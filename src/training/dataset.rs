//! The training dataset: a line-oriented, replayable format.
//!
//! ```text
//! # morstilia training data v1
//! W 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1  e2e4 g8f6 e4e5
//! D 8/8/8/4k3/8/8/4K3/8 w - - 0 1  <no pv>
//! L r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3  e8g8
//! ```
//!
//! One position per line: label, FEN, then the teacher's principal variation as
//! SAN (separated from the FEN by two spaces, so a FEN's own single spaces stay
//! unambiguous and a line can be split on whitespace without a parser).
//!
//! # Why a line format at all
//!
//! Because a training set is the one artefact that must outlive the code that
//! made it. A binary format couples the data to a struct layout; this one
//! couples it to nothing but FEN, which every chess tool speaks. It diffs, it
//! greps, it can be checked with `perft`-class tools, and a corpus split across
//! several machines merges with `cat`.
//!
//! # The PV is a *teacher's opinion*, not a fact
//!
//! It is stored because a policy head needs it, and it is stored per position
//! because it is specific to that position. It is never treated as ground truth
//! anywhere in this crate: [`Dataset::score_agreement`] measures how often the
//! net agrees with it, which is a statistic and not a requirement.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, bail};

use crate::board::Position;
use crate::training::config::Label;

/// The format version written into the header. Bumped on any incompatible
/// change; a reader refuses a version it does not know rather than guessing.
pub const DATA_FORMAT: u32 = 1;

/// The header line that starts every file.
pub const HEADER_PREFIX: &str = "# morstilia training data v";

/// One labelled position with the teacher's principal variation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub label: Label,
    pub fen: String,
    /// SAN plies, side-to-move relative. Empty when the teacher had none (a
    /// terminal position, or a tablebase verdict).
    pub pv: Vec<String>,
}

impl Sample {
    /// A sample with no PV — a terminal position or a tablebase-only label.
    pub fn without_pv(label: Label, fen: impl Into<String>) -> Sample {
        Sample {
            label,
            fen: fen.into(),
            pv: Vec::new(),
        }
    }

    /// A sample carrying a PV.
    pub fn new(label: Label, fen: impl Into<String>, pv: Vec<String>) -> Sample {
        Sample {
            label,
            fen: fen.into(),
            pv,
        }
    }

    /// The line as it is written to disk.
    pub fn to_line(&self) -> String {
        let mut s = String::with_capacity(self.fen.len() + 4 + self.pv.len() * 6);
        s.push(self.label.as_char());
        s.push(' ');
        s.push_str(&self.fen);
        if !self.pv.is_empty() {
            s.push_str("  ");
            s.push_str(&self.pv.join(" "));
        }
        s
    }

    /// Parses one data line. Returns `None` for a blank line or a comment, and
    /// an error for anything else — a malformed line is never skipped, because
    /// a silently dropped position is a hole in the training set that nobody
    /// will ever find.
    pub fn parse_line(line: &str) -> anyhow::Result<Option<Sample>> {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            return Ok(None);
        }
        let (head, pv_text) = match t.split_once("  ") {
            Some((h, p)) => (h, p),
            None => (t, ""),
        };
        let mut parts = head.split_whitespace();
        let label_text = parts
            .next()
            .with_context(|| format!("training data line has no label: {t:?}"))?;
        let label = Label::parse(label_text)
            .with_context(|| format!("unknown label {label_text:?} in: {t:?}"))?;
        let fen: String = parts.collect::<Vec<_>>().join(" ");
        if fen.is_empty() {
            bail!("training data line has no FEN: {t:?}");
        }
        let pv = pv_text
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>();
        Ok(Some(Sample { label, fen, pv }))
    }

    /// The position, parsed. The FEN is the record, so a bad one is an error and
    /// not a skipped row.
    pub fn position(&self) -> anyhow::Result<Position> {
        Position::from_fen(&self.fen)
            .map_err(|e| anyhow::anyhow!("invalid FEN {:?} in the dataset: {e}", self.fen))
    }

    /// Replays the teacher's PV and returns the SAN plies the position actually
    /// accepts, up to the first illegal one.
    ///
    /// Never errors on a bad move: a teacher PV that runs into a repetition the
    /// line did not record is a *truncated* PV, not a corrupt dataset. The
    /// count of samples that lost plies is the interesting number, and
    /// [`Dataset::validate`] reports it.
    pub fn legal_pv(&self) -> anyhow::Result<Vec<String>> {
        if self.pv.is_empty() {
            return Ok(Vec::new());
        }
        let mut pos = self.position()?;
        let mut out = Vec::with_capacity(self.pv.len());
        for san in &self.pv {
            match pos.play_san(san) {
                Ok((child, _)) => {
                    out.push(san.clone());
                    pos = child;
                }
                Err(_) => break,
            }
        }
        Ok(out)
    }
}

/// What an audit of a dataset found.
///
/// Each count is a number the operator can act on; none of them is fatal by
/// itself, which is exactly why they need to be visible. A dataset full of
/// duplicates still *trains* — it just learns the endgame prior twice as hard
/// as the endgame, and nothing in the loss curve says so.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Health {
    /// Lines that did not parse as FEN.
    pub unparseable_fens: usize,
    /// Samples whose PV the position does not accept in full.
    pub truncated_pvs: usize,
    /// Samples with an empty PV despite a non-terminal position.
    pub missing_pvs: usize,
    /// Duplicate FENs (the same position labelled twice).
    pub duplicates: usize,
    /// FENs that are not the standard starting position.
    pub non_standard_starts: usize,
}

impl Health {
    /// Whether nothing is wrong.
    pub fn is_clean(&self) -> bool {
        *self == Health::default()
    }

    /// A one-line summary.
    pub fn summary(&self) -> String {
        if self.is_clean() {
            return "dataset health: clean".to_string();
        }
        let mut parts: Vec<String> = Vec::new();
        if self.unparseable_fens > 0 {
            parts.push(format!("{} unparseable FENs", self.unparseable_fens));
        }
        if self.truncated_pvs > 0 {
            parts.push(format!("{} truncated PVs", self.truncated_pvs));
        }
        if self.missing_pvs > 0 {
            parts.push(format!("{} missing PVs", self.missing_pvs));
        }
        if self.duplicates > 0 {
            parts.push(format!("{} duplicate positions", self.duplicates));
        }
        if self.non_standard_starts > 0 {
            parts.push(format!(
                "{} positions not reachable from the start position",
                self.non_standard_starts
            ));
        }
        format!("dataset health: {}", parts.join(", "))
    }
}

/// A labelled dataset with a deterministic split.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// Where it came from, for reports.
    pub source: String,
    samples: Vec<Sample>,
}

impl Dataset {
    /// An empty dataset.
    pub fn empty(source: impl Into<String>) -> Dataset {
        Dataset {
            source: source.into(),
            samples: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    pub fn push(&mut self, s: Sample) {
        self.samples.push(s);
    }

    pub fn extend(&mut self, other: Dataset) {
        self.samples.extend(other.samples);
    }

    /// The W/D/L counts, in that order.
    pub fn wdl(&self) -> (usize, usize, usize) {
        let (mut w, mut d, mut l) = (0, 0, 0);
        for s in &self.samples {
            match s.label {
                Label::Win => w += 1,
                Label::Draw => d += 1,
                Label::Loss => l += 1,
            }
        }
        (w, d, l)
    }

    /// The fraction of draws, in `0.0 ..= 1.0`.
    pub fn draw_rate(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            let (_, d, _) = self.wdl();
            d as f64 / self.samples.len() as f64
        }
    }

    /// The mean PV length over the samples that have one.
    pub fn mean_pv_len(&self) -> f64 {
        let with = self.samples.iter().filter(|s| !s.pv.is_empty()).count();
        if with == 0 {
            return 0.0;
        }
        let total: usize = self.samples.iter().map(|s| s.pv.len()).sum();
        total as f64 / with as f64
    }

    /// A one-line summary for a log.
    pub fn stats(&self) -> String {
        let (w, d, l) = self.wdl();
        format!(
            "{} samples from {} (W {w} D {d} L {l}, draw rate {:.1}%, mean PV {:.1})",
            self.samples.len(),
            self.source,
            100.0 * self.draw_rate(),
            self.mean_pv_len()
        )
    }

    /// A deterministic train/test split.
    ///
    /// Shuffles with a caller-owned `SplitMix64` before slicing, so the split is
    /// reproducible from `(seed, fraction)` and does not depend on the order the
    /// games happened to be played in. A dataset written in game order is
    /// otherwise trivially biased: the first games and the last games are not
    /// exchangeable samples of the same distribution.
    pub fn split(&self, test_fraction: f64, seed: u64) -> (Dataset, Dataset) {
        if self.samples.is_empty() || test_fraction <= 0.0 {
            return (
                self.clone(),
                Dataset::empty(format!("{}-train", self.source)),
            );
        }
        let f = test_fraction.min(0.999_999);
        let mut order: Vec<usize> = (0..self.samples.len()).collect();
        let mut rng = crate::book::SplitMix64(seed);
        // Fisher-Yates, using the same generator the rest of the crate seeds
        // from, so "the shuffle" means one thing in this codebase.
        for i in (1..order.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let n_test = ((self.samples.len() as f64) * f).round() as usize;
        let n_test = n_test.min(self.samples.len().saturating_sub(1));
        let mut train = Dataset::empty(format!("{}-train", self.source));
        let mut test = Dataset::empty(format!("{}-test", self.source));
        for (rank, &idx) in order.iter().enumerate() {
            if rank < n_test {
                test.samples.push(self.samples[idx].clone());
            } else {
                train.samples.push(self.samples[idx].clone());
            }
        }
        (train, test)
    }

    /// Reads a dataset from text.
    ///
    /// The header is required and its version checked, so a file from a future
    /// format is refused instead of being half-read.
    pub fn from_text(text: &str, source: impl Into<String>) -> anyhow::Result<Dataset> {
        let source = source.into();
        let mut lines = text.lines();
        let header = lines
            .next()
            .with_context(|| format!("{source}: the file is empty"))?
            .trim();
        let version = header
            .strip_prefix(HEADER_PREFIX)
            .with_context(|| {
                format!(
                    "{source}: missing the {:?} header (is this a training data file?)",
                    HEADER_PREFIX
                )
            })?
            .trim()
            .parse::<u32>()
            .with_context(|| format!("{source}: unparseable data format version"))?;
        if version != DATA_FORMAT {
            bail!("{source}: data format v{version}, this build reads v{DATA_FORMAT}");
        }
        let mut ds = Dataset::empty(source.clone());
        for (n, line) in lines.enumerate() {
            if let Some(s) =
                Sample::parse_line(line).with_context(|| format!("{source}:{}", n + 2))?
            {
                ds.samples.push(s);
            }
        }
        Ok(ds)
    }

    /// Reads a dataset from a file.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Dataset> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read dataset {}: {e}", path.display()))?;
        Dataset::from_text(&text, path.display().to_string())
    }

    /// The dataset as text, header included.
    pub fn to_text(&self) -> String {
        let mut s = String::with_capacity(self.samples.len() * 96 + 32);
        s.push_str(HEADER_PREFIX);
        s.push_str(&DATA_FORMAT.to_string());
        s.push('\n');
        for sample in &self.samples {
            s.push_str(&sample.to_line());
            s.push('\n');
        }
        s
    }

    /// Writes the dataset, creating the parent directory.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", parent.display()))?;
        }
        std::fs::write(path, self.to_text())
            .map_err(|e| anyhow::anyhow!("cannot write dataset {}: {e}", path.display()))
    }

    /// Streams a dataset in, accepting either `#`-comments or blank lines
    /// between records. Written for the case where the corpus is too big to
    /// hold twice, so a comment cannot cost a full re-read.
    pub fn stream_in<R: BufRead>(mut reader: R, ds: &mut Dataset) -> anyhow::Result<()> {
        for line in reader.by_ref().lines() {
            let line = line.with_context(|| "cannot read a dataset line")?;
            if let Some(s) = Sample::parse_line(&line)? {
                ds.samples.push(s);
            }
        }
        Ok(())
    }

    /// Writes the dataset line by line, so a multi-gigabyte corpus never has to
    /// be materialised as one `String`.
    pub fn stream_out<W: Write>(&self, mut writer: W) -> anyhow::Result<()> {
        let w = &mut writer;
        writeln!(w, "{HEADER_PREFIX}{DATA_FORMAT}")?;
        for s in &self.samples {
            writeln!(w, "{}", s.to_line())?;
        }
        writer.flush().with_context(|| "cannot flush the dataset")?;
        Ok(())
    }

    /// The W/D/L balance per piece-count bucket, as `(bucket, wins, draws,
    /// losses)`. The classic NNUE sanity plot: a bucket that is 90% one label
    /// teaches the net the endgame prior, not the endgame.
    pub fn wdl_by_piece_bucket(
        &self,
        bucket_width: usize,
    ) -> BTreeMap<usize, (usize, usize, usize)> {
        let mut out: BTreeMap<usize, (usize, usize, usize)> = BTreeMap::new();
        let w = bucket_width.max(1);
        for s in &self.samples {
            let Ok(pos) = s.position() else { continue };
            let n = pos.piece_count();
            let e = out.entry(n / w).or_insert((0, 0, 0));
            match s.label {
                Label::Win => e.0 += 1,
                Label::Draw => e.1 += 1,
                Label::Loss => e.2 += 1,
            }
        }
        out
    }

    /// Audits the dataset. Parses every FEN, so it is `O(n)` in positions and
    /// is meant to run once, on a sample of a large corpus, not on every
    /// training epoch.
    pub fn validate(&self) -> Health {
        let mut h = Health::default();
        let start_fen = Position::startpos().fen();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for s in &self.samples {
            let Ok(pos) = s.position() else {
                h.unparseable_fens += 1;
                continue;
            };
            if !seen.insert(s.fen.as_str()) {
                h.duplicates += 1;
            }
            if s.fen != start_fen {
                h.non_standard_starts += 1;
            }
            if s.pv.is_empty() {
                // A terminal position legitimately has no PV; a non-terminal one
                // does not.
                if pos.legal_moves().is_empty() {
                    continue;
                }
                h.missing_pvs += 1;
                continue;
            }
            match s.legal_pv() {
                Ok(legal) if legal.len() == s.pv.len() => {}
                _ => h.truncated_pvs += 1,
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MID: &str = "r1bq1rk1/ppp2ppp/2n5/3p4/3P4/2P5/PP1B1PPP/RNBQ1RK1 w - - 0 8";

    fn sample() -> Sample {
        Sample::new(Label::Win, MID, vec!["c1g5".into(), "d8g8".into()])
    }

    #[test]
    fn a_sample_round_trips_through_its_line() {
        let s = sample();
        let back = Sample::parse_line(&s.to_line()).unwrap().unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn a_sample_without_a_pv_round_trips() {
        let s = Sample::without_pv(Label::Draw, MID);
        let back = Sample::parse_line(&s.to_line()).unwrap().unwrap();
        assert_eq!(back, s);
        assert!(back.pv.is_empty());
    }

    #[test]
    fn a_fen_with_spaces_survives_the_two_space_separator() {
        // The FEN has single spaces; the separator is two. Splitting on one
        // space would tear it apart, which is the whole reason the separator is
        // not a single space.
        let s = sample();
        let line = s.to_line();
        let (_, rest) = line.split_once("  ").unwrap();
        assert!(rest.contains(' '));
        assert_eq!(Sample::parse_line(&line).unwrap().unwrap().fen, MID);
    }

    #[test]
    fn comments_and_blanks_are_skipped_and_garbage_is_not() {
        assert!(Sample::parse_line("").unwrap().is_none());
        assert!(Sample::parse_line("   ").unwrap().is_none());
        assert!(Sample::parse_line("# a note").unwrap().is_none());
        // A line with a label but no FEN is an error, not a skip.
        assert!(Sample::parse_line("W").is_err());
        // A line with an unknown label is an error.
        assert!(Sample::parse_line("X not a fen").is_err());
    }

    #[test]
    fn a_dataset_round_trips_through_text() {
        let mut ds = Dataset::empty("test");
        ds.push(sample());
        ds.push(Sample::without_pv(Label::Loss, MID));
        ds.push(Sample::new(Label::Draw, MID, vec!["c1g5".into()]));
        let back = Dataset::from_text(&ds.to_text(), "test").unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back.samples(), ds.samples());
    }

    #[test]
    fn a_missing_or_wrong_header_is_refused() {
        assert!(Dataset::from_text("", "x").is_err());
        assert!(Dataset::from_text("W 8/8/8/8/8/8/8/8 w - - 0 1\n", "x").is_err());
        let future = format!("{HEADER_PREFIX}999\nW {MID}\n");
        let e = Dataset::from_text(&future, "x").unwrap_err().to_string();
        assert!(e.contains("v999"), "unhelpful error: {e}");
    }

    #[test]
    fn streaming_and_buffered_reads_agree() {
        let mut ds = Dataset::empty("test");
        for i in 0..50 {
            ds.push(Sample::without_pv(
                if i % 3 == 0 { Label::Win } else { Label::Loss },
                MID,
            ));
        }
        let mut streamed = Dataset::empty("stream");
        Dataset::stream_in(ds.to_text().as_bytes(), &mut streamed).unwrap();
        assert_eq!(streamed.samples(), ds.samples());

        let mut out: Vec<u8> = Vec::new();
        ds.stream_out(&mut out).unwrap();
        let reparsed = Dataset::from_text(std::str::from_utf8(&out).unwrap(), "x").unwrap();
        assert_eq!(reparsed.samples(), ds.samples());
    }

    #[test]
    fn the_split_is_deterministic_disjoint_and_complete() {
        let mut ds = Dataset::empty("test");
        for i in 0..100 {
            ds.push(Sample::without_pv(
                if i % 2 == 0 { Label::Win } else { Label::Loss },
                MID,
            ));
        }
        let (a_train, a_test) = ds.split(0.2, 99);
        let (_b_train, b_test) = ds.split(0.2, 99);
        assert_eq!(a_test.len(), 20);
        assert_eq!(a_train.len(), 80);
        // Same seed, same split.
        assert_eq!(
            a_test
                .samples()
                .iter()
                .map(|s| s.fen.clone())
                .collect::<Vec<_>>(),
            b_test
                .samples()
                .iter()
                .map(|s| s.fen.clone())
                .collect::<Vec<_>>()
        );
        // Different seed, (almost surely) different split.
        let (_, c_test) = ds.split(0.2, 100);
        assert_ne!(
            a_test.samples().iter().map(|s| s.label).collect::<Vec<_>>(),
            c_test.samples().iter().map(|s| s.label).collect::<Vec<_>>()
        );
        // Disjoint and covering.
        assert_eq!(a_train.len() + a_test.len(), ds.len());
    }

    #[test]
    fn a_zero_fraction_keeps_everything_for_training() {
        let mut ds = Dataset::empty("test");
        ds.push(sample());
        let (train, test) = ds.split(0.0, 1);
        assert_eq!(train.len(), 1);
        assert!(test.is_empty());
    }

    #[test]
    fn a_full_fraction_never_starves_the_training_half() {
        // A 1.0 fraction would leave nothing to train on, so it is clamped and
        // the training half always keeps at least one sample.
        let mut ds = Dataset::empty("test");
        for _ in 0..4 {
            ds.push(sample());
        }
        let (train, test) = ds.split(1.0, 1);
        assert!(!train.is_empty());
        assert!(!test.is_empty());
    }

    #[test]
    fn the_pv_is_replayed_and_truncates_rather_than_erroring() {
        let s = Sample::new(Label::Win, MID, vec!["c1g5".into(), "d8g8".into()]);
        assert_eq!(s.legal_pv().unwrap(), s.pv);

        // A second, illegal move truncates the line and returns the legal
        // prefix — a teacher PV that ran into an unrecorded repetition is
        // truncated evidence, not a corrupt dataset.
        let bad = Sample::new(Label::Win, MID, vec!["c1g5".into(), "a1a8".into()]);
        assert_eq!(bad.legal_pv().unwrap(), vec!["c1g5".to_string()]);

        assert!(
            Sample::without_pv(Label::Win, MID)
                .legal_pv()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn health_reports_what_is_actually_wrong() {
        let mut ds = Dataset::empty("test");
        ds.push(sample());
        let h = ds.validate();
        assert!(h.duplicates == 0, "one sample is not a duplicate");
        assert!(h.unparseable_fens == 0);
        assert!(h.missing_pvs == 0);
        assert!(h.truncated_pvs == 0);
        assert!(h.non_standard_starts == 1, "the test FEN is not the start");

        // The same FEN twice is a duplicate.
        ds.push(sample());
        assert_eq!(ds.validate().duplicates, 1);

        let mut broken = Dataset::empty("test");
        broken.push(Sample::new(Label::Win, "not a fen at all", vec![]));
        assert_eq!(broken.validate().unparseable_fens, 1);

        // A non-terminal position with no PV.
        let mut no_pv = Dataset::empty("test");
        no_pv.push(Sample::without_pv(Label::Win, MID));
        assert_eq!(no_pv.validate().missing_pvs, 1);

        // A truncated PV.
        let mut trunc = Dataset::empty("test");
        trunc.push(Sample::new(
            Label::Win,
            MID,
            vec!["c1g5".into(), "not-a-move".into()],
        ));
        assert_eq!(trunc.validate().truncated_pvs, 1);
    }

    #[test]
    fn health_summary_lists_every_symptom() {
        let h = Health {
            unparseable_fens: 2,
            truncated_pvs: 3,
            missing_pvs: 4,
            duplicates: 5,
            non_standard_starts: 6,
        };
        let s = h.summary();
        for n in ["2", "3", "4", "5", "6"] {
            assert!(s.contains(n), "summary missing {n}: {s}");
        }
        assert!(Health::default().is_clean());
        assert!(Health::default().summary().contains("clean"));
    }

    #[test]
    fn wdl_statistics_are_consistent() {
        let mut ds = Dataset::empty("test");
        ds.push(Sample::without_pv(Label::Win, MID));
        ds.push(Sample::without_pv(Label::Draw, MID));
        ds.push(Sample::without_pv(Label::Loss, MID));
        ds.push(Sample::without_pv(Label::Loss, MID));
        assert_eq!(ds.wdl(), (1, 1, 2));
        assert!((ds.draw_rate() - 0.25).abs() < 1e-12);
        assert!(ds.stats().contains("W 1 D 1 L 2"));
        assert_eq!(Dataset::empty("e").draw_rate(), 0.0);
    }

    #[test]
    fn the_piece_bucket_histogram_counts_every_sample_once() {
        let mut ds = Dataset::empty("test");
        for l in [Label::Win, Label::Draw, Label::Loss] {
            ds.push(Sample::without_pv(l, MID));
        }
        let buckets = ds.wdl_by_piece_bucket(4);
        let total: usize = buckets.values().map(|(w, d, l)| w + d + l).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn a_saved_dataset_reloads_identically() {
        let dir = std::env::temp_dir().join(format!("morstilia-ds-{}", std::process::id()));
        let path = dir.join("nested").join("data.txt");
        let mut ds = Dataset::empty("test");
        ds.push(sample());
        ds.save(&path).unwrap();
        assert_eq!(Dataset::load(&path).unwrap().samples(), ds.samples());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
