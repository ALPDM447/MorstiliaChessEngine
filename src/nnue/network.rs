//! The network itself: the feature transformer's weights, the eight layer
//! stacks, and the forward pass that turns an accumulator into a score.
//!
//! # Arithmetic
//!
//! Everything is integer, exactly as Stockfish's scalar (non-SIMD) path:
//!
//! 1. **Feature transform** — the accumulator is clipped to `[0, 255]` and the
//!    two halves are multiplied pairwise, `/512`:
//!    `out[512*p + j] = clip(acc[j], 0, 255) * clip(acc[j+512], 0, 255) / 512`.
//! 2. **`fc_0`** — `1024 -> 32` affine, `i8` weights against the `u8` features.
//! 3. **`ac_sqr_0` / `ac_0`** — `min(127, x² >> 21)` and `clamp(x >> 7, 0, 127)`.
//! 4. **`fc_1`** — `64 -> 32` affine on `[sqr_0 ‖ clip_0]`.
//! 5. **`ac_sqr_1` / `ac_1`** — `min(127, x² >> 19)` and `clamp(x >> 6, 0, 127)`.
//! 6. **`fc_2`** — `128 -> 1` affine on `[sqr_0 ‖ clip_0 ‖ sqr_1 ‖ clip_1]`.
//! 7. A skip connection adds `fc_0[30] - fc_0[31]`, and the result is rescaled
//!    by `600 * 16 / (128 * 64 * 2)`.
//!
//! The weight layout is the *scalar* one: `weights[j * PadIn + i]` for output
//! `j` and input `i`, and `get_weight_index(i) == i` — no SIMD chunk
//! interleaving, because there is no SIMD. Feeding a SIMD-shaped net to a
//! scalar reader (or the reverse) is exactly what the per-section hashes in
//! [`crate::nnue::format`] protect against.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Deref;
use std::path::Path;

use crate::nnue::accumulator::{Accumulator, L1, LAYER_STACKS, PSQT_BUCKETS};
use crate::nnue::features::{full_threats, half_ka_v2_hm, pp_3wide};
use crate::nnue::format::{self, FormatError, Reader};
use crate::nnue::types::Color;

// --- constants (Stockfish's `nnue_common.h`) ---------------------------------

/// Divisor turning the network's fixed-point output into centipawns.
pub const OUTPUT_SCALE: i32 = 16;
/// Fixed-point log2 scale of every `i8` weight.
pub const WEIGHT_SCALE_BITS: i32 = 6;
/// Clip bound of the feature transform.
pub const FT_MAX_VAL: i32 = 255;
/// The activation that represents 1.0 in the hidden layers.
pub const HIDDEN_ONE_VAL: i32 = 128;
/// Row stride quantum of the weight arrays (Stockfish's `MaxSimdWidth`).
const PAD_QUANTUM: usize = 32;

/// `HalfKAv2_hm::Dimensions` — rows of the PSQT weight block.
const PSQ_DIMENSIONS: usize = half_ka_v2_hm::DIMENSIONS;
/// `FullThreats::Dimensions` — first block of the threat/pawn-pair weights.
const THREAT_DIMENSIONS: usize = full_threats::DIMENSIONS;
/// `PP_3Wide::Dimensions` — second block, contiguous with the first so a single
/// index addresses either set.
const PAIR_DIMENSIONS: usize = pp_3wide::DIMENSIONS;
/// Both blocks together.
const OTHER_DIMENSIONS: usize = THREAT_DIMENSIONS + PAIR_DIMENSIONS;

/// Second fully connected layer width.
const L2: usize = 32;
/// Third fully connected layer width.
const L3: usize = 32;

// --- structural hashes --------------------------------------------------------

/// Stockfish's `combine_hash`: rotate left, then xor.
const fn combine_hash(parts: [u32; 3]) -> u32 {
    let mut hash = 0u32;
    let mut i = 0;
    while i < parts.len() {
        hash = (hash << 1) | (hash >> 31);
        hash ^= parts[i];
        i += 1;
    }
    hash
}

/// Hash of the feature-set trio *and* the transform width. A net trained on a
/// different feature set fails here before a single weight is read.
pub const FEATURE_TRANSFORMER_HASH: u32 = combine_hash([
    full_threats::HASH_VALUE,
    pp_3wide::HASH_VALUE,
    half_ka_v2_hm::HASH_VALUE,
]) ^ (L1 as u32 * 2);

/// `AffineTransform::get_hash_value`.
const fn affine_hash(prev: u32, outputs: u32) -> u32 {
    let h = 0xCC03_DAE4u32.wrapping_add(outputs);
    h ^ (prev >> 1) ^ (prev << 31)
}

/// `ClippedReLU::get_hash_value`.
const fn clipped_hash(prev: u32) -> u32 {
    0x538D_24C7u32.wrapping_add(prev)
}

/// Hash of the layer-stack shape. The `SqrClippedReLU` activations are
/// deliberately excluded (as in Stockfish): the trainer does not write them.
pub const LAYER_STACK_HASH: u32 = {
    let h = 0xEC42_E90Du32 ^ (L1 as u32 * 2);
    let h = affine_hash(h, L2 as u32); // fc_0: 1024 -> 32
    let h = clipped_hash(h); // ac_0
    let h = affine_hash(h, L3 as u32); // fc_1: 64 -> 32
    let h = clipped_hash(h); // ac_1
    affine_hash(h, 1) // fc_2: 128 -> 1
};

/// The structural hash in the file header: a net built for a different feature
/// set *or* a different network shape is rejected outright.
pub const NETWORK_HASH: u32 = FEATURE_TRANSFORMER_HASH ^ LAYER_STACK_HASH;

// --- the feature transformer ---------------------------------------------------

/// The first half of the network: 86 896 sparse input features, 1024 dense
/// outputs, plus the PSQT side that is summed per piece-count bucket.
///
/// The weights are stored **feature-major** (`row(idx)` is 1024 contiguous
/// values), matching `ft.weights[list[i] * Dimensions + j]` in Stockfish.
pub struct FeatureTransformer {
    /// The transformer's bias term.
    pub biases: Box<[i16; L1]>,
    /// `HalfKAv2_hm` weights, `PSQ_DIMENSIONS * L1` values.
    psq_weights: Box<[i16]>,
    /// `HalfKAv2_hm` PSQT weights, `PSQ_DIMENSIONS * PSQT_BUCKETS` values.
    psqt_weights: Box<[i32]>,
    /// `FullThreats` weights followed by `PP_3Wide` weights,
    /// `OTHER_DIMENSIONS * L1` values.
    other_weights: Box<[i8]>,
    /// The matching PSQT weights, `OTHER_DIMENSIONS * PSQT_BUCKETS` values.
    other_psqt: Box<[i32]>,
}

/// Number of sparse input features, for diagnostics.
pub const INPUT_DIMENSIONS: usize = PSQ_DIMENSIONS + OTHER_DIMENSIONS;

impl FeatureTransformer {
    /// The 1024 `i16` weights of one `HalfKAv2_hm` feature.
    #[inline]
    pub fn weight_row(&self, idx: usize) -> &[i16] {
        &self.psq_weights[idx * L1..(idx + 1) * L1]
    }

    /// The 1024 `i8` weights of one threat or pawn-pair feature.
    #[inline]
    pub fn other_weight_row(&self, idx: usize) -> &[i8] {
        &self.other_weights[idx * L1..(idx + 1) * L1]
    }

    /// The 8 bucket-expanded `i32` weights of one `HalfKAv2_hm` feature.
    #[inline]
    pub fn psqt_weight_row(&self, idx: usize) -> &[i32] {
        &self.psqt_weights[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS]
    }

    /// The 8 bucket-expanded `i32` weights of one threat or pawn-pair feature.
    #[inline]
    pub fn other_psqt_row(&self, idx: usize) -> &[i32] {
        &self.other_psqt[idx * PSQT_BUCKETS..(idx + 1) * PSQT_BUCKETS]
    }

    /// Reads the feature-transformer section (hash + six weight blocks).
    fn read<R: Read>(reader: &mut Reader<R>) -> format::Result<Self> {
        reader.read_section_hash(FEATURE_TRANSFORMER_HASH, "feature transformer")?;
        let mut ft = FeatureTransformer {
            biases: Box::new([0; L1]),
            psq_weights: vec![0i16; PSQ_DIMENSIONS * L1].into_boxed_slice(),
            psqt_weights: vec![0i32; PSQ_DIMENSIONS * PSQT_BUCKETS].into_boxed_slice(),
            other_weights: vec![0i8; OTHER_DIMENSIONS * L1].into_boxed_slice(),
            other_psqt: vec![0i32; OTHER_DIMENSIONS * PSQT_BUCKETS].into_boxed_slice(),
        };

        // The file order is biases, threat weights, threat PSQT, pawn-pair
        // weights, pawn-pair PSQT, then the two PSQT feature blocks. The two
        // `i8` blocks are contiguous in the file and in memory, so they can be
        // filled in place without copying.
        reader.read_leb128(&mut ft.biases[..], "feature transformer biases")?;
        let split = THREAT_DIMENSIONS * L1;
        reader.read_le_array(&mut ft.other_weights[..split], "threat weights")?;
        let psqt_split = THREAT_DIMENSIONS * PSQT_BUCKETS;
        reader.read_leb128(&mut ft.other_psqt[..psqt_split], "threat psqt weights")?;
        reader.read_le_array(&mut ft.other_weights[split..], "pawn pair weights")?;
        reader.read_leb128(&mut ft.other_psqt[psqt_split..], "pawn pair psqt weights")?;
        reader.read_leb128(&mut ft.psq_weights[..], "psq weights")?;
        reader.read_leb128(&mut ft.psqt_weights[..], "psqt weights")?;
        Ok(ft)
    }
}

// --- layers ---------------------------------------------------------------------

/// A fully connected layer: `i8` weights, `i32` biases, `u8` in, `i32` out.
///
/// The zero input columns are skipped, exactly like Stockfish's scalar
/// `affine_transform_non_ssse3` — the transformed features are mostly zero and
/// the products are anyway zero.
pub struct Affine<const IN: usize, const OUT: usize> {
    biases: Box<[i32]>,
    weights: Box<[i8]>,
}

impl<const IN: usize, const OUT: usize> Affine<IN, OUT> {
    /// Row stride of the weight block.
    const PAD_IN: usize = (IN + PAD_QUANTUM - 1) / PAD_QUANTUM * PAD_QUANTUM;

    /// Reads `OUT` `i32` biases then `OUT * PAD_IN` `i8` weights, both raw
    /// little-endian (the layer stacks are *not* LEB128-compressed).
    fn read<R: Read>(reader: &mut Reader<R>, what: &'static str) -> format::Result<Self> {
        let mut layer = Affine {
            biases: vec![0i32; OUT].into_boxed_slice(),
            weights: vec![0i8; OUT * Self::PAD_IN].into_boxed_slice(),
        };
        reader.read_le_array(&mut layer.biases[..], what)?;
        reader.read_le_array(&mut layer.weights[..], what)?;
        Ok(layer)
    }

    /// `output[j] = bias[j] + sum_i weights[j * PAD_IN + i] * input[i]`.
    fn propagate(&self, input: &[u8], output: &mut [i32]) {
        debug_assert!(input.len() >= IN);
        for (j, out) in output.iter_mut().enumerate().take(OUT) {
            let row = &self.weights[j * Self::PAD_IN..];
            let mut sum = self.biases[j];
            for (i, &v) in input.iter().enumerate().take(IN) {
                if v != 0 {
                    sum += i32::from(row[i]) * i32::from(v);
                }
            }
            *out = sum;
        }
    }
}

/// `clamp(x >> bits, 0, 127)`.
#[inline]
fn clipped_relu(x: i32, bits: i32) -> u8 {
    ((x >> bits).clamp(0, 127)) as u8
}

/// `min(127, x² >> (2 * bits + 7))` — the extra 7 bits are a deliberate
/// approximation the trainer compensates for.
#[inline]
fn sqr_clipped_relu(x: i32, bits: i32) -> u8 {
    let square = i64::from(x) * i64::from(x);
    (square >> (2 * bits + 7)).min(127) as u8
}

/// One of the eight evaluation heads: `fc_0`, `fc_1`, `fc_2` and the two
/// activation pairs between them.
pub struct LayerStack {
    fc_0: Affine<L1, L2>,
    fc_1: Affine<{ L2 * 2 }, L3>,
    fc_2: Affine<{ L2 * 2 + L3 * 2 }, 1>,
}

impl LayerStack {
    fn read<R: Read>(reader: &mut Reader<R>) -> format::Result<Self> {
        reader.read_section_hash(LAYER_STACK_HASH, "layer stack")?;
        Ok(LayerStack {
            fc_0: Affine::read(reader, "fc_0")?,
            fc_1: Affine::read(reader, "fc_1")?,
            fc_2: Affine::read(reader, "fc_2")?,
        })
    }

    /// The forward pass. `transformed` holds the 1024 dense input features.
    fn propagate(&self, transformed: &[u8; L1]) -> i32 {
        let mut fc_0_out = [0i32; L2];
        self.fc_0.propagate(transformed, &mut fc_0_out);

        // The concatenation buffer is shared between both activation pairs, so
        // it is written once and read twice by the next layer.
        let mut concat = [0u8; L2 * 2 + L3 * 2];
        for (i, &x) in fc_0_out.iter().enumerate() {
            concat[i] = sqr_clipped_relu(x, WEIGHT_SCALE_BITS + 1);
            concat[L2 + i] = clipped_relu(x, WEIGHT_SCALE_BITS + 1);
        }

        let mut fc_1_out = [0i32; L3];
        self.fc_1.propagate(&concat[..L2 * 2], &mut fc_1_out);
        for (i, &x) in fc_1_out.iter().enumerate() {
            concat[L2 * 2 + i] = sqr_clipped_relu(x, WEIGHT_SCALE_BITS);
            concat[L2 * 2 + L3 + i] = clipped_relu(x, WEIGHT_SCALE_BITS);
        }

        let mut fc_2_out = [0i32; 1];
        self.fc_2.propagate(&concat, &mut fc_2_out);

        // Skip connection from the first hidden layer's last two units.
        let fwd_out = fc_2_out[0] + (fc_0_out[L2 - 2] - fc_0_out[L2 - 1]);
        // 1.0 is `HiddenOneVal * (1 << WeightScaleBits) * 2` in the net's
        // quantisation but `600 * OutputScale` in centipawns, so the output is
        // rescaled — in 64 bits, to make overflow impossible.
        let multiplier = 600 * i64::from(OUTPUT_SCALE);
        let denominator = i64::from(HIDDEN_ONE_VAL) * (1i64 << WEIGHT_SCALE_BITS) * 2;
        ((i64::from(fwd_out) * multiplier) / denominator) as i32
    }
}

// --- the network ----------------------------------------------------------------

/// The network's raw output: the PSQT term and the positional term, both
/// already divided by [`OUTPUT_SCALE`], from the side to move's point of view.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NetworkOutput {
    pub psqt: i32,
    pub positional: i32,
}

impl NetworkOutput {
    /// `psqt + positional`, before the complexity damping in
    /// [`crate::nnue::blend`].
    pub fn raw(&self) -> i32 {
        self.psqt + self.positional
    }
}

/// A fully loaded, validated net.
///
/// Dereferences to its [`FeatureTransformer`] so accumulator code can name the
/// net directly where a transformer is expected.
pub struct Network {
    ft: FeatureTransformer,
    stacks: Box<[LayerStack; LAYER_STACKS]>,
    /// The trainer's own description line, kept for diagnostics.
    pub description: String,
}

impl Deref for Network {
    type Target = FeatureTransformer;

    #[inline]
    fn deref(&self) -> &FeatureTransformer {
        &self.ft
    }
}

impl Network {
    /// The network input features, for diagnostics.
    pub fn input_dimensions(&self) -> usize {
        INPUT_DIMENSIONS
    }

    /// The structural hash this net was validated against.
    pub fn hash(&self) -> u32 {
        NETWORK_HASH
    }

    /// The clamp-and-multiply transform of one perspective.
    ///
    /// `out[offset + j] = clip(acc[j]) * clip(acc[j + 512]) / 512` for
    /// `j < 512`. Both halves are clipped independently, so the product is
    /// never negative and the maximum output is `255 * 255 / 512 = 127`.
    #[inline]
    fn transform_perspective(acc: &[i16; L1], out: &mut [u8; L1], offset: usize) {
        for j in 0..L1 / 2 {
            let lo = i32::from(acc[j]).clamp(0, FT_MAX_VAL) as u32;
            let hi = i32::from(acc[j + L1 / 2]).clamp(0, FT_MAX_VAL) as u32;
            out[offset + j] = ((lo * hi) / 512) as u8;
        }
    }

    /// Evaluates one position from its (already computed) accumulator.
    ///
    /// `bucket` is the piece-count bucket `(piece_count - 1) / 4`, clamped to
    /// the eight heads.
    pub fn evaluate(&self, acc: &Accumulator, stm: Color, bucket: usize) -> NetworkOutput {
        let them = stm.other();
        let bucket = bucket.min(LAYER_STACKS - 1);

        // The PSQT accumulator is antisymmetric: each perspective sees its own
        // pieces' values, so halving the difference recovers the stm-relative
        // term without a second pass.
        let psqt = (acc.psqt[stm.idx()][bucket] - acc.psqt[them.idx()][bucket]) / 2;

        let mut transformed = [0u8; L1];
        Self::transform_perspective(&acc.accumulation[stm.idx()], &mut transformed, 0);
        Self::transform_perspective(&acc.accumulation[them.idx()], &mut transformed, L1 / 2);

        let positional = self.stacks[bucket].propagate(&transformed);
        NetworkOutput {
            psqt: psqt / OUTPUT_SCALE,
            positional: positional / OUTPUT_SCALE,
        }
    }
}

// --- loading ---------------------------------------------------------------------

/// Why a net could not be used. Both variants mean the same thing to the
/// caller: "keep playing with the classical evaluator".
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be opened or read.
    Io(std::io::Error),
    /// The file is not a net this engine can run.
    Format(FormatError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "cannot read the net file: {e}"),
            LoadError::Format(e) => write!(f, "invalid net: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Reads and validates a `.nnue` file.
///
/// Every failure mode — missing file, wrong version, wrong architecture hash,
/// a truncated or over-long section, trailing garbage — is reported as an
/// `Err` and leaves nothing behind; a caller that gets an `Err` must keep
/// running with whatever evaluator it had.
pub fn load_network(path: &Path) -> Result<Network, LoadError> {
    let file = File::open(path).map_err(LoadError::Io)?;
    // One MiB of user-space buffering: the net is read in 8 KiB LEB128 blocks
    // and in 4 KiB raw chunks, both of which re-enter the file constantly.
    let mut reader = Reader::new(BufReader::with_capacity(1 << 20, file));
    load_from(&mut reader).map_err(LoadError::Format)
}

/// Reads a net from an already-buffered reader. Split out from
/// [`load_network`] so the loader itself is testable over in-memory bytes.
pub fn load_from<R: Read>(reader: &mut Reader<R>) -> format::Result<Network> {
    let header = reader.read_header(NETWORK_HASH)?;
    let ft = FeatureTransformer::read(reader)?;

    let mut stacks: Vec<LayerStack> = Vec::with_capacity(LAYER_STACKS);
    for _ in 0..LAYER_STACKS {
        stacks.push(LayerStack::read(reader)?);
    }
    let stacks: Box<[LayerStack; LAYER_STACKS]> = stacks
        .try_into()
        .map_err(|_| FormatError::new("net file", "missing layer stacks"))?;

    reader.expect_eof("net file")?;
    Ok(Network {
        ft,
        stacks,
        description: header.description,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_hashes_match_stockfish() {
        // Values read straight out of `nn-1a298aa575a0.nnue`. If a hash ever
        // changes, the bundled net stops loading — this test says so first.
        assert_eq!(FEATURE_TRANSFORMER_HASH, 0xCB68_5313);
        assert_eq!(LAYER_STACK_HASH, 0x6333_7116);
        assert_eq!(NETWORK_HASH, 0xA85B_2205);
    }

    #[test]
    fn input_dimensions_are_the_sum_of_the_feature_sets() {
        assert_eq!(INPUT_DIMENSIONS, 22528 + 59808 + 4560);
        assert_eq!(INPUT_DIMENSIONS, 86_896);
    }

    #[test]
    fn row_accessors_return_the_right_slices() {
        // A one-element stand-in is enough: the accessors are pure offsets, and
        // the offsets are what a wrong net layout would corrupt. The weights
        // themselves are checked by address rather than by value, because a
        // flat `0..N` filler wraps around the narrow `i8` block.
        let idx = 7usize;
        let ft = FeatureTransformer {
            biases: Box::new([0; L1]),
            psq_weights: (0..PSQ_DIMENSIONS * L1).map(|i| i as i16).collect(),
            psqt_weights: (0..PSQ_DIMENSIONS * PSQT_BUCKETS)
                .map(|i| i as i32)
                .collect(),
            other_weights: (0..OTHER_DIMENSIONS * L1)
                .map(|i| (i % 251) as i8)
                .collect(),
            other_psqt: (0..OTHER_DIMENSIONS * PSQT_BUCKETS)
                .map(|i| i as i32)
                .collect(),
        };
        // Pointer arithmetic is in *bytes*, so each row is `idx * len * size_of`.
        let base = ft.psq_weights.as_ptr() as usize;
        let w = ft.weight_row(idx);
        assert_eq!(w.len(), L1);
        assert_eq!(w.as_ptr() as usize - base, idx * L1 * 2);
        let base = ft.other_weights.as_ptr() as usize;
        let o = ft.other_weight_row(idx);
        assert_eq!(o.len(), L1);
        assert_eq!(o.as_ptr() as usize - base, idx * L1 * 1);
        let base = ft.psqt_weights.as_ptr() as usize;
        let p = ft.psqt_weight_row(idx);
        assert_eq!(p.len(), PSQT_BUCKETS);
        assert_eq!(p.as_ptr() as usize - base, idx * PSQT_BUCKETS * 4);
        let base = ft.other_psqt.as_ptr() as usize;
        let q = ft.other_psqt_row(idx);
        assert_eq!(q.len(), PSQT_BUCKETS);
        assert_eq!(q.as_ptr() as usize - base, idx * PSQT_BUCKETS * 4);
        // And the first row really is the base of the block.
        assert_eq!(
            ft.weight_row(0).as_ptr() as usize,
            ft.psq_weights.as_ptr() as usize
        );
        assert_eq!(
            ft.other_weight_row(0).as_ptr() as usize,
            ft.other_weights.as_ptr() as usize
        );
        assert_eq!(
            ft.psqt_weight_row(0).as_ptr() as usize,
            ft.psqt_weights.as_ptr() as usize
        );
        assert_eq!(
            ft.other_psqt_row(0).as_ptr() as usize,
            ft.other_psqt.as_ptr() as usize
        );
    }

    #[test]
    fn affine_skips_zero_inputs_but_still_sums_correctly() {
        // Two outputs, four inputs. Row j is `weights[j * PAD_IN + i]`, and
        // PAD_IN is a multiple of 32, so the padding columns must be ignored.
        const IN: usize = 4;
        const OUT: usize = 2;
        let layer = Affine::<IN, OUT> {
            biases: Box::new([10, -10]),
            weights: (0..OUT * Affine::<IN, OUT>::PAD_IN)
                .map(|i| (i as i32 % 7 - 3) as i8)
                .collect(),
        };
        let input = [0u8, 5, 0, 3];
        let mut out = [0i32; OUT];
        layer.propagate(&input, &mut out);
        for j in 0..OUT {
            let mut expect = [10i32, -10][j];
            for i in 0..IN {
                expect += i32::from(layer.weights[j * Affine::<IN, OUT>::PAD_IN + i])
                    * i32::from(input[i]);
            }
            assert_eq!(out[j], expect, "output {j}");
        }
    }

    #[test]
    fn padding_is_a_multiple_of_thirty_two() {
        assert_eq!(Affine::<64, 32>::PAD_IN, 64);
        assert_eq!(Affine::<128, 1>::PAD_IN, 128);
        assert_eq!(Affine::<33, 4>::PAD_IN, 64);
    }

    #[test]
    fn activations_match_stockfish_semantics() {
        // Clipped: clamp(x >> bits, 0, 127).
        assert_eq!(clipped_relu(-1, 7), 0);
        assert_eq!(clipped_relu(0, 7), 0);
        assert_eq!(clipped_relu(127 << 7, 7), 127);
        assert_eq!(clipped_relu(1 << 20, 7), 127, "saturates at 127");
        assert_eq!(clipped_relu(-(1 << 20), 6), 0);
        // Squared: min(127, x*x >> (2*bits + 7)) — shift 21 at bits = 7, 19 at
        // bits = 6. The "extra 7 bits" are a deliberate approximation the
        // trainer compensates for, so the values look small.
        assert_eq!(sqr_clipped_relu(0, 7), 0);
        assert_eq!(sqr_clipped_relu(-128, 7), 0, "negative squares to zero");
        assert_eq!(sqr_clipped_relu(2048, 7), 2, "2048*2048 >> 21 == 2");
        assert_eq!(sqr_clipped_relu(16384, 7), 127, "saturates at 127");
        assert_eq!(sqr_clipped_relu(1024, 6), 2, "1024*1024 >> 19 == 2");
        assert_eq!(sqr_clipped_relu(8192, 6), 127, "saturates at 127");
        // A shift of 21 really is 21: one bit less would give a different
        // answer, which pins the constant.
        assert_eq!(sqr_clipped_relu(4096, 7), 8, "4096*4096 >> 21 == 8");
    }

    #[test]
    fn feature_transform_clips_both_halves_independently() {
        let mut acc = [0i16; L1];
        // Two units: one saturated pair, one with a negative half.
        acc[0] = 300;
        acc[L1 / 2] = 300;
        acc[1] = -50;
        acc[1 + L1 / 2] = 400;
        acc[2] = 4;
        acc[2 + L1 / 2] = 0;
        let mut out = [0u8; L1];
        Network::transform_perspective(&acc, &mut out, 0);
        assert_eq!(out[0], 127, "255*255/512 == 127");
        assert_eq!(out[1], 0, "a negative half zeroes the product");
        assert_eq!(out[2], 0);
        assert_eq!(out[L1 / 2..], [0u8; L1 / 2], "the second half is untouched");

        // The `offset` selects the perspective half of the same output array.
        let mut out = [0u8; L1];
        Network::transform_perspective(&acc, &mut out, L1 / 2);
        assert_eq!(out[L1 / 2], 127);
        assert_eq!(out[..L1 / 2], [0u8; L1 / 2]);
    }
}
