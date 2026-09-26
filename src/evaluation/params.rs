//! Tunable evaluation parameters (Stage 5).
//!
//! Every important classical-evaluation constant lives in [`EvalParams`]:
//! material values, piece-square tables, mobility weights, pawn-structure
//! terms, passed/protected-pawn bonuses, king safety, piece activity, bishop
//! pair, rook/file bonuses, space, threats, outposts/weak squares, endgame
//! king terms and every MG/EG tapered weight.
//!
//! The struct is serializable to/from a TOML file (see
//! [`EvalParams::to_toml_string`] / [`EvalParams::from_toml_str`]); the
//! shipped baseline file is `config/baseline_eval.toml` and the engine loads
//! any `EvalParamsPath` (UCI option) or `--eval-params` (CLI). The tuning
//! tools read the same format, so a tuned export can be dropped straight into
//! the engine.
//!
//! # Hot path
//!
//! The evaluation submodules read the parameters through a shared `&EvalParams`
//! reference (owned by the [`crate::search::Searcher`] and passed to every
//! worker through [`crate::search::SearchShared`]). Nothing is copied per
//! node; PST tables are plain `[i32; 64]` arrays.
//!
//! # Design notes
//!
//! * **Baseline parity**: [`EvalParams::default`] reproduces the pre-Stage-5
//!   hardcoded constants *exactly* (verified by `params_baseline_reproduces_legacy_eval`),
//!   so tuning is strictly a refinement, never a rewrite.
//! * **Knights/bishops/rooks/queens** historically shared one table between
//!   MG and EG; the parameterization exposes both phases (defaulting EG = MG
//!   for those roles) so the tuner *may* split them without changing the
//!   baseline behaviour.
//! * **Ordering**: [`EvalParams::to_vec`] / [`from_vec`] / [`bounds`] all use
//!   one flat ordering (a round-trip unit test guards against drift) — that
//!   ordering is what the SPSA tuner perturbs.

use serde::{Deserialize, Serialize};
use shakmaty::Role;

use crate::evaluation::Score;

/// `[0, P, N, B, R, Q, 0]` indexed directly by `Role as usize` — kept for
/// backward compatibility with SEE / MVV-LVA / qsearch code paths. The
/// tunable copy lives in [`EvalParams::piece_values`].
pub const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 0];

/// One flat parameter slot used by the tuner: name, current value, bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamDef {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
}

/// A whole-left/whole-right mode indicator for `to_vec`/`from_vec` sections.
impl EvalParams {
    /// Role index used for the `[7]`-sized parameter arrays.
    #[inline]
    fn role_idx(role: Role) -> usize {
        role as usize
    }
}

/// The PST tables: `[role = 1..=6: P N B R Q K][phase: 0 = mg, 1 = eg][square]`
/// for White (mirrored for Black). Role 0 is unused and kept zeroed.
///
/// serde only derives `[T; N]` up to `N = 32`, so the on-disk TOML form is a
/// nested `Vec` while the hot-path in-memory layout stays plain `[i32; 64]`
/// arrays (the conversion happens only when (de)serializing a parameter file).
#[derive(Debug, Clone)]
pub struct PstTables {
    /// Indexed `[role as usize][phase][square]`.
    pub tables: [[[i32; 64]; 2]; 7],
}

impl Default for PstTables {
    fn default() -> PstTables {
        PstTables {
            tables: [[[0i32; 64]; 2]; 7],
        }
    }
}

impl Serialize for PstTables {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let nested: Vec<Vec<Vec<i32>>> = self
            .tables
            .iter()
            .map(|phases| phases.iter().map(|table| table.to_vec()).collect())
            .collect();
        nested.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PstTables {
    fn deserialize<D>(deserializer: D) -> Result<PstTables, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let nested: Vec<Vec<Vec<i32>>> = Vec::deserialize(deserializer)?;
        if nested.len() != 7 {
            return Err(serde::de::Error::custom(format!(
                "pst: expected 7 role tables, got {}",
                nested.len()
            )));
        }
        let mut tables = [[[0i32; 64]; 2]; 7];
        for (role, phases) in nested.into_iter().enumerate() {
            if phases.len() != 2 {
                return Err(serde::de::Error::custom(format!(
                    "pst.role {role}: expected 2 phases, got {}",
                    phases.len()
                )));
            }
            for (phase, squares) in phases.into_iter().enumerate() {
                if squares.len() != 64 {
                    return Err(serde::de::Error::custom(format!(
                        "pst.role {role}.phase {phase}: expected 64 squares, got {}",
                        squares.len()
                    )));
                }
                tables[role][phase] = squares.try_into().map_err(|sqs: Vec<i32>| {
                    serde::de::Error::custom(format!(
                        "pst.role {role}.phase {phase}: expected 64 squares, got {}",
                        sqs.len()
                    ))
                })?;
            }
        }
        Ok(PstTables { tables })
    }
}

/// All tunable evaluation parameters (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalParams {
    // --- Material -------------------------------------------------------
    /// Piece values `[pawn, knight, bishop, rook, queen]` (centipawns, both
    /// phases — material is not tapered).
    pub piece_values: [i32; 5],
    /// Bishop-pair bonus `[mg, eg]`.
    pub bishop_pair: [i32; 2],

    // --- Piece-square tables -------------------------------------------
    /// PST values, see [`PstTables`].
    pub pst: PstTables,

    // --- Mobility -------------------------------------------------------
    /// Mobility weight `[role = 1..=5: N B R Q K][mg, eg]` (centipawns per
    /// extra attacked square).
    pub mobility: [[i32; 2]; 7],

    // --- Pawn structure --------------------------------------------------
    pub doubled: [i32; 2],
    pub isolated: [i32; 2],
    pub backward: [i32; 2],
    pub connected: [i32; 2],
    /// Per-island penalty *beyond the first*.
    pub island: [i32; 2],
    pub protected_pawn: [i32; 2],

    // --- Passed pawns ----------------------------------------------------
    /// Rank bonus `[rank - 2]` for ranks 2..=7, `[mg, eg]` each.
    pub passed_bonus: [[i32; 2]; 6],
    pub protected_passed: [i32; 2],
    pub connected_passed: [i32; 2],

    // --- King safety (midgame-oriented) ----------------------------------
    /// Cap of the per-king pawn-shield bonus.
    pub shield_cap: i32,
    /// Shield weight per rank in front of the king (`dist` 1..=3).
    pub shield_weight: [i32; 3],
    /// King-ring attack weight `[role = 1..=5: P N B R Q]`.
    pub attack_weight: [i32; 7],
    pub king_open_file: i32,
    pub king_fully_open_file: i32,
    pub king_open_files_cap: i32,
    pub king_hole: i32,
    pub king_holes_cap: i32,
    pub king_center_bonus: i32,
    pub king_passer_dist_penalty: i32,

    // --- Piece activity / coordination -------------------------------------
    pub knight_outpost: [i32; 2],
    pub bishop_outpost: [i32; 2],
    pub bad_bishop_pawn: [i32; 2],
    pub bad_bishop_cap: i32,
    /// Opening gate: the development term only applies when phase ≥ this.
    pub development_phase: i32,
    pub undeveloped: [i32; 2],
    pub rooks_connected: [i32; 2],
    pub rook_behind_passed: [i32; 2],
    pub queen_behind_passed: [i32; 2],
    pub queen_under_pawn_attack: [i32; 2],
    pub minor_coordination: [i32; 2],

    // --- Threats / rook files / seventh rank -------------------------------
    pub weak_minor: i32,
    pub pawn_attack_bonus: i32,
    pub rook_open: [i32; 2],
    pub rook_semi_open: [i32; 2],
    pub rook_seventh: [i32; 2],

    // --- Space ------------------------------------------------------------
    /// Centipawns per space unit, MG and EG.
    pub space_mg: i32,
    pub space_eg: i32,

    // --- Phase (taper weights) ----------------------------------------------
    pub phase_queen: i32,
    pub phase_rook: i32,
    pub phase_bishop: i32,
    pub phase_knight: i32,
    pub phase_max: i32,
}

/// Convenience accessors used by the evaluation submodules.
impl EvalParams {
    /// Material value of `role` (shared by MG and EG).
    #[inline]
    pub fn piece_value(&self, role: Role) -> i32 {
        if role == Role::King {
            0
        } else {
            self.piece_values[role_idx_of(role)]
        }
    }

    /// Bishop-pair bonus as a tapered [`Score`].
    #[inline]
    pub fn bishop_pair_score(&self) -> Score {
        Score::new(self.bishop_pair[0], self.bishop_pair[1])
    }

    /// PST value of `role` on `sq`-index (white-oriented) for one phase.
    #[inline]
    pub fn pst_value(&self, role: Role, phase: usize, idx: usize) -> i32 {
        self.pst.tables[Self::role_idx(role)][phase][idx]
    }

    /// Mobility weight for `role` as a `(mg, eg)` pair.
    #[inline]
    pub fn mobility_weight(&self, role: Role) -> (i32, i32) {
        let w = &self.mobility[Self::role_idx(role)];
        (w[0], w[1])
    }
}

/// `Role as usize - 1`: P=0, N=1, B=2, R=3, Q=4 (King excluded).
#[inline]
fn role_idx_of(role: Role) -> usize {
    role as usize - 1
}

/// Builds a PST table from a legacy `[i16; 64]` const.
pub(crate) const fn pst_from(table: &[i16; 64]) -> [i32; 64] {
    let mut out = [0i32; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = table[i] as i32;
        i += 1;
    }
    out
}

// Legacy tables the baseline defaults are taken from (kept private here so
// the parameter file is the single tunable source of truth).
use crate::evaluation::pieces::LEGACY_PST;

impl Default for EvalParams {
    fn default() -> EvalParams {
        let legacy = LEGACY_PST;
        // Default: knights/bishops/rooks/queens share one table between
        // phases (exactly the pre-Stage-5 behaviour); pawns and kings have
        // dedicated MG and EG tables.
        let mut pst = [[[0i32; 64]; 2]; 7];
        for (role, mg, eg) in [
            (1usize, legacy.pawn_mg, legacy.pawn_eg),
            (2usize, legacy.knight, legacy.knight),
            (3usize, legacy.bishop, legacy.bishop),
            (4usize, legacy.rook, legacy.rook),
            (5usize, legacy.queen, legacy.queen),
            (6usize, legacy.king_mg, legacy.king_eg),
        ] {
            pst[role][0] = pst_from(&mg);
            pst[role][1] = pst_from(&eg);
        }

        let mut mobility = [[0i32; 2]; 7];
        for (role, w) in [
            (2usize, [4, 4]), // knight
            (3usize, [4, 5]), // bishop
            (4usize, [2, 3]), // rook
            (5usize, [1, 2]), // queen
            (6usize, [0, 1]), // king (endgame activity)
        ] {
            mobility[role] = w;
        }

        let mut attack_weight = [0i32; 7];
        for (role, w) in [
            (1usize, 2), // pawn
            (2usize, 3), // knight
            (3usize, 3), // bishop
            (4usize, 4), // rook
            (5usize, 5), // queen
        ] {
            attack_weight[role] = w;
        }

        EvalParams {
            piece_values: [100, 320, 330, 500, 900],
            bishop_pair: [40, 60],
            pst: PstTables { tables: pst },
            mobility,
            doubled: [-20, -40],
            isolated: [-20, -30],
            backward: [-10, -15],
            connected: [15, 30],
            island: [-10, -15],
            protected_pawn: [6, 6],
            passed_bonus: [
                [0, 10],    // rank 2
                [10, 40],   // rank 3
                [25, 100],  // rank 4
                [50, 250],  // rank 5
                [110, 500], // rank 6
                [180, 800], // rank 7
            ],
            protected_passed: [10, 30],
            connected_passed: [15, 30],
            shield_cap: 40,
            shield_weight: [4, 3, 2],
            attack_weight,
            king_open_file: 8,
            king_fully_open_file: 5,
            king_open_files_cap: 30,
            king_hole: 6,
            king_holes_cap: 30,
            king_center_bonus: 3,
            king_passer_dist_penalty: 4,
            knight_outpost: [30, 20],
            bishop_outpost: [20, 10],
            bad_bishop_pawn: [-8, -4],
            bad_bishop_cap: 4,
            development_phase: 20,
            undeveloped: [-15, 0],
            rooks_connected: [15, 20],
            rook_behind_passed: [10, 30],
            queen_behind_passed: [15, 20],
            queen_under_pawn_attack: [-20, -10],
            minor_coordination: [5, 8],
            weak_minor: -15,
            pawn_attack_bonus: 18,
            rook_open: [25, 20],
            rook_semi_open: [10, 10],
            rook_seventh: [20, 40],
            space_mg: 4,
            space_eg: 2,
            phase_queen: 4,
            phase_rook: 2,
            phase_bishop: 1,
            phase_knight: 1,
            phase_max: 24,
        }
    }
}

impl EvalParams {
    /// The standard baseline (same contents as [`EvalParams::default`]).
    pub fn baseline() -> EvalParams {
        EvalParams::default()
    }

    /// Serializes to a TOML string (the `config/baseline_eval.toml` format).
    pub fn to_toml_string(&self) -> anyhow::Result<String> {
        toml::to_string(self).map_err(Into::into)
    }

    /// Parses a TOML string. Unknown fields are an error (guards against
    /// silently ignoring a malformed tuned file).
    pub fn from_toml_str(s: &str) -> anyhow::Result<EvalParams> {
        toml::from_str(s).map_err(Into::into)
    }

    /// Loads parameters from a TOML file.
    pub fn load(path: &str) -> anyhow::Result<EvalParams> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read eval params {path:?}: {e}"))?;
        EvalParams::from_toml_str(&s)
            .map_err(|e| anyhow::anyhow!("invalid eval params {path:?}: {e}"))
    }

    /// Writes parameters to a TOML file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let s = self.to_toml_string()?;
        std::fs::write(path, s)
            .map_err(|e| anyhow::anyhow!("cannot write eval params {path:?}: {e}"))
    }

    // --- Flat parameter vector (tuner interface) -------------------------

    /// The number of tunable parameters.
    pub fn param_count(&self) -> usize {
        self.to_vec().len()
    }

    /// The flat vector of all tunable parameters, in [`EvalParams::bounds`]
    /// order.
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(900);
        self.push_all(&mut v);
        v
    }

    /// Writes `v` (in [`EvalParams::bounds`] order) back into this struct.
    /// Values are not clamped here (the tuner clamps); out-of-range values
    /// stay as-is for diagnostics.
    pub fn from_vec(&mut self, v: &[f64]) {
        let mut i = 0;
        self.pop_all(v, &mut i);
    }

    /// Bounds (and names) of every tunable parameter, in [`EvalParams::to_vec`]
    /// order.
    pub fn bounds() -> Vec<ParamDef> {
        let mut out = Vec::with_capacity(900);
        // Scalars.
        for (name, min, max) in [
            ("material.pawn", 30.0, 200.0),
            ("material.knight", 120.0, 600.0),
            ("material.bishop", 120.0, 600.0),
            ("material.rook", 250.0, 900.0),
            ("material.queen", 600.0, 1500.0),
            ("bishop_pair.mg", 0.0, 200.0),
            ("bishop_pair.eg", 0.0, 250.0),
            ("pawns.doubled.mg", -150.0, 0.0),
            ("pawns.doubled.eg", -250.0, 0.0),
            ("pawns.isolated.mg", -150.0, 0.0),
            ("pawns.isolated.eg", -250.0, 0.0),
            ("pawns.backward.mg", -120.0, 0.0),
            ("pawns.backward.eg", -200.0, 0.0),
            ("pawns.connected.mg", 0.0, 150.0),
            ("pawns.connected.eg", 0.0, 250.0),
            ("pawns.island.mg", -120.0, 0.0),
            ("pawns.island.eg", -200.0, 0.0),
            ("pawns.protected_pawn.mg", 0.0, 60.0),
            ("pawns.protected_pawn.eg", 0.0, 80.0),
            ("passed.protected.mg", 0.0, 100.0),
            ("passed.protected.eg", 0.0, 200.0),
            ("passed.connected.mg", 0.0, 120.0),
            ("passed.connected.eg", 0.0, 250.0),
            ("king.shield_cap", 0.0, 250.0),
            ("king.shield_rank1", 0.0, 30.0),
            ("king.shield_rank2", 0.0, 30.0),
            ("king.shield_rank3", 0.0, 30.0),
            ("king.attack_pawn", 0.0, 30.0),
            ("king.attack_knight", 0.0, 30.0),
            ("king.attack_bishop", 0.0, 30.0),
            ("king.attack_rook", 0.0, 40.0),
            ("king.attack_queen", 0.0, 50.0),
            ("king.open_file", 0.0, 60.0),
            ("king.fully_open_file", 0.0, 40.0),
            ("king.open_files_cap", 0.0, 200.0),
            ("king.hole", 0.0, 40.0),
            ("king.holes_cap", 0.0, 200.0),
            ("king.center_bonus", 0.0, 30.0),
            ("king.passer_dist_penalty", 0.0, 40.0),
            ("activity.knight_outpost.mg", -150.0, 150.0),
            ("activity.knight_outpost.eg", -150.0, 200.0),
            ("activity.bishop_outpost.mg", -100.0, 120.0),
            ("activity.bishop_outpost.eg", -100.0, 150.0),
            ("activity.bad_bishop.mg", -120.0, 0.0),
            ("activity.bad_bishop.eg", -160.0, 0.0),
            ("activity.bad_bishop_cap", 1.0, 8.0),
            ("activity.development_phase", 0.0, 24.0),
            ("activity.undeveloped.mg", -120.0, 0.0),
            ("activity.undeveloped.eg", -100.0, 0.0),
            ("activity.rooks_connected.mg", 0.0, 120.0),
            ("activity.rooks_connected.eg", 0.0, 180.0),
            ("activity.rook_behind_passed.mg", 0.0, 150.0),
            ("activity.rook_behind_passed.eg", 0.0, 250.0),
            ("activity.queen_behind_passed.mg", 0.0, 120.0),
            ("activity.queen_behind_passed.eg", 0.0, 200.0),
            ("activity.queen_under_pawn_attack.mg", -150.0, 0.0),
            ("activity.queen_under_pawn_attack.eg", -120.0, 0.0),
            ("activity.minor_coordination.mg", 0.0, 60.0),
            ("activity.minor_coordination.eg", 0.0, 80.0),
            ("threats.weak_minor", -200.0, 0.0),
            ("threats.pawn_attack_bonus", 0.0, 120.0),
            ("threats.rook_open.mg", 0.0, 120.0),
            ("threats.rook_open.eg", 0.0, 150.0),
            ("threats.rook_semi_open.mg", 0.0, 80.0),
            ("threats.rook_semi_open.eg", 0.0, 100.0),
            ("threats.rook_seventh.mg", 0.0, 120.0),
            ("threats.rook_seventh.eg", 0.0, 250.0),
            ("space.mg", 0.0, 60.0),
            ("space.eg", 0.0, 60.0),
            ("phase.queen", 0.0, 8.0),
            ("phase.rook", 0.0, 6.0),
            ("phase.bishop", 0.0, 4.0),
            ("phase.knight", 0.0, 4.0),
            ("phase.max", 16.0, 32.0),
        ] {
            out.push(ParamDef { name, min, max });
        }
        // Passed-pawn rank bonuses.
        for r in 0..6 {
            out.push(ParamDef {
                name: concat_names(&format!("passed.rank{}_mg", r + 2)),
                min: 0.0,
                max: 500.0,
            });
            out.push(ParamDef {
                name: concat_names(&format!("passed.rank{}_eg", r + 2)),
                min: 0.0,
                max: 1400.0,
            });
        }
        // Mobility.
        for (name, max) in [
            ("mobility.knight", 20.0),
            ("mobility.bishop", 20.0),
            ("mobility.rook", 15.0),
            ("mobility.queen", 12.0),
            ("mobility.king", 10.0),
        ] {
            for phase in ["mg", "eg"] {
                out.push(ParamDef {
                    name: concat_names(&format!("{name}.{phase}")),
                    min: 0.0,
                    max,
                });
            }
        }
        // PST (role 1..=6, phase 0..=1, square 0..=63).
        let role_names = ["", "pawn", "knight", "bishop", "rook", "queen", "king"];
        let phase_names = ["mg", "eg"];
        for role in 1..=6usize {
            for phase in 0..=1usize {
                for sq in 0..64usize {
                    let (min, max) = pst_bounds(role);
                    out.push(ParamDef {
                        name: concat_names(&format!(
                            "pst.{}.{}.{}",
                            role_names[role], phase_names[phase], sq
                        )),
                        min,
                        max,
                    });
                }
            }
        }
        debug_assert_eq!(out.len(), 864, "parameter count drift");
        out
    }

    // --- internal: flat push/pop in the same order as bounds() -----------

    fn push_all(&self, v: &mut Vec<f64>) {
        // 1. material.
        for i in 0..5 {
            v.push(self.piece_values[i] as f64);
        }
        // 2. bishop pair.
        v.push(self.bishop_pair[0] as f64);
        v.push(self.bishop_pair[1] as f64);
        // 3. pawn structure.
        push_p((v, &self.doubled));
        push_p((v, &self.isolated));
        push_p((v, &self.backward));
        push_p((v, &self.connected));
        push_p((v, &self.island));
        push_p((v, &self.protected_pawn));
        // 4. passed pawns (protected + connected).
        push_p((v, &self.protected_passed));
        push_p((v, &self.connected_passed));
        // 5. king safety.
        v.push(self.shield_cap as f64);
        for i in 0..3 {
            v.push(self.shield_weight[i] as f64);
        }
        for role in 1..=5usize {
            v.push(self.attack_weight[role] as f64);
        }
        v.push(self.king_open_file as f64);
        v.push(self.king_fully_open_file as f64);
        v.push(self.king_open_files_cap as f64);
        v.push(self.king_hole as f64);
        v.push(self.king_holes_cap as f64);
        v.push(self.king_center_bonus as f64);
        v.push(self.king_passer_dist_penalty as f64);
        // 6. piece activity / coordination.
        push_p((v, &self.knight_outpost));
        push_p((v, &self.bishop_outpost));
        push_p((v, &self.bad_bishop_pawn));
        v.push(self.bad_bishop_cap as f64);
        v.push(self.development_phase as f64);
        push_p((v, &self.undeveloped));
        push_p((v, &self.rooks_connected));
        push_p((v, &self.rook_behind_passed));
        push_p((v, &self.queen_behind_passed));
        push_p((v, &self.queen_under_pawn_attack));
        push_p((v, &self.minor_coordination));
        // 7. threats / rook files.
        v.push(self.weak_minor as f64);
        v.push(self.pawn_attack_bonus as f64);
        push_p((v, &self.rook_open));
        push_p((v, &self.rook_semi_open));
        push_p((v, &self.rook_seventh));
        // 8. space.
        v.push(self.space_mg as f64);
        v.push(self.space_eg as f64);
        // 9. phase weights.
        v.push(self.phase_queen as f64);
        v.push(self.phase_rook as f64);
        v.push(self.phase_bishop as f64);
        v.push(self.phase_knight as f64);
        v.push(self.phase_max as f64);
        // 10. passed-pawn rank bonuses.
        for r in 0..6 {
            push_p((v, &self.passed_bonus[r]));
        }
        // 11. mobility.
        for role in 1..=5usize {
            v.push(self.mobility[role][0] as f64);
            v.push(self.mobility[role][1] as f64);
        }
        // 12. PST.
        for role in 1..=6usize {
            for phase in 0..=1usize {
                for sq in 0..64usize {
                    v.push(self.pst.tables[role][phase][sq] as f64);
                }
            }
        }
    }

    /// Reads slots in [`EvalParams::push_all`] order. Missing trailing values
    /// are skipped so a `from_vec` with a shorter vector leaves the remaining
    /// parameters untouched.
    fn pop_all(&mut self, v: &[f64], i: &mut usize) {
        let mut next = || -> Option<f64> {
            let x = v.get(*i).copied();
            *i += 1;
            x
        };
        // 1. material.
        for slot in self.piece_values.iter_mut() {
            if let Some(x) = next() {
                *slot = x as i32;
            }
        }
        // 2. bishop pair.
        if let Some(x) = next() {
            self.bishop_pair[0] = x as i32;
        }
        if let Some(x) = next() {
            self.bishop_pair[1] = x as i32;
        }
        // 3. pawn structure.
        pop_p(v, &mut self.doubled, &mut next);
        pop_p(v, &mut self.isolated, &mut next);
        pop_p(v, &mut self.backward, &mut next);
        pop_p(v, &mut self.connected, &mut next);
        pop_p(v, &mut self.island, &mut next);
        pop_p(v, &mut self.protected_pawn, &mut next);
        // 4. passed pawns (protected + connected).
        pop_p(v, &mut self.protected_passed, &mut next);
        pop_p(v, &mut self.connected_passed, &mut next);
        // 5. king safety.
        if let Some(x) = next() {
            self.shield_cap = x as i32;
        }
        for i in 0..3 {
            if let Some(x) = next() {
                self.shield_weight[i] = x as i32;
            }
        }
        for role in 1..=5usize {
            if let Some(x) = next() {
                self.attack_weight[role] = x as i32;
            }
        }
        if let Some(x) = next() {
            self.king_open_file = x as i32;
        }
        if let Some(x) = next() {
            self.king_fully_open_file = x as i32;
        }
        if let Some(x) = next() {
            self.king_open_files_cap = x as i32;
        }
        if let Some(x) = next() {
            self.king_hole = x as i32;
        }
        if let Some(x) = next() {
            self.king_holes_cap = x as i32;
        }
        if let Some(x) = next() {
            self.king_center_bonus = x as i32;
        }
        if let Some(x) = next() {
            self.king_passer_dist_penalty = x as i32;
        }
        // 6. piece activity / coordination.
        pop_p(v, &mut self.knight_outpost, &mut next);
        pop_p(v, &mut self.bishop_outpost, &mut next);
        pop_p(v, &mut self.bad_bishop_pawn, &mut next);
        if let Some(x) = next() {
            self.bad_bishop_cap = x as i32;
        }
        if let Some(x) = next() {
            self.development_phase = x as i32;
        }
        pop_p(v, &mut self.undeveloped, &mut next);
        pop_p(v, &mut self.rooks_connected, &mut next);
        pop_p(v, &mut self.rook_behind_passed, &mut next);
        pop_p(v, &mut self.queen_behind_passed, &mut next);
        pop_p(v, &mut self.queen_under_pawn_attack, &mut next);
        pop_p(v, &mut self.minor_coordination, &mut next);
        // 7. threats / rook files.
        if let Some(x) = next() {
            self.weak_minor = x as i32;
        }
        if let Some(x) = next() {
            self.pawn_attack_bonus = x as i32;
        }
        pop_p(v, &mut self.rook_open, &mut next);
        pop_p(v, &mut self.rook_semi_open, &mut next);
        pop_p(v, &mut self.rook_seventh, &mut next);
        // 8. space.
        if let Some(x) = next() {
            self.space_mg = x as i32;
        }
        if let Some(x) = next() {
            self.space_eg = x as i32;
        }
        // 9. phase weights.
        if let Some(x) = next() {
            self.phase_queen = x as i32;
        }
        if let Some(x) = next() {
            self.phase_rook = x as i32;
        }
        if let Some(x) = next() {
            self.phase_bishop = x as i32;
        }
        if let Some(x) = next() {
            self.phase_knight = x as i32;
        }
        if let Some(x) = next() {
            self.phase_max = x as i32;
        }
        // 10. passed-pawn rank bonuses.
        for r in 0..6 {
            pop_p(v, &mut self.passed_bonus[r], &mut next);
        }
        // 11. mobility.
        for role in 1..=5usize {
            if let Some(x) = next() {
                self.mobility[role][0] = x as i32;
            }
            if let Some(x) = next() {
                self.mobility[role][1] = x as i32;
            }
        }
        // 12. PST.
        for role in 1..=6usize {
            for phase in 0..=1usize {
                for sq in 0..64usize {
                    if let Some(x) = next() {
                        self.pst.tables[role][phase][sq] = x as i32;
                    }
                }
            }
        }
    }
}

/// Leaks a `String` into a `'static str` for `ParamDef` names (the tuner only
/// ever reads them; the count of leaked strings is bounded and tiny).
fn concat_names(s: &String) -> &'static str {
    Box::leak(s.clone().into_boxed_str())
}

/// PST bounds: material-sized swings for majors, smaller for pawns/kings.
fn pst_bounds(role: usize) -> (f64, f64) {
    match role {
        1 => (-300.0, 500.0), // pawn: promotions push values up
        6 => (-500.0, 500.0), // king: endgame centralization is large
        _ => (-400.0, 400.0),
    }
}

#[inline]
fn push_p((v, p): (&mut Vec<f64>, &[i32; 2])) {
    v.push(p[0] as f64);
    v.push(p[1] as f64);
}

#[inline]
fn pop_p(v: &[f64], p: &mut [i32; 2], next: &mut impl FnMut() -> Option<f64>) {
    let _ = v;
    if let Some(x) = next() {
        p[0] = x as i32;
    }
    if let Some(x) = next() {
        p[1] = x as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_vector_round_trips() {
        let p = EvalParams::default();
        let v = p.to_vec();
        let mut q = EvalParams::default();
        q.from_vec(&v);
        assert_eq!(p.to_vec(), q.to_vec(), "round-trip must be lossless");
    }

    #[test]
    fn param_vector_and_bounds_agree_in_length() {
        let v = EvalParams::default().to_vec();
        let b = EvalParams::bounds();
        assert_eq!(v.len(), b.len(), "vector and bounds must share ordering");
        assert_eq!(v.len(), 864);
        // Every bound is a valid interval and contains the baseline.
        let base = EvalParams::default().to_vec();
        for (i, def) in b.iter().enumerate() {
            assert!(def.min <= def.max, "invalid bounds for {}", def.name);
            assert!(
                base[i] >= def.min - 1e-9 && base[i] <= def.max + 1e-9,
                "baseline {} outside [{}, {}]",
                def.name,
                def.min,
                def.max
            );
        }
    }

    #[test]
    fn from_vec_applies_values() {
        let mut p = EvalParams::default();
        let mut v = p.to_vec();
        v[0] = 123.0; // material.pawn
        p.from_vec(&v);
        assert_eq!(p.piece_values[0], 123);
        // Sanity: a partial vector leaves the rest untouched.
        let mut q = EvalParams::default();
        q.from_vec(&[]);
        assert_eq!(q.piece_values[0], 100);
    }

    #[test]
    fn toml_round_trip_is_byte_equivalent() {
        let p = EvalParams::default();
        let s = p.to_toml_string().unwrap();
        let q = EvalParams::from_toml_str(&s).unwrap();
        assert_eq!(p.to_vec(), q.to_vec());
    }

    #[test]
    fn default_matches_legacy_material_values() {
        let p = EvalParams::default();
        for role in [
            Role::Pawn,
            Role::Knight,
            Role::Bishop,
            Role::Rook,
            Role::Queen,
        ] {
            assert_eq!(p.piece_value(role), PIECE_VALUES[role as usize]);
        }
    }

    #[test]
    fn pst_defaults_share_tables_for_majors() {
        // Baseline behaviour: majors use the same table for MG and EG.
        let p = EvalParams::default();
        for role in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            for sq in 0..64 {
                assert_eq!(
                    p.pst.tables[role as usize][0][sq], p.pst.tables[role as usize][1][sq],
                    "majors share a table at {sq}"
                );
            }
        }
    }
}
