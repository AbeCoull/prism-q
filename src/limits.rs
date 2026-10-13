//! Resource limits: a memory budget and per-cap overrides, set for one request with
//! [`Simulate::limits`](crate::Simulate::limits) or for the process with
//! [`set_default_limits`].

use std::mem::size_of;
use std::sync::{OnceLock, PoisonError, RwLock};

use num_complex::Complex64;

use crate::backend::memory;
use crate::error::{PrismError, ResourceKind};

/// Memory budget and per-cap overrides for a simulation.
///
/// A cap left unset derives from the memory budget, which defaults to half of physical
/// memory. Layers resolve from the request ([`Simulate::limits`]) down through the
/// process default ([`set_default_limits`]) and the `PRISM_MAX_*` environment variables
/// to detection, and the highest layer that sets a value wins. A layer that sets
/// `memory_budget_bytes` re-derives every memory cap from its budget and discards the
/// per-cap values of the layers below it; its own per-cap values still apply. The slice
/// budget and the bond dimension are not memory caps and survive a budget.
///
/// Each field names the environment variable it overrides.
///
/// [`Simulate::limits`]: crate::Simulate::limits
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// Bytes every memory cap derives from: a cap is the widest allocation of its kind
    /// that fits.
    pub memory_budget_bytes: Option<u64>,
    /// Dense statevector width (`PRISM_MAX_SV_QUBITS`). Auto routing sends a wider
    /// circuit to the sparse or MPS backend.
    pub max_statevector_qubits: Option<usize>,
    /// Density-matrix width (`PRISM_MAX_DM_QUBITS`), held to half the statevector cap
    /// because the `4^n` mixture is stored as a `2n`-qubit statevector.
    pub max_density_matrix_qubits: Option<usize>,
    /// Width of a dense probability output (`PRISM_MAX_PROB_QUBITS`).
    pub max_probability_qubits: Option<usize>,
    /// Width of a dense statevector export (`PRISM_MAX_EXPORT_QUBITS`).
    pub max_export_qubits: Option<usize>,
    /// Sparse map size as a power of two (`PRISM_MAX_SPARSE_QUBITS`): at most `2^q`
    /// stored entries.
    pub max_sparse_qubits: Option<usize>,
    /// Width of a merged factored block (`PRISM_MAX_FACTORED_MERGE_QUBITS`).
    pub max_factored_merge_qubits: Option<usize>,
    /// Width of a merged stabilizer cluster (`PRISM_MAX_STABILIZER_CLUSTER_QUBITS`).
    pub max_stabilizer_cluster_qubits: Option<usize>,
    /// MPS contraction workspace as `2^q` amplitudes (`PRISM_MAX_MPS_WORKSPACE_QUBITS`).
    pub max_mps_workspace_qubits: Option<usize>,
    /// Tensor-network peak intermediate as `2^q` elements (`PRISM_MAX_TN_PEAK_QUBITS`).
    pub max_tn_peak_qubits: Option<usize>,
    /// Slices a tensor-network contraction may sum over before its peak is reported
    /// over the cap (`PRISM_MAX_TN_SLICES`, default 1024). Zero reads as one, which
    /// turns slicing off.
    pub max_tn_slices: Option<usize>,
    /// Bond dimension of the MPS that auto routing picks past the statevector cap
    /// (default 256). Zero reads as one.
    pub auto_mps_bond_dim: Option<usize>,
}

impl Limits {
    const NONE: Limits = Limits {
        memory_budget_bytes: None,
        max_statevector_qubits: None,
        max_density_matrix_qubits: None,
        max_probability_qubits: None,
        max_export_qubits: None,
        max_sparse_qubits: None,
        max_factored_merge_qubits: None,
        max_stabilizer_cluster_qubits: None,
        max_mps_workspace_qubits: None,
        max_tn_peak_qubits: None,
        max_tn_slices: None,
        auto_mps_bond_dim: None,
    };

    /// Limits that set nothing, so every value comes from the layers below.
    pub fn new() -> Self {
        Self::NONE
    }

    pub fn memory_budget_bytes(mut self, bytes: u64) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    pub fn max_statevector_qubits(mut self, qubits: usize) -> Self {
        self.max_statevector_qubits = Some(qubits);
        self
    }

    pub fn max_density_matrix_qubits(mut self, qubits: usize) -> Self {
        self.max_density_matrix_qubits = Some(qubits);
        self
    }

    pub fn max_probability_qubits(mut self, qubits: usize) -> Self {
        self.max_probability_qubits = Some(qubits);
        self
    }

    pub fn max_export_qubits(mut self, qubits: usize) -> Self {
        self.max_export_qubits = Some(qubits);
        self
    }

    pub fn max_sparse_qubits(mut self, qubits: usize) -> Self {
        self.max_sparse_qubits = Some(qubits);
        self
    }

    pub fn max_factored_merge_qubits(mut self, qubits: usize) -> Self {
        self.max_factored_merge_qubits = Some(qubits);
        self
    }

    pub fn max_stabilizer_cluster_qubits(mut self, qubits: usize) -> Self {
        self.max_stabilizer_cluster_qubits = Some(qubits);
        self
    }

    pub fn max_mps_workspace_qubits(mut self, qubits: usize) -> Self {
        self.max_mps_workspace_qubits = Some(qubits);
        self
    }

    pub fn max_tn_peak_qubits(mut self, qubits: usize) -> Self {
        self.max_tn_peak_qubits = Some(qubits);
        self
    }

    pub fn max_tn_slices(mut self, slices: usize) -> Self {
        self.max_tn_slices = Some(slices);
        self
    }

    pub fn auto_mps_bond_dim(mut self, bond_dim: usize) -> Self {
        self.auto_mps_bond_dim = Some(bond_dim);
        self
    }
}

/// Layer a resolved cap came from, lowest precedence first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LimitSource {
    /// Derived from detected physical memory, or the built-in default of a count that
    /// is not a memory cap.
    Detected,
    /// A `PRISM_MAX_*` environment variable.
    Env,
    /// [`set_default_limits`].
    ProcessDefault,
    /// The request's own [`Simulate::limits`](crate::Simulate::limits).
    Request,
}

/// Cap a [`ResourceLimit`](crate::ResourceLimit) error ran into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LimitKnob {
    StatevectorQubits,
    DensityMatrixQubits,
    ProbabilityQubits,
    ExportQubits,
    SparseQubits,
    FactoredMergeQubits,
    StabilizerClusterQubits,
    MpsWorkspaceQubits,
    TnPeakQubits,
}

impl LimitKnob {
    /// Name of the [`Limits`] field and setter that moves this cap.
    pub fn name(self) -> &'static str {
        match self {
            LimitKnob::StatevectorQubits => "max_statevector_qubits",
            LimitKnob::DensityMatrixQubits => "max_density_matrix_qubits",
            LimitKnob::ProbabilityQubits => "max_probability_qubits",
            LimitKnob::ExportQubits => "max_export_qubits",
            LimitKnob::SparseQubits => "max_sparse_qubits",
            LimitKnob::FactoredMergeQubits => "max_factored_merge_qubits",
            LimitKnob::StabilizerClusterQubits => "max_stabilizer_cluster_qubits",
            LimitKnob::MpsWorkspaceQubits => "max_mps_workspace_qubits",
            LimitKnob::TnPeakQubits => "max_tn_peak_qubits",
        }
    }

    /// Environment advice for the cap. Both density-matrix variables bind, since that
    /// cap is held to half the statevector cap.
    pub(crate) fn env_var(self) -> &'static str {
        match self {
            LimitKnob::StatevectorQubits => "PRISM_MAX_SV_QUBITS",
            LimitKnob::DensityMatrixQubits => "PRISM_MAX_DM_QUBITS and PRISM_MAX_SV_QUBITS",
            LimitKnob::ProbabilityQubits => "PRISM_MAX_PROB_QUBITS",
            LimitKnob::ExportQubits => "PRISM_MAX_EXPORT_QUBITS",
            LimitKnob::SparseQubits => "PRISM_MAX_SPARSE_QUBITS",
            LimitKnob::FactoredMergeQubits => "PRISM_MAX_FACTORED_MERGE_QUBITS",
            LimitKnob::StabilizerClusterQubits => "PRISM_MAX_STABILIZER_CLUSTER_QUBITS",
            LimitKnob::MpsWorkspaceQubits => "PRISM_MAX_MPS_WORKSPACE_QUBITS",
            LimitKnob::TnPeakQubits => "PRISM_MAX_TN_PEAK_QUBITS",
        }
    }
}

static PROCESS_DEFAULT: RwLock<Limits> = RwLock::new(Limits::NONE);

/// Replace the process default limits, which every request without its own
/// [`Simulate::limits`](crate::Simulate::limits) resolves against, along with
/// [`run_qasm`](crate::run_qasm), batches, dynamic programs and the QEC runner.
///
/// Takes effect on the next call that resolves limits; a backend already built keeps
/// the caps it was built with. [`Limits::new`] restores the environment and detected
/// values.
pub fn set_default_limits(limits: Limits) {
    *PROCESS_DEFAULT
        .write()
        .unwrap_or_else(PoisonError::into_inner) = limits;
}

/// The limits last passed to [`set_default_limits`], empty when it was never called.
pub fn default_limits() -> Limits {
    *PROCESS_DEFAULT
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

/// One resolved cap and the layer it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limit {
    pub(crate) value: usize,
    pub(crate) source: LimitSource,
}

/// Every cap one request runs under, resolved once per call and passed by reference
/// to dispatch and the backends it builds. Declared `pub` only so the sealed
/// `Backend` supertrait can return it; this module is private, so it has no public
/// path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLimits {
    pub(crate) statevector_qubits: Limit,
    pub(crate) density_matrix_qubits: Limit,
    pub(crate) probability_qubits: Limit,
    /// The probability cap priced for the tensor network, which holds an amplitude
    /// beside each probability; set together with `probability_qubits`.
    pub(crate) tensor_probability_qubits: Limit,
    pub(crate) export_qubits: Limit,
    /// Measured bits past which terminal sampling streams instead of materializing
    /// the outcome distribution and its CDF. A routing threshold, not an error cap.
    pub(crate) dense_outcome_bits: usize,
    pub(crate) sparse_qubits: Limit,
    pub(crate) factored_merge_qubits: Limit,
    pub(crate) stabilizer_cluster_qubits: Limit,
    pub(crate) mps_workspace_qubits: Limit,
    pub(crate) tn_peak_qubits: Limit,
    pub(crate) tn_slices: Limit,
    pub(crate) auto_mps_bond_dim: Limit,
}

/// Slicing trades a multiplicative time factor for a divided peak, so the slice budget
/// is the worst-case slowdown a contraction takes in place of a rejection.
const DEFAULT_TN_SLICES: usize = 1 << 10;
const DEFAULT_AUTO_MPS_BOND_DIM: usize = 256;

/// Bytes per stored sparse entry: the worst-case peak across the double-buffered
/// maps (24 payload bytes, table slot overhead, and load-factor headroom on both).
const SPARSE_ENTRY_BYTES: u64 = 64;

impl ResolvedLimits {
    /// Environment and detection overlaid with the process default.
    pub(crate) fn process_default() -> Self {
        Self::resolve(None)
    }

    /// Environment and detection overlaid with the process default, then `request`.
    pub(crate) fn resolve(request: Option<&Limits>) -> Self {
        let mut layers = *base_layers();
        layers.apply(&default_limits(), LimitSource::ProcessDefault);
        if let Some(request) = request {
            layers.apply(request, LimitSource::Request);
        }
        layers.finish()
    }

    /// Sparse map entry cap, `2^q`, saturating when detection is off.
    pub(crate) fn sparse_entries(&self) -> usize {
        shifted_cap(self.sparse_qubits.value)
    }

    /// MPS workspace cap in amplitudes, `u128::MAX` when detection is off.
    pub(crate) fn mps_workspace_elements(&self) -> u128 {
        let q = self.mps_workspace_qubits.value;
        if q >= u128::BITS as usize {
            u128::MAX
        } else {
            1u128 << q
        }
    }

    /// Tensor-network peak budget in `Complex64` elements, `usize::MAX` when detection
    /// is off.
    pub(crate) fn tn_peak_elements(&self) -> usize {
        let q = self.tn_peak_qubits.value;
        if q >= usize::BITS as usize {
            usize::MAX
        } else {
            1usize << q
        }
    }

    /// The export cap as a shift-safe width.
    pub(crate) fn export_cap(&self) -> usize {
        self.export_qubits.value.min(usize::BITS as usize - 1)
    }
}

fn shifted_cap(q: usize) -> usize {
    if q >= usize::BITS as usize - 1 {
        usize::MAX
    } else {
        1usize << q
    }
}

/// [`PrismError::ResourceLimit`] for an allocation of `required` over the cap `limit`,
/// which `knob` moves and `source` set.
#[cold]
pub(crate) fn resource_limit_error(
    backend: &str,
    operation: impl Into<String>,
    resource: ResourceKind,
    required: u128,
    limit: u128,
    knob: LimitKnob,
    source: LimitSource,
) -> PrismError {
    let env_var = match source {
        LimitSource::Detected | LimitSource::Env => Some(knob.env_var()),
        LimitSource::ProcessDefault | LimitSource::Request => None,
    };
    PrismError::ResourceLimit(Box::new(crate::error::ResourceLimit {
        backend: backend.to_string(),
        operation: operation.into(),
        resource,
        required,
        limit,
        env_var,
        knob: Some(knob),
        limit_source: source,
    }))
}

/// Resolution in progress: the final density-matrix cap waits on the final
/// statevector cap, so its explicit override is carried separately until
/// [`Layers::finish`].
#[derive(Debug, Clone, Copy)]
struct Layers {
    caps: ResolvedLimits,
    density_matrix_override: Option<Limit>,
}

/// Detection overlaid with the environment, read once per process.
fn base_layers() -> &'static Layers {
    static BASE: OnceLock<Layers> = OnceLock::new();
    BASE.get_or_init(|| {
        let budget = memory::detected_budget_bytes();
        let env = memory::env_limits();
        if budget.is_none() {
            eprintln!(
                "warning: could not detect system memory; memory caps without an override are \
                 disabled, and large allocations may abort. Set Limits::memory_budget_bytes or \
                 the PRISM_MAX_* variables to bound them."
            );
        }
        Layers::base(budget, &env.limits, env.dense_outcome_bits)
    })
}

impl Layers {
    /// Caps from a detected `budget` (`None` when detection failed, which disables
    /// the memory caps) overlaid with the environment.
    fn base(budget: Option<u64>, env: &Limits, env_outcome_bits: Option<usize>) -> Self {
        let detected = |value| Limit {
            value,
            source: LimitSource::Detected,
        };
        let mut layers = Layers {
            caps: ResolvedLimits {
                statevector_qubits: detected(0),
                density_matrix_qubits: detected(0),
                probability_qubits: detected(0),
                tensor_probability_qubits: detected(0),
                export_qubits: detected(0),
                dense_outcome_bits: 0,
                sparse_qubits: detected(0),
                factored_merge_qubits: detected(0),
                stabilizer_cluster_qubits: detected(0),
                mps_workspace_qubits: detected(0),
                tn_peak_qubits: detected(0),
                tn_slices: detected(DEFAULT_TN_SLICES),
                auto_mps_bond_dim: detected(DEFAULT_AUTO_MPS_BOND_DIM),
            },
            density_matrix_override: None,
        };
        layers.derive_from_budget(budget, LimitSource::Detected);
        layers.apply(env, LimitSource::Env);
        if let Some(bits) = env_outcome_bits {
            layers.caps.dense_outcome_bits = bits;
        }
        layers
    }

    fn derive_from_budget(&mut self, budget: Option<u64>, source: LimitSource) {
        let dense = |bytes_per_basis_state: usize| Limit {
            value: budget.map_or(usize::MAX, |b| {
                dense_qubits_for_budget(b, bytes_per_basis_state as u64)
            }),
            source,
        };
        let amplitude = size_of::<Complex64>();
        let caps = &mut self.caps;
        caps.statevector_qubits = dense(amplitude);
        caps.probability_qubits = dense(size_of::<f64>());
        caps.tensor_probability_qubits = dense(amplitude + size_of::<f64>());
        caps.export_qubits = dense(amplitude);
        caps.dense_outcome_bits = dense(2 * size_of::<f64>()).value;
        caps.sparse_qubits = dense(SPARSE_ENTRY_BYTES as usize);
        caps.factored_merge_qubits = dense(amplitude);
        caps.mps_workspace_qubits = dense(amplitude);
        caps.tn_peak_qubits = dense(amplitude);
        caps.stabilizer_cluster_qubits = Limit {
            value: budget.map_or(usize::MAX, stabilizer_cluster_qubits_for_budget),
            source,
        };
        self.density_matrix_override = None;
    }

    fn apply(&mut self, limits: &Limits, source: LimitSource) {
        if let Some(bytes) = limits.memory_budget_bytes {
            self.derive_from_budget(Some(bytes), source);
        }
        let set = |slot: &mut Limit, value: Option<usize>| {
            if let Some(value) = value {
                *slot = Limit { value, source };
            }
        };
        let caps = &mut self.caps;
        set(&mut caps.statevector_qubits, limits.max_statevector_qubits);
        set(&mut caps.probability_qubits, limits.max_probability_qubits);
        set(
            &mut caps.tensor_probability_qubits,
            limits.max_probability_qubits,
        );
        set(&mut caps.export_qubits, limits.max_export_qubits);
        set(&mut caps.sparse_qubits, limits.max_sparse_qubits);
        set(
            &mut caps.factored_merge_qubits,
            limits.max_factored_merge_qubits,
        );
        set(
            &mut caps.stabilizer_cluster_qubits,
            limits.max_stabilizer_cluster_qubits,
        );
        set(
            &mut caps.mps_workspace_qubits,
            limits.max_mps_workspace_qubits,
        );
        set(&mut caps.tn_peak_qubits, limits.max_tn_peak_qubits);
        set(&mut caps.tn_slices, limits.max_tn_slices);
        set(&mut caps.auto_mps_bond_dim, limits.auto_mps_bond_dim);
        if let Some(value) = limits.max_density_matrix_qubits {
            self.density_matrix_override = Some(Limit { value, source });
        }
    }

    fn finish(self) -> ResolvedLimits {
        let mut caps = self.caps;
        let half = Limit {
            value: caps.statevector_qubits.value / 2,
            source: caps.statevector_qubits.source,
        };
        caps.density_matrix_qubits = match self.density_matrix_override {
            Some(cap) if cap.value <= half.value => cap,
            _ => half,
        };
        caps.tn_slices.value = caps.tn_slices.value.max(1);
        caps.auto_mps_bond_dim.value = caps.auto_mps_bond_dim.value.max(1);
        caps
    }
}

/// Widest register whose `2^n` basis states of `bytes_per_basis_state` fit in
/// `budget_bytes`; zero when not even one fits.
fn dense_qubits_for_budget(budget_bytes: u64, bytes_per_basis_state: u64) -> usize {
    let max_elements = budget_bytes / bytes_per_basis_state.max(1);
    if max_elements == 0 {
        return 0;
    }
    ((u64::BITS - 1 - max_elements.leading_zeros()) as usize).min(usize::BITS as usize - 1)
}

/// Words a stabilizer tableau of `n` qubits occupies: `2n + 1` rows of
/// `2 * ceil(n / 64)` words each.
fn stabilizer_tableau_words(n: u128) -> u128 {
    (2 * n + 1) * 2 * n.div_ceil(64)
}

/// Widest stabilizer cluster a merge may build under `budget_bytes`. A merge
/// allocates `O(n^2 / 64)` words rather than `2^n` amplitudes, and the old pair is
/// still live when the joint tableau allocates and strictly smaller than it, so peak
/// is under twice the merged size.
fn stabilizer_cluster_qubits_for_budget(budget_bytes: u64) -> usize {
    let budget_words = u128::from(budget_bytes) / 8 / 2;
    let mut n: u128 = 1;
    while stabilizer_tableau_words(2 * n) <= budget_words && 2 * n < u128::from(u32::MAX) {
        n *= 2;
    }
    let mut step = n / 2;
    while step > 0 {
        if stabilizer_tableau_words(n + step) <= budget_words {
            n += step;
        }
        step /= 2;
    }
    usize::try_from(n).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn resolve(
        budget: Option<u64>,
        env: Limits,
        default: Limits,
        request: Option<Limits>,
    ) -> ResolvedLimits {
        let mut layers = Layers::base(budget, &env, None);
        layers.apply(&default, LimitSource::ProcessDefault);
        if let Some(request) = request {
            layers.apply(&request, LimitSource::Request);
        }
        layers.finish()
    }

    #[test]
    fn dense_budget_counts_fit_elements() {
        assert_eq!(dense_qubits_for_budget(8, 8), 0);
        assert_eq!(dense_qubits_for_budget(16, 8), 1);
        assert_eq!(dense_qubits_for_budget(31, 8), 1);
        assert_eq!(dense_qubits_for_budget(32, 8), 2);
        assert_eq!(dense_qubits_for_budget(7, 8), 0);
    }

    #[test]
    fn detection_sets_every_cap_from_half_of_memory() {
        let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), None);
        assert_eq!(caps.statevector_qubits.value, 30);
        assert_eq!(caps.statevector_qubits.source, LimitSource::Detected);
        assert_eq!(caps.density_matrix_qubits.value, 15);
        assert_eq!(caps.probability_qubits.value, 31);
        assert_eq!(caps.sparse_qubits.value, 28);
        assert_eq!(caps.tn_slices.value, DEFAULT_TN_SLICES);
        assert_eq!(caps.auto_mps_bond_dim.value, DEFAULT_AUTO_MPS_BOND_DIM);
    }

    #[test]
    fn failed_detection_disables_the_memory_caps() {
        let caps = resolve(None, Limits::new(), Limits::new(), None);
        assert_eq!(caps.statevector_qubits.value, usize::MAX);
        assert_eq!(caps.stabilizer_cluster_qubits.value, usize::MAX);
        assert_eq!(caps.sparse_entries(), usize::MAX);
    }

    #[test]
    fn each_layer_overrides_the_one_below() {
        let env = Limits::new()
            .max_statevector_qubits(24)
            .max_export_qubits(20);
        let default = Limits::new().max_statevector_qubits(22);
        let request = Limits::new().max_statevector_qubits(18);

        let caps = resolve(Some(16 * GIB), env, Limits::new(), None);
        assert_eq!(
            caps.statevector_qubits,
            Limit {
                value: 24,
                source: LimitSource::Env
            }
        );
        let caps = resolve(Some(16 * GIB), env, default, None);
        assert_eq!(
            caps.statevector_qubits,
            Limit {
                value: 22,
                source: LimitSource::ProcessDefault
            }
        );
        let caps = resolve(Some(16 * GIB), env, default, Some(request));
        assert_eq!(
            caps.statevector_qubits,
            Limit {
                value: 18,
                source: LimitSource::Request
            }
        );
        assert_eq!(
            caps.export_qubits,
            Limit {
                value: 20,
                source: LimitSource::Env
            },
            "a cap no higher layer sets keeps the lower layer's value"
        );
        let wider = Limits::new().max_statevector_qubits(28);
        let caps = resolve(Some(16 * GIB), env, default, Some(wider));
        assert_eq!(
            caps.statevector_qubits.value, 28,
            "a request may raise a cap the process default lowered"
        );
    }

    #[test]
    fn a_budget_replaces_lower_per_cap_values() {
        let env = Limits::new()
            .max_statevector_qubits(34)
            .max_density_matrix_qubits(10)
            .max_tn_slices(8);
        let tenant = Limits::new().memory_budget_bytes(4 * GIB);
        let caps = resolve(Some(16 * GIB), env, Limits::new(), Some(tenant));
        assert_eq!(
            caps.statevector_qubits,
            Limit {
                value: 28,
                source: LimitSource::Request
            },
            "a 4 GiB budget must not inherit PRISM_MAX_SV_QUBITS=34"
        );
        assert_eq!(caps.density_matrix_qubits.value, 14);
        assert_eq!(caps.density_matrix_qubits.source, LimitSource::Request);
        assert_eq!(
            caps.tn_slices.value, 8,
            "the slice budget is not a memory cap"
        );

        let both = tenant.max_export_qubits(12);
        let caps = resolve(Some(16 * GIB), env, Limits::new(), Some(both));
        assert_eq!(
            caps.export_qubits.value, 12,
            "the budget's own layer still applies its per-cap values"
        );

        let caps = resolve(Some(16 * GIB), env, tenant, None);
        assert_eq!(caps.statevector_qubits.source, LimitSource::ProcessDefault);
        assert_eq!(caps.statevector_qubits.value, 28);
    }

    #[test]
    fn the_density_matrix_cap_stays_within_half_the_statevector_cap() {
        let caps = resolve(
            Some(16 * GIB),
            Limits::new().max_density_matrix_qubits(40),
            Limits::new(),
            None,
        );
        assert_eq!(
            caps.density_matrix_qubits,
            Limit {
                value: 15,
                source: LimitSource::Detected
            }
        );
        let request = Limits::new().max_statevector_qubits(12);
        let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
        assert_eq!(
            caps.density_matrix_qubits,
            Limit {
                value: 6,
                source: LimitSource::Request
            }
        );
        let request = Limits::new().max_density_matrix_qubits(4);
        let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
        assert_eq!(caps.density_matrix_qubits.value, 4);
    }

    #[test]
    fn the_probability_knob_sets_both_probability_caps() {
        let request = Limits::new().max_probability_qubits(9);
        let caps = resolve(Some(16 * GIB), Limits::new(), Limits::new(), Some(request));
        assert_eq!(caps.probability_qubits.value, 9);
        assert_eq!(caps.tensor_probability_qubits.value, 9);
    }

    #[test]
    fn zero_counts_read_as_one() {
        let request = Limits::new().max_tn_slices(0).auto_mps_bond_dim(0);
        let caps = resolve(Some(GIB), Limits::new(), Limits::new(), Some(request));
        assert_eq!(caps.tn_slices.value, 1);
        assert_eq!(caps.auto_mps_bond_dim.value, 1);
    }

    #[test]
    fn a_request_error_names_the_setter_and_no_variable() {
        let err = resource_limit_error(
            "statevector",
            "the circuit",
            ResourceKind::Qubits,
            20,
            18,
            LimitKnob::StatevectorQubits,
            LimitSource::Request,
        );
        let text = err.to_string();
        assert!(text.contains("set by this request's limits"), "{text}");
        assert!(text.contains("Limits::max_statevector_qubits"), "{text}");
        assert!(!text.contains("PRISM_MAX_SV_QUBITS"), "{text}");
        let PrismError::ResourceLimit(data) = err else {
            panic!("expected ResourceLimit");
        };
        assert_eq!(data.env_var, None);
        assert_eq!(data.knob, Some(LimitKnob::StatevectorQubits));
    }

    #[test]
    fn a_detected_error_names_the_setter_and_the_variable() {
        let text = resource_limit_error(
            "statevector",
            "the circuit",
            ResourceKind::Qubits,
            40,
            30,
            LimitKnob::StatevectorQubits,
            LimitSource::Detected,
        )
        .to_string();
        assert!(text.contains("on this machine"), "{text}");
        assert!(text.contains("Limits::max_statevector_qubits"), "{text}");
        assert!(text.contains("PRISM_MAX_SV_QUBITS"), "{text}");
    }
}
