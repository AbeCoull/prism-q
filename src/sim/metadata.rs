//! Provenance carried by every simulation result: which engine ran, whether the
//! answer is exact, and where the state lived.

/// Expectation values with the uncertainty of the estimate that produced them.
///
/// Returned by [`Simulate::expectation_values_reported`]; the plain
/// [`Simulate::expectation_values`] returns the values alone.
///
/// [`Simulate::expectation_values`]: crate::sim::Simulate::expectation_values
/// [`Simulate::expectation_values_reported`]: crate::sim::Simulate::expectation_values_reported
#[derive(Debug, Clone)]
pub struct ExpectationResult {
    /// One value per observable, in the order they were supplied.
    pub values: Vec<f64>,
    /// One standard error per value, `None` when the route evaluates rather
    /// than samples. Such a route reports exactness through
    /// [`RunMetadata::exactness`] instead of an interval of width zero.
    pub std_errors: Option<Vec<f64>>,
    pub metadata: RunMetadata,
}

impl ExpectationResult {
    pub fn into_values(self) -> Vec<f64> {
        self.values
    }
}

/// Engine a run resolved to, after [`BackendKind::Auto`] dispatch.
///
/// [`BackendKind::Auto`]: crate::sim::BackendKind::Auto
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    ProductState,
    Stabilizer,
    FactoredStabilizer,
    Sparse,
    Mps,
    Factored,
    TensorNetwork,
    Statevector,
    DensityMatrix,
    Distributed,
    StabilizerRank,
    StochasticPauli,
    DeterministicPauli,
    /// The compiled Clifford sampler, which answers from a propagated parity
    /// map rather than from a `Backend`.
    CompiledStabilizer,
    /// One engine per independent block, merged. The decomposed route holds no
    /// joint state, so no single engine names the result.
    Decomposed,
    /// A backend outside the built-in set, named by [`Backend::name`].
    ///
    /// [`Backend::name`]: crate::backend::Backend::name
    Other(&'static str),
}

/// Whether a result is exact for the circuit as given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Exactness {
    Exact,
    /// The engine that ran can discard state weight or estimate by sampling.
    ///
    /// This marks the route, not the run: an MPS at bond 256 on a circuit that
    /// never fills a bond truncates nothing and still reports `Approximate`,
    /// with `fidelity_lower_bound` of 1.0. The variant answers whether the
    /// answer could have been approximated, the bound answers whether it was.
    /// `None` means the engine reports no bound.
    Approximate {
        fidelity_lower_bound: Option<f64>,
    },
}

/// Where the state lived during the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Host,
    Device,
}

/// How a result was produced, attached to every [`Simulate`] result type.
///
/// A GPU-attached run below the device crossover reports [`Placement::Host`],
/// and automatic dispatch onto an approximate backend reports
/// [`Exactness::Approximate`].
///
/// [`Simulate`]: crate::sim::Simulate
#[derive(Debug, Clone)]
pub struct RunMetadata {
    pub backend: ResolvedBackend,
    pub exactness: Exactness,
    pub placement: Placement,
    /// Shots drawn, for a result estimated by sampling. `None` for an analytic
    /// result.
    pub shots: Option<usize>,
}

impl RunMetadata {
    pub(crate) fn new(
        backend: ResolvedBackend,
        exactness: Exactness,
        placement: Placement,
    ) -> Self {
        Self {
            backend,
            exactness,
            placement,
            shots: None,
        }
    }

    pub(crate) fn exact(backend: ResolvedBackend) -> Self {
        Self::new(backend, Exactness::Exact, Placement::Host)
    }

    pub(crate) fn approximate(backend: ResolvedBackend) -> Self {
        Self::new(
            backend,
            Exactness::Approximate {
                fidelity_lower_bound: None,
            },
            Placement::Host,
        )
    }

    pub(crate) fn with_shots(mut self, shots: usize) -> Self {
        self.shots = Some(shots);
        self
    }

    pub fn is_exact(&self) -> bool {
        matches!(self.exactness, Exactness::Exact)
    }

    /// Lower bound on the fidelity of the produced state, `None` when the
    /// result is exact or the engine reports no bound.
    pub fn fidelity_lower_bound(&self) -> Option<f64> {
        match self.exactness {
            Exactness::Exact => None,
            Exactness::Approximate {
                fidelity_lower_bound,
            } => fidelity_lower_bound,
        }
    }

    /// Fold in one more run of the same route, keeping the weaker claim. Shots
    /// on a per-shot route each evolve their own state, so the ensemble is only
    /// as exact as its worst member and the bound is a minimum, not a product.
    pub(crate) fn weaken_with(&mut self, other: &RunMetadata) {
        if other.placement != self.placement {
            self.placement = Placement::Host;
        }
        match (self.exactness, other.exactness) {
            (_, Exactness::Exact) => {}
            (Exactness::Exact, approx) => self.exactness = approx,
            (
                Exactness::Approximate {
                    fidelity_lower_bound: a,
                },
                Exactness::Approximate {
                    fidelity_lower_bound: b,
                },
            ) => {
                self.exactness = Exactness::Approximate {
                    fidelity_lower_bound: match (a, b) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        _ => None,
                    },
                };
            }
        }
    }

    /// Metadata for a run whose blocks were evolved separately. Each block holds
    /// part of one state, so the bound is the product; exactness is the weakest
    /// of the parts and placement is `Device` only when every part was, so the
    /// merged result never claims more than the block that claimed least. A part
    /// that reports no bound poisons the product, since an unquantified
    /// truncation bounds nothing.
    pub(crate) fn decomposed(parts: impl IntoIterator<Item = RunMetadata>) -> Self {
        let mut exact = true;
        let mut bound = Some(1.0f64);
        let mut all_device = true;
        for part in parts {
            all_device &= part.placement == Placement::Device;
            if let Exactness::Approximate {
                fidelity_lower_bound,
            } = part.exactness
            {
                exact = false;
                bound = match (bound, fidelity_lower_bound) {
                    (Some(acc), Some(b)) => Some(acc * b),
                    _ => None,
                };
            }
        }
        let exactness = if exact {
            Exactness::Exact
        } else {
            Exactness::Approximate {
                fidelity_lower_bound: bound,
            }
        };
        let placement = if all_device {
            Placement::Device
        } else {
            Placement::Host
        };
        Self::new(ResolvedBackend::Decomposed, exactness, placement)
    }
}
