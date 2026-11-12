//! Transmon sites and the device that couples them.

/// One transmon site, a Duffing oscillator with energies `frequency * n + anharmonicity
/// * n (n - 1) / 2` in Hz, truncated to `levels`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transmon {
    /// The 0 to 1 transition frequency in Hz; also the default carrier of the site's drive
    /// line.
    pub frequency: f64,
    /// `f12 - f01` in Hz, negative for a transmon.
    pub anharmonicity: f64,
    pub levels: usize,
}

/// One to three [`Transmon`] sites with exchange couplings and one drive line per site.
///
/// The device basis is mixed radix with site 0 the least significant digit, the order a
/// [`Circuit`](crate::Circuit) gives qubits: basis index `n0 + d0 * (n1 + d1 * n2)` for
/// levels `n_j` and per-site dimensions `d_j`.
///
/// # Panics
///
/// [`Device::couple`] panics on a site index past the last site or on a pair naming one
/// site twice.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    sites: Vec<Transmon>,
    couplings: Vec<(usize, usize, f64)>,
}

impl Device {
    pub fn new(sites: impl IntoIterator<Item = Transmon>) -> Self {
        Device {
            sites: sites.into_iter().collect(),
            couplings: Vec::new(),
        }
    }

    /// Add the exchange coupling `g (a_a^dagger a_b + a_b^dagger a_a)`, `g` in Hz.
    pub fn couple(mut self, a: usize, b: usize, g: f64) -> Self {
        let sites = self.sites.len();
        assert!(
            a < sites && b < sites,
            "coupling ({a}, {b}) names a site past the last of {sites}"
        );
        assert_ne!(a, b, "a coupling needs two distinct sites, got ({a}, {a})");
        self.couplings.push((a, b, g));
        self
    }

    pub fn sites(&self) -> &[Transmon] {
        &self.sites
    }

    /// Product of the per-site levels, the length of a state [`evolve`](super::evolve)
    /// takes.
    pub fn dimension(&self) -> usize {
        self.sites
            .iter()
            .fold(1usize, |dim, site| dim.saturating_mul(site.levels))
    }

    pub(super) fn couplings(&self) -> &[(usize, usize, f64)] {
        &self.couplings
    }

    /// Basis index step of one level on each site.
    pub(super) fn strides(&self) -> Vec<usize> {
        self.sites
            .iter()
            .scan(1usize, |stride, site| {
                let current = *stride;
                *stride *= site.levels;
                Some(current)
            })
            .collect()
    }
}
