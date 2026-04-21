# Changelog

All notable changes to PRISM-Q will be documented in this file.

## [0.9.0] - 2026-04-21

### Features

- **gpu:** Stabilizer GPU scaffol (#34)([30d7d42](https://github.com/AbeCoull/prism-q/commit/30d7d42df89ac1758b3baa1aeba2a7063f427fd3))
## [0.8.0] - 2026-04-20

### Features

- **gpu:** Observability  and benchmarking infra for gpu code (#33)([d133066](https://github.com/AbeCoull/prism-q/commit/d133066c1abf8f61253c1606f4e2a738d737d9f6))

### Miscellaneous

- **release:** 0.8.0([c5120bb](https://github.com/AbeCoull/prism-q/commit/c5120bb70846f176196bbd6943643542e981b9ab))
## [0.7.0] - 2026-04-19

### Documentation

- Update docs and add a pull request template (#31)([13b381e](https://github.com/AbeCoull/prism-q/commit/13b381efd7da4556ca8ed84f3fbafb67a2dfffd7))

### Features

- **gpu:** Dispatch-level crossover and decomposition-aware routing (#32)([1c2627d](https://github.com/AbeCoull/prism-q/commit/1c2627d92c3ecdaca9d62da092f73f8b857292f7))

### Miscellaneous

- **release:** 0.7.0([d4c3007](https://github.com/AbeCoull/prism-q/commit/d4c3007153e6bd0fe09c02573e664eb3ef70d231))
## [0.6.0] - 2026-04-18

### Features

- **gpu:** Batched kernels for fused gate variants (#30)([02bb5fb](https://github.com/AbeCoull/prism-q/commit/02bb5fb15fe24024696210c93562eefb88ce717c))

### Miscellaneous

- **release:** 0.6.0([3b0df64](https://github.com/AbeCoull/prism-q/commit/3b0df64d62418cf90c330547ba3f610939736736))
## [0.5.0] - 2026-04-18

### Features

- **statevector:** Add optional CUDA GPU backend for processing gates (#29)([61d56a1](https://github.com/AbeCoull/prism-q/commit/61d56a1300cd63592dc2c053476752814c248a08))

### Miscellaneous

- **release:** 0.5.0([f9f6eb2](https://github.com/AbeCoull/prism-q/commit/f9f6eb2ef66bbeaa994593d08abd7ec55a59e01d))
## [0.4.0] - 2026-04-17

### Features

- **statevector:** Add optional GPU acceleration (#27)([c99c752](https://github.com/AbeCoull/prism-q/commit/c99c7528bcb16c0acc38261e006077183a97721c))

### Miscellaneous

- **release:** 0.4.0([036a349](https://github.com/AbeCoull/prism-q/commit/036a3491ff560e4847d992bd4d88302d3c35814b))
## [0.3.0] - 2026-04-17

### Features

- **gpu:** Shared GPU execution resource scaffold (#26)([9132747](https://github.com/AbeCoull/prism-q/commit/9132747bc5e30032c52b235b775f596b8c3b360c))

### Miscellaneous

- **release:** 0.3.0([5781bd7](https://github.com/AbeCoull/prism-q/commit/5781bd71d1dccf0ccf77b2b2364c06a39c03bffb))
## [0.2.4] - 2026-04-16

### Miscellaneous

- **release:** 0.2.4([fa01a49](https://github.com/AbeCoull/prism-q/commit/fa01a49171c9510b0756ef4371b0984e0a9a27a1))

### Performance

- **compiled:** Skip deterministic rows in BTS sampling (#25)([533495b](https://github.com/AbeCoull/prism-q/commit/533495ba0995d6b161fcf286be71e5910526481b))

### Testing

- **dispatch:** Add validation, error path, and smoke tests for Backe… (#24)([f6950b8](https://github.com/AbeCoull/prism-q/commit/f6950b8d7534c9467882e1d99aebc3a51a3fc34b))
## [0.2.3] - 2026-04-16

### Miscellaneous

- **release:** 0.2.3([ce31d89](https://github.com/AbeCoull/prism-q/commit/ce31d898219da3e4915c36fba1606674633345b9))

### Performance

- **compiled:** AVX2 DAG kernel + parallel BTS DAG pass-through (#23)([4be7d31](https://github.com/AbeCoull/prism-q/commit/4be7d313c11fc01e612294da3906be573a34aa96))
## [0.2.2] - 2026-04-14

### Miscellaneous

- **release:** 0.2.2([77e2bcd](https://github.com/AbeCoull/prism-q/commit/77e2bcd77e3fc6a52e25915fa172c35fec9ed909))

### Performance

- **mps:** Keep routed qubits in a persistent logical layout (#22)([24d3788](https://github.com/AbeCoull/prism-q/commit/24d3788a969a5ee6727eda793dab7afbfbea30c8))
## [0.2.1] - 2026-04-10

### Miscellaneous

- **release:** 0.2.1([ef6e5c7](https://github.com/AbeCoull/prism-q/commit/ef6e5c7c145f1e716a83af25de3feaa96a12dc8c))
## [0.2.0] - 2026-04-10

### Bug Fixes

- Update release flows (#21)([31ba60a](https://github.com/AbeCoull/prism-q/commit/31ba60a35788cc23b189fc8b41d268ff87e1b5c6))
## [0.1.0] - 2026-04-10

### Bug Fixes

- Add shot accumulator for clifford sims (#12)([9c8ab55](https://github.com/AbeCoull/prism-q/commit/9c8ab55fd3d29b43f75cfbcaee680d5035effd4a))
- Update coverage github badge action (#11)([5be3a43](https://github.com/AbeCoull/prism-q/commit/5be3a43316303f6baeeb3f0e29a6e7c056b6036e))
- Update shot processing to end of the simulation (#2)([3a2a9ca](https://github.com/AbeCoull/prism-q/commit/3a2a9cad39718e5bdaaf8b6684a881317b0de5aa))

### CI

- Update to include coverage and docs runs (#4)([709a629](https://github.com/AbeCoull/prism-q/commit/709a6292bef82d5210dd372303327376c3c227fd))

### Features

- Add crate publishing workflow (#20)([9307d81](https://github.com/AbeCoull/prism-q/commit/9307d8165af2cc63942d21e90860e54f8f37aea7))
- Add quantum trajectories (#19)([ec30253](https://github.com/AbeCoull/prism-q/commit/ec302534e93341b7dcbd8305d4fcbe614647a70d))
- Add circuit visualizer (#18)([88251f6](https://github.com/AbeCoull/prism-q/commit/88251f64ed0630a5f54117eccbea14038a8da0e0))
- Refactor the stabilizer backend (#7)([7648541](https://github.com/AbeCoull/prism-q/commit/7648541ff9a8bc272d5eeceff2c426b66335bf7e))
- Add arm support for kernel (#1)([7016588](https://github.com/AbeCoull/prism-q/commit/70165887d9b0ea7f3d9476e46c5583f1f97d5744))

### Performance

- Add better stabilizer perf at higher shot count and benchmarking (#14)([cfe251d](https://github.com/AbeCoull/prism-q/commit/cfe251d44c5df1b3741c4d31de29ca4eb05ab06b))
- Optimize stabilizer measurement and Gaussian elimination (#6)([6ff849b](https://github.com/AbeCoull/prism-q/commit/6ff849b0259e75973302aa12b70ee8ab03503a1c))
- Cache fusion across shots in run_shots_with slow path (#5)([7a0f7b9](https://github.com/AbeCoull/prism-q/commit/7a0f7b91dcfc8d4659fd820953c94bb435fab17c))
- Optimize rowmul phase and inline measurement rowmul (#3)([c579931](https://github.com/AbeCoull/prism-q/commit/c5799311309ea1f8beb95857076995dcab12d3a1))

### Refactor

- Move items to shared file path and fix visibility (#17)([1583b9a](https://github.com/AbeCoull/prism-q/commit/1583b9a3eced48274a4c44245b1e2f30ddbec13e))
- Clean up the way users interface with the package (#16)([263ecdc](https://github.com/AbeCoull/prism-q/commit/263ecdc81109f4c1010fe21f1b0930925520df3e))
- Split larger files and better organize dispatch utils (#15)([bca4c1d](https://github.com/AbeCoull/prism-q/commit/bca4c1d165c69fe223c604bb74f55bc4e474d17b))
- Add nullAccum for testing paths and speed up histogram (#13)([9846b5f](https://github.com/AbeCoull/prism-q/commit/9846b5f43bb7a896eed5d66746f0bfc600edcab0))
- Rewriting to speed up shots at higher values and changing output type (#10)([b3f1ab5](https://github.com/AbeCoull/prism-q/commit/b3f1ab55dfc09685975743a7065c326f9861e8d2))
- Replace quasi_prob with unified SPP/SPD backends (#9)([c1e1d25](https://github.com/AbeCoull/prism-q/commit/c1e1d25744e2b746e5a1853850db50edaca804fa))

### `fix

- Fusion pass fixes with temporal blocking (#8)([eb9fc76](https://github.com/AbeCoull/prism-q/commit/eb9fc762a122e66903defb793a1345b3b3e2a5ad))

