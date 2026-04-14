# Changelog

All notable changes to PRISM-Q will be documented in this file.

## [0.2.2] - 2026-04-14

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

