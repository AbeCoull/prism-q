use super::*;

mod gpu_crossover_tests {
    use super::*;
    use crate::gates::Gate;

    fn stub_kind() -> BackendKind {
        BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    fn run_query(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
        crate::sim::simulate(circuit).backend(kind).seed(seed).run()
    }

    // The builder GPU shortcut must compose identically to constructing the
    // variant manually. Uses the stub context at a small circuit so crossover
    // fires and proves the composition is side-effect equivalent.
    #[test]
    fn builder_gpu_wraps_statevector_gpu_variant() {
        let ctx = GpuContext::stub_for_tests();
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);

        let direct = crate::sim::simulate(&circuit)
            .gpu(ctx.clone())
            .seed(42)
            .run()
            .expect("builder GPU shortcut must honor crossover and route to CPU");
        let manual = crate::sim::simulate(&circuit)
            .backend(stub_kind())
            .seed(42)
            .run()
            .expect("manual variant reference");

        let dp = direct.probabilities.expect("direct probs").to_vec();
        let mp = manual.probabilities.expect("manual probs").to_vec();
        assert_eq!(dp, mp);
    }

    // A 4q circuit is far below the default 14q threshold. If the dispatch
    // layer were to build a GPU backend anyway, `GpuState::new` on the stub
    // context would return `BackendUnsupported`. Success proves the
    // crossover in `select_dispatch` is routing small circuits to the host
    // path.
    #[test]
    fn small_circuit_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);

        let result = run_query(stub_kind(), &circuit, 42)
            .expect("stub context must not be touched for a 4q circuit");
        let probs = result
            .probabilities
            .expect("probabilities missing")
            .to_vec();

        let mut expected = [0.0_f64; 16];
        expected[0b0000] = 0.25;
        expected[0b0011] = 0.25;
        expected[0b1100] = 0.25;
        expected[0b1111] = 0.25;
        for (i, (p, e)) in probs.iter().zip(&expected).enumerate() {
            assert!((p - e).abs() < 1e-10, "p[{i}] = {p}, expected {e}");
        }
    }

    // `independent_bell_pairs(8)` spans 16 qubits but decomposes into 8
    // independent 2q blocks. With `BackendKind::StatevectorGpu`, each
    // sub-block is below the 14q threshold and must route to CPU. If
    // decomposition failed to fire, the 16q monolithic path would attempt
    // `GpuState::new` through the stub and return `BackendUnsupported`.
    // Success here proves decomposition survives across the GPU dispatch.
    #[test]
    fn decomposable_16q_circuit_runs_per_block_on_cpu() {
        let circuit = crate::circuits::independent_bell_pairs(8);
        assert_eq!(circuit.num_qubits, 16);

        let cpu = run_query(BackendKind::Statevector, &circuit, 42).expect("cpu baseline");
        let gpu = run_query(stub_kind(), &circuit, 42).expect("stub must stay out of the way");

        let cpu_p = cpu.probabilities.expect("cpu probs").to_vec();
        let gpu_p = gpu.probabilities.expect("gpu probs").to_vec();
        assert_eq!(cpu_p.len(), gpu_p.len());
        for (i, (c, g)) in cpu_p.iter().zip(gpu_p.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-10,
                "prob[{i}] cpu={c}, gpu={g}, diff={}",
                (c - g).abs()
            );
        }
    }

    fn stabilizer_stub_kind() -> BackendKind {
        BackendKind::StabilizerGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    // A 4q Clifford circuit is far below the stabilizer GPU threshold, so the
    // stub context must never be touched. Produces the same measurement bits
    // as a plain CPU stabilizer run.
    #[test]
    fn stabilizer_gpu_small_circuit_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 4);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        circuit.add_measure(2, 2);
        circuit.add_measure(3, 3);

        let cpu_run = run_query(BackendKind::Stabilizer, &circuit, 42).expect("cpu baseline");
        let gpu_run = run_query(stabilizer_stub_kind(), &circuit, 42)
            .expect("stub must stay out of the way for small circuits");
        assert_eq!(cpu_run.classical_bits, gpu_run.classical_bits);
    }

    // Non-Clifford circuits are rejected at dispatch time with the same error
    // shape as `BackendKind::Stabilizer`.
    #[test]
    fn stabilizer_gpu_rejects_non_clifford_at_dispatch() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_gate(Gate::T, &[0]);
        let err = run_query(stabilizer_stub_kind(), &circuit, 42).unwrap_err();
        assert!(matches!(err, PrismError::IncompatibleBackend { .. }));
    }

    fn auto_gpu_stub_kind() -> BackendKind {
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    fn assert_probs_match(a: &RunOutcome, b: &RunOutcome) {
        let ap = a.probabilities.as_ref().expect("probs a").to_vec();
        let bp = b.probabilities.as_ref().expect("probs b").to_vec();
        assert_eq!(ap.len(), bp.len());
        for (i, (x, y)) in ap.iter().zip(bp.iter()).enumerate() {
            assert!((x - y).abs() < 1e-10, "prob[{i}]: {x} vs {y}");
        }
    }

    // A small Clifford circuit selects the stabilizer choice, which sits below
    // the stabilizer GPU crossover, so `AutoGpu` builds a CPU stabilizer and
    // never touches the stub. Results match the plain `Auto` path.
    #[test]
    fn auto_gpu_small_clifford_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42)
            .expect("stub must stay out of the way below the stabilizer crossover");
        assert_probs_match(&cpu, &gpu);
    }

    // `independent_bell_pairs(8)` spans 16 qubits but decomposes into 8
    // independent 2q blocks, each below the statevector crossover. Every block
    // stays on CPU under `AutoGpu`; if decomposition failed, the monolithic 16q
    // path would still hit the VRAM gate and fall back rather than error.
    #[test]
    fn auto_gpu_decomposable_16q_runs_per_block_on_cpu() {
        let circuit = crate::circuits::independent_bell_pairs(8);
        assert_eq!(circuit.num_qubits, 16);

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42).expect("stub stays out of the way");
        assert_probs_match(&cpu, &gpu);
    }

    // A 16q entangled non-Clifford circuit selects the statevector choice and
    // clears the 14q crossover, so `AutoGpu` reaches the GPU decision. Because
    // the stub cannot report VRAM, the fits check fails closed and the block
    // runs on CPU: same results as `Auto`, no error. The explicit
    // `StatevectorGpu` path has no VRAM gate, so the same circuit touches the
    // stub and surfaces `BackendUnsupported`, isolating the added fallback.
    #[test]
    fn auto_gpu_large_block_falls_back_to_cpu_on_stub() {
        let mut circuit = Circuit::new(16, 0);
        for q in 0..16 {
            circuit.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..15 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42)
            .expect("stub VRAM query fails closed, so AutoGpu must fall back to CPU without error");
        assert_probs_match(&cpu, &gpu);

        let explicit = BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        };
        assert!(matches!(
            run_query(explicit, &circuit, 42).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }
}

mod accel_tests {
    use super::*;

    fn stub() -> Arc<GpuContext> {
        GpuContext::stub_for_tests()
    }

    fn is_cpu(accel: &Accel) -> bool {
        matches!(accel, Accel::Cpu)
    }

    #[test]
    fn auto_gpu_statevector_below_crossover_is_cpu() {
        let kind = BackendKind::AutoGpu { context: stub() };
        let n = crate::gpu::min_qubits() - 1;
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n)));
    }

    // Above the crossover the stub cannot report VRAM, so the fit gate fails
    // closed and the soft path resolves to the host.
    #[test]
    fn auto_gpu_statevector_fits_fails_closed_on_stub() {
        let kind = BackendKind::AutoGpu { context: stub() };
        let n = crate::gpu::min_qubits() + 2;
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n)));
    }

    #[test]
    fn auto_gpu_cpu_only_families_stay_cpu() {
        let kind = BackendKind::AutoGpu { context: stub() };
        for family in [
            Family::ProductState,
            Family::Sparse,
            Family::Mps,
            Family::Factored,
            Family::FactoredStabilizer,
            Family::TensorNetwork,
        ] {
            assert!(is_cpu(&accel_for(&kind, family, 1 << 10)), "{family:?}");
        }
    }

    #[test]
    fn explicit_statevector_gpu_is_hard_at_crossover_and_cpu_below() {
        let kind = BackendKind::StatevectorGpu { context: stub() };
        let n = crate::gpu::min_qubits();
        assert!(matches!(
            accel_for(&kind, Family::Statevector, n),
            Accel::Gpu { soft: false, .. }
        ));
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n - 1)));
    }

    #[test]
    fn explicit_stabilizer_gpu_is_hard_at_crossover_and_cpu_below() {
        let kind = BackendKind::StabilizerGpu { context: stub() };
        let n = crate::gpu::stabilizer_min_qubits();
        assert!(matches!(
            accel_for(&kind, Family::Stabilizer, n),
            Accel::Gpu { soft: false, .. }
        ));
        assert!(is_cpu(&accel_for(&kind, Family::Stabilizer, n - 1)));
    }

    // An explicit GPU kind accelerates only its own family; any other family
    // resolves to the host regardless of size.
    #[test]
    fn explicit_kind_other_family_is_cpu() {
        let sv = BackendKind::StatevectorGpu { context: stub() };
        assert!(is_cpu(&accel_for(&sv, Family::Stabilizer, 1 << 20)));
        let stab = BackendKind::StabilizerGpu { context: stub() };
        assert!(is_cpu(&accel_for(&stab, Family::Statevector, 1 << 20)));
    }

    #[test]
    fn cpu_kinds_are_cpu_everywhere() {
        for family in [Family::Statevector, Family::Stabilizer] {
            assert!(is_cpu(&accel_for(&BackendKind::Auto, family, 1 << 20)));
            assert!(is_cpu(&accel_for(
                &BackendKind::Statevector,
                family,
                1 << 20
            )));
        }
    }

    // `init_from_state` on a soft GPU backend degrades to the host when the
    // device upload fails, preserving the supplied amplitudes.
    #[test]
    fn init_from_state_soft_falls_back_to_host_on_stub() {
        use num_complex::Complex64;
        let mut sv = StatevectorBackend::new(42).with_gpu_auto(stub());
        let amp = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let state = vec![amp, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), amp];
        sv.init_from_state(state.clone(), 0).unwrap();
        let exported = sv.export_statevector().unwrap();
        for (e, s) in exported.iter().zip(&state) {
            assert!((e - s).norm() < 1e-12);
        }
    }

    #[test]
    fn init_from_state_hard_errors_on_stub() {
        use num_complex::Complex64;
        let mut sv = StatevectorBackend::new(42).with_gpu(stub());
        let err = sv
            .init_from_state(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)], 0)
            .unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }
}
