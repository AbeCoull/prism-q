//! Run a repetition-code memory experiment and decode it with the built-in decoder.

use prism_q::{
    Gate, QecBasis, QecNoise, QecOptions, QecProgram, QecRecordRef, UnionFindDecoder,
    run_qec_program,
};

fn repetition_memory(distance: usize, rounds: usize, p: f64, shots: usize) -> QecProgram {
    let data: Vec<usize> = (0..distance).map(|i| 2 * i).collect();
    let ancillas: Vec<usize> = (0..distance - 1).map(|i| 2 * i + 1).collect();
    let options = QecOptions {
        shots,
        seed: 42,
        ..QecOptions::default()
    };
    let mut qp = QecProgram::with_options(2 * distance - 1, options);
    for &q in data.iter().chain(&ancillas) {
        qp.reset(QecBasis::Z, q).unwrap();
    }

    let mut previous: Vec<usize> = Vec::new();
    for _ in 0..rounds {
        qp.noise(QecNoise::XError(p), &data).unwrap();
        for &a in &ancillas {
            qp.push_gate(Gate::Cx, &[a - 1, a]).unwrap();
            qp.push_gate(Gate::Cx, &[a + 1, a]).unwrap();
        }
        let mut current = Vec::with_capacity(ancillas.len());
        for &a in &ancillas {
            current.push(qp.measure_z(a).unwrap());
            qp.reset(QecBasis::Z, a).unwrap();
        }
        for (i, &record) in current.iter().enumerate() {
            let mut refs = vec![QecRecordRef::absolute(record)];
            if let Some(&before) = previous.get(i) {
                refs.push(QecRecordRef::absolute(before));
            }
            qp.detector(&refs).unwrap();
        }
        previous = current;
    }

    let last: Vec<usize> = data.iter().map(|&q| qp.measure_z(q).unwrap()).collect();
    for (i, &record) in previous.iter().enumerate() {
        let refs = [last[i], last[i + 1], record].map(QecRecordRef::absolute);
        qp.detector(&refs).unwrap();
    }
    qp.observable_include(0, &[QecRecordRef::absolute(last[0])])
        .unwrap();
    qp
}

fn main() {
    for distance in [3, 5, 7] {
        let qp = repetition_memory(distance, distance, 0.05, 20_000);
        let model = qp
            .detector_error_model()
            .and_then(|m| m.decompose_graphlike())
            .expect("no graphlike model");
        let decoder = UnionFindDecoder::from_model(&model).unwrap();
        let result = run_qec_program(&qp).unwrap();
        let predicted = decoder.decode_packed(&result.detectors).unwrap();
        let failures = (0..result.total_shots)
            .filter(|&shot| predicted.get_bit(shot, 0) != result.observables.get_bit(shot, 0))
            .count();
        println!(
            "d={distance}: {} detectors, raw {:.4}, decoded {:.4}",
            qp.num_detectors(),
            result.logical_error_rates()[0],
            failures as f64 / result.total_shots as f64,
        );
    }
}
