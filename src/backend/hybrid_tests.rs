use super::*;

fn core_record(class: u8, group: u16, mask: u64) -> Vec<u8> {
    let mut record = vec![0u8; 48];
    record[4..8].copy_from_slice(&48u32.to_le_bytes());
    record[9] = class;
    record[30..32].copy_from_slice(&1u16.to_le_bytes());
    record[32..40].copy_from_slice(&mask.to_le_bytes());
    record[40..42].copy_from_slice(&group.to_le_bytes());
    record
}

#[test]
fn records_parse_class_group_and_mask_per_core() {
    let buf = [
        core_record(1, 0, 0b1),
        core_record(0, 0, 0b10),
        core_record(1, 1, 0b100),
    ]
    .concat();
    let records: Vec<_> = processor_core_records(&buf).collect();
    assert_eq!(records, vec![(1, 0, 0b1), (0, 0, 0b10), (1, 1, 0b100)]);
}

#[test]
fn a_truncated_record_ends_the_walk() {
    let mut buf = [core_record(1, 0, 0b1), core_record(0, 0, 0b10)].concat();
    buf.truncate(48 + 20);
    assert_eq!(processor_core_records(&buf).count(), 1);
}

#[test]
fn one_core_class_is_not_hybrid() {
    assert_eq!(classify([(0, 0, 0b1), (0, 0, 0b10)]), None);
    assert_eq!(classify(std::iter::empty()), None);
}

#[test]
fn the_highest_class_is_the_performance_set() {
    let cores = [
        (1, 0, 0b0011),
        (0, 0, 0b0100),
        (1, 0, 0b1000),
        (0, 0, 0b1_0000),
    ];
    assert_eq!(classify(cores), Some(PerformanceCores { count: 3 }));
}

#[test]
fn performance_cores_in_any_group_count() {
    let cores = [(1, 0, 0b1), (1, 1, 0b1), (0, 0, 0b10)];
    assert_eq!(classify(cores), Some(PerformanceCores { count: 2 }));
}

#[test]
fn only_mps_and_the_statevector_band_take_the_narrow_pool() {
    use crate::sim::ResolvedBackend;
    let threshold = super::super::PARALLEL_THRESHOLD_QUBITS;
    assert!(route_wants_narrow_pool(ResolvedBackend::Mps, 64));
    assert!(!route_wants_narrow_pool(ResolvedBackend::TensorNetwork, 64));
    assert!(route_wants_narrow_pool(
        ResolvedBackend::Statevector,
        threshold
    ));
    assert!(route_wants_narrow_pool(
        ResolvedBackend::Statevector,
        MAX_NARROW_STATEVECTOR_QUBITS
    ));
    assert!(!route_wants_narrow_pool(
        ResolvedBackend::Statevector,
        threshold - 1
    ));
    assert!(!route_wants_narrow_pool(
        ResolvedBackend::Statevector,
        MAX_NARROW_STATEVECTOR_QUBITS + 1
    ));
    assert!(!route_wants_narrow_pool(ResolvedBackend::Stabilizer, 64));
}
