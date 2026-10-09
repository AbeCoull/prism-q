use super::*;
use crate::circuit::dynamic::{BinaryOp, UnaryOp};

fn eval(expr: &ClassicalExpr, bits: &[bool], vars: &[ClassicalValue]) -> ClassicalValue {
    expr.eval(bits, vars).unwrap()
}

#[test]
fn integer_types_wrap_to_their_width() {
    let int3 = ClassicalType::Int { width: 3 };
    let uint2 = ClassicalType::Uint { width: 2 };
    assert_eq!(int3.store(ClassicalValue::Int(4)), ClassicalValue::Int(-4));
    assert_eq!(int3.store(ClassicalValue::Int(-5)), ClassicalValue::Int(3));
    assert_eq!(uint2.store(ClassicalValue::Int(5)), ClassicalValue::Int(1));
    assert_eq!(uint2.store(ClassicalValue::Int(-1)), ClassicalValue::Int(3));
    let uint64 = ClassicalType::Uint { width: 64 };
    assert_eq!(
        uint64.store(ClassicalValue::Int(-1)),
        ClassicalValue::Int(i128::from(u64::MAX))
    );
    assert_eq!(
        uint2.store(ClassicalValue::Float(6.9)),
        ClassicalValue::Int(2)
    );
    assert_eq!(
        ClassicalType::Bool.store(ClassicalValue::Int(2)),
        ClassicalValue::Bool(true)
    );
    let angle = ClassicalType::Angle.store(ClassicalValue::Float(-std::f64::consts::FRAC_PI_2));
    assert_eq!(angle, ClassicalValue::Float(1.5 * std::f64::consts::PI));
}

#[test]
fn register_reads_weight_the_low_bit_first() {
    let bits = [true, false, true, true];
    let register = ClassicalExpr::Register { offset: 1, size: 3 };
    assert_eq!(eval(&register, &bits, &[]), ClassicalValue::Int(0b110));
    assert_eq!(
        eval(&ClassicalExpr::Bit(0), &bits, &[]),
        ClassicalValue::Bool(true)
    );
}

#[test]
fn integer_arithmetic_stays_integral_and_floats_promote() {
    let int = |v: i64| ClassicalExpr::from(v);
    let div = ClassicalExpr::binary(BinaryOp::Div, int(-7), int(2));
    assert_eq!(eval(&div, &[], &[]), ClassicalValue::Int(-3));
    let mixed = ClassicalExpr::binary(BinaryOp::Div, int(7), ClassicalExpr::from(2.0));
    assert_eq!(eval(&mixed, &[], &[]), ClassicalValue::Float(3.5));
    let pow = ClassicalExpr::binary(BinaryOp::Pow, int(3), int(4));
    assert_eq!(eval(&pow, &[], &[]), ClassicalValue::Int(81));
    let shifted = ClassicalExpr::binary(BinaryOp::Shl, int(1), int(5));
    assert_eq!(eval(&shifted, &[], &[]), ClassicalValue::Int(32));
    let not = ClassicalExpr::unary(UnaryOp::BitNot, int(0));
    assert_eq!(eval(&not, &[], &[]), ClassicalValue::Int(-1));
    let compare = ClassicalExpr::binary(BinaryOp::Lt, int(2), ClassicalExpr::from(2.5));
    assert_eq!(eval(&compare, &[], &[]), ClassicalValue::Bool(true));
}

#[test]
fn integer_division_by_zero_errors_rather_than_panicking() {
    let div = ClassicalExpr::binary(BinaryOp::Rem, 1i64.into(), 0i64.into());
    assert!(matches!(
        div.eval(&[], &[]),
        Err(PrismError::InvalidParameter { .. })
    ));
}

#[test]
fn logical_operators_short_circuit() {
    let failing = ClassicalExpr::binary(BinaryOp::Div, 1i64.into(), 0i64.into());
    let and = ClassicalExpr::binary(BinaryOp::And, false.into(), failing.clone());
    assert_eq!(eval(&and, &[], &[]), ClassicalValue::Bool(false));
    let or = ClassicalExpr::binary(BinaryOp::Or, true.into(), failing);
    assert_eq!(eval(&or, &[], &[]), ClassicalValue::Bool(true));
}

fn block(circuit: Circuit, terminator: Terminator) -> BasicBlock {
    BasicBlock {
        circuit,
        actions: Vec::new(),
        terminator,
        loop_name: None,
    }
}

#[test]
fn construction_rejects_a_missing_block_variable_or_bit() {
    let jump = vec![block(Circuit::new(1, 1), Terminator::Jump(BlockId::new(3)))];
    assert!(DynamicProgram::new(1, 1, Vec::new(), jump).is_err());

    let mut assign = block(Circuit::new(1, 1), Terminator::End);
    assign.actions.push(Action::Assign {
        var: VarId::new(0),
        value: 1i64.into(),
    });
    assert!(DynamicProgram::new(1, 1, Vec::new(), vec![assign]).is_err());

    let branch = block(
        Circuit::new(1, 1),
        Terminator::Branch {
            condition: ClassicalExpr::Bit(4),
            then: BlockId::new(0),
            otherwise: BlockId::new(0),
        },
    );
    assert!(matches!(
        DynamicProgram::new(1, 1, Vec::new(), vec![branch]),
        Err(PrismError::InvalidClassicalBit { index: 4, .. })
    ));

    let mut measured = Circuit::new(1, 1);
    measured.instructions.push(Instruction::Measure {
        qubit: 2,
        classical_bit: 0,
    });
    assert!(matches!(
        DynamicProgram::new(1, 1, Vec::new(), vec![block(measured, Terminator::End)]),
        Err(PrismError::InvalidQubit { index: 2, .. })
    ));
}

#[test]
fn construction_rejects_a_save_point_and_a_bitwise_float() {
    let mut saving = Circuit::new(1, 0);
    saving.add_save(crate::circuit::SaveSpec::Probabilities, "p");
    assert!(DynamicProgram::new(1, 0, Vec::new(), vec![block(saving, Terminator::End)]).is_err());

    let variables = vec![Variable {
        name: "t".into(),
        ty: ClassicalType::Float,
        initial: ClassicalValue::Float(0.5),
    }];
    let branch = block(
        Circuit::new(1, 0),
        Terminator::Branch {
            condition: ClassicalExpr::binary(BinaryOp::BitAnd, VarId::new(0).into(), 1i64.into()),
            then: BlockId::new(0),
            otherwise: BlockId::new(0),
        },
    );
    assert!(DynamicProgram::new(1, 0, variables, vec![branch]).is_err());
}

#[test]
fn construction_wraps_an_initial_value_to_its_type() {
    let variables = vec![Variable {
        name: "n".into(),
        ty: ClassicalType::Uint { width: 3 },
        initial: ClassicalValue::Int(9),
    }];
    let program = DynamicProgram::new(
        1,
        0,
        variables,
        vec![block(Circuit::new(1, 0), Terminator::End)],
    )
    .unwrap();
    assert_eq!(program.variables()[0].initial, ClassicalValue::Int(1));
}

#[test]
fn a_straight_line_program_is_its_circuit() {
    let mut b = DynamicProgramBuilder::new(2, 1);
    b.add_gate(crate::Gate::H, &[0])
        .add_gate(crate::Gate::Cx, &[0, 1]);
    b.add_measure(1, 0);
    let program = b.build().unwrap();
    let circuit = program.static_circuit().expect("no control flow");
    assert_eq!(circuit.instructions.len(), 3);
}

#[test]
fn an_if_and_its_else_share_one_branch_and_a_closed_if_reopens() {
    let mut b = DynamicProgramBuilder::new(1, 1);
    b.add_measure(0, 0);
    b.begin_if(ClassicalExpr::Bit(0));
    b.add_gate(crate::Gate::X, &[0]);
    b.end().unwrap();
    b.begin_else().unwrap();
    b.add_gate(crate::Gate::Z, &[0]);
    b.end().unwrap();
    assert!(b.begin_else().is_err(), "the `if` already has its `else`");
    let program = b.build().unwrap();
    let branches = program
        .blocks()
        .iter()
        .filter(|block| matches!(block.terminator, Terminator::Branch { .. }))
        .count();
    assert_eq!(branches, 1);
}

#[test]
fn builder_rejects_unbalanced_structure() {
    let mut b = DynamicProgramBuilder::new(1, 0);
    assert!(b.end().is_err());
    assert!(b.break_loop().is_err());
    assert!(b.continue_loop().is_err());
    b.begin_while("open", true.into());
    assert!(b.build().is_err());
}

#[test]
fn builder_expressions_read_variables_and_bits() {
    let mut b = DynamicProgramBuilder::new(1, 3);
    let n = b.declare("n", ClassicalType::Int { width: 8 }, 2i64.into());
    let expr = b.expr("n * 2 + c[1] == 5 && c < 4").unwrap();
    let vars = [ClassicalValue::Int(2)];
    assert_eq!(
        expr.eval(&[false, true, false], &vars).unwrap(),
        ClassicalValue::Bool(true)
    );
    assert_eq!(
        b.expr("n").unwrap(),
        ClassicalExpr::Var(n),
        "a bare name reads the variable"
    );
    assert_eq!(
        b.expr("2 * pi").unwrap(),
        ClassicalExpr::Const(ClassicalValue::Float(std::f64::consts::TAU))
    );
    assert!(b.expr("m + 1").is_err());
    assert!(b.expr("sin(n)").is_err());
}
