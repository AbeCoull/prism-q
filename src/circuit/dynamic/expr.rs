//! Runtime classical values, variable types, and the expressions a dynamic
//! program evaluates once per shot.

use crate::error::{PrismError, Result};

/// Index of a runtime classical variable in a [`DynamicProgram`](super::DynamicProgram).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(u32);

impl VarId {
    /// # Panics
    ///
    /// Panics when `index` does not fit in 32 bits.
    pub fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("variable index fits in 32 bits"))
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Declared type of a runtime classical variable, which decides how a stored
/// value is wrapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClassicalType {
    Bool,
    /// Two's complement over `width` bits, 1 to 64, wrapping on overflow.
    Int {
        width: u32,
    },
    /// Unsigned over `width` bits, 1 to 64, wrapping on overflow.
    Uint {
        width: u32,
    },
    Float,
    /// A float held modulo `2 pi`, in `[0, 2 pi)`.
    Angle,
}

impl ClassicalType {
    pub(crate) fn kind(self) -> ValueKind {
        match self {
            ClassicalType::Bool => ValueKind::Bool,
            ClassicalType::Int { .. } | ClassicalType::Uint { .. } => ValueKind::Int,
            ClassicalType::Float | ClassicalType::Angle => ValueKind::Float,
        }
    }

    pub(crate) fn check(self) -> Result<()> {
        match self {
            ClassicalType::Int { width } | ClassicalType::Uint { width }
                if !(1..=64).contains(&width) =>
            {
                Err(PrismError::InvalidParameter {
                    message: format!("integer width {width} is outside 1 to 64"),
                })
            }
            _ => Ok(()),
        }
    }

    /// `value` converted to this type: truthiness for `bool`, truncation
    /// toward zero and then wraparound for the integers, and the reduction
    /// into `[0, 2 pi)` for `angle`.
    pub(crate) fn store(self, value: ClassicalValue) -> ClassicalValue {
        match self {
            ClassicalType::Bool => ClassicalValue::Bool(value.truthy()),
            ClassicalType::Int { width } => {
                let modulus = 1i128 << width;
                let wrapped = value.to_int().rem_euclid(modulus);
                ClassicalValue::Int(if wrapped >= modulus / 2 {
                    wrapped - modulus
                } else {
                    wrapped
                })
            }
            ClassicalType::Uint { width } => {
                ClassicalValue::Int(value.to_int().rem_euclid(1i128 << width))
            }
            ClassicalType::Float => ClassicalValue::Float(value.to_float()),
            ClassicalType::Angle => {
                ClassicalValue::Float(value.to_float().rem_euclid(std::f64::consts::TAU))
            }
        }
    }
}

/// A runtime classical value. Integers carry 128 bits, so every 64-bit `int`
/// and `uint` is exact before its declared width wraps it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassicalValue {
    Bool(bool),
    Int(i128),
    Float(f64),
}

impl ClassicalValue {
    pub(crate) fn truthy(self) -> bool {
        match self {
            ClassicalValue::Bool(value) => value,
            ClassicalValue::Int(value) => value != 0,
            ClassicalValue::Float(value) => value != 0.0,
        }
    }

    pub(crate) fn to_float(self) -> f64 {
        match self {
            ClassicalValue::Bool(value) => f64::from(u8::from(value)),
            ClassicalValue::Int(value) => value as f64,
            ClassicalValue::Float(value) => value,
        }
    }

    /// Truncated toward zero; a NaN reads as zero and an infinity saturates.
    pub(crate) fn to_int(self) -> i128 {
        match self {
            ClassicalValue::Bool(value) => i128::from(value),
            ClassicalValue::Int(value) => value,
            ClassicalValue::Float(value) => value as i128,
        }
    }

    pub(crate) fn kind(self) -> ValueKind {
        match self {
            ClassicalValue::Bool(_) => ValueKind::Bool,
            ClassicalValue::Int(_) => ValueKind::Int,
            ClassicalValue::Float(_) => ValueKind::Float,
        }
    }
}

impl From<bool> for ClassicalValue {
    fn from(value: bool) -> Self {
        ClassicalValue::Bool(value)
    }
}

impl From<i64> for ClassicalValue {
    fn from(value: i64) -> Self {
        ClassicalValue::Int(i128::from(value))
    }
}

impl From<f64> for ClassicalValue {
    fn from(value: f64) -> Self {
        ClassicalValue::Float(value)
    }
}

/// What an expression evaluates to, for the checks made at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Bool,
    Int,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryOp {
    Neg,
    /// Logical negation of the operand's truthiness.
    Not,
    /// Bitwise complement of an integer.
    BitNot,
}

/// Binary operators. Arithmetic on two integers stays integral, with `/`
/// truncating toward zero; a float on either side makes it a float.
/// Comparisons and the logical pair return a `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl BinaryOp {
    fn is_bitwise(self) -> bool {
        matches!(
            self,
            BinaryOp::Shl | BinaryOp::Shr | BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
        )
    }
}

/// A classical expression over runtime variables and measured bits.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClassicalExpr {
    Const(ClassicalValue),
    Var(VarId),
    /// Classical bit `index`, read as a `bool`.
    Bit(usize),
    /// Bits `offset..offset + size` read as an unsigned integer, bit
    /// `offset + i` weighted `1 << i`. At most 64 bits.
    Register {
        offset: usize,
        size: usize,
    },
    Unary {
        op: UnaryOp,
        operand: Box<ClassicalExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ClassicalExpr>,
        rhs: Box<ClassicalExpr>,
    },
}

impl ClassicalExpr {
    pub fn unary(op: UnaryOp, operand: ClassicalExpr) -> Self {
        ClassicalExpr::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    pub fn binary(op: BinaryOp, lhs: ClassicalExpr, rhs: ClassicalExpr) -> Self {
        ClassicalExpr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Check every reference against the program's widths and return the kind
    /// of value the expression produces.
    pub(crate) fn check(
        &self,
        variables: &[ClassicalType],
        num_classical_bits: usize,
    ) -> Result<ValueKind> {
        match self {
            ClassicalExpr::Const(value) => Ok(value.kind()),
            ClassicalExpr::Var(var) => {
                variables
                    .get(var.index())
                    .map(|ty| ty.kind())
                    .ok_or_else(|| PrismError::InvalidParameter {
                        message: format!(
                            "expression reads variable {} of {} declared",
                            var.index(),
                            variables.len()
                        ),
                    })
            }
            ClassicalExpr::Bit(bit) => {
                check_bit(*bit, num_classical_bits)?;
                Ok(ValueKind::Bool)
            }
            ClassicalExpr::Register { offset, size } => {
                if *size == 0 || *size > 64 {
                    return Err(PrismError::InvalidParameter {
                        message: format!("a register read spans {size} bits, outside 1 to 64"),
                    });
                }
                check_bit(offset + size - 1, num_classical_bits)?;
                Ok(ValueKind::Int)
            }
            ClassicalExpr::Unary { op, operand } => {
                let kind = operand.check(variables, num_classical_bits)?;
                match op {
                    UnaryOp::Not => Ok(ValueKind::Bool),
                    UnaryOp::Neg if kind == ValueKind::Float => Ok(ValueKind::Float),
                    UnaryOp::Neg => Ok(ValueKind::Int),
                    UnaryOp::BitNot if kind == ValueKind::Float => Err(float_operand("~")),
                    UnaryOp::BitNot => Ok(ValueKind::Int),
                }
            }
            ClassicalExpr::Binary { op, lhs, rhs } => {
                let left = lhs.check(variables, num_classical_bits)?;
                let right = rhs.check(variables, num_classical_bits)?;
                let float = left == ValueKind::Float || right == ValueKind::Float;
                Ok(match op {
                    op if op.is_bitwise() && float => {
                        return Err(float_operand(&format!("{op:?}")));
                    }
                    op if op.is_bitwise() => ValueKind::Int,
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem
                    | BinaryOp::Pow => {
                        if float {
                            ValueKind::Float
                        } else {
                            ValueKind::Int
                        }
                    }
                    _ => ValueKind::Bool,
                })
            }
        }
    }

    /// Evaluate against one shot's classical bits and variables.
    pub(crate) fn eval(&self, bits: &[bool], vars: &[ClassicalValue]) -> Result<ClassicalValue> {
        Ok(match self {
            ClassicalExpr::Const(value) => *value,
            ClassicalExpr::Var(var) => vars[var.index()],
            ClassicalExpr::Bit(bit) => ClassicalValue::Bool(bits[*bit]),
            ClassicalExpr::Register { offset, size } => ClassicalValue::Int(
                bits[*offset..offset + size]
                    .iter()
                    .rev()
                    .fold(0i128, |acc, &bit| (acc << 1) | i128::from(bit)),
            ),
            ClassicalExpr::Unary { op, operand } => {
                let value = operand.eval(bits, vars)?;
                match (op, value) {
                    (UnaryOp::Not, value) => ClassicalValue::Bool(!value.truthy()),
                    (UnaryOp::Neg, ClassicalValue::Float(value)) => ClassicalValue::Float(-value),
                    (UnaryOp::Neg, value) => ClassicalValue::Int(value.to_int().wrapping_neg()),
                    (UnaryOp::BitNot, ClassicalValue::Float(_)) => return Err(float_operand("~")),
                    (UnaryOp::BitNot, value) => ClassicalValue::Int(!value.to_int()),
                }
            }
            ClassicalExpr::Binary { op, lhs, rhs } => {
                let left = lhs.eval(bits, vars)?;
                match op {
                    BinaryOp::And if !left.truthy() => return Ok(ClassicalValue::Bool(false)),
                    BinaryOp::Or if left.truthy() => return Ok(ClassicalValue::Bool(true)),
                    BinaryOp::And | BinaryOp::Or => {
                        return Ok(ClassicalValue::Bool(rhs.eval(bits, vars)?.truthy()));
                    }
                    _ => {}
                }
                binary(*op, left, rhs.eval(bits, vars)?)?
            }
        })
    }
}

impl From<VarId> for ClassicalExpr {
    fn from(var: VarId) -> Self {
        ClassicalExpr::Var(var)
    }
}

impl From<bool> for ClassicalExpr {
    fn from(value: bool) -> Self {
        ClassicalExpr::Const(ClassicalValue::Bool(value))
    }
}

impl From<i64> for ClassicalExpr {
    fn from(value: i64) -> Self {
        ClassicalExpr::Const(value.into())
    }
}

impl From<f64> for ClassicalExpr {
    fn from(value: f64) -> Self {
        ClassicalExpr::Const(ClassicalValue::Float(value))
    }
}

pub(crate) fn check_bit(bit: usize, num_classical_bits: usize) -> Result<()> {
    if bit >= num_classical_bits {
        return Err(PrismError::InvalidClassicalBit {
            index: bit,
            register_size: num_classical_bits,
        });
    }
    Ok(())
}

fn float_operand(op: &str) -> PrismError {
    PrismError::InvalidParameter {
        message: format!("bitwise `{op}` on a float"),
    }
}

fn binary(op: BinaryOp, left: ClassicalValue, right: ClassicalValue) -> Result<ClassicalValue> {
    use ClassicalValue::{Bool, Float, Int};
    let float = matches!(left, Float(_)) || matches!(right, Float(_));
    if op.is_bitwise() {
        if float {
            return Err(float_operand(&format!("{op:?}")));
        }
        let (a, b) = (left.to_int(), right.to_int());
        let shift = b.clamp(0, 127) as u32;
        return Ok(Int(match op {
            BinaryOp::Shl if b > 127 => 0,
            BinaryOp::Shl => a << shift,
            BinaryOp::Shr => a >> shift,
            BinaryOp::BitAnd => a & b,
            BinaryOp::BitOr => a | b,
            _ => a ^ b,
        }));
    }
    let compare = |ordering: Option<std::cmp::Ordering>| -> Result<ClassicalValue> {
        use std::cmp::Ordering::{Equal, Greater, Less};
        Ok(Bool(match (op, ordering) {
            (BinaryOp::Ne, None) => true,
            (_, None) => false,
            (BinaryOp::Eq, Some(o)) => o == Equal,
            (BinaryOp::Ne, Some(o)) => o != Equal,
            (BinaryOp::Lt, Some(o)) => o == Less,
            (BinaryOp::Le, Some(o)) => o != Greater,
            (BinaryOp::Gt, Some(o)) => o == Greater,
            (_, Some(o)) => o != Less,
        }))
    };
    if matches!(
        op,
        BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
    ) {
        return if float {
            compare(left.to_float().partial_cmp(&right.to_float()))
        } else {
            compare(Some(left.to_int().cmp(&right.to_int())))
        };
    }
    if float {
        let (a, b) = (left.to_float(), right.to_float());
        return Ok(Float(match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Rem => a % b,
            _ => a.powf(b),
        }));
    }
    let (a, b) = (left.to_int(), right.to_int());
    let zero_divisor = || PrismError::InvalidParameter {
        message: format!("integer {op:?} by zero in a dynamic program"),
    };
    Ok(match op {
        BinaryOp::Add => Int(a.wrapping_add(b)),
        BinaryOp::Sub => Int(a.wrapping_sub(b)),
        BinaryOp::Mul => Int(a.wrapping_mul(b)),
        BinaryOp::Div => Int(a.checked_div(b).ok_or_else(zero_divisor)?),
        BinaryOp::Rem => Int(a.checked_rem(b).ok_or_else(zero_divisor)?),
        _ => match u32::try_from(b) {
            Ok(exponent) => Int(a.wrapping_pow(exponent)),
            Err(_) if b < 0 => Float((a as f64).powf(b as f64)),
            Err(_) => Int(a.wrapping_pow(u32::MAX)),
        },
    })
}
