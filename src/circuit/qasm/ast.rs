//! Syntax tree for an OpenQASM program.
//!
//! Names borrow the source text. Statements and operands carry their source line,
//! so a diagnostic from inside an expanded `for` or `gate` body points at the source.

use std::fmt;

use smallvec::SmallVec;

use super::expr::Expr;

pub(crate) type Block<'a> = Vec<Stmt<'a>>;

#[derive(Clone, Debug)]
pub(crate) struct Stmt<'a> {
    pub line: usize,
    pub kind: StmtKind<'a>,
}

#[derive(Clone, Debug)]
pub(crate) enum StmtKind<'a> {
    Version(&'a str),
    /// The gates a program may include are built in, so the path is dropped.
    Include,
    /// `qubit[n] q;`, `bit[n] c;`, and the `qreg`/`creg` spellings, which
    /// differ only in where the size is written.
    RegisterDecl {
        kind: RegisterKind,
        name: &'a str,
        size: Option<Expr<'a>>,
    },
    InputDecl {
        ty: &'a str,
        name: &'a str,
    },
    /// `output bit[3] c;`, which declares a classical register of that width.
    OutputDecl {
        ty: &'a str,
        name: &'a str,
        size: Option<Expr<'a>>,
    },
    ClassicalDecl {
        constant: bool,
        ty: &'a str,
        name: &'a str,
        value: Option<Expr<'a>>,
    },
    Assign {
        target: &'a str,
        op: Option<AssignOp>,
        value: Expr<'a>,
    },
    Alias {
        name: &'a str,
        sources: Vec<Operand<'a>>,
    },
    /// A gate application, a `def` call, or `gphase`, which share a spelling
    /// and are told apart by what is declared rather than by syntax.
    Call {
        modifiers: Vec<Modifier<'a>>,
        name: &'a str,
        params: SmallVec<[Argument<'a>; 1]>,
        operands: SmallVec<[Operand<'a>; 2]>,
    },
    /// Boxed because two operands would otherwise set the width of every
    /// statement in the tree.
    Measure(Box<Measure<'a>>),
    /// A lone `;`.
    Empty,
    Reset {
        targets: Vec<Operand<'a>>,
    },
    Barrier {
        /// Empty for a bare `barrier;`, which spans every qubit declared so far.
        targets: Vec<Operand<'a>>,
    },
    /// `delay[d] q;`. Empty targets mean every qubit.
    Delay {
        duration: Expr<'a>,
        targets: Vec<Operand<'a>>,
    },
    If(Box<Conditional<'a>>),
    For {
        variable: &'a str,
        range: ForRange<'a>,
        body: Block<'a>,
    },
    Switch {
        operand: Operand<'a>,
        arms: Vec<SwitchArm<'a>>,
    },
    GateDef {
        name: &'a str,
        params: Vec<&'a str>,
        qubits: Vec<&'a str>,
        body: Block<'a>,
    },
    DefDef {
        name: &'a str,
        args: Vec<DefParam<'a>>,
        result: Option<BitResult<'a>>,
        body: Block<'a>,
    },
    /// `return;`, `return measure q;` or `return r;`.
    Return(Option<ReturnValue<'a>>),
    /// `c[0] = f(q[0]);`, a call whose result is assigned. Boxed because the
    /// target and the arguments would otherwise set the width of every
    /// statement in the tree.
    CallAssign(Box<CallAssign<'a>>),
    /// `box { ... }` or `box[100ns] { ... }`.
    Box {
        duration: Option<Expr<'a>>,
        body: Block<'a>,
    },
    /// A `#pragma` line, text included.
    Pragma(&'a str),
}

#[derive(Clone, Debug)]
pub(crate) struct CallAssign<'a> {
    pub target: Operand<'a>,
    pub name: &'a str,
    pub args: SmallVec<[Argument<'a>; 1]>,
}

/// A `def` declared `-> bit` or `-> bit[n]`, the only result type it may have.
#[derive(Clone, Debug)]
pub(crate) struct BitResult<'a> {
    pub width: Option<Expr<'a>>,
}

#[derive(Clone, Debug)]
pub(crate) enum ReturnValue<'a> {
    Measure(Operand<'a>),
    Value(Expr<'a>),
}

#[derive(Clone, Debug)]
pub(crate) struct Measure<'a> {
    pub source: Operand<'a>,
    pub target: Operand<'a>,
}

#[derive(Clone, Debug)]
pub(crate) struct Conditional<'a> {
    pub condition: Condition<'a>,
    pub then_body: Block<'a>,
    pub else_body: Option<Block<'a>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RegisterKind {
    Qubit,
    Classical,
}

impl RegisterKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            RegisterKind::Qubit => "qubit",
            RegisterKind::Classical => "bit",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AssignOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Debug)]
pub(crate) enum Modifier<'a> {
    Inv,
    /// `pow(k)`. Real, so a fractional power reaches the matrix path.
    Pow(Expr<'a>),
    /// `ctrl` and `negctrl`, which differ only in the control polarity.
    Ctrl {
        negated: bool,
    },
}

/// One entry inside a call's parentheses. A gate takes angles; a `def` takes
/// qubits beside them, and only the declaration says which is which.
#[derive(Clone, Debug)]
pub(crate) enum Argument<'a> {
    Operand(Operand<'a>),
    Value(Expr<'a>),
}

/// A qubit or classical-bit reference: a register name or a physical index,
/// with an optional subscript.
#[derive(Clone, Debug)]
pub(crate) struct Operand<'a> {
    pub name: OperandName<'a>,
    pub index: Option<Index<'a>>,
    pub line: usize,
}

impl<'a> Operand<'a> {
    /// How the reference reads back in a diagnostic, subscript included.
    pub(crate) fn describe(&self) -> String {
        let base = match self.name {
            OperandName::Register(name) => name.to_string(),
            OperandName::Physical(index) => format!("${index}"),
        };
        match &self.index {
            Some(index) => format!("{base}[{index}]"),
            None => base,
        }
    }

    pub(crate) fn register(&self) -> Option<&'a str> {
        match self.name {
            OperandName::Register(name) => Some(name),
            OperandName::Physical(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum OperandName<'a> {
    Register(&'a str),
    Physical(usize),
}

#[derive(Clone, Debug)]
pub(crate) enum Index<'a> {
    Single(Expr<'a>),
    /// `start:stop` or `start:step:stop`, inclusive at both ends, with either
    /// bound open. Boxed because a range is rare and three bounds would
    /// otherwise set the width of every subscript.
    Range(Box<Range<'a>>),
    /// `{a, b, c}`, an explicit list in the order written.
    Set(Vec<Expr<'a>>),
}

impl fmt::Display for Index<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Index::Single(expr) => write!(f, "{expr}"),
            Index::Set(entries) => {
                f.write_str("{")?;
                for (at, entry) in entries.iter().enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{entry}")?;
                }
                f.write_str("}")
            }
            Index::Range(range) => {
                if let Some(start) = &range.start {
                    write!(f, "{start}")?;
                }
                f.write_str(":")?;
                if let Some(step) = &range.step {
                    write!(f, "{step}:")?;
                }
                match &range.stop {
                    Some(stop) => write!(f, "{stop}"),
                    None => Ok(()),
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Range<'a> {
    pub start: Option<Expr<'a>>,
    pub step: Option<Expr<'a>>,
    pub stop: Option<Expr<'a>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CmpOp {
    Equal,
    NotEqual,
}

#[derive(Clone, Debug)]
pub(crate) enum Condition<'a> {
    /// `if (c[0])`, true when the bit is set.
    Truthy(Operand<'a>),
    /// `if (!c[0])`, true when it is clear.
    Negated(Operand<'a>),
    /// `if (c == 1)` over a register, or over one bit.
    Compare {
        lhs: Operand<'a>,
        op: CmpOp,
        rhs: Expr<'a>,
    },
    /// `if (c[0] ^ c[2])`, optionally compared against `0` or `1`.
    Parity {
        bits: Vec<Operand<'a>>,
        compare: Option<(CmpOp, Expr<'a>)>,
    },
}

#[derive(Clone, Debug)]
pub(crate) enum ForRange<'a> {
    /// `[start:stop]` or `[start:step:stop]`, inclusive at both ends.
    Range {
        start: Expr<'a>,
        step: Option<Expr<'a>>,
        stop: Expr<'a>,
    },
    /// `{a, b, c}`.
    Set(Vec<Expr<'a>>),
}

#[derive(Clone, Debug)]
pub(crate) struct SwitchArm<'a> {
    /// `None` for the `default` arm.
    pub labels: Option<Vec<Expr<'a>>>,
    pub body: Block<'a>,
    pub line: usize,
}

#[derive(Clone, Debug)]
pub(crate) enum DefParam<'a> {
    Qubit(&'a str),
    Value {
        name: &'a str,
        integral: bool,
    },
    /// A `duration` or `stretch` parameter.
    Duration(&'a str),
    /// `bit` or `bit[n]`, passed by value.
    Bit {
        name: &'a str,
        width: Option<Expr<'a>>,
    },
}
