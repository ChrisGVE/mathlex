//! Display implementations for operator and constant enums, and `IntegralBounds`.

use crate::ast::{
    BinaryOp, Direction, InequalityOp, IntegralBounds, LogicalOp, MathConstant, RelationOp, UnaryOp,
};
use std::fmt;

impl fmt::Display for MathConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathConstant::Pi => write!(f, "pi"),
            MathConstant::E => write!(f, "e"),
            MathConstant::I => write!(f, "i"),
            MathConstant::J => write!(f, "j"),
            MathConstant::K => write!(f, "k"),
            MathConstant::Infinity => write!(f, "inf"),
            MathConstant::NegInfinity => write!(f, "-inf"),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "^"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::PlusMinus => write!(f, "±"),
            BinaryOp::MinusPlus => write!(f, "∓"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Pos => write!(f, "+"),
            UnaryOp::Factorial => write!(f, "!"),
            UnaryOp::Transpose => write!(f, "'"),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Left => write!(f, "-"),
            Direction::Right => write!(f, "+"),
            Direction::Both => write!(f, ""),
        }
    }
}

impl fmt::Display for InequalityOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InequalityOp::Lt => write!(f, "<"),
            InequalityOp::Le => write!(f, "<="),
            InequalityOp::Gt => write!(f, ">"),
            InequalityOp::Ge => write!(f, ">="),
            InequalityOp::Ne => write!(f, "!="),
        }
    }
}

impl fmt::Display for LogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalOp::And => write!(f, "∧"),
            LogicalOp::Or => write!(f, "∨"),
            LogicalOp::Not => write!(f, "¬"),
            LogicalOp::Implies => write!(f, "→"),
            LogicalOp::Iff => write!(f, "↔"),
        }
    }
}

impl fmt::Display for RelationOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RelationOp::Similar => write!(f, "~"),
            RelationOp::Equivalent => write!(f, "≡"),
            RelationOp::Congruent => write!(f, "≅"),
            RelationOp::Approx => write!(f, "≈"),
        }
    }
}

impl fmt::Display for IntegralBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.lower, self.upper)
    }
}
