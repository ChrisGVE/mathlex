use super::trait_def::ToLatex;
use crate::ast::{BinaryOp, Direction, InequalityOp, LogicalOp, MathConstant, RelationOp};

impl ToLatex for MathConstant {
    fn to_latex(&self) -> String {
        match self {
            MathConstant::Pi => r"\pi".to_string(),
            MathConstant::E => "e".to_string(),
            MathConstant::I => "i".to_string(),
            MathConstant::J => r"\mathbf{j}".to_string(),
            MathConstant::K => r"\mathbf{k}".to_string(),
            MathConstant::Infinity => r"\infty".to_string(),
            MathConstant::NegInfinity => r"-\infty".to_string(),
        }
    }
}

impl ToLatex for BinaryOp {
    fn to_latex(&self) -> String {
        match self {
            BinaryOp::Add => "+".to_string(),
            BinaryOp::Sub => "-".to_string(),
            BinaryOp::Mul => r"\cdot".to_string(),
            BinaryOp::Div => "/".to_string(), // Division is handled specially in Expression
            BinaryOp::Pow => "^".to_string(),
            BinaryOp::Mod => r"\bmod".to_string(),
            BinaryOp::PlusMinus => r"\pm".to_string(),
            BinaryOp::MinusPlus => r"\mp".to_string(),
        }
    }
}

impl ToLatex for InequalityOp {
    fn to_latex(&self) -> String {
        match self {
            InequalityOp::Lt => "<".to_string(),
            InequalityOp::Le => r"\leq".to_string(),
            InequalityOp::Gt => ">".to_string(),
            InequalityOp::Ge => r"\geq".to_string(),
            InequalityOp::Ne => r"\neq".to_string(),
        }
    }
}

impl ToLatex for Direction {
    fn to_latex(&self) -> String {
        match self {
            Direction::Left => "^-".to_string(),
            Direction::Right => "^+".to_string(),
            Direction::Both => "".to_string(),
        }
    }
}

impl ToLatex for LogicalOp {
    fn to_latex(&self) -> String {
        match self {
            LogicalOp::And => r"\land".to_string(),
            LogicalOp::Or => r"\lor".to_string(),
            LogicalOp::Not => r"\lnot".to_string(),
            LogicalOp::Implies => r"\implies".to_string(),
            LogicalOp::Iff => r"\iff".to_string(),
        }
    }
}

impl ToLatex for RelationOp {
    fn to_latex(&self) -> String {
        match self {
            RelationOp::Similar => r"\sim".to_string(),
            RelationOp::Equivalent => r"\equiv".to_string(),
            RelationOp::Congruent => r"\cong".to_string(),
            RelationOp::Approx => r"\approx".to_string(),
        }
    }
}
