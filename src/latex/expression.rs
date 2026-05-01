use super::helpers::{
    to_latex_binary, to_latex_calculus, to_latex_function, to_latex_literal, to_latex_unary,
};
use super::helpers_linalg::{to_latex_linear_algebra, to_latex_logic_sets, to_latex_relations};
use super::trait_def::ToLatex;
use crate::ast::{ExprKind, Expression};

impl ToLatex for ExprKind {
    fn to_latex(&self) -> String {
        Expression::from(self.clone()).to_latex()
    }
}

impl ToLatex for Expression {
    fn to_latex(&self) -> String {
        match &self.kind {
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Rational { .. }
            | ExprKind::Complex { .. }
            | ExprKind::Quaternion { .. }
            | ExprKind::Variable(_)
            | ExprKind::Constant(_) => to_latex_literal(self),

            ExprKind::Binary { op, left, right } => to_latex_binary(op, left, right),

            ExprKind::Unary { op, operand } => to_latex_unary(op, operand),

            ExprKind::Function { name, args } => to_latex_function(name, args),

            ExprKind::Derivative { .. }
            | ExprKind::PartialDerivative { .. }
            | ExprKind::Integral { .. }
            | ExprKind::MultipleIntegral { .. }
            | ExprKind::ClosedIntegral { .. }
            | ExprKind::Limit { .. }
            | ExprKind::Sum { .. }
            | ExprKind::Product { .. } => to_latex_calculus(self),

            ExprKind::Vector(_)
            | ExprKind::Matrix(_)
            | ExprKind::MarkedVector { .. }
            | ExprKind::DotProduct { .. }
            | ExprKind::CrossProduct { .. }
            | ExprKind::OuterProduct { .. }
            | ExprKind::Gradient { .. }
            | ExprKind::Divergence { .. }
            | ExprKind::Curl { .. }
            | ExprKind::Laplacian { .. }
            | ExprKind::Nabla
            | ExprKind::Determinant { .. }
            | ExprKind::Trace { .. }
            | ExprKind::Rank { .. }
            | ExprKind::ConjugateTranspose { .. }
            | ExprKind::MatrixInverse { .. }
            | ExprKind::Tensor { .. }
            | ExprKind::KroneckerDelta { .. }
            | ExprKind::LeviCivita { .. } => to_latex_linear_algebra(self),

            ExprKind::Equation { .. }
            | ExprKind::Inequality { .. }
            | ExprKind::ForAll { .. }
            | ExprKind::Exists { .. }
            | ExprKind::Logical { .. }
            | ExprKind::SetOperation { .. }
            | ExprKind::SetRelationExpr { .. }
            | ExprKind::SetBuilder { .. }
            | ExprKind::EmptySet
            | ExprKind::PowerSet { .. }
            | ExprKind::NumberSetExpr(_) => to_latex_logic_sets(self),

            ExprKind::FunctionSignature { .. }
            | ExprKind::Composition { .. }
            | ExprKind::Differential { .. }
            | ExprKind::WedgeProduct { .. }
            | ExprKind::Relation { .. } => to_latex_relations(self),
        }
    }
}
