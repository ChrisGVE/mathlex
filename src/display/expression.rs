//! `fmt::Display` implementation for `Expression`.

use crate::ast::{ExprKind, Expression};
use std::fmt;

use super::helpers::{
    fmt_binary, fmt_calculus, fmt_function, fmt_linear_algebra, fmt_literal, fmt_logic_sets,
    fmt_relations, fmt_unary,
};

impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Expression::from(self.clone()))
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            // Literals
            ExprKind::Integer(_)
            | ExprKind::Float(_)
            | ExprKind::Rational { .. }
            | ExprKind::Complex { .. }
            | ExprKind::Quaternion { .. }
            | ExprKind::Variable(_)
            | ExprKind::Constant(_) => fmt_literal(self, f),

            // Arithmetic
            ExprKind::Binary { .. } => fmt_binary(self, f),
            ExprKind::Unary { .. } => fmt_unary(self, f),
            ExprKind::Function { .. } => fmt_function(self, f),

            // Calculus
            ExprKind::Derivative { .. }
            | ExprKind::PartialDerivative { .. }
            | ExprKind::Integral { .. }
            | ExprKind::MultipleIntegral { .. }
            | ExprKind::ClosedIntegral { .. }
            | ExprKind::Limit { .. }
            | ExprKind::Sum { .. }
            | ExprKind::Product { .. } => fmt_calculus(self, f),

            // Linear algebra and tensors
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
            | ExprKind::LeviCivita { .. } => fmt_linear_algebra(self, f),

            // Logic and sets
            ExprKind::Equation { .. }
            | ExprKind::Inequality { .. }
            | ExprKind::ForAll { .. }
            | ExprKind::Exists { .. }
            | ExprKind::Logical { .. }
            | ExprKind::NumberSetExpr(_)
            | ExprKind::SetOperation { .. }
            | ExprKind::SetRelationExpr { .. }
            | ExprKind::SetBuilder { .. }
            | ExprKind::EmptySet
            | ExprKind::PowerSet { .. } => fmt_logic_sets(self, f),

            // Relations and misc
            ExprKind::FunctionSignature { .. }
            | ExprKind::Composition { .. }
            | ExprKind::Differential { .. }
            | ExprKind::WedgeProduct { .. }
            | ExprKind::Relation { .. } => fmt_relations(self, f),
        }
    }
}
