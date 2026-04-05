//! `fmt::Display` implementation for `Expression`.

use crate::ast::Expression;
use std::fmt;

use super::helpers::{
    fmt_binary, fmt_calculus, fmt_function, fmt_linear_algebra, fmt_literal, fmt_logic_sets,
    fmt_relations, fmt_unary,
};

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Literals
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Rational { .. }
            | Expression::Complex { .. }
            | Expression::Quaternion { .. }
            | Expression::Variable(_)
            | Expression::Constant(_) => fmt_literal(self, f),

            // Arithmetic
            Expression::Binary { .. } => fmt_binary(self, f),
            Expression::Unary { .. } => fmt_unary(self, f),
            Expression::Function { .. } => fmt_function(self, f),

            // Calculus
            Expression::Derivative { .. }
            | Expression::PartialDerivative { .. }
            | Expression::Integral { .. }
            | Expression::MultipleIntegral { .. }
            | Expression::ClosedIntegral { .. }
            | Expression::Limit { .. }
            | Expression::Sum { .. }
            | Expression::Product { .. } => fmt_calculus(self, f),

            // Linear algebra and tensors
            Expression::Vector(_)
            | Expression::Matrix(_)
            | Expression::MarkedVector { .. }
            | Expression::DotProduct { .. }
            | Expression::CrossProduct { .. }
            | Expression::OuterProduct { .. }
            | Expression::Gradient { .. }
            | Expression::Divergence { .. }
            | Expression::Curl { .. }
            | Expression::Laplacian { .. }
            | Expression::Nabla
            | Expression::Determinant { .. }
            | Expression::Trace { .. }
            | Expression::Rank { .. }
            | Expression::ConjugateTranspose { .. }
            | Expression::MatrixInverse { .. }
            | Expression::Tensor { .. }
            | Expression::KroneckerDelta { .. }
            | Expression::LeviCivita { .. } => fmt_linear_algebra(self, f),

            // Logic and sets
            Expression::Equation { .. }
            | Expression::Inequality { .. }
            | Expression::ForAll { .. }
            | Expression::Exists { .. }
            | Expression::Logical { .. }
            | Expression::NumberSetExpr(_)
            | Expression::SetOperation { .. }
            | Expression::SetRelationExpr { .. }
            | Expression::SetBuilder { .. }
            | Expression::EmptySet
            | Expression::PowerSet { .. } => fmt_logic_sets(self, f),

            // Relations and misc
            Expression::FunctionSignature { .. }
            | Expression::Composition { .. }
            | Expression::Differential { .. }
            | Expression::WedgeProduct { .. }
            | Expression::Relation { .. } => fmt_relations(self, f),
        }
    }
}
