use super::helpers::{
    to_latex_binary, to_latex_calculus, to_latex_function, to_latex_literal, to_latex_unary,
};
use super::helpers_linalg::{to_latex_linear_algebra, to_latex_logic_sets, to_latex_relations};
use super::trait_def::ToLatex;
use crate::ast::Expression;

impl ToLatex for Expression {
    fn to_latex(&self) -> String {
        match self {
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Rational { .. }
            | Expression::Complex { .. }
            | Expression::Quaternion { .. }
            | Expression::Variable(_)
            | Expression::Constant(_) => to_latex_literal(self),

            Expression::Binary { op, left, right } => to_latex_binary(op, left, right),

            Expression::Unary { op, operand } => to_latex_unary(op, operand),

            Expression::Function { name, args } => to_latex_function(name, args),

            Expression::Derivative { .. }
            | Expression::PartialDerivative { .. }
            | Expression::Integral { .. }
            | Expression::MultipleIntegral { .. }
            | Expression::ClosedIntegral { .. }
            | Expression::Limit { .. }
            | Expression::Sum { .. }
            | Expression::Product { .. } => to_latex_calculus(self),

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
            | Expression::LeviCivita { .. } => to_latex_linear_algebra(self),

            Expression::Equation { .. }
            | Expression::Inequality { .. }
            | Expression::ForAll { .. }
            | Expression::Exists { .. }
            | Expression::Logical { .. }
            | Expression::SetOperation { .. }
            | Expression::SetRelationExpr { .. }
            | Expression::SetBuilder { .. }
            | Expression::EmptySet
            | Expression::PowerSet { .. }
            | Expression::NumberSetExpr(_) => to_latex_logic_sets(self),

            Expression::FunctionSignature { .. }
            | Expression::Composition { .. }
            | Expression::Differential { .. }
            | Expression::WedgeProduct { .. }
            | Expression::Relation { .. } => to_latex_relations(self),
        }
    }
}
