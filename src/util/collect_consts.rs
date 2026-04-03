//! Helpers for collecting mathematical constants from an expression tree.

use crate::ast::{Expression, MathConstant};
use std::collections::HashSet;

pub(super) fn cc_core(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Variable(_) => {}
        Expression::Constant(c) => {
            cs.insert(*c);
        }
        Expression::Rational {
            numerator: l,
            denominator: r,
        }
        | Expression::Complex {
            real: l,
            imaginary: r,
        }
        | Expression::Equation { left: l, right: r }
        | Expression::Binary {
            left: l, right: r, ..
        }
        | Expression::Inequality {
            left: l, right: r, ..
        } => {
            l.collect_constants(cs);
            r.collect_constants(cs);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_constants(cs);
            i.collect_constants(cs);
            j.collect_constants(cs);
            k.collect_constants(cs);
        }
        Expression::Unary { operand, .. } => operand.collect_constants(cs),
        Expression::Function { args, .. } => {
            for a in args {
                a.collect_constants(cs);
            }
        }
        _ => cc_calculus_and_rest(expr, cs),
    }
}

fn cc_calculus_and_rest(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            e.collect_constants(cs)
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            integrand.collect_constants(cs);
            if let Some(b) = bounds {
                b.lower.collect_constants(cs);
                b.upper.collect_constants(cs);
            }
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            integrand.collect_constants(cs);
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_constants(cs);
                    ib.upper.collect_constants(cs);
                }
            }
        }
        Expression::ClosedIntegral { integrand, .. } => integrand.collect_constants(cs),
        Expression::Limit { expr: e, to, .. } => {
            e.collect_constants(cs);
            to.collect_constants(cs);
        }
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            lower.collect_constants(cs);
            upper.collect_constants(cs);
            body.collect_constants(cs);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_constants(cs);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_constants(cs);
                }
            }
        }
        _ => cc_logic_sets_and_rest(expr, cs),
    }
}

fn cc_logic_sets_and_rest(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                d.collect_constants(cs);
            }
            body.collect_constants(cs);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_constants(cs);
            }
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => {}
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            left.collect_constants(cs);
            right.collect_constants(cs);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_constants(cs);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_constants(cs);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_constants(cs);
        }
        _ => cc_sets_theory(expr, cs),
    }
}

fn cc_sets_theory(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_constants(cs);
            set.collect_constants(cs);
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            if let Some(d) = domain {
                d.collect_constants(cs);
            }
            predicate.collect_constants(cs);
        }
        Expression::PowerSet { set } => set.collect_constants(cs),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_constants(cs);
            codomain.collect_constants(cs);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_constants(cs);
            inner.collect_constants(cs);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_constants(cs);
            right.collect_constants(cs);
        }
        _ => {}
    }
}
