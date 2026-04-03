//! Helpers for collecting function names from an expression tree.

use crate::ast::Expression;
use std::collections::HashSet;

pub(super) fn cf_core(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_) => {}
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
            l.collect_functions(fns);
            r.collect_functions(fns);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_functions(fns);
            i.collect_functions(fns);
            j.collect_functions(fns);
            k.collect_functions(fns);
        }
        Expression::Unary { operand, .. } => operand.collect_functions(fns),
        Expression::Function { name, args } => {
            fns.insert(name.clone());
            for a in args {
                a.collect_functions(fns);
            }
        }
        _ => cf_calculus_and_rest(expr, fns),
    }
}

fn cf_calculus_and_rest(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            e.collect_functions(fns)
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            integrand.collect_functions(fns);
            if let Some(b) = bounds {
                b.lower.collect_functions(fns);
                b.upper.collect_functions(fns);
            }
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            integrand.collect_functions(fns);
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_functions(fns);
                    ib.upper.collect_functions(fns);
                }
            }
        }
        Expression::ClosedIntegral { integrand, .. } => integrand.collect_functions(fns),
        Expression::Limit { expr: e, to, .. } => {
            e.collect_functions(fns);
            to.collect_functions(fns);
        }
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            lower.collect_functions(fns);
            upper.collect_functions(fns);
            body.collect_functions(fns);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_functions(fns);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_functions(fns);
                }
            }
        }
        _ => cf_logic_sets_and_rest(expr, fns),
    }
}

fn cf_logic_sets_and_rest(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                d.collect_functions(fns);
            }
            body.collect_functions(fns);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_functions(fns);
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
            left.collect_functions(fns);
            right.collect_functions(fns);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_functions(fns);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_functions(fns);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_functions(fns);
        }
        _ => cf_sets_theory(expr, fns),
    }
}

fn cf_sets_theory(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_functions(fns);
            set.collect_functions(fns);
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            if let Some(d) = domain {
                d.collect_functions(fns);
            }
            predicate.collect_functions(fns);
        }
        Expression::PowerSet { set } => set.collect_functions(fns),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_functions(fns);
            codomain.collect_functions(fns);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_functions(fns);
            inner.collect_functions(fns);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_functions(fns);
            right.collect_functions(fns);
        }
        _ => {}
    }
}
