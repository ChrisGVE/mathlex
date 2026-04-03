//! Helpers for collecting variable names from an expression tree.

use crate::ast::Expression;
use std::collections::HashSet;

pub(super) fn cv_core(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => {}
        Expression::Variable(name) => {
            vars.insert(name.clone());
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
            l.collect_variables(vars);
            r.collect_variables(vars);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_variables(vars);
            i.collect_variables(vars);
            j.collect_variables(vars);
            k.collect_variables(vars);
        }
        Expression::Unary { operand, .. } => operand.collect_variables(vars),
        Expression::Function { args, .. } => {
            for a in args {
                a.collect_variables(vars);
            }
        }
        _ => cv_calculus_and_rest(expr, vars),
    }
}

fn cv_calculus_and_rest(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::Derivative { expr: e, var, .. }
        | Expression::PartialDerivative { expr: e, var, .. } => {
            e.collect_variables(vars);
            vars.insert(var.clone());
        }
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            integrand.collect_variables(vars);
            vars.insert(var.clone());
            if let Some(b) = bounds {
                b.lower.collect_variables(vars);
                b.upper.collect_variables(vars);
            }
        }
        Expression::MultipleIntegral {
            integrand,
            vars: ivars,
            bounds,
            ..
        } => {
            integrand.collect_variables(vars);
            for v in ivars {
                vars.insert(v.clone());
            }
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_variables(vars);
                    ib.upper.collect_variables(vars);
                }
            }
        }
        Expression::ClosedIntegral { integrand, var, .. } => {
            integrand.collect_variables(vars);
            vars.insert(var.clone());
        }
        Expression::Limit {
            expr: e, var, to, ..
        } => {
            e.collect_variables(vars);
            vars.insert(var.clone());
            to.collect_variables(vars);
        }
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        }
        | Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            vars.insert(index.clone());
            lower.collect_variables(vars);
            upper.collect_variables(vars);
            body.collect_variables(vars);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_variables(vars);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_variables(vars);
                }
            }
        }
        _ => cv_logic_sets_and_rest(expr, vars),
    }
}

fn cv_logic_sets_and_rest(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::ForAll {
            variable,
            domain,
            body,
        }
        | Expression::Exists {
            variable,
            domain,
            body,
            ..
        } => {
            vars.insert(variable.clone());
            if let Some(d) = domain {
                d.collect_variables(vars);
            }
            body.collect_variables(vars);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_variables(vars);
            }
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla => {}
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            left.collect_variables(vars);
            right.collect_variables(vars);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_variables(vars);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_variables(vars);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_variables(vars);
        }
        _ => cv_sets_tensors_theory(expr, vars),
    }
}

fn cv_sets_tensors_theory(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_variables(vars);
            set.collect_variables(vars);
        }
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => {
            vars.insert(variable.clone());
            if let Some(d) = domain {
                d.collect_variables(vars);
            }
            predicate.collect_variables(vars);
        }
        Expression::PowerSet { set } => set.collect_variables(vars),
        Expression::Tensor { indices, .. }
        | Expression::KroneckerDelta { indices }
        | Expression::LeviCivita { indices } => {
            for idx in indices {
                vars.insert(idx.name.clone());
            }
        }
        Expression::Differential { var } => {
            vars.insert(var.clone());
        }
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_variables(vars);
            codomain.collect_variables(vars);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_variables(vars);
            inner.collect_variables(vars);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_variables(vars);
            right.collect_variables(vars);
        }
        _ => {}
    }
}
