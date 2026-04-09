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

// ── contains_variable helpers ────────────────────────────────────────────────

pub(super) fn cv_contains(expr: &Expression, name: &str) -> bool {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => false,
        Expression::Variable(n) => n == name,
        Expression::Rational {
            numerator,
            denominator,
        } => cv_contains(numerator, name) || cv_contains(denominator, name),
        Expression::Complex { real, imaginary } => {
            cv_contains(real, name) || cv_contains(imaginary, name)
        }
        Expression::Quaternion { real, i, j, k } => {
            cv_contains(real, name)
                || cv_contains(i, name)
                || cv_contains(j, name)
                || cv_contains(k, name)
        }
        Expression::Binary { left, right, .. }
        | Expression::Equation { left, right }
        | Expression::Inequality { left, right, .. } => {
            cv_contains(left, name) || cv_contains(right, name)
        }
        Expression::Unary { operand, .. } => cv_contains(operand, name),
        Expression::Function { args, .. } => args.iter().any(|a| cv_contains(a, name)),
        _ => cv_contains_calculus_and_rest(expr, name),
    }
}

fn cv_contains_calculus_and_rest(expr: &Expression, name: &str) -> bool {
    match expr {
        Expression::Derivative { expr: e, var, .. }
        | Expression::PartialDerivative { expr: e, var, .. } => var == name || cv_contains(e, name),
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            var == name
                || cv_contains(integrand, name)
                || bounds
                    .as_ref()
                    .is_some_and(|b| cv_contains(&b.lower, name) || cv_contains(&b.upper, name))
        }
        Expression::MultipleIntegral {
            integrand,
            vars,
            bounds,
            ..
        } => {
            vars.iter().any(|v| v == name)
                || cv_contains(integrand, name)
                || bounds.as_ref().is_some_and(|b| {
                    b.bounds
                        .iter()
                        .any(|ib| cv_contains(&ib.lower, name) || cv_contains(&ib.upper, name))
                })
        }
        Expression::ClosedIntegral { integrand, var, .. } => {
            var == name || cv_contains(integrand, name)
        }
        Expression::Limit {
            expr: e, var, to, ..
        } => var == name || cv_contains(e, name) || cv_contains(to, name),
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
            index == name
                || cv_contains(lower, name)
                || cv_contains(upper, name)
                || cv_contains(body, name)
        }
        Expression::Vector(elems) => elems.iter().any(|e| cv_contains(e, name)),
        Expression::Matrix(rows) => rows
            .iter()
            .flat_map(|r| r.iter())
            .any(|e| cv_contains(e, name)),
        _ => cv_contains_logic_sets_and_rest(expr, name),
    }
}

fn cv_contains_logic_sets_and_rest(expr: &Expression, name: &str) -> bool {
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
            variable == name
                || domain.as_ref().is_some_and(|d| cv_contains(d, name))
                || cv_contains(body, name)
        }
        Expression::Logical { operands, .. } => operands.iter().any(|o| cv_contains(o, name)),
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla => false,
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            cv_contains(left, name) || cv_contains(right, name)
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => cv_contains(expr, name),
        Expression::Divergence { field } | Expression::Curl { field } => cv_contains(field, name),
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => cv_contains(matrix, name),
        _ => cv_contains_sets_tensors(expr, name),
    }
}

fn cv_contains_sets_tensors(expr: &Expression, name: &str) -> bool {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            cv_contains(element, name) || cv_contains(set, name)
        }
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => {
            variable == name
                || domain.as_ref().is_some_and(|d| cv_contains(d, name))
                || cv_contains(predicate, name)
        }
        Expression::PowerSet { set } => cv_contains(set, name),
        Expression::Tensor { indices, .. }
        | Expression::KroneckerDelta { indices }
        | Expression::LeviCivita { indices } => indices.iter().any(|idx| idx.name == name),
        Expression::Differential { var } => var == name,
        Expression::FunctionSignature {
            domain, codomain, ..
        } => cv_contains(domain, name) || cv_contains(codomain, name),
        Expression::Composition { outer, inner } => {
            cv_contains(outer, name) || cv_contains(inner, name)
        }
        Expression::Relation { left, right, .. } => {
            cv_contains(left, name) || cv_contains(right, name)
        }
        _ => false,
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
