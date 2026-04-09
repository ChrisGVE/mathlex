//! Helpers for collecting variable names from an expression tree.

use crate::ast::Expression;
use std::collections::HashSet;

use super::walker::for_each_child;

/// Collect all variable names (including index/bound variables) into `vars`.
pub(super) fn cv_core(expr: &Expression, vars: &mut HashSet<String>) {
    // Extract variable names from the current node
    match expr {
        Expression::Variable(name) => {
            vars.insert(name.clone());
        }
        Expression::Derivative { var, .. }
        | Expression::PartialDerivative { var, .. }
        | Expression::Integral { var, .. }
        | Expression::ClosedIntegral { var, .. } => {
            vars.insert(var.clone());
        }
        Expression::MultipleIntegral { vars: ivars, .. } => {
            for v in ivars {
                vars.insert(v.clone());
            }
        }
        Expression::Limit { var, .. } => {
            vars.insert(var.clone());
        }
        Expression::Sum { index, .. } | Expression::Product { index, .. } => {
            vars.insert(index.clone());
        }
        Expression::ForAll { variable, .. } | Expression::Exists { variable, .. } => {
            vars.insert(variable.clone());
        }
        Expression::SetBuilder { variable, .. } => {
            vars.insert(variable.clone());
        }
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
        _ => {}
    }
    // Recurse into child expressions
    for_each_child(expr, |child| cv_core(child, vars));
}

/// Check whether `expr` contains a variable with the given `name`.
pub(super) fn cv_contains(expr: &Expression, name: &str) -> bool {
    // Check the current node for a matching variable name
    let found = match expr {
        Expression::Variable(n) => n == name,
        Expression::Derivative { var, .. }
        | Expression::PartialDerivative { var, .. }
        | Expression::Integral { var, .. }
        | Expression::ClosedIntegral { var, .. }
        | Expression::Limit { var, .. } => var == name,
        Expression::MultipleIntegral { vars, .. } => vars.iter().any(|v| v == name),
        Expression::Sum { index, .. } | Expression::Product { index, .. } => index == name,
        Expression::ForAll { variable, .. }
        | Expression::Exists { variable, .. }
        | Expression::SetBuilder { variable, .. } => variable == name,
        Expression::Tensor { indices, .. }
        | Expression::KroneckerDelta { indices }
        | Expression::LeviCivita { indices } => indices.iter().any(|idx| idx.name == name),
        Expression::Differential { var } => var == name,
        _ => false,
    };
    if found {
        return true;
    }
    // Recurse into children
    let mut result = false;
    for_each_child(expr, |child| {
        if !result {
            result = cv_contains(child, name);
        }
    });
    result
}
