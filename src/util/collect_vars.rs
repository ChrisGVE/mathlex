//! Helpers for collecting variable names from an expression tree.

use crate::ast::{ExprKind, Expression};
use std::collections::HashSet;

use super::walker::for_each_child;

/// Collect all variable names (including index/bound variables) into `vars`.
pub(super) fn cv_core(expr: &Expression, vars: &mut HashSet<String>) {
    // Extract variable names from the current node
    match &expr.kind {
        ExprKind::Variable(name) => {
            vars.insert(name.clone());
        }
        ExprKind::Derivative { var, .. }
        | ExprKind::PartialDerivative { var, .. }
        | ExprKind::Integral { var, .. }
        | ExprKind::ClosedIntegral { var, .. } => {
            vars.insert(var.clone());
        }
        ExprKind::MultipleIntegral { vars: ivars, .. } => {
            for v in ivars {
                vars.insert(v.clone());
            }
        }
        ExprKind::Limit { var, .. } => {
            vars.insert(var.clone());
        }
        ExprKind::Sum { index, .. } | ExprKind::Product { index, .. } => {
            vars.insert(index.clone());
        }
        ExprKind::ForAll { variable, .. } | ExprKind::Exists { variable, .. } => {
            vars.insert(variable.clone());
        }
        ExprKind::SetBuilder { variable, .. } => {
            vars.insert(variable.clone());
        }
        ExprKind::Tensor { indices, .. }
        | ExprKind::KroneckerDelta { indices }
        | ExprKind::LeviCivita { indices } => {
            for idx in indices {
                vars.insert(idx.name.clone());
            }
        }
        ExprKind::Differential { var } => {
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
    let found = match &expr.kind {
        ExprKind::Variable(n) => n == name,
        ExprKind::Derivative { var, .. }
        | ExprKind::PartialDerivative { var, .. }
        | ExprKind::Integral { var, .. }
        | ExprKind::ClosedIntegral { var, .. }
        | ExprKind::Limit { var, .. } => var == name,
        ExprKind::MultipleIntegral { vars, .. } => vars.iter().any(|v| v == name),
        ExprKind::Sum { index, .. } | ExprKind::Product { index, .. } => index == name,
        ExprKind::ForAll { variable, .. }
        | ExprKind::Exists { variable, .. }
        | ExprKind::SetBuilder { variable, .. } => variable == name,
        ExprKind::Tensor { indices, .. }
        | ExprKind::KroneckerDelta { indices }
        | ExprKind::LeviCivita { indices } => indices.iter().any(|idx| idx.name == name),
        ExprKind::Differential { var } => var == name,
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
