//! Helpers for collecting function names from an expression tree.

use crate::ast::Expression;
use std::collections::HashSet;

use super::walker::for_each_child;

pub(super) fn cf_core(expr: &Expression, fns: &mut HashSet<String>) {
    if let Expression::Function { name, .. } = expr {
        fns.insert(name.clone());
    }
    for_each_child(expr, |child| cf_core(child, fns));
}
