//! Helpers for collecting mathematical constants from an expression tree.

use crate::ast::{Expression, MathConstant};
use std::collections::HashSet;

use super::walker::for_each_child;

pub(super) fn cc_core(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    if let Expression::Constant(c) = expr {
        cs.insert(*c);
    }
    for_each_child(expr, |child| cc_core(child, cs));
}
