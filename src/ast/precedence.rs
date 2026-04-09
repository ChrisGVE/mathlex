//! Shared precedence and parenthesization logic for Display and ToLatex.

use super::{BinaryOp, Expression, UnaryOp};

/// Get the precedence level of a binary operator.
///
/// Lower numbers bind less tightly (evaluated later).
pub(crate) fn precedence(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::PlusMinus | BinaryOp::MinusPlus => 1,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        BinaryOp::Pow => 3,
    }
}

/// Determine if an expression needs parentheses when used as a child of a binary operation.
///
/// Parentheses are needed when:
/// - The child is a binary operation with lower precedence than the parent
/// - The child is on the right side of a non-commutative operation with equal precedence
/// - The child is a unary prefix operator used as the base of a power
pub(crate) fn needs_parens(child: &Expression, parent_op: BinaryOp, is_right: bool) -> bool {
    match child {
        Expression::Binary { op: child_op, .. } => {
            let parent_prec = precedence(parent_op);
            let child_prec = precedence(*child_op);

            if child_prec < parent_prec {
                return true;
            }

            if child_prec == parent_prec {
                match (parent_op, *child_op) {
                    // Power is right-associative: a^b^c means a^(b^c)
                    (BinaryOp::Pow, BinaryOp::Pow) => return true,
                    // Sub and Div are left-associative, so right side needs parens
                    (BinaryOp::Sub, BinaryOp::Sub) | (BinaryOp::Div, BinaryOp::Div) => {
                        return is_right
                    }
                    // Add and Mul are commutative, no parens needed
                    _ => {}
                }
            }

            false
        }
        // Unary prefix operators (Neg, Pos) need parens when used as base of power
        // because -1^2 parses as -(1^2), not (-1)^2
        Expression::Unary { op, .. } => {
            matches!(
                (parent_op, op, is_right),
                (BinaryOp::Pow, UnaryOp::Neg | UnaryOp::Pos, false)
            )
        }
        _ => false,
    }
}
