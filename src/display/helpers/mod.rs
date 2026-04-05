//! Helper functions for formatting AST nodes as plain text.

mod advanced;

pub(crate) use advanced::{fmt_calculus, fmt_linear_algebra, fmt_logic_sets, fmt_relations};

use crate::ast::{BinaryOp, Expression, UnaryOp};
use std::fmt;

/// Get the precedence level of a binary operator.
///
/// Lower numbers bind less tightly (evaluated later).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(precedence(BinaryOp::Add), 1);
/// assert_eq!(precedence(BinaryOp::Mul), 2);
/// assert_eq!(precedence(BinaryOp::Pow), 3);
/// ```
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
///
/// # Arguments
///
/// - `child`: The child expression
/// - `parent_op`: The parent binary operator
/// - `is_right`: Whether the child is the right operand
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
                    _ => {}
                }
            }

            false
        }
        _ => false,
    }
}

/// Format literal and value expressions: Integer, Float, Rational, Complex,
/// Quaternion, Variable, Constant.
pub(crate) fn fmt_literal(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        Expression::Integer(n) => write!(f, "{}", n),
        Expression::Float(x) => write!(f, "{}", x),
        Expression::Rational {
            numerator,
            denominator,
        } => write!(f, "{}/{}", numerator, denominator),
        Expression::Complex { real, imaginary } => write!(f, "{} + {}i", real, imaginary),
        Expression::Quaternion { real, i, j, k } => {
            write!(f, "{} + {}i + {}j + {}k", real, i, j, k)
        }
        Expression::Variable(name) => write!(f, "{}", name),
        Expression::Constant(c) => write!(f, "{}", c),
        _ => unreachable!("fmt_literal called on non-literal"),
    }
}

/// Format binary operations, applying minimal parenthesization via precedence.
pub(crate) fn fmt_binary(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let Expression::Binary { op, left, right } = expr else {
        unreachable!("fmt_binary called on non-binary");
    };
    if needs_parens(left, *op, false) {
        write!(f, "({})", left)?;
    } else {
        write!(f, "{}", left)?;
    }
    write!(f, " {} ", op)?;
    if needs_parens(right, *op, true) {
        write!(f, "({})", right)
    } else {
        write!(f, "{}", right)
    }
}

/// Format unary operations (prefix and postfix).
pub(crate) fn fmt_unary(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let Expression::Unary { op, operand } = expr else {
        unreachable!("fmt_unary called on non-unary");
    };
    let is_binary = matches!(**operand, Expression::Binary { .. });
    match op {
        UnaryOp::Factorial | UnaryOp::Transpose => {
            if is_binary {
                write!(f, "({}){}", operand, op)
            } else {
                write!(f, "{}{}", operand, op)
            }
        }
        UnaryOp::Neg | UnaryOp::Pos => {
            if is_binary {
                write!(f, "{}({})", op, operand)
            } else {
                write!(f, "{}{}", op, operand)
            }
        }
    }
}

/// Format function-call expressions.
pub(crate) fn fmt_function(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let Expression::Function { name, args } = expr else {
        unreachable!("fmt_function called on non-function");
    };
    write!(f, "{}(", name)?;
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", arg)?;
    }
    write!(f, ")")
}
