//! Helper functions for formatting AST nodes as plain text.

mod advanced;

pub(crate) use crate::ast::precedence::{needs_parens, precedence};
pub(crate) use advanced::{fmt_calculus, fmt_linear_algebra, fmt_logic_sets, fmt_relations};

use crate::ast::{Expression, UnaryOp};
use std::fmt;

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
