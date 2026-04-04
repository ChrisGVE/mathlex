use crate::ast::{BinaryOp, Expression, UnaryOp};

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
pub(super) fn precedence(op: BinaryOp) -> u8 {
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
pub(super) fn needs_parens(child: &Expression, parent_op: BinaryOp, is_right: bool) -> bool {
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

/// Wrap an expression in parentheses if it has additive-level precedence (Add, Sub).
///
/// This is used by product-like operators (dot product, cross product, outer product)
/// which have higher precedence than addition/subtraction.
pub(super) fn wrap_if_additive(expr: &Expression) -> String {
    match expr {
        Expression::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::PlusMinus | BinaryOp::MinusPlus,
            ..
        } => format!(r"\left({}\right)", expr.to_latex()),
        _ => expr.to_latex(),
    }
}

/// Trait for converting AST types to LaTeX notation.
///
/// This trait provides a method to convert mathematical expressions and
/// operators into valid LaTeX strings.
///
/// # Examples
///
/// ```
/// use mathlex::ast::{Expression, MathConstant};
/// use mathlex::latex::ToLatex;
///
/// let pi = Expression::Constant(MathConstant::Pi);
/// assert_eq!(pi.to_latex(), r"\pi");
/// ```
pub trait ToLatex {
    /// Converts the value to a LaTeX string.
    fn to_latex(&self) -> String;
}

/// Known trigonometric and mathematical functions that have LaTeX commands.
pub(super) const KNOWN_FUNCTIONS: &[&str] = &[
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "arccot", "arcsec",
    "arccsc", "sinh", "cosh", "tanh", "coth", "sech", "csch", "ln", "log", "exp", "lg", "det",
    "dim", "ker", "hom", "arg", "deg", "gcd", "lcm", "max", "min", "sup", "inf", "lim", "limsup",
    "liminf",
];
