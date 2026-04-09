use crate::ast::precedence::needs_parens;
use crate::ast::{BinaryOp, Expression};

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
