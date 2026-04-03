//! Integral bound types for mathematical expressions.

use super::Expression;

/// Bounds for definite integrals.
///
/// Represents the lower and upper bounds of integration.
///
/// # Examples
///
/// ```
/// use mathlex::ast::{IntegralBounds, Expression};
///
/// // Integral from 0 to 1
/// let bounds = IntegralBounds {
///     lower: Box::new(Expression::Integer(0)),
///     upper: Box::new(Expression::Integer(1)),
/// };
///
/// match (*bounds.lower, *bounds.upper) {
///     (Expression::Integer(0), Expression::Integer(1)) => println!("Bounds are 0 to 1"),
///     _ => panic!("Unexpected bounds"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntegralBounds {
    /// Lower bound of integration
    pub lower: Box<Expression>,

    /// Upper bound of integration
    pub upper: Box<Expression>,
}

/// Bounds for multiple integrals (region specification).
///
/// Contains bounds for each variable of integration in order.
/// For a double integral ∬_R f dA, the region R may be specified
/// as separate bounds for each variable.
///
/// # Examples
///
/// ```ignore
/// // Bounds for ∬_[0,1]×[0,2] f(x,y) dy dx
/// MultipleBounds {
///     bounds: vec![
///         IntegralBounds { lower: 0, upper: 1 },  // x bounds
///         IntegralBounds { lower: 0, upper: 2 },  // y bounds
///     ],
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultipleBounds {
    /// Bounds for each variable, in integration order
    pub bounds: Vec<IntegralBounds>,
}
