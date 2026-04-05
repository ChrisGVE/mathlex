//! # Display Trait Implementations for AST (Plain Text)
//!
//! This module provides `std::fmt::Display` implementations for all AST types,
//! converting them back to plain text mathematical notation.
//!
//! ## Design Philosophy
//!
//! - **Minimal Parentheses**: Use operator precedence to minimize parenthesization
//! - **Readable Output**: Generate human-readable plain text
//! - **Round-trip Capable**: Output can be parsed back (though not guaranteed identical AST)
//!
//! ## Precedence Levels
//!
//! 1. Addition, Subtraction: precedence = 1
//! 2. Multiplication, Division, Modulo: precedence = 2
//! 3. Exponentiation: precedence = 3
//! 4. Unary operations: precedence = 4 (implicit, highest)
//!
//! ## Examples
//!
//! ```ignore
//! use mathlex::ast::{Expression, BinaryOp};
//!
//! // 2 + 3 * 4 → "2 + 3 * 4" (no parens needed)
//! let expr = Expression::Binary {
//!     op: BinaryOp::Add,
//!     left: Box::new(Expression::Integer(2)),
//!     right: Box::new(Expression::Binary {
//!         op: BinaryOp::Mul,
//!         left: Box::new(Expression::Integer(3)),
//!         right: Box::new(Expression::Integer(4)),
//!     }),
//! };
//! assert_eq!(format!("{}", expr), "2 + 3 * 4");
//! ```

pub(crate) mod helpers;

mod expression;
mod types;

#[cfg(test)]
mod tests;
