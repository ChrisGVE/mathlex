//! # AST Utility Functions
//!
//! This module provides utility methods for querying and analyzing the AST.
//! These methods allow consumers to extract information about expressions
//! without manually traversing the tree structure.
//!
//! ## Key Features
//!
//! - **Variable Discovery**: Find all variables in an expression
//! - **Function Discovery**: Find all function calls in an expression
//! - **Constant Discovery**: Find all mathematical constants used
//! - **Tree Metrics**: Calculate depth and node count
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Create expression: 2 * π * x
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Binary {
//!         op: BinaryOp::Mul,
//!         left: Box::new(Expression::Integer(2)),
//!         right: Box::new(Expression::Constant(MathConstant::Pi)),
//!     }),
//!     right: Box::new(Expression::Variable("x".to_string())),
//! };
//!
//! // Query the expression
//! assert_eq!(expr.find_variables().len(), 1); // {x}
//! assert_eq!(expr.find_constants().len(), 1); // {π}
//! assert_eq!(expr.depth(), 3);
//! assert_eq!(expr.node_count(), 5);
//! ```

mod collect_consts;
mod collect_fns;
mod collect_vars;

mod collect;
mod metrics;
mod substitute;
mod validation;

#[cfg(test)]
mod tests;
