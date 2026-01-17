//! # mathlex
//!
//! A mathematical expression parser for LaTeX and plain text notation,
//! producing a language-agnostic Abstract Syntax Tree (AST).
//!
//! ## Overview
//!
//! mathlex is a pure parsing library that converts mathematical expressions
//! in LaTeX or plain text format into a well-defined AST. The library does
//! NOT perform any evaluation or mathematical operations - interpretation
//! of the AST is entirely the responsibility of consuming libraries.
//!
//! ## Features
//!
//! - **LaTeX Parsing**: Parse mathematical LaTeX notation
//! - **Plain Text Parsing**: Parse standard mathematical expressions
//! - **Rich AST**: Comprehensive AST for algebra, calculus, linear algebra
//! - **Utilities**: Variable extraction, substitution, string conversion
//!
//! ## Quick Start
//!
//! ```ignore
//! use mathlex::{parse, parse_latex};
//!
//! // Parse plain text
//! let expr = parse("2*x + sin(y)").unwrap();
//!
//! // Parse LaTeX
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//! ```
//!
//! ## Design Philosophy
//!
//! mathlex follows the principle of single responsibility:
//!
//! - **Parse** text into AST
//! - **Convert** AST back to text (plain or LaTeX)
//! - **Query** AST (find variables, functions)
//! - **Transform** AST (substitution)
//!
//! What mathlex does NOT do:
//!
//! - Evaluate expressions
//! - Simplify expressions
//! - Solve equations
//! - Perform any mathematical computation
//!
//! This design allows mathlex to serve as a shared foundation for both
//! symbolic computation systems (like thales) and numerical libraries
//! (like NumericSwift) without creating dependencies between them.

#![warn(missing_docs)]
#![warn(clippy::all)]

// Modules will be added as development progresses
pub mod ast;
// pub mod parser;
pub mod error;
// pub mod util;

// Re-export key types at crate root for convenience
pub use ast::{
    BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, MathConstant, UnaryOp,
};

/// Placeholder for library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
