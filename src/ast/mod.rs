//! # Abstract Syntax Tree (AST) Types
//!
//! This module defines the core AST types used to represent mathematical expressions.
//! The AST is the contract between mathlex parsers and consuming libraries.
//!
//! ## Design Philosophy
//!
//! - **Format Agnostic**: The same mathematical concept produces the same AST regardless
//!   of input format (LaTeX or plain text)
//! - **Structural Representation**: AST nodes represent syntax, not evaluated values
//! - **Complete Coverage**: Supports algebra, calculus, linear algebra, and equations
//!
//! ## Key Types
//!
//! - [`Expression`]: The main AST node type representing any mathematical expression
//! - [`MathConstant`]: Mathematical constants (π, e, i, ∞)
//! - [`BinaryOp`]: Binary operators (+, -, *, /, ^, %)
//! - [`UnaryOp`]: Unary operators (negation, factorial, transpose)
//! - [`MathFloat`]: Wrapper for f64 with proper equality and hashing semantics
//!
//! ## AST Semantics and Conventions
//!
//! ### Expression Types
//!
//! - **`Rational`**: Contains `Expression` fields (not `i64`), allowing symbolic rationals
//!   like `x/y`. This enables representation of unevaluated rational expressions where
//!   numerator and denominator are arbitrary expressions.
//! - **`Complex`**: Contains `Expression` fields for real and imaginary parts, enabling
//!   symbolic complex numbers like `(a+b)+(c+d)i` rather than just numeric values.
//! - **`MathFloat`**: Wraps `OrderedFloat<f64>` to provide proper `Hash` and `Eq` implementations.
//!   NaN values are comparable (NaN == NaN), which differs from standard IEEE 754 semantics
//!   but is necessary for use in hash-based collections.
//!
//! ### Known Limitations
//!
//! - **`Rational` and `Complex` variants are not produced by parsers** in the current implementation.
//!   These variants are available for programmatic construction by consumers of the AST,
//!   allowing symbolic manipulation libraries to build complex expressions.
//! - **Some ASTs don't round-trip perfectly** due to precedence and formatting choices.
//!   For example, `(2 + 3) * 4` and `2 + 3 * 4` have different ASTs but the first may display
//!   without parentheses depending on context.
//! - **`MathConstant::NegInfinity` is produced by parsers** when unary minus is applied to
//!   infinity. Both `-∞` (plain text) and `-\infty` (LaTeX) parse directly as
//!   `Constant(NegInfinity)`.
//!
//! ### Serialization Notes
//!
//! - **`Display` trait**: Uses minimal parentheses based on operator precedence. The output
//!   is human-readable but may omit parentheses where they can be inferred from precedence rules.
//! - **`ToLatex` trait**: Produces standard LaTeX notation that can be re-parsed by the LaTeX
//!   parser. This is the recommended format for round-trip serialization.
//! - **Special float values**: When using JSON serialization (via serde), NaN and Infinity
//!   values serialize to `null` per JSON specification. For lossless serialization of special
//!   floats, use binary formats like bincode.
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Representing: 2 * π
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Integer(2)),
//!     right: Box::new(Expression::Constant(MathConstant::Pi)),
//! };
//!
//! // Verify structure
//! match expr {
//!     Expression::Binary { op: BinaryOp::Mul, .. } => println!("It's multiplication!"),
//!     _ => panic!("Unexpected expression type"),
//! }
//! ```

mod bounds;
mod constants;
mod expression;
mod linear_algebra;
mod operators;
mod sets;

pub use bounds::{IntegralBounds, MultipleBounds};
pub use constants::{MathConstant, MathFloat};
pub use expression::Expression;
pub use linear_algebra::{IndexType, TensorIndex, VectorNotation};
pub use operators::{BinaryOp, Direction, InequalityOp, LogicalOp, RelationOp, UnaryOp};
pub use sets::{NumberSet, SetOp, SetRelation};

#[cfg(test)]
mod tests;
