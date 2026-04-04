//! # LaTeX Conversion for AST
//!
//! This module provides the `ToLatex` trait for converting AST types back to LaTeX notation.
//!
//! ## Design Philosophy
//!
//! - **Standard LaTeX**: Generate valid LaTeX mathematical notation
//! - **Readability**: Produce clean, readable LaTeX code
//! - **Round-trip Capable**: Output can be parsed back (though not guaranteed identical AST)
//!
//! ## LaTeX Mappings
//!
//! - Integers/Floats: Direct string representation
//! - Rational: `\frac{numerator}{denominator}`
//! - Binary Division: `\frac{left}{right}`
//! - Functions: `\sin`, `\cos`, etc. for known functions, `\operatorname{name}` for others
//! - Square Root: `\sqrt{x}` or `\sqrt[n]{x}`
//! - Derivatives: `\frac{d}{dx}` or `\frac{d^n}{dx^n}`
//! - Partial Derivatives: `\frac{\partial}{\partial x}`
//! - Integrals: `\int_{lower}^{upper} expr dx`
//! - Limits: `\lim_{x \to value^{direction}}`
//! - Sum/Product: `\sum_{i=lower}^{upper}`, `\prod_{i=lower}^{upper}`
//! - Vectors/Matrices: `\begin{pmatrix}...\end{pmatrix}`
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::Expression;
//! use mathlex::latex::ToLatex;
//!
//! // 1/2 → "\frac{1}{2}"
//! let expr = Expression::Rational {
//!     numerator: Box::new(Expression::Integer(1)),
//!     denominator: Box::new(Expression::Integer(2)),
//! };
//! assert_eq!(expr.to_latex(), r"\frac{1}{2}");
//! ```

mod expression;
mod helpers;
mod helpers_linalg;
mod impl_types;
mod trait_def;

pub use trait_def::ToLatex;

#[cfg(test)]
mod tests;
