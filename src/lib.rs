// Allow approx_constant - tests parse "3.14" which should give 3.14, not π
#![allow(clippy::approx_constant)]
// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]
// Allow unnecessary cast - swift-bridge macro generates these
#![allow(clippy::unnecessary_cast)]

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
//! ```
//! use mathlex::{parse, parse_latex, Expression, BinaryOp};
//!
//! // Parse plain text
//! let expr = parse("2*x + sin(y)").unwrap();
//!
//! // Parse LaTeX
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//!
//! // Verify the parsed structure
//! if let Expression::Binary { op: BinaryOp::Div, .. } = expr {
//!     println!("Parsed as division");
//! }
//! ```
//!
//! ## Configuration
//!
//! For advanced parsing options, use [`ParserConfig`]:
//!
//! ```
//! use mathlex::{parse_with_config, ParserConfig};
//!
//! let config = ParserConfig {
//!     implicit_multiplication: true,
//! };
//!
//! // Parse with custom configuration (config reserved for future use)
//! let expr = parse_with_config("2*x + 3", &config).unwrap();
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

use std::collections::HashSet;

// Modules will be added as development progresses
pub mod ast;
pub mod display;
pub mod error;
pub mod latex;
pub mod parser;
pub mod util;

#[cfg(feature = "ffi")]
pub mod ffi;

// Re-export key types at crate root for convenience
pub use ast::{
    BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, LogicalOp, MathConstant,
    UnaryOp, VectorNotation,
};
pub use error::{ParseError, ParseErrorKind, ParseResult, Position, Span};
pub use latex::ToLatex;

// Re-export parser functions
pub use parser::{parse, parse_latex};

/// Number system context for parsing.
///
/// Indicates the assumed number system for parsing expressions.
/// This can help disambiguate notation in context-dependent scenarios.
///
/// # Examples
///
/// ```
/// use mathlex::NumberSystem;
///
/// let real = NumberSystem::Real;
/// let complex = NumberSystem::Complex;
/// let quaternion = NumberSystem::Quaternion;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumberSystem {
    /// Real number system (default)
    #[default]
    Real,
    /// Complex number system
    Complex,
    /// Quaternion number system
    Quaternion,
}

/// Index convention for tensor notation.
///
/// Specifies how repeated indices should be interpreted in expressions.
///
/// # Examples
///
/// ```
/// use mathlex::IndexConvention;
///
/// let standard = IndexConvention::Standard;
/// let einstein = IndexConvention::EinsteinSummation;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexConvention {
    /// Standard index notation (default)
    #[default]
    Standard,
    /// Einstein summation convention (implied summation over repeated indices)
    EinsteinSummation,
}

/// Configuration options for the mathematical expression parser.
///
/// This struct controls various parsing behaviors including implicit
/// multiplication, number system context, and symbol declarations.
///
/// # Examples
///
/// ```
/// use mathlex::{ParserConfig, NumberSystem, IndexConvention};
/// use std::collections::HashSet;
///
/// // Use default configuration
/// let config = ParserConfig::default();
/// assert_eq!(config.implicit_multiplication, true);
///
/// // Create custom configuration with context hints
/// let mut declared_vectors = HashSet::new();
/// declared_vectors.insert("v".to_string());
/// declared_vectors.insert("u".to_string());
///
/// let config = ParserConfig {
///     implicit_multiplication: false,
///     number_system: NumberSystem::Complex,
///     index_convention: IndexConvention::EinsteinSummation,
///     declared_vectors,
///     declared_matrices: HashSet::new(),
///     declared_scalars: HashSet::new(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParserConfig {
    /// Enable implicit multiplication (e.g., `2x` means `2*x`).
    ///
    /// When `true`, expressions like `2x` or `(a)(b)` are interpreted
    /// as multiplication. When `false`, such expressions may result in
    /// parse errors.
    ///
    /// Default: `true`
    pub implicit_multiplication: bool,

    /// Number system context for parsing.
    ///
    /// Indicates the assumed number system which can help disambiguate
    /// notation in context-dependent scenarios.
    ///
    /// Default: `NumberSystem::Real`
    pub number_system: NumberSystem,

    /// Index convention for tensor notation.
    ///
    /// Specifies how repeated indices should be interpreted in expressions,
    /// particularly for Einstein summation convention.
    ///
    /// Default: `IndexConvention::Standard`
    pub index_convention: IndexConvention,

    /// Declared vector symbols.
    ///
    /// A set of symbol names that should be treated as vectors during parsing.
    /// This helps the parser disambiguate between scalar and vector quantities.
    ///
    /// Default: empty set
    pub declared_vectors: HashSet<String>,

    /// Declared matrix symbols.
    ///
    /// A set of symbol names that should be treated as matrices during parsing.
    /// This helps the parser disambiguate between scalar and matrix quantities.
    ///
    /// Default: empty set
    pub declared_matrices: HashSet<String>,

    /// Declared scalar symbols.
    ///
    /// A set of symbol names that should be explicitly treated as scalars.
    /// Useful when certain symbols might otherwise be ambiguous.
    ///
    /// Default: empty set
    pub declared_scalars: HashSet<String>,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            implicit_multiplication: true,
            number_system: NumberSystem::default(),
            index_convention: IndexConvention::default(),
            declared_vectors: HashSet::new(),
            declared_matrices: HashSet::new(),
            declared_scalars: HashSet::new(),
        }
    }
}

/// Parses a plain text mathematical expression with custom configuration.
///
/// This function allows parsing with custom configuration options.
///
/// # Arguments
///
/// * `input` - The mathematical expression string to parse
/// * `config` - Parser configuration options
///
/// # Returns
///
/// A `ParseResult<Expression>` containing the parsed AST or an error.
///
/// # Examples
///
/// ```
/// use mathlex::{parse_with_config, ParserConfig};
///
/// let config = ParserConfig::default();
/// let expr = parse_with_config("sin(x) + 2", &config).unwrap();
/// ```
///
/// ```
/// use mathlex::{parse_with_config, ParserConfig, Expression, BinaryOp};
///
/// let config = ParserConfig {
///     implicit_multiplication: true,
/// };
///
/// // Parse expression with implicit multiplication
/// let expr = parse_with_config("2x", &config).unwrap();
/// match expr {
///     Expression::Binary { op: BinaryOp::Mul, .. } => println!("Parsed as multiplication"),
///     _ => panic!("Unexpected expression type"),
/// }
/// ```
pub fn parse_with_config(input: &str, config: &ParserConfig) -> ParseResult<Expression> {
    parser::parse_with_config(input, config)
}

/// Placeholder for library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.1");
    }

    #[test]
    fn test_parser_config_default() {
        let config = ParserConfig::default();
        assert!(config.implicit_multiplication);
    }

    #[test]
    fn test_parser_config_custom() {
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        assert!(!config.implicit_multiplication);
    }

    #[test]
    fn test_parser_config_equality() {
        let config1 = ParserConfig::default();
        let config2 = ParserConfig {
            implicit_multiplication: true,
            ..ParserConfig::default()
        };
        let config3 = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };

        assert_eq!(config1, config2);
        assert_ne!(config1, config3);
    }

    #[test]
    fn test_parser_config_clone() {
        let config = ParserConfig::default();
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    #[test]
    fn test_parse_simple() {
        let expr = parse("2 + 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_latex_simple() {
        let expr = parse_latex(r"\frac{1}{2}").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_with_config_default() {
        let config = ParserConfig::default();
        let expr = parse_with_config("sin(x) + 2", &config).unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_with_config_custom() {
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        let expr = parse_with_config("2 + 3", &config).unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_all_type_exports() {
        // Ensure all required types are exported
        let _expr: Expression = Expression::Integer(42);
        let _op: BinaryOp = BinaryOp::Add;
        let _unary: UnaryOp = UnaryOp::Neg;
        let _const: MathConstant = MathConstant::Pi;
        let _dir: Direction = Direction::Both;
        let _ineq: InequalityOp = InequalityOp::Lt;
        let _bounds: IntegralBounds = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };
        let _pos: Position = Position::start();
        let _span: Span = Span::start();
        let _err: ParseError = ParseError::empty_expression(None);
    }
}
