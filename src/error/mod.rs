//! Error types for mathlex parsing operations.
//!
//! This module defines comprehensive error types for reporting parsing failures,
//! including precise location information and helpful error messages.
//!
//! # Error Architecture
//!
//! - [`Position`]: A point in source text (line, column, offset)
//! - [`Span`]: A range in source text (start and end positions)
//! - [`ParseErrorKind`]: The specific type of parsing error
//! - [`ParseError`]: Complete error with kind, location, and context
//! - [`ParseResult<T>`]: Standard Result type for parsing operations
//!
//! # Example
//!
//! ```
//! use mathlex::error::{Position, Span, ParseError, ParseErrorKind};
//!
//! let pos = Position::new(1, 5, 5);
//! let error = ParseError::new(
//!     ParseErrorKind::UnexpectedEof {
//!         expected: vec!["number".to_string()],
//!     },
//!     Some(Span::new(pos, pos)),
//! );
//!
//! assert_eq!(error.to_string(), "unexpected end of input, expected number at 1:5");
//! ```

mod builder;
mod constructors;
mod display;
mod output;
mod suggestions;
mod tests;
mod types;

pub(crate) use builder::ErrorBuilder;
pub use output::ParseOutput;
pub(crate) use suggestions::levenshtein;
pub use suggestions::suggest_function;
pub use types::{ParseError, ParseErrorKind, ParseResult, Position, Span};
