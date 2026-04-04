//! Tokenizer for plain text mathematical expressions.
//!
//! This module provides tokenization of plain text mathematical notation into
//! a stream of tokens with precise span information for error reporting.
//!
//! # Example
//!
//! ```
//! use mathlex::parser::tokenize;
//!
//! let tokens = tokenize("2 + x").unwrap();
//! assert_eq!(tokens.len(), 3);
//! ```

mod scanner;
mod tests;
mod token_types;

pub use scanner::tokenize;
pub use token_types::{SpannedToken, Token};
