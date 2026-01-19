//! Parser module for mathlex.
//!
//! This module provides tokenization and parsing capabilities for both
//! plain text and LaTeX mathematical expressions.
//!
//! # Modules
//!
//! - [`tokenizer`]: Plain text tokenizer that converts input into tokens
//! - [`latex_tokenizer`]: LaTeX tokenizer that converts LaTeX input into tokens
//! - [`text`]: Plain text expression parser
//! - [`latex`]: LaTeX expression parser

pub mod latex;
pub mod latex_tokenizer;
pub mod text;
pub mod tokenizer;

// Re-export key types
pub use latex::parse_latex;
pub use latex_tokenizer::{tokenize_latex, LatexToken};
pub use text::{parse, parse_with_config};
pub use tokenizer::tokenize;

/// A spanned token - combines a token with its source location.
///
/// This type alias is used throughout the parser to track where tokens
/// originated in the source text for error reporting.
pub type Spanned<T> = (T, crate::error::Span);
