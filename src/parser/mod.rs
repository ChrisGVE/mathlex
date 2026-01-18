//! Parser module for mathlex.
//!
//! This module provides tokenization and parsing capabilities for both
//! plain text and LaTeX mathematical expressions.
//!
//! # Modules
//!
//! - [`latex_tokenizer`]: LaTeX tokenizer that converts LaTeX input into tokens
//! - [`text`]: Plain text expression parser (to be implemented)
//! - [`latex`]: LaTeX expression parser (to be implemented)

pub mod latex_tokenizer;
// pub mod text;
// pub mod latex;

// Re-export key types
pub use latex_tokenizer::{tokenize_latex, LatexToken};

/// A spanned token - combines a token with its source location.
///
/// This type alias is used throughout the parser to track where tokens
/// originated in the source text for error reporting.
pub type Spanned<T> = (T, crate::error::Span);
