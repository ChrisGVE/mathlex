//! Display implementations for error types.

use std::fmt;

use super::types::{ParseError, ParseErrorKind};

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnexpectedToken { expected, found } => {
                if expected.is_empty() {
                    write!(f, "unexpected token '{}'", found)
                } else if expected.len() == 1 {
                    write!(f, "unexpected token '{}', expected {}", found, expected[0])
                } else {
                    write!(
                        f,
                        "unexpected token '{}', expected one of: {}",
                        found,
                        expected.join(", ")
                    )
                }
            }
            ParseErrorKind::UnexpectedEof { expected } => {
                if expected.is_empty() {
                    write!(f, "unexpected end of input")
                } else if expected.len() == 1 {
                    write!(f, "unexpected end of input, expected {}", expected[0])
                } else {
                    write!(
                        f,
                        "unexpected end of input, expected one of: {}",
                        expected.join(", ")
                    )
                }
            }
            ParseErrorKind::UnmatchedDelimiter { opening, position } => {
                write!(
                    f,
                    "unmatched opening delimiter '{}' at {}",
                    opening, position
                )
            }
            ParseErrorKind::InvalidNumber { value, reason } => {
                write!(f, "invalid number '{}': {}", value, reason)
            }
            ParseErrorKind::InvalidLatexCommand { command } => {
                write!(f, "invalid LaTeX command '{}'", command)
            }
            ParseErrorKind::UnknownFunction { name } => {
                write!(f, "unknown function '{}'", name)
            }
            ParseErrorKind::InvalidSubscript { reason } => {
                write!(f, "invalid subscript: {}", reason)
            }
            ParseErrorKind::InvalidSuperscript { reason } => {
                write!(f, "invalid superscript: {}", reason)
            }
            ParseErrorKind::MalformedMatrix { reason } => {
                write!(f, "malformed matrix: {}", reason)
            }
            ParseErrorKind::EmptyExpression => {
                write!(f, "empty expression")
            }
            ParseErrorKind::Custom(msg) => {
                write!(f, "{}", msg)
            }
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.span {
            Some(span) => write!(f, "{} at {}", self.kind, span)?,
            None => write!(f, "{}", self.kind)?,
        }

        if let Some(ctx) = &self.context {
            write!(f, " ({})", ctx)?;
        }

        if let Some(suggestion) = &self.suggestion {
            write!(f, " {}", suggestion)?;
        }

        Ok(())
    }
}

impl std::error::Error for ParseError {}
