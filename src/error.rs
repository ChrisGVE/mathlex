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

use std::fmt;

/// A position in the source text.
///
/// Tracks line number (1-indexed), column number (1-indexed), and byte offset
/// (0-indexed) for precise error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Byte offset from start of input (0-indexed)
    pub offset: usize,
}

impl Position {
    /// Creates a new position.
    ///
    /// # Arguments
    ///
    /// * `line` - Line number (1-indexed)
    /// * `column` - Column number (1-indexed)
    /// * `offset` - Byte offset (0-indexed)
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::Position;
    ///
    /// let pos = Position::new(1, 1, 0);
    /// assert_eq!(pos.line, 1);
    /// assert_eq!(pos.column, 1);
    /// assert_eq!(pos.offset, 0);
    /// ```
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Creates a position at the start of input.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::Position;
    ///
    /// let pos = Position::start();
    /// assert_eq!(pos.line, 1);
    /// assert_eq!(pos.column, 1);
    /// assert_eq!(pos.offset, 0);
    /// ```
    pub fn start() -> Self {
        Self::new(1, 1, 0)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A span representing a range in the source text.
///
/// Contains start and end positions to identify the exact location of a token
/// or error in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    /// Start position of the span
    pub start: Position,
    /// End position of the span
    pub end: Position,
}

impl Span {
    /// Creates a new span.
    ///
    /// # Arguments
    ///
    /// * `start` - Start position
    /// * `end` - End position
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{Position, Span};
    ///
    /// let start = Position::new(1, 1, 0);
    /// let end = Position::new(1, 5, 4);
    /// let span = Span::new(start, end);
    /// assert_eq!(span.to_string(), "1:1-1:5");
    /// ```
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }

    /// Creates a span at a single position.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{Position, Span};
    ///
    /// let pos = Position::new(1, 5, 4);
    /// let span = Span::at(pos);
    /// assert_eq!(span.start, span.end);
    /// ```
    pub fn at(pos: Position) -> Self {
        Self::new(pos, pos)
    }

    /// Creates a span from the start of input.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{Position, Span};
    ///
    /// let span = Span::start();
    /// assert_eq!(span.start, Position::start());
    /// ```
    pub fn start() -> Self {
        Self::at(Position::start())
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

/// The kind of parsing error that occurred.
///
/// Each variant provides specific context about what went wrong during parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// An unexpected token was encountered.
    UnexpectedToken {
        /// Tokens that were expected at this position
        expected: Vec<String>,
        /// The token that was actually found
        found: String,
    },

    /// Unexpected end of input.
    UnexpectedEof {
        /// Tokens that were expected before end of input
        expected: Vec<String>,
    },

    /// An opening delimiter was never closed.
    UnmatchedDelimiter {
        /// The opening delimiter character
        opening: char,
        /// Position where the opening delimiter was found
        position: Position,
    },

    /// A number could not be parsed.
    InvalidNumber {
        /// The string that failed to parse as a number
        value: String,
        /// Reason why the number is invalid
        reason: String,
    },

    /// An unknown or malformed LaTeX command was encountered.
    InvalidLatexCommand {
        /// The invalid command
        command: String,
    },

    /// An unknown function name was encountered.
    UnknownFunction {
        /// The unknown function name
        name: String,
    },

    /// A subscript expression is malformed.
    InvalidSubscript {
        /// Reason why the subscript is invalid
        reason: String,
    },

    /// A superscript expression is malformed.
    InvalidSuperscript {
        /// Reason why the superscript is invalid
        reason: String,
    },

    /// A matrix expression is malformed.
    MalformedMatrix {
        /// Reason why the matrix is malformed
        reason: String,
    },

    /// An empty expression was encountered where one was required.
    EmptyExpression,

    /// A custom error with a free-form message.
    Custom(String),
}

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

/// A parsing error with location and context information.
///
/// This is the main error type returned by parsing operations. It combines
/// a [`ParseErrorKind`] with optional location and context information.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// The specific kind of error
    pub kind: ParseErrorKind,
    /// The location in the source where the error occurred
    pub span: Option<Span>,
    /// Additional context about the error
    pub context: Option<String>,
}

impl ParseError {
    /// Creates a new parse error.
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of parsing error
    /// * `span` - Optional span indicating where the error occurred
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{ParseError, ParseErrorKind};
    ///
    /// let error = ParseError::new(
    ///     ParseErrorKind::EmptyExpression,
    ///     None,
    /// );
    /// ```
    pub fn new(kind: ParseErrorKind, span: Option<Span>) -> Self {
        Self {
            kind,
            span,
            context: None,
        }
    }

    /// Adds context to this error.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context information
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{ParseError, ParseErrorKind};
    ///
    /// let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
    ///     .with_context("while parsing function arguments");
    /// ```
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Creates an unexpected token error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::unexpected_token(
    ///     vec!["number"],
    ///     "+",
    ///     None,
    /// );
    /// ```
    pub fn unexpected_token<S1, S2>(
        expected: Vec<S1>,
        found: S2,
        span: Option<Span>,
    ) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
    {
        Self::new(
            ParseErrorKind::UnexpectedToken {
                expected: expected.into_iter().map(|s| s.into()).collect(),
                found: found.into(),
            },
            span,
        )
    }

    /// Creates an unexpected end of input error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::unexpected_eof(
    ///     vec!["closing parenthesis"],
    ///     None,
    /// );
    /// ```
    pub fn unexpected_eof<S>(expected: Vec<S>, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::UnexpectedEof {
                expected: expected.into_iter().map(|s| s.into()).collect(),
            },
            span,
        )
    }

    /// Creates an unmatched delimiter error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{Position, ParseError};
    ///
    /// let error = ParseError::unmatched_delimiter(
    ///     '(',
    ///     Position::new(1, 1, 0),
    ///     None,
    /// );
    /// ```
    pub fn unmatched_delimiter(
        opening: char,
        position: Position,
        span: Option<Span>,
    ) -> Self {
        Self::new(
            ParseErrorKind::UnmatchedDelimiter { opening, position },
            span,
        )
    }

    /// Creates an invalid number error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::invalid_number(
    ///     "123.45.67",
    ///     "multiple decimal points",
    ///     None,
    /// );
    /// ```
    pub fn invalid_number<S1, S2>(value: S1, reason: S2, span: Option<Span>) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
    {
        Self::new(
            ParseErrorKind::InvalidNumber {
                value: value.into(),
                reason: reason.into(),
            },
            span,
        )
    }

    /// Creates an invalid LaTeX command error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::invalid_latex_command(r"\unknowncommand", None);
    /// ```
    pub fn invalid_latex_command<S>(command: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::InvalidLatexCommand {
                command: command.into(),
            },
            span,
        )
    }

    /// Creates an unknown function error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::unknown_function("unknownfunc", None);
    /// ```
    pub fn unknown_function<S>(name: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::UnknownFunction { name: name.into() },
            span,
        )
    }

    /// Creates an invalid subscript error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::invalid_subscript("missing expression", None);
    /// ```
    pub fn invalid_subscript<S>(reason: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::InvalidSubscript {
                reason: reason.into(),
            },
            span,
        )
    }

    /// Creates an invalid superscript error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::invalid_superscript("missing expression", None);
    /// ```
    pub fn invalid_superscript<S>(reason: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::InvalidSuperscript {
                reason: reason.into(),
            },
            span,
        )
    }

    /// Creates a malformed matrix error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::malformed_matrix("inconsistent row lengths", None);
    /// ```
    pub fn malformed_matrix<S>(reason: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(
            ParseErrorKind::MalformedMatrix {
                reason: reason.into(),
            },
            span,
        )
    }

    /// Creates an empty expression error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::empty_expression(None);
    /// ```
    pub fn empty_expression(span: Option<Span>) -> Self {
        Self::new(ParseErrorKind::EmptyExpression, span)
    }

    /// Creates a custom error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::ParseError;
    ///
    /// let error = ParseError::custom("something went wrong", None);
    /// ```
    pub fn custom<S>(message: S, span: Option<Span>) -> Self
    where
        S: Into<String>,
    {
        Self::new(ParseErrorKind::Custom(message.into()), span)
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

        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// Result type for parsing operations.
///
/// This is a convenience alias for `Result<T, ParseError>`.
pub type ParseResult<T> = Result<T, ParseError>;

/// Builder for constructing parse errors ergonomically.
///
/// # Example
///
/// ```
/// use mathlex::error::{ErrorBuilder, Position, ParseErrorKind};
///
/// let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
///     .at_position(Position::new(1, 1, 0))
///     .with_context("in function body")
///     .build();
/// ```
#[derive(Debug)]
pub struct ErrorBuilder {
    kind: ParseErrorKind,
    span: Option<Span>,
    context: Option<String>,
}

impl ErrorBuilder {
    /// Creates a new error builder.
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of parsing error
    pub fn new(kind: ParseErrorKind) -> Self {
        Self {
            kind,
            span: None,
            context: None,
        }
    }

    /// Sets the span for this error.
    ///
    /// # Arguments
    ///
    /// * `span` - The span where the error occurred
    pub fn at_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Sets the span to a single position.
    ///
    /// # Arguments
    ///
    /// * `position` - The position where the error occurred
    pub fn at_position(mut self, position: Position) -> Self {
        self.span = Some(Span::at(position));
        self
    }

    /// Adds context to this error.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context information
    pub fn with_context<S: Into<String>>(mut self, context: S) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Builds the parse error.
    pub fn build(self) -> ParseError {
        ParseError {
            kind: self.kind,
            span: self.span,
            context: self.context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_new() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.line, 5);
        assert_eq!(pos.column, 10);
        assert_eq!(pos.offset, 42);
    }

    #[test]
    fn test_position_start() {
        let pos = Position::start();
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);
    }

    #[test]
    fn test_position_display() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.to_string(), "5:10");
    }

    #[test]
    fn test_position_equality() {
        let pos1 = Position::new(1, 1, 0);
        let pos2 = Position::new(1, 1, 0);
        let pos3 = Position::new(1, 2, 1);

        assert_eq!(pos1, pos2);
        assert_ne!(pos1, pos3);
    }

    #[test]
    fn test_span_new() {
        let start = Position::new(1, 1, 0);
        let end = Position::new(1, 5, 4);
        let span = Span::new(start, end);

        assert_eq!(span.start, start);
        assert_eq!(span.end, end);
    }

    #[test]
    fn test_span_at() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);

        assert_eq!(span.start, pos);
        assert_eq!(span.end, pos);
    }

    #[test]
    fn test_span_start() {
        let span = Span::start();
        assert_eq!(span.start, Position::start());
        assert_eq!(span.end, Position::start());
    }

    #[test]
    fn test_span_display_single_position() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        assert_eq!(span.to_string(), "1:5");
    }

    #[test]
    fn test_span_display_range() {
        let start = Position::new(1, 1, 0);
        let end = Position::new(1, 5, 4);
        let span = Span::new(start, end);
        assert_eq!(span.to_string(), "1:1-1:5");
    }

    #[test]
    fn test_parse_error_kind_unexpected_token() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec!["number".to_string()],
            found: "+".to_string(),
        };
        assert_eq!(kind.to_string(), "unexpected token '+', expected number");
    }

    #[test]
    fn test_parse_error_kind_unexpected_token_multiple_expected() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec!["number".to_string(), "variable".to_string()],
            found: "+".to_string(),
        };
        assert_eq!(
            kind.to_string(),
            "unexpected token '+', expected one of: number, variable"
        );
    }

    #[test]
    fn test_parse_error_kind_unexpected_token_no_expected() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec![],
            found: "+".to_string(),
        };
        assert_eq!(kind.to_string(), "unexpected token '+'");
    }

    #[test]
    fn test_parse_error_kind_unexpected_eof() {
        let kind = ParseErrorKind::UnexpectedEof {
            expected: vec!["number".to_string()],
        };
        assert_eq!(kind.to_string(), "unexpected end of input, expected number");
    }

    #[test]
    fn test_parse_error_kind_unexpected_eof_multiple_expected() {
        let kind = ParseErrorKind::UnexpectedEof {
            expected: vec!["number".to_string(), "variable".to_string()],
        };
        assert_eq!(
            kind.to_string(),
            "unexpected end of input, expected one of: number, variable"
        );
    }

    #[test]
    fn test_parse_error_kind_unmatched_delimiter() {
        let pos = Position::new(1, 5, 4);
        let kind = ParseErrorKind::UnmatchedDelimiter {
            opening: '(',
            position: pos,
        };
        assert_eq!(kind.to_string(), "unmatched opening delimiter '(' at 1:5");
    }

    #[test]
    fn test_parse_error_kind_invalid_number() {
        let kind = ParseErrorKind::InvalidNumber {
            value: "123.45.67".to_string(),
            reason: "multiple decimal points".to_string(),
        };
        assert_eq!(
            kind.to_string(),
            "invalid number '123.45.67': multiple decimal points"
        );
    }

    #[test]
    fn test_parse_error_kind_invalid_latex_command() {
        let kind = ParseErrorKind::InvalidLatexCommand {
            command: r"\unknowncommand".to_string(),
        };
        assert_eq!(kind.to_string(), r"invalid LaTeX command '\unknowncommand'");
    }

    #[test]
    fn test_parse_error_kind_unknown_function() {
        let kind = ParseErrorKind::UnknownFunction {
            name: "unknownfunc".to_string(),
        };
        assert_eq!(kind.to_string(), "unknown function 'unknownfunc'");
    }

    #[test]
    fn test_parse_error_kind_invalid_subscript() {
        let kind = ParseErrorKind::InvalidSubscript {
            reason: "missing expression".to_string(),
        };
        assert_eq!(kind.to_string(), "invalid subscript: missing expression");
    }

    #[test]
    fn test_parse_error_kind_invalid_superscript() {
        let kind = ParseErrorKind::InvalidSuperscript {
            reason: "missing expression".to_string(),
        };
        assert_eq!(kind.to_string(), "invalid superscript: missing expression");
    }

    #[test]
    fn test_parse_error_kind_malformed_matrix() {
        let kind = ParseErrorKind::MalformedMatrix {
            reason: "inconsistent row lengths".to_string(),
        };
        assert_eq!(kind.to_string(), "malformed matrix: inconsistent row lengths");
    }

    #[test]
    fn test_parse_error_kind_empty_expression() {
        let kind = ParseErrorKind::EmptyExpression;
        assert_eq!(kind.to_string(), "empty expression");
    }

    #[test]
    fn test_parse_error_kind_custom() {
        let kind = ParseErrorKind::Custom("custom error message".to_string());
        assert_eq!(kind.to_string(), "custom error message");
    }

    #[test]
    fn test_parse_error_new() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None);
        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, None);
        assert_eq!(error.context, None);
    }

    #[test]
    fn test_parse_error_with_context() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_context("while parsing function arguments");

        assert_eq!(error.context, Some("while parsing function arguments".to_string()));
    }

    #[test]
    fn test_parse_error_display_no_span() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None);
        assert_eq!(error.to_string(), "empty expression");
    }

    #[test]
    fn test_parse_error_display_with_span() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        let error = ParseError::new(ParseErrorKind::EmptyExpression, Some(span));
        assert_eq!(error.to_string(), "empty expression at 1:5");
    }

    #[test]
    fn test_parse_error_display_with_context() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_context("while parsing function arguments");
        assert_eq!(
            error.to_string(),
            "empty expression (while parsing function arguments)"
        );
    }

    #[test]
    fn test_parse_error_display_with_span_and_context() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        let error = ParseError::new(ParseErrorKind::EmptyExpression, Some(span))
            .with_context("while parsing function arguments");
        assert_eq!(
            error.to_string(),
            "empty expression at 1:5 (while parsing function arguments)"
        );
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        let error = ParseError::unexpected_token(
            vec!["number"],
            "+",
            None,
        );

        assert_eq!(
            error.kind,
            ParseErrorKind::UnexpectedToken {
                expected: vec!["number".to_string()],
                found: "+".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let error = ParseError::unexpected_eof(
            vec!["closing parenthesis"],
            None,
        );

        assert_eq!(
            error.kind,
            ParseErrorKind::UnexpectedEof {
                expected: vec!["closing parenthesis".to_string()],
            }
        );
    }

    #[test]
    fn test_parse_error_unmatched_delimiter() {
        let pos = Position::new(1, 1, 0);
        let error = ParseError::unmatched_delimiter('(', pos, None);

        assert_eq!(
            error.kind,
            ParseErrorKind::UnmatchedDelimiter {
                opening: '(',
                position: pos,
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_number() {
        let error = ParseError::invalid_number(
            "123.45.67",
            "multiple decimal points",
            None,
        );

        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidNumber {
                value: "123.45.67".to_string(),
                reason: "multiple decimal points".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_latex_command() {
        let error = ParseError::invalid_latex_command(r"\unknowncommand", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidLatexCommand {
                command: r"\unknowncommand".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_unknown_function() {
        let error = ParseError::unknown_function("unknownfunc", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::UnknownFunction {
                name: "unknownfunc".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_subscript() {
        let error = ParseError::invalid_subscript("missing expression", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidSubscript {
                reason: "missing expression".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_superscript() {
        let error = ParseError::invalid_superscript("missing expression", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidSuperscript {
                reason: "missing expression".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_malformed_matrix() {
        let error = ParseError::malformed_matrix("inconsistent row lengths", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::MalformedMatrix {
                reason: "inconsistent row lengths".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_empty_expression() {
        let error = ParseError::empty_expression(None);
        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
    }

    #[test]
    fn test_parse_error_custom() {
        let error = ParseError::custom("custom error message", None);

        assert_eq!(
            error.kind,
            ParseErrorKind::Custom("custom error message".to_string())
        );
    }

    #[test]
    fn test_error_builder_basic() {
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression).build();

        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, None);
        assert_eq!(error.context, None);
    }

    #[test]
    fn test_error_builder_with_span() {
        let span = Span::at(Position::new(1, 5, 4));
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_span(span)
            .build();

        assert_eq!(error.span, Some(span));
    }

    #[test]
    fn test_error_builder_with_position() {
        let pos = Position::new(1, 5, 4);
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_position(pos)
            .build();

        assert_eq!(error.span, Some(Span::at(pos)));
    }

    #[test]
    fn test_error_builder_with_context() {
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .with_context("in function body")
            .build();

        assert_eq!(error.context, Some("in function body".to_string()));
    }

    #[test]
    fn test_error_builder_complete() {
        let pos = Position::new(1, 5, 4);
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_position(pos)
            .with_context("in function body")
            .build();

        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, Some(Span::at(pos)));
        assert_eq!(error.context, Some("in function body".to_string()));
    }

    #[test]
    fn test_parse_result_ok() {
        let result: ParseResult<i32> = Ok(42);
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_parse_result_err() {
        let result: ParseResult<i32> = Err(ParseError::empty_expression(None));
        assert!(result.is_err());
    }
}
