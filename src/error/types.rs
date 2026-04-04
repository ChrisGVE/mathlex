//! Core type definitions: Position, Span, ParseErrorKind, ParseError.

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
    /// Helpful suggestion for fixing the error
    pub suggestion: Option<String>,
}

/// Result type for parsing operations.
///
/// This is a convenience alias for `Result<T, ParseError>`.
pub type ParseResult<T> = Result<T, ParseError>;
