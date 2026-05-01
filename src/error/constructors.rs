//! Constructor methods for ParseError.

use super::suggestions::suggest_function;
use super::types::{ParseError, ParseErrorKind, Span};

/// Generates a simple error constructor that wraps a single `Into<String>` field.
macro_rules! simple_error_ctor {
    ($name:ident, $kind:ident, $field:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<S: Into<String>>($field: S, span: Option<Span>) -> Self {
            Self::new(
                ParseErrorKind::$kind {
                    $field: $field.into(),
                },
                span,
            )
        }
    };
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
            suggestion: None,
        }
    }

    /// Adds context to this error.
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

    /// Adds a suggestion to this error.
    ///
    /// # Example
    ///
    /// ```
    /// use mathlex::error::{ParseError, ParseErrorKind};
    ///
    /// let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
    ///     .with_suggestion("Did you mean 'sin'?");
    /// ```
    pub fn with_suggestion<S: Into<String>>(mut self, suggestion: S) -> Self {
        self.suggestion = Some(suggestion.into());
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
    pub fn unexpected_token<S1, S2>(expected: Vec<S1>, found: S2, span: Option<Span>) -> Self
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
        position: super::types::Position,
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

    /// Creates an unknown function error.
    ///
    /// Automatically adds a suggestion if a similar known function is found.
    pub fn unknown_function<S: Into<String>>(name: S, span: Option<Span>) -> Self {
        let name_str = name.into();
        let suggestion = suggest_function(&name_str);
        let mut error = Self::new(ParseErrorKind::UnknownFunction { name: name_str }, span);
        error.suggestion = suggestion;
        error
    }

    simple_error_ctor!(
        invalid_latex_command,
        InvalidLatexCommand,
        command,
        "Creates an invalid LaTeX command error."
    );
    simple_error_ctor!(
        invalid_subscript,
        InvalidSubscript,
        reason,
        "Creates an invalid subscript error."
    );
    simple_error_ctor!(
        invalid_superscript,
        InvalidSuperscript,
        reason,
        "Creates an invalid superscript error."
    );
    simple_error_ctor!(
        malformed_matrix,
        MalformedMatrix,
        reason,
        "Creates a malformed matrix error."
    );

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
