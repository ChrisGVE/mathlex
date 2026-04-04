//! ErrorBuilder for ergonomic parse error construction.

use super::types::{ParseError, ParseErrorKind, Position, Span};

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
    suggestion: Option<String>,
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
            suggestion: None,
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

    /// Adds a suggestion to this error.
    ///
    /// # Arguments
    ///
    /// * `suggestion` - A helpful suggestion for fixing the error
    pub fn with_suggestion<S: Into<String>>(mut self, suggestion: S) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Builds the parse error.
    pub fn build(self) -> ParseError {
        ParseError {
            kind: self.kind,
            span: self.span,
            context: self.context,
            suggestion: self.suggestion,
        }
    }
}
