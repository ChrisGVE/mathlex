//! ParseOutput struct for lenient (error-recovering) parsing results.

use super::types::{ParseError, ParseResult};

/// Output from lenient (error-recovering) parsing.
///
/// Contains a partial AST (if any portion was parseable) alongside
/// all errors encountered during parsing. This allows callers to
/// report multiple errors at once instead of stopping at the first.
///
/// # Examples
///
/// ```
/// use mathlex::error::ParseOutput;
/// use mathlex::ast::Expression;
///
/// // Successful parse with no errors
/// let output = ParseOutput {
///     expression: Some(Expression::Integer(42)),
///     errors: vec![],
/// };
/// assert!(output.is_ok());
///
/// // Failed parse with errors
/// let output: ParseOutput = ParseOutput {
///     expression: None,
///     errors: vec![],
/// };
/// assert!(!output.is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ParseOutput {
    /// The (possibly partial) parsed expression.
    /// `None` if no valid expression could be recovered.
    pub expression: Option<crate::ast::Expression>,
    /// All errors encountered during parsing, in source order.
    pub errors: Vec<ParseError>,
}

impl ParseOutput {
    /// Returns `true` if parsing produced an expression with no errors.
    pub fn is_ok(&self) -> bool {
        self.expression.is_some() && self.errors.is_empty()
    }

    /// Returns `true` if any errors were encountered.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Converts a strict `ParseResult` into a `ParseOutput`.
    pub fn from_result(result: ParseResult<crate::ast::Expression>) -> Self {
        match result {
            Ok(expr) => Self {
                expression: Some(expr),
                errors: vec![],
            },
            Err(err) => Self {
                expression: None,
                errors: vec![err],
            },
        }
    }
}
