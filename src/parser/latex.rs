// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

//! LaTeX expression parser for mathematical notation.
//!
//! This module provides parsing capabilities for LaTeX mathematical expressions,
//! converting tokenized LaTeX input into an Abstract Syntax Tree (AST).
//!
//! # Supported Constructs
//!
//! - **Fractions**: `\frac{num}{denom}` → Binary division
//! - **Roots**: `\sqrt{x}`, `\sqrt[n]{x}` → Function calls
//! - **Powers**: `x^2`, `x^{expr}` → Binary exponentiation
//! - **Subscripts**: `x_1`, `x_i`, `x_{i+1}` → Variables with subscripts (supports expressions)
//! - **Greek letters**: `\alpha`, `\beta`, etc. → Variables
//! - **Constants**: `\pi`, `\infty`, `e`, `i` → Mathematical constants
//! - **Trigonometric functions**: `\sin`, `\cos`, `\tan`, etc. → Functions
//! - **Basic operators**: `+`, `-`, `*`, `/`
//!
//! # Context-Aware Parsing of `e` and `i`
//!
//! The letters `e` and `i` receive special treatment to distinguish between their use as
//! mathematical constants (Euler's number and the imaginary unit) versus as variables.
//!
//! ## Explicit Markers (Always Constants)
//!
//! Use explicit markers to unambiguously specify constants:
//! - `\mathrm{e}` → Euler's number `e ≈ 2.71828`
//! - `\mathrm{i}` → Imaginary unit `i`
//! - `\imath` → Imaginary unit (mathematical notation)
//! - `\jmath` → Imaginary unit (engineering notation)
//!
//! ## Bound Variables in Iterators
//!
//! Index variables in `\sum` and `\prod` take precedence over constant interpretation:
//! - `\sum_{i=1}^{n} i` → `i` is a variable (the summation index)
//! - `\prod_{e=1}^{n} e` → `e` is a variable (the product index)
//!
//! ## Default Behavior
//!
//! When unbound and without explicit markers:
//! - `e` defaults to `Constant(E)` (Euler's number)
//! - `i` defaults to `Constant(I)` (imaginary unit)
//!
//! ## Exponential Normalization
//!
//! When `e` (Euler's number) is raised to a power, it's normalized to `exp()`:
//! - `e^x` → `Function("exp", [x])`
//! - `e^{i\pi}` → `Function("exp", [Constant(I) * Constant(Pi)])`
//!
//! This ensures equivalence with `\exp{x}`.
//!
//! ## Known Limitations
//!
//! 1. **Integral scope timing**: In `\int f(i) di`, the integrand is parsed before
//!    the differential variable is known. The `i` in `f(i)` will be `Constant(I)`.
//!    Workaround: use `\mathrm{i}` explicitly if you need `i` as a variable.
//!
//! 2. **Single-letter index only**: Index variables in `\sum` and `\prod` must be
//!    single ASCII letters.
//!
//! 3. **No complex pattern detection**: Patterns like `a + bi` are not specially
//!    detected; `i` defaults to constant regardless of context.
//!
//! # Example
//!
//! ```ignore
//! use mathlex::parser::parse_latex;
//!
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//! // Returns: Binary { op: Div, left: Integer(1), right: Integer(2) }
//! ```

use std::collections::HashSet;

use crate::ast::{
    BinaryOp, Direction, Expression, IndexType, InequalityOp, IntegralBounds, LogicalOp,
    MathConstant, MathFloat, RelationOp, SetOp, SetRelation, TensorIndex, VectorNotation,
};
use crate::error::{ParseError, ParseOutput, ParseResult, Span};
use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
use crate::parser::Spanned;

mod arithmetic;
mod calculus;
mod commands;
mod derivatives;
mod expression;
mod linear_algebra;
mod primary;

/// Parses a LaTeX mathematical expression.
///
/// # Arguments
///
/// * `input` - The LaTeX string to parse
///
/// # Returns
///
/// A parsed AST expression or a parse error.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_latex;
///
/// // Simple fraction
/// let expr = parse_latex(r"\frac{1}{2}").unwrap();
///
/// // Square root
/// let expr = parse_latex(r"\sqrt{x}").unwrap();
///
/// // Power
/// let expr = parse_latex(r"x^{2+3}").unwrap();
/// ```
pub fn parse_latex(input: &str) -> ParseResult<Expression> {
    let tokens = tokenize_latex(input)?;
    let parser = LatexParser::new(tokens, false);
    parser.parse_strict()
}

/// Parses a LaTeX mathematical expression in lenient (error-recovering) mode.
///
/// Instead of stopping at the first error, this function collects all errors
/// and returns a partial AST where possible.
///
/// # Arguments
///
/// * `input` - The LaTeX string to parse
///
/// # Returns
///
/// A [`ParseOutput`] containing the partial AST and all errors found.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_latex_lenient;
///
/// let output = parse_latex_lenient(r"\frac{1}{} + x");
/// assert!(output.has_errors());
/// ```
pub fn parse_latex_lenient(input: &str) -> ParseOutput {
    let tokens = match tokenize_latex(input) {
        Ok(tokens) => tokens,
        Err(err) => {
            return ParseOutput {
                expression: None,
                errors: vec![err],
            }
        }
    };
    let parser = LatexParser::new(tokens, true);
    parser.parse_lenient()
}

/// Internal parser state for LaTeX expressions.
struct LatexParser {
    /// Token stream with positions
    tokens: Vec<Spanned<LatexToken>>,
    /// Current position in token stream
    pos: usize,
    /// Stack of bound variable scopes (for sum/product index variables)
    bound_scopes: Vec<HashSet<String>>,
    /// Flag indicating we're parsing inside an integral context where
    /// 'dx' should not be parsed as a Differential
    in_integral_context: bool,
    /// Flag indicating we're parsing inside a fraction where
    /// 'dx' should not be parsed as a Differential (to allow d/dx derivative notation)
    in_fraction_context: bool,
    /// Errors collected during lenient parsing
    collected_errors: Vec<ParseError>,
}

impl LatexParser {
    /// Creates a new parser from a token stream.
    fn new(tokens: Vec<Spanned<LatexToken>>, _lenient: bool) -> Self {
        Self {
            tokens,
            pos: 0,
            bound_scopes: Vec::new(),
            in_integral_context: false,
            in_fraction_context: false,
            collected_errors: Vec::new(),
        }
    }

    /// Pushes a new scope with the given bound variables.
    fn push_scope(&mut self, vars: impl IntoIterator<Item = String>) {
        self.bound_scopes.push(vars.into_iter().collect());
    }

    /// Pops the current scope.
    fn pop_scope(&mut self) {
        self.bound_scopes.pop();
    }

    /// Checks if a variable name is bound in any current scope.
    fn is_bound(&self, name: &str) -> bool {
        self.bound_scopes.iter().any(|scope| scope.contains(name))
    }

    /// Resolves a letter to either a constant (for `e` and `i`) or a variable.
    ///
    /// Resolution rules:
    /// 1. Bound variables (in sum/product scopes) are always variables
    /// 2. Explicit markers (`\mathrm{e}`, `\mathrm{i}`, `\imath`, `\jmath`) are always constants
    /// 3. By default, `e` is Euler's number and `i` is the imaginary unit
    fn resolve_letter(&self, ch: char, is_explicit: bool) -> Expression {
        let name = ch.to_string();

        // Rule 1: Bound variables are always variables
        if self.is_bound(&name) {
            return Expression::Variable(name);
        }

        // Rule 2: Explicit markers are always constants
        // This includes e, i (complex) and j, k (quaternion) from \mathrm{x}
        if is_explicit {
            return match ch {
                'e' => Expression::Constant(MathConstant::E),
                'i' => Expression::Constant(MathConstant::I),
                'j' => Expression::Constant(MathConstant::J),
                'k' => Expression::Constant(MathConstant::K),
                _ => Expression::Variable(name),
            };
        }

        // Rule 3: Default for e and i (unbound) - these are constants
        // Note: j and k default to variables (quaternion context requires explicit markers)
        if ch == 'e' || ch == 'i' {
            return match ch {
                'e' => Expression::Constant(MathConstant::E),
                'i' => Expression::Constant(MathConstant::I),
                _ => unreachable!(),
            };
        }

        Expression::Variable(name)
    }

    /// Returns the current token without consuming it.
    fn peek(&self) -> Option<&Spanned<LatexToken>> {
        self.tokens.get(self.pos)
    }

    /// Returns the token at offset positions ahead without consuming it.
    fn peek_ahead(&self, offset: usize) -> Option<&Spanned<LatexToken>> {
        self.tokens.get(self.pos + offset)
    }

    /// Returns the current token and advances position.
    fn next(&mut self) -> Option<Spanned<LatexToken>> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Returns the current position/span for error reporting.
    fn current_span(&self) -> Span {
        self.peek().map(|(_, span)| *span).unwrap_or_else(|| {
            // Use the last token's end position if we're at EOF
            if let Some((_, last_span)) = self.tokens.last() {
                Span::at(last_span.end)
            } else {
                Span::start()
            }
        })
    }

    /// Checks if current token matches a pattern without consuming.
    fn check(&self, expected: &LatexToken) -> bool {
        self.peek().map(|(tok, _)| tok == expected).unwrap_or(false)
    }

    /// Consumes a token if it matches the expected token.
    fn consume(&mut self, expected: LatexToken) -> ParseResult<Span> {
        if let Some((token, span)) = self.next() {
            if token == expected {
                Ok(span)
            } else {
                Err(ParseError::unexpected_token(
                    vec![format!("{:?}", expected)],
                    format!("{:?}", token),
                    Some(span),
                ))
            }
        } else {
            Err(ParseError::unexpected_eof(
                vec![format!("{:?}", expected)],
                Some(self.current_span()),
            ))
        }
    }

    /// Returns true if the token is a synchronization point for error recovery.
    fn is_sync_token(token: &LatexToken) -> bool {
        matches!(
            token,
            LatexToken::RBrace
                | LatexToken::Plus
                | LatexToken::Minus
                | LatexToken::Equals
                | LatexToken::Eof
        ) || matches!(token, LatexToken::Command(cmd) if matches!(
            cmd.as_str(),
            "frac" | "sqrt" | "sum" | "prod" | "int" | "iint" | "iiint" | "oint" | "lim"
        ))
    }

    /// Advances past tokens until a synchronization point is found.
    /// Used in lenient mode to recover from errors.
    fn synchronize(&mut self) {
        while let Some((token, _)) = self.peek() {
            if Self::is_sync_token(token) {
                return;
            }
            self.next();
        }
    }

    /// Strict entry point: fails on first error.
    fn parse_strict(mut self) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;

        // Ensure we consumed all non-EOF tokens
        if let Some((token, span)) = self.peek() {
            if !matches!(token, LatexToken::Eof) {
                return Err(ParseError::unexpected_token(
                    vec!["end of input"],
                    format!("{:?}", token),
                    Some(*span),
                ));
            }
        }

        Ok(expr)
    }

    /// Lenient entry point: collects errors and returns partial AST.
    fn parse_lenient(mut self) -> ParseOutput {
        let mut parts: Vec<Expression> = Vec::new();

        while self.peek().is_some() && !matches!(self.peek(), Some((LatexToken::Eof, _))) {
            match self.parse_expression() {
                Ok(expr) => {
                    parts.push(expr);
                    // If there are remaining non-EOF tokens that aren't consumed,
                    // try to continue parsing
                    if let Some((token, _)) = self.peek() {
                        if !matches!(token, LatexToken::Eof) {
                            // Record the unconsumed token as an error but keep going
                            let span = self.current_span();
                            self.collected_errors.push(ParseError::unexpected_token(
                                vec!["end of input or operator"],
                                format!("{:?}", token),
                                Some(span),
                            ));
                            self.synchronize();
                        }
                    }
                }
                Err(err) => {
                    self.collected_errors.push(err);
                    self.synchronize();
                    // Skip the sync token itself if it's not EOF
                    if let Some((token, _)) = self.peek() {
                        if !matches!(token, LatexToken::Eof) && matches!(token, LatexToken::RBrace)
                        {
                            self.next();
                        }
                    }
                }
            }
        }

        let expression = match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => {
                // Multiple successfully-parsed segments: return the first
                Some(parts.remove(0))
            }
        };

        ParseOutput {
            expression,
            errors: self.collected_errors,
        }
    }
}

#[cfg(test)]
#[path = "latex/tests/latex_tests_greek.rs"]
mod greek_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_fractions.rs"]
mod fractions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_roots.rs"]
#[allow(clippy::approx_constant)]
mod roots_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_powers_subscripts.rs"]
mod powers_subscripts_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_functions.rs"]
#[allow(clippy::approx_constant)]
mod functions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_calculus.rs"]
mod calculus_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_constants.rs"]
mod constants_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_errors.rs"]
mod errors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_tensors.rs"]
mod tensors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_vectors.rs"]
mod vectors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_multiple_integrals.rs"]
mod multiple_integrals_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_logic.rs"]
mod logic_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_sets.rs"]
mod sets_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_quaternions.rs"]
mod quaternions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_differential_forms.rs"]
mod differential_forms_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_inline.rs"]
#[allow(clippy::approx_constant)]
mod tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_log_base.rs"]
mod log_base_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_floor_ceil.rs"]
mod floor_ceil_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_abs_sgn.rs"]
mod abs_sgn_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_det.rs"]
mod det_tests;
