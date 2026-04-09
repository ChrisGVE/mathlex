//! Plain text mathematical expression parser.
//!
//! This module implements a recursive descent parser for plain text
//! mathematical notation. It takes tokens from the tokenizer and builds an AST.
//!
//! # Operator Precedence (lowest to highest)
//!
//! 1. Logical operators (iff, implies, or, and, not)
//! 2. Quantifiers (forall, exists)
//! 3. Set operations (union, intersect, in, notin)
//! 4. Relations (=, <, >, <=, >=, !=)
//! 5. Addition, Subtraction (+, -)
//! 6. Multiplication, Division, Modulo (*, /, %)
//! 7. Unary prefix operators (-, +)
//! 8. Power (^) - RIGHT ASSOCIATIVE
//! 9. Postfix operators (!)
//! 10. Function calls and atoms
//!
//! # Examples
//!
//! ```
//! use mathlex::parser::parse;
//!
//! let expr = parse("2 + 3 * 4").unwrap();
//! // Parses as: 2 + (3 * 4)
//!
//! let expr = parse("2^3^4").unwrap();
//! // Parses as: 2^(3^4) - right associative
//! ```

use crate::ast::{
    BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, LogicalOp, MathConstant,
    MathFloat, SetOp, SetRelation, UnaryOp,
};
use crate::error::{ParseError, ParseOutput, ParseResult, Span};
use crate::parser::tokenizer::{tokenize, SpannedToken, Token};
use crate::ParserConfig;

mod arithmetic;
mod expression;
mod primary;
mod set_ops;
#[cfg(test)]
mod tests;

/// Parses a plain text mathematical expression with default configuration.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse;
///
/// let expr = parse("sin(x) + 2").unwrap();
/// ```
pub fn parse(input: &str) -> ParseResult<Expression> {
    parse_with_config(input, &ParserConfig::default())
}

/// Parses a plain text mathematical expression with custom configuration.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_with_config;
/// use mathlex::ParserConfig;
///
/// let config = ParserConfig {
///     implicit_multiplication: true,
///     ..Default::default()
/// };
/// let expr = parse_with_config("2x", &config).unwrap();
/// ```
pub fn parse_with_config(input: &str, config: &ParserConfig) -> ParseResult<Expression> {
    let tokens = tokenize(input)?;
    let parser = TextParser::new(tokens, config.clone(), false);
    parser.parse_strict()
}

/// Parses a plain text mathematical expression in lenient mode.
///
/// Instead of stopping at the first error, collects all errors and
/// returns a partial AST where possible.
pub fn parse_lenient(input: &str) -> ParseOutput {
    parse_lenient_with_config(input, &ParserConfig::default())
}

/// Parses a semicolon-delimited string of equations into a vector of expressions.
///
/// Each segment separated by `;` is parsed as an independent expression.
/// Empty segments (e.g., from trailing semicolons) are ignored.
///
/// # Errors
///
/// Returns a [`ParseError`] from the first segment that fails to parse.
///
/// # Examples
///
/// ```
/// use mathlex::parser::text::parse_equation_system;
/// use mathlex::Expression;
///
/// let exprs = parse_equation_system("x + y = 5; 2*x - y = 1").unwrap();
/// assert_eq!(exprs.len(), 2);
/// ```
pub fn parse_equation_system(input: &str) -> ParseResult<Vec<Expression>> {
    parse_equation_system_with_config(input, &ParserConfig::default())
}

/// Parses a semicolon-delimited string of equations with custom configuration.
///
/// Each segment separated by `;` is parsed as an independent expression using
/// the supplied [`ParserConfig`]. Empty segments are ignored.
///
/// # Errors
///
/// Returns a [`ParseError`] from the first segment that fails to parse.
///
/// # Examples
///
pub(crate) fn parse_equation_system_with_config(
    input: &str,
    config: &ParserConfig,
) -> ParseResult<Vec<Expression>> {
    input
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|part| parse_with_config(part, config))
        .collect()
}

/// Parses a plain text mathematical expression in lenient mode with config.
///
/// Instead of stopping at the first error, collects all errors and
/// returns a partial AST where possible.
pub fn parse_lenient_with_config(input: &str, config: &ParserConfig) -> ParseOutput {
    let tokens = match tokenize(input) {
        Ok(tokens) => tokens,
        Err(err) => {
            return ParseOutput {
                expression: None,
                errors: vec![err],
            }
        }
    };
    let parser = TextParser::new(tokens, config.clone(), true);
    parser.parse_lenient()
}

/// Internal parser state for text expressions.
struct TextParser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    config: ParserConfig,
    collected_errors: Vec<ParseError>,
}

impl TextParser {
    fn new(tokens: Vec<SpannedToken>, config: ParserConfig, _lenient: bool) -> Self {
        Self {
            tokens,
            pos: 0,
            config,
            collected_errors: Vec::new(),
        }
    }

    fn peek(&self) -> Option<&SpannedToken> {
        self.tokens.get(self.pos)
    }

    fn peek_ahead(&self, offset: usize) -> Option<&SpannedToken> {
        self.tokens.get(self.pos + offset)
    }

    fn next(&mut self) -> Option<SpannedToken> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn current_span(&self) -> Span {
        self.peek().map(|token| token.span).unwrap_or_else(|| {
            if let Some(last_token) = self.tokens.last() {
                Span::at(last_token.span.end)
            } else {
                Span::start()
            }
        })
    }

    fn check(&self, expected: &Token) -> bool {
        self.peek()
            .map(|token| &token.value == expected)
            .unwrap_or(false)
    }

    /// Determines if implicit multiplication should be inserted.
    ///
    /// Returns true when config has implicit_multiplication enabled AND
    /// the next token is an identifier or left parenthesis.
    fn should_insert_implicit_mult(&self, _left: &Expression) -> bool {
        if !self.config.implicit_multiplication {
            return false;
        }
        let next_token = match self.peek() {
            Some(token) => &token.value,
            None => return false,
        };
        matches!(next_token, Token::Identifier(_) | Token::LParen)
    }

    fn consume(&mut self, expected: Token) -> ParseResult<Span> {
        if let Some(token) = self.next() {
            if token.value == expected {
                Ok(token.span)
            } else {
                Err(ParseError::unexpected_token(
                    vec![format!("{}", expected)],
                    format!("{}", token.value),
                    Some(token.span),
                ))
            }
        } else {
            Err(ParseError::unexpected_eof(
                vec![format!("{}", expected)],
                Some(self.current_span()),
            ))
        }
    }

    fn is_sync_token(token: &Token) -> bool {
        matches!(
            token,
            Token::RParen
                | Token::Plus
                | Token::Minus
                | Token::Equals
                | Token::Semicolon
                | Token::Comma
        )
    }

    fn synchronize(&mut self) {
        while let Some(token) = self.peek() {
            if Self::is_sync_token(&token.value) {
                return;
            }
            self.next();
        }
    }

    fn parse_strict(mut self) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;
        if let Some(token) = self.peek() {
            return Err(ParseError::unexpected_token(
                vec!["end of input"],
                format!("{}", token.value),
                Some(token.span),
            ));
        }
        Ok(expr)
    }

    fn parse_lenient(mut self) -> ParseOutput {
        let mut parts: Vec<Expression> = Vec::new();
        while self.peek().is_some() {
            match self.parse_expression() {
                Ok(expr) => {
                    parts.push(expr);
                    if let Some(token) = self.peek() {
                        self.collected_errors.push(ParseError::unexpected_token(
                            vec!["end of input or operator"],
                            format!("{}", token.value),
                            Some(token.span),
                        ));
                        self.synchronize();
                        if let Some(t) = self.peek() {
                            if matches!(t.value, Token::RParen) {
                                self.next();
                            }
                        }
                    }
                }
                Err(err) => {
                    self.collected_errors.push(err);
                    self.synchronize();
                    if let Some(t) = self.peek() {
                        if matches!(t.value, Token::RParen) {
                            self.next();
                        }
                    }
                }
            }
        }
        let expression = match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => Some(parts.remove(0)),
        };
        ParseOutput {
            expression,
            errors: self.collected_errors,
        }
    }
}
