//! Tokenizer for plain text mathematical expressions.
//!
//! This module provides tokenization of plain text mathematical notation into
//! a stream of tokens with precise span information for error reporting.
//!
//! # Example
//!
//! ```
//! use mathlex::parser::tokenize;
//!
//! let tokens = tokenize("2 + x").unwrap();
//! assert_eq!(tokens.len(), 3);
//! ```

use crate::error::{ParseResult, ParseError, Position, Span};
use chumsky::prelude::*;

/// A token in a mathematical expression.
///
/// Represents the atomic elements of a mathematical expression including
/// literals, operators, delimiters, and special symbols.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    /// Integer literal (e.g., 42, -17)
    Integer(i64),
    /// Floating point literal (e.g., 3.14, 1.5e-3)
    Float(f64),
    /// Identifier/variable name (e.g., x, theta, var_1)
    Identifier(String),

    // Operators
    /// Addition operator (+)
    Plus,
    /// Subtraction operator (-)
    Minus,
    /// Multiplication operator (*)
    Star,
    /// Division operator (/)
    Slash,
    /// Exponentiation operator (^)
    Caret,
    /// Modulo operator (%)
    Percent,
    /// Factorial operator (!)
    Bang,

    // Delimiters
    /// Left parenthesis (()
    LParen,
    /// Right parenthesis ())
    RParen,
    /// Left square bracket ([)
    LBracket,
    /// Right square bracket (])
    RBracket,
    /// Left curly brace ({)
    LBrace,
    /// Right curly brace (})
    RBrace,
    /// Comma separator (,)
    Comma,
    /// Semicolon separator (;)
    Semicolon,

    // Relations
    /// Equality (=)
    Equals,
    /// Inequality (!=)
    NotEquals,
    /// Less than (<)
    Less,
    /// Less than or equal (<=)
    LessEq,
    /// Greater than (>)
    Greater,
    /// Greater than or equal (>=)
    GreaterEq,

    // Special
    /// Underscore for subscripts (_)
    Underscore,
    /// End of input
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Integer(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::Percent => write!(f, "%"),
            Token::Bang => write!(f, "!"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Equals => write!(f, "="),
            Token::NotEquals => write!(f, "!="),
            Token::Less => write!(f, "<"),
            Token::LessEq => write!(f, "<="),
            Token::Greater => write!(f, ">"),
            Token::GreaterEq => write!(f, ">="),
            Token::Underscore => write!(f, "_"),
            Token::Eof => write!(f, "<EOF>"),
        }
    }
}

/// A value with an associated span in the source text.
///
/// This wrapper type attaches position information to tokens for
/// precise error reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// The wrapped value
    pub value: T,
    /// The span in the source text
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Creates a new spanned value.
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }
}

/// A token with span information.
pub type SpannedToken = Spanned<Token>;

/// Tokenizes a mathematical expression string.
///
/// Converts the input string into a vector of tokens with span information.
/// Each token includes its position in the source for error reporting.
///
/// # Arguments
///
/// * `input` - The mathematical expression to tokenize
///
/// # Returns
///
/// A vector of spanned tokens, or a parse error if tokenization fails.
///
/// # Example
///
/// ```
/// use mathlex::parser::tokenize;
///
/// let tokens = tokenize("2 + x * 3.14").unwrap();
/// assert_eq!(tokens.len(), 5);
/// ```
pub fn tokenize(input: &str) -> ParseResult<Vec<SpannedToken>> {
    // Build the token parser
    let lexer = build_lexer();

    // Parse the input
    match lexer.parse(input) {
        Ok(tokens) => Ok(tokens),
        Err(errors) => {
            // Convert chumsky errors to our ParseError type
            // Take the first error for simplicity
            if let Some(err) = errors.into_iter().next() {
                Err(convert_chumsky_error(err, input))
            } else {
                Err(ParseError::custom("unknown tokenization error", None))
            }
        }
    }
}

/// Builds the chumsky lexer for mathematical expressions.
fn build_lexer() -> impl Parser<char, Vec<SpannedToken>, Error = Simple<char>> {
    // Number parser: handles integers and floats (including scientific notation)
    let number = text::int(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .chain::<char, _, _>(
            just('e')
                .or(just('E'))
                .chain(just('+').or(just('-')).or_not())
                .chain::<char, _, _>(text::digits(10))
                .or_not()
                .flatten()
        )
        .collect::<String>()
        .map(|s| {
            // Try parsing as integer first, then as float
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s.parse::<f64>()
                    .map(Token::Float)
                    .unwrap_or_else(|_| Token::Float(0.0))
            } else {
                s.parse::<i64>()
                    .map(Token::Integer)
                    .unwrap_or_else(|_| Token::Integer(0))
            }
        });

    // Identifier parser: starts with letter, followed by alphanumerics
    let identifier = filter(|c: &char| c.is_ascii_alphabetic())
        .chain::<char, _, _>(filter(|c: &char| c.is_ascii_alphanumeric() || *c == '_').repeated())
        .collect::<String>()
        .map(Token::Identifier);

    // Multi-character operators (must come before single-char versions)
    let not_equals = just("!=").to(Token::NotEquals);
    let less_eq = just("<=").to(Token::LessEq);
    let greater_eq = just(">=").to(Token::GreaterEq);

    // Single-character operators and delimiters
    let plus = just('+').to(Token::Plus);
    let minus = just('-').to(Token::Minus);
    let star = just('*').to(Token::Star);
    let slash = just('/').to(Token::Slash);
    let caret = just('^').to(Token::Caret);
    let percent = just('%').to(Token::Percent);
    let bang = just('!').to(Token::Bang);

    let lparen = just('(').to(Token::LParen);
    let rparen = just(')').to(Token::RParen);
    let lbracket = just('[').to(Token::LBracket);
    let rbracket = just(']').to(Token::RBracket);
    let lbrace = just('{').to(Token::LBrace);
    let rbrace = just('}').to(Token::RBrace);

    let comma = just(',').to(Token::Comma);
    let semicolon = just(';').to(Token::Semicolon);

    let equals = just('=').to(Token::Equals);
    let less = just('<').to(Token::Less);
    let greater = just('>').to(Token::Greater);

    let underscore = just('_').to(Token::Underscore);

    // Combine all token parsers
    // Order matters: try multi-char operators before single-char
    let token = choice((
        number,
        identifier,
        not_equals,
        less_eq,
        greater_eq,
        plus,
        minus,
        star,
        slash,
        caret,
        percent,
        bang,
        lparen,
        rparen,
        lbracket,
        rbracket,
        lbrace,
        rbrace,
        comma,
        semicolon,
        equals,
        less,
        greater,
        underscore,
    ))
    .map_with_span(|token, span: std::ops::Range<usize>| {
        // Convert byte span to Position-based Span
        Spanned::new(token, byte_span_to_span(span))
    });

    // Skip whitespace and parse tokens
    token
        .padded()
        .repeated()
        .then_ignore(end())
}

/// Converts a byte-offset span to a Position-based Span.
///
/// For now, we use a simplified approach assuming single-line input.
/// A full implementation would track line/column information.
fn byte_span_to_span(range: std::ops::Range<usize>) -> Span {
    let start = Position::new(1, range.start + 1, range.start);
    let end = Position::new(1, range.end + 1, range.end);
    Span::new(start, end)
}

/// Converts a chumsky error to our ParseError type.
fn convert_chumsky_error(err: Simple<char>, _input: &str) -> ParseError {
    match err.reason() {
        chumsky::error::SimpleReason::Unexpected => {
            let found = err.found()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "end of input".to_string());

            let expected: Vec<String> = err.expected()
                .map(|opt_c| {
                    opt_c.map(|c| c.to_string())
                        .unwrap_or_else(|| "end of input".to_string())
                })
                .collect();

            let span = byte_span_to_span(err.span());

            if err.found().is_none() {
                ParseError::unexpected_eof(expected, Some(span))
            } else {
                ParseError::unexpected_token(expected, found, Some(span))
            }
        }
        chumsky::error::SimpleReason::Unclosed { span, delimiter } => {
            let pos_span = byte_span_to_span(span.clone());
            ParseError::unmatched_delimiter(
                *delimiter,
                pos_span.start,
                Some(byte_span_to_span(err.span()))
            )
        }
        chumsky::error::SimpleReason::Custom(msg) => {
            ParseError::custom(msg.clone(), Some(byte_span_to_span(err.span())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_integer() {
        let tokens = tokenize("42").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Integer(42));
    }

    #[test]
    fn test_tokenize_negative_integer() {
        let tokens = tokenize("-17").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Minus);
        assert_eq!(tokens[1].value, Token::Integer(17));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("3.14").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Float(3.14));
    }

    #[test]
    fn test_tokenize_scientific_notation() {
        let tokens = tokenize("1.5e-3").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Float(1.5e-3));
    }

    #[test]
    fn test_tokenize_scientific_notation_positive_exp() {
        let tokens = tokenize("2.5E+10").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Float(2.5e10));
    }

    #[test]
    fn test_tokenize_identifier() {
        let tokens = tokenize("x").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
    }

    #[test]
    fn test_tokenize_identifier_multi_char() {
        let tokens = tokenize("theta").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("theta".to_string()));
    }

    #[test]
    fn test_tokenize_identifier_with_numbers() {
        let tokens = tokenize("var_1").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("var_1".to_string()));
    }

    #[test]
    fn test_tokenize_plus() {
        let tokens = tokenize("+").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Plus);
    }

    #[test]
    fn test_tokenize_minus() {
        let tokens = tokenize("-").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Minus);
    }

    #[test]
    fn test_tokenize_star() {
        let tokens = tokenize("*").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Star);
    }

    #[test]
    fn test_tokenize_slash() {
        let tokens = tokenize("/").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Slash);
    }

    #[test]
    fn test_tokenize_caret() {
        let tokens = tokenize("^").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Caret);
    }

    #[test]
    fn test_tokenize_percent() {
        let tokens = tokenize("%").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Percent);
    }

    #[test]
    fn test_tokenize_bang() {
        let tokens = tokenize("!").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Bang);
    }

    #[test]
    fn test_tokenize_lparen() {
        let tokens = tokenize("(").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::LParen);
    }

    #[test]
    fn test_tokenize_rparen() {
        let tokens = tokenize(")").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::RParen);
    }

    #[test]
    fn test_tokenize_lbracket() {
        let tokens = tokenize("[").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::LBracket);
    }

    #[test]
    fn test_tokenize_rbracket() {
        let tokens = tokenize("]").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::RBracket);
    }

    #[test]
    fn test_tokenize_lbrace() {
        let tokens = tokenize("{").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::LBrace);
    }

    #[test]
    fn test_tokenize_rbrace() {
        let tokens = tokenize("}").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::RBrace);
    }

    #[test]
    fn test_tokenize_comma() {
        let tokens = tokenize(",").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Comma);
    }

    #[test]
    fn test_tokenize_semicolon() {
        let tokens = tokenize(";").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Semicolon);
    }

    #[test]
    fn test_tokenize_equals() {
        let tokens = tokenize("=").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Equals);
    }

    #[test]
    fn test_tokenize_not_equals() {
        let tokens = tokenize("!=").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::NotEquals);
    }

    #[test]
    fn test_tokenize_less() {
        let tokens = tokenize("<").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Less);
    }

    #[test]
    fn test_tokenize_less_eq() {
        let tokens = tokenize("<=").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::LessEq);
    }

    #[test]
    fn test_tokenize_greater() {
        let tokens = tokenize(">").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Greater);
    }

    #[test]
    fn test_tokenize_greater_eq() {
        let tokens = tokenize(">=").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::GreaterEq);
    }

    #[test]
    fn test_tokenize_underscore() {
        let tokens = tokenize("_").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Underscore);
    }

    #[test]
    fn test_tokenize_simple_expression() {
        let tokens = tokenize("2 + x").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Plus);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
    }

    #[test]
    fn test_tokenize_complex_expression() {
        let tokens = tokenize("2 * x + 3.14 / y").unwrap();
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Star);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::Plus);
        assert_eq!(tokens[4].value, Token::Float(3.14));
        assert_eq!(tokens[5].value, Token::Slash);
        assert_eq!(tokens[6].value, Token::Identifier("y".to_string()));
    }

    #[test]
    fn test_tokenize_with_parentheses() {
        let tokens = tokenize("(x + y) * z").unwrap();
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0].value, Token::LParen);
        assert_eq!(tokens[1].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[2].value, Token::Plus);
        assert_eq!(tokens[3].value, Token::Identifier("y".to_string()));
        assert_eq!(tokens[4].value, Token::RParen);
        assert_eq!(tokens[5].value, Token::Star);
        assert_eq!(tokens[6].value, Token::Identifier("z".to_string()));
    }

    #[test]
    fn test_tokenize_subscript() {
        let tokens = tokenize("x_1").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Underscore);
        assert_eq!(tokens[2].value, Token::Integer(1));
    }

    #[test]
    fn test_tokenize_power() {
        let tokens = tokenize("x^2").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Caret);
        assert_eq!(tokens[2].value, Token::Integer(2));
    }

    #[test]
    fn test_tokenize_factorial() {
        let tokens = tokenize("5!").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Integer(5));
        assert_eq!(tokens[1].value, Token::Bang);
    }

    #[test]
    fn test_tokenize_comparison() {
        let tokens = tokenize("x <= 5").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::LessEq);
        assert_eq!(tokens[2].value, Token::Integer(5));
    }

    #[test]
    fn test_tokenize_whitespace_handling() {
        let tokens = tokenize("  x   +   y  ").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Plus);
        assert_eq!(tokens[2].value, Token::Identifier("y".to_string()));
    }

    #[test]
    fn test_tokenize_no_whitespace() {
        let tokens = tokenize("x+y").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Plus);
        assert_eq!(tokens[2].value, Token::Identifier("y".to_string()));
    }

    #[test]
    fn test_tokenize_matrix_brackets() {
        let tokens = tokenize("[1, 2; 3, 4]").unwrap();
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0].value, Token::LBracket);
        assert_eq!(tokens[1].value, Token::Integer(1));
        assert_eq!(tokens[2].value, Token::Comma);
        assert_eq!(tokens[3].value, Token::Integer(2));
        assert_eq!(tokens[4].value, Token::Semicolon);
        assert_eq!(tokens[5].value, Token::Integer(3));
        assert_eq!(tokens[6].value, Token::Comma);
        assert_eq!(tokens[7].value, Token::Integer(4));
        assert_eq!(tokens[8].value, Token::RBracket);
    }

    #[test]
    fn test_token_display_integer() {
        let token = Token::Integer(42);
        assert_eq!(token.to_string(), "42");
    }

    #[test]
    fn test_token_display_float() {
        let token = Token::Float(3.14);
        assert_eq!(token.to_string(), "3.14");
    }

    #[test]
    fn test_token_display_identifier() {
        let token = Token::Identifier("x".to_string());
        assert_eq!(token.to_string(), "x");
    }

    #[test]
    fn test_token_display_operators() {
        assert_eq!(Token::Plus.to_string(), "+");
        assert_eq!(Token::Minus.to_string(), "-");
        assert_eq!(Token::Star.to_string(), "*");
        assert_eq!(Token::Slash.to_string(), "/");
        assert_eq!(Token::Caret.to_string(), "^");
    }

    #[test]
    fn test_spanned_new() {
        let span = Span::new(Position::new(1, 1, 0), Position::new(1, 3, 2));
        let spanned = Spanned::new(Token::Integer(42), span);
        assert_eq!(spanned.value, Token::Integer(42));
        assert_eq!(spanned.span, span);
    }

    #[test]
    fn test_span_information_preserved() {
        let tokens = tokenize("x + y").unwrap();

        // First token 'x' should start at column 1
        assert_eq!(tokens[0].span.start.column, 1);

        // Second token '+' should be at a different position
        assert_ne!(tokens[1].span.start.offset, tokens[0].span.start.offset);

        // Third token 'y' should be at yet another position
        assert_ne!(tokens[2].span.start.offset, tokens[1].span.start.offset);
    }

    #[test]
    fn test_tokenize_empty_string() {
        let tokens = tokenize("").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenize_only_whitespace() {
        let tokens = tokenize("   ").unwrap();
        assert_eq!(tokens.len(), 0);
    }
}
