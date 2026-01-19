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

use crate::error::{ParseError, ParseResult, Position, Span};

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

/// Tokenizer state with position tracking.
struct Tokenizer<'a> {
    /// Input string
    input: &'a str,
    /// Current byte offset
    offset: usize,
    /// Current line number (1-indexed)
    line: usize,
    /// Current column number (1-indexed)
    column: usize,
}

impl<'a> Tokenizer<'a> {
    /// Creates a new tokenizer from input string.
    fn new(input: &'a str) -> Self {
        Self {
            input,
            offset: 0,
            line: 1,
            column: 1,
        }
    }

    /// Returns the current position.
    fn position(&self) -> Position {
        Position::new(self.line, self.column, self.offset)
    }

    /// Peeks at the current character without consuming it.
    fn peek(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    /// Peeks at the character at the given offset ahead.
    fn peek_ahead(&self, n: usize) -> Option<char> {
        self.input[self.offset..].chars().nth(n)
    }

    /// Consumes and returns the current character.
    fn consume(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.offset += ch.len_utf8();

        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }

        Some(ch)
    }

    /// Skips whitespace characters.
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.consume();
            } else {
                break;
            }
        }
    }

    /// Scans a number (integer or float, including scientific notation).
    fn scan_number(&mut self) -> ParseResult<(Token, Span)> {
        let start = self.position();
        let mut number_str = String::new();
        let mut has_dot = false;
        let mut has_exp = false;

        // Scan integer part
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.consume();
            } else {
                break;
            }
        }

        // Check for decimal point
        if self.peek() == Some('.') {
            // Look ahead to ensure it's followed by a digit
            if let Some(next) = self.peek_ahead(1) {
                if next.is_ascii_digit() {
                    has_dot = true;
                    number_str.push('.');
                    self.consume();

                    // Scan fractional part
                    while let Some(ch) = self.peek() {
                        if ch.is_ascii_digit() {
                            number_str.push(ch);
                            self.consume();
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Check for exponent
        if let Some(ch) = self.peek() {
            if ch == 'e' || ch == 'E' {
                has_exp = true;
                number_str.push(ch);
                self.consume();

                // Optional sign
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        number_str.push(sign);
                        self.consume();
                    }
                }

                // Exponent digits
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        number_str.push(ch);
                        self.consume();
                    } else {
                        break;
                    }
                }
            }
        }

        let end = self.position();
        let span = Span::new(start, end);

        // Parse the number
        if has_dot || has_exp {
            match number_str.parse::<f64>() {
                Ok(n) => Ok((Token::Float(n), span)),
                Err(_) => Err(ParseError::invalid_number(
                    &number_str,
                    "invalid float",
                    Some(span),
                )),
            }
        } else {
            match number_str.parse::<i64>() {
                Ok(n) => Ok((Token::Integer(n), span)),
                Err(_) => Err(ParseError::invalid_number(
                    &number_str,
                    "invalid integer",
                    Some(span),
                )),
            }
        }
    }

    /// Scans an identifier or keyword.
    fn scan_identifier(&mut self) -> (Token, Span) {
        let start = self.position();
        let mut ident = String::new();

        // First character (must be alphabetic)
        if let Some(ch) = self.peek() {
            if ch.is_ascii_alphabetic() {
                ident.push(ch);
                self.consume();
            }
        }

        // Rest of identifier (alphanumeric only - underscores are separate tokens for subscripts)
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() {
                ident.push(ch);
                self.consume();
            } else {
                break;
            }
        }

        let end = self.position();
        let span = Span::new(start, end);

        (Token::Identifier(ident), span)
    }

    /// Scans the next token.
    fn scan_token(&mut self) -> ParseResult<Option<SpannedToken>> {
        self.skip_whitespace();

        let Some(ch) = self.peek() else {
            return Ok(None);
        };

        let start = self.position();

        // Numbers
        if ch.is_ascii_digit() {
            let (token, span) = self.scan_number()?;
            return Ok(Some(SpannedToken::new(token, span)));
        }

        // Identifiers
        if ch.is_ascii_alphabetic() {
            let (token, span) = self.scan_identifier();
            return Ok(Some(SpannedToken::new(token, span)));
        }

        // Multi-character operators and Unicode symbols
        match ch {
            '!' => {
                self.consume();
                if self.peek() == Some('=') {
                    self.consume();
                    let end = self.position();
                    return Ok(Some(SpannedToken::new(
                        Token::NotEquals,
                        Span::new(start, end),
                    )));
                }
                let end = self.position();
                return Ok(Some(SpannedToken::new(Token::Bang, Span::new(start, end))));
            }
            '<' => {
                self.consume();
                if self.peek() == Some('=') {
                    self.consume();
                    let end = self.position();
                    return Ok(Some(SpannedToken::new(
                        Token::LessEq,
                        Span::new(start, end),
                    )));
                }
                let end = self.position();
                return Ok(Some(SpannedToken::new(Token::Less, Span::new(start, end))));
            }
            '>' => {
                self.consume();
                if self.peek() == Some('=') {
                    self.consume();
                    let end = self.position();
                    return Ok(Some(SpannedToken::new(
                        Token::GreaterEq,
                        Span::new(start, end),
                    )));
                }
                let end = self.position();
                return Ok(Some(SpannedToken::new(
                    Token::Greater,
                    Span::new(start, end),
                )));
            }
            // Unicode inequality symbols
            '≤' => {
                self.consume();
                let end = self.position();
                return Ok(Some(SpannedToken::new(
                    Token::LessEq,
                    Span::new(start, end),
                )));
            }
            '≥' => {
                self.consume();
                let end = self.position();
                return Ok(Some(SpannedToken::new(
                    Token::GreaterEq,
                    Span::new(start, end),
                )));
            }
            '≠' => {
                self.consume();
                let end = self.position();
                return Ok(Some(SpannedToken::new(
                    Token::NotEquals,
                    Span::new(start, end),
                )));
            }
            _ => {}
        }

        // Single-character tokens
        self.consume();
        let end = self.position();
        let span = Span::new(start, end);

        let token = match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '^' => Token::Caret,
            '%' => Token::Percent,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            '=' => Token::Equals,
            '_' => Token::Underscore,
            _ => {
                return Err(ParseError::unexpected_token(
                    vec!["valid token".to_string()],
                    ch.to_string(),
                    Some(span),
                ));
            }
        };

        Ok(Some(SpannedToken::new(token, span)))
    }

    /// Tokenizes the entire input.
    fn tokenize_all(&mut self) -> ParseResult<Vec<SpannedToken>> {
        let mut tokens = Vec::new();

        while let Some(token) = self.scan_token()? {
            tokens.push(token);
        }

        Ok(tokens)
    }
}

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
    let mut tokenizer = Tokenizer::new(input);
    tokenizer.tokenize_all()
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_integer() {
        let tokens = tokenize("42").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Integer(42));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("3.14").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].value, Token::Float(f) if (f - 3.14).abs() < 0.001));
    }

    #[test]
    fn test_tokenize_scientific_notation() {
        let tokens = tokenize("1.5e-3").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].value, Token::Float(f) if (f - 0.0015).abs() < 0.0001));
    }

    #[test]
    fn test_tokenize_identifier() {
        let tokens = tokenize("x").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
    }

    #[test]
    fn test_tokenize_multi_char_identifier() {
        let tokens = tokenize("theta").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("theta".to_string()));
    }

    #[test]
    fn test_tokenize_operators() {
        let tokens = tokenize("+ - * / ^ %").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::Plus);
        assert_eq!(tokens[1].value, Token::Minus);
        assert_eq!(tokens[2].value, Token::Star);
        assert_eq!(tokens[3].value, Token::Slash);
        assert_eq!(tokens[4].value, Token::Caret);
        assert_eq!(tokens[5].value, Token::Percent);
    }

    #[test]
    fn test_tokenize_delimiters() {
        let tokens = tokenize("( ) [ ] { }").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::LParen);
        assert_eq!(tokens[1].value, Token::RParen);
        assert_eq!(tokens[2].value, Token::LBracket);
        assert_eq!(tokens[3].value, Token::RBracket);
        assert_eq!(tokens[4].value, Token::LBrace);
        assert_eq!(tokens[5].value, Token::RBrace);
    }

    #[test]
    fn test_tokenize_relations() {
        let tokens = tokenize("= != < <= > >=").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::Equals);
        assert_eq!(tokens[1].value, Token::NotEquals);
        assert_eq!(tokens[2].value, Token::Less);
        assert_eq!(tokens[3].value, Token::LessEq);
        assert_eq!(tokens[4].value, Token::Greater);
        assert_eq!(tokens[5].value, Token::GreaterEq);
    }

    #[test]
    fn test_tokenize_unicode_relations() {
        let tokens = tokenize("≤ ≥ ≠").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::LessEq);
        assert_eq!(tokens[1].value, Token::GreaterEq);
        assert_eq!(tokens[2].value, Token::NotEquals);
    }

    #[test]
    fn test_tokenize_expression() {
        let tokens = tokenize("2 + x * 3.14").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Plus);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::Star);
        assert!(matches!(tokens[4].value, Token::Float(_)));
    }

    #[test]
    fn test_tokenize_function_call() {
        let tokens = tokenize("sin(x)").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].value, Token::Identifier("sin".to_string()));
        assert_eq!(tokens[1].value, Token::LParen);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::RParen);
    }

    #[test]
    fn test_tokenize_factorial() {
        let tokens = tokenize("5!").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Integer(5));
        assert_eq!(tokens[1].value, Token::Bang);
    }

    #[test]
    fn test_tokenize_underscore() {
        let tokens = tokenize("x_1").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Underscore);
        assert_eq!(tokens[2].value, Token::Integer(1));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = tokenize("   ").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_invalid_character() {
        let result = tokenize("@");
        assert!(result.is_err());
    }

    #[test]
    fn test_span_tracking() {
        let tokens = tokenize("x + y").unwrap();
        assert_eq!(tokens.len(), 3);

        // Check first token span
        assert_eq!(tokens[0].span.start.column, 1);
        assert_eq!(tokens[0].span.end.column, 2);

        // Check third token span
        assert_eq!(tokens[2].span.start.column, 5);
        assert_eq!(tokens[2].span.end.column, 6);
    }
}
