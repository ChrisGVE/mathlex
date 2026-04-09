//! Tokenizer struct and scanning methods for plain text expressions.

use crate::error::{ParseError, ParseResult, Position, Span};

use super::token_types::{SpannedToken, Token};

/// Tokenizer state with position tracking.
pub(super) struct Tokenizer<'a> {
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
    pub(super) fn new(input: &'a str) -> Self {
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

    /// Scans the optional decimal fraction into `buf`; returns true if consumed.
    fn scan_decimal_part(&mut self, buf: &mut String) -> bool {
        if self.peek() == Some('.') && matches!(self.peek_ahead(1), Some(d) if d.is_ascii_digit()) {
            buf.push('.');
            self.consume();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    buf.push(ch);
                    self.consume();
                } else {
                    break;
                }
            }
            return true;
        }
        false
    }

    /// Scans the optional scientific-notation exponent into `buf`; returns true if consumed.
    fn scan_exponent_part(&mut self, buf: &mut String) -> bool {
        if matches!(self.peek(), Some('e' | 'E')) {
            buf.push(self.peek().unwrap());
            self.consume();
            if matches!(self.peek(), Some('+' | '-')) {
                buf.push(self.peek().unwrap());
                self.consume();
            }
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    buf.push(ch);
                    self.consume();
                } else {
                    break;
                }
            }
            return true;
        }
        false
    }

    /// Scans a number (integer or float, including scientific notation).
    fn scan_number(&mut self) -> ParseResult<(Token, Span)> {
        let start = self.position();
        let mut number_str = String::new();

        // Scan integer part
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                self.consume();
            } else {
                break;
            }
        }

        let has_dot = self.scan_decimal_part(&mut number_str);
        let has_exp = self.scan_exponent_part(&mut number_str);

        let end = self.position();
        let span = Span::new(start, end);

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

        // Rest: alphanumeric only — underscores are separate tokens
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

        let token = match ident.as_str() {
            "dot" => Token::Dot,
            "cross" => Token::Cross,
            "grad" => Token::Grad,
            "div" => Token::Div,
            "curl" => Token::Curl,
            "laplacian" => Token::Laplacian,
            "forall" => Token::ForAll,
            "exists" => Token::Exists,
            "union" => Token::Union,
            "intersect" => Token::Intersect,
            "in" => Token::In,
            "notin" => Token::NotIn,
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            "implies" => Token::Implies,
            "iff" => Token::Iff,
            _ => Token::Identifier(ident),
        };

        (token, span)
    }

    /// Scans a two-character operator: if `follow` matches the next char, return `long_tok`,
    /// otherwise return `short_tok`.
    fn scan_two_char_op(
        &mut self,
        start: Position,
        follow: char,
        short_tok: Token,
        long_tok: Token,
    ) -> SpannedToken {
        self.consume();
        if self.peek() == Some(follow) {
            self.consume();
            SpannedToken::new(long_tok, Span::new(start, self.position()))
        } else {
            SpannedToken::new(short_tok, Span::new(start, self.position()))
        }
    }

    /// Scans a single unicode character as a token.
    fn scan_unicode_token(&mut self, start: Position, tok: Token) -> SpannedToken {
        self.consume();
        SpannedToken::new(tok, Span::new(start, self.position()))
    }

    /// Scans multi-character operators and Unicode symbol tokens.
    fn scan_multi_char_op(
        &mut self,
        ch: char,
        start: Position,
    ) -> ParseResult<Option<SpannedToken>> {
        let token = match ch {
            '!' => self.scan_two_char_op(start, '=', Token::Bang, Token::NotEquals),
            '<' => self.scan_two_char_op(start, '=', Token::Less, Token::LessEq),
            '>' => self.scan_two_char_op(start, '=', Token::Greater, Token::GreaterEq),
            '*' => self.scan_two_char_op(start, '*', Token::Star, Token::DoubleStar),
            '≤' => self.scan_unicode_token(start, Token::LessEq),
            '≥' => self.scan_unicode_token(start, Token::GreaterEq),
            '≠' => self.scan_unicode_token(start, Token::NotEquals),
            'π' => self.scan_unicode_token(start, Token::Pi),
            '∞' => self.scan_unicode_token(start, Token::Infinity),
            '√' => self.scan_unicode_token(start, Token::Sqrt),
            _ => return Ok(None),
        };
        Ok(Some(token))
    }

    /// Scans the next token.
    fn scan_token(&mut self) -> ParseResult<Option<SpannedToken>> {
        self.skip_whitespace();

        let Some(ch) = self.peek() else {
            return Ok(None);
        };

        let start = self.position();

        if ch.is_ascii_digit() {
            let (token, span) = self.scan_number()?;
            return Ok(Some(SpannedToken::new(token, span)));
        }

        if ch.is_ascii_alphabetic() {
            let (token, span) = self.scan_identifier();
            return Ok(Some(SpannedToken::new(token, span)));
        }

        if let Some(tok) = self.scan_multi_char_op(ch, start)? {
            return Ok(Some(tok));
        }

        // Single-character tokens
        self.consume();
        let end = self.position();
        let span = Span::new(start, end);

        let token = match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
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
            '\'' => Token::Apostrophe,
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
    pub(super) fn tokenize_all(&mut self) -> ParseResult<Vec<SpannedToken>> {
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
