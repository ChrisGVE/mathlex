//! LatexTokenizer struct and core scanning methods.

use crate::error::{ParseError, ParseResult, Position, Span};

use super::token_types::LatexToken;

/// Tokenizer state with position tracking.
pub(super) struct Tokenizer<'a> {
    /// Input string
    pub(super) input: &'a str,
    /// Current byte offset
    pub(super) offset: usize,
    /// Current line number (1-indexed)
    pub(super) line: usize,
    /// Current column number (1-indexed)
    pub(super) column: usize,
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
    pub(super) fn position(&self) -> Position {
        Position::new(self.line, self.column, self.offset)
    }

    /// Peeks at the current character without consuming it.
    pub(super) fn peek(&self) -> Option<char> {
        self.input[self.offset..].chars().next()
    }

    /// Peeks at the character at the given offset ahead.
    pub(super) fn peek_ahead(&self, n: usize) -> Option<char> {
        self.input[self.offset..].chars().nth(n)
    }

    /// Consumes and returns the current character.
    pub(super) fn consume(&mut self) -> Option<char> {
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
    pub(super) fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.consume();
            } else {
                break;
            }
        }
    }

    /// Scans a command starting with backslash.
    pub(super) fn scan_command(&mut self) -> ParseResult<(String, Span)> {
        let start = self.position();

        // Consume the backslash
        self.consume();

        // Special case: check for double backslash
        if self.peek() == Some('\\') {
            self.consume();
            let end = self.position();
            return Ok(("\\\\".to_string(), Span::new(start, end)));
        }

        // Scan command name (letters only)
        let mut name = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphabetic() {
                name.push(ch);
                self.consume();
            } else {
                break;
            }
        }

        let end = self.position();

        if name.is_empty() {
            return Err(ParseError::invalid_latex_command(
                "\\",
                Some(Span::new(start, end)),
            ));
        }

        Ok((name, Span::new(start, end)))
    }

    /// Scans an environment name from \begin{name} or \end{name}.
    pub(super) fn scan_environment(&mut self, is_begin: bool) -> ParseResult<(String, Span)> {
        let start = self.position();

        // Skip whitespace before {
        self.skip_whitespace();

        if self.peek() != Some('{') {
            let cmd = if is_begin { "\\begin" } else { "\\end" };
            return Err(ParseError::custom(
                format!("{} must be followed by {{name}}", cmd),
                Some(Span::new(start, self.position())),
            ));
        }

        self.consume(); // consume {

        let mut name = String::new();
        while let Some(ch) = self.peek() {
            if ch == '}' {
                self.consume();
                let end = self.position();
                return Ok((name, Span::new(start, end)));
            } else if ch.is_ascii_alphanumeric() || ch == '*' {
                name.push(ch);
                self.consume();
            } else {
                return Err(ParseError::custom(
                    format!("invalid character '{}' in environment name", ch),
                    Some(Span::new(start, self.position())),
                ));
            }
        }

        Err(ParseError::unexpected_eof(
            vec!["closing brace"],
            Some(Span::new(start, self.position())),
        ))
    }

    /// Scans a number (integer or float).
    pub(super) fn scan_number(&mut self) -> (String, Span) {
        let start = self.position();
        let mut num = String::new();
        let mut has_decimal = false;

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num.push(ch);
                self.consume();
            } else if ch == '.' && !has_decimal {
                // Check if next char is a digit
                if let Some(next) = self.peek_ahead(1) {
                    if next.is_ascii_digit() {
                        has_decimal = true;
                        num.push(ch);
                        self.consume();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let end = self.position();
        (num, Span::new(start, end))
    }

    /// Tokenizes the next token from the input.
    pub(super) fn next_token(&mut self) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();

        let start = self.position();

        let ch = match self.peek() {
            Some(ch) => ch,
            None => return Ok((LatexToken::Eof, Span::at(start))),
        };

        if ch == '\\' {
            let (cmd, span) = self.scan_command()?;
            if cmd == "\\\\" {
                return Ok((LatexToken::DoubleBackslash, span));
            }
            return self.resolve_command(cmd, span, start);
        }

        // Single character tokens
        self.consume();
        let end = self.position();
        let span = Span::new(start, end);

        match ch {
            '+' => Ok((LatexToken::Plus, span)),
            '-' => Ok((LatexToken::Minus, span)),
            '*' => Ok((LatexToken::Star, span)),
            '/' => Ok((LatexToken::Slash, span)),
            '^' => Ok((LatexToken::Caret, span)),
            '_' => Ok((LatexToken::Underscore, span)),
            '=' => Ok((LatexToken::Equals, span)),
            '<' => Ok((LatexToken::Less, span)),
            '>' => Ok((LatexToken::Greater, span)),
            '{' => Ok((LatexToken::LBrace, span)),
            '}' => Ok((LatexToken::RBrace, span)),
            '(' => Ok((LatexToken::LParen, span)),
            ')' => Ok((LatexToken::RParen, span)),
            '[' => Ok((LatexToken::LBracket, span)),
            ']' => Ok((LatexToken::RBracket, span)),
            '|' => Ok((LatexToken::Pipe, span)),
            '&' => Ok((LatexToken::Ampersand, span)),
            ',' => Ok((LatexToken::Comma, span)),
            ':' => Ok((LatexToken::Colon, span)),
            _ if ch.is_ascii_digit() => {
                // Back up one character since we already consumed it
                self.offset -= 1;
                self.column -= 1;
                let (num, num_span) = self.scan_number();
                Ok((LatexToken::Number(num), num_span))
            }
            _ if ch.is_ascii_alphabetic() => Ok((LatexToken::Letter(ch), span)),
            _ => Err(ParseError::custom(
                format!("unexpected character '{}'", ch),
                Some(span),
            )),
        }
    }
}

/// Tokenizes a LaTeX mathematical expression.
///
/// # Arguments
///
/// * `input` - The LaTeX string to tokenize
///
/// # Returns
///
/// A vector of tokens with their source locations, or a parse error.
///
/// # Example
///
/// ```
/// use mathlex::parser::tokenize_latex;
///
/// let tokens = tokenize_latex(r"\frac{1}{2}").unwrap();
/// assert!(!tokens.is_empty());
/// ```
pub fn tokenize_latex(input: &str) -> ParseResult<Vec<(LatexToken, Span)>> {
    let mut tokenizer = Tokenizer::new(input);
    let mut tokens = Vec::new();

    loop {
        let (token, span) = tokenizer.next_token()?;

        if matches!(token, LatexToken::Eof) {
            tokens.push((token, span));
            break;
        }

        tokens.push((token, span));
    }

    Ok(tokens)
}
