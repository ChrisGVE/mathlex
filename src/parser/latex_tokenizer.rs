//! LaTeX tokenizer for mathematical expressions.
//!
//! This module provides tokenization of LaTeX mathematical notation into
//! a stream of tokens suitable for parsing.

use crate::error::{ParseError, ParseResult, Position, Span};

/// A LaTeX token representing a lexical element in LaTeX math mode.
#[derive(Debug, Clone, PartialEq)]
pub enum LatexToken {
    // Commands
    /// LaTeX command without backslash (e.g., "frac", "sin", "alpha")
    Command(String),

    // Literals
    /// Number literal (will be parsed as int or float later)
    Number(String),
    /// Single letter variable
    Letter(char),
    /// Explicit constant from \mathrm{e}, \mathrm{i}, \imath, \jmath
    ExplicitConstant(char),

    // Operators
    /// Plus operator (+)
    Plus,
    /// Minus operator (-)
    Minus,
    /// Multiplication operator (*)
    Star,
    /// Division operator (/)
    Slash,
    /// Exponentiation operator (^)
    Caret,
    /// Subscript operator (_)
    Underscore,
    /// Equals operator (=)
    Equals,
    /// Less than operator (<)
    Less,
    /// Greater than operator (>)
    Greater,

    // Delimiters
    /// Left brace ({)
    LBrace,
    /// Right brace (})
    RBrace,
    /// Left parenthesis (()
    LParen,
    /// Right parenthesis ())
    RParen,
    /// Left bracket ([)
    LBracket,
    /// Right bracket (])
    RBracket,
    /// Pipe (|) for absolute value
    Pipe,

    // Environment
    /// Begin environment (\begin{name})
    BeginEnv(String),
    /// End environment (\end{name})
    EndEnv(String),
    /// Ampersand (&) for column separator
    Ampersand,
    /// Double backslash (\\) for row separator
    DoubleBackslash,

    // Special
    /// Comma (,)
    Comma,
    /// Arrow (\to)
    To,
    /// Infinity (\infty)
    Infty,

    // Multiple integrals
    /// \iint - double integral
    DoubleIntegral,
    /// \iiint - triple integral
    TripleIntegral,
    /// \iiiint - quadruple integral (rare)
    QuadIntegral,

    // Closed integrals
    /// \oint - closed line integral
    ClosedIntegral,
    /// \oiint - closed surface integral
    ClosedSurface,
    /// \oiiint - closed volume integral
    ClosedVolume,

    // Quantifiers
    /// \forall - universal quantifier
    ForAll,
    /// \exists - existential quantifier
    Exists,

    // Logical connectives
    /// \land, \wedge - logical AND
    Land,
    /// \lor, \vee - logical OR
    Lor,
    /// \lnot, \neg - logical NOT
    Lnot,
    /// \implies, \Rightarrow - implication
    Implies,
    /// \iff, \Leftrightarrow - if and only if
    Iff,

    // Membership
    /// \in - element of
    In,
    /// \notin - not element of
    NotIn,


    // Number sets (via \mathbb{X})
    /// \mathbb{N} - natural numbers
    Naturals,
    /// \mathbb{Z} - integers
    Integers,
    /// \mathbb{Q} - rationals
    Rationals,
    /// \mathbb{R} - reals
    Reals,
    /// \mathbb{C} - complex numbers
    Complexes,
    /// \mathbb{H} - quaternions
    Quaternions,

    // Set operations
    /// \cup - union
    Cup,
    /// \cap - intersection
    Cap,
    /// \setminus - set difference
    Setminus,
    /// \triangle, \bigtriangleup - symmetric difference
    Triangle,

    // Set relations
    /// \subset - proper subset
    Subset,
    /// \subseteq - subset or equal
    SubsetEq,
    /// \supset - proper superset
    Superset,
    /// \supseteq - superset or equal
    SupersetEq,

    // Set notation
    /// \emptyset, \varnothing - empty set
    EmptySet,
    /// \mid - set builder separator
    SetMid,
    /// \mathcal{P} - power set
    PowerSet,

    // Vector notation
    /// \mathbf{...} - bold vector notation
    Mathbf,
    /// \boldsymbol{...} - bold symbol
    Boldsymbol,
    /// \vec{...} - arrow vector notation
    Vec,
    /// \overrightarrow{...} - long arrow
    Overrightarrow,
    /// \hat{...} - unit vector (hat)
    Hat,
    /// \underline{...} - underline notation
    Underline,

    // Vector/tensor operations
    /// \cdot used as dot product operator
    Cdot,
    /// \bullet - alternative dot
    Bullet,
    /// \otimes - tensor/outer product
    Otimes,
    /// \wedge - wedge product
    Wedge,

    // Nabla
    /// \nabla - del/nabla operator
    Nabla,

    /// End of file marker
    Eof,
}

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

    /// Scans a command starting with backslash.
    fn scan_command(&mut self) -> ParseResult<(String, Span)> {
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
    fn scan_environment(&mut self, is_begin: bool) -> ParseResult<(String, Span)> {
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
    fn scan_number(&mut self) -> (String, Span) {
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
    fn next_token(&mut self) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();

        let start = self.position();

        let ch = match self.peek() {
            Some(ch) => ch,
            None => return Ok((LatexToken::Eof, Span::at(start))),
        };

        // Check for backslash (command or environment)
        if ch == '\\' {
            let (cmd, span) = self.scan_command()?;

            // Special case: double backslash
            if cmd == "\\\\" {
                return Ok((LatexToken::DoubleBackslash, span));
            }

            // Check for special commands
            return match cmd.as_str() {
                "begin" => {
                    let (name, span) = self.scan_environment(true)?;
                    Ok((LatexToken::BeginEnv(name), span))
                }
                "end" => {
                    let (name, span) = self.scan_environment(false)?;
                    Ok((LatexToken::EndEnv(name), span))
                }
                "to" => Ok((LatexToken::To, span)),
                "infty" => Ok((LatexToken::Infty, span)),
                "cdot" | "times" => Ok((LatexToken::Star, span)),
                "mathrm" => {
                    // Parse \mathrm{e} or \mathrm{i} as explicit constants
                    self.skip_whitespace();
                    if self.peek() != Some('{') {
                        return Err(ParseError::custom(
                            "\\mathrm must be followed by {content}".to_string(),
                            Some(span),
                        ));
                    }
                    self.consume(); // consume {
                    let ch = self.peek();
                    if ch.is_none() {
                        return Err(ParseError::unexpected_eof(vec!["letter"], Some(span)));
                    }
                    let ch = ch.unwrap();
                    self.consume(); // consume letter
                    if self.peek() != Some('}') {
                        return Err(ParseError::custom(
                            "\\mathrm{} must contain exactly one character for constant notation"
                                .to_string(),
                            Some(span),
                        ));
                    }
                    self.consume(); // consume }
                    let end = self.position();
                    match ch {
                        'e' | 'i' | 'j' | 'k' => {
                            Ok((LatexToken::ExplicitConstant(ch), Span::new(start, end)))
                        }
                        _ => Ok((
                            LatexToken::Command(format!("mathrm_{}", ch)),
                            Span::new(start, end),
                        )),
                    }
                }
                "imath" | "jmath" => Ok((LatexToken::ExplicitConstant('i'), span)),
                "left" => {
                    // Handle \left( or \left[ or \left|
                    self.skip_whitespace();
                    match self.peek() {
                        Some('(') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::LParen, Span::new(start, end)))
                        }
                        Some('[') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::LBracket, Span::new(start, end)))
                        }
                        Some('|') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::Pipe, Span::new(start, end)))
                        }
                        _ => Ok((LatexToken::Command("left".to_string()), span)),
                    }
                }
                "right" => {
                    // Handle \right) or \right] or \right|
                    self.skip_whitespace();
                    match self.peek() {
                        Some(')') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::RParen, Span::new(start, end)))
                        }
                        Some(']') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::RBracket, Span::new(start, end)))
                        }
                        Some('|') => {
                            self.consume();
                            let end = self.position();
                            Ok((LatexToken::Pipe, Span::new(start, end)))
                        }
                        _ => Ok((LatexToken::Command("right".to_string()), span)),
                    }
                }

                // Quantifiers
                "forall" => Ok((LatexToken::ForAll, span)),
                "exists" => Ok((LatexToken::Exists, span)),

                // Logical connectives (handle aliases)
                "land" | "wedge" => Ok((LatexToken::Land, span)),
                "lor" | "vee" => Ok((LatexToken::Lor, span)),
                "lnot" | "neg" => Ok((LatexToken::Lnot, span)),
                "implies" | "Rightarrow" => Ok((LatexToken::Implies, span)),
                "iff" | "Leftrightarrow" => Ok((LatexToken::Iff, span)),

                // Set membership
                "in" => Ok((LatexToken::In, span)),
                "notin" => Ok((LatexToken::NotIn, span)),

                // Multiple integrals
                "iint" => Ok((LatexToken::DoubleIntegral, span)),
                "iiint" => Ok((LatexToken::TripleIntegral, span)),
                "iiiint" => Ok((LatexToken::QuadIntegral, span)),

                // Closed integrals
                "oint" => Ok((LatexToken::ClosedIntegral, span)),
                "oiint" => Ok((LatexToken::ClosedSurface, span)),
                "oiiint" => Ok((LatexToken::ClosedVolume, span)),


                // Set operations
                "cup" => Ok((LatexToken::Cup, span)),
                "cap" => Ok((LatexToken::Cap, span)),
                "setminus" => Ok((LatexToken::Setminus, span)),
                "triangle" | "bigtriangleup" => Ok((LatexToken::Triangle, span)),

                // Set relations
                "subset" => Ok((LatexToken::Subset, span)),
                "subseteq" => Ok((LatexToken::SubsetEq, span)),
                "supset" => Ok((LatexToken::Superset, span)),
                "supseteq" => Ok((LatexToken::SupersetEq, span)),

                // Set notation
                "emptyset" | "varnothing" => Ok((LatexToken::EmptySet, span)),
                "mid" => Ok((LatexToken::SetMid, span)),

                // \mathbb{X} - number sets
                "mathbb" => {
                    self.skip_whitespace();
                    if self.peek() != Some('{') {
                        return Err(ParseError::custom(
                            "\\mathbb must be followed by {letter}".to_string(),
                            Some(span),
                        ));
                    }
                    self.consume(); // consume {
                    let ch = self.peek();
                    if ch.is_none() {
                        return Err(ParseError::unexpected_eof(vec!["letter"], Some(span)));
                    }
                    let ch = ch.unwrap();
                    self.consume(); // consume letter
                    if self.peek() != Some('}') {
                        return Err(ParseError::custom(
                            "\\mathbb{} must contain exactly one character".to_string(),
                            Some(span),
                        ));
                    }
                    self.consume(); // consume }
                    let end = self.position();
                    match ch {
                        'N' => Ok((LatexToken::Naturals, Span::new(start, end))),
                        'Z' => Ok((LatexToken::Integers, Span::new(start, end))),
                        'Q' => Ok((LatexToken::Rationals, Span::new(start, end))),
                        'R' => Ok((LatexToken::Reals, Span::new(start, end))),
                        'C' => Ok((LatexToken::Complexes, Span::new(start, end))),
                        'H' => Ok((LatexToken::Quaternions, Span::new(start, end))),
                        _ => Ok((LatexToken::Command(format!("mathbb_{}", ch)), Span::new(start, end))),
                    }
                }

                // \mathcal{P} - power set
                "mathcal" => {
                    self.skip_whitespace();
                    if self.peek() != Some('{') {
                        return Ok((LatexToken::Command("mathcal".to_string()), span));
                    }
                    self.consume(); // consume {
                    let ch = self.peek();
                    if ch.is_none() {
                        return Err(ParseError::unexpected_eof(vec!["letter"], Some(span)));
                    }
                    let ch = ch.unwrap();
                    self.consume(); // consume letter
                    if self.peek() != Some('}') {
                        return Err(ParseError::custom(
                            "\\mathcal{} must contain exactly one character".to_string(),
                            Some(span),
                        ));
                    }
                    self.consume(); // consume }
                    let end = self.position();
                    match ch {
                        'P' => Ok((LatexToken::PowerSet, Span::new(start, end))),
                        _ => Ok((LatexToken::Command(format!("mathcal_{}", ch)), Span::new(start, end))),
                    }
                }
                _ => Ok((LatexToken::Command(cmd), span)),
            };
        }

        // Single character tokens
        self.consume();
        let end = self.position();
        let span = Span::new(start, end);

        match ch {
            // Operators
            '+' => Ok((LatexToken::Plus, span)),
            '-' => Ok((LatexToken::Minus, span)),
            '*' => Ok((LatexToken::Star, span)),
            '/' => Ok((LatexToken::Slash, span)),
            '^' => Ok((LatexToken::Caret, span)),
            '_' => Ok((LatexToken::Underscore, span)),
            '=' => Ok((LatexToken::Equals, span)),
            '<' => Ok((LatexToken::Less, span)),
            '>' => Ok((LatexToken::Greater, span)),

            // Delimiters
            '{' => Ok((LatexToken::LBrace, span)),
            '}' => Ok((LatexToken::RBrace, span)),
            '(' => Ok((LatexToken::LParen, span)),
            ')' => Ok((LatexToken::RParen, span)),
            '[' => Ok((LatexToken::LBracket, span)),
            ']' => Ok((LatexToken::RBracket, span)),
            '|' => Ok((LatexToken::Pipe, span)),

            // Special
            '&' => Ok((LatexToken::Ampersand, span)),
            ',' => Ok((LatexToken::Comma, span)),

            // Numbers and letters
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize_latex("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].0, LatexToken::Eof));
    }

    #[test]
    fn test_tokenize_simple_number() {
        let tokens = tokenize_latex("42").unwrap();
        assert_eq!(tokens.len(), 2); // number + eof
        assert_eq!(tokens[0].0, LatexToken::Number("42".to_string()));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize_latex("3.14").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Number("3.14".to_string()));
    }

    #[test]
    fn test_tokenize_letter() {
        let tokens = tokenize_latex("x").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
    }

    #[test]
    fn test_tokenize_operators() {
        let input = "+ - * / ^ _ = < >";
        let tokens = tokenize_latex(input).unwrap();

        assert!(matches!(tokens[0].0, LatexToken::Plus));
        assert!(matches!(tokens[1].0, LatexToken::Minus));
        assert!(matches!(tokens[2].0, LatexToken::Star));
        assert!(matches!(tokens[3].0, LatexToken::Slash));
        assert!(matches!(tokens[4].0, LatexToken::Caret));
        assert!(matches!(tokens[5].0, LatexToken::Underscore));
        assert!(matches!(tokens[6].0, LatexToken::Equals));
        assert!(matches!(tokens[7].0, LatexToken::Less));
        assert!(matches!(tokens[8].0, LatexToken::Greater));
    }

    #[test]
    fn test_tokenize_delimiters() {
        let input = "{ } ( ) [ ] |";
        let tokens = tokenize_latex(input).unwrap();

        assert!(matches!(tokens[0].0, LatexToken::LBrace));
        assert!(matches!(tokens[1].0, LatexToken::RBrace));
        assert!(matches!(tokens[2].0, LatexToken::LParen));
        assert!(matches!(tokens[3].0, LatexToken::RParen));
        assert!(matches!(tokens[4].0, LatexToken::LBracket));
        assert!(matches!(tokens[5].0, LatexToken::RBracket));
        assert!(matches!(tokens[6].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_command() {
        let tokens = tokenize_latex(r"\frac").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("frac".to_string()));
    }

    #[test]
    fn test_tokenize_greek_letter() {
        let tokens = tokenize_latex(r"\alpha").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("alpha".to_string()));
    }

    #[test]
    fn test_tokenize_special_to() {
        let tokens = tokenize_latex(r"\to").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::To));
    }

    #[test]
    fn test_tokenize_special_infty() {
        let tokens = tokenize_latex(r"\infty").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Infty));
    }

    #[test]
    fn test_tokenize_double_backslash() {
        let tokens = tokenize_latex(r"\\").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::DoubleBackslash));
    }

    #[test]
    fn test_tokenize_begin_env() {
        let tokens = tokenize_latex(r"\begin{matrix}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::BeginEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_end_env() {
        let tokens = tokenize_latex(r"\end{matrix}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::EndEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_ampersand() {
        let tokens = tokenize_latex("&").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Ampersand));
    }

    #[test]
    fn test_tokenize_comma() {
        let tokens = tokenize_latex(",").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Comma));
    }

    #[test]
    fn test_tokenize_left_paren() {
        let tokens = tokenize_latex(r"\left(").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::LParen));
    }

    #[test]
    fn test_tokenize_right_paren() {
        let tokens = tokenize_latex(r"\right)").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::RParen));
    }

    #[test]
    fn test_tokenize_left_bracket() {
        let tokens = tokenize_latex(r"\left[").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::LBracket));
    }

    #[test]
    fn test_tokenize_right_bracket() {
        let tokens = tokenize_latex(r"\right]").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::RBracket));
    }

    #[test]
    fn test_tokenize_left_pipe() {
        let tokens = tokenize_latex(r"\left|").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_right_pipe() {
        let tokens = tokenize_latex(r"\right|").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_frac_expression() {
        let tokens = tokenize_latex(r"\frac{1}{2}").unwrap();
        // \frac { 1 } { 2 } eof = 8 tokens
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0].0, LatexToken::Command("frac".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LBrace));
        assert_eq!(tokens[2].0, LatexToken::Number("1".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::RBrace));
        assert!(matches!(tokens[4].0, LatexToken::LBrace));
        assert_eq!(tokens[5].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::RBrace));
        assert!(matches!(tokens[7].0, LatexToken::Eof));
    }

    #[test]
    fn test_tokenize_superscript() {
        let tokens = tokenize_latex("x^2").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[1].0, LatexToken::Caret));
        assert_eq!(tokens[2].0, LatexToken::Number("2".to_string()));
    }

    #[test]
    fn test_tokenize_subscript() {
        let tokens = tokenize_latex("x_i").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[1].0, LatexToken::Underscore));
        assert_eq!(tokens[2].0, LatexToken::Letter('i'));
    }

    #[test]
    fn test_tokenize_complex_expression() {
        let tokens = tokenize_latex(r"\sin(x) + \cos(y)").unwrap();

        // \sin ( x ) + \cos ( y ) eof
        assert_eq!(tokens[0].0, LatexToken::Command("sin".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LParen));
        assert_eq!(tokens[2].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[3].0, LatexToken::RParen));
        assert!(matches!(tokens[4].0, LatexToken::Plus));
        assert_eq!(tokens[5].0, LatexToken::Command("cos".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::LParen));
        assert_eq!(tokens[7].0, LatexToken::Letter('y'));
        assert!(matches!(tokens[8].0, LatexToken::RParen));
    }

    #[test]
    fn test_tokenize_matrix() {
        let input = r"\begin{matrix}
            1 & 2 \\
            3 & 4
        \end{matrix}";

        let tokens = tokenize_latex(input).unwrap();

        assert_eq!(tokens[0].0, LatexToken::BeginEnv("matrix".to_string()));
        assert_eq!(tokens[1].0, LatexToken::Number("1".to_string()));
        assert!(matches!(tokens[2].0, LatexToken::Ampersand));
        assert_eq!(tokens[3].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[4].0, LatexToken::DoubleBackslash));
        assert_eq!(tokens[5].0, LatexToken::Number("3".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::Ampersand));
        assert_eq!(tokens[7].0, LatexToken::Number("4".to_string()));
        assert_eq!(tokens[8].0, LatexToken::EndEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_whitespace_ignored() {
        let tokens1 = tokenize_latex("x+y").unwrap();
        let tokens2 = tokenize_latex("x + y").unwrap();
        let tokens3 = tokenize_latex("  x  +  y  ").unwrap();

        // All should produce the same tokens (ignoring spans)
        assert_eq!(tokens1.len(), tokens2.len());
        assert_eq!(tokens1.len(), tokens3.len());

        for i in 0..tokens1.len() {
            assert_eq!(tokens1[i].0, tokens2[i].0);
            assert_eq!(tokens1[i].0, tokens3[i].0);
        }
    }

    #[test]
    fn test_tokenize_position_tracking() {
        let tokens = tokenize_latex("x+y").unwrap();

        // Check first token (x)
        assert_eq!(tokens[0].1.start.line, 1);
        assert_eq!(tokens[0].1.start.column, 1);
        assert_eq!(tokens[0].1.start.offset, 0);

        // Check second token (+)
        assert_eq!(tokens[1].1.start.line, 1);
        assert_eq!(tokens[1].1.start.column, 2);
        assert_eq!(tokens[1].1.start.offset, 1);

        // Check third token (y)
        assert_eq!(tokens[2].1.start.line, 1);
        assert_eq!(tokens[2].1.start.column, 3);
        assert_eq!(tokens[2].1.start.offset, 2);
    }

    #[test]
    fn test_tokenize_multiline_position_tracking() {
        let tokens = tokenize_latex("x\n+\ny").unwrap();

        // Check first token (x) - line 1
        assert_eq!(tokens[0].1.start.line, 1);

        // Check second token (+) - line 2
        assert_eq!(tokens[1].1.start.line, 2);
        assert_eq!(tokens[1].1.start.column, 1);

        // Check third token (y) - line 3
        assert_eq!(tokens[2].1.start.line, 3);
        assert_eq!(tokens[2].1.start.column, 1);
    }

    #[test]
    fn test_error_invalid_command() {
        let result = tokenize_latex(r"\");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_begin_without_brace() {
        let result = tokenize_latex(r"\begin");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_begin_unclosed() {
        let result = tokenize_latex(r"\begin{matrix");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unexpected_character() {
        let result = tokenize_latex("@");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenize_multiple_digits() {
        let tokens = tokenize_latex("123").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Number("123".to_string()));
    }

    #[test]
    fn test_tokenize_float_at_end() {
        // "3." is not valid LaTeX math - trailing dot without digits
        // The tokenizer returns an error for invalid characters
        let result = tokenize_latex("3.");
        // This is expected to fail - "3." is not valid
        assert!(
            result.is_err() || {
                // Or if it succeeds, check tokens
                let tokens = result.unwrap();
                tokens.len() >= 2 && tokens[0].0 == LatexToken::Number("3".to_string())
            }
        );
    }

    #[test]
    fn test_tokenize_integral() {
        let tokens = tokenize_latex(r"\int_0^\infty").unwrap();
        assert_eq!(tokens[0].0, LatexToken::Command("int".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::Underscore));
        assert_eq!(tokens[2].0, LatexToken::Number("0".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::Caret));
        assert!(matches!(tokens[4].0, LatexToken::Infty));
    }

    #[test]
    fn test_tokenize_sqrt() {
        let tokens = tokenize_latex(r"\sqrt{2}").unwrap();
        assert_eq!(tokens[0].0, LatexToken::Command("sqrt".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LBrace));
        assert_eq!(tokens[2].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::RBrace));
    }

    #[test]
    fn test_tokenize_cdot() {
        let tokens = tokenize_latex(r"\cdot").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Star));
    }

    #[test]
    fn test_tokenize_times() {
        let tokens = tokenize_latex(r"\times").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Star));
    }

    #[test]
    fn test_tokenize_cdot_multiplication() {
        let tokens = tokenize_latex(r"a \cdot b").unwrap();
        assert_eq!(tokens.len(), 4); // a, \cdot, b, eof
        assert_eq!(tokens[0].0, LatexToken::Letter('a'));
        assert!(matches!(tokens[1].0, LatexToken::Star));
        assert_eq!(tokens[2].0, LatexToken::Letter('b'));
    }

    #[test]
    fn test_tokenize_times_multiplication() {
        let tokens = tokenize_latex(r"2 \times 3").unwrap();
        assert_eq!(tokens.len(), 4); // 2, \times, 3, eof
        assert_eq!(tokens[0].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::Star));
        assert_eq!(tokens[2].0, LatexToken::Number("3".to_string()));
    }

    #[test]
    fn test_tokenize_mathrm_j() {
        let tokens = tokenize_latex(r"\mathrm{j}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('j'));
    }

    #[test]
    fn test_tokenize_mathrm_k() {
        let tokens = tokenize_latex(r"\mathrm{k}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('k'));
    }

    #[test]
    fn test_bare_j_is_letter() {
        let tokens = tokenize_latex("j").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('j'));
    }

    #[test]
    fn test_bare_k_is_letter() {
        let tokens = tokenize_latex("k").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('k'));
    }

    #[test]
    fn test_mathrm_i_still_works() {
        let tokens = tokenize_latex(r"\mathrm{i}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('i'));
    }

    #[test]
    fn test_mathrm_e_still_works() {
        let tokens = tokenize_latex(r"\mathrm{e}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('e'));
    }

    #[test]
    fn test_tokenize_forall() {
        let tokens = tokenize_latex(r"\forall").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ForAll));
    }

    #[test]
    fn test_tokenize_exists() {
        let tokens = tokenize_latex(r"\exists").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Exists));
    }

    #[test]
    fn test_tokenize_land() {
        let tokens = tokenize_latex(r"\land").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Land));
    }

    #[test]
    fn test_tokenize_lor() {
        let tokens = tokenize_latex(r"\lor").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lor));
    }

    #[test]
    fn test_tokenize_lnot() {
        let tokens = tokenize_latex(r"\lnot").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lnot));
    }

    #[test]
    fn test_tokenize_neg_alias() {
        let tokens = tokenize_latex(r"\neg").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lnot));
    }

    #[test]
    fn test_tokenize_implies() {
        let tokens = tokenize_latex(r"\implies").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Implies));
    }

    #[test]
    fn test_tokenize_iff() {
        let tokens = tokenize_latex(r"\iff").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Iff));
    }

    #[test]
    fn test_tokenize_in() {
        let tokens = tokenize_latex(r"\in").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::In));
    }

    #[test]
    fn test_tokenize_notin() {
        let tokens = tokenize_latex(r"\notin").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::NotIn));
    }

    #[test]
    fn test_tokenize_double_integral() {
        let tokens = tokenize_latex(r"\iint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::DoubleIntegral));
    }

    #[test]
    fn test_tokenize_triple_integral() {
        let tokens = tokenize_latex(r"\iiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::TripleIntegral));
    }

    #[test]
    fn test_tokenize_quad_integral() {
        let tokens = tokenize_latex(r"\iiiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::QuadIntegral));
    }

    #[test]
    fn test_tokenize_closed_integral() {
        let tokens = tokenize_latex(r"\oint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedIntegral));
    }

    #[test]
    fn test_tokenize_closed_surface() {
        let tokens = tokenize_latex(r"\oiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedSurface));
    }

    #[test]
    fn test_tokenize_closed_volume() {
        let tokens = tokenize_latex(r"\oiiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedVolume));
    }

    #[test]
    fn test_tokenize_single_int_still_works() {
        // Verify \int still works as before
        let tokens = tokenize_latex(r"\int").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("int".to_string()));
    }
}
