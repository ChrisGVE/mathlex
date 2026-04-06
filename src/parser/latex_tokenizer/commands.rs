//! Command resolution: resolve_command and \mathXX sub-helpers.

use crate::error::{ParseError, ParseResult, Span};

use super::scanner::Tokenizer;
use super::token_types::LatexToken;

impl<'a> Tokenizer<'a> {
    /// Scans a braced word, returning its string content.
    ///
    /// Expects the next character to be `{`, reads letters until `}`, and
    /// returns the accumulated word.  Errors if the brace or closing brace is
    /// absent, or if a non-letter character is found inside.
    fn scan_braced_word(&mut self, cmd_span: Span, cmd_name: &str) -> ParseResult<(String, Span)> {
        let token_start = self.position();
        self.skip_whitespace();
        if self.peek() != Some('{') {
            return Err(ParseError::custom(
                format!("\\{} must be followed by {{content}}", cmd_name),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume {
        let mut word = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(ParseError::unexpected_eof(
                        vec!["letter or }"],
                        Some(cmd_span),
                    ));
                }
                Some('}') => {
                    self.consume(); // consume }
                    break;
                }
                Some(ch) if ch.is_ascii_alphabetic() => {
                    word.push(ch);
                    self.consume();
                }
                Some(ch) => {
                    return Err(ParseError::custom(
                        format!(
                            "\\{} content must contain only letters, got '{}'",
                            cmd_name, ch
                        ),
                        Some(cmd_span),
                    ));
                }
            }
        }
        let end = self.position();
        Ok((word, Span::new(token_start, end)))
    }

    /// Parses `\text{word}` and returns a `NaNConstant` token when the word is
    /// `NaN` or `nan`, or a generic `Command` token otherwise.
    pub(super) fn scan_text(
        &mut self,
        cmd_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        let (word, word_span) = self.scan_braced_word(cmd_span, "text")?;
        let end = word_span.end;
        let token = match word.as_str() {
            "NaN" | "nan" => LatexToken::NaNConstant,
            _ => LatexToken::Command(format!("text_{}", word)),
        };
        Ok((token, Span::new(token_start, end)))
    }

    /// Parses `\mathrm{X}` and returns an explicit constant or fallback command.
    pub(super) fn scan_mathrm(
        &mut self,
        cmd_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();
        if self.peek() != Some('{') {
            return Err(ParseError::custom(
                "\\mathrm must be followed by {content}".to_string(),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume {

        // Peek ahead to detect multi-character words (e.g. "NaN") vs single chars.
        let first_ch = self
            .peek()
            .ok_or_else(|| ParseError::unexpected_eof(vec!["letter"], Some(cmd_span)))?;

        // Check whether the content is a multi-character word.
        let second_ch = self.peek_ahead(1);
        if second_ch.is_some() && second_ch != Some('}') {
            // Multi-character content — read the full word.
            let (word, word_span) = {
                let mut w = String::new();
                loop {
                    match self.peek() {
                        Some('}') | None => break,
                        Some(ch) if ch.is_ascii_alphabetic() => {
                            w.push(ch);
                            self.consume();
                        }
                        Some(ch) => {
                            return Err(ParseError::custom(
                                format!(
                                    "\\mathrm{{}} content must contain only letters, got '{}'",
                                    ch
                                ),
                                Some(cmd_span),
                            ));
                        }
                    }
                }
                let end = self.position();
                (w, end)
            };
            if self.peek() != Some('}') {
                return Err(ParseError::custom(
                    "\\mathrm{} missing closing brace".to_string(),
                    Some(cmd_span),
                ));
            }
            self.consume(); // consume }
            let end = self.position();
            return match word.as_str() {
                "NaN" | "nan" => Ok((LatexToken::NaNConstant, Span::new(token_start, end))),
                _ => Ok((
                    LatexToken::Command(format!("mathrm_{}", word)),
                    Span::new(token_start, word_span),
                )),
            };
        }

        // Single-character content (original behaviour).
        let ch = first_ch;
        self.consume(); // consume letter
        if self.peek() != Some('}') {
            return Err(ParseError::custom(
                "\\mathrm{} must contain exactly one character for constant notation".to_string(),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume }
        let end = self.position();
        match ch {
            'e' | 'i' | 'j' | 'k' => Ok((
                LatexToken::ExplicitConstant(ch),
                Span::new(token_start, end),
            )),
            _ => Ok((
                LatexToken::Command(format!("mathrm_{}", ch)),
                Span::new(token_start, end),
            )),
        }
    }

    /// Parses `\mathbb{X}` and returns the matching number-set token.
    pub(super) fn scan_mathbb(
        &mut self,
        cmd_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();
        if self.peek() != Some('{') {
            return Err(ParseError::custom(
                "\\mathbb must be followed by {letter}".to_string(),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume {
        let ch = self
            .peek()
            .ok_or_else(|| ParseError::unexpected_eof(vec!["letter"], Some(cmd_span)))?;
        self.consume(); // consume letter
        if self.peek() != Some('}') {
            return Err(ParseError::custom(
                "\\mathbb{} must contain exactly one character".to_string(),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume }
        let end = self.position();
        let tok = match ch {
            'N' => LatexToken::Naturals,
            'Z' => LatexToken::Integers,
            'Q' => LatexToken::Rationals,
            'R' => LatexToken::Reals,
            'C' => LatexToken::Complexes,
            'H' => LatexToken::Quaternions,
            _ => LatexToken::Command(format!("mathbb_{}", ch)),
        };
        Ok((tok, Span::new(token_start, end)))
    }

    /// Parses `\mathcal{X}` and returns a power-set token or fallback command.
    pub(super) fn scan_mathcal(
        &mut self,
        cmd_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();
        if self.peek() != Some('{') {
            return Ok((LatexToken::Command("mathcal".to_string()), cmd_span));
        }
        self.consume(); // consume {
        let ch = self
            .peek()
            .ok_or_else(|| ParseError::unexpected_eof(vec!["letter"], Some(cmd_span)))?;
        self.consume(); // consume letter
        if self.peek() != Some('}') {
            return Err(ParseError::custom(
                "\\mathcal{} must contain exactly one character".to_string(),
                Some(cmd_span),
            ));
        }
        self.consume(); // consume }
        let end = self.position();
        let tok = match ch {
            'P' => LatexToken::PowerSet,
            _ => LatexToken::Command(format!("mathcal_{}", ch)),
        };
        Ok((tok, Span::new(token_start, end)))
    }

    /// Parses the bracket after `\left` into the appropriate delimiter token.
    pub(super) fn scan_left_delimiter(
        &mut self,
        fallback_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();
        match self.peek() {
            Some('(') => {
                self.consume();
                Ok((LatexToken::LParen, Span::new(token_start, self.position())))
            }
            Some('[') => {
                self.consume();
                Ok((
                    LatexToken::LBracket,
                    Span::new(token_start, self.position()),
                ))
            }
            Some('|') => {
                self.consume();
                Ok((LatexToken::Pipe, Span::new(token_start, self.position())))
            }
            _ => Ok((LatexToken::Command("left".to_string()), fallback_span)),
        }
    }

    /// Parses the bracket after `\right` into the appropriate delimiter token.
    pub(super) fn scan_right_delimiter(
        &mut self,
        fallback_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        self.skip_whitespace();
        match self.peek() {
            Some(')') => {
                self.consume();
                Ok((LatexToken::RParen, Span::new(token_start, self.position())))
            }
            Some(']') => {
                self.consume();
                Ok((
                    LatexToken::RBracket,
                    Span::new(token_start, self.position()),
                ))
            }
            Some('|') => {
                self.consume();
                Ok((LatexToken::Pipe, Span::new(token_start, self.position())))
            }
            _ => Ok((LatexToken::Command("right".to_string()), fallback_span)),
        }
    }

    /// Resolves a scanned command name to its `LatexToken`.
    ///
    /// `cmd_span` covers the full `\name` span; `token_start` is the backslash position.
    pub(super) fn resolve_command(
        &mut self,
        cmd: String,
        cmd_span: Span,
        token_start: crate::error::Position,
    ) -> ParseResult<(LatexToken, Span)> {
        match cmd.as_str() {
            "begin" => {
                let (name, span) = self.scan_environment(true)?;
                Ok((LatexToken::BeginEnv(name), span))
            }
            "end" => {
                let (name, span) = self.scan_environment(false)?;
                Ok((LatexToken::EndEnv(name), span))
            }
            "to" => Ok((LatexToken::To, cmd_span)),
            "infty" => Ok((LatexToken::Infty, cmd_span)),
            "cdot" => Ok((LatexToken::Cdot, cmd_span)),
            "times" => Ok((LatexToken::Cross, cmd_span)),
            "mathrm" => self.scan_mathrm(cmd_span, token_start),
            "text" => self.scan_text(cmd_span, token_start),
            "imath" | "jmath" => Ok((LatexToken::ExplicitConstant('i'), cmd_span)),
            "left" => self.scan_left_delimiter(cmd_span, token_start),
            "right" => self.scan_right_delimiter(cmd_span, token_start),
            "forall" => Ok((LatexToken::ForAll, cmd_span)),
            "exists" => Ok((LatexToken::Exists, cmd_span)),
            "land" => Ok((LatexToken::Land, cmd_span)),
            "lor" | "vee" => Ok((LatexToken::Lor, cmd_span)),
            "lnot" | "neg" => Ok((LatexToken::Lnot, cmd_span)),
            "implies" | "Rightarrow" => Ok((LatexToken::Implies, cmd_span)),
            "iff" | "Leftrightarrow" => Ok((LatexToken::Iff, cmd_span)),
            "in" => Ok((LatexToken::In, cmd_span)),
            "notin" => Ok((LatexToken::NotIn, cmd_span)),
            "iint" => Ok((LatexToken::DoubleIntegral, cmd_span)),
            "iiint" => Ok((LatexToken::TripleIntegral, cmd_span)),
            "iiiint" => Ok((LatexToken::QuadIntegral, cmd_span)),
            "oint" => Ok((LatexToken::ClosedIntegral, cmd_span)),
            "oiint" => Ok((LatexToken::ClosedSurface, cmd_span)),
            "oiiint" => Ok((LatexToken::ClosedVolume, cmd_span)),
            "cup" => Ok((LatexToken::Cup, cmd_span)),
            "cap" => Ok((LatexToken::Cap, cmd_span)),
            "setminus" => Ok((LatexToken::Setminus, cmd_span)),
            "triangle" | "bigtriangleup" => Ok((LatexToken::Triangle, cmd_span)),
            "subset" => Ok((LatexToken::Subset, cmd_span)),
            "subseteq" => Ok((LatexToken::SubsetEq, cmd_span)),
            "supset" => Ok((LatexToken::Superset, cmd_span)),
            "supseteq" => Ok((LatexToken::SupersetEq, cmd_span)),
            "emptyset" | "varnothing" => Ok((LatexToken::EmptySet, cmd_span)),
            "mid" => Ok((LatexToken::SetMid, cmd_span)),
            "mathbb" => self.scan_mathbb(cmd_span, token_start),
            "mathcal" => self.scan_mathcal(cmd_span, token_start),
            "mathbf" => Ok((LatexToken::Mathbf, cmd_span)),
            "boldsymbol" => Ok((LatexToken::Boldsymbol, cmd_span)),
            "vec" => Ok((LatexToken::Vec, cmd_span)),
            "overrightarrow" => Ok((LatexToken::Overrightarrow, cmd_span)),
            "hat" => Ok((LatexToken::Hat, cmd_span)),
            "underline" => Ok((LatexToken::Underline, cmd_span)),
            "bullet" => Ok((LatexToken::Bullet, cmd_span)),
            "otimes" => Ok((LatexToken::Otimes, cmd_span)),
            "wedge" => Ok((LatexToken::Wedge, cmd_span)),
            "nabla" => Ok((LatexToken::Nabla, cmd_span)),
            "sim" => Ok((LatexToken::Sim, cmd_span)),
            "equiv" => Ok((LatexToken::Equiv, cmd_span)),
            "cong" => Ok((LatexToken::Cong, cmd_span)),
            "approx" => Ok((LatexToken::Approx, cmd_span)),
            "circ" => Ok((LatexToken::Circ, cmd_span)),
            _ => Ok((LatexToken::Command(cmd), cmd_span)),
        }
    }
}
