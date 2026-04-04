// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses a letter token, handling differential detection and e/i constant resolution.
    pub(super) fn parse_letter_token(&mut self, ch: char, _span: Span) -> ParseResult<Expression> {
        // Detect `d<var>` as a differential outside integral/fraction context
        if ch == 'd' && !self.in_integral_context && !self.in_fraction_context {
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek_ahead(1) {
                let var_ch = *var_ch;
                self.next(); // consume 'd'
                self.next(); // consume variable letter
                return Ok(Expression::Differential {
                    var: var_ch.to_string(),
                });
            }
        }
        self.next();
        if ch == 'e' || ch == 'i' {
            Ok(self.resolve_letter(ch, false))
        } else {
            Ok(Expression::Variable(ch.to_string()))
        }
    }

    /// Parses a number-set token (\mathbb{N/Z/Q/R/C/H}).
    pub(super) fn parse_number_set_token(&mut self, token: &LatexToken) -> Option<Expression> {
        use crate::ast::NumberSet;
        let set = match token {
            LatexToken::Naturals => NumberSet::Natural,
            LatexToken::Integers => NumberSet::Integer,
            LatexToken::Rationals => NumberSet::Rational,
            LatexToken::Reals => NumberSet::Real,
            LatexToken::Complexes => NumberSet::Complex,
            _ => return None,
        };
        self.next();
        Some(Expression::NumberSetExpr(set))
    }

    /// Parses vector notation, nabla, multiple/closed integrals, and matrix environments.
    /// Returns `None` when the token does not match any of those forms.
    pub(super) fn parse_vector_integral_token(
        &mut self,
        token: &LatexToken,
        span: Span,
    ) -> ParseResult<Option<Expression>> {
        let expr = match token {
            LatexToken::Nabla => {
                self.next();
                self.parse_nabla()?
            }
            LatexToken::Mathbf | LatexToken::Boldsymbol => {
                self.next();
                self.parse_marked_vector(VectorNotation::Bold)?
            }
            LatexToken::Vec | LatexToken::Overrightarrow => {
                self.next();
                self.parse_marked_vector(VectorNotation::Arrow)?
            }
            LatexToken::Hat => {
                self.next();
                self.parse_marked_vector(VectorNotation::Hat)?
            }
            LatexToken::Underline => {
                self.next();
                self.parse_marked_vector(VectorNotation::Underline)?
            }
            LatexToken::DoubleIntegral => {
                self.next();
                self.parse_multiple_integral(2)?
            }
            LatexToken::TripleIntegral => {
                self.next();
                self.parse_multiple_integral(3)?
            }
            LatexToken::QuadIntegral => {
                self.next();
                self.parse_multiple_integral(4)?
            }
            LatexToken::ClosedIntegral => {
                self.next();
                self.parse_closed_integral(1)?
            }
            LatexToken::ClosedSurface => {
                self.next();
                self.parse_closed_integral(2)?
            }
            LatexToken::ClosedVolume => {
                self.next();
                self.parse_closed_integral(3)?
            }
            LatexToken::BeginEnv(env_name) => {
                let env_name = env_name.clone();
                self.next();
                self.parse_matrix_environment(&env_name)?
            }
            _ => return Ok(None),
        };
        let _ = span; // span available for future error reporting
        Ok(Some(expr))
    }

    /// Parses set/logic atoms, quantifiers, power-set, number sets, vectors, and integrals.
    pub(super) fn parse_primary_extended(
        &mut self,
        token: &LatexToken,
        span: Span,
    ) -> ParseResult<Expression> {
        if let Some(expr) = self.parse_number_set_token(token) {
            return Ok(expr);
        }

        if let Some(expr) = self.parse_vector_integral_token(token, span)? {
            return Ok(expr);
        }

        match token {
            LatexToken::PowerSet => {
                self.next();
                let set = if self.check(&LatexToken::LParen) {
                    self.next();
                    let expr = self.parse_expression()?;
                    self.consume(LatexToken::RParen)?;
                    expr
                } else if self.check(&LatexToken::LBrace) {
                    self.braced(|p| p.parse_expression())?
                } else {
                    self.parse_power()?
                };
                Ok(Expression::PowerSet { set: Box::new(set) })
            }
            LatexToken::EmptySet => {
                self.next();
                Ok(Expression::EmptySet)
            }
            LatexToken::Lnot => {
                self.next();
                let operand = self.parse_power()?;
                Ok(Expression::Logical {
                    op: crate::ast::LogicalOp::Not,
                    operands: vec![operand],
                })
            }
            LatexToken::ForAll => {
                self.next();
                self.parse_forall()
            }
            LatexToken::Exists => {
                self.next();
                self.parse_exists()
            }
            _ => Err(ParseError::unexpected_token(
                vec!["expression"],
                format!("{:?}", token),
                Some(span),
            )),
        }
    }

    /// Parses primary expressions (atoms, commands, parenthesized expressions).
    pub(super) fn parse_primary(&mut self) -> ParseResult<Expression> {
        // Clone both token and span upfront so the immutable borrow on `self` ends
        // before any `&mut self` call below.
        let (token, span) = match self.peek() {
            Some((t, s)) => (t.clone(), *s),
            None => {
                return Err(ParseError::unexpected_eof(
                    vec!["expression"],
                    Some(self.current_span()),
                ))
            }
        };

        match token {
            LatexToken::Number(ref num_str) => {
                let num_str = num_str.clone();
                self.next();
                self.parse_number(&num_str, span)
            }
            LatexToken::Letter(ch) => self.parse_letter_token(ch, span),
            LatexToken::ExplicitConstant(ch) => {
                self.next();
                Ok(self.resolve_letter(ch, true))
            }
            LatexToken::Command(ref cmd) => {
                let cmd = cmd.clone();
                self.next();
                self.parse_command(&cmd, span)
            }
            LatexToken::LParen => {
                self.next();
                let expr = self.parse_expression()?;
                self.consume(LatexToken::RParen)?;
                Ok(expr)
            }
            LatexToken::LBrace => self.braced(|parser| parser.parse_expression()),
            LatexToken::Pipe => {
                self.next();
                let expr = self.parse_expression()?;
                self.consume(LatexToken::Pipe)?;
                Ok(Expression::Function {
                    name: "abs".to_string(),
                    args: vec![expr],
                })
            }
            LatexToken::Minus => {
                self.next();
                let operand = self.parse_power()?;
                if matches!(operand, Expression::Constant(MathConstant::Infinity)) {
                    Ok(Expression::Constant(MathConstant::NegInfinity))
                } else {
                    Ok(Expression::Unary {
                        op: crate::ast::UnaryOp::Neg,
                        operand: Box::new(operand),
                    })
                }
            }
            LatexToken::Plus => {
                self.next();
                let operand = self.parse_power()?;
                Ok(Expression::Unary {
                    op: crate::ast::UnaryOp::Pos,
                    operand: Box::new(operand),
                })
            }
            LatexToken::Infty => {
                self.next();
                Ok(Expression::Constant(MathConstant::Infinity))
            }
            // All remaining tokens are handled by the extended helper.
            // Pass the already-cloned token to avoid re-borrowing self.
            ref t => {
                let t = t.clone();
                self.parse_primary_extended(&t, span)
            }
        }
    }
}
