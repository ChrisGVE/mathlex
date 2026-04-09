//! Derivative parsing: diff(), partial(), Leibniz notation, prime notation.

use super::*;

impl TextParser {
    /// Parses `diff(expr, var)` or `diff(expr, var, order)` into `Expression::Derivative`.
    pub(super) fn parse_diff_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let expr = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let var = self.expect_identifier("variable name")?;
        let order = if self.check(&Token::Comma) {
            self.next();
            self.expect_positive_integer("derivative order")?
        } else {
            1
        };
        self.consume(Token::RParen)?;
        Ok(Expression::Derivative {
            expr: Box::new(expr),
            var,
            order,
        })
    }

    /// Parses `partial(expr, var[, order_or_var2[, ...]])` into `Expression::PartialDerivative`.
    ///
    /// - `partial(f, x)` → first partial w.r.t. x
    /// - `partial(f, x, 2)` → second partial w.r.t. x
    /// - `partial(f, x, y)` → mixed partial ∂²f/∂x∂y (nested)
    pub(super) fn parse_partial_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let expr = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let first_var = self.expect_identifier("variable name")?;

        // Check for optional third argument
        if self.check(&Token::Comma) {
            self.next();
            // Peek: integer → order for first_var; identifier → mixed partial
            if let Some(token) = self.peek() {
                match &token.value {
                    Token::Integer(n) => {
                        let order = *n as u32;
                        self.next();
                        // Collect any additional variable args for mixed partials
                        let mut result = Expression::PartialDerivative {
                            expr: Box::new(expr),
                            var: first_var,
                            order,
                        };
                        while self.check(&Token::Comma) {
                            self.next();
                            let var = self.expect_identifier("variable name")?;
                            let ord = if self.check(&Token::Comma) {
                                // Peek ahead: if next after comma is integer, consume it
                                if let Some(t) = self.peek_ahead(1) {
                                    if matches!(t.value, Token::Integer(_)) {
                                        self.next(); // comma
                                        self.expect_positive_integer("derivative order")?
                                    } else {
                                        1
                                    }
                                } else {
                                    1
                                }
                            } else {
                                1
                            };
                            result = Expression::PartialDerivative {
                                expr: Box::new(result),
                                var,
                                order: ord,
                            };
                        }
                        self.consume(Token::RParen)?;
                        return Ok(result);
                    }
                    Token::Identifier(_) => {
                        // Mixed partial: partial(f, x, y) → ∂/∂x(∂/∂y(f))
                        let second_var = self.expect_identifier("variable name")?;
                        let mut result = Expression::PartialDerivative {
                            expr: Box::new(expr),
                            var: second_var,
                            order: 1,
                        };
                        result = Expression::PartialDerivative {
                            expr: Box::new(result),
                            var: first_var,
                            order: 1,
                        };
                        // Support more than 2 vars: partial(f, x, y, z)
                        while self.check(&Token::Comma) {
                            self.next();
                            let var = self.expect_identifier("variable name")?;
                            result = Expression::PartialDerivative {
                                expr: Box::new(result),
                                var,
                                order: 1,
                            };
                        }
                        self.consume(Token::RParen)?;
                        return Ok(result);
                    }
                    _ => {
                        let span = token.span;
                        return Err(ParseError::unexpected_token(
                            vec!["integer order or variable name"],
                            format!("{}", token.value),
                            Some(span),
                        ));
                    }
                }
            }
        }

        self.consume(Token::RParen)?;
        Ok(Expression::PartialDerivative {
            expr: Box::new(expr),
            var: first_var,
            order: 1,
        })
    }

    /// Tries to parse Leibniz derivative notation: `dy/dx`, `d2y/dx2`, `d(expr)/dx`.
    ///
    /// Returns `None` if the identifier doesn't match the pattern or lookahead fails.
    pub(super) fn try_parse_leibniz_derivative(
        &mut self,
        name: &str,
    ) -> ParseResult<Option<Expression>> {
        // Must start with 'd'
        if !name.starts_with('d') {
            return Ok(None);
        }

        // Special case: d(expr)/dx — operator form
        if name == "d" && self.check(&Token::LParen) {
            return self.try_parse_operator_derivative();
        }

        if name.len() < 2 {
            return Ok(None);
        }

        // Lookahead: next must be `/` followed by identifier starting with `d`
        let is_slash = self
            .peek()
            .map(|t| matches!(t.value, Token::Slash))
            .unwrap_or(false);
        if !is_slash {
            return Ok(None);
        }

        let denom_name = self
            .peek_ahead(1)
            .and_then(|t| {
                if let Token::Identifier(s) = &t.value {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        if !denom_name.starts_with('d') || denom_name.len() < 2 {
            return Ok(None);
        }

        // Parse numerator: d[order]<func_name>
        let after_d = &name[1..];
        let (num_order, func_name) = split_order_prefix(after_d);

        if func_name.is_empty() {
            return Ok(None);
        }

        // Parse denominator: d<var>[order]
        let after_d_denom = &denom_name[1..];
        let (denom_order, var_name) = split_order_suffix(after_d_denom);

        if var_name.is_empty() {
            return Ok(None);
        }

        // Orders must match (or numerator has no explicit order = 1)
        let order = if num_order > 0 && denom_order > 0 {
            if num_order != denom_order {
                return Ok(None); // Mismatched orders, not a valid derivative
            }
            num_order
        } else if num_order > 0 {
            num_order
        } else if denom_order > 0 {
            denom_order
        } else {
            1
        };

        // Commit: consume `/` and denominator identifier
        self.next(); // Slash
        self.next(); // denominator identifier

        Ok(Some(Expression::Derivative {
            expr: Box::new(Expression::Variable(func_name.to_string())),
            var: var_name.to_string(),
            order,
        }))
    }

    /// Tries to parse `d(expr)/dx` or `d(expr)/d(var)` operator-form derivative.
    ///
    /// Called when we've consumed identifier `d` and the next token is `(`.
    /// Returns `None` if the pattern doesn't match (falls back to function call).
    fn try_parse_operator_derivative(&mut self) -> ParseResult<Option<Expression>> {
        // We need: ( expr ) / d<var> or ( expr ) / d ( var )
        // Lookahead to verify the /d pattern before committing
        // Save position for backtrack
        let saved_pos = self.pos;

        // Parse the parenthesized expression
        self.next(); // consume (
        let expr = match self.parse_expression() {
            Ok(e) => e,
            Err(_) => {
                self.pos = saved_pos;
                return Ok(None);
            }
        };
        if self.consume(Token::RParen).is_err() {
            self.pos = saved_pos;
            return Ok(None);
        }

        // Expect /
        if !self.check(&Token::Slash) {
            self.pos = saved_pos;
            return Ok(None);
        }
        self.next(); // consume /

        // Expect d<var> or d(<var>)
        let var = if let Some(token) = self.peek() {
            match &token.value {
                Token::Identifier(s) if s.starts_with('d') && s.len() > 1 => {
                    let v = s[1..].to_string();
                    self.next();
                    v
                }
                Token::Identifier(s) if s == "d" => {
                    self.next(); // consume d
                                 // Expect (var)
                    if self.consume(Token::LParen).is_err() {
                        self.pos = saved_pos;
                        return Ok(None);
                    }
                    let v = match self.expect_identifier("variable") {
                        Ok(v) => v,
                        Err(_) => {
                            self.pos = saved_pos;
                            return Ok(None);
                        }
                    };
                    if self.consume(Token::RParen).is_err() {
                        self.pos = saved_pos;
                        return Ok(None);
                    }
                    v
                }
                _ => {
                    self.pos = saved_pos;
                    return Ok(None);
                }
            }
        } else {
            self.pos = saved_pos;
            return Ok(None);
        };

        Ok(Some(Expression::Derivative {
            expr: Box::new(expr),
            var,
            order: 1,
        }))
    }

    /// Parses prime notation: `y'`, `y''`, `y'''`.
    ///
    /// The derivative variable is left empty since prime notation doesn't specify it.
    pub(super) fn parse_prime_derivative(&mut self, name: String) -> ParseResult<Expression> {
        let mut order = 0u32;
        while self.check(&Token::Apostrophe) {
            self.next();
            order += 1;
        }
        Ok(Expression::Derivative {
            expr: Box::new(self.identifier_to_expression(name)),
            var: String::new(),
            order,
        })
    }

    /// Consumes an identifier token and returns its name, or errors with context.
    pub(super) fn expect_identifier(&mut self, context: &str) -> ParseResult<String> {
        if let Some(token) = self.next() {
            if let Token::Identifier(name) = token.value {
                return Ok(name);
            }
            return Err(ParseError::unexpected_token(
                vec![context],
                format!("{}", token.value),
                Some(token.span),
            ));
        }
        Err(ParseError::unexpected_eof(
            vec![context],
            Some(self.current_span()),
        ))
    }

    /// Consumes an integer token and returns it as u32, or errors with context.
    pub(super) fn expect_positive_integer(&mut self, context: &str) -> ParseResult<u32> {
        if let Some(token) = self.next() {
            if let Token::Integer(n) = token.value {
                if n > 0 {
                    return Ok(n as u32);
                }
            }
            return Err(ParseError::unexpected_token(
                vec![context],
                format!("{}", token.value),
                Some(token.span),
            ));
        }
        Err(ParseError::unexpected_eof(
            vec![context],
            Some(self.current_span()),
        ))
    }
}
