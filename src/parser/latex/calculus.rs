// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses an integral: \int f(x) dx or \int_a^b f(x) dx
    pub(super) fn parse_integral(&mut self) -> ParseResult<Expression> {
        // Check for subscript (lower bound)
        let bounds = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let lower = self.parse_braced_or_atom()?;

            // Must have superscript (upper bound) if we have subscript
            if !self.check(&LatexToken::Caret) {
                return Err(ParseError::custom(
                    "integral with lower bound must also have upper bound".to_string(),
                    Some(self.current_span()),
                ));
            }
            self.next(); // consume ^
            let upper = self.parse_braced_or_atom()?;

            Some(IntegralBounds {
                lower: Box::new(lower),
                upper: Box::new(upper),
            })
        } else if self.check(&LatexToken::Caret) {
            // Upper bound without lower bound is an error
            return Err(ParseError::custom(
                "integral with upper bound must also have lower bound".to_string(),
                Some(self.current_span()),
            ));
        } else {
            None
        };

        // Parse integrand - use multiplicative level so x + 1 parses as (int x) + 1
        // Set integral context to prevent 'dx' from being parsed as a Differential
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Expect 'd' followed by variable name
        if let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'

            // Next should be the variable
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                let var = var_ch.to_string();
                self.next(); // consume variable

                Ok(Expression::Integral {
                    integrand: Box::new(integrand),
                    var,
                    bounds,
                })
            } else {
                Err(ParseError::custom(
                    "expected variable name after 'd' in integral".to_string(),
                    Some(self.current_span()),
                ))
            }
        } else {
            Err(ParseError::custom(
                "expected 'd' followed by variable in integral".to_string(),
                Some(self.current_span()),
            ))
        }
    }

    /// Parses a multiple integral: \iint, \iiint, \iiiint
    /// Format: \iint f(x,y) dy dx or \iint_{bounds} f(x,y) dy dx
    pub(super) fn parse_multiple_integral(&mut self, dimension: u8) -> ParseResult<Expression> {
        // Check for optional bounds subscript
        let bounds = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let _region = self.parse_braced_or_atom()?;
            // For simplicity, we don't track the region expression
            None
        } else {
            None
        };

        // Parse integrand with integral context set
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Parse the differential variables (dy dx, dz dy dx, etc.)
        let mut vars = Vec::new();
        while let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                vars.push(var_ch.to_string());
                self.next(); // consume variable
            } else {
                return Err(ParseError::custom(
                    "expected variable name after 'd' in multiple integral".to_string(),
                    Some(self.current_span()),
                ));
            }
        }

        if vars.is_empty() {
            return Err(ParseError::custom(
                format!(
                    "expected {} differential variables for {}-dimensional integral",
                    dimension,
                    match dimension {
                        2 => "double",
                        3 => "triple",
                        4 => "quadruple",
                        _ => "multiple",
                    }
                ),
                Some(self.current_span()),
            ));
        }

        Ok(Expression::MultipleIntegral {
            dimension,
            integrand: Box::new(integrand),
            bounds,
            vars,
        })
    }

    /// Parses a closed/contour integral: \oint, \oiint, \oiiint
    /// Format: \oint_C f(x) dx or \oiint_S f(x,y) dA
    pub(super) fn parse_closed_integral(&mut self, dimension: u8) -> ParseResult<Expression> {
        // Check for optional surface/curve subscript
        let surface = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let surface_expr = self.parse_braced_or_atom()?;
            // Extract name if it's a variable
            match surface_expr {
                Expression::Variable(name) => Some(name),
                _ => None,
            }
        } else {
            None
        };

        // Parse integrand with integral context set
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Parse the differential variable
        let var = if let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                let v = var_ch.to_string();
                self.next(); // consume variable
                v
            } else {
                // Could be dA, dS, etc. - check for uppercase
                if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                    let v = var_ch.to_string();
                    self.next();
                    v
                } else {
                    "".to_string() // Will error below
                }
            }
        } else {
            "".to_string()
        };

        if var.is_empty() {
            return Err(ParseError::custom(
                "expected differential variable in closed integral".to_string(),
                Some(self.current_span()),
            ));
        }

        Ok(Expression::ClosedIntegral {
            dimension,
            integrand: Box::new(integrand),
            surface,
            var,
        })
    }

    // ============================================================
    // Quantifier Parsing
    // ============================================================

    /// Parses a universal quantifier: \forall x P(x) or \forall x \in S P(x)
    pub(super) fn parse_forall(&mut self) -> ParseResult<Expression> {
        // Expect variable
        let variable = match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                let v = ch.to_string();
                self.next();
                v
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Greek letter variable
                let v = cmd.clone();
                self.next();
                v
            }
            _ => {
                return Err(ParseError::custom(
                    "expected variable after \\forall".to_string(),
                    Some(self.current_span()),
                ));
            }
        };

        // Check for optional domain: \in S
        let domain = if let Some((LatexToken::In, _)) = self.peek() {
            self.next(); // consume \in
            let set = self.parse_power()?;
            Some(Box::new(set))
        } else {
            None
        };

        // Parse the body expression
        let body = self.parse_expression()?;

        Ok(Expression::ForAll {
            variable,
            domain,
            body: Box::new(body),
        })
    }

    /// Parses an existential quantifier: \exists x P(x) or \exists! x P(x)
    pub(super) fn parse_exists(&mut self) -> ParseResult<Expression> {
        // Check for unique existence: \exists!
        let unique = if let Some((LatexToken::Command(cmd), _)) = self.peek() {
            if cmd == "!" {
                self.next(); // consume !
                true
            } else {
                false
            }
        } else {
            false
        };

        // Expect variable
        let variable = match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                let v = ch.to_string();
                self.next();
                v
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Greek letter variable
                let v = cmd.clone();
                self.next();
                v
            }
            _ => {
                return Err(ParseError::custom(
                    "expected variable after \\exists".to_string(),
                    Some(self.current_span()),
                ));
            }
        };

        // Check for optional domain: \in S
        let domain = if let Some((LatexToken::In, _)) = self.peek() {
            self.next(); // consume \in
            let set = self.parse_power()?;
            Some(Box::new(set))
        } else {
            None
        };

        // Parse the body expression
        let body = self.parse_expression()?;

        Ok(Expression::Exists {
            variable,
            domain,
            body: Box::new(body),
            unique,
        })
    }

    /// Parses a limit: \lim_{x \to a} or \lim_{x \to a^+}
    pub(super) fn parse_limit(&mut self) -> ParseResult<Expression> {
        // Expect subscript with pattern: var \to value
        if !self.check(&LatexToken::Underscore) {
            return Err(ParseError::custom(
                "limit must have subscript with approach pattern".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume _

        // Parse the subscript content
        self.consume(LatexToken::LBrace)?;

        // Expect variable
        let var = if let Some((LatexToken::Letter(ch), _)) = self.peek() {
            let v = ch.to_string();
            self.next(); // consume variable
            v
        } else {
            return Err(ParseError::custom(
                "expected variable in limit subscript".to_string(),
                Some(self.current_span()),
            ));
        };

        // Expect \to
        if let Some((LatexToken::To, _)) = self.peek() {
            self.next(); // consume \to
        } else {
            return Err(ParseError::custom(
                "expected \\to in limit subscript".to_string(),
                Some(self.current_span()),
            ));
        }

        // Parse approach value
        let to = self.parse_primary()?;

        // Check for direction (^+ or ^-) before the closing brace
        let direction = if self.check(&LatexToken::Caret) {
            self.next(); // consume ^

            match self.peek() {
                Some((LatexToken::Plus, _)) => {
                    self.next();
                    Direction::Right
                }
                Some((LatexToken::Minus, _)) => {
                    self.next();
                    Direction::Left
                }
                _ => {
                    return Err(ParseError::custom(
                        "expected + or - after ^ in limit direction".to_string(),
                        Some(self.current_span()),
                    ));
                }
            }
        } else {
            Direction::Both
        };

        self.consume(LatexToken::RBrace)?;

        // Parse the expression - use parse_multiplicative to capture full expressions
        let expr = self.parse_multiplicative()?;

        Ok(Expression::Limit {
            expr: Box::new(expr),
            var,
            to: Box::new(to),
            direction,
        })
    }

    /// Parses a sum: \sum_{i=1}^{n} expr
    pub(super) fn parse_sum(&mut self) -> ParseResult<Expression> {
        let (index, lower, upper) = self.parse_iterator_bounds()?;
        // Bind the index variable in scope while parsing the body
        self.push_scope(std::iter::once(index.clone()));
        let body = self.parse_multiplicative()?;
        self.pop_scope();

        Ok(Expression::Sum {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Parses a product: \prod_{i=1}^{n} expr
    pub(super) fn parse_product(&mut self) -> ParseResult<Expression> {
        let (index, lower, upper) = self.parse_iterator_bounds()?;
        // Bind the index variable in scope while parsing the body
        self.push_scope(std::iter::once(index.clone()));
        let body = self.parse_multiplicative()?;
        self.pop_scope();

        Ok(Expression::Product {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Helper to parse iterator bounds: _{var=lower}^{upper}
    pub(super) fn parse_iterator_bounds(
        &mut self,
    ) -> ParseResult<(String, Expression, Expression)> {
        // Expect subscript with pattern: var = value
        if !self.check(&LatexToken::Underscore) {
            return Err(ParseError::custom(
                "iterator must have subscript with index=lower pattern".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume _

        // Parse the subscript content
        self.consume(LatexToken::LBrace)?;

        // Expect variable
        let index = if let Some((LatexToken::Letter(ch), _)) = self.peek() {
            let v = ch.to_string();
            self.next(); // consume variable
            v
        } else {
            return Err(ParseError::custom(
                "expected index variable in iterator subscript".to_string(),
                Some(self.current_span()),
            ));
        };

        // Expect =
        if let Some((LatexToken::Equals, _)) = self.peek() {
            self.next(); // consume =
        } else {
            return Err(ParseError::custom(
                "expected = in iterator subscript".to_string(),
                Some(self.current_span()),
            ));
        }

        // Parse lower bound
        let lower = self.parse_additive()?;

        self.consume(LatexToken::RBrace)?;

        // Expect superscript with upper bound
        if !self.check(&LatexToken::Caret) {
            return Err(ParseError::custom(
                "iterator must have superscript with upper bound".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume ^

        let upper = self.parse_braced_or_atom()?;

        Ok((index, lower, upper))
    }
}
