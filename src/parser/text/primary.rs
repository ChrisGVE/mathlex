//! Primary expression parsing: atoms, identifiers, functions, vectors, subscripts.

use super::*;

/// Splits a string like "2y" into (2, "y") — order prefix then remaining.
/// Returns (0, s) if no leading digits.
fn split_order_prefix(s: &str) -> (u32, &str) {
    let digit_end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if digit_end == 0 {
        (0, s)
    } else {
        let order = s[..digit_end].parse::<u32>().unwrap_or(0);
        (order, &s[digit_end..])
    }
}

/// Splits a string like "x2" into (2, "x") — trailing order then remaining.
/// Returns (0, s) if no trailing digits.
fn split_order_suffix(s: &str) -> (u32, &str) {
    let alpha_end = s
        .find(|c: char| !c.is_ascii_alphabetic())
        .unwrap_or(s.len());
    if alpha_end == s.len() {
        (0, s)
    } else {
        let order = s[alpha_end..].parse::<u32>().unwrap_or(0);
        (order, &s[..alpha_end])
    }
}

/// Normalizes function name aliases to their canonical mathlex names.
///
/// Maps common alternative spellings used by thales and other consumers to the
/// canonical names recognized throughout the mathlex AST.
fn normalize_function_name(name: &str) -> &str {
    match name {
        "asin" => "arcsin",
        "acos" => "arccos",
        "atan" => "arctan",
        "sign" => "sgn",
        "log2" => "lg",
        _ => name,
    }
}

impl TextParser {
    pub(super) fn parse_primary(&mut self) -> ParseResult<Expression> {
        let token = self.peek().ok_or_else(|| {
            ParseError::unexpected_eof(vec!["expression"], Some(self.current_span()))
        })?;
        match &token.value {
            Token::Integer(_) | Token::Float(_) => self.parse_number_token(),
            Token::Identifier(_) => self.parse_identifier_token(),
            Token::Pi => {
                self.next();
                Ok(Expression::Constant(MathConstant::Pi))
            }
            Token::Infinity => {
                self.next();
                Ok(Expression::Constant(MathConstant::Infinity))
            }
            Token::Sqrt => self.parse_sqrt_token(),
            Token::Dot
            | Token::Cross
            | Token::Grad
            | Token::Div
            | Token::Curl
            | Token::Laplacian => self.parse_vector_token(),
            Token::LParen => self.parse_paren_group(),
            _ => {
                let token = self.next().unwrap();
                Err(ParseError::unexpected_token(
                    vec!["number", "variable", "("],
                    format!("{}", token.value),
                    Some(token.span),
                ))
            }
        }
    }

    pub(super) fn parse_number_token(&mut self) -> ParseResult<Expression> {
        let token = self.next().expect("caller checked token exists");
        match token.value {
            Token::Integer(n) => Ok(Expression::Integer(n)),
            Token::Float(f) => Ok(Expression::Float(MathFloat::from(f))),
            _ => unreachable!("caller guarantees Integer or Float token"),
        }
    }

    pub(super) fn parse_identifier_token(&mut self) -> ParseResult<Expression> {
        let name_raw = match &self.peek().expect("caller checked token exists").value {
            Token::Identifier(n) => n.clone(),
            _ => unreachable!("caller guarantees Identifier token"),
        };
        self.next();
        let name = self.parse_subscript(name_raw)?;

        // diff(expr, var) or diff(expr, var, order) → Derivative
        if name == "diff" && self.check(&Token::LParen) {
            return self.parse_diff_function();
        }

        // partial(expr, var) or partial(expr, var, order/var2) → PartialDerivative
        if name == "partial" && self.check(&Token::LParen) {
            return self.parse_partial_function();
        }

        // Leibniz notation: dy/dx, d2y/dx2, df/dx
        if let Some(deriv) = self.try_parse_leibniz_derivative(&name)? {
            return Ok(deriv);
        }

        // Prime notation: y', y'', y'''
        if self.check(&Token::Apostrophe) {
            return self.parse_prime_derivative(name);
        }

        if self.check(&Token::LParen) {
            self.parse_function_args(name)
        } else {
            Ok(self.identifier_to_expression(name))
        }
    }

    /// Parses an optional subscript `_<number|identifier>`, returns combined name.
    pub(super) fn parse_subscript(&mut self, base: String) -> ParseResult<String> {
        if !self.check(&Token::Underscore) {
            return Ok(base);
        }
        self.next();
        let subscript = if let Some(token) = self.peek() {
            match &token.value {
                Token::Integer(n) => {
                    let sub = n.to_string();
                    self.next();
                    sub
                }
                Token::Identifier(id) => {
                    let sub = id.clone();
                    self.next();
                    sub
                }
                _ => {
                    return Err(ParseError::unexpected_token(
                        vec!["number", "identifier"],
                        format!("{}", token.value),
                        Some(token.span),
                    ));
                }
            }
        } else {
            return Err(ParseError::unexpected_eof(
                vec!["subscript"],
                Some(self.current_span()),
            ));
        };
        Ok(format!("{}_{}", base, subscript))
    }

    pub(super) fn parse_sqrt_token(&mut self) -> ParseResult<Expression> {
        self.next();
        let arg = if self.check(&Token::LParen) {
            self.next();
            let expr = self.parse_expression()?;
            self.consume(Token::RParen)?;
            expr
        } else {
            self.parse_primary()?
        };
        Ok(Expression::Function {
            name: "sqrt".to_string(),
            args: vec![arg],
        })
    }

    pub(super) fn parse_vector_token(&mut self) -> ParseResult<Expression> {
        let token = self.next().expect("caller checked token exists");
        match token.value {
            Token::Dot => self.parse_binary_vector_op("dot"),
            Token::Cross => self.parse_binary_vector_op("cross"),
            Token::Grad => self.parse_unary_vector_calculus("grad"),
            Token::Div => self.parse_unary_vector_calculus("div"),
            Token::Curl => self.parse_unary_vector_calculus("curl"),
            Token::Laplacian => self.parse_unary_vector_calculus("laplacian"),
            _ => unreachable!("caller guarantees vector token"),
        }
    }

    pub(super) fn parse_paren_group(&mut self) -> ParseResult<Expression> {
        self.next();
        let expr = self.parse_expression()?;
        self.consume(Token::RParen)?;
        Ok(expr)
    }

    pub(super) fn parse_function_args(&mut self, name: String) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let mut args = Vec::new();
        if self.check(&Token::RParen) {
            let span = self.current_span();
            return Err(ParseError::unexpected_token(
                vec!["expression"],
                ")".to_string(),
                Some(span),
            ));
        }
        args.push(self.parse_expression()?);
        while self.check(&Token::Comma) {
            self.next();
            args.push(self.parse_expression()?);
        }
        self.consume(Token::RParen)?;
        Ok(Expression::Function {
            name: normalize_function_name(&name).to_string(),
            args,
        })
    }

    pub(super) fn parse_binary_vector_op(&mut self, op_name: &str) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let left = Box::new(self.parse_expression()?);
        self.consume(Token::Comma)?;
        let right = Box::new(self.parse_expression()?);
        self.consume(Token::RParen)?;
        match op_name {
            "dot" => Ok(Expression::DotProduct { left, right }),
            "cross" => Ok(Expression::CrossProduct { left, right }),
            _ => unreachable!(),
        }
    }

    pub(super) fn parse_unary_vector_calculus(&mut self, op_name: &str) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let arg = Box::new(self.parse_expression()?);
        self.consume(Token::RParen)?;
        match op_name {
            "grad" => Ok(Expression::Gradient { expr: arg }),
            "div" => Ok(Expression::Divergence { field: arg }),
            "curl" => Ok(Expression::Curl { field: arg }),
            "laplacian" => Ok(Expression::Laplacian { expr: arg }),
            _ => unreachable!(),
        }
    }

    /// Parses `diff(expr, var)` or `diff(expr, var, order)` into `Expression::Derivative`.
    fn parse_diff_function(&mut self) -> ParseResult<Expression> {
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
    fn parse_partial_function(&mut self) -> ParseResult<Expression> {
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

    /// Tries to parse Leibniz derivative notation: `dy/dx`, `d2y/dx2`.
    ///
    /// Returns `None` if the identifier doesn't match the pattern or lookahead fails.
    fn try_parse_leibniz_derivative(&mut self, name: &str) -> ParseResult<Option<Expression>> {
        // Must start with 'd'
        if !name.starts_with('d') || name.len() < 2 {
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

    /// Parses prime notation: `y'`, `y''`, `y'''`.
    ///
    /// The derivative variable is left empty since prime notation doesn't specify it.
    fn parse_prime_derivative(&mut self, name: String) -> ParseResult<Expression> {
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
    fn expect_identifier(&mut self, context: &str) -> ParseResult<String> {
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
    fn expect_positive_integer(&mut self, context: &str) -> ParseResult<u32> {
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

    pub(super) fn identifier_to_expression(&self, name: String) -> Expression {
        match name.as_str() {
            "pi" => Expression::Constant(MathConstant::Pi),
            "e" => Expression::Constant(MathConstant::E),
            "i" => Expression::Constant(MathConstant::I),
            "inf" => Expression::Constant(MathConstant::Infinity),
            "nan" | "NaN" => Expression::Constant(MathConstant::NaN),
            _ => Expression::Variable(name),
        }
    }
}
