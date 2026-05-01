//! Primary expression parsing: atoms, identifiers, functions, vectors, subscripts.

use super::*;

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
                let token = self.next().expect("peek confirmed token exists");
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

        // integrate/integral/int(expr, var[, lower, upper]) → Integral
        if matches!(name.as_str(), "integrate" | "integral" | "int") && self.check(&Token::LParen) {
            return self.parse_integrate_function();
        }

        // sum/summation(expr, var, lower, upper) → Sum
        if matches!(name.as_str(), "sum" | "summation" | "Sum") && self.check(&Token::LParen) {
            return self.parse_sum_function();
        }

        // product/prod(expr, var, lower, upper) → Product
        if matches!(name.as_str(), "product" | "prod" | "Product") && self.check(&Token::LParen) {
            return self.parse_product_function();
        }

        // limit/lim(expr, var, point[, direction]) → Limit
        if matches!(name.as_str(), "limit" | "lim" | "Limit") && self.check(&Token::LParen) {
            return self.parse_limit_function();
        }

        // Leibniz notation: dy/dx, d2y/dx2, df/dx, d(expr)/dx
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
        let arg = if self.check(&Token::LParen) {
            self.next();
            let expr = self.parse_expression()?;
            self.consume(Token::RParen)?;
            Box::new(expr)
        } else {
            Box::new(self.parse_primary()?)
        };
        match op_name {
            "grad" => Ok(Expression::Gradient { expr: arg }),
            "div" => Ok(Expression::Divergence { field: arg }),
            "curl" => Ok(Expression::Curl { field: arg }),
            "laplacian" => Ok(Expression::Laplacian { expr: arg }),
            _ => unreachable!(),
        }
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
