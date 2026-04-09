//! Calculus function parsing: integrate, sum, product, limit.

use super::*;

impl TextParser {
    /// Parses `integrate(expr, var)` or `integrate(expr, var, lower, upper)`.
    ///
    /// The `var` argument should be a plain identifier (the variable of integration).
    /// Aliases: `integral`, `int`.
    pub(super) fn parse_integrate_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let integrand = self.parse_expression()?;
        self.consume(Token::Comma)?;

        // Second arg: variable of integration — may be bare identifier or d-prefixed (dx)
        let var_raw = self.expect_identifier("variable of integration")?;
        let var = if var_raw.starts_with('d') && var_raw.len() > 1 {
            var_raw[1..].to_string() // strip leading 'd' from dx, dt, etc.
        } else {
            var_raw
        };

        let bounds = if self.check(&Token::Comma) {
            self.next();
            let lower = self.parse_expression()?;
            self.consume(Token::Comma)?;
            let upper = self.parse_expression()?;
            Some(IntegralBounds {
                lower: Box::new(lower),
                upper: Box::new(upper),
            })
        } else {
            None
        };

        self.consume(Token::RParen)?;
        Ok(Expression::Integral {
            integrand: Box::new(integrand),
            var,
            bounds,
        })
    }

    /// Parses `sum(expr, var, lower, upper)`.
    ///
    /// Aliases: `summation`, `Sum`.
    pub(super) fn parse_sum_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let body = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let index = self.expect_identifier("index variable")?;
        self.consume(Token::Comma)?;
        let lower = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let upper = self.parse_expression()?;
        self.consume(Token::RParen)?;
        Ok(Expression::Sum {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Parses `product(expr, var, lower, upper)`.
    ///
    /// Aliases: `prod`, `Product`.
    pub(super) fn parse_product_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let body = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let index = self.expect_identifier("index variable")?;
        self.consume(Token::Comma)?;
        let lower = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let upper = self.parse_expression()?;
        self.consume(Token::RParen)?;
        Ok(Expression::Product {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Parses `limit(expr, var, point[, direction])`.
    ///
    /// Direction is optional: `"+"` or `"right"` for right-hand,
    /// `"-"` or `"left"` for left-hand, omitted for two-sided.
    /// Aliases: `lim`, `Limit`.
    pub(super) fn parse_limit_function(&mut self) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;
        let expr = self.parse_expression()?;
        self.consume(Token::Comma)?;
        let var = self.expect_identifier("limit variable")?;
        self.consume(Token::Comma)?;
        let to = self.parse_expression()?;

        let direction = if self.check(&Token::Comma) {
            self.next();
            // Expect a direction identifier: +, -, left, right
            let dir_token = self.next().ok_or_else(|| {
                ParseError::unexpected_eof(
                    vec!["direction (+, -, left, right)"],
                    Some(self.current_span()),
                )
            })?;
            match &dir_token.value {
                Token::Plus => Direction::Right,
                Token::Minus => Direction::Left,
                Token::Identifier(s) => match s.as_str() {
                    "right" | "Right" => Direction::Right,
                    "left" | "Left" => Direction::Left,
                    "both" | "Both" => Direction::Both,
                    _ => {
                        return Err(ParseError::unexpected_token(
                            vec!["direction (+, -, left, right)"],
                            format!("{}", dir_token.value),
                            Some(dir_token.span),
                        ));
                    }
                },
                _ => {
                    return Err(ParseError::unexpected_token(
                        vec!["direction (+, -, left, right)"],
                        format!("{}", dir_token.value),
                        Some(dir_token.span),
                    ));
                }
            }
        } else {
            Direction::Both
        };

        self.consume(Token::RParen)?;
        Ok(Expression::Limit {
            expr: Box::new(expr),
            var,
            to: Box::new(to),
            direction,
        })
    }
}
