//! Arithmetic expression parsing: additive, multiplicative, unary, power, postfix.

use super::*;

impl TextParser {
    pub(super) fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_multiplicative()?;
        while let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.next();
            let right = self.parse_multiplicative()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    pub(super) fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_unary()?;
        loop {
            let op = if let Some(token) = self.peek() {
                match &token.value {
                    Token::Star => Some(BinaryOp::Mul),
                    Token::Slash => Some(BinaryOp::Div),
                    Token::Percent => Some(BinaryOp::Mod),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(op) = op {
                self.next();
                let right = self.parse_unary()?;
                left = Expression::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else if self.should_insert_implicit_mult(&left) {
                let right = self.parse_unary()?;
                left = Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else {
                break;
            }
        }
        Ok(left)
    }

    pub(super) fn parse_unary(&mut self) -> ParseResult<Expression> {
        if let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Minus => Some(UnaryOp::Neg),
                Token::Plus => Some(UnaryOp::Pos),
                _ => None,
            };
            if let Some(op) = op {
                self.next();
                let operand = self.parse_unary()?;
                if matches!(op, UnaryOp::Neg)
                    && matches!(operand, Expression::Constant(MathConstant::Infinity))
                {
                    return Ok(Expression::Constant(MathConstant::NegInfinity));
                }
                return Ok(Expression::Unary {
                    op,
                    operand: Box::new(operand),
                });
            }
        }
        self.parse_power()
    }

    pub(super) fn parse_power(&mut self) -> ParseResult<Expression> {
        let left = self.parse_postfix()?;
        if self.check(&Token::Caret) || self.check(&Token::DoubleStar) {
            self.next();
            let right = self.parse_power()?;
            Ok(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    pub(super) fn parse_postfix(&mut self) -> ParseResult<Expression> {
        let mut expr = self.parse_primary()?;
        while self.check(&Token::Bang) {
            self.next();
            expr = Expression::Unary {
                op: UnaryOp::Factorial,
                operand: Box::new(expr),
            };
        }
        Ok(expr)
    }
}
