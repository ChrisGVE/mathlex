//! Expression and logical operator parsing for the text parser.

use super::*;

impl TextParser {
    pub(super) fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_logical()
    }

    pub(super) fn parse_logical(&mut self) -> ParseResult<Expression> {
        if let Some(token) = self.peek() {
            if matches!(token.value, Token::Not) {
                self.next();
                let operand = self.parse_logical()?;
                return Ok(Expression::Logical {
                    op: LogicalOp::Not,
                    operands: vec![operand],
                });
            }
        }
        let mut left = self.parse_quantifier()?;
        while let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Iff => LogicalOp::Iff,
                Token::Implies => LogicalOp::Implies,
                Token::Or => LogicalOp::Or,
                Token::And => LogicalOp::And,
                _ => break,
            };
            self.next();
            let right = self.parse_quantifier()?;
            left = Expression::Logical {
                op,
                operands: vec![left, right],
            };
        }
        Ok(left)
    }

    pub(super) fn parse_quantifier(&mut self) -> ParseResult<Expression> {
        if let Some(token) = self.peek() {
            match &token.value {
                Token::ForAll => {
                    self.next();
                    return self.parse_quantifier_body(false);
                }
                Token::Exists => {
                    self.next();
                    return self.parse_quantifier_body(true);
                }
                _ => {}
            }
        }
        self.parse_set_operation()
    }

    /// Parses the body of a quantifier: variable [in domain], body
    pub(super) fn parse_quantifier_body(&mut self, is_exists: bool) -> ParseResult<Expression> {
        let variable = if let Some(token) = self.peek() {
            if let Token::Identifier(name) = &token.value {
                let var = name.clone();
                self.next();
                var
            } else {
                return Err(ParseError::unexpected_token(
                    vec!["variable"],
                    format!("{}", token.value),
                    Some(token.span),
                ));
            }
        } else {
            return Err(ParseError::unexpected_eof(
                vec!["variable"],
                Some(self.current_span()),
            ));
        };

        let domain = if let Some(token) = self.peek() {
            if matches!(token.value, Token::In) {
                self.next();
                Some(Box::new(self.parse_set_operation()?))
            } else {
                None
            }
        } else {
            None
        };

        self.consume(Token::Comma)?;
        let body = Box::new(self.parse_logical()?);

        if is_exists {
            Ok(Expression::Exists {
                variable,
                domain,
                body,
                unique: false,
            })
        } else {
            Ok(Expression::ForAll {
                variable,
                domain,
                body,
            })
        }
    }
}
