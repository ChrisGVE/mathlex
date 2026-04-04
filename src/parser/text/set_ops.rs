//! Set operation and relational expression parsing for the text parser.

use super::*;

impl TextParser {
    pub(super) fn parse_set_operation(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_relation()?;
        while let Some(token) = self.peek() {
            match &token.value {
                Token::Union => {
                    self.next();
                    let right = self.parse_relation()?;
                    left = Expression::SetOperation {
                        op: SetOp::Union,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                Token::Intersect => {
                    self.next();
                    let right = self.parse_relation()?;
                    left = Expression::SetOperation {
                        op: SetOp::Intersection,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                }
                Token::In => {
                    self.next();
                    let right = self.parse_relation()?;
                    left = Expression::SetRelationExpr {
                        relation: SetRelation::In,
                        element: Box::new(left),
                        set: Box::new(right),
                    };
                }
                Token::NotIn => {
                    self.next();
                    let right = self.parse_relation()?;
                    left = Expression::SetRelationExpr {
                        relation: SetRelation::NotIn,
                        element: Box::new(left),
                        set: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        Ok(left)
    }

    pub(super) fn parse_relation(&mut self) -> ParseResult<Expression> {
        let left = self.parse_additive()?;
        if let Some(token) = self.peek() {
            let relation = match &token.value {
                Token::Equals => Some(None),
                Token::Less => Some(Some(InequalityOp::Lt)),
                Token::Greater => Some(Some(InequalityOp::Gt)),
                Token::LessEq => Some(Some(InequalityOp::Le)),
                Token::GreaterEq => Some(Some(InequalityOp::Ge)),
                Token::NotEquals => Some(Some(InequalityOp::Ne)),
                _ => None,
            };
            if let Some(rel_op) = relation {
                self.next();
                let right = self.parse_additive()?;
                if let Some(next_token) = self.peek() {
                    if matches!(
                        next_token.value,
                        Token::Equals
                            | Token::Less
                            | Token::Greater
                            | Token::LessEq
                            | Token::GreaterEq
                            | Token::NotEquals
                    ) {
                        return Err(ParseError::custom(
                            "chained relations are not supported; use explicit grouping \
                             (e.g., (a < b) and (b < c))"
                                .to_string(),
                            Some(next_token.span),
                        ));
                    }
                }
                return Ok(match rel_op {
                    None => Expression::Equation {
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    Some(op) => Expression::Inequality {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                });
            }
        }
        Ok(left)
    }
}
