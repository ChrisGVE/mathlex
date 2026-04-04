// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses an expression (entry point for recursive descent).
    pub(super) fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_function_signature()
    }

    /// Parses function signature: f: A → B
    /// Lowest precedence (type annotations).
    pub(super) fn parse_function_signature(&mut self) -> ParseResult<Expression> {
        let left = self.parse_logical_iff()?;

        // Check for function signature (colon followed by \to)
        if let Some((LatexToken::Colon, _)) = self.peek() {
            // Extract function name (must be a simple variable)
            let name = match &left {
                Expression::Variable(n) => n.clone(),
                _ => {
                    // Not a valid function signature, just return the expression
                    return Ok(left);
                }
            };

            self.next(); // consume colon
            let domain = self.parse_logical_iff()?;

            // Expect \to token
            if let Some((LatexToken::To, _)) = self.peek() {
                self.next(); // consume \to
                let codomain = self.parse_logical_iff()?;

                return Ok(Expression::FunctionSignature {
                    name,
                    domain: Box::new(domain),
                    codomain: Box::new(codomain),
                });
            } else {
                // Missing \to, return error
                return Err(ParseError::custom(
                    "expected \\to after domain in function signature".to_string(),
                    Some(self.current_span()),
                ));
            }
        }

        Ok(left)
    }

    /// Parses biconditional (iff) expressions: P \iff Q
    /// Lowest logical precedence.
    pub(super) fn parse_logical_iff(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_implies()?;

        while let Some((LatexToken::Iff, _)) = self.peek() {
            self.next(); // consume \iff
            let right = self.parse_logical_implies()?;
            left = Expression::Logical {
                op: LogicalOp::Iff,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses implication expressions: P \implies Q
    pub(super) fn parse_logical_implies(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_or()?;

        while let Some((LatexToken::Implies, _)) = self.peek() {
            self.next(); // consume \implies
            let right = self.parse_logical_or()?;
            left = Expression::Logical {
                op: LogicalOp::Implies,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses logical OR expressions: P \lor Q
    pub(super) fn parse_logical_or(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_and()?;

        while let Some((LatexToken::Lor, _)) = self.peek() {
            self.next(); // consume \lor
            let right = self.parse_logical_and()?;
            left = Expression::Logical {
                op: LogicalOp::Or,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses logical AND expressions: P \land Q
    pub(super) fn parse_logical_and(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_set_membership()?;

        while let Some((LatexToken::Land, _)) = self.peek() {
            self.next(); // consume \land
            let right = self.parse_set_membership()?;
            left = Expression::Logical {
                op: LogicalOp::And,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses set membership and subset relations.
    /// Handles: x \in S, x \notin S, A \subset B, A \subseteq B, A \supset B, A \supseteq B
    pub(super) fn parse_set_membership(&mut self) -> ParseResult<Expression> {
        let left = self.parse_relation()?;

        // Check for set membership and subset operators
        if let Some((token, _)) = self.peek() {
            let relation = match token {
                LatexToken::In => Some(SetRelation::In),
                LatexToken::NotIn => Some(SetRelation::NotIn),
                LatexToken::Subset => Some(SetRelation::Subset),
                LatexToken::SubsetEq => Some(SetRelation::SubsetEq),
                LatexToken::Superset => Some(SetRelation::Superset),
                LatexToken::SupersetEq => Some(SetRelation::SupersetEq),
                _ => None,
            };

            if let Some(rel) = relation {
                self.next(); // consume the relation token
                let right = self.parse_relation()?;
                return Ok(Expression::SetRelationExpr {
                    relation: rel,
                    element: Box::new(left),
                    set: Box::new(right),
                });
            }
        }

        Ok(left)
    }

    /// Matches a similarity/equivalence relation token to its operator.
    fn match_math_relation(token: &LatexToken) -> Option<RelationOp> {
        match token {
            LatexToken::Sim => Some(RelationOp::Similar),
            LatexToken::Equiv => Some(RelationOp::Equivalent),
            LatexToken::Cong => Some(RelationOp::Congruent),
            LatexToken::Approx => Some(RelationOp::Approx),
            _ => None,
        }
    }

    /// Matches an equation/inequality token to an optional InequalityOp (None = Equation).
    fn match_ineq_relation(token: &LatexToken) -> Option<Option<InequalityOp>> {
        match token {
            LatexToken::Equals => Some(None),
            LatexToken::Less => Some(Some(InequalityOp::Lt)),
            LatexToken::Greater => Some(Some(InequalityOp::Gt)),
            LatexToken::Command(cmd) => match cmd.as_str() {
                "lt" => Some(Some(InequalityOp::Lt)),
                "gt" => Some(Some(InequalityOp::Gt)),
                "leq" | "le" => Some(Some(InequalityOp::Le)),
                "geq" | "ge" => Some(Some(InequalityOp::Ge)),
                "neq" | "ne" => Some(Some(InequalityOp::Ne)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns true if the token is any relation operator (used to detect chaining).
    fn is_relation_token(token: &LatexToken) -> bool {
        matches!(
            token,
            LatexToken::Equals
                | LatexToken::Less
                | LatexToken::Greater
                | LatexToken::Sim
                | LatexToken::Equiv
                | LatexToken::Cong
                | LatexToken::Approx
        ) || matches!(
            token,
            LatexToken::Command(cmd) if matches!(
                cmd.as_str(),
                "lt" | "gt" | "leq" | "le" | "geq" | "ge" | "neq" | "ne"
            )
        )
    }

    /// Builds an Equation or Inequality expression from two operands.
    fn build_equation_or_ineq(
        rel_op: Option<InequalityOp>,
        left: Expression,
        right: Expression,
    ) -> Expression {
        match rel_op {
            None => Expression::Equation {
                left: Box::new(left),
                right: Box::new(right),
            },
            Some(op) => Expression::Inequality {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }

    /// Parses relational expressions (=, <, >, \leq, \geq, \neq, etc.).
    pub(super) fn parse_relation(&mut self) -> ParseResult<Expression> {
        let left = self.parse_additive()?;

        if let Some((token, span)) = self.peek() {
            let span = *span;

            // Check similarity/equivalence/congruence/approximation relations
            if let Some(rel_op) = Self::match_math_relation(token) {
                self.next();
                let right = self.parse_additive()?;
                return Ok(Expression::Relation {
                    op: rel_op,
                    left: Box::new(left),
                    right: Box::new(right),
                });
            }

            // Check equation/inequality relations
            if let Some(rel_op) = Self::match_ineq_relation(token) {
                self.next();
                let right = self.parse_additive()?;

                // Reject chained relations
                if let Some((next_token, next_span)) = self.peek() {
                    if Self::is_relation_token(next_token) {
                        return Err(ParseError::custom(
                            "chained relations are not supported; use explicit grouping"
                                .to_string(),
                            Some(*next_span),
                        ));
                    }
                }

                return Ok(Self::build_equation_or_ineq(rel_op, left, right));
            }

            let _ = span; // suppress unused warning
        }

        Ok(left)
    }
}
