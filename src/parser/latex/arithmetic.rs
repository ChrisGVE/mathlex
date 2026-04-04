// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses additive expressions (+, -, \pm, \mp) and set union/difference.
    pub(super) fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        while let Some((token, _)) = self.peek() {
            // Check for set operations first (union, difference)
            match token {
                LatexToken::Cup => {
                    self.next(); // consume \cup
                    let right = self.parse_multiplicative()?;
                    left = Expression::SetOperation {
                        op: SetOp::Union,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                    continue;
                }
                LatexToken::Setminus => {
                    self.next(); // consume \setminus
                    let right = self.parse_multiplicative()?;
                    left = Expression::SetOperation {
                        op: SetOp::Difference,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                    continue;
                }
                _ => {}
            }

            // Standard arithmetic operators
            let op = match token {
                LatexToken::Plus => BinaryOp::Add,
                LatexToken::Minus => BinaryOp::Sub,
                LatexToken::Command(cmd) if cmd == "pm" => BinaryOp::PlusMinus,
                LatexToken::Command(cmd) if cmd == "mp" => BinaryOp::MinusPlus,
                _ => break,
            };

            self.next(); // consume operator
            let right = self.parse_multiplicative()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Tries to parse a named product operator (\circ, \bullet, \otimes, \wedge, \times).
    /// Returns `Ok(new_left)` if a product operator was consumed, `Err(left)` otherwise.
    pub(super) fn try_parse_named_product(
        &mut self,
        token: &LatexToken,
        left: Expression,
    ) -> ParseResult<Result<Expression, Expression>> {
        match token {
            LatexToken::Circ => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::Composition {
                    outer: Box::new(left),
                    inner: Box::new(right),
                }))
            }
            LatexToken::Bullet => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::DotProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Otimes => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::OuterProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Wedge => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::WedgeProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Cross => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::CrossProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            _ => Ok(Err(left)),
        }
    }

    /// Parses multiplicative expressions (*, /) and set intersection.
    pub(super) fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_power()?;

        while let Some((t, _)) = self.peek() {
            // Clone the token so we drop the immutable borrow before any `&mut self` call.
            let token = t.clone();

            // Set intersection
            if matches!(token, LatexToken::Cap) {
                self.next();
                let right = self.parse_power()?;
                left = Expression::SetOperation {
                    op: SetOp::Intersection,
                    left: Box::new(left),
                    right: Box::new(right),
                };
                continue;
            }

            // Named product operators (\circ, \bullet, \otimes, \wedge, \times)
            match self.try_parse_named_product(&token, left)? {
                Ok(new_left) => {
                    left = new_left;
                    continue;
                }
                Err(returned) => {
                    left = returned;
                }
            }

            // Regular scalar multiplication (token was already cloned above)
            let op = match &token {
                LatexToken::Star | LatexToken::Cdot => BinaryOp::Mul,
                LatexToken::Slash => BinaryOp::Div,
                _ => {
                    if self.should_insert_implicit_mult(&left) {
                        BinaryOp::Mul
                    } else {
                        break;
                    }
                }
            };

            // Consume the explicit operator (not for implicit multiplication)
            if matches!(
                token,
                LatexToken::Star | LatexToken::Cdot | LatexToken::Slash
            ) {
                self.next();
            }

            let right = self.parse_power()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Determines if implicit multiplication should be inserted in LaTeX.
    /// This is used for patterns like 2x, xy, 2\pi, i\pi, etc.
    pub(super) fn should_insert_implicit_mult(&self, left: &Expression) -> bool {
        // Only insert implicit mult when left is a simple variable, number, constant, or differential
        let is_valid_left = matches!(
            left,
            Expression::Variable(_)
                | Expression::Integer(_)
                | Expression::Float(_)
                | Expression::Constant(_)
                | Expression::Differential { .. }
        );
        if !is_valid_left {
            return false;
        }

        // Check if next token is something that could start a multiplicand
        match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                // In integral context, don't trigger implicit mult for 'd' followed by a letter
                // This allows the integral parser to handle 'dx' properly
                if self.in_integral_context && *ch == 'd' {
                    if let Some((LatexToken::Letter(_), _)) = self.peek_ahead(1) {
                        return false;
                    }
                }
                true
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Exclude relation commands and right delimiters - they should not trigger implicit mult
                !matches!(
                    cmd.as_str(),
                    "lt" | "gt"
                        | "leq"
                        | "le"
                        | "geq"
                        | "ge"
                        | "neq"
                        | "ne"
                        | "pm"
                        | "mp"
                        | "cdot"
                        | "times"
                        | "div"
                        | "rfloor"
                        | "rceil"
                )
            }
            Some((LatexToken::LParen, _)) => true,
            Some((LatexToken::LBrace, _)) => true,
            _ => false,
        }
    }

    /// Parses power expressions (^) and subscripts (_).
    ///
    /// Note: When the base is Euler's number `e` (Constant(E)), the expression
    /// `e^x` is normalized to `exp(x)` for consistency with `\exp{x}`.
    pub(super) fn parse_power(&mut self) -> ParseResult<Expression> {
        let mut base = self.parse_postfix()?;

        // Handle superscript (power)
        if self.check(&LatexToken::Caret) {
            self.next(); // consume ^
            let exponent = self.parse_braced_or_atom()?;

            // Normalize e^{...} to exp(...)
            if matches!(base, Expression::Constant(MathConstant::E)) {
                return Ok(Expression::Function {
                    name: "exp".to_string(),
                    args: vec![exponent],
                });
            }

            base = Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(base),
                right: Box::new(exponent),
            };
        }

        // Handle subscript (append to variable name if base is a variable)
        if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let subscript = self.parse_braced_or_atom()?;

            // Convert base to variable with subscript
            base = match base {
                Expression::Variable(name) => {
                    // Format: var_subscript
                    let subscript_str = self.expression_to_subscript_string(&subscript)?;
                    Expression::Variable(format!("{}_{}", name, subscript_str))
                }
                _ => {
                    return Err(ParseError::invalid_subscript(
                        "subscript can only be applied to variables",
                        Some(self.current_span()),
                    ));
                }
            };
        }

        Ok(base)
    }

    /// Parses postfix expressions (currently just primary, extensible for factorial, etc.).
    pub(super) fn parse_postfix(&mut self) -> ParseResult<Expression> {
        self.parse_primary()
    }
}
