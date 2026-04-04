// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses a number (integer or float).
    pub(super) fn parse_number(&self, num_str: &str, span: Span) -> ParseResult<Expression> {
        if num_str.contains('.') {
            // Float
            num_str
                .parse::<f64>()
                .map(|f| Expression::Float(MathFloat::from(f)))
                .map_err(|_| ParseError::invalid_number(num_str, "invalid float", Some(span)))
        } else {
            // Integer
            num_str
                .parse::<i64>()
                .map(Expression::Integer)
                .map_err(|_| ParseError::invalid_number(num_str, "invalid integer", Some(span)))
        }
    }

    /// Parses \frac{num}{denom}, promoting to a derivative when the pattern matches.
    pub(super) fn parse_frac_command(&mut self) -> ParseResult<Expression> {
        self.in_fraction_context = true;
        let numerator = self.braced(|p| p.parse_expression())?;
        let denominator = self.braced(|p| p.parse_expression())?;
        self.in_fraction_context = false;

        if let Some(deriv) = self.try_parse_derivative(numerator.clone(), denominator.clone())? {
            return Ok(deriv);
        }

        Ok(Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(numerator),
            right: Box::new(denominator),
        })
    }

    /// Parses \log (with optional base subscript) or \ln.
    pub(super) fn parse_log_command(&mut self, is_log: bool) -> ParseResult<Expression> {
        if is_log && self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let base = self.parse_braced_or_atom()?;
            let arg = self.parse_function_arg()?;
            return Ok(Expression::Function {
                name: "log".to_string(),
                args: vec![arg, base],
            });
        }
        let arg = self.parse_function_arg()?;
        Ok(Expression::Function {
            name: if is_log { "log" } else { "ln" }.to_string(),
            args: vec![arg],
        })
    }

    /// Parses \lfloor expr \rfloor or \lceil expr \rceil.
    pub(super) fn parse_floor_ceil_command(&mut self, is_floor: bool) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;
        let (close_cmd, fn_name, err_msg) = if is_floor {
            ("rfloor", "floor", "expected \\rfloor after \\lfloor")
        } else {
            ("rceil", "ceil", "expected \\rceil after \\lceil")
        };
        if let Some((LatexToken::Command(c), _)) = self.peek() {
            if c == close_cmd {
                self.next();
                return Ok(Expression::Function {
                    name: fn_name.to_string(),
                    args: vec![expr],
                });
            }
        }
        Err(ParseError::custom(
            err_msg.to_string(),
            Some(self.current_span()),
        ))
    }

    /// Parses \delta (Kronecker) or \varepsilon / \epsilon (Levi-Civita).
    /// Falls back to a plain variable when no tensor indices follow.
    pub(super) fn parse_tensor_symbol_command(&mut self, cmd: &str) -> ParseResult<Expression> {
        if self.looks_like_tensor_index() {
            let indices = self.parse_tensor_indices()?;
            if !indices.is_empty() {
                return Ok(if cmd == "delta" {
                    Expression::KroneckerDelta { indices }
                } else {
                    Expression::LeviCivita { indices }
                });
            }
        }
        Ok(Expression::Variable(cmd.to_string()))
    }

    /// Dispatches a LaTeX command to the appropriate sub-parser.
    pub(super) fn parse_command(&mut self, cmd: &str, span: Span) -> ParseResult<Expression> {
        match cmd {
            "frac" => self.parse_frac_command(),

            "sqrt" => {
                if self.check(&LatexToken::LBracket) {
                    let n = self.bracketed(|p| p.parse_expression())?;
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "root".to_string(),
                        args: vec![x, n],
                    })
                } else {
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "sqrt".to_string(),
                        args: vec![x],
                    })
                }
            }

            "delta" | "varepsilon" | "epsilon" => self.parse_tensor_symbol_command(cmd),

            // Greek letters → variables (\pi is a constant)
            "alpha" | "beta" | "gamma" | "zeta" | "eta" | "theta" | "iota" | "kappa" | "lambda"
            | "mu" | "nu" | "xi" | "omicron" | "pi" | "rho" | "sigma" | "tau" | "upsilon"
            | "phi" | "chi" | "psi" | "omega" | "Gamma" | "Delta" | "Theta" | "Lambda" | "Xi"
            | "Pi" | "Sigma" | "Upsilon" | "Phi" | "Psi" | "Omega" => {
                if cmd == "pi" {
                    Ok(Expression::Constant(MathConstant::Pi))
                } else {
                    Ok(Expression::Variable(cmd.to_string()))
                }
            }

            "partial" => Ok(Expression::Variable("partial".to_string())),

            // Single-argument functions
            "sin" | "cos" | "tan" | "sec" | "csc" | "cot" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "exp" | "det" | "min" | "max" | "gcd" | "lcm" | "abs"
            | "floor" | "ceil" | "sgn" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: cmd.to_string(),
                    args: vec![arg],
                })
            }

            "ln" => self.parse_log_command(false),
            "log" => self.parse_log_command(true),
            "lfloor" => self.parse_floor_ceil_command(true),
            "lceil" => self.parse_floor_ceil_command(false),

            "int" => self.parse_integral(),
            "lim" => self.parse_limit(),
            "sum" => self.parse_sum(),
            "prod" => self.parse_product(),

            _ => Err(ParseError::invalid_latex_command(cmd, Some(span))),
        }
    }

    /// Parses a function argument (either braced or a primary expression).
    pub(super) fn parse_function_arg(&mut self) -> ParseResult<Expression> {
        if self.check(&LatexToken::LBrace) {
            self.braced(|p| p.parse_expression())
        } else if self.check(&LatexToken::LParen) {
            self.next(); // consume (
            let expr = self.parse_expression()?;
            self.consume(LatexToken::RParen)?;
            Ok(expr)
        } else {
            // For LaTeX, functions can take unbraced simple arguments: \sin x
            self.parse_power()
        }
    }

    /// Parses an expression in braces {...} or a single atom.
    pub(super) fn parse_braced_or_atom(&mut self) -> ParseResult<Expression> {
        if self.check(&LatexToken::LBrace) {
            self.braced(|p| p.parse_expression())
        } else {
            // Parse a single atom (number, letter, etc.)
            self.parse_primary()
        }
    }

    /// Helper: parses content within braces {...}.
    pub(super) fn braced<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBrace)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBrace)?;
        Ok(result)
    }

    /// Helper: parses content within brackets [...].
    pub(super) fn bracketed<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBracket)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBracket)?;
        Ok(result)
    }

    /// Checks if the next subscript/superscript looks like a tensor index (letters)
    /// rather than a power or regular subscript (numbers/expressions).
    /// This helps distinguish between \delta^i_j (tensor) and \delta^2 (power).
    pub(super) fn looks_like_tensor_index(&self) -> bool {
        // Must have ^ or _ next
        if !self.check(&LatexToken::Caret) && !self.check(&LatexToken::Underscore) {
            return false;
        }

        // Look at what follows ^ or _
        // We need to peek 2 tokens ahead
        let next_pos = self.pos + 1;
        if let Some((token, _)) = self.tokens.get(next_pos) {
            match token {
                // Single letter index: ^i or _j
                LatexToken::Letter(_) => true,
                // Greek letter index: ^\mu or _\nu
                LatexToken::Command(_) => true,
                // Braced group: ^{ij} or _{kl} - check first char inside braces
                LatexToken::LBrace => {
                    // Look inside the braces
                    if let Some((inner, _)) = self.tokens.get(next_pos + 1) {
                        matches!(inner, LatexToken::Letter(_) | LatexToken::Command(_))
                    } else {
                        false
                    }
                }
                // Number means this is a power, not tensor index
                LatexToken::Number(_) => false,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Parses tensor indices from the current position.
    /// Handles patterns like ^{ij}_{kl}, ^i_j, _{ij}, etc.
    /// Returns a vector of TensorIndex with the appropriate index types.
    pub(super) fn parse_tensor_indices(&mut self) -> ParseResult<Vec<TensorIndex>> {
        let mut indices = Vec::new();

        // Parse upper indices (^{...} or ^x)
        if self.check(&LatexToken::Caret) {
            self.next(); // consume ^
            let upper_indices = self.parse_index_group(IndexType::Upper)?;
            indices.extend(upper_indices);
        }

        // Parse lower indices (_{...} or _x)
        if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let lower_indices = self.parse_index_group(IndexType::Lower)?;
            indices.extend(lower_indices);
        }

        // Handle mixed notation: T^i_j^k (rare but valid)
        // Check for additional upper indices after lower
        if self.check(&LatexToken::Caret) {
            self.next();
            let more_upper = self.parse_index_group(IndexType::Upper)?;
            indices.extend(more_upper);
        }

        Ok(indices)
    }

    /// Parses a group of indices (either braced or single character).
    /// Returns a vector of TensorIndex all with the specified index type.
    pub(super) fn parse_index_group(
        &mut self,
        index_type: IndexType,
    ) -> ParseResult<Vec<TensorIndex>> {
        let mut indices = Vec::new();

        if self.check(&LatexToken::LBrace) {
            // Braced group: ^{ij} or _{kl}
            self.next(); // consume {

            // Parse letters inside braces until we hit }
            while !self.check(&LatexToken::RBrace) && !self.check(&LatexToken::Eof) {
                match self.peek() {
                    Some((LatexToken::Letter(ch), _)) => {
                        let ch = *ch;
                        self.next();
                        indices.push(TensorIndex {
                            name: ch.to_string(),
                            index_type,
                        });
                    }
                    Some((LatexToken::Command(cmd), _)) => {
                        // Handle Greek letter indices like μ, ν
                        let cmd = cmd.clone();
                        self.next();
                        indices.push(TensorIndex {
                            name: cmd,
                            index_type,
                        });
                    }
                    Some((_, span)) => {
                        return Err(ParseError::custom(
                            "expected letter in tensor index".to_string(),
                            Some(*span),
                        ));
                    }
                    None => {
                        return Err(ParseError::unexpected_eof(
                            vec!["tensor index"],
                            Some(self.current_span()),
                        ));
                    }
                }
            }

            self.consume(LatexToken::RBrace)?;
        } else {
            // Single character: ^i or _j
            match self.peek() {
                Some((LatexToken::Letter(ch), _)) => {
                    let ch = *ch;
                    self.next();
                    indices.push(TensorIndex {
                        name: ch.to_string(),
                        index_type,
                    });
                }
                Some((LatexToken::Command(cmd), _)) => {
                    // Greek letter index
                    let cmd = cmd.clone();
                    self.next();
                    indices.push(TensorIndex {
                        name: cmd,
                        index_type,
                    });
                }
                Some((_, span)) => {
                    return Err(ParseError::custom(
                        "expected letter in tensor index".to_string(),
                        Some(*span),
                    ));
                }
                None => {
                    return Err(ParseError::unexpected_eof(
                        vec!["tensor index"],
                        Some(self.current_span()),
                    ));
                }
            }
        }

        Ok(indices)
    }
}
