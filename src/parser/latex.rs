//! LaTeX expression parser for mathematical notation.
//!
//! This module provides parsing capabilities for LaTeX mathematical expressions,
//! converting tokenized LaTeX input into an Abstract Syntax Tree (AST).
//!
//! # Supported Constructs
//!
//! - **Fractions**: `\frac{num}{denom}` → Binary division
//! - **Roots**: `\sqrt{x}`, `\sqrt[n]{x}` → Function calls
//! - **Powers**: `x^2`, `x^{expr}` → Binary exponentiation
//! - **Subscripts**: `x_1`, `x_{sub}` → Variables with subscripts
//! - **Greek letters**: `\alpha`, `\beta`, etc. → Variables
//! - **Constants**: `\pi`, `\infty` → Mathematical constants
//! - **Trigonometric functions**: `\sin`, `\cos`, `\tan`, etc. → Functions
//! - **Basic operators**: `+`, `-`, `*`, `/`
//!
//! # Example
//!
//! ```ignore
//! use mathlex::parser::parse_latex;
//!
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//! // Returns: Binary { op: Div, left: Integer(1), right: Integer(2) }
//! ```

use crate::ast::{BinaryOp, Expression, InequalityOp, MathConstant, MathFloat};
use crate::error::{ParseError, ParseResult, Span};
use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
use crate::parser::Spanned;

/// Parses a LaTeX mathematical expression.
///
/// # Arguments
///
/// * `input` - The LaTeX string to parse
///
/// # Returns
///
/// A parsed AST expression or a parse error.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_latex;
///
/// // Simple fraction
/// let expr = parse_latex(r"\frac{1}{2}").unwrap();
///
/// // Square root
/// let expr = parse_latex(r"\sqrt{x}").unwrap();
///
/// // Power
/// let expr = parse_latex(r"x^{2+3}").unwrap();
/// ```
pub fn parse_latex(input: &str) -> ParseResult<Expression> {
    let tokens = tokenize_latex(input)?;
    let parser = LatexParser::new(tokens);
    parser.parse()
}

/// Internal parser state for LaTeX expressions.
struct LatexParser {
    /// Token stream with positions
    tokens: Vec<Spanned<LatexToken>>,
    /// Current position in token stream
    pos: usize,
}

impl LatexParser {
    /// Creates a new parser from a token stream.
    fn new(tokens: Vec<Spanned<LatexToken>>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Returns the current token without consuming it.
    fn peek(&self) -> Option<&Spanned<LatexToken>> {
        self.tokens.get(self.pos)
    }

    /// Returns the current token and advances position.
    fn next(&mut self) -> Option<Spanned<LatexToken>> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Returns the current position/span for error reporting.
    fn current_span(&self) -> Span {
        self.peek().map(|(_, span)| *span).unwrap_or_else(|| {
            // Use the last token's end position if we're at EOF
            if let Some((_, last_span)) = self.tokens.last() {
                Span::at(last_span.end)
            } else {
                Span::start()
            }
        })
    }

    /// Checks if current token matches a pattern without consuming.
    fn check(&self, expected: &LatexToken) -> bool {
        self.peek().map(|(tok, _)| tok == expected).unwrap_or(false)
    }

    /// Consumes a token if it matches the expected token.
    fn consume(&mut self, expected: LatexToken) -> ParseResult<Span> {
        if let Some((token, span)) = self.next() {
            if token == expected {
                Ok(span)
            } else {
                Err(ParseError::unexpected_token(
                    vec![format!("{:?}", expected)],
                    format!("{:?}", token),
                    Some(span),
                ))
            }
        } else {
            Err(ParseError::unexpected_eof(
                vec![format!("{:?}", expected)],
                Some(self.current_span()),
            ))
        }
    }

    /// Main entry point for parsing.
    fn parse(mut self) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;

        // Ensure we consumed all non-EOF tokens
        if let Some((token, span)) = self.peek() {
            if !matches!(token, LatexToken::Eof) {
                return Err(ParseError::unexpected_token(
                    vec!["end of input"],
                    format!("{:?}", token),
                    Some(*span),
                ));
            }
        }

        Ok(expr)
    }

    /// Parses an expression (entry point for recursive descent).
    fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_relation()
    }

    /// Parses relational expressions (=, <, >, \leq, \geq, \neq, etc.).
    fn parse_relation(&mut self) -> ParseResult<Expression> {
        let left = self.parse_additive()?;

        // Check for relation operator
        if let Some((token, span)) = self.peek() {
            let span = *span;
            let relation = match token {
                LatexToken::Equals => Some((None, span)), // None indicates equation
                LatexToken::Less => Some((Some(InequalityOp::Lt), span)),
                LatexToken::Greater => Some((Some(InequalityOp::Gt), span)),
                LatexToken::Command(cmd) => match cmd.as_str() {
                    "lt" => Some((Some(InequalityOp::Lt), span)),
                    "gt" => Some((Some(InequalityOp::Gt), span)),
                    "leq" | "le" => Some((Some(InequalityOp::Le), span)),
                    "geq" | "ge" => Some((Some(InequalityOp::Ge), span)),
                    "neq" | "ne" => Some((Some(InequalityOp::Ne), span)),
                    _ => None,
                },
                _ => None,
            };

            if let Some((rel_op, _)) = relation {
                self.next(); // consume relation operator
                let right = self.parse_additive()?;

                // Check for chained relations and error if found
                if let Some((next_token, next_span)) = self.peek() {
                    let is_relation = matches!(
                        next_token,
                        LatexToken::Equals | LatexToken::Less | LatexToken::Greater
                    ) || matches!(
                        next_token,
                        LatexToken::Command(cmd) if matches!(
                            cmd.as_str(),
                            "lt" | "gt" | "leq" | "le" | "geq" | "ge" | "neq" | "ne"
                        )
                    );

                    if is_relation {
                        return Err(ParseError::custom(
                            "chained relations are not supported; use explicit grouping"
                                .to_string(),
                            Some(*next_span),
                        ));
                    }
                }

                // Return Equation or Inequality
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

    /// Parses additive expressions (+ and -).
    fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        while let Some((token, _)) = self.peek() {
            let op = match token {
                LatexToken::Plus => BinaryOp::Add,
                LatexToken::Minus => BinaryOp::Sub,
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

    /// Parses multiplicative expressions (*, /).
    fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_power()?;

        while let Some((token, _)) = self.peek() {
            let op = match token {
                LatexToken::Star => BinaryOp::Mul,
                LatexToken::Slash => BinaryOp::Div,
                _ => break,
            };

            self.next(); // consume operator
            let right = self.parse_power()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parses power expressions (^) and subscripts (_).
    fn parse_power(&mut self) -> ParseResult<Expression> {
        let mut base = self.parse_postfix()?;

        // Handle superscript (power)
        if self.check(&LatexToken::Caret) {
            self.next(); // consume ^
            let exponent = self.parse_braced_or_atom()?;
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
    fn parse_postfix(&mut self) -> ParseResult<Expression> {
        self.parse_primary()
    }

    /// Parses primary expressions (atoms, commands, parenthesized expressions).
    fn parse_primary(&mut self) -> ParseResult<Expression> {
        match self.peek() {
            Some((token, span)) => {
                let span = *span;
                match token {
                    LatexToken::Number(num_str) => {
                        let num_str = num_str.clone();
                        self.next(); // consume
                        self.parse_number(&num_str, span)
                    }
                    LatexToken::Letter(ch) => {
                        let ch = *ch;
                        self.next(); // consume
                        Ok(Expression::Variable(ch.to_string()))
                    }
                    LatexToken::Command(cmd) => {
                        let cmd = cmd.clone();
                        self.next(); // consume
                        self.parse_command(&cmd, span)
                    }
                    LatexToken::LParen => {
                        self.next(); // consume (
                        let expr = self.parse_expression()?;
                        self.consume(LatexToken::RParen)?;
                        Ok(expr)
                    }
                    LatexToken::LBrace => self.braced(|parser| parser.parse_expression()),
                    LatexToken::Pipe => {
                        // Absolute value: |expr|
                        self.next(); // consume |
                        let expr = self.parse_expression()?;
                        self.consume(LatexToken::Pipe)?;
                        Ok(Expression::Function {
                            name: "abs".to_string(),
                            args: vec![expr],
                        })
                    }
                    LatexToken::Minus => {
                        // Unary minus
                        self.next(); // consume -
                        let operand = self.parse_power()?;
                        Ok(Expression::Unary {
                            op: crate::ast::UnaryOp::Neg,
                            operand: Box::new(operand),
                        })
                    }
                    LatexToken::Plus => {
                        // Unary plus
                        self.next(); // consume +
                        let operand = self.parse_power()?;
                        Ok(Expression::Unary {
                            op: crate::ast::UnaryOp::Pos,
                            operand: Box::new(operand),
                        })
                    }
                    LatexToken::Infty => {
                        self.next(); // consume
                        Ok(Expression::Constant(MathConstant::Infinity))
                    }
                    _ => Err(ParseError::unexpected_token(
                        vec!["expression"],
                        format!("{:?}", token),
                        Some(span),
                    )),
                }
            }
            None => Err(ParseError::unexpected_eof(
                vec!["expression"],
                Some(self.current_span()),
            )),
        }
    }

    /// Parses a number (integer or float).
    fn parse_number(&self, num_str: &str, span: Span) -> ParseResult<Expression> {
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

    /// Parses a LaTeX command.
    fn parse_command(&mut self, cmd: &str, span: Span) -> ParseResult<Expression> {
        match cmd {
            // Fractions: \frac{num}{denom}
            "frac" => {
                let numerator = self.braced(|p| p.parse_expression())?;
                let denominator = self.braced(|p| p.parse_expression())?;
                Ok(Expression::Binary {
                    op: BinaryOp::Div,
                    left: Box::new(numerator),
                    right: Box::new(denominator),
                })
            }

            // Square root: \sqrt{x} or \sqrt[n]{x}
            "sqrt" => {
                // Check for optional [n] parameter
                if self.check(&LatexToken::LBracket) {
                    // nth root: \sqrt[n]{x}
                    let n = self.bracketed(|p| p.parse_expression())?;
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "root".to_string(),
                        args: vec![x, n],
                    })
                } else {
                    // square root: \sqrt{x}
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "sqrt".to_string(),
                        args: vec![x],
                    })
                }
            }

            // Greek letters -> Variables
            "alpha" | "beta" | "gamma" | "delta" | "epsilon" | "zeta" | "eta" | "theta"
            | "iota" | "kappa" | "lambda" | "mu" | "nu" | "xi" | "omicron" | "pi" | "rho"
            | "sigma" | "tau" | "upsilon" | "phi" | "chi" | "psi" | "omega" | "Gamma" | "Delta"
            | "Theta" | "Lambda" | "Xi" | "Pi" | "Sigma" | "Upsilon" | "Phi" | "Psi" | "Omega" => {
                // Special case: \pi is a constant
                if cmd == "pi" {
                    Ok(Expression::Constant(MathConstant::Pi))
                } else {
                    Ok(Expression::Variable(cmd.to_string()))
                }
            }

            // Trigonometric functions
            "sin" | "cos" | "tan" | "sec" | "csc" | "cot" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: cmd.to_string(),
                    args: vec![arg],
                })
            }

            // Logarithms
            "ln" | "log" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: cmd.to_string(),
                    args: vec![arg],
                })
            }

            // Exponential
            "exp" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: "exp".to_string(),
                    args: vec![arg],
                })
            }

            // Other common functions
            "min" | "max" | "gcd" | "lcm" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: cmd.to_string(),
                    args: vec![arg],
                })
            }

            _ => Err(ParseError::invalid_latex_command(cmd, Some(span))),
        }
    }

    /// Parses a function argument (either braced or a primary expression).
    fn parse_function_arg(&mut self) -> ParseResult<Expression> {
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
    fn parse_braced_or_atom(&mut self) -> ParseResult<Expression> {
        if self.check(&LatexToken::LBrace) {
            self.braced(|p| p.parse_expression())
        } else {
            // Parse a single atom (number, letter, etc.)
            self.parse_primary()
        }
    }

    /// Helper: parses content within braces {...}.
    fn braced<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBrace)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBrace)?;
        Ok(result)
    }

    /// Helper: parses content within brackets [...].
    fn bracketed<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBracket)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBracket)?;
        Ok(result)
    }

    /// Converts an expression to a subscript string representation.
    fn expression_to_subscript_string(&self, expr: &Expression) -> ParseResult<String> {
        match expr {
            Expression::Integer(n) => Ok(n.to_string()),
            Expression::Variable(s) => Ok(s.clone()),
            _ => Err(ParseError::invalid_subscript(
                "subscript must be a simple value (number or variable)",
                Some(self.current_span()),
            )),
        }
    }

    /// Parses a matrix environment (\begin{matrix}...\end{matrix} and variants).
    fn parse_matrix_environment(&mut self, env_name: &str) -> ParseResult<Expression> {
        // Validate environment name
        match env_name {
            "matrix" | "bmatrix" | "pmatrix" | "vmatrix" | "Bmatrix" | "Vmatrix" => {}
            _ => {
                return Err(ParseError::invalid_latex_command(
                    &format!("\\begin{{{}}}", env_name),
                    Some(self.current_span()),
                ));
            }
        }

        let mut rows: Vec<Vec<Expression>> = Vec::new();
        let mut current_row: Vec<Expression> = Vec::new();

        // Parse matrix content
        loop {
            // Check for end of environment
            if let Some((LatexToken::EndEnv(end_name), _)) = self.peek() {
                let end_name = end_name.clone();
                self.next(); // consume EndEnv

                // Validate matching environment name
                if end_name != env_name {
                    return Err(ParseError::custom(
                        format!(
                            "mismatched environment: \\begin{{{}}} ended with \\end{{{}}}",
                            env_name, end_name
                        ),
                        Some(self.current_span()),
                    ));
                }

                // Add last row if not empty
                if !current_row.is_empty() {
                    rows.push(current_row);
                }
                break;
            }

            // Parse expression
            let expr = self.parse_expression()?;
            current_row.push(expr);

            // Check what comes next
            match self.peek() {
                Some((LatexToken::Ampersand, _)) => {
                    self.next(); // consume &
                    // Continue parsing current row
                }
                Some((LatexToken::DoubleBackslash, _)) => {
                    self.next(); // consume \\
                    // End current row and start new one
                    rows.push(current_row);
                    current_row = Vec::new();
                }
                Some((LatexToken::EndEnv(_), _)) => {
                    // Will be handled in next iteration
                }
                Some((token, span)) => {
                    return Err(ParseError::unexpected_token(
                        vec!["&", "\\\\", "\\end"],
                        format!("{:?}", token),
                        Some(*span),
                    ));
                }
                None => {
                    return Err(ParseError::unexpected_eof(
                        vec!["&", "\\\\", "\\end"],
                        Some(self.current_span()),
                    ));
                }
            }
        }

        // Validate all rows have the same number of columns
        if !rows.is_empty() {
            let first_col_count = rows[0].len();
            for (i, row) in rows.iter().enumerate() {
                if row.len() != first_col_count {
                    return Err(ParseError::custom(
                        format!(
                            "inconsistent matrix row lengths: row 0 has {} columns, row {} has {} columns",
                            first_col_count, i, row.len()
                        ),
                        Some(self.current_span()),
                    ));
                }
            }
        }

        // Convert single-column matrices to vectors
        if !rows.is_empty() && rows[0].len() == 1 {
            // All rows have exactly 1 column - this is a column vector
            let elements: Vec<Expression> = rows.into_iter().map(|mut row| row.remove(0)).collect();
            Ok(Expression::Vector(elements))
        } else {
            // Regular matrix
            Ok(Expression::Matrix(rows))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_number() {
        let expr = parse_latex("42").unwrap();
        assert_eq!(expr, Expression::Integer(42));
    }

    #[test]
    fn test_parse_float() {
        let expr = parse_latex("3.14").unwrap();
        match expr {
            Expression::Float(f) => {
                assert!((f.value() - 3.14).abs() < 1e-10);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_latex("x").unwrap();
        assert_eq!(expr, Expression::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_latex("1 + 2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse_latex("5 - 3").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Sub);
                assert_eq!(*left, Expression::Integer(5));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse_latex("2 * 3").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Mul);
                assert_eq!(*left, Expression::Integer(2));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_division() {
        let expr = parse_latex("6 / 2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Div);
                assert_eq!(*left, Expression::Integer(6));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_power() {
        let expr = parse_latex("x^2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Pow);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_power_braced() {
        let expr = parse_latex("x^{2+3}").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Add, ..
                    } => {}
                    _ => panic!("Expected addition in exponent"),
                }
            }
            _ => panic!("Expected power expression"),
        }
    }

    #[test]
    fn test_parse_frac() {
        let expr = parse_latex(r"\frac{1}{2}").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Div);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary division"),
        }
    }

    #[test]
    fn test_parse_sqrt() {
        let expr = parse_latex(r"\sqrt{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sqrt_nth() {
        let expr = parse_latex(r"\sqrt[3]{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "root");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
                assert_eq!(args[1], Expression::Integer(3));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_subscript() {
        let expr = parse_latex("x_1").unwrap();
        assert_eq!(expr, Expression::Variable("x_1".to_string()));
    }

    #[test]
    fn test_parse_subscript_braced() {
        let expr = parse_latex("x_{12}").unwrap();
        assert_eq!(expr, Expression::Variable("x_12".to_string()));
    }

    #[test]
    fn test_parse_greek_letter() {
        let expr = parse_latex(r"\alpha").unwrap();
        assert_eq!(expr, Expression::Variable("alpha".to_string()));
    }

    #[test]
    fn test_parse_pi_constant() {
        let expr = parse_latex(r"\pi").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Pi));
    }

    #[test]
    fn test_parse_infinity() {
        let expr = parse_latex(r"\infty").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_sin() {
        let expr = parse_latex(r"\sin{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sin_unbraced() {
        let expr = parse_latex(r"\sin x").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sin_parentheses() {
        let expr = parse_latex(r"\sin(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse_latex("(1 + 2)").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_absolute_value() {
        let expr = parse_latex("|x|").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "abs");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_unary_minus() {
        let expr = parse_latex("-x").unwrap();
        match expr {
            Expression::Unary { op, operand } => {
                assert_eq!(op, crate::ast::UnaryOp::Neg);
                assert_eq!(*operand, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected unary expression"),
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        // (2 + 3) * 4
        let expr = parse_latex("(2 + 3) * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Add, ..
                    } => {}
                    _ => panic!("Expected addition in left"),
                }
                assert_eq!(*right, Expression::Integer(4));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_operator_precedence() {
        // 2 + 3 * 4 should be 2 + (3 * 4)
        let expr = parse_latex("2 + 3 * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Mul, ..
                    } => {}
                    _ => panic!("Expected multiplication in right"),
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_power_precedence() {
        // 2 * x^3 should be 2 * (x^3)
        let expr = parse_latex("2 * x^3").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Pow, ..
                    } => {}
                    _ => panic!("Expected power in right"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_nested_frac() {
        // \frac{\frac{1}{2}}{3}
        let expr = parse_latex(r"\frac{\frac{1}{2}}{3}").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Div, ..
                    } => {}
                    _ => panic!("Expected nested division"),
                }
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected division"),
        }
    }

    // Relation tests

    #[test]
    fn test_latex_simple_equation() {
        let expr = parse_latex("x = 5").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_latex_inequality_less() {
        let expr = parse_latex("x < 5").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Lt);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_less_command() {
        let expr = parse_latex(r"x \lt 5").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Lt);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_greater() {
        let expr = parse_latex("x > 0").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Gt);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_greater_command() {
        let expr = parse_latex(r"x \gt 0").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Gt);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_leq() {
        let expr = parse_latex(r"x \leq 3").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Le);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_le() {
        let expr = parse_latex(r"x \le 3").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Le);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_geq() {
        let expr = parse_latex(r"x \geq -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_ge() {
        let expr = parse_latex(r"x \ge -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_neq() {
        let expr = parse_latex(r"x \neq 0").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Ne);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_ne() {
        let expr = parse_latex(r"a \ne b").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ne);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_complex_equation() {
        // \frac{x}{2} = 3
        let expr = parse_latex(r"\frac{x}{2} = 3").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_latex_complex_inequality() {
        // a + b < c + d
        let expr = parse_latex("a + b < c + d").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Lt);
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_chained_relation_error() {
        // a < b < c should error
        let result = parse_latex("a < b < c");
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("chained relations"));
        }
    }

    #[test]
    fn test_latex_relation_precedence() {
        // 2 + 3 = 5 should parse as (2 + 3) = 5
        let expr = parse_latex("2 + 3 = 5").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Equation variant"),
        }
    }
}
