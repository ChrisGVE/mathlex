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

use crate::ast::{BinaryOp, Expression, MathConstant, MathFloat};
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
        self.peek()
            .map(|(_, span)| *span)
            .unwrap_or_else(|| {
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
        self.parse_additive()
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
                    LatexToken::LBrace => {
                        self.braced(|parser| parser.parse_expression())
                    }
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
            num_str.parse::<f64>()
                .map(|f| Expression::Float(MathFloat::from(f)))
                .map_err(|_| ParseError::invalid_number(num_str, "invalid float", Some(span)))
        } else {
            // Integer
            num_str.parse::<i64>()
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
            "alpha" | "beta" | "gamma" | "delta" | "epsilon" | "zeta" | "eta" | "theta" |
            "iota" | "kappa" | "lambda" | "mu" | "nu" | "xi" | "omicron" | "pi" | "rho" |
            "sigma" | "tau" | "upsilon" | "phi" | "chi" | "psi" | "omega" |
            "Gamma" | "Delta" | "Theta" | "Lambda" | "Xi" | "Pi" | "Sigma" | "Upsilon" |
            "Phi" | "Psi" | "Omega" => {
                // Special case: \pi is a constant
                if cmd == "pi" {
                    Ok(Expression::Constant(MathConstant::Pi))
                } else {
                    Ok(Expression::Variable(cmd.to_string()))
                }
            }

            // Trigonometric functions
            "sin" | "cos" | "tan" | "sec" | "csc" | "cot" |
            "arcsin" | "arccos" | "arctan" | "sinh" | "cosh" | "tanh" => {
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
            Expression::Binary { op: BinaryOp::Pow, left, right } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                match *right {
                    Expression::Binary { op: BinaryOp::Add, .. } => {}
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
            Expression::Binary { op: BinaryOp::Mul, left, right } => {
                match *left {
                    Expression::Binary { op: BinaryOp::Add, .. } => {}
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
            Expression::Binary { op: BinaryOp::Add, left, right } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary { op: BinaryOp::Mul, .. } => {}
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
            Expression::Binary { op: BinaryOp::Mul, left, right } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary { op: BinaryOp::Pow, .. } => {}
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
            Expression::Binary { op: BinaryOp::Div, left, right } => {
                match *left {
                    Expression::Binary { op: BinaryOp::Div, .. } => {}
                    _ => panic!("Expected nested division"),
                }
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected division"),
        }
    }
}
