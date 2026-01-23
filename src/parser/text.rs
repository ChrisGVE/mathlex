//! Plain text mathematical expression parser.
//!
//! This module implements a recursive descent parser for plain text
//! mathematical notation. It takes tokens from the tokenizer and builds an AST.
//!
//! # Operator Precedence (lowest to highest)
//!
//! 1. Relations (=, <, >, <=, >=, !=)
//! 2. Addition, Subtraction (+, -)
//! 3. Multiplication, Division, Modulo (*, /, %)
//! 4. Unary prefix operators (-, +)
//! 5. Power (^) - RIGHT ASSOCIATIVE
//! 6. Postfix operators (!)
//! 7. Function calls and atoms
//!
//! # Examples
//!
//! ```
//! use mathlex::parser::parse;
//!
//! let expr = parse("2 + 3 * 4").unwrap();
//! // Parses as: 2 + (3 * 4)
//!
//! let expr = parse("2^3^4").unwrap();
//! // Parses as: 2^(3^4) - right associative
//! ```

use crate::ast::{BinaryOp, Expression, InequalityOp, MathConstant, MathFloat, UnaryOp};
use crate::error::{ParseError, ParseResult, Span};
use crate::parser::tokenizer::{tokenize, SpannedToken, Token};
use crate::ParserConfig;

/// Parses a plain text mathematical expression with default configuration.
///
/// # Arguments
///
/// * `input` - The mathematical expression string to parse
///
/// # Returns
///
/// A `ParseResult<Expression>` containing the parsed AST or an error.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse;
///
/// let expr = parse("sin(x) + 2").unwrap();
/// ```
pub fn parse(input: &str) -> ParseResult<Expression> {
    parse_with_config(input, &ParserConfig::default())
}

/// Parses a plain text mathematical expression with custom configuration.
///
/// # Arguments
///
/// * `input` - The mathematical expression string to parse
/// * `config` - Parser configuration options
///
/// # Returns
///
/// A `ParseResult<Expression>` containing the parsed AST or an error.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_with_config;
/// use mathlex::ParserConfig;
///
/// let config = ParserConfig {
///     implicit_multiplication: true,
///     ..Default::default()
/// };
/// let expr = parse_with_config("2x", &config).unwrap();
/// ```
pub fn parse_with_config(input: &str, config: &ParserConfig) -> ParseResult<Expression> {
    let tokens = tokenize(input)?;
    let parser = TextParser::new(tokens, config.clone());
    parser.parse()
}

/// Internal parser state for text expressions.
struct TextParser {
    /// Token stream with positions
    tokens: Vec<SpannedToken>,
    /// Current position in token stream
    pos: usize,
    /// Parser configuration
    config: ParserConfig,
}

impl TextParser {
    /// Creates a new parser from a token stream.
    fn new(tokens: Vec<SpannedToken>, config: ParserConfig) -> Self {
        Self {
            tokens,
            pos: 0,
            config,
        }
    }

    /// Returns the current token without consuming it.
    fn peek(&self) -> Option<&SpannedToken> {
        self.tokens.get(self.pos)
    }

    /// Returns the current token and advances position.
    fn next(&mut self) -> Option<SpannedToken> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Returns the current position/span for error reporting.
    fn current_span(&self) -> Span {
        self.peek().map(|token| token.span).unwrap_or_else(|| {
            // Use the last token's end position if we're at EOF
            if let Some(last_token) = self.tokens.last() {
                Span::at(last_token.span.end)
            } else {
                Span::start()
            }
        })
    }

    /// Checks if current token matches a pattern without consuming.
    fn check(&self, expected: &Token) -> bool {
        self.peek()
            .map(|token| &token.value == expected)
            .unwrap_or(false)
    }

    /// Determines if implicit multiplication should be inserted.
    ///
    /// Returns true when:
    /// - Config has implicit_multiplication enabled AND
    /// - Next token is an identifier or left parenthesis
    ///
    /// This enables natural mathematical notation:
    ///   - `2x` → `2*x`
    ///   - `2(x+1)` → `2*(x+1)`
    ///   - `x y` → `x*y`
    ///   - `x y z` → `(x*y)*z`
    ///   - `(a+b)(c+d)` → `(a+b)*(c+d)`
    fn should_insert_implicit_mult(&self, _left: &Expression) -> bool {
        if !self.config.implicit_multiplication {
            return false;
        }

        let next_token = match self.peek() {
            Some(token) => &token.value,
            None => return false,
        };

        // Allow implicit multiplication when followed by identifier or lparen
        matches!(next_token, Token::Identifier(_) | Token::LParen)
    }

    /// Consumes a token if it matches the expected token.
    fn consume(&mut self, expected: Token) -> ParseResult<Span> {
        if let Some(token) = self.next() {
            if token.value == expected {
                Ok(token.span)
            } else {
                Err(ParseError::unexpected_token(
                    vec![format!("{}", expected)],
                    format!("{}", token.value),
                    Some(token.span),
                ))
            }
        } else {
            Err(ParseError::unexpected_eof(
                vec![format!("{}", expected)],
                Some(self.current_span()),
            ))
        }
    }

    /// Main entry point for parsing.
    fn parse(mut self) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;

        // Ensure we consumed all tokens
        if self.peek().is_some() {
            let token = self.peek().unwrap();
            return Err(ParseError::unexpected_token(
                vec!["end of input"],
                format!("{}", token.value),
                Some(token.span),
            ));
        }

        Ok(expr)
    }

    /// Parses an expression (entry point for recursive descent).
    fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_relation()
    }

    /// Parses relational expressions (=, <, >, <=, >=, !=).
    fn parse_relation(&mut self) -> ParseResult<Expression> {
        let left = self.parse_additive()?;

        // Check for relation operator
        if let Some(token) = self.peek() {
            let relation = match &token.value {
                Token::Equals => Some(None), // None indicates equation
                Token::Less => Some(Some(InequalityOp::Lt)),
                Token::Greater => Some(Some(InequalityOp::Gt)),
                Token::LessEq => Some(Some(InequalityOp::Le)),
                Token::GreaterEq => Some(Some(InequalityOp::Ge)),
                Token::NotEquals => Some(Some(InequalityOp::Ne)),
                _ => None,
            };

            if let Some(rel_op) = relation {
                self.next(); // consume relation operator
                let right = self.parse_additive()?;

                // Check for chained relations and error if found
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
                            "chained relations are not supported; use explicit grouping (e.g., (a < b) and (b < c))".to_string(),
                            Some(next_token.span),
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

        while let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
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

    /// Parses multiplicative expressions (*, /, %).
    fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_unary()?;

        loop {
            // Check for explicit multiplication operators
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
                // Explicit operator found
                self.next(); // consume operator
                let right = self.parse_unary()?;

                left = Expression::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else if self.should_insert_implicit_mult(&left) {
                // Implicit multiplication detected
                let right = self.parse_unary()?;

                left = Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(left),
                    right: Box::new(right),
                };
            } else {
                // No more multiplication operations
                break;
            }
        }

        Ok(left)
    }

    /// Parses unary prefix operators (-, +).
    fn parse_unary(&mut self) -> ParseResult<Expression> {
        if let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Minus => Some(UnaryOp::Neg),
                Token::Plus => Some(UnaryOp::Pos),
                _ => None,
            };

            if let Some(op) = op {
                self.next(); // consume operator
                let operand = self.parse_unary()?; // allow multiple unary operators

                return Ok(Expression::Unary {
                    op,
                    operand: Box::new(operand),
                });
            }
        }

        self.parse_power()
    }

    /// Parses power expressions (^ or **) - right associative.
    fn parse_power(&mut self) -> ParseResult<Expression> {
        let left = self.parse_postfix()?;

        if self.check(&Token::Caret) || self.check(&Token::DoubleStar) {
            self.next(); // consume ^ or **
            let right = self.parse_power()?; // right associative - recurse

            Ok(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(left),
                right: Box::new(right),
            })
        } else {
            Ok(left)
        }
    }

    /// Parses postfix operators (!).
    fn parse_postfix(&mut self) -> ParseResult<Expression> {
        let mut expr = self.parse_primary()?;

        while self.check(&Token::Bang) {
            self.next(); // consume !
            expr = Expression::Unary {
                op: UnaryOp::Factorial,
                operand: Box::new(expr),
            };
        }

        Ok(expr)
    }

    /// Parses primary expressions (atoms, functions, parenthesized).
    fn parse_primary(&mut self) -> ParseResult<Expression> {
        let token = self.peek().ok_or_else(|| {
            ParseError::unexpected_eof(vec!["expression"], Some(self.current_span()))
        })?;

        match &token.value {
            Token::Integer(n) => {
                let value = *n;
                self.next();
                Ok(Expression::Integer(value))
            }
            Token::Float(f) => {
                let value = *f;
                self.next();
                Ok(Expression::Float(MathFloat::from(value)))
            }
            Token::Identifier(name) => {
                let mut name = name.clone();
                self.next();

                // Check for subscript (followed by '_')
                if self.check(&Token::Underscore) {
                    self.next(); // consume '_'

                    // Parse subscript: single char/number or identifier
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

                    // Combine identifier with subscript
                    name = format!("{}_{}", name, subscript);
                }

                // Check if it's a function call (followed by '(')
                if self.check(&Token::LParen) {
                    self.parse_function_args(name)
                } else {
                    // It's a variable or constant
                    Ok(self.identifier_to_expression(name))
                }
            }
            Token::Pi => {
                self.next();
                Ok(Expression::Constant(MathConstant::Pi))
            }
            Token::Infinity => {
                self.next();
                Ok(Expression::Constant(MathConstant::Infinity))
            }
            Token::Sqrt => {
                self.next();
                // Parse √x or √(expr)
                let arg = if self.check(&Token::LParen) {
                    self.next(); // consume (
                    let expr = self.parse_expression()?;
                    self.consume(Token::RParen)?;
                    expr
                } else {
                    // Parse a single primary expression (number, variable, or function call)
                    self.parse_primary()?
                };
                Ok(Expression::Function {
                    name: "sqrt".to_string(),
                    args: vec![arg],
                })
            }
            Token::LParen => {
                self.next(); // consume (
                let expr = self.parse_expression()?;
                self.consume(Token::RParen)?;
                Ok(expr)
            }
            _ => {
                let token = self.next().unwrap();
                Err(ParseError::unexpected_token(
                    vec!["number", "variable", "("],
                    format!("{}", token.value),
                    Some(token.span),
                ))
            }
        }
    }

    /// Parses function arguments.
    fn parse_function_args(&mut self, name: String) -> ParseResult<Expression> {
        self.consume(Token::LParen)?;

        let mut args = Vec::new();

        // Check for empty argument list and reject it
        if self.check(&Token::RParen) {
            let span = self.current_span();
            return Err(ParseError::unexpected_token(
                vec!["expression"],
                ")".to_string(),
                Some(span),
            ));
        }

        // Parse first argument
        args.push(self.parse_expression()?);

        // Parse remaining arguments
        while self.check(&Token::Comma) {
            self.next(); // consume comma
            args.push(self.parse_expression()?);
        }

        self.consume(Token::RParen)?;

        Ok(Expression::Function { name, args })
    }

    /// Converts an identifier to the appropriate expression (constant or variable).
    fn identifier_to_expression(&self, name: String) -> Expression {
        match name.as_str() {
            "pi" => Expression::Constant(MathConstant::Pi),
            "e" => Expression::Constant(MathConstant::E),
            "i" => Expression::Constant(MathConstant::I),
            "inf" => Expression::Constant(MathConstant::Infinity),
            _ => Expression::Variable(name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer() {
        let expr = parse("42").unwrap();
        assert_eq!(expr, Expression::Integer(42));
    }

    #[test]
    fn test_parse_negative_integer() {
        let expr = parse("-17").unwrap();
        assert!(matches!(
            expr,
            Expression::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_float() {
        let expr = parse("3.14").unwrap();
        assert!(matches!(expr, Expression::Float(_)));
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse("x").unwrap();
        assert_eq!(expr, Expression::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_constant_pi() {
        let expr = parse("pi").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Pi));
    }

    #[test]
    fn test_parse_constant_e() {
        let expr = parse("e").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::E));
    }

    #[test]
    fn test_parse_constant_i() {
        let expr = parse("i").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::I));
    }

    #[test]
    fn test_parse_constant_inf() {
        let expr = parse("inf").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_simple_addition() {
        let expr = parse("2 + 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_simple_subtraction() {
        let expr = parse("5 - 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_simple_multiplication() {
        let expr = parse("2 * 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_simple_division() {
        let expr = parse("6 / 2").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_modulo() {
        let expr = parse("7 % 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Mod,
                ..
            }
        ));
    }

    #[test]
    fn test_parse_power() {
        let expr = parse("2 ^ 3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Pow,
                ..
            }
        ));
    }

    #[test]
    fn test_operator_precedence_mul_over_add() {
        // 2 + 3 * 4 should parse as 2 + (3 * 4)
        let expr = parse("2 + 3 * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_operator_precedence_power_over_mul() {
        // 2 * 3 ^ 4 should parse as 2 * (3 ^ 4)
        let expr = parse("2 * 3 ^ 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication at top level"),
        }
    }

    #[test]
    fn test_power_right_associative() {
        // 2 ^ 3 ^ 4 should parse as 2 ^ (3 ^ 4)
        let expr = parse("2 ^ 3 ^ 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected power at top level"),
        }
    }

    #[test]
    fn test_parse_double_star_power() {
        let expr = parse("2**3").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Pow,
                ..
            }
        ));
    }

    #[test]
    fn test_double_star_with_variable() {
        let expr = parse("x**2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected power"),
        }
    }

    #[test]
    fn test_double_star_right_associative() {
        // 2**3**4 should parse as 2**(3**4)
        let expr = parse("2**3**4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected power at top level"),
        }
    }

    #[test]
    fn test_star_vs_double_star_in_expression() {
        // 2*3**4 should parse as 2*(3**4)
        let expr = parse("2*3**4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication at top level"),
        }
    }

    #[test]
    fn test_mixed_caret_and_double_star() {
        // 2^3**4 should parse as 2^(3**4)
        let expr = parse("2^3**4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected power at top level"),
        }
    }

    #[test]
    fn test_parentheses_override_precedence() {
        // (2 + 3) * 4 should parse as (2 + 3) * 4
        let expr = parse("(2 + 3) * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(*right, Expression::Integer(4)));
            }
            _ => panic!("Expected multiplication at top level"),
        }
    }

    #[test]
    fn test_unary_negation() {
        let expr = parse("-5").unwrap();
        assert!(matches!(
            expr,
            Expression::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn test_unary_positive() {
        let expr = parse("+5").unwrap();
        assert!(matches!(
            expr,
            Expression::Unary {
                op: UnaryOp::Pos,
                ..
            }
        ));
    }

    #[test]
    fn test_factorial() {
        let expr = parse("5!").unwrap();
        assert!(matches!(
            expr,
            Expression::Unary {
                op: UnaryOp::Factorial,
                ..
            }
        ));
    }

    #[test]
    fn test_double_factorial() {
        let expr = parse("5!!").unwrap();
        // Should parse as (5!)!
        match expr {
            Expression::Unary {
                op: UnaryOp::Factorial,
                operand,
            } => {
                assert!(matches!(
                    *operand,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
            }
            _ => panic!("Expected factorial"),
        }
    }

    #[test]
    fn test_function_call_no_args_errors() {
        // Empty argument list should be rejected
        let result = parse("f()");
        assert!(result.is_err());
    }

    #[test]
    fn test_function_call_one_arg() {
        let expr = parse("sin(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(_)));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_function_call_multiple_args() {
        let expr = parse("max(1, 2, 3)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_nested_function_calls() {
        let expr = parse("sin(cos(x))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Function { .. }));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_function_with_expression_args() {
        let expr = parse("pow(x, 2 + 3)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expression::Variable(_)));
                assert!(matches!(args[1], Expression::Binary { .. }));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_complex_expression() {
        // (2 + 3) * sin(x) - 4^2
        let expr = parse("(2 + 3) * sin(x) - 4^2").unwrap();
        // Should parse correctly without panicking
        assert!(matches!(expr, Expression::Binary { .. }));
    }

    // Comprehensive function tests

    #[test]
    fn test_trig_functions() {
        // sin(x)
        let expr = parse("sin(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        // cos(2*pi)
        let expr = parse("cos(2*pi)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "cos");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Binary { .. }));
            }
            _ => panic!("Expected function"),
        }

        // tan(x)
        let expr = parse("tan(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => {
                assert_eq!(name, "tan");
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_inverse_trig_functions() {
        // asin(x)
        let expr = parse("asin(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "asin"),
            _ => panic!("Expected function"),
        }

        // acos(x)
        let expr = parse("acos(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "acos"),
            _ => panic!("Expected function"),
        }

        // atan(x)
        let expr = parse("atan(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "atan"),
            _ => panic!("Expected function"),
        }

        // atan2(y, x) - two arguments
        let expr = parse("atan2(y, x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "atan2");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_hyperbolic_functions() {
        // sinh(x)
        let expr = parse("sinh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sinh"),
            _ => panic!("Expected function"),
        }

        // cosh(x)
        let expr = parse("cosh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "cosh"),
            _ => panic!("Expected function"),
        }

        // tanh(x)
        let expr = parse("tanh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "tanh"),
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_logarithmic_functions() {
        // log(2, 8) - two arguments
        let expr = parse("log(2, 8)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "log");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }

        // ln(x)
        let expr = parse("ln(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "ln");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        // exp(-x)
        let expr = parse("exp(-x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "exp");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Unary { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_other_functions() {
        // sqrt(x)
        let expr = parse("sqrt(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sqrt"),
            _ => panic!("Expected function"),
        }

        // abs(x)
        let expr = parse("abs(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "abs"),
            _ => panic!("Expected function"),
        }

        // floor(x)
        let expr = parse("floor(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "floor"),
            _ => panic!("Expected function"),
        }

        // ceil(x)
        let expr = parse("ceil(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "ceil"),
            _ => panic!("Expected function"),
        }

        // sgn(x)
        let expr = parse("sgn(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sgn"),
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_multi_argument_functions() {
        // max(a, b)
        let expr = parse("max(a, b)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }

        // min(a, b, c) - three arguments
        let expr = parse("min(a, b, c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "min");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_deeply_nested_functions() {
        // sin(cos(x))
        let expr = parse("sin(cos(x))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "cos");
                        assert_eq!(args.len(), 1);
                    }
                    _ => panic!("Expected nested function"),
                }
            }
            _ => panic!("Expected function"),
        }

        // max(min(a, b), c)
        let expr = parse("max(min(a, b), c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 2);
                match &args[0] {
                    Expression::Function { name, .. } => assert_eq!(name, "min"),
                    _ => panic!("Expected nested function"),
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_functions_with_complex_expressions() {
        // sin(x + y)
        let expr = parse("sin(x + y)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }

        // log(2, x^2 + 1)
        let expr = parse("log(2, x^2 + 1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "log");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expression::Integer(2)));
                assert!(matches!(
                    args[1],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }

        // sqrt(x^2 + y^2)
        let expr = parse("sqrt(x^2 + y^2)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_custom_function_names() {
        // myFunc(x) - unknown function name should be preserved
        let expr = parse("myFunc(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "myFunc");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        // customFunction(a, b, c)
        let expr = parse("customFunction(a, b, c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "customFunction");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_in_complex_expression() {
        // 2 * sin(x) + cos(y)
        let expr = parse("2 * sin(x) + cos(y)").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Mul, ..
                    } => {}
                    _ => panic!("Expected multiplication on left"),
                }
                match *right {
                    Expression::Function { name, .. } => assert_eq!(name, "cos"),
                    _ => panic!("Expected function on right"),
                }
            }
            _ => panic!("Expected binary addition"),
        }

        // pow(x, 2) equivalent to x^2 but as function
        let expr = parse("pow(x, 2)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_nested_parentheses() {
        let expr = parse("((2 + 3) * (4 + 5))").unwrap();
        assert!(matches!(expr, Expression::Binary { .. }));
    }

    #[test]
    fn test_multiple_unary_operators() {
        let expr = parse("--5").unwrap();
        // Should parse as -(-5)
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert!(matches!(
                    *operand,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("Expected negation"),
        }
    }

    #[test]
    fn test_whitespace_handling() {
        let expr1 = parse("2+3").unwrap();
        let expr2 = parse("2 + 3").unwrap();
        let expr3 = parse("  2   +   3  ").unwrap();

        assert_eq!(expr1, expr2);
        assert_eq!(expr2, expr3);
    }

    #[test]
    fn test_empty_string() {
        let result = parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_trailing_operator() {
        let result = parse("2 +");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_operand() {
        let result = parse("+ 2");
        // This is valid as unary plus
        assert!(result.is_ok());
    }

    #[test]
    fn test_unmatched_parenthesis() {
        let result = parse("(2 + 3");
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_closing_parenthesis() {
        let result = parse("2 + 3)");
        assert!(result.is_err());
    }

    // Relation tests

    #[test]
    fn test_parse_simple_equation() {
        let expr = parse("x = 5").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_parse_inequality_less_than() {
        let expr = parse("x < 5").unwrap();
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
    fn test_parse_inequality_greater_than() {
        let expr = parse("x > 0").unwrap();
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
    fn test_parse_inequality_less_equal() {
        let expr = parse("x <= 3").unwrap();
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
    fn test_parse_inequality_greater_equal() {
        let expr = parse("x >= -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_parse_inequality_not_equal() {
        let expr = parse("x != 0").unwrap();
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
    fn test_parse_inequality_unicode_le() {
        let expr = parse("x ≤ 3").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Le);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_parse_inequality_unicode_ge() {
        let expr = parse("x ≥ -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_parse_inequality_unicode_ne() {
        let expr = parse("a ≠ b").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ne);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_parse_complex_equation() {
        // 2*x + 1 = 5
        let expr = parse("2*x + 1 = 5").unwrap();
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

    #[test]
    fn test_parse_complex_inequality() {
        // a + b < c + d
        let expr = parse("a + b < c + d").unwrap();
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
    fn test_chained_relation_error() {
        // a < b < c should error
        let result = parse("a < b < c");
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("chained relations"));
        }
    }

    #[test]
    fn test_relation_precedence_over_addition() {
        // 2 + 3 = 5 should parse as (2 + 3) = 5
        let expr = parse("2 + 3 = 5").unwrap();
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

    #[test]
    fn test_relation_with_parentheses() {
        // (x + 1) > y
        let expr = parse("(x + 1) > y").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Gt);
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("y".to_string()));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    // Implicit multiplication tests

    #[test]
    fn test_implicit_mult_number_variable() {
        // 2x should parse as 2*x
        let config = ParserConfig::default();
        let expr = parse_with_config("2x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_float_variable() {
        // 3.14r should parse as 3.14*r
        let config = ParserConfig::default();
        let expr = parse_with_config("3.14r", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Float(_)));
                assert_eq!(*right, Expression::Variable("r".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_parens() {
        // 2(x+1) should parse as 2*(x+1)
        let config = ParserConfig::default();
        let expr = parse_with_config("2(x+1)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_variable_variable() {
        // x y (with space) should parse as x*y
        // Note: xy (without space) is a single identifier per tokenizer rules
        let config = ParserConfig::default();
        let expr = parse_with_config("x y", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Variable("y".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_variable_chain() {
        // x y z (with spaces) should parse as (x*y)*z due to left-associativity
        // Note: xyz (without spaces) is a single identifier per tokenizer rules
        let config = ParserConfig::default();
        let expr = parse_with_config("x y z", &config).unwrap();
        // Due to left-associativity, this will be (x*y)*z
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("z".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_constant_variable() {
        // pi x should parse as pi*x
        let config = ParserConfig::default();
        let expr = parse_with_config("pi x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Constant(MathConstant::Pi));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    #[ignore] // TODO: Requires tracking parenthesized expressions through parser state
    fn test_implicit_mult_parens_parens() {
        // (a)(b) should parse as (a)*(b)
        // Currently not working - needs enhancement to track when expr came from parens
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)(b)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                assert_eq!(*right, Expression::Variable("b".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parens_variable() {
        // (a)x should parse as (a)*x
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_complex_expression() {
        // 2x + 3y should parse as (2*x) + (3*y)
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 3y", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_implicit_mult_with_power() {
        // 2x^2 should parse as 2*(x^2)
        let config = ParserConfig::default();
        let expr = parse_with_config("2x^2", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_no_implicit_mult_function_call() {
        // sin(x) should remain a function call, NOT s*i*n*(x)
        let config = ParserConfig::default();
        let expr = parse_with_config("sin(x)", &config).unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function call, not implicit multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_disabled() {
        // With implicit multiplication disabled, 2x should fail
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        let result = parse_with_config("2x", &config);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // TODO: Requires resolving tokenizer multi-character identifier issue
    fn test_implicit_mult_mixed_with_explicit() {
        // 2x * 3y should parse as (2*x) * (3*y)
        // Currently fails because "3y" is tokenized as single identifier
        let config = ParserConfig::default();
        let expr = parse_with_config("2x * 3y", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parenthesized_sum() {
        // (a + b)(c + d) should parse as (a+b)*(c+d)
        let config = ParserConfig::default();
        let expr = parse_with_config("(a + b)(c + d)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
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
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_function() {
        // 2sin(x) should parse as 2*sin(x)
        let config = ParserConfig::default();
        let expr = parse_with_config("2sin(x)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Function { name, .. } => assert_eq!(name, "sin"),
                    _ => panic!("Expected function on right"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_default_config() {
        // Default config should have implicit multiplication enabled
        let expr = parse("2x").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul, ..
            } => {}
            _ => panic!("Expected multiplication with default config"),
        }
    }

    #[test]
    fn test_implicit_mult_precedence() {
        // 2x + 1 should parse as (2*x) + 1, not 2*(x+1)
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 1", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected addition at top level"),
        }
    }
}

// Additional comprehensive tests organized by category

#[cfg(test)]
mod extended_basic_arithmetic {
    use super::*;

    #[test]
    fn test_scientific_notation_positive_exponent() {
        let expr = parse("1.5e10").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 1.5e10);
            }
            _ => panic!("Expected float with scientific notation"),
        }
    }

    #[test]
    fn test_scientific_notation_negative_exponent() {
        let expr = parse("2.5e-5").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 2.5e-5);
            }
            _ => panic!("Expected float with negative exponent"),
        }
    }

    #[test]
    fn test_scientific_notation_uppercase_e() {
        let expr = parse("3.14E8").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 3.14e8);
            }
            _ => panic!("Expected float with uppercase E"),
        }
    }

    #[test]
    fn test_scientific_notation_with_positive_sign() {
        let expr = parse("1e+3").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 1000.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_very_large_integer() {
        let expr = parse("9223372036854775807").unwrap(); // i64::MAX
        assert!(matches!(expr, Expression::Integer(_)));
    }

    #[test]
    fn test_zero() {
        let expr = parse("0").unwrap();
        assert_eq!(expr, Expression::Integer(0));
    }

    #[test]
    fn test_zero_float() {
        let expr = parse("0.0").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 0.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_negative_zero() {
        let expr = parse("-0").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert_eq!(*operand, Expression::Integer(0));
            }
            _ => panic!("Expected negation of zero"),
        }
    }

    #[test]
    fn test_mixed_int_float_operations() {
        // 2 + 3.5
        let expr = parse("2 + 3.5").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(*right, Expression::Float(_)));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_division_by_zero_parses() {
        // Should parse (evaluation is not parser's concern)
        let expr = parse("1 / 0").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
    }
}

#[cfg(test)]
mod extended_operator_precedence {
    use super::*;

    #[test]
    fn test_unary_minus_with_power() {
        // -x^2 parses as -(x^2) following standard mathematical convention
        // where exponentiation has higher precedence than unary minus
        let expr = parse("-x^2").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                // The operand should be x^2
                match *operand {
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        left,
                        right,
                    } => {
                        assert_eq!(*left, Expression::Variable("x".to_string()));
                        assert_eq!(*right, Expression::Integer(2));
                    }
                    _ => panic!("Expected power as operand"),
                }
            }
            _ => panic!("Expected negation of power"),
        }
    }

    #[test]
    fn test_parenthesized_negation_with_power() {
        // (-x)^2 should parse as (-x)^2
        let expr = parse("(-x)^2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected power of negation"),
        }
    }

    #[test]
    fn test_factorial_then_addition() {
        // 5! + 1 should parse as (5!) + 1
        let expr = parse("5! + 1").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_factorial_then_multiplication() {
        // 5! * 2 should parse as (5!) * 2
        let expr = parse("5! * 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_factorial_then_division() {
        // 5! / 2 should parse as (5!) / 2
        let expr = parse("5! / 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_complex_precedence_chain() {
        // a + b * c ^ d should parse as a + (b * (c ^ d))
        let expr = parse("a + b * c ^ d").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        left: mul_left,
                        right: mul_right,
                    } => {
                        assert_eq!(*mul_left, Expression::Variable("b".to_string()));
                        assert!(matches!(
                            *mul_right,
                            Expression::Binary {
                                op: BinaryOp::Pow,
                                ..
                            }
                        ));
                    }
                    _ => panic!("Expected multiplication on right"),
                }
            }
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_left_associativity_of_subtraction() {
        // 10 - 5 - 2 should parse as (10 - 5) - 2 = 3, not 10 - (5 - 2) = 7
        let expr = parse("10 - 5 - 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Sub,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected subtraction"),
        }
    }

    #[test]
    fn test_left_associativity_of_division() {
        // 20 / 4 / 2 should parse as (20 / 4) / 2 = 2.5, not 20 / (4 / 2) = 10
        let expr = parse("20 / 4 / 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_multiple_unary_negations() {
        // ---5 should parse as -(-(-(5)))
        let expr = parse("---5").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match *operand {
                Expression::Unary {
                    op: UnaryOp::Neg,
                    operand: inner,
                } => {
                    assert!(matches!(
                        *inner,
                        Expression::Unary {
                            op: UnaryOp::Neg,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected nested negation"),
            },
            _ => panic!("Expected negation"),
        }
    }

    #[test]
    fn test_mixed_unary_operators() {
        // -+5 should parse as -(+(5))
        let expr = parse("-+5").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert!(matches!(
                    *operand,
                    Expression::Unary {
                        op: UnaryOp::Pos,
                        ..
                    }
                ));
            }
            _ => panic!("Expected negation of positive"),
        }
    }

    #[test]
    fn test_triple_factorial() {
        // 5!!! should parse as ((5!)!)!
        let expr = parse("5!!!").unwrap();
        let mut current = &expr;
        let mut factorial_count = 0;

        while let Expression::Unary {
            op: UnaryOp::Factorial,
            operand,
        } = current
        {
            factorial_count += 1;
            current = operand;
        }

        assert_eq!(factorial_count, 3);
        assert_eq!(*current, Expression::Integer(5));
    }

    #[test]
    fn test_negative_exponent() {
        // 2^-3 requires parentheses as 2^(-3)
        // Without parens, the - is ambiguous and not supported
        let result = parse("2^-3");
        assert!(result.is_err(), "2^-3 should require parentheses");

        // 2^(-3) should parse correctly
        let expr = parse("2^(-3)").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("Expected power"),
        }
    }

    #[test]
    fn test_power_zero_and_one() {
        // x^0
        let expr = parse("x^0").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected power"),
        }

        // x^1
        let expr = parse("x^1").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected power"),
        }
    }
}

#[cfg(test)]
mod extended_functions {
    use super::*;

    #[test]
    fn test_many_function_arguments() {
        // Test function with 10 arguments
        let expr = parse("f(a, b, c, d, e, f, g, h, i, j)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 10);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_deeply_nested_functions_three_levels() {
        // f(g(h(x)))
        let expr = parse("f(g(h(x)))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "g");
                        assert_eq!(args.len(), 1);
                        match &args[0] {
                            Expression::Function { name, args } => {
                                assert_eq!(name, "h");
                                assert_eq!(args.len(), 1);
                                assert_eq!(args[0], Expression::Variable("x".to_string()));
                            }
                            _ => panic!("Expected third level function"),
                        }
                    }
                    _ => panic!("Expected second level function"),
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_factorial_argument() {
        // sin(5!)
        let expr = parse("sin(5!)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_equation_argument() {
        // solve(x = 5) - equation as argument should parse
        let expr = parse("solve(x = 5)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "solve");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Equation { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_inequality_argument() {
        // filter(x > 0)
        let expr = parse("filter(x > 0)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "filter");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Inequality { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_all_operator_types() {
        // f(a+b, c*d, e^f, g!)
        let expr = parse("f(a+b, c*d, e^f, g!)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 4);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    args[1],
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    args[2],
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(
                    args[3],
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_nested_parentheses_argument() {
        // f(((x)))
        let expr = parse("f(((x)))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function"),
        }
    }
}

#[cfg(test)]
mod extended_implicit_multiplication {
    use super::*;

    #[test]
    fn test_implicit_mult_factorial_then_variable() {
        // 5!x should parse as (5!)*x
        let config = ParserConfig::default();
        let expr = parse_with_config("5!x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_factorial_then_parens() {
        // 5!(x+1) should parse as (5!)*(x+1)
        let config = ParserConfig::default();
        let expr = parse_with_config("5!(x+1)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
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
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_chained_three_variables() {
        // a b c should parse as (a*b)*c
        let config = ParserConfig::default();
        let expr = parse_with_config("a b c", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("c".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_constant_then_function() {
        // pi sin(x) should parse as pi*sin(x)
        let config = ParserConfig::default();
        let expr = parse_with_config("pi sin(x)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Constant(MathConstant::Pi));
                assert!(matches!(*right, Expression::Function { .. }));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_multiple_numbers_fails() {
        // 2 3 with implicit mult should fail (numbers separated by space are ambiguous)
        let config = ParserConfig::default();
        // Note: This actually tokenizes as two separate integers
        // and would need explicit multiplication. Parser should error on unexpected token.
        let result = parse_with_config("2 3", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_implicit_mult_power_chain() {
        // 2x^3 should parse as 2*(x^3), not (2*x)^3
        let config = ParserConfig::default();
        let expr = parse_with_config("2x^3", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_with_relation() {
        // 2x = 5 should parse as (2*x) = 5
        let config = ParserConfig::default();
        let expr = parse_with_config("2x = 5", &config).unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected equation"),
        }
    }
}

#[cfg(test)]
mod extended_error_cases {
    use super::*;

    #[test]
    fn test_error_leading_comma_in_function() {
        let result = parse("f(,x)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_trailing_comma_in_function() {
        let result = parse("f(x,)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_double_comma_in_function() {
        let result = parse("f(x,,y)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_lone_star_operator() {
        let result = parse("*");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_lone_slash_operator() {
        let result = parse("/");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_lone_caret_operator() {
        let result = parse("^");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_lone_percent_operator() {
        let result = parse("%");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_empty_parens_not_function() {
        // () without function name should error
        let result = parse("()");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_just_whitespace() {
        let result = parse("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_mismatched_brackets() {
        // Parser doesn't use brackets for primary expressions
        // but tokenizer recognizes them
        let result = parse("[1]");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_double_operators() {
        // Note: ** is now a valid power operator (2 ** 3 = 2^3)
        // And ++ parses as 2 + (+3), -- parses as 2 - (-3) since +/- are unary operators
        // Test actual invalid double operators
        let result = parse("2 */ 3");
        assert!(result.is_err(), "Mix of * and / should be an error");
        let result = parse("2 ^^ 3");
        assert!(result.is_err(), "Double caret should be an error");
    }

    #[test]
    fn test_error_triple_equals() {
        let result = parse("x === 5");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_incomplete_inequality() {
        let result = parse("x <");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parentheses_mismatch_extra_open() {
        let result = parse("((2 + 3)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parentheses_mismatch_extra_close() {
        let result = parse("(2 + 3))");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_function_name() {
        // Numbers can't start function names
        // With implicit mult, 2func(x) parses as 2*func(x) which is valid
        // Without implicit mult, it's an error
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        let result = parse_with_config("2func(x)", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_function_call() {
        let result = parse("sin(x");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_comma_outside_function() {
        let result = parse("x, y");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_deeply_nested_parentheses() {
        // 10 levels of nesting
        let expr = parse("((((((((((x))))))))))").unwrap();
        assert_eq!(expr, Expression::Variable("x".to_string()));
    }

    #[test]
    fn test_very_long_expression() {
        // Long chain of additions
        let expr = parse("1+2+3+4+5+6+7+8+9+10").unwrap();
        // Just verify it parses without error
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_nested_expression() {
        // Combining many features
        let expr = parse("2*sin(x^2 + 1)! - cos(y)/(z + 1)").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_many_nested_functions() {
        // Multiple function calls at same level
        let expr = parse("f(g(x), h(y), i(z), j(a), k(b))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 5);
                for arg in args {
                    assert!(matches!(arg, Expression::Function { .. }));
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_alternating_operators() {
        // a + b - c + d - e
        let expr = parse("a + b - c + d - e").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_precedence_expression() {
        // ((a + b) * c - d / e) ^ f
        let expr = parse("((a + b) * c - d / e) ^ f").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected power"),
        }
    }
    #[test]
    fn test_parse_unicode_pi() {
        let expr = parse("2*π").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(*right, Expression::Constant(MathConstant::Pi)));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_parse_unicode_infinity() {
        let expr = parse("∞").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_unicode_sqrt() {
        let expr = parse("√4").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Integer(4)));
            }
            _ => panic!("Expected sqrt function call"),
        }
    }

    #[test]
    fn test_parse_unicode_sqrt_with_parens() {
        let expr = parse("√(x+1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected sqrt function call"),
        }
    }

    // Tests for subscripts
    #[test]
    fn test_parse_subscript_with_single_digit() {
        let expr = parse("x_1").unwrap();
        assert_eq!(expr, Expression::Variable("x_1".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_identifier() {
        let expr = parse("alpha_i").unwrap();
        assert_eq!(expr, Expression::Variable("alpha_i".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multiple_digits() {
        let expr = parse("x_12").unwrap();
        assert_eq!(expr, Expression::Variable("x_12".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multi_char() {
        let expr = parse("x_ij").unwrap();
        assert_eq!(expr, Expression::Variable("x_ij".to_string()));
    }

    #[test]
    fn test_parse_subscript_in_expression() {
        let expr = parse("x_1 + y_2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x_1".to_string()));
                assert_eq!(*right, Expression::Variable("y_2".to_string()));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_parse_subscript_round_trip() {
        let input = "x_1";
        let expr = parse(input).unwrap();
        let output = format!("{}", expr);
        assert_eq!(output, input);
    }
}
