//! Plain text mathematical expression parser.
//!
//! This module implements a recursive descent parser for plain text
//! mathematical notation. It takes tokens from the tokenizer and builds an AST.
//!
//! # Operator Precedence (lowest to highest)
//!
//! 1. Addition, Subtraction (+, -)
//! 2. Multiplication, Division, Modulo (*, /, %)
//! 3. Power (^) - RIGHT ASSOCIATIVE
//! 4. Unary operators (-, +, !)
//! 5. Function calls and atoms
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

use crate::ast::{BinaryOp, Expression, MathConstant, MathFloat, UnaryOp};
use crate::error::{ParseError, ParseResult, Span};
use crate::parser::tokenizer::{tokenize, SpannedToken, Token};

/// Parses a plain text mathematical expression.
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
    let tokens = tokenize(input)?;
    let parser = TextParser::new(tokens);
    parser.parse()
}

/// Internal parser state for text expressions.
struct TextParser {
    /// Token stream with positions
    tokens: Vec<SpannedToken>,
    /// Current position in token stream
    pos: usize,
}

impl TextParser {
    /// Creates a new parser from a token stream.
    fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
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
        self.peek()
            .map(|token| token.span)
            .unwrap_or_else(|| {
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
        self.peek().map(|token| &token.value == expected).unwrap_or(false)
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
        self.parse_additive()
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
        let mut left = self.parse_power()?;

        while let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                Token::Percent => BinaryOp::Mod,
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

    /// Parses power expressions (^) - right associative.
    fn parse_power(&mut self) -> ParseResult<Expression> {
        let left = self.parse_postfix()?;

        if self.check(&Token::Caret) {
            self.next(); // consume ^
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
        let mut expr = self.parse_unary_prefix()?;

        while self.check(&Token::Bang) {
            self.next(); // consume !
            expr = Expression::Unary {
                op: UnaryOp::Factorial,
                operand: Box::new(expr),
            };
        }

        Ok(expr)
    }

    /// Parses prefix unary operators (-, +).
    fn parse_unary_prefix(&mut self) -> ParseResult<Expression> {
        if let Some(token) = self.peek() {
            let op = match &token.value {
                Token::Minus => Some(UnaryOp::Neg),
                Token::Plus => Some(UnaryOp::Pos),
                _ => None,
            };

            if let Some(op) = op {
                self.next(); // consume operator
                let operand = self.parse_unary_prefix()?; // allow multiple unary operators

                return Ok(Expression::Unary {
                    op,
                    operand: Box::new(operand),
                });
            }
        }

        self.parse_primary()
    }

    /// Parses primary expressions (atoms, functions, parenthesized).
    fn parse_primary(&mut self) -> ParseResult<Expression> {
        let token = self.peek().ok_or_else(|| {
            ParseError::unexpected_eof(
                vec!["expression"],
                Some(self.current_span()),
            )
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
                let name = name.clone();
                self.next();

                // Check if it's a function call (followed by '(')
                if self.check(&Token::LParen) {
                    self.parse_function_args(name)
                } else {
                    // It's a variable or constant
                    Ok(self.identifier_to_expression(name))
                }
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

        // Handle empty argument list
        if self.check(&Token::RParen) {
            self.next();
            return Ok(Expression::Function { name, args });
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
    fn test_function_call_no_args() {
        let expr = parse("f()").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 0);
            }
            _ => panic!("Expected function call"),
        }
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
}
