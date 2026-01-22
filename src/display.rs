//! # Display Trait Implementations for AST (Plain Text)
//!
//! This module provides `std::fmt::Display` implementations for all AST types,
//! converting them back to plain text mathematical notation.
//!
//! ## Design Philosophy
//!
//! - **Minimal Parentheses**: Use operator precedence to minimize parenthesization
//! - **Readable Output**: Generate human-readable plain text
//! - **Round-trip Capable**: Output can be parsed back (though not guaranteed identical AST)
//!
//! ## Precedence Levels
//!
//! 1. Addition, Subtraction: precedence = 1
//! 2. Multiplication, Division, Modulo: precedence = 2
//! 3. Exponentiation: precedence = 3
//! 4. Unary operations: precedence = 4 (implicit, highest)
//!
//! ## Examples
//!
//! ```ignore
//! use mathlex::ast::{Expression, BinaryOp};
//!
//! // 2 + 3 * 4 → "2 + 3 * 4" (no parens needed)
//! let expr = Expression::Binary {
//!     op: BinaryOp::Add,
//!     left: Box::new(Expression::Integer(2)),
//!     right: Box::new(Expression::Binary {
//!         op: BinaryOp::Mul,
//!         left: Box::new(Expression::Integer(3)),
//!         right: Box::new(Expression::Integer(4)),
//!     }),
//! };
//! assert_eq!(format!("{}", expr), "2 + 3 * 4");
//! ```

use crate::ast::*;
use std::fmt;

/// Get the precedence level of a binary operator.
///
/// Lower numbers bind less tightly (evaluated later).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(precedence(BinaryOp::Add), 1);
/// assert_eq!(precedence(BinaryOp::Mul), 2);
/// assert_eq!(precedence(BinaryOp::Pow), 3);
/// ```
fn precedence(op: BinaryOp) -> u8 {
    match op {
        BinaryOp::Add | BinaryOp::Sub => 1,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 2,
        BinaryOp::Pow => 3,
    }
}

/// Determine if an expression needs parentheses when used as a child of a binary operation.
///
/// Parentheses are needed when:
/// - The child is a binary operation with lower precedence than the parent
/// - The child is on the right side of a non-commutative operation with equal precedence
///
/// # Arguments
///
/// - `child`: The child expression
/// - `parent_op`: The parent binary operator
/// - `is_right`: Whether the child is the right operand
///
/// # Examples
///
/// ```ignore
/// // (2 + 3) * 4 needs parens
/// // 2 * (3 + 4) needs parens
/// // 2 * 3 * 4 doesn't need parens
/// ```
fn needs_parens(child: &Expression, parent_op: BinaryOp, is_right: bool) -> bool {
    match child {
        Expression::Binary { op: child_op, .. } => {
            let parent_prec = precedence(parent_op);
            let child_prec = precedence(*child_op);

            // Lower precedence always needs parens
            if child_prec < parent_prec {
                return true;
            }

            // For equal precedence, handle associativity
            if child_prec == parent_prec {
                match (parent_op, *child_op) {
                    // Power is right-associative by convention: a^b^c means a^(b^c)
                    // So we need parens when power appears as operand of power
                    (BinaryOp::Pow, BinaryOp::Pow) => return true,
                    // Sub and Div are left-associative, so right side needs parens
                    (BinaryOp::Sub, BinaryOp::Sub) | (BinaryOp::Div, BinaryOp::Div) => {
                        return is_right
                    }
                    // Add and Mul are commutative, no parens needed
                    _ => {}
                }
            }

            false
        }
        // Unary expressions don't need parens (they have highest precedence)
        _ => false,
    }
}

// Display implementations for enums

impl fmt::Display for MathConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathConstant::Pi => write!(f, "pi"),
            MathConstant::E => write!(f, "e"),
            MathConstant::I => write!(f, "i"),
            MathConstant::Infinity => write!(f, "inf"),
            MathConstant::NegInfinity => write!(f, "-inf"),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Pow => write!(f, "^"),
            BinaryOp::Mod => write!(f, "%"),
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Pos => write!(f, "+"),
            UnaryOp::Factorial => write!(f, "!"),
            UnaryOp::Transpose => write!(f, "'"),
        }
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Left => write!(f, "-"),
            Direction::Right => write!(f, "+"),
            Direction::Both => write!(f, ""),
        }
    }
}

impl fmt::Display for InequalityOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InequalityOp::Lt => write!(f, "<"),
            InequalityOp::Le => write!(f, "<="),
            InequalityOp::Gt => write!(f, ">"),
            InequalityOp::Ge => write!(f, ">="),
            InequalityOp::Ne => write!(f, "!="),
        }
    }
}

impl fmt::Display for IntegralBounds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.lower, self.upper)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Integer(n) => write!(f, "{}", n),

            Expression::Float(x) => write!(f, "{}", x),

            Expression::Rational {
                numerator,
                denominator,
            } => {
                write!(f, "{}/{}", numerator, denominator)
            }

            Expression::Complex { real, imaginary } => {
                write!(f, "{} + {}i", real, imaginary)
            }

            Expression::Variable(name) => write!(f, "{}", name),

            Expression::Constant(c) => write!(f, "{}", c),

            Expression::Binary { op, left, right } => {
                // Determine if we need parentheses
                let left_needs_parens = needs_parens(left, *op, false);
                let right_needs_parens = needs_parens(right, *op, true);

                // Write left operand
                if left_needs_parens {
                    write!(f, "({})", left)?;
                } else {
                    write!(f, "{}", left)?;
                }

                // Write operator
                write!(f, " {} ", op)?;

                // Write right operand
                if right_needs_parens {
                    write!(f, "({})", right)?;
                } else {
                    write!(f, "{}", right)?;
                }

                Ok(())
            }

            Expression::Unary { op, operand } => {
                match op {
                    UnaryOp::Factorial | UnaryOp::Transpose => {
                        // Postfix operators - need parens for binary operands
                        if matches!(**operand, Expression::Binary { .. }) {
                            write!(f, "({}){}", operand, op)
                        } else {
                            write!(f, "{}{}", operand, op)
                        }
                    }
                    UnaryOp::Neg | UnaryOp::Pos => {
                        // Prefix operators - need parens for binary operands
                        if matches!(**operand, Expression::Binary { .. }) {
                            write!(f, "{}({})", op, operand)
                        } else {
                            write!(f, "{}{}", op, operand)
                        }
                    }
                }
            }

            Expression::Function { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }

            Expression::Derivative { expr, var, order } => {
                if *order == 1 {
                    write!(f, "d/d{}({})", var, expr)
                } else {
                    write!(f, "d^{}/d{}^{}({})", order, var, order, expr)
                }
            }

            Expression::PartialDerivative { expr, var, order } => {
                if *order == 1 {
                    write!(f, "∂/∂{}({})", var, expr)
                } else {
                    write!(f, "∂^{}/∂{}^{}({})", order, var, order, expr)
                }
            }

            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                if let Some(bounds) = bounds {
                    write!(f, "int({}, d{}, {})", integrand, var, bounds)
                } else {
                    write!(f, "int({}, d{})", integrand, var)
                }
            }

            Expression::Limit {
                expr,
                var,
                to,
                direction,
            } => {
                let dir_str = match direction {
                    Direction::Both => String::new(),
                    Direction::Left => "-".to_string(),
                    Direction::Right => "+".to_string(),
                };
                write!(f, "lim({}->{}{})({})", var, to, dir_str, expr)
            }

            Expression::Sum {
                index,
                lower,
                upper,
                body,
            } => {
                write!(f, "sum({}={}, {}, {})", index, lower, upper, body)
            }

            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                write!(f, "prod({}={}, {}, {})", index, lower, upper, body)
            }

            Expression::Vector(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }

            Expression::Matrix(rows) => {
                write!(f, "[")?;
                for (i, row) in rows.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for (j, elem) in row.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", elem)?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")
            }

            Expression::Equation { left, right } => {
                write!(f, "{} = {}", left, right)
            }

            Expression::Inequality { op, left, right } => {
                write!(f, "{} {} {}", left, op, right)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    // Tests for MathConstant Display

    #[test]
    fn test_math_constant_pi() {
        assert_eq!(format!("{}", MathConstant::Pi), "pi");
    }

    #[test]
    fn test_math_constant_e() {
        assert_eq!(format!("{}", MathConstant::E), "e");
    }

    #[test]
    fn test_math_constant_i() {
        assert_eq!(format!("{}", MathConstant::I), "i");
    }

    #[test]
    fn test_math_constant_infinity() {
        assert_eq!(format!("{}", MathConstant::Infinity), "inf");
    }

    #[test]
    fn test_math_constant_neg_infinity() {
        assert_eq!(format!("{}", MathConstant::NegInfinity), "-inf");
    }

    // Tests for BinaryOp Display

    #[test]
    fn test_binary_op_add() {
        assert_eq!(format!("{}", BinaryOp::Add), "+");
    }

    #[test]
    fn test_binary_op_sub() {
        assert_eq!(format!("{}", BinaryOp::Sub), "-");
    }

    #[test]
    fn test_binary_op_mul() {
        assert_eq!(format!("{}", BinaryOp::Mul), "*");
    }

    #[test]
    fn test_binary_op_div() {
        assert_eq!(format!("{}", BinaryOp::Div), "/");
    }

    #[test]
    fn test_binary_op_pow() {
        assert_eq!(format!("{}", BinaryOp::Pow), "^");
    }

    #[test]
    fn test_binary_op_mod() {
        assert_eq!(format!("{}", BinaryOp::Mod), "%");
    }

    // Tests for UnaryOp Display

    #[test]
    fn test_unary_op_neg() {
        assert_eq!(format!("{}", UnaryOp::Neg), "-");
    }

    #[test]
    fn test_unary_op_pos() {
        assert_eq!(format!("{}", UnaryOp::Pos), "+");
    }

    #[test]
    fn test_unary_op_factorial() {
        assert_eq!(format!("{}", UnaryOp::Factorial), "!");
    }

    #[test]
    fn test_unary_op_transpose() {
        assert_eq!(format!("{}", UnaryOp::Transpose), "'");
    }

    // Tests for Direction Display

    #[test]
    fn test_direction_left() {
        assert_eq!(format!("{}", Direction::Left), "-");
    }

    #[test]
    fn test_direction_right() {
        assert_eq!(format!("{}", Direction::Right), "+");
    }

    #[test]
    fn test_direction_both() {
        assert_eq!(format!("{}", Direction::Both), "");
    }

    // Tests for InequalityOp Display

    #[test]
    fn test_inequality_op_lt() {
        assert_eq!(format!("{}", InequalityOp::Lt), "<");
    }

    #[test]
    fn test_inequality_op_le() {
        assert_eq!(format!("{}", InequalityOp::Le), "<=");
    }

    #[test]
    fn test_inequality_op_gt() {
        assert_eq!(format!("{}", InequalityOp::Gt), ">");
    }

    #[test]
    fn test_inequality_op_ge() {
        assert_eq!(format!("{}", InequalityOp::Ge), ">=");
    }

    #[test]
    fn test_inequality_op_ne() {
        assert_eq!(format!("{}", InequalityOp::Ne), "!=");
    }

    // Tests for IntegralBounds Display

    #[test]
    fn test_integral_bounds_simple() {
        let bounds = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };
        assert_eq!(format!("{}", bounds), "0, 1");
    }

    #[test]
    fn test_integral_bounds_variables() {
        let bounds = IntegralBounds {
            lower: Box::new(Expression::Variable("a".to_string())),
            upper: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(format!("{}", bounds), "a, b");
    }

    // Tests for Expression::Integer Display

    #[test]
    fn test_expression_integer_positive() {
        let expr = Expression::Integer(42);
        assert_eq!(format!("{}", expr), "42");
    }

    #[test]
    fn test_expression_integer_negative() {
        let expr = Expression::Integer(-17);
        assert_eq!(format!("{}", expr), "-17");
    }

    #[test]
    fn test_expression_integer_zero() {
        let expr = Expression::Integer(0);
        assert_eq!(format!("{}", expr), "0");
    }

    // Tests for Expression::Float Display

    #[test]
    fn test_expression_float_positive() {
        let expr = Expression::Float(3.14.into());
        assert_eq!(format!("{}", expr), "3.14");
    }

    #[test]
    fn test_expression_float_negative() {
        let expr = Expression::Float((-2.5).into());
        assert_eq!(format!("{}", expr), "-2.5");
    }

    #[test]
    fn test_expression_float_scientific() {
        let expr = Expression::Float(1e-10.into());
        let output = format!("{}", expr);
        // Output may be in scientific or decimal notation
        assert!(!output.is_empty());
        // Verify the value is preserved (very small number)
        assert!(output.parse::<f64>().unwrap() < 1e-9);
    }

    // Tests for Expression::Rational Display

    #[test]
    fn test_expression_rational_simple() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(1)),
            denominator: Box::new(Expression::Integer(2)),
        };
        assert_eq!(format!("{}", expr), "1/2");
    }

    #[test]
    fn test_expression_rational_negative() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(-3)),
            denominator: Box::new(Expression::Integer(4)),
        };
        assert_eq!(format!("{}", expr), "-3/4");
    }

    #[test]
    fn test_expression_rational_variables() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Variable("a".to_string())),
            denominator: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(format!("{}", expr), "a/b");
    }

    // Tests for Expression::Complex Display

    #[test]
    fn test_expression_complex_simple() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(3)),
            imaginary: Box::new(Expression::Integer(4)),
        };
        assert_eq!(format!("{}", expr), "3 + 4i");
    }

    #[test]
    fn test_expression_complex_negative_imaginary() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(2)),
            imaginary: Box::new(Expression::Integer(-5)),
        };
        assert_eq!(format!("{}", expr), "2 + -5i");
    }

    #[test]
    fn test_expression_complex_pure_imaginary() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(0)),
            imaginary: Box::new(Expression::Integer(1)),
        };
        assert_eq!(format!("{}", expr), "0 + 1i");
    }

    // Tests for Expression::Variable Display

    #[test]
    fn test_expression_variable_simple() {
        let expr = Expression::Variable("x".to_string());
        assert_eq!(format!("{}", expr), "x");
    }

    #[test]
    fn test_expression_variable_greek() {
        let expr = Expression::Variable("theta".to_string());
        assert_eq!(format!("{}", expr), "theta");
    }

    #[test]
    fn test_expression_variable_subscript() {
        let expr = Expression::Variable("x_1".to_string());
        assert_eq!(format!("{}", expr), "x_1");
    }

    // Tests for Expression::Constant Display

    #[test]
    fn test_expression_constant_pi() {
        let expr = Expression::Constant(MathConstant::Pi);
        assert_eq!(format!("{}", expr), "pi");
    }

    #[test]
    fn test_expression_constant_e() {
        let expr = Expression::Constant(MathConstant::E);
        assert_eq!(format!("{}", expr), "e");
    }

    // Tests for Expression::Binary Display with precedence

    #[test]
    fn test_expression_binary_add_simple() {
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(3)),
        };
        assert_eq!(format!("{}", expr), "2 + 3");
    }

    #[test]
    fn test_expression_binary_mul_simple() {
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(3)),
        };
        assert_eq!(format!("{}", expr), "2 * 3");
    }

    #[test]
    fn test_expression_binary_precedence_add_mul() {
        // 2 + 3 * 4 (no parens needed)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::Integer(3)),
                right: Box::new(Expression::Integer(4)),
            }),
        };
        assert_eq!(format!("{}", expr), "2 + 3 * 4");
    }

    #[test]
    fn test_expression_binary_precedence_mul_add() {
        // (2 + 3) * 4 (parens needed)
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Integer(3)),
            }),
            right: Box::new(Expression::Integer(4)),
        };
        assert_eq!(format!("{}", expr), "(2 + 3) * 4");
    }

    #[test]
    fn test_expression_binary_sub_sub_left_associative() {
        // (5 - 3) - 1 (no parens needed, left-to-right)
        let expr = Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::Integer(5)),
                right: Box::new(Expression::Integer(3)),
            }),
            right: Box::new(Expression::Integer(1)),
        };
        assert_eq!(format!("{}", expr), "5 - 3 - 1");
    }

    #[test]
    fn test_expression_binary_sub_sub_right_needs_parens() {
        // 5 - (3 - 1) (parens needed on right)
        let expr = Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Integer(5)),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::Integer(3)),
                right: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(format!("{}", expr), "5 - (3 - 1)");
    }

    #[test]
    fn test_expression_binary_pow_right_associative() {
        // 2 ^ (3 ^ 4) (parens needed on right for clarity)
        let expr = Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Integer(3)),
                right: Box::new(Expression::Integer(4)),
            }),
        };
        assert_eq!(format!("{}", expr), "2 ^ (3 ^ 4)");
    }

    #[test]
    fn test_expression_binary_complex_nested() {
        // (2 + 3) * (4 - 5)
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Integer(3)),
            }),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::Integer(4)),
                right: Box::new(Expression::Integer(5)),
            }),
        };
        assert_eq!(format!("{}", expr), "(2 + 3) * (4 - 5)");
    }

    // Tests for Expression::Unary Display

    #[test]
    fn test_expression_unary_neg() {
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Integer(5)),
        };
        assert_eq!(format!("{}", expr), "-5");
    }

    #[test]
    fn test_expression_unary_pos() {
        let expr = Expression::Unary {
            op: UnaryOp::Pos,
            operand: Box::new(Expression::Integer(5)),
        };
        assert_eq!(format!("{}", expr), "+5");
    }

    #[test]
    fn test_expression_unary_factorial() {
        let expr = Expression::Unary {
            op: UnaryOp::Factorial,
            operand: Box::new(Expression::Variable("n".to_string())),
        };
        assert_eq!(format!("{}", expr), "n!");
    }

    #[test]
    fn test_expression_unary_transpose() {
        let expr = Expression::Unary {
            op: UnaryOp::Transpose,
            operand: Box::new(Expression::Variable("A".to_string())),
        };
        assert_eq!(format!("{}", expr), "A'");
    }

    #[test]
    fn test_expression_unary_nested() {
        // -(-5)
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Expression::Integer(5)),
            }),
        };
        assert_eq!(format!("{}", expr), "--5");
    }

    // Tests for Expression::Function Display

    #[test]
    fn test_expression_function_no_args() {
        let expr = Expression::Function {
            name: "f".to_string(),
            args: vec![],
        };
        assert_eq!(format!("{}", expr), "f()");
    }

    #[test]
    fn test_expression_function_one_arg() {
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(format!("{}", expr), "sin(x)");
    }

    #[test]
    fn test_expression_function_multiple_args() {
        let expr = Expression::Function {
            name: "max".to_string(),
            args: vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
            ],
        };
        assert_eq!(format!("{}", expr), "max(1, 2, 3)");
    }

    #[test]
    fn test_expression_function_nested() {
        let expr = Expression::Function {
            name: "f".to_string(),
            args: vec![Expression::Function {
                name: "g".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }],
        };
        assert_eq!(format!("{}", expr), "f(g(x))");
    }

    // Tests for Expression::Derivative Display

    #[test]
    fn test_expression_derivative_first_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        assert_eq!(format!("{}", expr), "d/dx(f)");
    }

    #[test]
    fn test_expression_derivative_second_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        };
        assert_eq!(format!("{}", expr), "d^2/dx^2(f)");
    }

    #[test]
    fn test_expression_derivative_third_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "t".to_string(),
            order: 3,
        };
        assert_eq!(format!("{}", expr), "d^3/dt^3(f)");
    }

    // Tests for Expression::PartialDerivative Display

    #[test]
    fn test_expression_partial_derivative_first_order() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        assert_eq!(format!("{}", expr), "∂/∂x(f)");
    }

    #[test]
    fn test_expression_partial_derivative_second_order() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "y".to_string(),
            order: 2,
        };
        assert_eq!(format!("{}", expr), "∂^2/∂y^2(f)");
    }

    // Tests for Expression::Integral Display

    #[test]
    fn test_expression_integral_indefinite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };
        assert_eq!(format!("{}", expr), "int(x, dx)");
    }

    #[test]
    fn test_expression_integral_definite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(format!("{}", expr), "int(x, dx, 0, 1)");
    }

    #[test]
    fn test_expression_integral_complex_bounds() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Function {
                name: "f".to_string(),
                args: vec![Expression::Variable("t".to_string())],
            }),
            var: "t".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Variable("a".to_string())),
                upper: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "int(f(t), dt, a, b)");
    }

    // Tests for Expression::Limit Display

    #[test]
    fn test_expression_limit_both() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };
        assert_eq!(format!("{}", expr), "lim(x->0)(f)");
    }

    #[test]
    fn test_expression_limit_left() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Left,
        };
        assert_eq!(format!("{}", expr), "lim(x->0-)(f)");
    }

    #[test]
    fn test_expression_limit_right() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Right,
        };
        assert_eq!(format!("{}", expr), "lim(x->0+)(f)");
    }

    #[test]
    fn test_expression_limit_to_infinity() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Function {
                name: "f".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: Direction::Both,
        };
        assert_eq!(format!("{}", expr), "lim(x->inf)(f(x))");
    }

    // Tests for Expression::Sum Display

    #[test]
    fn test_expression_sum_simple() {
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(format!("{}", expr), "sum(i=1, n, i)");
    }

    #[test]
    fn test_expression_sum_complex_body() {
        let expr = Expression::Sum {
            index: "k".to_string(),
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(10)),
            body: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Variable("k".to_string())),
                right: Box::new(Expression::Integer(2)),
            }),
        };
        assert_eq!(format!("{}", expr), "sum(k=0, 10, k ^ 2)");
    }

    // Tests for Expression::Product Display

    #[test]
    fn test_expression_product_simple() {
        let expr = Expression::Product {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(format!("{}", expr), "prod(i=1, n, i)");
    }

    #[test]
    fn test_expression_product_complex() {
        let expr = Expression::Product {
            index: "k".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Integer(5)),
            body: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("k".to_string())),
                right: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(format!("{}", expr), "prod(k=1, 5, k + 1)");
    }

    // Tests for Expression::Vector Display

    #[test]
    fn test_expression_vector_empty() {
        let expr = Expression::Vector(vec![]);
        assert_eq!(format!("{}", expr), "[]");
    }

    #[test]
    fn test_expression_vector_single() {
        let expr = Expression::Vector(vec![Expression::Integer(1)]);
        assert_eq!(format!("{}", expr), "[1]");
    }

    #[test]
    fn test_expression_vector_multiple() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ]);
        assert_eq!(format!("{}", expr), "[1, 2, 3]");
    }

    #[test]
    fn test_expression_vector_mixed() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Variable("x".to_string()),
            Expression::Float(2.5.into()),
        ]);
        assert_eq!(format!("{}", expr), "[1, x, 2.5]");
    }

    // Tests for Expression::Matrix Display

    #[test]
    fn test_expression_matrix_empty() {
        let expr = Expression::Matrix(vec![]);
        assert_eq!(format!("{}", expr), "[]");
    }

    #[test]
    fn test_expression_matrix_1x1() {
        let expr = Expression::Matrix(vec![vec![Expression::Integer(1)]]);
        assert_eq!(format!("{}", expr), "[[1]]");
    }

    #[test]
    fn test_expression_matrix_2x2() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);
        assert_eq!(format!("{}", expr), "[[1, 2], [3, 4]]");
    }

    #[test]
    fn test_expression_matrix_3x2() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
            vec![Expression::Integer(5), Expression::Integer(6)],
        ]);
        assert_eq!(format!("{}", expr), "[[1, 2], [3, 4], [5, 6]]");
    }

    // Tests for Expression::Equation Display

    #[test]
    fn test_expression_equation_simple() {
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        assert_eq!(format!("{}", expr), "x = 5");
    }

    #[test]
    fn test_expression_equation_complex() {
        // y = 2*x + 1
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expression::Integer(2)),
                    right: Box::new(Expression::Variable("x".to_string())),
                }),
                right: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(format!("{}", expr), "y = 2 * x + 1");
    }

    // Tests for Expression::Inequality Display

    #[test]
    fn test_expression_inequality_lt() {
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        assert_eq!(format!("{}", expr), "x < 5");
    }

    #[test]
    fn test_expression_inequality_le() {
        let expr = Expression::Inequality {
            op: InequalityOp::Le,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(10)),
        };
        assert_eq!(format!("{}", expr), "x <= 10");
    }

    #[test]
    fn test_expression_inequality_ge() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ge,
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Integer(0)),
        };
        assert_eq!(format!("{}", expr), "y >= 0");
    }

    #[test]
    fn test_expression_inequality_ne() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ne,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(format!("{}", expr), "a != b");
    }

    // Tests for precedence helper functions

    #[test]
    fn test_precedence_add() {
        assert_eq!(precedence(BinaryOp::Add), 1);
    }

    #[test]
    fn test_precedence_sub() {
        assert_eq!(precedence(BinaryOp::Sub), 1);
    }

    #[test]
    fn test_precedence_mul() {
        assert_eq!(precedence(BinaryOp::Mul), 2);
    }

    #[test]
    fn test_precedence_div() {
        assert_eq!(precedence(BinaryOp::Div), 2);
    }

    #[test]
    fn test_precedence_mod() {
        assert_eq!(precedence(BinaryOp::Mod), 2);
    }

    #[test]
    fn test_precedence_pow() {
        assert_eq!(precedence(BinaryOp::Pow), 3);
    }

    #[test]
    fn test_needs_parens_lower_precedence() {
        let child = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };
        assert!(needs_parens(&child, BinaryOp::Mul, false));
        assert!(needs_parens(&child, BinaryOp::Mul, true));
    }

    #[test]
    fn test_needs_parens_equal_precedence_left() {
        let child = Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };
        assert!(!needs_parens(&child, BinaryOp::Sub, false));
    }

    #[test]
    fn test_needs_parens_equal_precedence_right_sub() {
        let child = Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };
        assert!(needs_parens(&child, BinaryOp::Sub, true));
    }

    #[test]
    fn test_needs_parens_higher_precedence() {
        let child = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };
        assert!(!needs_parens(&child, BinaryOp::Add, false));
        assert!(!needs_parens(&child, BinaryOp::Add, true));
    }

    #[test]
    fn test_needs_parens_non_binary() {
        let child = Expression::Integer(5);
        assert!(!needs_parens(&child, BinaryOp::Add, false));
        assert!(!needs_parens(&child, BinaryOp::Mul, true));
    }

    // Tests for precedence-safe parentheses with unary operators

    #[test]
    fn test_unary_neg_with_binary_operand() {
        // -(a + b) should print as "-(a + b)"
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "-(a + b)");
    }

    #[test]
    fn test_unary_pos_with_binary_operand() {
        // +(a * b) should print as "+(a * b)"
        let expr = Expression::Unary {
            op: UnaryOp::Pos,
            operand: Box::new(Expression::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "+(a * b)");
    }

    #[test]
    fn test_factorial_with_binary_operand() {
        // (a + b)! should print as "(a + b)!"
        let expr = Expression::Unary {
            op: UnaryOp::Factorial,
            operand: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "(a + b)!");
    }

    #[test]
    fn test_transpose_with_binary_operand() {
        // (A + B)' should print as "(A + B)'"
        let expr = Expression::Unary {
            op: UnaryOp::Transpose,
            operand: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("A".to_string())),
                right: Box::new(Expression::Variable("B".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "(A + B)'");
    }

    #[test]
    fn test_power_left_associative() {
        // (a^b)^c should print as "(a ^ b) ^ c"
        let expr = Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
            right: Box::new(Expression::Variable("c".to_string())),
        };
        assert_eq!(format!("{}", expr), "(a ^ b) ^ c");
    }

    #[test]
    fn test_power_right_associative() {
        // a^(b^c) should print as "a ^ (b ^ c)"
        let expr = Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Variable("b".to_string())),
                right: Box::new(Expression::Variable("c".to_string())),
            }),
        };
        assert_eq!(format!("{}", expr), "a ^ (b ^ c)");
    }

    #[test]
    fn test_complex_precedence_example() {
        // -(a + b) * c should print as "-(a + b) * c"
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Expression::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::Variable("a".to_string())),
                    right: Box::new(Expression::Variable("b".to_string())),
                }),
            }),
            right: Box::new(Expression::Variable("c".to_string())),
        };
        assert_eq!(format!("{}", expr), "-(a + b) * c");
    }

    #[test]
    fn test_unary_with_non_binary_operand() {
        // -x should print as "-x" (no parens needed)
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(format!("{}", expr), "-x");
    }

    #[test]
    fn test_factorial_with_non_binary_operand() {
        // n! should print as "n!" (no parens needed)
        let expr = Expression::Unary {
            op: UnaryOp::Factorial,
            operand: Box::new(Expression::Variable("n".to_string())),
        };
        assert_eq!(format!("{}", expr), "n!");
    }
}
