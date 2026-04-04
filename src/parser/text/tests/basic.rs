//! Basic parsing tests: atoms, operators, precedence, errors.

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
fn test_parse_constant_neg_inf() {
    let expr = parse("-inf").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::NegInfinity));
}

#[test]
fn test_parse_neg_unicode_infinity() {
    let expr = parse("-\u{221E}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::NegInfinity));
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
            assert!(matches!(args[0], Expression::Function { .. }));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_complex_expression() {
    let expr = parse("(2 + 3) * sin(x) - 4^2").unwrap();
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
    assert!(parse("").is_err());
}

#[test]
fn test_trailing_operator() {
    assert!(parse("2 +").is_err());
}

#[test]
fn test_missing_operand_unary() {
    assert!(parse("+ 2").is_ok());
}

#[test]
fn test_unmatched_parenthesis() {
    assert!(parse("(2 + 3").is_err());
}

#[test]
fn test_extra_closing_parenthesis() {
    assert!(parse("2 + 3)").is_err());
}
