// Power and subscript tests for LaTeX parser
use super::*;

// Power tests

#[test]
fn test_power_simple() {
    let expr = parse_latex("x^2").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary power"),
    }
}

#[test]
fn test_power_braced() {
    let expr = parse_latex("x^{10}").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(10));
        }
        _ => panic!("Expected binary power"),
    }
}

#[test]
fn test_power_expression_simple() {
    let expr = parse_latex("x^{n+1}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_expression_complex() {
    let expr = parse_latex("x^{2*n+1}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            // Right should be addition with multiplication on left
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left: add_left,
                    ..
                } => {
                    assert!(matches!(
                        *add_left,
                        Expression::Binary {
                            op: BinaryOp::Mul,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected addition in exponent"),
            }
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_nested() {
    let expr = parse_latex("x^{y^z}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            // Right should be another power
            match *right {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    left: inner_left,
                    right: inner_right,
                } => {
                    assert_eq!(*inner_left, Expression::Variable("y".to_string()));
                    assert_eq!(*inner_right, Expression::Variable("z".to_string()));
                }
                _ => panic!("Expected nested power"),
            }
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_triple_nested() {
    let expr = parse_latex("a^{b^{c^d}}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("a".to_string()));
            // Verify triple nesting
            match *right {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    right: r1,
                    ..
                } => {
                    assert!(matches!(
                        *r1,
                        Expression::Binary {
                            op: BinaryOp::Pow,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected nested powers"),
            }
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_of_number() {
    let expr = parse_latex("2^3").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected binary power"),
    }
}

#[test]
fn test_power_negative_exponent() {
    let expr = parse_latex("x^{-1}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            // -1 is parsed as unary negation of 1
            match *right {
                Expression::Unary {
                    op: crate::ast::UnaryOp::Neg,
                    operand,
                } => {
                    assert_eq!(*operand, Expression::Integer(1));
                }
                _ => panic!("Expected unary negation"),
            }
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_fraction_exponent() {
    let expr = parse_latex(r"x^{\frac{1}{2}}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_greek_letter() {
    let expr = parse_latex(r"x^\alpha").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Variable("alpha".to_string()));
        }
        _ => panic!("Expected binary power"),
    }
}

// Subscript tests

#[test]
fn test_subscript_simple() {
    let expr = parse_latex("x_1").unwrap();
    assert_eq!(expr, Expression::Variable("x_1".to_string()));
}

#[test]
fn test_subscript_braced() {
    let expr = parse_latex("x_{12}").unwrap();
    assert_eq!(expr, Expression::Variable("x_12".to_string()));
}

#[test]
fn test_subscript_variable() {
    let expr = parse_latex("x_i").unwrap();
    assert_eq!(expr, Expression::Variable("x_i".to_string()));
}

#[test]
fn test_subscript_letter() {
    let expr = parse_latex("x_n").unwrap();
    assert_eq!(expr, Expression::Variable("x_n".to_string()));
}

#[test]
fn test_subscript_braced_variable() {
    // Note: {max} would try to parse "max" as an expression, which fails
    // because subscripts can only be simple integers or single variables
    let result = parse_latex("x_{max}");
    assert!(result.is_err());
}

#[test]
fn test_subscript_zero() {
    let expr = parse_latex("x_0").unwrap();
    assert_eq!(expr, Expression::Variable("x_0".to_string()));
}

#[test]
fn test_subscript_three_digits() {
    let expr = parse_latex("x_{123}").unwrap();
    assert_eq!(expr, Expression::Variable("x_123".to_string()));
}

// Combined subscript and superscript tests

#[test]
fn test_subscript_then_superscript() {
    // The parser consumes subscript first (x_i), then tries to parse ^n
    // But the expression is already complete, so it errors on ^
    let result = parse_latex("x_i^n");
    assert!(result.is_err());
}

#[test]
fn test_superscript_then_subscript() {
    // Parser processes ^ first, creating x^n, then _ tries to apply to the result
    // Since the result is a Binary (power), not a Variable, subscript fails
    let result = parse_latex("x^n_i");
    assert!(result.is_err());
}

#[test]
fn test_subscript_and_superscript_braced() {
    // Same issue: subscript is consumed first, making x_i, then ^ causes error
    let result = parse_latex("x_{i}^{n}");
    assert!(result.is_err());
}

#[test]
fn test_subscript_power_combined_complex() {
    // Subscript with expression should fail per parser rules
    // Subscripts can only be simple integers or variables
    let result = parse_latex("x_{i+1}^{2}");
    assert!(result.is_err());
}

// Powers in expressions

#[test]
fn test_power_in_addition() {
    let expr = parse_latex("x^2 + y^2").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Pow,
                    ..
                }
            ));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Pow,
                    ..
                }
            ));
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_power_in_multiplication() {
    let expr = parse_latex("2 * x^2").unwrap();
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
fn test_power_of_sum() {
    let expr = parse_latex("(x+y)^2").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
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
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_power_of_product() {
    let expr = parse_latex("(x*y)^n").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
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
            assert_eq!(*right, Expression::Variable("n".to_string()));
        }
        _ => panic!("Expected power expression"),
    }
}

// Edge cases

#[test]
fn test_multiple_subscripts_in_expression() {
    let expr = parse_latex("x_1 + x_2 + x_3").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            // Left should be x_1 + x_2, right should be x_3
            match *left {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left: ll,
                    right: lr,
                } => {
                    assert_eq!(*ll, Expression::Variable("x_1".to_string()));
                    assert_eq!(*lr, Expression::Variable("x_2".to_string()));
                }
                _ => panic!("Expected addition"),
            }
            assert_eq!(*right, Expression::Variable("x_3".to_string()));
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_power_zero() {
    let expr = parse_latex("x^0").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(0));
        }
        _ => panic!("Expected binary power"),
    }
}

#[test]
fn test_power_one() {
    let expr = parse_latex("x^1").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected binary power"),
    }
}
