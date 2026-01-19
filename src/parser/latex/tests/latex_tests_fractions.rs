// Fraction tests for LaTeX parser
use super::*;

#[test]
fn test_frac_simple() {
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
fn test_frac_variables() {
    let expr = parse_latex(r"\frac{x}{y}").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Div);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Variable("y".to_string()));
        }
        _ => panic!("Expected binary division"),
    }
}

#[test]
fn test_frac_nested_numerator() {
    let expr = parse_latex(r"\frac{\frac{1}{2}}{3}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Numerator should be a division
            match *left {
                Expression::Binary {
                    op: BinaryOp::Div,
                    left: inner_left,
                    right: inner_right,
                } => {
                    assert_eq!(*inner_left, Expression::Integer(1));
                    assert_eq!(*inner_right, Expression::Integer(2));
                }
                _ => panic!("Expected nested division in numerator"),
            }
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_nested_denominator() {
    let expr = parse_latex(r"\frac{1}{\frac{2}{3}}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(1));
            // Denominator should be a division
            match *right {
                Expression::Binary {
                    op: BinaryOp::Div,
                    left: inner_left,
                    right: inner_right,
                } => {
                    assert_eq!(*inner_left, Expression::Integer(2));
                    assert_eq!(*inner_right, Expression::Integer(3));
                }
                _ => panic!("Expected nested division in denominator"),
            }
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_nested_both() {
    let expr = parse_latex(r"\frac{\frac{1}{2}}{\frac{3}{4}}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Both numerator and denominator should be divisions
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_nested_three_levels() {
    let expr = parse_latex(r"\frac{\frac{\frac{1}{2}}{3}}{4}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Numerator should be a division containing a nested division
            match *left {
                Expression::Binary {
                    op: BinaryOp::Div,
                    left: inner_left,
                    ..
                } => {
                    assert!(matches!(
                        *inner_left,
                        Expression::Binary {
                            op: BinaryOp::Div,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected nested divisions"),
            }
            assert_eq!(*right, Expression::Integer(4));
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_complex_numerator() {
    let expr = parse_latex(r"\frac{x+1}{2}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Numerator should be an addition
            match *left {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in numerator"),
            }
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_complex_denominator() {
    let expr = parse_latex(r"\frac{1}{x+y}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(1));
            // Denominator should be an addition
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in denominator"),
            }
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_both_complex() {
    let expr = parse_latex(r"\frac{x^2+1}{y-2}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Numerator should be an addition with power
            match *left {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in numerator"),
            }
            // Denominator should be a subtraction
            match *right {
                Expression::Binary {
                    op: BinaryOp::Sub, ..
                } => {}
                _ => panic!("Expected subtraction in denominator"),
            }
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_with_multiplication() {
    let expr = parse_latex(r"\frac{2*x}{3*y}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
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
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_with_powers() {
    let expr = parse_latex(r"\frac{x^2}{y^3}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
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
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_with_floats() {
    let expr = parse_latex(r"\frac{1.5}{2.7}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => match (*left, *right) {
            (Expression::Float(f1), Expression::Float(f2)) => {
                assert!((f1.value() - 1.5).abs() < 1e-10);
                assert!((f2.value() - 2.7).abs() < 1e-10);
            }
            _ => panic!("Expected floats"),
        },
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_frac_in_expression() {
    // Test: 1 + \frac{2}{3}
    let expr = parse_latex(r"1 + \frac{2}{3}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(1));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_multiple_fracs() {
    // Test: \frac{1}{2} + \frac{3}{4}
    let expr = parse_latex(r"\frac{1}{2} + \frac{3}{4}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
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
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_frac_with_parentheses() {
    let expr = parse_latex(r"\frac{(x+1)}{(y-2)}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            // Parentheses are transparent in the AST
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
                    op: BinaryOp::Sub,
                    ..
                }
            ));
        }
        _ => panic!("Expected division"),
    }
}
