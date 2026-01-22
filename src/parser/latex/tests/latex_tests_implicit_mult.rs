//! Tests for LaTeX implicit multiplication parsing.

use crate::ast::{BinaryOp, Expression, MathConstant};
use crate::parser::parse_latex;

#[test]
fn test_implicit_mult_number_letter() {
    // 2x should parse as 2*x
    let expr = parse_latex("2x").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_letter_letter() {
    // xy should parse as x*y
    let expr = parse_latex("xy").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Variable("y".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_number_paren() {
    // 2(x+1) should parse as 2*(x+1)
    let expr = parse_latex("2(x+1)").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in right operand"),
            }
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_number_pi() {
    // 2\pi should parse as 2*π
    let expr = parse_latex(r"2\pi").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Constant(MathConstant::Pi));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_function_not_applied() {
    // \sin(x) should remain a function call, NOT implicit multiplication
    let expr = parse_latex(r"\sin(x)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call, not implicit multiplication"),
    }
}

#[test]
fn test_implicit_mult_var_paren() {
    // x(y) should parse as x*(y), not a function call
    let expr = parse_latex("x(y)").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Variable("y".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_paren_paren() {
    // (a)(b) should parse as (a)*(b)
    let expr = parse_latex("(a)(b)").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Variable("a".to_string()));
            assert_eq!(*right, Expression::Variable("b".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_complex_expr() {
    // 2xy should parse as (2*x)*y
    let expr = parse_latex("2xy").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            // Left should be 2*x
            match *left {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left: ref ll,
                    right: ref lr,
                } => {
                    assert_eq!(**ll, Expression::Integer(2));
                    assert_eq!(**lr, Expression::Variable("x".to_string()));
                }
                _ => panic!("Expected 2*x on left"),
            }
            // Right should be y
            assert_eq!(*right, Expression::Variable("y".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_implicit_mult_with_addition() {
    // 2x + 3y should parse as (2*x) + (3*y)
    let expr = parse_latex("2x + 3y").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            // Left: 2*x
            match *left {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication on left"),
            }
            // Right: 3*y
            match *right {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication on right"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_implicit_mult_with_power() {
    // 2x^2 should parse as 2*(x^2), not (2x)^2
    let expr = parse_latex("2x^2").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            // Right should be x^2
            match *right {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    left: ref pow_left,
                    right: ref pow_right,
                } => {
                    assert_eq!(**pow_left, Expression::Variable("x".to_string()));
                    assert_eq!(**pow_right, Expression::Integer(2));
                }
                _ => panic!("Expected power on right"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_implicit_mult_brace() {
    // 2{x+1} should parse as 2*(x+1)
    let expr = parse_latex("2{x+1}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in braces"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_no_implicit_mult_with_explicit() {
    // 2*x should still work with explicit multiplication
    let expr1 = parse_latex("2*x").unwrap();
    let expr2 = parse_latex("2x").unwrap();

    // Both should produce the same AST
    assert_eq!(expr1, expr2);
}

#[test]
fn test_implicit_mult_three_letters() {
    // abc should parse as (a*b)*c
    let expr = parse_latex("abc").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            // Left should be a*b
            match *left {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left: ref ll,
                    right: ref lr,
                } => {
                    assert_eq!(**ll, Expression::Variable("a".to_string()));
                    assert_eq!(**lr, Expression::Variable("b".to_string()));
                }
                _ => panic!("Expected a*b on left"),
            }
            // Right should be c
            assert_eq!(*right, Expression::Variable("c".to_string()));
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_implicit_mult_complex() {
    // 2x(y+1) should parse as (2*x)*(y+1)
    let expr = parse_latex("2x(y+1)").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            // Left should be 2*x
            match *left {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected 2*x on left"),
            }
            // Right should be (y+1)
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition on right"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}
