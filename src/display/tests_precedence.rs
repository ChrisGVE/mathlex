/// Tests for precedence helpers and unary-with-binary-operand patterns.
use crate::ast::{BinaryOp, Expression, UnaryOp};

use crate::display::helpers::{needs_parens, precedence};

// Tests for precedence helper

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

// Tests for needs_parens helper

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

// Tests for unary operators with binary operands

#[test]
fn test_unary_neg_with_binary_operand() {
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
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Variable("x".to_string())),
    };
    assert_eq!(format!("{}", expr), "-x");
}

#[test]
fn test_factorial_with_non_binary_operand() {
    let expr = Expression::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::Variable("n".to_string())),
    };
    assert_eq!(format!("{}", expr), "n!");
}
