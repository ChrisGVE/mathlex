/// Tests for precedence helpers and unary-with-binary-operand patterns.
use crate::ast::{BinaryOp, ExprKind, Expression, UnaryOp};

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
    let child: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    assert!(needs_parens(&child, BinaryOp::Mul, false));
    assert!(needs_parens(&child, BinaryOp::Mul, true));
}

#[test]
fn test_needs_parens_equal_precedence_left() {
    let child: Expression = ExprKind::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    assert!(!needs_parens(&child, BinaryOp::Sub, false));
}

#[test]
fn test_needs_parens_equal_precedence_right_sub() {
    let child: Expression = ExprKind::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    assert!(needs_parens(&child, BinaryOp::Sub, true));
}

#[test]
fn test_needs_parens_higher_precedence() {
    let child: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    assert!(!needs_parens(&child, BinaryOp::Add, false));
    assert!(!needs_parens(&child, BinaryOp::Add, true));
}

#[test]
fn test_needs_parens_non_binary() {
    let child = Expression::integer(5);
    assert!(!needs_parens(&child, BinaryOp::Add, false));
    assert!(!needs_parens(&child, BinaryOp::Mul, true));
}

// Tests for unary operators with binary operands

#[test]
fn test_unary_neg_with_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("a".to_string())),
                right: Box::new(Expression::variable("b".to_string())),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "-(a + b)");
}

#[test]
fn test_unary_pos_with_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Pos,
        operand: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::variable("a".to_string())),
                right: Box::new(Expression::variable("b".to_string())),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "+(a * b)");
}

#[test]
fn test_factorial_with_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("a".to_string())),
                right: Box::new(Expression::variable("b".to_string())),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "(a + b)!");
}

#[test]
fn test_transpose_with_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Transpose,
        operand: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("A".to_string())),
                right: Box::new(Expression::variable("B".to_string())),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "(A + B)'");
}

#[test]
fn test_power_left_associative() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::variable("a".to_string())),
                right: Box::new(Expression::variable("b".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("c".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "(a ^ b) ^ c");
}

#[test]
fn test_power_right_associative() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::variable("a".to_string())),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::variable("b".to_string())),
                right: Box::new(Expression::variable("c".to_string())),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "a ^ (b ^ c)");
}

#[test]
fn test_complex_precedence_example() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(Expression::variable("a".to_string())),
                        right: Box::new(Expression::variable("b".to_string())),
                    }
                    .into(),
                ),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("c".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "-(a + b) * c");
}

#[test]
fn test_unary_with_non_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "-x");
}

#[test]
fn test_factorial_with_non_binary_operand() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::variable("n".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "n!");
}
