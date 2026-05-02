/// Tests for Expression Display: literals, arithmetic (Binary, Unary, Function).
#[allow(clippy::approx_constant)]
use crate::ast::{BinaryOp, ExprKind, Expression, MathConstant, UnaryOp};

// Tests for ExprKind::Integer Display

#[test]
fn test_expression_integer_positive() {
    let expr = Expression::integer(42);
    assert_eq!(format!("{}", expr), "42");
}

#[test]
fn test_expression_integer_negative() {
    let expr = Expression::integer(-17);
    assert_eq!(format!("{}", expr), "-17");
}

#[test]
fn test_expression_integer_zero() {
    let expr = Expression::integer(0);
    assert_eq!(format!("{}", expr), "0");
}

// Tests for ExprKind::Float Display

#[test]
fn test_expression_float_positive() {
    let expr = Expression::float(3.14.into());
    assert_eq!(format!("{}", expr), "3.14");
}

#[test]
fn test_expression_float_negative() {
    let expr = Expression::float((-2.5).into());
    assert_eq!(format!("{}", expr), "-2.5");
}

#[test]
fn test_expression_float_scientific() {
    let expr = Expression::float(1e-10.into());
    let output = format!("{}", expr);
    assert!(!output.is_empty());
    assert!(output.parse::<f64>().unwrap() < 1e-9);
}

// Tests for ExprKind::Rational Display

#[test]
fn test_expression_rational_simple() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::integer(1)),
        denominator: Box::new(Expression::integer(2)),
    }
    .into();
    assert_eq!(format!("{}", expr), "1/2");
}

#[test]
fn test_expression_rational_negative() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::integer(-3)),
        denominator: Box::new(Expression::integer(4)),
    }
    .into();
    assert_eq!(format!("{}", expr), "-3/4");
}

#[test]
fn test_expression_rational_variables() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::variable("a".to_string())),
        denominator: Box::new(Expression::variable("b".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "a/b");
}

// Tests for ExprKind::Complex Display

#[test]
fn test_expression_complex_simple() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(3)),
        imaginary: Box::new(Expression::integer(4)),
    }
    .into();
    assert_eq!(format!("{}", expr), "3 + 4i");
}

#[test]
fn test_expression_complex_negative_imaginary() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(2)),
        imaginary: Box::new(Expression::integer(-5)),
    }
    .into();
    assert_eq!(format!("{}", expr), "2 + -5i");
}

#[test]
fn test_expression_complex_pure_imaginary() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(0)),
        imaginary: Box::new(Expression::integer(1)),
    }
    .into();
    assert_eq!(format!("{}", expr), "0 + 1i");
}

// Tests for ExprKind::Variable Display

#[test]
fn test_expression_variable_simple() {
    let expr = Expression::variable("x".to_string());
    assert_eq!(format!("{}", expr), "x");
}

#[test]
fn test_expression_variable_greek() {
    let expr = Expression::variable("theta".to_string());
    assert_eq!(format!("{}", expr), "theta");
}

#[test]
fn test_expression_variable_subscript() {
    let expr = Expression::variable("x_1".to_string());
    assert_eq!(format!("{}", expr), "x_1");
}

// Tests for ExprKind::Constant Display

#[test]
fn test_expression_constant_pi() {
    let expr = Expression::constant(MathConstant::Pi);
    assert_eq!(format!("{}", expr), "pi");
}

#[test]
fn test_expression_constant_e() {
    let expr = Expression::constant(MathConstant::E);
    assert_eq!(format!("{}", expr), "e");
}

// Tests for ExprKind::Binary Display with precedence

#[test]
fn test_expression_binary_add_simple() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::integer(3)),
    }
    .into();
    assert_eq!(format!("{}", expr), "2 + 3");
}

#[test]
fn test_expression_binary_mul_simple() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::integer(3)),
    }
    .into();
    assert_eq!(format!("{}", expr), "2 * 3");
}

#[test]
fn test_expression_binary_precedence_add_mul() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(2)),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::integer(3)),
                right: Box::new(Expression::integer(4)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "2 + 3 * 4");
}

#[test]
fn test_expression_binary_precedence_mul_add() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(2)),
                right: Box::new(Expression::integer(3)),
            }
            .into(),
        ),
        right: Box::new(Expression::integer(4)),
    }
    .into();
    assert_eq!(format!("{}", expr), "(2 + 3) * 4");
}

#[test]
fn test_expression_binary_sub_sub_left_associative() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Sub,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::integer(5)),
                right: Box::new(Expression::integer(3)),
            }
            .into(),
        ),
        right: Box::new(Expression::integer(1)),
    }
    .into();
    assert_eq!(format!("{}", expr), "5 - 3 - 1");
}

#[test]
fn test_expression_binary_sub_sub_right_needs_parens() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::integer(5)),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::integer(3)),
                right: Box::new(Expression::integer(1)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "5 - (3 - 1)");
}

#[test]
fn test_expression_binary_pow_right_associative() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::integer(2)),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::integer(3)),
                right: Box::new(Expression::integer(4)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "2 ^ (3 ^ 4)");
}

#[test]
fn test_expression_binary_complex_nested() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(2)),
                right: Box::new(Expression::integer(3)),
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::integer(4)),
                right: Box::new(Expression::integer(5)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "(2 + 3) * (4 - 5)");
}

// Tests for ExprKind::Unary Display

#[test]
fn test_expression_unary_neg() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::integer(5)),
    }
    .into();
    assert_eq!(format!("{}", expr), "-5");
}

#[test]
fn test_expression_unary_pos() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Pos,
        operand: Box::new(Expression::integer(5)),
    }
    .into();
    assert_eq!(format!("{}", expr), "+5");
}

#[test]
fn test_expression_unary_factorial() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::variable("n".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "n!");
}

#[test]
fn test_expression_unary_transpose() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Transpose,
        operand: Box::new(Expression::variable("A".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "A'");
}

#[test]
fn test_expression_unary_nested() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Expression::integer(5)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "--5");
}

// Tests for ExprKind::Function Display

#[test]
fn test_expression_function_no_args() {
    let expr: Expression = ExprKind::Function {
        name: "f".to_string(),
        args: vec![],
    }
    .into();
    assert_eq!(format!("{}", expr), "f()");
}

#[test]
fn test_expression_function_one_arg() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    }
    .into();
    assert_eq!(format!("{}", expr), "sin(x)");
}

#[test]
fn test_expression_function_multiple_args() {
    let expr: Expression = ExprKind::Function {
        name: "max".to_string(),
        args: vec![
            Expression::integer(1),
            Expression::integer(2),
            Expression::integer(3),
        ],
    }
    .into();
    assert_eq!(format!("{}", expr), "max(1, 2, 3)");
}

#[test]
fn test_expression_function_nested() {
    let expr: Expression = ExprKind::Function {
        name: "f".to_string(),
        args: vec![ExprKind::Function {
            name: "g".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }
        .into()],
    }
    .into();
    assert_eq!(format!("{}", expr), "f(g(x))");
}
