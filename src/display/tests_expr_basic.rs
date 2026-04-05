/// Tests for Expression Display: literals, arithmetic (Binary, Unary, Function).
#[allow(clippy::approx_constant)]
use crate::ast::{BinaryOp, Expression, MathConstant, UnaryOp};

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
    assert!(!output.is_empty());
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
