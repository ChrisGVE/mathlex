/// Tests for Expression Display: calculus, linear algebra, logic, equations.
use crate::ast::{BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, MathConstant};

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
