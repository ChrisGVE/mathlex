/// Tests for Expression Display: calculus, linear algebra, logic, equations.
use crate::ast::{
    BinaryOp, Direction, ExprKind, Expression, InequalityOp, IntegralBounds, MathConstant,
};

// Tests for ExprKind::Derivative Display

#[test]
fn test_expression_derivative_first_order() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    assert_eq!(format!("{}", expr), "d/dx(f)");
}

#[test]
fn test_expression_derivative_second_order() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    }
    .into();
    assert_eq!(format!("{}", expr), "d^2/dx^2(f)");
}

#[test]
fn test_expression_derivative_third_order() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "t".to_string(),
        order: 3,
    }
    .into();
    assert_eq!(format!("{}", expr), "d^3/dt^3(f)");
}

// Tests for ExprKind::PartialDerivative Display

#[test]
fn test_expression_partial_derivative_first_order() {
    let expr: Expression = ExprKind::PartialDerivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    assert_eq!(format!("{}", expr), "∂/∂x(f)");
}

#[test]
fn test_expression_partial_derivative_second_order() {
    let expr: Expression = ExprKind::PartialDerivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "y".to_string(),
        order: 2,
    }
    .into();
    assert_eq!(format!("{}", expr), "∂^2/∂y^2(f)");
}

// Tests for ExprKind::Integral Display

#[test]
fn test_expression_integral_indefinite() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    }
    .into();
    assert_eq!(format!("{}", expr), "int(x, dx)");
}

#[test]
fn test_expression_integral_definite() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    assert_eq!(format!("{}", expr), "int(x, dx, 0, 1)");
}

#[test]
fn test_expression_integral_complex_bounds() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("t".to_string())],
            }
            .into(),
        ),
        var: "t".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::variable("a".to_string())),
            upper: Box::new(Expression::variable("b".to_string())),
        }),
    }
    .into();
    assert_eq!(format!("{}", expr), "int(f(t), dt, a, b)");
}

// Tests for ExprKind::Limit Display

#[test]
fn test_expression_limit_both() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Both,
    }
    .into();
    assert_eq!(format!("{}", expr), "lim(x->0)(f)");
}

#[test]
fn test_expression_limit_left() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Left,
    }
    .into();
    assert_eq!(format!("{}", expr), "lim(x->0-)(f)");
}

#[test]
fn test_expression_limit_right() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Right,
    }
    .into();
    assert_eq!(format!("{}", expr), "lim(x->0+)(f)");
}

#[test]
fn test_expression_limit_to_infinity() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x".to_string())],
            }
            .into(),
        ),
        var: "x".to_string(),
        to: Box::new(Expression::constant(MathConstant::Infinity)),
        direction: Direction::Both,
    }
    .into();
    assert_eq!(format!("{}", expr), "lim(x->inf)(f(x))");
}

// Tests for ExprKind::Sum Display

#[test]
fn test_expression_sum_simple() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "sum(i=1, n, i)");
}

#[test]
fn test_expression_sum_complex_body() {
    let expr: Expression = ExprKind::Sum {
        index: "k".to_string(),
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(10)),
        body: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::variable("k".to_string())),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "sum(k=0, 10, k ^ 2)");
}

// Tests for ExprKind::Product Display

#[test]
fn test_expression_product_simple() {
    let expr: Expression = ExprKind::Product {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "prod(i=1, n, i)");
}

#[test]
fn test_expression_product_complex() {
    let expr: Expression = ExprKind::Product {
        index: "k".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::integer(5)),
        body: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("k".to_string())),
                right: Box::new(Expression::integer(1)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "prod(k=1, 5, k + 1)");
}

// Tests for ExprKind::Vector Display

#[test]
fn test_expression_vector_empty() {
    let expr: Expression = ExprKind::Vector(vec![]).into();
    assert_eq!(format!("{}", expr), "[]");
}

#[test]
fn test_expression_vector_single() {
    let expr: Expression = ExprKind::Vector(vec![Expression::integer(1)]).into();
    assert_eq!(format!("{}", expr), "[1]");
}

#[test]
fn test_expression_vector_multiple() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::integer(2),
        Expression::integer(3),
    ])
    .into();
    assert_eq!(format!("{}", expr), "[1, 2, 3]");
}

#[test]
fn test_expression_vector_mixed() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::variable("x".to_string()),
        Expression::float(2.5.into()),
    ])
    .into();
    assert_eq!(format!("{}", expr), "[1, x, 2.5]");
}

// Tests for ExprKind::Matrix Display

#[test]
fn test_expression_matrix_empty() {
    let expr: Expression = ExprKind::Matrix(vec![]).into();
    assert_eq!(format!("{}", expr), "[]");
}

#[test]
fn test_expression_matrix_1x1() {
    let expr: Expression = ExprKind::Matrix(vec![vec![Expression::integer(1)]]).into();
    assert_eq!(format!("{}", expr), "[[1]]");
}

#[test]
fn test_expression_matrix_2x2() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ])
    .into();
    assert_eq!(format!("{}", expr), "[[1, 2], [3, 4]]");
}

#[test]
fn test_expression_matrix_3x2() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
        vec![Expression::integer(5), Expression::integer(6)],
    ])
    .into();
    assert_eq!(format!("{}", expr), "[[1, 2], [3, 4], [5, 6]]");
}

// Tests for ExprKind::Equation Display

#[test]
fn test_expression_equation_simple() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(5)),
    }
    .into();
    assert_eq!(format!("{}", expr), "x = 5");
}

#[test]
fn test_expression_equation_complex() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("y".to_string())),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(Expression::integer(2)),
                        right: Box::new(Expression::variable("x".to_string())),
                    }
                    .into(),
                ),
                right: Box::new(Expression::integer(1)),
            }
            .into(),
        ),
    }
    .into();
    assert_eq!(format!("{}", expr), "y = 2 * x + 1");
}

// Tests for ExprKind::Inequality Display

#[test]
fn test_expression_inequality_lt() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(5)),
    }
    .into();
    assert_eq!(format!("{}", expr), "x < 5");
}

#[test]
fn test_expression_inequality_le() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Le,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(10)),
    }
    .into();
    assert_eq!(format!("{}", expr), "x <= 10");
}

#[test]
fn test_expression_inequality_ge() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Ge,
        left: Box::new(Expression::variable("y".to_string())),
        right: Box::new(Expression::integer(0)),
    }
    .into();
    assert_eq!(format!("{}", expr), "y >= 0");
}

#[test]
fn test_expression_inequality_ne() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Ne,
        left: Box::new(Expression::variable("a".to_string())),
        right: Box::new(Expression::variable("b".to_string())),
    }
    .into();
    assert_eq!(format!("{}", expr), "a != b");
}
