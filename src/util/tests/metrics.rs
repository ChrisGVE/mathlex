use crate::ast::{
    BinaryOp, ExprKind, Expression, InequalityOp, IntegralBounds, MathConstant, UnaryOp,
};

// Tests for depth

#[test]
fn test_depth_leaf_nodes() {
    assert_eq!(Expression::integer(42).depth(), 1);
    assert_eq!(
        Expression::float(crate::ast::MathFloat::from(3.14)).depth(),
        1
    );
    assert_eq!(Expression::variable("x".to_string()).depth(), 1);
    assert_eq!(Expression::constant(MathConstant::Pi).depth(), 1);
}

#[test]
fn test_depth_unary() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_nested() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("x".to_string())),
                right: Box::new(Expression::variable("y".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("z".to_string())),
    }
    .into();
    assert_eq!(expr.depth(), 3);
}

#[test]
fn test_depth_asymmetric() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        left: Box::new(Expression::variable("x".to_string())),
                        right: Box::new(Expression::variable("y".to_string())),
                    }
                    .into(),
                ),
                right: Box::new(Expression::variable("z".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("w".to_string())),
    }
    .into();
    assert_eq!(expr.depth(), 4);
}

#[test]
fn test_depth_function() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    }
    .into();
    assert_eq!(expr.depth(), 2);

    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![ExprKind::Function {
            name: "cos".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }
        .into()],
    }
    .into();
    assert_eq!(expr.depth(), 3);
}

#[test]
fn test_depth_vector() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::variable("x".to_string()),
        Expression::variable("y".to_string()),
    ])
    .into();
    assert_eq!(expr.depth(), 2);

    let expr: Expression = ExprKind::Vector(vec![]).into();
    assert_eq!(expr.depth(), 1);
}

#[test]
fn test_depth_matrix() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![
            Expression::variable("x".to_string()),
            Expression::variable("y".to_string()),
        ],
        vec![
            Expression::variable("z".to_string()),
            Expression::variable("w".to_string()),
        ],
    ])
    .into();
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_integral() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    assert_eq!(expr.depth(), 2);
}

// Tests for node_count

#[test]
fn test_node_count_leaf_nodes() {
    assert_eq!(Expression::integer(42).node_count(), 1);
    assert_eq!(
        Expression::float(crate::ast::MathFloat::from(3.14)).node_count(),
        1
    );
    assert_eq!(Expression::variable("x".to_string()).node_count(), 1);
    assert_eq!(Expression::constant(MathConstant::Pi).node_count(), 1);
}

#[test]
fn test_node_count_unary() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 2);
}

#[test]
fn test_node_count_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_nested() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("x".to_string())),
                right: Box::new(Expression::variable("y".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("z".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 5);
}

#[test]
fn test_node_count_function() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![],
    }
    .into();
    assert_eq!(expr.node_count(), 1);

    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    }
    .into();
    assert_eq!(expr.node_count(), 2);

    let expr: Expression = ExprKind::Function {
        name: "max".to_string(),
        args: vec![
            Expression::variable("x".to_string()),
            Expression::variable("y".to_string()),
            Expression::variable("z".to_string()),
        ],
    }
    .into();
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_vector() {
    let expr: Expression = ExprKind::Vector(vec![]).into();
    assert_eq!(expr.node_count(), 1);

    let expr: Expression = ExprKind::Vector(vec![
        Expression::variable("x".to_string()),
        Expression::variable("y".to_string()),
        Expression::variable("z".to_string()),
    ])
    .into();
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_matrix() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![
            Expression::variable("x".to_string()),
            Expression::variable("y".to_string()),
        ],
        vec![
            Expression::variable("z".to_string()),
            Expression::variable("w".to_string()),
        ],
    ])
    .into();
    assert_eq!(expr.node_count(), 5);
}

#[test]
fn test_node_count_integral() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    }
    .into();
    assert_eq!(expr.node_count(), 2);

    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_sum() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_equation() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_inequality() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_complex_expression() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::integer(2)),
                right: Box::new(Expression::constant(MathConstant::Pi)),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    assert_eq!(expr.node_count(), 5);
}
