use crate::ast::{BinaryOp, Expression, InequalityOp, IntegralBounds, MathConstant, UnaryOp};

// Tests for depth

#[test]
fn test_depth_leaf_nodes() {
    assert_eq!(Expression::Integer(42).depth(), 1);
    assert_eq!(
        Expression::Float(crate::ast::MathFloat::from(3.14)).depth(),
        1
    );
    assert_eq!(Expression::Variable("x".to_string()).depth(), 1);
    assert_eq!(Expression::Constant(MathConstant::Pi).depth(), 1);
}

#[test]
fn test_depth_unary() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Variable("x".to_string())),
    };
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_nested() {
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        }),
        right: Box::new(Expression::Variable("z".to_string())),
    };
    assert_eq!(expr.depth(), 3);
}

#[test]
fn test_depth_asymmetric() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            right: Box::new(Expression::Variable("z".to_string())),
        }),
        right: Box::new(Expression::Variable("w".to_string())),
    };
    assert_eq!(expr.depth(), 4);
}

#[test]
fn test_depth_function() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };
    assert_eq!(expr.depth(), 2);

    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }],
    };
    assert_eq!(expr.depth(), 3);
}

#[test]
fn test_depth_vector() {
    let expr = Expression::Vector(vec![
        Expression::Variable("x".to_string()),
        Expression::Variable("y".to_string()),
    ]);
    assert_eq!(expr.depth(), 2);

    let expr = Expression::Vector(vec![]);
    assert_eq!(expr.depth(), 1);
}

#[test]
fn test_depth_matrix() {
    let expr = Expression::Matrix(vec![
        vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
        ],
        vec![
            Expression::Variable("z".to_string()),
            Expression::Variable("w".to_string()),
        ],
    ]);
    assert_eq!(expr.depth(), 2);
}

#[test]
fn test_depth_integral() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };
    assert_eq!(expr.depth(), 2);
}

// Tests for node_count

#[test]
fn test_node_count_leaf_nodes() {
    assert_eq!(Expression::Integer(42).node_count(), 1);
    assert_eq!(
        Expression::Float(crate::ast::MathFloat::from(3.14)).node_count(),
        1
    );
    assert_eq!(Expression::Variable("x".to_string()).node_count(), 1);
    assert_eq!(Expression::Constant(MathConstant::Pi).node_count(), 1);
}

#[test]
fn test_node_count_unary() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Variable("x".to_string())),
    };
    assert_eq!(expr.node_count(), 2);
}

#[test]
fn test_node_count_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_nested() {
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        }),
        right: Box::new(Expression::Variable("z".to_string())),
    };
    assert_eq!(expr.node_count(), 5);
}

#[test]
fn test_node_count_function() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![],
    };
    assert_eq!(expr.node_count(), 1);

    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };
    assert_eq!(expr.node_count(), 2);

    let expr = Expression::Function {
        name: "max".to_string(),
        args: vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
            Expression::Variable("z".to_string()),
        ],
    };
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_vector() {
    let expr = Expression::Vector(vec![]);
    assert_eq!(expr.node_count(), 1);

    let expr = Expression::Vector(vec![
        Expression::Variable("x".to_string()),
        Expression::Variable("y".to_string()),
        Expression::Variable("z".to_string()),
    ]);
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_matrix() {
    let expr = Expression::Matrix(vec![
        vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
        ],
        vec![
            Expression::Variable("z".to_string()),
            Expression::Variable("w".to_string()),
        ],
    ]);
    assert_eq!(expr.node_count(), 5);
}

#[test]
fn test_node_count_integral() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    };
    assert_eq!(expr.node_count(), 2);

    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_sum() {
    let expr = Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    };
    assert_eq!(expr.node_count(), 4);
}

#[test]
fn test_node_count_equation() {
    let expr = Expression::Equation {
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_inequality() {
    let expr = Expression::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    assert_eq!(expr.node_count(), 3);
}

#[test]
fn test_node_count_complex_expression() {
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Constant(MathConstant::Pi)),
        }),
        right: Box::new(Expression::Variable("x".to_string())),
    };
    assert_eq!(expr.node_count(), 5);
}
