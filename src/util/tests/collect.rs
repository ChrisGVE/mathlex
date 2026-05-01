use crate::ast::{
    BinaryOp, Direction, ExprKind, Expression, IntegralBounds, MathConstant, UnaryOp,
};

// Tests for find_variables

#[test]
fn test_find_variables_leaf_nodes() {
    let expr = Expression::integer(42);
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::float(crate::ast::MathFloat::from(3.14));
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::constant(MathConstant::Pi);
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::variable("x".to_string());
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_duplicate() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_rational() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::variable("x".to_string())),
        denominator: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_complex() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::variable("a".to_string())),
        imaginary: Box::new(Expression::variable("b".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
}

#[test]
fn test_find_variables_unary() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_function() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![
            Expression::variable("x".to_string()),
            Expression::variable("y".to_string()),
        ],
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_derivative() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("f"));
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_integral() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_integral_with_variable_bounds() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::variable("a".to_string())),
            upper: Box::new(Expression::variable("b".to_string())),
        }),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains("x"));
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
}

#[test]
fn test_find_variables_limit() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Both,
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("f"));
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_sum() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("i"));
    assert!(vars.contains("n"));
}

#[test]
fn test_find_variables_vector() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::variable("x".to_string()),
        Expression::variable("y".to_string()),
        Expression::variable("z".to_string()),
    ])
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
    assert!(vars.contains("z"));
}

#[test]
fn test_find_variables_matrix() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![
            Expression::variable("a".to_string()),
            Expression::variable("b".to_string()),
        ],
        vec![
            Expression::variable("c".to_string()),
            Expression::variable("d".to_string()),
        ],
    ])
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 4);
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
    assert!(vars.contains("c"));
    assert!(vars.contains("d"));
}

#[test]
fn test_find_variables_equation() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

// Tests for find_functions

#[test]
fn test_find_functions_leaf_nodes() {
    let expr = Expression::variable("x".to_string());
    assert_eq!(expr.find_functions().len(), 0);
}

#[test]
fn test_find_functions_simple() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    }
    .into();
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

#[test]
fn test_find_functions_multiple() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(
            ExprKind::Function {
                name: "sin".to_string(),
                args: vec![Expression::variable("x".to_string())],
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Function {
                name: "cos".to_string(),
                args: vec![Expression::variable("y".to_string())],
            }
            .into(),
        ),
    }
    .into();
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 2);
    assert!(funcs.contains("sin"));
    assert!(funcs.contains("cos"));
}

#[test]
fn test_find_functions_nested() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![ExprKind::Function {
            name: "cos".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }
        .into()],
    }
    .into();
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 2);
    assert!(funcs.contains("sin"));
    assert!(funcs.contains("cos"));
}

#[test]
fn test_find_functions_duplicate() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(
            ExprKind::Function {
                name: "sin".to_string(),
                args: vec![Expression::variable("x".to_string())],
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Function {
                name: "sin".to_string(),
                args: vec![Expression::variable("y".to_string())],
            }
            .into(),
        ),
    }
    .into();
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

#[test]
fn test_find_functions_in_integral() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(
            ExprKind::Function {
                name: "sin".to_string(),
                args: vec![Expression::variable("x".to_string())],
            }
            .into(),
        ),
        var: "x".to_string(),
        bounds: None,
    }
    .into();
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

// Tests for find_constants

#[test]
fn test_find_constants_none() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(1)),
    }
    .into();
    assert_eq!(expr.find_constants().len(), 0);
}

#[test]
fn test_find_constants_single() {
    let expr = Expression::constant(MathConstant::Pi);
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Pi));
}

#[test]
fn test_find_constants_multiple() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::constant(MathConstant::Pi)),
        right: Box::new(Expression::constant(MathConstant::E)),
    }
    .into();
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 2);
    assert!(consts.contains(&MathConstant::Pi));
    assert!(consts.contains(&MathConstant::E));
}

#[test]
fn test_find_constants_duplicate() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::constant(MathConstant::Pi)),
        right: Box::new(Expression::constant(MathConstant::Pi)),
    }
    .into();
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Pi));
}

#[test]
fn test_find_constants_all_types() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::constant(MathConstant::Pi),
        Expression::constant(MathConstant::E),
        Expression::constant(MathConstant::I),
        Expression::constant(MathConstant::Infinity),
        Expression::constant(MathConstant::NegInfinity),
    ])
    .into();
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 5);
    assert!(consts.contains(&MathConstant::Pi));
    assert!(consts.contains(&MathConstant::E));
    assert!(consts.contains(&MathConstant::I));
    assert!(consts.contains(&MathConstant::Infinity));
    assert!(consts.contains(&MathConstant::NegInfinity));
}

#[test]
fn test_find_constants_in_limit() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::constant(MathConstant::Infinity)),
        direction: Direction::Both,
    }
    .into();
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Infinity));
}

// Tests for contains_variable

#[test]
fn test_contains_variable_present() {
    let expr = Expression::variable("x".to_string());
    assert!(expr.contains_variable("x"));
}

#[test]
fn test_contains_variable_absent() {
    let expr = Expression::variable("x".to_string());
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_in_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(1)),
    }
    .into();
    assert!(expr.contains_variable("x"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_deeply_nested() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(1)),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Function {
                name: "sin".to_string(),
                args: vec![Expression::variable("x".to_string())],
            }
            .into(),
        ),
    }
    .into();
    assert!(expr.contains_variable("x"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_in_derivative() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    assert!(expr.contains_variable("x"));
    assert!(expr.contains_variable("f"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_leaf_types() {
    assert!(!Expression::integer(42).contains_variable("x"));
    assert!(!Expression::constant(MathConstant::Pi).contains_variable("x"));
    assert!(!Expression::empty_set().contains_variable("x"));
    assert!(!Expression::nabla().contains_variable("x"));
}
