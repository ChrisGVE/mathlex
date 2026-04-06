use crate::ast::{BinaryOp, Direction, Expression, IntegralBounds, MathConstant, UnaryOp};

// Tests for find_variables

#[test]
fn test_find_variables_leaf_nodes() {
    let expr = Expression::Integer(42);
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::Float(crate::ast::MathFloat::from(3.14));
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::Constant(MathConstant::Pi);
    assert_eq!(expr.find_variables().len(), 0);

    let expr = Expression::Variable("x".to_string());
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_duplicate() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("x".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_rational() {
    let expr = Expression::Rational {
        numerator: Box::new(Expression::Variable("x".to_string())),
        denominator: Box::new(Expression::Variable("y".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_complex() {
    let expr = Expression::Complex {
        real: Box::new(Expression::Variable("a".to_string())),
        imaginary: Box::new(Expression::Variable("b".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
}

#[test]
fn test_find_variables_unary() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Variable("x".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_function() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
        ],
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

#[test]
fn test_find_variables_derivative() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("f"));
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_integral() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_integral_with_variable_bounds() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Variable("a".to_string())),
            upper: Box::new(Expression::Variable("b".to_string())),
        }),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains("x"));
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
}

#[test]
fn test_find_variables_limit() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(0)),
        direction: Direction::Both,
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("f"));
    assert!(vars.contains("x"));
}

#[test]
fn test_find_variables_sum() {
    let expr = Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("i"));
    assert!(vars.contains("n"));
}

#[test]
fn test_find_variables_vector() {
    let expr = Expression::Vector(vec![
        Expression::Variable("x".to_string()),
        Expression::Variable("y".to_string()),
        Expression::Variable("z".to_string()),
    ]);
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
    assert!(vars.contains("z"));
}

#[test]
fn test_find_variables_matrix() {
    let expr = Expression::Matrix(vec![
        vec![
            Expression::Variable("a".to_string()),
            Expression::Variable("b".to_string()),
        ],
        vec![
            Expression::Variable("c".to_string()),
            Expression::Variable("d".to_string()),
        ],
    ]);
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 4);
    assert!(vars.contains("a"));
    assert!(vars.contains("b"));
    assert!(vars.contains("c"));
    assert!(vars.contains("d"));
}

#[test]
fn test_find_variables_equation() {
    let expr = Expression::Equation {
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains("x"));
    assert!(vars.contains("y"));
}

// Tests for find_functions

#[test]
fn test_find_functions_leaf_nodes() {
    let expr = Expression::Variable("x".to_string());
    assert_eq!(expr.find_functions().len(), 0);
}

#[test]
fn test_find_functions_simple() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

#[test]
fn test_find_functions_multiple() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        right: Box::new(Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("y".to_string())],
        }),
    };
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 2);
    assert!(funcs.contains("sin"));
    assert!(funcs.contains("cos"));
}

#[test]
fn test_find_functions_nested() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }],
    };
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 2);
    assert!(funcs.contains("sin"));
    assert!(funcs.contains("cos"));
}

#[test]
fn test_find_functions_duplicate() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        right: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("y".to_string())],
        }),
    };
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

#[test]
fn test_find_functions_in_integral() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        var: "x".to_string(),
        bounds: None,
    };
    let funcs = expr.find_functions();
    assert_eq!(funcs.len(), 1);
    assert!(funcs.contains("sin"));
}

// Tests for find_constants

#[test]
fn test_find_constants_none() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(1)),
    };
    assert_eq!(expr.find_constants().len(), 0);
}

#[test]
fn test_find_constants_single() {
    let expr = Expression::Constant(MathConstant::Pi);
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Pi));
}

#[test]
fn test_find_constants_multiple() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Constant(MathConstant::Pi)),
        right: Box::new(Expression::Constant(MathConstant::E)),
    };
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 2);
    assert!(consts.contains(&MathConstant::Pi));
    assert!(consts.contains(&MathConstant::E));
}

#[test]
fn test_find_constants_duplicate() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Constant(MathConstant::Pi)),
        right: Box::new(Expression::Constant(MathConstant::Pi)),
    };
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Pi));
}

#[test]
fn test_find_constants_all_types() {
    let expr = Expression::Vector(vec![
        Expression::Constant(MathConstant::Pi),
        Expression::Constant(MathConstant::E),
        Expression::Constant(MathConstant::I),
        Expression::Constant(MathConstant::Infinity),
        Expression::Constant(MathConstant::NegInfinity),
    ]);
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
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Constant(MathConstant::Infinity)),
        direction: Direction::Both,
    };
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Infinity));
}

// Tests for contains_variable

#[test]
fn test_contains_variable_present() {
    let expr = Expression::Variable("x".to_string());
    assert!(expr.contains_variable("x"));
}

#[test]
fn test_contains_variable_absent() {
    let expr = Expression::Variable("x".to_string());
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_in_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(1)),
    };
    assert!(expr.contains_variable("x"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_deeply_nested() {
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
    };
    assert!(expr.contains_variable("x"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_in_derivative() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };
    assert!(expr.contains_variable("x"));
    assert!(expr.contains_variable("f"));
    assert!(!expr.contains_variable("y"));
}

#[test]
fn test_contains_variable_leaf_types() {
    assert!(!Expression::Integer(42).contains_variable("x"));
    assert!(!Expression::Constant(MathConstant::Pi).contains_variable("x"));
    assert!(!Expression::EmptySet.contains_variable("x"));
    assert!(!Expression::Nabla.contains_variable("x"));
}
