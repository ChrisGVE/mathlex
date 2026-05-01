#![allow(clippy::approx_constant)]

use super::*;

// Tests for MathConstant
#[test]
fn test_math_constant_variants() {
    let pi = MathConstant::Pi;
    let e = MathConstant::E;
    let i = MathConstant::I;
    let inf = MathConstant::Infinity;
    let neg_inf = MathConstant::NegInfinity;

    // Verify they are all distinct
    assert_ne!(pi, e);
    assert_ne!(e, i);
    assert_ne!(i, inf);
    assert_ne!(inf, neg_inf);
}

#[test]
fn test_math_constant_copy() {
    let pi = MathConstant::Pi;
    let pi_copy = pi;
    assert_eq!(pi, pi_copy);
}

#[test]
fn test_math_constant_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(MathConstant::Pi);
    set.insert(MathConstant::E);
    set.insert(MathConstant::Pi); // Duplicate

    assert_eq!(set.len(), 2); // Only Pi and E
}

// Tests for BinaryOp
#[test]
fn test_binary_op_variants() {
    let add = BinaryOp::Add;
    let sub = BinaryOp::Sub;
    let mul = BinaryOp::Mul;
    let div = BinaryOp::Div;
    let pow = BinaryOp::Pow;
    let modulo = BinaryOp::Mod;

    assert_ne!(add, sub);
    assert_ne!(mul, div);
    assert_ne!(pow, modulo);
}

#[test]
fn test_binary_op_copy() {
    let add = BinaryOp::Add;
    let add_copy = add;
    assert_eq!(add, add_copy);
}

#[test]
fn test_binary_op_hash() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(BinaryOp::Add, "addition");
    map.insert(BinaryOp::Mul, "multiplication");

    assert_eq!(map.get(&BinaryOp::Add), Some(&"addition"));
    assert_eq!(map.get(&BinaryOp::Mul), Some(&"multiplication"));
}

// Tests for UnaryOp
#[test]
fn test_unary_op_variants() {
    let neg = UnaryOp::Neg;
    let pos = UnaryOp::Pos;
    let fact = UnaryOp::Factorial;
    let transpose = UnaryOp::Transpose;

    assert_ne!(neg, pos);
    assert_ne!(fact, transpose);
}

#[test]
fn test_unary_op_copy() {
    let neg = UnaryOp::Neg;
    let neg_copy = neg;
    assert_eq!(neg, neg_copy);
}

// Tests for Direction
#[test]
fn test_direction_variants() {
    let left = Direction::Left;
    let right = Direction::Right;
    let both = Direction::Both;

    assert_ne!(left, right);
    assert_ne!(right, both);
    assert_ne!(left, both);
}

#[test]
fn test_direction_copy() {
    let left = Direction::Left;
    let left_copy = left;
    assert_eq!(left, left_copy);
}

// Tests for InequalityOp
#[test]
fn test_inequality_op_variants() {
    let lt = InequalityOp::Lt;
    let le = InequalityOp::Le;
    let gt = InequalityOp::Gt;
    let ge = InequalityOp::Ge;
    let ne = InequalityOp::Ne;

    assert_ne!(lt, le);
    assert_ne!(gt, ge);
    assert_ne!(lt, gt);
    assert_ne!(ne, lt);
}

#[test]
fn test_inequality_op_copy() {
    let lt = InequalityOp::Lt;
    let lt_copy = lt;
    assert_eq!(lt, lt_copy);
}

// Tests for IntegralBounds
#[test]
fn test_integral_bounds_creation() {
    let bounds = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    assert!(matches!(bounds.lower.kind, ExprKind::Integer(0)));
    assert!(matches!(bounds.upper.kind, ExprKind::Integer(1)));
}

#[test]
fn test_integral_bounds_clone() {
    let bounds = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    let bounds_clone = bounds.clone();

    assert!(matches!(bounds_clone.lower.kind, ExprKind::Integer(0)));
    assert!(matches!(bounds_clone.upper.kind, ExprKind::Integer(1)));
}

// Tests for Expression - Integer
#[test]
fn test_expression_integer() {
    let expr = Expression::integer(42);
    match &expr.kind {
        ExprKind::Integer(n) => assert_eq!(*n, 42),
        _ => panic!("Expected Integer variant"),
    }
}

#[test]
fn test_expression_integer_negative() {
    let expr = Expression::integer(-17);
    match &expr.kind {
        ExprKind::Integer(n) => assert_eq!(*n, -17),
        _ => panic!("Expected Integer variant"),
    }
}

#[test]
fn test_expression_integer_clone() {
    let expr = Expression::integer(42);
    let expr_clone = expr.clone();

    assert_eq!(expr.kind, expr_clone.kind);
}

// Tests for Expression - Float
#[test]
fn test_expression_float() {
    let expr = Expression::float(MathFloat::from(42.5));
    match &expr.kind {
        ExprKind::Float(f) => {
            let value: f64 = (*f).into();
            assert!((value - 42.5).abs() < 1e-10);
        }
        _ => panic!("Expected Float variant"),
    }
}

#[test]
fn test_expression_float_negative() {
    let expr = Expression::float(MathFloat::from(-2.5));
    match &expr.kind {
        ExprKind::Float(f) => {
            let value: f64 = (*f).into();
            assert!((value + 2.5).abs() < 1e-10);
        }
        _ => panic!("Expected Float variant"),
    }
}

// Tests for Expression - Rational
#[test]
fn test_expression_rational() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::integer(1)),
        denominator: Box::new(Expression::integer(2)),
    }
    .into();

    match &expr.kind {
        ExprKind::Rational {
            numerator,
            denominator,
        } => {
            assert!(matches!(numerator.kind, ExprKind::Integer(1)));
            assert!(matches!(denominator.kind, ExprKind::Integer(2)));
        }
        _ => panic!("Expected Rational variant"),
    }
}

#[test]
fn test_expression_rational_clone() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::integer(3)),
        denominator: Box::new(Expression::integer(4)),
    }
    .into();

    let expr_clone = expr.clone();

    match &expr_clone.kind {
        ExprKind::Rational {
            numerator,
            denominator,
        } => {
            assert!(matches!(numerator.kind, ExprKind::Integer(3)));
            assert!(matches!(denominator.kind, ExprKind::Integer(4)));
        }
        _ => panic!("Expected Rational variant"),
    }
}

// Tests for Expression - Complex
#[test]
fn test_expression_complex() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(3)),
        imaginary: Box::new(Expression::integer(4)),
    }
    .into();

    match &expr.kind {
        ExprKind::Complex { real, imaginary } => {
            assert!(matches!(real.kind, ExprKind::Integer(3)));
            assert!(matches!(imaginary.kind, ExprKind::Integer(4)));
        }
        _ => panic!("Expected Complex variant"),
    }
}

#[test]
fn test_expression_complex_pure_imaginary() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(0)),
        imaginary: Box::new(Expression::integer(1)),
    }
    .into();

    match &expr.kind {
        ExprKind::Complex { real, imaginary } => {
            assert!(matches!(real.kind, ExprKind::Integer(0)));
            assert!(matches!(imaginary.kind, ExprKind::Integer(1)));
        }
        _ => panic!("Expected Complex variant"),
    }
}

// Tests for Expression - Variable
#[test]
fn test_expression_variable() {
    let expr = Expression::variable("x".to_string());
    match &expr.kind {
        ExprKind::Variable(name) => assert_eq!(name, "x"),
        _ => panic!("Expected Variable variant"),
    }
}

#[test]
fn test_expression_variable_greek() {
    let expr = Expression::variable("theta".to_string());
    match &expr.kind {
        ExprKind::Variable(name) => assert_eq!(name, "theta"),
        _ => panic!("Expected Variable variant"),
    }
}

#[test]
fn test_expression_variable_subscript() {
    let expr = Expression::variable("x_1".to_string());
    match &expr.kind {
        ExprKind::Variable(name) => assert_eq!(name, "x_1"),
        _ => panic!("Expected Variable variant"),
    }
}

// Tests for Expression - Constant
#[test]
fn test_expression_constant_pi() {
    let expr = Expression::constant(MathConstant::Pi);
    match &expr.kind {
        ExprKind::Constant(c) => assert_eq!(*c, MathConstant::Pi),
        _ => panic!("Expected Constant variant"),
    }
}

#[test]
fn test_expression_constant_e() {
    let expr = Expression::constant(MathConstant::E);
    match &expr.kind {
        ExprKind::Constant(c) => assert_eq!(*c, MathConstant::E),
        _ => panic!("Expected Constant variant"),
    }
}

// Tests for Expression - Binary
#[test]
fn test_expression_binary_add() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::integer(3)),
    }
    .into();

    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            assert_eq!(op, &BinaryOp::Add);
            assert!(matches!(left.kind, ExprKind::Integer(2)));
            assert!(matches!(right.kind, ExprKind::Integer(3)));
        }
        _ => panic!("Expected Binary variant"),
    }
}

#[test]
fn test_expression_binary_nested() {
    // (2 + 3) * 4
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

    match &expr.kind {
        ExprKind::Binary { op, left, .. } => {
            assert_eq!(op, &BinaryOp::Mul);
            match &left.kind {
                ExprKind::Binary { op, .. } => assert_eq!(op, &BinaryOp::Add),
                _ => panic!("Expected nested Binary"),
            }
        }
        _ => panic!("Expected Binary variant"),
    }
}

// Tests for Expression - Unary
#[test]
fn test_expression_unary_neg() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::integer(5)),
    }
    .into();

    match &expr.kind {
        ExprKind::Unary { op, operand } => {
            assert_eq!(op, &UnaryOp::Neg);
            assert!(matches!(operand.kind, ExprKind::Integer(5)));
        }
        _ => panic!("Expected Unary variant"),
    }
}

#[test]
fn test_expression_unary_factorial() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::variable("n".to_string())),
    }
    .into();

    match &expr.kind {
        ExprKind::Unary { op, operand } => {
            assert_eq!(op, &UnaryOp::Factorial);
            match &operand.kind {
                ExprKind::Variable(name) => assert_eq!(name, "n"),
                _ => panic!("Expected Variable operand"),
            }
        }
        _ => panic!("Expected Unary variant"),
    }
}

// Tests for Expression - Function
#[test]
fn test_expression_function_no_args() {
    let expr: Expression = ExprKind::Function {
        name: "f".to_string(),
        args: vec![],
    }
    .into();

    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "f");
            assert_eq!(args.len(), 0);
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_expression_function_one_arg() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    }
    .into();

    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            match &args[0].kind {
                ExprKind::Variable(v) => assert_eq!(v, "x"),
                _ => panic!("Expected Variable argument"),
            }
        }
        _ => panic!("Expected Function variant"),
    }
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

    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "max");
            assert_eq!(args.len(), 3);
        }
        _ => panic!("Expected Function variant"),
    }
}

// Tests for Expression - Derivative
#[test]
fn test_expression_derivative_first_order() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();

    match &expr.kind {
        ExprKind::Derivative { expr, var, order } => {
            assert!(matches!(expr.kind, ExprKind::Variable(_)));
            assert_eq!(var, "x");
            assert_eq!(*order, 1);
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_expression_derivative_second_order() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    }
    .into();

    match &expr.kind {
        ExprKind::Derivative { order, .. } => assert_eq!(*order, 2),
        _ => panic!("Expected Derivative variant"),
    }
}

// Tests for Expression - PartialDerivative
#[test]
fn test_expression_partial_derivative() {
    let expr: Expression = ExprKind::PartialDerivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    }
    .into();

    match &expr.kind {
        ExprKind::PartialDerivative { expr, var, order } => {
            assert!(matches!(expr.kind, ExprKind::Variable(_)));
            assert_eq!(var, "x");
            assert_eq!(*order, 1);
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_expression_partial_derivative_higher_order() {
    let expr: Expression = ExprKind::PartialDerivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "y".to_string(),
        order: 3,
    }
    .into();

    match &expr.kind {
        ExprKind::PartialDerivative { var, order, .. } => {
            assert_eq!(var, "y");
            assert_eq!(*order, 3);
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

// Tests for Expression - Integral
#[test]
fn test_expression_integral_indefinite() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    }
    .into();

    match &expr.kind {
        ExprKind::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert!(matches!(integrand.kind, ExprKind::Variable(_)));
            assert_eq!(var, "x");
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
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

    match &expr.kind {
        ExprKind::Integral { bounds, .. } => {
            assert!(bounds.is_some());
            let bounds = bounds.as_ref().unwrap();
            assert!(matches!(bounds.lower.kind, ExprKind::Integer(0)));
            assert!(matches!(bounds.upper.kind, ExprKind::Integer(1)));
        }
        _ => panic!("Expected Integral variant"),
    }
}

// Tests for Expression - Limit
#[test]
fn test_expression_limit_both_sides() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Both,
    }
    .into();

    match &expr.kind {
        ExprKind::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            assert!(matches!(expr.kind, ExprKind::Variable(_)));
            assert_eq!(var, "x");
            assert!(matches!(to.kind, ExprKind::Integer(0)));
            assert_eq!(*direction, Direction::Both);
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_expression_limit_from_left() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Left,
    }
    .into();

    match &expr.kind {
        ExprKind::Limit { direction, .. } => assert_eq!(*direction, Direction::Left),
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_expression_limit_to_infinity() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::constant(MathConstant::Infinity)),
        direction: Direction::Both,
    }
    .into();

    match &expr.kind {
        ExprKind::Limit { to, .. } => {
            assert!(matches!(
                to.kind,
                ExprKind::Constant(MathConstant::Infinity)
            ));
        }
        _ => panic!("Expected Limit variant"),
    }
}

// Tests for Expression - Sum
#[test]
fn test_expression_sum() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();

    match &expr.kind {
        ExprKind::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert!(matches!(lower.kind, ExprKind::Integer(1)));
            assert!(matches!(upper.kind, ExprKind::Variable(_)));
            assert!(matches!(body.kind, ExprKind::Variable(_)));
        }
        _ => panic!("Expected Sum variant"),
    }
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

    match &expr.kind {
        ExprKind::Sum { body, .. } => {
            assert!(matches!(body.kind, ExprKind::Binary { .. }));
        }
        _ => panic!("Expected Sum variant"),
    }
}

// Tests for Expression - Product
#[test]
fn test_expression_product() {
    let expr: Expression = ExprKind::Product {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    }
    .into();

    match &expr.kind {
        ExprKind::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert!(matches!(lower.kind, ExprKind::Integer(1)));
            assert!(matches!(upper.kind, ExprKind::Variable(_)));
            assert!(matches!(body.kind, ExprKind::Variable(_)));
        }
        _ => panic!("Expected Product variant"),
    }
}

// Tests for Expression - Vector
#[test]
fn test_expression_vector_empty() {
    let expr: Expression = ExprKind::Vector(vec![]).into();
    match &expr.kind {
        ExprKind::Vector(elements) => assert_eq!(elements.len(), 0),
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_single() {
    let expr: Expression = ExprKind::Vector(vec![Expression::integer(1)]).into();
    match &expr.kind {
        ExprKind::Vector(elements) => {
            assert_eq!(elements.len(), 1);
            assert!(matches!(elements[0].kind, ExprKind::Integer(1)));
        }
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_multiple() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::integer(2),
        Expression::integer(3),
    ])
    .into();

    match &expr.kind {
        ExprKind::Vector(elements) => {
            assert_eq!(elements.len(), 3);
            assert!(matches!(elements[0].kind, ExprKind::Integer(1)));
            assert!(matches!(elements[1].kind, ExprKind::Integer(2)));
            assert!(matches!(elements[2].kind, ExprKind::Integer(3)));
        }
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_mixed_types() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::variable("x".to_string()),
        Expression::float(MathFloat::from(2.5)),
    ])
    .into();

    match &expr.kind {
        ExprKind::Vector(elements) => assert_eq!(elements.len(), 3),
        _ => panic!("Expected Vector variant"),
    }
}

// Tests for Expression - Matrix
#[test]
fn test_expression_matrix_empty() {
    let expr: Expression = ExprKind::Matrix(vec![]).into();
    match &expr.kind {
        ExprKind::Matrix(rows) => assert_eq!(rows.len(), 0),
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_single_element() {
    let expr: Expression = ExprKind::Matrix(vec![vec![Expression::integer(1)]]).into();

    match &expr.kind {
        ExprKind::Matrix(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 1);
            assert!(matches!(rows[0][0].kind, ExprKind::Integer(1)));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_2x2() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ])
    .into();

    match &expr.kind {
        ExprKind::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
            assert_eq!(rows[1].len(), 2);
            assert!(matches!(rows[0][0].kind, ExprKind::Integer(1)));
            assert!(matches!(rows[1][1].kind, ExprKind::Integer(4)));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_rectangular() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![
            Expression::integer(1),
            Expression::integer(2),
            Expression::integer(3),
        ],
        vec![
            Expression::integer(4),
            Expression::integer(5),
            Expression::integer(6),
        ],
    ])
    .into();

    match &expr.kind {
        ExprKind::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[1].len(), 3);
        }
        _ => panic!("Expected Matrix variant"),
    }
}

// Tests for Expression - Equation
#[test]
fn test_expression_equation_simple() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(5)),
    }
    .into();

    match &expr.kind {
        ExprKind::Equation { left, right } => {
            assert!(matches!(left.kind, ExprKind::Variable(_)));
            assert!(matches!(right.kind, ExprKind::Integer(5)));
        }
        _ => panic!("Expected Equation variant"),
    }
}

#[test]
fn test_expression_equation_complex() {
    // y = 2x + 1
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

    match &expr.kind {
        ExprKind::Equation { right, .. } => {
            assert!(matches!(right.kind, ExprKind::Binary { .. }));
        }
        _ => panic!("Expected Equation variant"),
    }
}

// Tests for Expression - Inequality
#[test]
fn test_expression_inequality_less_than() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(5)),
    }
    .into();

    match &expr.kind {
        ExprKind::Inequality { op, left, right } => {
            assert_eq!(*op, InequalityOp::Lt);
            assert!(matches!(left.kind, ExprKind::Variable(_)));
            assert!(matches!(right.kind, ExprKind::Integer(5)));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_expression_inequality_greater_equal() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Ge,
        left: Box::new(Expression::variable("y".to_string())),
        right: Box::new(Expression::integer(0)),
    }
    .into();

    match &expr.kind {
        ExprKind::Inequality { op, .. } => assert_eq!(*op, InequalityOp::Ge),
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_expression_inequality_not_equal() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Ne,
        left: Box::new(Expression::variable("a".to_string())),
        right: Box::new(Expression::variable("b".to_string())),
    }
    .into();

    match &expr.kind {
        ExprKind::Inequality { op, .. } => assert_eq!(*op, InequalityOp::Ne),
        _ => panic!("Expected Inequality variant"),
    }
}

// Test ExprKind clone
#[test]
fn test_expression_clone_deep() {
    let kind = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::variable("x".to_string())),
    };

    let kind_clone = kind.clone();

    match (kind, kind_clone) {
        (ExprKind::Binary { op: op1, .. }, ExprKind::Binary { op: op2, .. }) => {
            assert_eq!(op1, op2);
        }
        _ => panic!("Clone failed"),
    }
}

// Test Debug trait
#[test]
fn test_expression_debug() {
    let expr = Expression::integer(42);
    let debug_str = format!("{:?}", expr);
    assert!(debug_str.contains("Integer"));
    assert!(debug_str.contains("42"));
}

// Tests for MathFloat
#[test]
fn test_math_float_creation() {
    let f1 = MathFloat::new(3.14);
    let f2 = MathFloat::from(3.14);
    assert_eq!(f1, f2);
}

#[test]
fn test_math_float_value_extraction() {
    let f = MathFloat::from(2.718);
    assert_eq!(f.value(), 2.718);

    let val: f64 = f.into();
    assert_eq!(val, 2.718);
}

#[test]
fn test_math_float_equality() {
    let f1 = MathFloat::from(1.5);
    let f2 = MathFloat::from(1.5);
    let f3 = MathFloat::from(2.5);

    assert_eq!(f1, f2);
    assert_ne!(f1, f3);
}

#[test]
fn test_math_float_copy() {
    let f1 = MathFloat::from(3.14);
    let f2 = f1;
    assert_eq!(f1, f2);
}

#[test]
fn test_math_float_display() {
    let f = MathFloat::from(3.14159);
    let display_str = format!("{}", f);
    assert_eq!(display_str, "3.14159");
}

#[test]
fn test_math_float_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(MathFloat::from(1.0));
    set.insert(MathFloat::from(2.0));
    set.insert(MathFloat::from(1.0)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&MathFloat::from(1.0)));
    assert!(set.contains(&MathFloat::from(2.0)));
}

#[test]
fn test_math_float_nan_equality() {
    // NaN values should compare equal to themselves in MathFloat
    let nan1 = MathFloat::from(f64::NAN);
    let nan2 = MathFloat::from(f64::NAN);
    assert_eq!(nan1, nan2);
}

#[test]
fn test_math_float_infinity() {
    let inf = MathFloat::from(f64::INFINITY);
    let neg_inf = MathFloat::from(f64::NEG_INFINITY);

    assert_ne!(inf, neg_inf);
    assert_eq!(inf, MathFloat::from(f64::INFINITY));
}

// Tests for IntegralBounds equality and hashing
#[test]
fn test_integral_bounds_equality() {
    let bounds1 = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    let bounds2 = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    let bounds3 = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(2)),
    };

    assert_eq!(bounds1, bounds2);
    assert_ne!(bounds1, bounds3);
}

#[test]
fn test_integral_bounds_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let bounds1 = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    let bounds2 = IntegralBounds {
        lower: Box::new(Expression::integer(0)),
        upper: Box::new(Expression::integer(1)),
    };

    set.insert(bounds1);
    set.insert(bounds2); // Should be considered duplicate

    assert_eq!(set.len(), 1);
}

// Tests for Expression equality
#[test]
fn test_expression_integer_equality() {
    let e1 = Expression::integer(42);
    let e2 = Expression::integer(42);
    let e3 = Expression::integer(43);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_float_equality() {
    let e1 = Expression::float(MathFloat::from(3.14));
    let e2 = Expression::float(MathFloat::from(3.14));
    let e3 = Expression::float(MathFloat::from(2.71));

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_variable_equality() {
    let e1 = Expression::variable("x".to_string());
    let e2 = Expression::variable("x".to_string());
    let e3 = Expression::variable("y".to_string());

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_binary_equality() {
    let e1 = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    };

    let e2 = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    };

    let e3 = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_nested_equality() {
    // (1 + 2) * 3
    let e1 = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(1)),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        right: Box::new(Expression::integer(3)),
    };

    let e2 = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(1)),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        right: Box::new(Expression::integer(3)),
    };

    assert_eq!(e1, e2);
}

#[test]
fn test_expression_vector_equality() {
    let e1 = ExprKind::Vector(vec![Expression::integer(1), Expression::integer(2)]);

    let e2 = ExprKind::Vector(vec![Expression::integer(1), Expression::integer(2)]);

    let e3 = ExprKind::Vector(vec![Expression::integer(1), Expression::integer(3)]);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_matrix_equality() {
    let e1 = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ]);

    let e2 = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ]);

    let e3 = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(5)],
    ]);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

// Tests for Expression hashing and use in collections
#[test]
fn test_expression_hash_set() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(Expression::integer(1));
    set.insert(Expression::integer(2));
    set.insert(Expression::integer(1)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&Expression::integer(1)));
    assert!(set.contains(&Expression::integer(2)));
}

#[test]
fn test_expression_hash_map() {
    use std::collections::HashMap;
    let mut map = HashMap::new();

    map.insert(Expression::variable("x".to_string()), 42);
    map.insert(Expression::variable("y".to_string()), 17);
    map.insert(Expression::variable("x".to_string()), 99); // Update

    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&Expression::variable("x".to_string())), Some(&99));
    assert_eq!(map.get(&Expression::variable("y".to_string())), Some(&17));
}

#[test]
fn test_expression_complex_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let expr1 = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    };

    let expr2 = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    };

    set.insert(expr1);
    set.insert(expr2); // Should be considered duplicate

    assert_eq!(set.len(), 1);
}

#[test]
fn test_expression_float_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(Expression::float(MathFloat::from(3.14)));
    set.insert(Expression::float(MathFloat::from(2.71)));
    set.insert(Expression::float(MathFloat::from(3.14))); // Duplicate

    assert_eq!(set.len(), 2);
}

#[test]
fn test_expression_function_equality() {
    let e1 = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    };

    let e2 = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    };

    let e3 = ExprKind::Function {
        name: "cos".to_string(),
        args: vec![Expression::variable("x".to_string())],
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_derivative_equality() {
    let e1 = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    let e2 = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    let e3 = ExprKind::Derivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_integral_equality() {
    let e1 = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    };

    let e2 = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    };

    let e3 = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

// Tests for serde serialization/deserialization
#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_integer() {
        let expr = Expression::integer(42);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_float() {
        let expr = Expression::float(MathFloat::from(3.14159));
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_variable() {
        let expr = Expression::variable("x".to_string());
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_constant() {
        let expr = Expression::constant(MathConstant::Pi);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_binary() {
        let expr: Expression = ExprKind::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::integer(2)),
            right: Box::new(Expression::variable("x".to_string())),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_unary() {
        let expr: Expression = ExprKind::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::integer(5)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_function() {
        let expr: Expression = ExprKind::Function {
            name: "sin".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_rational() {
        let expr: Expression = ExprKind::Rational {
            numerator: Box::new(Expression::integer(1)),
            denominator: Box::new(Expression::integer(2)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_complex() {
        let expr: Expression = ExprKind::Complex {
            real: Box::new(Expression::integer(3)),
            imaginary: Box::new(Expression::integer(4)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_derivative() {
        let expr: Expression = ExprKind::Derivative {
            expr: Box::new(Expression::variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_partial_derivative() {
        let expr: Expression = ExprKind::PartialDerivative {
            expr: Box::new(Expression::variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_integral_indefinite() {
        let expr: Expression = ExprKind::Integral {
            integrand: Box::new(Expression::variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_integral_definite() {
        let expr: Expression = ExprKind::Integral {
            integrand: Box::new(Expression::variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::integer(0)),
                upper: Box::new(Expression::integer(1)),
            }),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_limit() {
        let expr: Expression = ExprKind::Limit {
            expr: Box::new(Expression::variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::integer(0)),
            direction: Direction::Both,
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_sum() {
        let expr: Expression = ExprKind::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::integer(1)),
            upper: Box::new(Expression::variable("n".to_string())),
            body: Box::new(Expression::variable("i".to_string())),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_product() {
        let expr: Expression = ExprKind::Product {
            index: "i".to_string(),
            lower: Box::new(Expression::integer(1)),
            upper: Box::new(Expression::variable("n".to_string())),
            body: Box::new(Expression::variable("i".to_string())),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_vector() {
        let expr: Expression = ExprKind::Vector(vec![
            Expression::integer(1),
            Expression::integer(2),
            Expression::integer(3),
        ])
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_matrix() {
        let expr: Expression = ExprKind::Matrix(vec![
            vec![Expression::integer(1), Expression::integer(2)],
            vec![Expression::integer(3), Expression::integer(4)],
        ])
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_equation() {
        let expr: Expression = ExprKind::Equation {
            left: Box::new(Expression::variable("x".to_string())),
            right: Box::new(Expression::integer(5)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_inequality() {
        let expr: Expression = ExprKind::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::variable("x".to_string())),
            right: Box::new(Expression::integer(5)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_nested_expression() {
        // (2 + x) * 3
        let expr: Expression = ExprKind::Binary {
            op: BinaryOp::Mul,
            left: Box::new(
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::integer(2)),
                    right: Box::new(Expression::variable("x".to_string())),
                }
                .into(),
            ),
            right: Box::new(Expression::integer(3)),
        }
        .into();
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_math_float() {
        let float = MathFloat::from(3.14159);
        let json = serde_json::to_string(&float).unwrap();
        let parsed: MathFloat = serde_json::from_str(&json).unwrap();
        assert_eq!(float, parsed);
    }

    #[test]
    fn test_serialize_deserialize_all_constants() {
        let constants = vec![
            MathConstant::Pi,
            MathConstant::E,
            MathConstant::I,
            MathConstant::Infinity,
            MathConstant::NegInfinity,
        ];
        for constant in constants {
            let json = serde_json::to_string(&constant).unwrap();
            let parsed: MathConstant = serde_json::from_str(&json).unwrap();
            assert_eq!(constant, parsed);
        }
    }

    #[test]
    fn test_serialize_deserialize_all_binary_ops() {
        let ops = vec![
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Mod,
        ];
        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let parsed: BinaryOp = serde_json::from_str(&json).unwrap();
            assert_eq!(op, parsed);
        }
    }

    #[test]
    fn test_serialize_deserialize_all_unary_ops() {
        let ops = vec![
            UnaryOp::Neg,
            UnaryOp::Pos,
            UnaryOp::Factorial,
            UnaryOp::Transpose,
        ];
        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let parsed: UnaryOp = serde_json::from_str(&json).unwrap();
            assert_eq!(op, parsed);
        }
    }

    #[test]
    fn test_serialize_deserialize_all_directions() {
        let directions = vec![Direction::Left, Direction::Right, Direction::Both];
        for direction in directions {
            let json = serde_json::to_string(&direction).unwrap();
            let parsed: Direction = serde_json::from_str(&json).unwrap();
            assert_eq!(direction, parsed);
        }
    }

    #[test]
    fn test_serialize_deserialize_all_inequality_ops() {
        let ops = vec![
            InequalityOp::Lt,
            InequalityOp::Le,
            InequalityOp::Gt,
            InequalityOp::Ge,
            InequalityOp::Ne,
        ];
        for op in ops {
            let json = serde_json::to_string(&op).unwrap();
            let parsed: InequalityOp = serde_json::from_str(&json).unwrap();
            assert_eq!(op, parsed);
        }
    }

    #[test]
    fn test_serialize_deserialize_integral_bounds() {
        let bounds = IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(10)),
        };
        let json = serde_json::to_string(&bounds).unwrap();
        let parsed: IntegralBounds = serde_json::from_str(&json).unwrap();
        assert_eq!(bounds, parsed);
    }

    #[test]
    fn test_math_float_nan_serialization() {
        // Note: JSON doesn't natively support NaN, it serializes to null
        // This is expected behavior from ordered-float's serde implementation
        let nan = MathFloat::from(f64::NAN);
        let json = serde_json::to_string(&nan).unwrap();
        assert_eq!(json, "null");

        // For actual round-trip preservation of NaN, use binary formats like bincode
        // JSON explicitly doesn't support NaN per spec
    }

    #[test]
    fn test_math_float_infinity_serialization() {
        // Note: JSON doesn't natively support Infinity, it serializes to null
        // This is expected behavior from ordered-float's serde implementation
        let inf = MathFloat::from(f64::INFINITY);
        let json = serde_json::to_string(&inf).unwrap();
        assert_eq!(json, "null");

        let neg_inf = MathFloat::from(f64::NEG_INFINITY);
        let json = serde_json::to_string(&neg_inf).unwrap();
        assert_eq!(json, "null");

        // For actual round-trip preservation of special floats, use binary formats
    }
}
