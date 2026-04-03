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
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    match (*bounds.lower, *bounds.upper) {
        (Expression::Integer(l), Expression::Integer(u)) => {
            assert_eq!(l, 0);
            assert_eq!(u, 1);
        }
        _ => panic!("Expected integer bounds"),
    }
}

#[test]
fn test_integral_bounds_clone() {
    let bounds = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    let bounds_clone = bounds.clone();

    match (*bounds_clone.lower, *bounds_clone.upper) {
        (Expression::Integer(l), Expression::Integer(u)) => {
            assert_eq!(l, 0);
            assert_eq!(u, 1);
        }
        _ => panic!("Expected integer bounds"),
    }
}

// Tests for Expression - Integer
#[test]
fn test_expression_integer() {
    let expr = Expression::Integer(42);
    match expr {
        Expression::Integer(n) => assert_eq!(n, 42),
        _ => panic!("Expected Integer variant"),
    }
}

#[test]
fn test_expression_integer_negative() {
    let expr = Expression::Integer(-17);
    match expr {
        Expression::Integer(n) => assert_eq!(n, -17),
        _ => panic!("Expected Integer variant"),
    }
}

#[test]
fn test_expression_integer_clone() {
    let expr = Expression::Integer(42);
    let expr_clone = expr.clone();

    match (expr, expr_clone) {
        (Expression::Integer(a), Expression::Integer(b)) => assert_eq!(a, b),
        _ => panic!("Expected Integer variants"),
    }
}

// Tests for Expression - Float
#[test]
fn test_expression_float() {
    let expr = Expression::Float(MathFloat::from(42.5));
    match expr {
        Expression::Float(f) => {
            let value: f64 = f.into();
            assert!((value - 42.5).abs() < 1e-10);
        }
        _ => panic!("Expected Float variant"),
    }
}

#[test]
fn test_expression_float_negative() {
    let expr = Expression::Float(MathFloat::from(-2.5));
    match expr {
        Expression::Float(f) => {
            let value: f64 = f.into();
            assert!((value + 2.5).abs() < 1e-10);
        }
        _ => panic!("Expected Float variant"),
    }
}

// Tests for Expression - Rational
#[test]
fn test_expression_rational() {
    let expr = Expression::Rational {
        numerator: Box::new(Expression::Integer(1)),
        denominator: Box::new(Expression::Integer(2)),
    };

    match expr {
        Expression::Rational {
            numerator,
            denominator,
        } => {
            assert!(matches!(*numerator, Expression::Integer(1)));
            assert!(matches!(*denominator, Expression::Integer(2)));
        }
        _ => panic!("Expected Rational variant"),
    }
}

#[test]
fn test_expression_rational_clone() {
    let expr = Expression::Rational {
        numerator: Box::new(Expression::Integer(3)),
        denominator: Box::new(Expression::Integer(4)),
    };

    let expr_clone = expr.clone();

    match expr_clone {
        Expression::Rational {
            numerator,
            denominator,
        } => {
            assert!(matches!(*numerator, Expression::Integer(3)));
            assert!(matches!(*denominator, Expression::Integer(4)));
        }
        _ => panic!("Expected Rational variant"),
    }
}

// Tests for Expression - Complex
#[test]
fn test_expression_complex() {
    let expr = Expression::Complex {
        real: Box::new(Expression::Integer(3)),
        imaginary: Box::new(Expression::Integer(4)),
    };

    match expr {
        Expression::Complex { real, imaginary } => {
            assert!(matches!(*real, Expression::Integer(3)));
            assert!(matches!(*imaginary, Expression::Integer(4)));
        }
        _ => panic!("Expected Complex variant"),
    }
}

#[test]
fn test_expression_complex_pure_imaginary() {
    let expr = Expression::Complex {
        real: Box::new(Expression::Integer(0)),
        imaginary: Box::new(Expression::Integer(1)),
    };

    match expr {
        Expression::Complex { real, imaginary } => {
            assert!(matches!(*real, Expression::Integer(0)));
            assert!(matches!(*imaginary, Expression::Integer(1)));
        }
        _ => panic!("Expected Complex variant"),
    }
}

// Tests for Expression - Variable
#[test]
fn test_expression_variable() {
    let expr = Expression::Variable("x".to_string());
    match expr {
        Expression::Variable(name) => assert_eq!(name, "x"),
        _ => panic!("Expected Variable variant"),
    }
}

#[test]
fn test_expression_variable_greek() {
    let expr = Expression::Variable("theta".to_string());
    match expr {
        Expression::Variable(name) => assert_eq!(name, "theta"),
        _ => panic!("Expected Variable variant"),
    }
}

#[test]
fn test_expression_variable_subscript() {
    let expr = Expression::Variable("x_1".to_string());
    match expr {
        Expression::Variable(name) => assert_eq!(name, "x_1"),
        _ => panic!("Expected Variable variant"),
    }
}

// Tests for Expression - Constant
#[test]
fn test_expression_constant_pi() {
    let expr = Expression::Constant(MathConstant::Pi);
    match expr {
        Expression::Constant(c) => assert_eq!(c, MathConstant::Pi),
        _ => panic!("Expected Constant variant"),
    }
}

#[test]
fn test_expression_constant_e() {
    let expr = Expression::Constant(MathConstant::E);
    match expr {
        Expression::Constant(c) => assert_eq!(c, MathConstant::E),
        _ => panic!("Expected Constant variant"),
    }
}

// Tests for Expression - Binary
#[test]
fn test_expression_binary_add() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(2)),
        right: Box::new(Expression::Integer(3)),
    };

    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert!(matches!(*left, Expression::Integer(2)));
            assert!(matches!(*right, Expression::Integer(3)));
        }
        _ => panic!("Expected Binary variant"),
    }
}

#[test]
fn test_expression_binary_nested() {
    // (2 + 3) * 4
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(3)),
        }),
        right: Box::new(Expression::Integer(4)),
    };

    match expr {
        Expression::Binary { op, left, .. } => {
            assert_eq!(op, BinaryOp::Mul);
            match *left {
                Expression::Binary { op, .. } => assert_eq!(op, BinaryOp::Add),
                _ => panic!("Expected nested Binary"),
            }
        }
        _ => panic!("Expected Binary variant"),
    }
}

// Tests for Expression - Unary
#[test]
fn test_expression_unary_neg() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Integer(5)),
    };

    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, UnaryOp::Neg);
            assert!(matches!(*operand, Expression::Integer(5)));
        }
        _ => panic!("Expected Unary variant"),
    }
}

#[test]
fn test_expression_unary_factorial() {
    let expr = Expression::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::Variable("n".to_string())),
    };

    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, UnaryOp::Factorial);
            match *operand {
                Expression::Variable(ref name) => assert_eq!(name, "n"),
                _ => panic!("Expected Variable operand"),
            }
        }
        _ => panic!("Expected Unary variant"),
    }
}

// Tests for Expression - Function
#[test]
fn test_expression_function_no_args() {
    let expr = Expression::Function {
        name: "f".to_string(),
        args: vec![],
    };

    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "f");
            assert_eq!(args.len(), 0);
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_expression_function_one_arg() {
    let expr = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };

    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Variable(v) => assert_eq!(v, "x"),
                _ => panic!("Expected Variable argument"),
            }
        }
        _ => panic!("Expected Function variant"),
    }
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

    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "max");
            assert_eq!(args.len(), 3);
        }
        _ => panic!("Expected Function variant"),
    }
}

// Tests for Expression - Derivative
#[test]
fn test_expression_derivative_first_order() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    match expr {
        Expression::Derivative { expr, var, order } => {
            assert!(matches!(*expr, Expression::Variable(_)));
            assert_eq!(var, "x");
            assert_eq!(order, 1);
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_expression_derivative_second_order() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    };

    match expr {
        Expression::Derivative { order, .. } => assert_eq!(order, 2),
        _ => panic!("Expected Derivative variant"),
    }
}

// Tests for Expression - PartialDerivative
#[test]
fn test_expression_partial_derivative() {
    let expr = Expression::PartialDerivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert!(matches!(*expr, Expression::Variable(_)));
            assert_eq!(var, "x");
            assert_eq!(order, 1);
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_expression_partial_derivative_higher_order() {
    let expr = Expression::PartialDerivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "y".to_string(),
        order: 3,
    };

    match expr {
        Expression::PartialDerivative { var, order, .. } => {
            assert_eq!(var, "y");
            assert_eq!(order, 3);
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

// Tests for Expression - Integral
#[test]
fn test_expression_integral_indefinite() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    };

    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert!(matches!(*integrand, Expression::Variable(_)));
            assert_eq!(var, "x");
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
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

    match expr {
        Expression::Integral { bounds, .. } => {
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            assert!(matches!(*bounds.lower, Expression::Integer(0)));
            assert!(matches!(*bounds.upper, Expression::Integer(1)));
        }
        _ => panic!("Expected Integral variant"),
    }
}

// Tests for Expression - Limit
#[test]
fn test_expression_limit_both_sides() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(0)),
        direction: Direction::Both,
    };

    match expr {
        Expression::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            assert!(matches!(*expr, Expression::Variable(_)));
            assert_eq!(var, "x");
            assert!(matches!(*to, Expression::Integer(0)));
            assert_eq!(direction, Direction::Both);
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_expression_limit_from_left() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(0)),
        direction: Direction::Left,
    };

    match expr {
        Expression::Limit { direction, .. } => assert_eq!(direction, Direction::Left),
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_expression_limit_to_infinity() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Constant(MathConstant::Infinity)),
        direction: Direction::Both,
    };

    match expr {
        Expression::Limit { to, .. } => {
            assert!(matches!(*to, Expression::Constant(MathConstant::Infinity)));
        }
        _ => panic!("Expected Limit variant"),
    }
}

// Tests for Expression - Sum
#[test]
fn test_expression_sum() {
    let expr = Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    };

    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert!(matches!(*lower, Expression::Integer(1)));
            assert!(matches!(*upper, Expression::Variable(_)));
            assert!(matches!(*body, Expression::Variable(_)));
        }
        _ => panic!("Expected Sum variant"),
    }
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

    match expr {
        Expression::Sum { body, .. } => {
            assert!(matches!(*body, Expression::Binary { .. }));
        }
        _ => panic!("Expected Sum variant"),
    }
}

// Tests for Expression - Product
#[test]
fn test_expression_product() {
    let expr = Expression::Product {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    };

    match expr {
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert!(matches!(*lower, Expression::Integer(1)));
            assert!(matches!(*upper, Expression::Variable(_)));
            assert!(matches!(*body, Expression::Variable(_)));
        }
        _ => panic!("Expected Product variant"),
    }
}

// Tests for Expression - Vector
#[test]
fn test_expression_vector_empty() {
    let expr = Expression::Vector(vec![]);
    match expr {
        Expression::Vector(elements) => assert_eq!(elements.len(), 0),
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_single() {
    let expr = Expression::Vector(vec![Expression::Integer(1)]);
    match expr {
        Expression::Vector(elements) => {
            assert_eq!(elements.len(), 1);
            assert!(matches!(elements[0], Expression::Integer(1)));
        }
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_multiple() {
    let expr = Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Integer(2),
        Expression::Integer(3),
    ]);

    match expr {
        Expression::Vector(elements) => {
            assert_eq!(elements.len(), 3);
            assert!(matches!(elements[0], Expression::Integer(1)));
            assert!(matches!(elements[1], Expression::Integer(2)));
            assert!(matches!(elements[2], Expression::Integer(3)));
        }
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_expression_vector_mixed_types() {
    let expr = Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Variable("x".to_string()),
        Expression::Float(MathFloat::from(2.5)),
    ]);

    match expr {
        Expression::Vector(elements) => assert_eq!(elements.len(), 3),
        _ => panic!("Expected Vector variant"),
    }
}

// Tests for Expression - Matrix
#[test]
fn test_expression_matrix_empty() {
    let expr = Expression::Matrix(vec![]);
    match expr {
        Expression::Matrix(rows) => assert_eq!(rows.len(), 0),
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_single_element() {
    let expr = Expression::Matrix(vec![vec![Expression::Integer(1)]]);

    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 1);
            assert!(matches!(rows[0][0], Expression::Integer(1)));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_2x2() {
    let expr = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(4)],
    ]);

    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
            assert_eq!(rows[1].len(), 2);
            assert!(matches!(rows[0][0], Expression::Integer(1)));
            assert!(matches!(rows[1][1], Expression::Integer(4)));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_expression_matrix_rectangular() {
    let expr = Expression::Matrix(vec![
        vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ],
        vec![
            Expression::Integer(4),
            Expression::Integer(5),
            Expression::Integer(6),
        ],
    ]);

    match expr {
        Expression::Matrix(rows) => {
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
    let expr = Expression::Equation {
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(5)),
    };

    match expr {
        Expression::Equation { left, right } => {
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Integer(5)));
        }
        _ => panic!("Expected Equation variant"),
    }
}

#[test]
fn test_expression_equation_complex() {
    // y = 2x + 1
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

    match expr {
        Expression::Equation { right, .. } => {
            assert!(matches!(*right, Expression::Binary { .. }));
        }
        _ => panic!("Expected Equation variant"),
    }
}

// Tests for Expression - Inequality
#[test]
fn test_expression_inequality_less_than() {
    let expr = Expression::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(5)),
    };

    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Lt);
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Integer(5)));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_expression_inequality_greater_equal() {
    let expr = Expression::Inequality {
        op: InequalityOp::Ge,
        left: Box::new(Expression::Variable("y".to_string())),
        right: Box::new(Expression::Integer(0)),
    };

    match expr {
        Expression::Inequality { op, .. } => assert_eq!(op, InequalityOp::Ge),
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_expression_inequality_not_equal() {
    let expr = Expression::Inequality {
        op: InequalityOp::Ne,
        left: Box::new(Expression::Variable("a".to_string())),
        right: Box::new(Expression::Variable("b".to_string())),
    };

    match expr {
        Expression::Inequality { op, .. } => assert_eq!(op, InequalityOp::Ne),
        _ => panic!("Expected Inequality variant"),
    }
}

// Test Expression::Clone
#[test]
fn test_expression_clone_deep() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(2)),
        right: Box::new(Expression::Variable("x".to_string())),
    };

    let expr_clone = expr.clone();

    match (expr, expr_clone) {
        (Expression::Binary { op: op1, .. }, Expression::Binary { op: op2, .. }) => {
            assert_eq!(op1, op2);
        }
        _ => panic!("Clone failed"),
    }
}

// Test Debug trait
#[test]
fn test_expression_debug() {
    let expr = Expression::Integer(42);
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
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    let bounds2 = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    let bounds3 = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(2)),
    };

    assert_eq!(bounds1, bounds2);
    assert_ne!(bounds1, bounds3);
}

#[test]
fn test_integral_bounds_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let bounds1 = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    let bounds2 = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };

    set.insert(bounds1);
    set.insert(bounds2); // Should be considered duplicate

    assert_eq!(set.len(), 1);
}

// Tests for Expression equality
#[test]
fn test_expression_integer_equality() {
    let e1 = Expression::Integer(42);
    let e2 = Expression::Integer(42);
    let e3 = Expression::Integer(43);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_float_equality() {
    let e1 = Expression::Float(MathFloat::from(3.14));
    let e2 = Expression::Float(MathFloat::from(3.14));
    let e3 = Expression::Float(MathFloat::from(2.71));

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_variable_equality() {
    let e1 = Expression::Variable("x".to_string());
    let e2 = Expression::Variable("x".to_string());
    let e3 = Expression::Variable("y".to_string());

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_binary_equality() {
    let e1 = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };

    let e2 = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };

    let e3 = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_nested_equality() {
    // (1 + 2) * 3
    let e1 = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Integer(3)),
    };

    let e2 = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Integer(3)),
    };

    assert_eq!(e1, e2);
}

#[test]
fn test_expression_vector_equality() {
    let e1 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(2)]);

    let e2 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(2)]);

    let e3 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(3)]);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_matrix_equality() {
    let e1 = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(4)],
    ]);

    let e2 = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(4)],
    ]);

    let e3 = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(5)],
    ]);

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

// Tests for Expression hashing and use in collections
#[test]
fn test_expression_hash_set() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(Expression::Integer(1));
    set.insert(Expression::Integer(2));
    set.insert(Expression::Integer(1)); // Duplicate

    assert_eq!(set.len(), 2);
    assert!(set.contains(&Expression::Integer(1)));
    assert!(set.contains(&Expression::Integer(2)));
}

#[test]
fn test_expression_hash_map() {
    use std::collections::HashMap;
    let mut map = HashMap::new();

    map.insert(Expression::Variable("x".to_string()), 42);
    map.insert(Expression::Variable("y".to_string()), 17);
    map.insert(Expression::Variable("x".to_string()), 99); // Update

    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&Expression::Variable("x".to_string())), Some(&99));
    assert_eq!(map.get(&Expression::Variable("y".to_string())), Some(&17));
}

#[test]
fn test_expression_complex_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    let expr1 = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };

    let expr2 = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };

    set.insert(expr1);
    set.insert(expr2); // Should be considered duplicate

    assert_eq!(set.len(), 1);
}

#[test]
fn test_expression_float_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();

    set.insert(Expression::Float(MathFloat::from(3.14)));
    set.insert(Expression::Float(MathFloat::from(2.71)));
    set.insert(Expression::Float(MathFloat::from(3.14))); // Duplicate

    assert_eq!(set.len(), 2);
}

#[test]
fn test_expression_function_equality() {
    let e1 = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };

    let e2 = Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };

    let e3 = Expression::Function {
        name: "cos".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_derivative_equality() {
    let e1 = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    let e2 = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    };

    let e3 = Expression::Derivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    };

    assert_eq!(e1, e2);
    assert_ne!(e1, e3);
}

#[test]
fn test_expression_integral_equality() {
    let e1 = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };

    let e2 = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };

    let e3 = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
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
        let expr = Expression::Integer(42);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_float() {
        let expr = Expression::Float(MathFloat::from(3.14159));
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_variable() {
        let expr = Expression::Variable("x".to_string());
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_constant() {
        let expr = Expression::Constant(MathConstant::Pi);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_binary() {
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Variable("x".to_string())),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_unary() {
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Integer(5)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_function() {
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_rational() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(1)),
            denominator: Box::new(Expression::Integer(2)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_complex() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(3)),
            imaginary: Box::new(Expression::Integer(4)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_derivative() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_partial_derivative() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_integral_indefinite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_integral_definite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_limit() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_sum() {
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_product() {
        let expr = Expression::Product {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_vector() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ]);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_matrix() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_equation() {
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_inequality() {
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let parsed: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, parsed);
    }

    #[test]
    fn test_serialize_deserialize_nested_expression() {
        // (2 + x) * 3
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Variable("x".to_string())),
            }),
            right: Box::new(Expression::Integer(3)),
        };
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
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(10)),
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
