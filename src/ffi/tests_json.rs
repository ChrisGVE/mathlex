#![cfg(all(feature = "ffi", feature = "serde"))]

use super::*;
use crate::ast::{BinaryOp, Direction, IntegralBounds, MathConstant, NumberSet, SetOp, UnaryOp};
use crate::Expression;

// ----------------------------------------------------------------
// Basic value types
// ----------------------------------------------------------------

#[test]
fn test_json_integer() {
    let expr = parse_text("42").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert_eq!(json, r#"{"Integer":42}"#);
}

#[test]
fn test_json_float() {
    let expr = parse_text("3.14").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("Float"),
        "expected Float variant, got: {json}"
    );
}

#[test]
fn test_json_variable() {
    let expr = parse_text("x").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert_eq!(json, r#"{"Variable":"x"}"#);
}

// ----------------------------------------------------------------
// All MathConstant variants
// ----------------------------------------------------------------

#[test]
fn test_json_constant_pi() {
    let expr = parse_text("pi").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Constant"), "expected Constant, got: {json}");
    assert!(json.contains("Pi"), "expected Pi, got: {json}");
}

#[test]
fn test_json_constant_e() {
    let expr = Expression::Constant(MathConstant::E);
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Constant"), "expected Constant, got: {json}");
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "E");
}

#[test]
fn test_json_constant_i() {
    let expr = Expression::Constant(MathConstant::I);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "I");
}

#[test]
fn test_json_constant_j() {
    let expr = Expression::Constant(MathConstant::J);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "J");
}

#[test]
fn test_json_constant_k() {
    let expr = Expression::Constant(MathConstant::K);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "K");
}

#[test]
fn test_json_constant_infinity() {
    let expr = Expression::Constant(MathConstant::Infinity);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "Infinity");
}

#[test]
fn test_json_constant_neg_infinity() {
    let expr = Expression::Constant(MathConstant::NegInfinity);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "NegInfinity");
}

#[test]
fn test_json_constant_nan() {
    let expr = Expression::Constant(MathConstant::NaN);
    let json = expression_to_json(&expr).unwrap();
    let val: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(val["Constant"], "NaN");
}

// ----------------------------------------------------------------
// All BinaryOp variants
// ----------------------------------------------------------------

fn assert_binary_op(input: &str, expected_op: &str) {
    let expr = parse_text(input).unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Binary"), "expected Binary, got: {json}");
    assert!(
        json.contains(expected_op),
        "expected op {expected_op}, got: {json}"
    );
}

#[test]
fn test_json_binary_add() {
    assert_binary_op("x + y", "Add");
}

#[test]
fn test_json_binary_sub() {
    assert_binary_op("x - y", "Sub");
}

#[test]
fn test_json_binary_mul() {
    assert_binary_op("x * y", "Mul");
}

#[test]
fn test_json_binary_div() {
    assert_binary_op("x / y", "Div");
}

#[test]
fn test_json_binary_pow() {
    assert_binary_op("x^2", "Pow");
}

#[test]
fn test_json_binary_mod() {
    assert_binary_op("x % y", "Mod");
}

#[test]
fn test_json_binary_plus_minus() {
    let expr = Expression::Binary {
        op: BinaryOp::PlusMinus,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(1)),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("PlusMinus"),
        "expected PlusMinus, got: {json}"
    );
}

// ----------------------------------------------------------------
// Unary operators
// ----------------------------------------------------------------

#[test]
fn test_json_unary_neg() {
    let expr = parse_text("-x").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("Neg") || json.contains("Unary"),
        "expected negation, got: {json}"
    );
}

#[test]
fn test_json_unary_factorial() {
    let expr = Expression::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::Variable("n".to_string())),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("Factorial"),
        "expected Factorial, got: {json}"
    );
}

// ----------------------------------------------------------------
// Function call
// ----------------------------------------------------------------

#[test]
fn test_json_function_call() {
    let expr = parse_text("sin(x)").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Function"), "expected Function, got: {json}");
    assert!(json.contains("sin"), "expected sin, got: {json}");
}

#[test]
fn test_json_function_multi_arg() {
    let expr = parse_text("max(a, b)").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Function"), "expected Function, got: {json}");
    assert!(json.contains("max"), "expected max, got: {json}");
}

// ----------------------------------------------------------------
// Calculus nodes (constructed directly to avoid parser syntax coupling)
// ----------------------------------------------------------------

#[test]
fn test_json_derivative_round_trip() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(2)),
        }),
        var: "x".to_string(),
        order: 1,
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("Derivative"),
        "expected Derivative, got: {json}"
    );
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_partial_derivative_round_trip() {
    let expr = Expression::PartialDerivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 2,
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("PartialDerivative"),
        "expected PartialDerivative, got: {json}"
    );
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_integral_indefinite_round_trip() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Integral"), "expected Integral, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_integral_definite_round_trip() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(2)),
        }),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("bounds"),
        "expected bounds field, got: {json}"
    );
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_limit_round_trip() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            right: Box::new(Expression::Variable("x".to_string())),
        }),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(0)),
        direction: Direction::Both,
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Limit"), "expected Limit, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_sum_round_trip() {
    let expr = Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Sum"), "expected Sum, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_product_round_trip() {
    let expr = Expression::Product {
        index: "k".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("k".to_string())),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Product"), "expected Product, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Linear algebra
// ----------------------------------------------------------------

#[test]
fn test_json_vector_round_trip() {
    let expr = Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Integer(2),
        Expression::Integer(3),
    ]);
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Vector"), "expected Vector, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_matrix_round_trip() {
    let expr = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(0)],
        vec![Expression::Integer(0), Expression::Integer(1)],
    ]);
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Matrix"), "expected Matrix, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Set operations
// ----------------------------------------------------------------

#[test]
fn test_json_set_operation_round_trip() {
    let expr = Expression::SetOperation {
        op: SetOp::Union,
        left: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
        right: Box::new(Expression::NumberSetExpr(NumberSet::Integer)),
    };
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("SetOperation"),
        "expected SetOperation, got: {json}"
    );
    assert!(json.contains("Union"), "expected Union, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_empty_set_round_trip() {
    let expr = Expression::EmptySet;
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("EmptySet"), "expected EmptySet, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Equation and Inequality
// ----------------------------------------------------------------

#[test]
fn test_json_equation_round_trip() {
    let expr = parse_text("x = 5").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(json.contains("Equation"), "expected Equation, got: {json}");
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_inequality_round_trip() {
    let expr = parse_text("x < 5").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("Inequality"),
        "expected Inequality, got: {json}"
    );
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Nested / complex expressions
// ----------------------------------------------------------------

#[test]
fn test_json_nested_expression() {
    let expr = parse_text("sin(x)^2 + cos(x)^2").unwrap();
    let json = expression_to_json(&expr).unwrap();
    assert!(
        json.contains("sin"),
        "expected sin in nested expr, got: {json}"
    );
    assert!(
        json.contains("cos"),
        "expected cos in nested expr, got: {json}"
    );
}

#[test]
fn test_json_deeply_nested_round_trip() {
    // Build a deeply nested expression: ((x + 1) * (y - 2)) ^ 3
    let expr = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Integer(1)),
            }),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::Variable("y".to_string())),
                right: Box::new(Expression::Integer(2)),
            }),
        }),
        right: Box::new(Expression::Integer(3)),
    };
    let json = expression_to_json(&expr).unwrap();
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Pretty-print output
// ----------------------------------------------------------------

#[test]
fn test_json_pretty_is_multiline() {
    let expr = parse_text("x + y").unwrap();
    let pretty = expression_to_json_pretty(&expr).unwrap();
    assert!(pretty.contains('\n'), "pretty JSON should be multi-line");
}

#[test]
fn test_json_pretty_contains_same_data() {
    let expr = parse_text("x + y").unwrap();
    let compact = expression_to_json(&expr).unwrap();
    let pretty = expression_to_json_pretty(&expr).unwrap();
    let compact_val: serde_json::Value = serde_json::from_str(&compact).unwrap();
    let pretty_val: serde_json::Value = serde_json::from_str(&pretty).unwrap();
    assert_eq!(compact_val, pretty_val);
}

// ----------------------------------------------------------------
// Round-trip tests
// ----------------------------------------------------------------

#[test]
fn test_json_round_trip() {
    let expr = parse_text("2 * x + 3").unwrap();
    let json = expression_to_json(&expr).unwrap();
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_round_trip_function() {
    let expr = parse_text("sin(x)").unwrap();
    let json = expression_to_json(&expr).unwrap();
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

#[test]
fn test_json_round_trip_nested() {
    let expr = parse_text("sin(x)^2 + cos(x)^2").unwrap();
    let json = expression_to_json(&expr).unwrap();
    let restored: Expression = serde_json::from_str(&json).unwrap();
    assert_eq!(expr, restored);
}

// ----------------------------------------------------------------
// Error cases
// ----------------------------------------------------------------

#[test]
fn test_json_deserialize_invalid_json_returns_error() {
    let result: Result<Expression, _> = serde_json::from_str("not valid json");
    assert!(result.is_err(), "deserializing invalid JSON should fail");
}

#[test]
fn test_json_deserialize_unknown_variant_returns_error() {
    let result: Result<Expression, _> = serde_json::from_str(r#"{"UnknownVariant": 42}"#);
    assert!(result.is_err(), "deserializing unknown variant should fail");
}
