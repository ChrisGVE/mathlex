#![cfg(feature = "serde")]
//! Serde JSON round-trip tests for all Expression variants.
//!
//! Each test verifies that serialize→deserialize produces an equivalent
//! expression. NaN behavior is documented separately since IEEE 754 NaN
//! is not equal to itself, but MathFloat treats NaN == NaN.

use mathlex::ast::{
    BinaryOp, Direction, IndexType, InequalityOp, IntegralBounds, LogicalOp, MathConstant,
    MathFloat, MultipleBounds, NumberSet, RelationOp, SetOp, SetRelation, TensorIndex, UnaryOp,
    VectorNotation,
};
use mathlex::Expression;

fn roundtrip(expr: &Expression) -> Expression {
    let json = serde_json::to_string(expr).expect("serialize failed");
    serde_json::from_str(&json).expect("deserialize failed")
}

fn assert_roundtrip(expr: &Expression) {
    let restored = roundtrip(expr);
    assert_eq!(
        expr, &restored,
        "Round-trip failed.\nOriginal: {:?}\nRestored: {:?}",
        expr, restored
    );
}

// ---------------------------------------------------------------------------
// Basic values
// ---------------------------------------------------------------------------

#[test]
fn integer_positive() {
    assert_roundtrip(&Expression::Integer(42));
}

#[test]
fn integer_negative() {
    assert_roundtrip(&Expression::Integer(-1));
}

#[test]
fn integer_zero() {
    assert_roundtrip(&Expression::Integer(0));
}

#[test]
fn float_decimal() {
    assert_roundtrip(&Expression::Float(MathFloat::from(1.5)));
}

#[test]
fn float_zero() {
    assert_roundtrip(&Expression::Float(MathFloat::from(0.0)));
}

/// JSON does not support non-finite floats; serde_json serializes them as `null`,
/// which means INFINITY, NEG_INFINITY, and NAN cannot round-trip through JSON.
/// Consumers that need to serialize non-finite floats should use a format that
/// supports them (e.g. MessagePack, bincode) or encode them as
/// `Constant(MathConstant::Infinity)` / `Constant(MathConstant::NegInfinity)` instead.
#[test]
fn float_infinity_serializes_to_null() {
    let expr = Expression::Float(MathFloat::from(f64::INFINITY));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    // serde_json encodes non-finite f64 as null — round-trip is not possible
    assert!(
        json.contains("null"),
        "Expected null in JSON for INFINITY, got: {json}"
    );
}

#[test]
fn float_neg_infinity_serializes_to_null() {
    let expr = Expression::Float(MathFloat::from(f64::NEG_INFINITY));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    assert!(
        json.contains("null"),
        "Expected null in JSON for NEG_INFINITY, got: {json}"
    );
}

#[test]
fn float_nan_serializes_to_null() {
    let expr = Expression::Float(MathFloat::from(f64::NAN));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    assert!(
        json.contains("null"),
        "Expected null in JSON for NAN, got: {json}"
    );
}

#[test]
fn variable_simple() {
    assert_roundtrip(&Expression::Variable("x".to_string()));
}

#[test]
fn variable_subscript() {
    assert_roundtrip(&Expression::Variable("x_1".to_string()));
}

#[test]
fn constant_pi() {
    assert_roundtrip(&Expression::Constant(MathConstant::Pi));
}

#[test]
fn constant_e() {
    assert_roundtrip(&Expression::Constant(MathConstant::E));
}

#[test]
fn constant_i() {
    assert_roundtrip(&Expression::Constant(MathConstant::I));
}

#[test]
fn constant_j() {
    assert_roundtrip(&Expression::Constant(MathConstant::J));
}

#[test]
fn constant_k() {
    assert_roundtrip(&Expression::Constant(MathConstant::K));
}

#[test]
fn constant_infinity() {
    assert_roundtrip(&Expression::Constant(MathConstant::Infinity));
}

#[test]
fn constant_neg_infinity() {
    assert_roundtrip(&Expression::Constant(MathConstant::NegInfinity));
}

// ---------------------------------------------------------------------------
// Compound types
// ---------------------------------------------------------------------------

#[test]
fn rational() {
    assert_roundtrip(&Expression::Rational {
        numerator: Box::new(Expression::Integer(1)),
        denominator: Box::new(Expression::Integer(2)),
    });
}

#[test]
fn complex() {
    assert_roundtrip(&Expression::Complex {
        real: Box::new(Expression::Integer(1)),
        imaginary: Box::new(Expression::Integer(2)),
    });
}

#[test]
fn quaternion() {
    assert_roundtrip(&Expression::Quaternion {
        real: Box::new(Expression::Integer(1)),
        i: Box::new(Expression::Integer(2)),
        j: Box::new(Expression::Integer(3)),
        k: Box::new(Expression::Integer(4)),
    });
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[test]
fn binary_add() {
    assert_roundtrip(&Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    });
}

#[test]
fn binary_all_ops() {
    let ops = [
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Mod,
        BinaryOp::PlusMinus,
        BinaryOp::MinusPlus,
    ];
    for op in ops {
        assert_roundtrip(&Expression::Binary {
            op,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        });
    }
}

#[test]
fn unary_all_ops() {
    let ops = [
        UnaryOp::Neg,
        UnaryOp::Pos,
        UnaryOp::Factorial,
        UnaryOp::Transpose,
    ];
    for op in ops {
        assert_roundtrip(&Expression::Unary {
            op,
            operand: Box::new(Expression::Variable("x".to_string())),
        });
    }
}

#[test]
fn function_one_arg() {
    assert_roundtrip(&Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    });
}

#[test]
fn function_two_args() {
    assert_roundtrip(&Expression::Function {
        name: "log".to_string(),
        args: vec![
            Expression::Variable("x".to_string()),
            Expression::Integer(2),
        ],
    });
}

#[test]
fn function_no_args() {
    assert_roundtrip(&Expression::Function {
        name: "f".to_string(),
        args: vec![],
    });
}

// ---------------------------------------------------------------------------
// Calculus
// ---------------------------------------------------------------------------

#[test]
fn derivative_first_order() {
    assert_roundtrip(&Expression::Derivative {
        expr: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        order: 1,
    });
}

#[test]
fn derivative_second_order() {
    assert_roundtrip(&Expression::Derivative {
        expr: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        var: "x".to_string(),
        order: 2,
    });
}

#[test]
fn partial_derivative() {
    assert_roundtrip(&Expression::PartialDerivative {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    });
}

#[test]
fn integral_indefinite() {
    assert_roundtrip(&Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    });
}

#[test]
fn integral_definite() {
    assert_roundtrip(&Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    });
}

#[test]
fn multiple_integral_no_bounds() {
    assert_roundtrip(&Expression::MultipleIntegral {
        dimension: 2,
        integrand: Box::new(Expression::Variable("f".to_string())),
        bounds: None,
        vars: vec!["x".to_string(), "y".to_string()],
    });
}

#[test]
fn multiple_integral_with_bounds() {
    assert_roundtrip(&Expression::MultipleIntegral {
        dimension: 2,
        integrand: Box::new(Expression::Variable("f".to_string())),
        bounds: Some(MultipleBounds {
            bounds: vec![
                IntegralBounds {
                    lower: Box::new(Expression::Integer(0)),
                    upper: Box::new(Expression::Integer(1)),
                },
                IntegralBounds {
                    lower: Box::new(Expression::Integer(0)),
                    upper: Box::new(Expression::Integer(2)),
                },
            ],
        }),
        vars: vec!["x".to_string(), "y".to_string()],
    });
}

#[test]
fn closed_integral_no_surface() {
    assert_roundtrip(&Expression::ClosedIntegral {
        dimension: 1,
        integrand: Box::new(Expression::Variable("F".to_string())),
        surface: None,
        var: "r".to_string(),
    });
}

#[test]
fn closed_integral_with_surface() {
    assert_roundtrip(&Expression::ClosedIntegral {
        dimension: 2,
        integrand: Box::new(Expression::Variable("F".to_string())),
        surface: Some("S".to_string()),
        var: "r".to_string(),
    });
}

#[test]
fn limit_both_direction() {
    assert_roundtrip(&Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(0)),
        direction: Direction::Both,
    });
}

#[test]
fn limit_left_and_right() {
    for dir in [Direction::Left, Direction::Right] {
        assert_roundtrip(&Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: dir,
        });
    }
}

#[test]
fn sum() {
    assert_roundtrip(&Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Variable("n".to_string())),
        body: Box::new(Expression::Variable("i".to_string())),
    });
}

#[test]
fn product() {
    assert_roundtrip(&Expression::Product {
        index: "k".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Integer(10)),
        body: Box::new(Expression::Variable("k".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Linear Algebra
// ---------------------------------------------------------------------------

#[test]
fn vector_numeric() {
    assert_roundtrip(&Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Integer(2),
        Expression::Integer(3),
    ]));
}

#[test]
fn vector_empty() {
    assert_roundtrip(&Expression::Vector(vec![]));
}

#[test]
fn matrix_2x2() {
    assert_roundtrip(&Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(0)],
        vec![Expression::Integer(0), Expression::Integer(1)],
    ]));
}

#[test]
fn marked_vector_all_notations() {
    for notation in [
        VectorNotation::Bold,
        VectorNotation::Arrow,
        VectorNotation::Hat,
        VectorNotation::Underline,
        VectorNotation::Plain,
    ] {
        assert_roundtrip(&Expression::MarkedVector {
            name: "v".to_string(),
            notation,
        });
    }
}

#[test]
fn dot_product() {
    assert_roundtrip(&Expression::DotProduct {
        left: Box::new(Expression::Variable("u".to_string())),
        right: Box::new(Expression::Variable("v".to_string())),
    });
}

#[test]
fn cross_product() {
    assert_roundtrip(&Expression::CrossProduct {
        left: Box::new(Expression::Variable("u".to_string())),
        right: Box::new(Expression::Variable("v".to_string())),
    });
}

#[test]
fn outer_product() {
    assert_roundtrip(&Expression::OuterProduct {
        left: Box::new(Expression::Variable("u".to_string())),
        right: Box::new(Expression::Variable("v".to_string())),
    });
}

#[test]
fn determinant() {
    assert_roundtrip(&Expression::Determinant {
        matrix: Box::new(Expression::Variable("A".to_string())),
    });
}

#[test]
fn trace() {
    assert_roundtrip(&Expression::Trace {
        matrix: Box::new(Expression::Variable("A".to_string())),
    });
}

#[test]
fn rank() {
    assert_roundtrip(&Expression::Rank {
        matrix: Box::new(Expression::Variable("A".to_string())),
    });
}

#[test]
fn conjugate_transpose() {
    assert_roundtrip(&Expression::ConjugateTranspose {
        matrix: Box::new(Expression::Variable("A".to_string())),
    });
}

#[test]
fn matrix_inverse() {
    assert_roundtrip(&Expression::MatrixInverse {
        matrix: Box::new(Expression::Variable("A".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Vector Calculus
// ---------------------------------------------------------------------------

#[test]
fn gradient() {
    assert_roundtrip(&Expression::Gradient {
        expr: Box::new(Expression::Variable("f".to_string())),
    });
}

#[test]
fn divergence() {
    assert_roundtrip(&Expression::Divergence {
        field: Box::new(Expression::Variable("F".to_string())),
    });
}

#[test]
fn curl() {
    assert_roundtrip(&Expression::Curl {
        field: Box::new(Expression::Variable("F".to_string())),
    });
}

#[test]
fn laplacian() {
    assert_roundtrip(&Expression::Laplacian {
        expr: Box::new(Expression::Variable("f".to_string())),
    });
}

#[test]
fn nabla() {
    assert_roundtrip(&Expression::Nabla);
}

// ---------------------------------------------------------------------------
// Equations and Relations
// ---------------------------------------------------------------------------

#[test]
fn equation() {
    assert_roundtrip(&Expression::Equation {
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(5)),
    });
}

#[test]
fn inequality_all_ops() {
    let ops = [
        InequalityOp::Lt,
        InequalityOp::Le,
        InequalityOp::Gt,
        InequalityOp::Ge,
        InequalityOp::Ne,
    ];
    for op in ops {
        assert_roundtrip(&Expression::Inequality {
            op,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(0)),
        });
    }
}

#[test]
fn relation_all_ops() {
    let ops = [
        RelationOp::Similar,
        RelationOp::Equivalent,
        RelationOp::Congruent,
        RelationOp::Approx,
    ];
    for op in ops {
        assert_roundtrip(&Expression::Relation {
            op,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        });
    }
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

#[test]
fn for_all_no_domain() {
    assert_roundtrip(&Expression::ForAll {
        variable: "x".to_string(),
        domain: None,
        body: Box::new(Expression::Variable("P".to_string())),
    });
}

#[test]
fn for_all_with_domain() {
    assert_roundtrip(&Expression::ForAll {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::NumberSetExpr(NumberSet::Real))),
        body: Box::new(Expression::Variable("P".to_string())),
    });
}

#[test]
fn exists_non_unique() {
    assert_roundtrip(&Expression::Exists {
        variable: "x".to_string(),
        domain: None,
        body: Box::new(Expression::Variable("P".to_string())),
        unique: false,
    });
}

#[test]
fn exists_unique() {
    assert_roundtrip(&Expression::Exists {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::NumberSetExpr(NumberSet::Integer))),
        body: Box::new(Expression::Variable("Q".to_string())),
        unique: true,
    });
}

#[test]
fn logical_all_ops() {
    let ops = [
        LogicalOp::And,
        LogicalOp::Or,
        LogicalOp::Not,
        LogicalOp::Implies,
        LogicalOp::Iff,
    ];
    for op in ops {
        assert_roundtrip(&Expression::Logical {
            op,
            operands: vec![
                Expression::Variable("P".to_string()),
                Expression::Variable("Q".to_string()),
            ],
        });
    }
}

// ---------------------------------------------------------------------------
// Sets
// ---------------------------------------------------------------------------

#[test]
fn number_set_all_variants() {
    let sets = [
        NumberSet::Natural,
        NumberSet::Integer,
        NumberSet::Rational,
        NumberSet::Real,
        NumberSet::Complex,
        NumberSet::Quaternion,
    ];
    for set in sets {
        assert_roundtrip(&Expression::NumberSetExpr(set));
    }
}

#[test]
fn set_operation_all_ops() {
    let ops = [
        SetOp::Union,
        SetOp::Intersection,
        SetOp::Difference,
        SetOp::SymmetricDiff,
        SetOp::CartesianProd,
    ];
    for op in ops {
        assert_roundtrip(&Expression::SetOperation {
            op,
            left: Box::new(Expression::Variable("A".to_string())),
            right: Box::new(Expression::Variable("B".to_string())),
        });
    }
}

#[test]
fn set_relation_all_variants() {
    let rels = [
        SetRelation::In,
        SetRelation::NotIn,
        SetRelation::Subset,
        SetRelation::SubsetEq,
        SetRelation::Superset,
        SetRelation::SupersetEq,
    ];
    for relation in rels {
        assert_roundtrip(&Expression::SetRelationExpr {
            relation,
            element: Box::new(Expression::Variable("x".to_string())),
            set: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
        });
    }
}

#[test]
fn set_builder_no_domain() {
    assert_roundtrip(&Expression::SetBuilder {
        variable: "x".to_string(),
        domain: None,
        predicate: Box::new(Expression::Inequality {
            op: InequalityOp::Gt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(0)),
        }),
    });
}

#[test]
fn set_builder_with_domain() {
    assert_roundtrip(&Expression::SetBuilder {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::NumberSetExpr(NumberSet::Real))),
        predicate: Box::new(Expression::Inequality {
            op: InequalityOp::Gt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(0)),
        }),
    });
}

#[test]
fn empty_set() {
    assert_roundtrip(&Expression::EmptySet);
}

#[test]
fn power_set() {
    assert_roundtrip(&Expression::PowerSet {
        set: Box::new(Expression::Variable("A".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Tensors and Differential Forms
// ---------------------------------------------------------------------------

fn idx(name: &str, index_type: IndexType) -> TensorIndex {
    TensorIndex {
        name: name.to_string(),
        index_type,
    }
}

#[test]
fn tensor_mixed_indices() {
    assert_roundtrip(&Expression::Tensor {
        name: "T".to_string(),
        indices: vec![idx("i", IndexType::Upper), idx("j", IndexType::Lower)],
    });
}

#[test]
fn kronecker_delta() {
    assert_roundtrip(&Expression::KroneckerDelta {
        indices: vec![idx("i", IndexType::Upper), idx("j", IndexType::Lower)],
    });
}

#[test]
fn levi_civita() {
    assert_roundtrip(&Expression::LeviCivita {
        indices: vec![
            idx("i", IndexType::Upper),
            idx("j", IndexType::Upper),
            idx("k", IndexType::Upper),
        ],
    });
}

#[test]
fn differential() {
    assert_roundtrip(&Expression::Differential {
        var: "x".to_string(),
    });
}

#[test]
fn wedge_product() {
    assert_roundtrip(&Expression::WedgeProduct {
        left: Box::new(Expression::Differential {
            var: "x".to_string(),
        }),
        right: Box::new(Expression::Differential {
            var: "y".to_string(),
        }),
    });
}

#[test]
fn function_signature() {
    assert_roundtrip(&Expression::FunctionSignature {
        name: "f".to_string(),
        domain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
        codomain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
    });
}

#[test]
fn composition() {
    assert_roundtrip(&Expression::Composition {
        outer: Box::new(Expression::Variable("f".to_string())),
        inner: Box::new(Expression::Variable("g".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Nested expressions (deep round-trip)
// ---------------------------------------------------------------------------

#[test]
fn nested_expression() {
    // sin(x)² + cos(x)²
    let sin_sq = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        right: Box::new(Expression::Integer(2)),
    };
    let cos_sq = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        }),
        right: Box::new(Expression::Integer(2)),
    };
    assert_roundtrip(&Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(sin_sq),
        right: Box::new(cos_sq),
    });
}
