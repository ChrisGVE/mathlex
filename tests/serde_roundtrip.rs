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
use mathlex::{ExprKind, Expression};

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
    assert_roundtrip(&ExprKind::Integer(42));
}

#[test]
fn integer_negative() {
    assert_roundtrip(&ExprKind::Integer(-1));
}

#[test]
fn integer_zero() {
    assert_roundtrip(&ExprKind::Integer(0));
}

#[test]
fn float_decimal() {
    assert_roundtrip(&ExprKind::Float(MathFloat::from(1.5)));
}

#[test]
fn float_zero() {
    assert_roundtrip(&ExprKind::Float(MathFloat::from(0.0)));
}

/// JSON does not support non-finite floats; serde_json serializes them as `null`,
/// which means INFINITY, NEG_INFINITY, and NAN cannot round-trip through JSON.
/// Consumers that need to serialize non-finite floats should use a format that
/// supports them (e.g. MessagePack, bincode) or encode them as
/// `Constant(MathConstant::Infinity)` / `Constant(MathConstant::NegInfinity)` instead.
#[test]
fn float_infinity_serializes_to_null() {
    let expr = Expression::float(MathFloat::from(f64::INFINITY));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    // serde_json encodes non-finite f64 as null — round-trip is not possible
    assert!(
        json.contains("null"),
        "Expected null in JSON for INFINITY, got: {json}"
    );
}

#[test]
fn float_neg_infinity_serializes_to_null() {
    let expr = Expression::float(MathFloat::from(f64::NEG_INFINITY));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    assert!(
        json.contains("null"),
        "Expected null in JSON for NEG_INFINITY, got: {json}"
    );
}

#[test]
fn float_nan_serializes_to_null() {
    let expr = Expression::float(MathFloat::from(f64::NAN));
    let json = serde_json::to_string(&expr).expect("serialize failed");
    assert!(
        json.contains("null"),
        "Expected null in JSON for NAN, got: {json}"
    );
}

#[test]
fn variable_simple() {
    assert_roundtrip(&ExprKind::Variable("x".to_string()));
}

#[test]
fn variable_subscript() {
    assert_roundtrip(&ExprKind::Variable("x_1".to_string()));
}

#[test]
fn constant_pi() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::Pi));
}

#[test]
fn constant_e() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::E));
}

#[test]
fn constant_i() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::I));
}

#[test]
fn constant_j() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::J));
}

#[test]
fn constant_k() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::K));
}

#[test]
fn constant_infinity() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::Infinity));
}

#[test]
fn constant_neg_infinity() {
    assert_roundtrip(&ExprKind::Constant(MathConstant::NegInfinity));
}

// ---------------------------------------------------------------------------
// Compound types
// ---------------------------------------------------------------------------

#[test]
fn rational() {
    assert_roundtrip(&ExprKind::Rational {
        numerator: Box::new(Expression::integer(1)),
        denominator: Box::new(Expression::integer(2)),
    });
}

#[test]
fn complex() {
    assert_roundtrip(&ExprKind::Complex {
        real: Box::new(Expression::integer(1)),
        imaginary: Box::new(Expression::integer(2)),
    });
}

#[test]
fn quaternion() {
    assert_roundtrip(&ExprKind::Quaternion {
        real: Box::new(Expression::integer(1)),
        i: Box::new(Expression::integer(2)),
        j: Box::new(Expression::integer(3)),
        k: Box::new(Expression::integer(4)),
    });
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[test]
fn binary_add() {
    assert_roundtrip(&ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
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
        assert_roundtrip(&ExprKind::Binary {
            op,
            left: Box::new(Expression::variable("a".to_string())),
            right: Box::new(Expression::variable("b".to_string())),
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
        assert_roundtrip(&ExprKind::Unary {
            op,
            operand: Box::new(Expression::variable("x".to_string())),
        });
    }
}

#[test]
fn function_one_arg() {
    assert_roundtrip(&ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x".to_string())],
    });
}

#[test]
fn function_two_args() {
    assert_roundtrip(&ExprKind::Function {
        name: "log".to_string(),
        args: vec![
            ExprKind::Variable("x".to_string()),
            ExprKind::Integer(2),
        ],
    });
}

#[test]
fn function_no_args() {
    assert_roundtrip(&ExprKind::Function {
        name: "f".to_string(),
        args: vec![],
    });
}

// ---------------------------------------------------------------------------
// Calculus
// ---------------------------------------------------------------------------

#[test]
fn derivative_first_order() {
    assert_roundtrip(&ExprKind::Derivative {
        expr: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        order: 1,
    });
}

#[test]
fn derivative_second_order() {
    assert_roundtrip(&ExprKind::Derivative {
        expr: Box::new(ExprKind::Function {
            name: "sin".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }),
        var: "x".to_string(),
        order: 2,
    });
}

#[test]
fn partial_derivative() {
    assert_roundtrip(&ExprKind::PartialDerivative {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        order: 1,
    });
}

#[test]
fn integral_indefinite() {
    assert_roundtrip(&ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: None,
    });
}

#[test]
fn integral_definite() {
    assert_roundtrip(&ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    });
}

#[test]
fn multiple_integral_no_bounds() {
    assert_roundtrip(&ExprKind::MultipleIntegral {
        dimension: 2,
        integrand: Box::new(Expression::variable("f".to_string())),
        bounds: None,
        vars: vec!["x".to_string(), "y".to_string()],
    });
}

#[test]
fn multiple_integral_with_bounds() {
    assert_roundtrip(&ExprKind::MultipleIntegral {
        dimension: 2,
        integrand: Box::new(Expression::variable("f".to_string())),
        bounds: Some(MultipleBounds {
            bounds: vec![
                IntegralBounds {
                    lower: Box::new(Expression::integer(0)),
                    upper: Box::new(Expression::integer(1)),
                },
                IntegralBounds {
                    lower: Box::new(Expression::integer(0)),
                    upper: Box::new(Expression::integer(2)),
                },
            ],
        }),
        vars: vec!["x".to_string(), "y".to_string()],
    });
}

#[test]
fn closed_integral_no_surface() {
    assert_roundtrip(&ExprKind::ClosedIntegral {
        dimension: 1,
        integrand: Box::new(Expression::variable("F".to_string())),
        surface: None,
        var: "r".to_string(),
    });
}

#[test]
fn closed_integral_with_surface() {
    assert_roundtrip(&ExprKind::ClosedIntegral {
        dimension: 2,
        integrand: Box::new(Expression::variable("F".to_string())),
        surface: Some("S".to_string()),
        var: "r".to_string(),
    });
}

#[test]
fn limit_both_direction() {
    assert_roundtrip(&ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Both,
    });
}

#[test]
fn limit_left_and_right() {
    for dir in [Direction::Left, Direction::Right] {
        assert_roundtrip(&ExprKind::Limit {
            expr: Box::new(Expression::variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::constant(MathConstant::Infinity)),
            direction: dir,
        });
    }
}

#[test]
fn sum() {
    assert_roundtrip(&ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n".to_string())),
        body: Box::new(Expression::variable("i".to_string())),
    });
}

#[test]
fn product() {
    assert_roundtrip(&ExprKind::Product {
        index: "k".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::integer(10)),
        body: Box::new(Expression::variable("k".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Linear Algebra
// ---------------------------------------------------------------------------

#[test]
fn vector_numeric() {
    assert_roundtrip(&ExprKind::Vector(vec![
        ExprKind::Integer(1),
        ExprKind::Integer(2),
        ExprKind::Integer(3),
    ]));
}

#[test]
fn vector_empty() {
    assert_roundtrip(&ExprKind::Vector(vec![]));
}

#[test]
fn matrix_2x2() {
    assert_roundtrip(&ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(0)],
        vec![Expression::integer(0), Expression::integer(1)],
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
        assert_roundtrip(&ExprKind::MarkedVector {
            name: "v".to_string(),
            notation,
        });
    }
}

#[test]
fn dot_product() {
    assert_roundtrip(&ExprKind::DotProduct {
        left: Box::new(Expression::variable("u".to_string())),
        right: Box::new(Expression::variable("v".to_string())),
    });
}

#[test]
fn cross_product() {
    assert_roundtrip(&ExprKind::CrossProduct {
        left: Box::new(Expression::variable("u".to_string())),
        right: Box::new(Expression::variable("v".to_string())),
    });
}

#[test]
fn outer_product() {
    assert_roundtrip(&ExprKind::OuterProduct {
        left: Box::new(Expression::variable("u".to_string())),
        right: Box::new(Expression::variable("v".to_string())),
    });
}

#[test]
fn determinant() {
    assert_roundtrip(&ExprKind::Determinant {
        matrix: Box::new(Expression::variable("A".to_string())),
    });
}

#[test]
fn trace() {
    assert_roundtrip(&ExprKind::Trace {
        matrix: Box::new(Expression::variable("A".to_string())),
    });
}

#[test]
fn rank() {
    assert_roundtrip(&ExprKind::Rank {
        matrix: Box::new(Expression::variable("A".to_string())),
    });
}

#[test]
fn conjugate_transpose() {
    assert_roundtrip(&ExprKind::ConjugateTranspose {
        matrix: Box::new(Expression::variable("A".to_string())),
    });
}

#[test]
fn matrix_inverse() {
    assert_roundtrip(&ExprKind::MatrixInverse {
        matrix: Box::new(Expression::variable("A".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Vector Calculus
// ---------------------------------------------------------------------------

#[test]
fn gradient() {
    assert_roundtrip(&ExprKind::Gradient {
        expr: Box::new(Expression::variable("f".to_string())),
    });
}

#[test]
fn divergence() {
    assert_roundtrip(&ExprKind::Divergence {
        field: Box::new(Expression::variable("F".to_string())),
    });
}

#[test]
fn curl() {
    assert_roundtrip(&ExprKind::Curl {
        field: Box::new(Expression::variable("F".to_string())),
    });
}

#[test]
fn laplacian() {
    assert_roundtrip(&ExprKind::Laplacian {
        expr: Box::new(Expression::variable("f".to_string())),
    });
}

#[test]
fn nabla() {
    assert_roundtrip(&ExprKind::Nabla);
}

// ---------------------------------------------------------------------------
// Equations and Relations
// ---------------------------------------------------------------------------

#[test]
fn equation() {
    assert_roundtrip(&ExprKind::Equation {
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::integer(5)),
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
        assert_roundtrip(&ExprKind::Inequality {
            op,
            left: Box::new(Expression::variable("x".to_string())),
            right: Box::new(Expression::integer(0)),
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
        assert_roundtrip(&ExprKind::Relation {
            op,
            left: Box::new(Expression::variable("a".to_string())),
            right: Box::new(Expression::variable("b".to_string())),
        });
    }
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

#[test]
fn for_all_no_domain() {
    assert_roundtrip(&ExprKind::ForAll {
        variable: "x".to_string(),
        domain: None,
        body: Box::new(Expression::variable("P".to_string())),
    });
}

#[test]
fn for_all_with_domain() {
    assert_roundtrip(&ExprKind::ForAll {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Real))),
        body: Box::new(Expression::variable("P".to_string())),
    });
}

#[test]
fn exists_non_unique() {
    assert_roundtrip(&ExprKind::Exists {
        variable: "x".to_string(),
        domain: None,
        body: Box::new(Expression::variable("P".to_string())),
        unique: false,
    });
}

#[test]
fn exists_unique() {
    assert_roundtrip(&ExprKind::Exists {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Integer))),
        body: Box::new(Expression::variable("Q".to_string())),
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
        assert_roundtrip(&ExprKind::Logical {
            op,
            operands: vec![
                ExprKind::Variable("P".to_string()),
                ExprKind::Variable("Q".to_string()),
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
        assert_roundtrip(&ExprKind::NumberSetExpr(set));
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
        assert_roundtrip(&ExprKind::SetOperation {
            op,
            left: Box::new(Expression::variable("A".to_string())),
            right: Box::new(Expression::variable("B".to_string())),
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
        assert_roundtrip(&ExprKind::SetRelationExpr {
            relation,
            element: Box::new(Expression::variable("x".to_string())),
            set: Box::new(Expression::number_set(NumberSet::Real)),
        });
    }
}

#[test]
fn set_builder_no_domain() {
    assert_roundtrip(&ExprKind::SetBuilder {
        variable: "x".to_string(),
        domain: None,
        predicate: Box::new(ExprKind::Inequality {
            op: InequalityOp::Gt,
            left: Box::new(Expression::variable("x".to_string())),
            right: Box::new(Expression::integer(0)),
        }),
    });
}

#[test]
fn set_builder_with_domain() {
    assert_roundtrip(&ExprKind::SetBuilder {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Real))),
        predicate: Box::new(ExprKind::Inequality {
            op: InequalityOp::Gt,
            left: Box::new(Expression::variable("x".to_string())),
            right: Box::new(Expression::integer(0)),
        }),
    });
}

#[test]
fn empty_set() {
    assert_roundtrip(&ExprKind::EmptySet);
}

#[test]
fn power_set() {
    assert_roundtrip(&ExprKind::PowerSet {
        set: Box::new(Expression::variable("A".to_string())),
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
    assert_roundtrip(&ExprKind::Tensor {
        name: "T".to_string(),
        indices: vec![idx("i", IndexType::Upper), idx("j", IndexType::Lower)],
    });
}

#[test]
fn kronecker_delta() {
    assert_roundtrip(&ExprKind::KroneckerDelta {
        indices: vec![idx("i", IndexType::Upper), idx("j", IndexType::Lower)],
    });
}

#[test]
fn levi_civita() {
    assert_roundtrip(&ExprKind::LeviCivita {
        indices: vec![
            idx("i", IndexType::Upper),
            idx("j", IndexType::Upper),
            idx("k", IndexType::Upper),
        ],
    });
}

#[test]
fn differential() {
    assert_roundtrip(&ExprKind::Differential {
        var: "x".to_string(),
    });
}

#[test]
fn wedge_product() {
    assert_roundtrip(&ExprKind::WedgeProduct {
        left: Box::new(ExprKind::Differential {
            var: "x".to_string(),
        }),
        right: Box::new(ExprKind::Differential {
            var: "y".to_string(),
        }),
    });
}

#[test]
fn function_signature() {
    assert_roundtrip(&ExprKind::FunctionSignature {
        name: "f".to_string(),
        domain: Box::new(Expression::number_set(NumberSet::Real)),
        codomain: Box::new(Expression::number_set(NumberSet::Real)),
    });
}

#[test]
fn composition() {
    assert_roundtrip(&ExprKind::Composition {
        outer: Box::new(Expression::variable("f".to_string())),
        inner: Box::new(Expression::variable("g".to_string())),
    });
}

// ---------------------------------------------------------------------------
// Nested expressions (deep round-trip)
// ---------------------------------------------------------------------------

#[test]
fn nested_expression() {
    // sin(x)² + cos(x)²
    let sin_sq = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(ExprKind::Function {
            name: "sin".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }),
        right: Box::new(Expression::integer(2)),
    };
    let cos_sq = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(ExprKind::Function {
            name: "cos".to_string(),
            args: vec![Expression::variable("x".to_string())],
        }),
        right: Box::new(Expression::integer(2)),
    };
    assert_roundtrip(&ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(sin_sq),
        right: Box::new(cos_sq),
    });
}
