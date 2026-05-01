#![cfg(feature = "serde")]
//! Golden fixture tests for all ExprKind variants.
//!
//! Run with MATHLEX_UPDATE_GOLDEN=1 to regenerate fixtures.
//! Run without the env var to verify fixtures are stable.

#[path = "common/mod.rs"]
mod common;

use mathlex::ast::{
    BinaryOp, Direction, IndexType, InequalityOp, IntegralBounds, LogicalOp, MathConstant,
    MathFloat, MultipleBounds, NumberSet, RelationOp, SetOp, SetRelation, TensorIndex, UnaryOp,
    VectorNotation,
};
use mathlex::{ExprKind, Expression};

// ── Basic values ─────────────────────────────────────────────────────────────

#[test]
fn integer() {
    common::assert_golden(&Expression::integer(42), "integer");
}

#[test]
fn float() {
    common::assert_golden(&Expression::float(MathFloat::from(3.14)), "float");
}

#[test]
fn rational() {
    let expr: Expression = ExprKind::Rational {
        numerator: Box::new(Expression::integer(3)),
        denominator: Box::new(Expression::integer(4)),
    }
    .into();
    common::assert_golden(&expr, "rational");
}

#[test]
fn complex() {
    let expr: Expression = ExprKind::Complex {
        real: Box::new(Expression::integer(2)),
        imaginary: Box::new(Expression::integer(3)),
    }
    .into();
    common::assert_golden(&expr, "complex");
}

#[test]
fn quaternion() {
    let expr: Expression = ExprKind::Quaternion {
        real: Box::new(Expression::integer(1)),
        i: Box::new(Expression::integer(2)),
        j: Box::new(Expression::integer(3)),
        k: Box::new(Expression::integer(4)),
    }
    .into();
    common::assert_golden(&expr, "quaternion");
}

#[test]
fn variable() {
    common::assert_golden(&Expression::variable("x"), "variable");
}

#[test]
fn constant() {
    common::assert_golden(&Expression::constant(MathConstant::Pi), "constant");
}

// ── Operations ───────────────────────────────────────────────────────────────

#[test]
fn binary_add() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    common::assert_golden(&expr, "binary_add");
}

#[test]
fn unary_neg() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::variable("x")),
    }
    .into();
    common::assert_golden(&expr, "unary_neg");
}

#[test]
fn function() {
    let expr: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x")],
    }
    .into();
    common::assert_golden(&expr, "function");
}

// ── Calculus ─────────────────────────────────────────────────────────────────

#[test]
fn derivative() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x")],
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    common::assert_golden(&expr, "derivative");
}

#[test]
fn partial_derivative() {
    let expr: Expression = ExprKind::PartialDerivative {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x"), Expression::variable("y")],
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    common::assert_golden(&expr, "partial_derivative");
}

#[test]
fn integral() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x")),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    common::assert_golden(&expr, "integral");
}

#[test]
fn multiple_integral() {
    let expr: Expression = ExprKind::MultipleIntegral {
        dimension: 2,
        integrand: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x"), Expression::variable("y")],
            }
            .into(),
        ),
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
    }
    .into();
    common::assert_golden(&expr, "multiple_integral");
}

#[test]
fn closed_integral() {
    let expr: Expression = ExprKind::ClosedIntegral {
        dimension: 1,
        integrand: Box::new(Expression::variable("f")),
        surface: Some("S".to_string()),
        var: "s".to_string(),
    }
    .into();
    common::assert_golden(&expr, "closed_integral");
}

#[test]
fn limit() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x")],
            }
            .into(),
        ),
        var: "x".to_string(),
        to: Box::new(Expression::integer(0)),
        direction: Direction::Both,
    }
    .into();
    common::assert_golden(&expr, "limit");
}

#[test]
fn sum() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n")),
        body: Box::new(Expression::variable("i")),
    }
    .into();
    common::assert_golden(&expr, "sum");
}

#[test]
fn product() {
    let expr: Expression = ExprKind::Product {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n")),
        body: Box::new(Expression::variable("i")),
    }
    .into();
    common::assert_golden(&expr, "product");
}

// ── Collections ──────────────────────────────────────────────────────────────

#[test]
fn vector() {
    let expr = Expression::vector(vec![
        Expression::integer(1),
        Expression::integer(2),
        Expression::integer(3),
    ]);
    common::assert_golden(&expr, "vector");
}

#[test]
fn matrix() {
    let expr = Expression::matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ]);
    common::assert_golden(&expr, "matrix");
}

// ── Equations & Inequalities ─────────────────────────────────────────────────

#[test]
fn equation() {
    let expr: Expression = ExprKind::Equation {
        left: Box::new(Expression::variable("x")),
        right: Box::new(Expression::integer(5)),
    }
    .into();
    common::assert_golden(&expr, "equation");
}

#[test]
fn inequality() {
    let expr: Expression = ExprKind::Inequality {
        op: InequalityOp::Lt,
        left: Box::new(Expression::variable("x")),
        right: Box::new(Expression::integer(10)),
    }
    .into();
    common::assert_golden(&expr, "inequality");
}

// ── Quantifiers & Logic ──────────────────────────────────────────────────────

#[test]
fn for_all() {
    let expr: Expression = ExprKind::ForAll {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Real))),
        body: Box::new(
            ExprKind::Inequality {
                op: InequalityOp::Ge,
                left: Box::new(
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(Expression::variable("x")),
                        right: Box::new(Expression::variable("x")),
                    }
                    .into(),
                ),
                right: Box::new(Expression::integer(0)),
            }
            .into(),
        ),
    }
    .into();
    common::assert_golden(&expr, "for_all");
}

#[test]
fn exists() {
    let expr: Expression = ExprKind::Exists {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Real))),
        body: Box::new(
            ExprKind::Equation {
                left: Box::new(
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        left: Box::new(Expression::variable("x")),
                        right: Box::new(Expression::variable("x")),
                    }
                    .into(),
                ),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        unique: false,
    }
    .into();
    common::assert_golden(&expr, "exists");
}

#[test]
fn logical() {
    let expr: Expression = ExprKind::Logical {
        op: LogicalOp::And,
        operands: vec![Expression::variable("p"), Expression::variable("q")],
    }
    .into();
    common::assert_golden(&expr, "logical");
}

// ── Vectors & Products ───────────────────────────────────────────────────────

#[test]
fn marked_vector() {
    let expr: Expression = ExprKind::MarkedVector {
        name: "v".to_string(),
        notation: VectorNotation::Arrow,
    }
    .into();
    common::assert_golden(&expr, "marked_vector");
}

#[test]
fn dot_product() {
    let expr: Expression = ExprKind::DotProduct {
        left: Box::new(Expression::variable("u")),
        right: Box::new(Expression::variable("v")),
    }
    .into();
    common::assert_golden(&expr, "dot_product");
}

#[test]
fn cross_product() {
    let expr: Expression = ExprKind::CrossProduct {
        left: Box::new(Expression::variable("u")),
        right: Box::new(Expression::variable("v")),
    }
    .into();
    common::assert_golden(&expr, "cross_product");
}

#[test]
fn outer_product() {
    let expr: Expression = ExprKind::OuterProduct {
        left: Box::new(Expression::variable("u")),
        right: Box::new(Expression::variable("v")),
    }
    .into();
    common::assert_golden(&expr, "outer_product");
}

// ── Vector Calculus ──────────────────────────────────────────────────────────

#[test]
fn gradient() {
    let expr: Expression = ExprKind::Gradient {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x"), Expression::variable("y")],
            }
            .into(),
        ),
    }
    .into();
    common::assert_golden(&expr, "gradient");
}

#[test]
fn divergence() {
    let expr: Expression = ExprKind::Divergence {
        field: Box::new(Expression::variable("F")),
    }
    .into();
    common::assert_golden(&expr, "divergence");
}

#[test]
fn curl() {
    let expr: Expression = ExprKind::Curl {
        field: Box::new(Expression::variable("F")),
    }
    .into();
    common::assert_golden(&expr, "curl");
}

#[test]
fn laplacian() {
    let expr: Expression = ExprKind::Laplacian {
        expr: Box::new(Expression::variable("f")),
    }
    .into();
    common::assert_golden(&expr, "laplacian");
}

#[test]
fn nabla() {
    common::assert_golden(&Expression::nabla(), "nabla");
}

// ── Linear Algebra ───────────────────────────────────────────────────────────

#[test]
fn determinant() {
    let expr: Expression = ExprKind::Determinant {
        matrix: Box::new(Expression::matrix(vec![
            vec![Expression::integer(1), Expression::integer(2)],
            vec![Expression::integer(3), Expression::integer(4)],
        ])),
    }
    .into();
    common::assert_golden(&expr, "determinant");
}

#[test]
fn trace() {
    let expr: Expression = ExprKind::Trace {
        matrix: Box::new(Expression::variable("A")),
    }
    .into();
    common::assert_golden(&expr, "trace");
}

#[test]
fn rank() {
    let expr: Expression = ExprKind::Rank {
        matrix: Box::new(Expression::variable("A")),
    }
    .into();
    common::assert_golden(&expr, "rank");
}

#[test]
fn conjugate_transpose() {
    let expr: Expression = ExprKind::ConjugateTranspose {
        matrix: Box::new(Expression::variable("A")),
    }
    .into();
    common::assert_golden(&expr, "conjugate_transpose");
}

#[test]
fn matrix_inverse() {
    let expr: Expression = ExprKind::MatrixInverse {
        matrix: Box::new(Expression::variable("A")),
    }
    .into();
    common::assert_golden(&expr, "matrix_inverse");
}

// ── Set Theory ───────────────────────────────────────────────────────────────

#[test]
fn number_set_expr() {
    common::assert_golden(&Expression::number_set(NumberSet::Real), "number_set_expr");
}

#[test]
fn set_operation() {
    let expr: Expression = ExprKind::SetOperation {
        op: SetOp::Union,
        left: Box::new(Expression::variable("A")),
        right: Box::new(Expression::variable("B")),
    }
    .into();
    common::assert_golden(&expr, "set_operation");
}

#[test]
fn set_relation_expr() {
    let expr: Expression = ExprKind::SetRelationExpr {
        relation: SetRelation::In,
        element: Box::new(Expression::variable("x")),
        set: Box::new(Expression::number_set(NumberSet::Real)),
    }
    .into();
    common::assert_golden(&expr, "set_relation_expr");
}

#[test]
fn set_builder() {
    let expr: Expression = ExprKind::SetBuilder {
        variable: "x".to_string(),
        domain: Some(Box::new(Expression::number_set(NumberSet::Real))),
        predicate: Box::new(
            ExprKind::Inequality {
                op: InequalityOp::Gt,
                left: Box::new(Expression::variable("x")),
                right: Box::new(Expression::integer(0)),
            }
            .into(),
        ),
    }
    .into();
    common::assert_golden(&expr, "set_builder");
}

#[test]
fn empty_set() {
    common::assert_golden(&Expression::empty_set(), "empty_set");
}

#[test]
fn power_set() {
    let expr: Expression = ExprKind::PowerSet {
        set: Box::new(Expression::variable("S")),
    }
    .into();
    common::assert_golden(&expr, "power_set");
}

// ── Tensor Notation ──────────────────────────────────────────────────────────

#[test]
fn tensor() {
    let expr: Expression = ExprKind::Tensor {
        name: "T".to_string(),
        indices: vec![
            TensorIndex {
                name: "i".to_string(),
                index_type: IndexType::Upper,
            },
            TensorIndex {
                name: "j".to_string(),
                index_type: IndexType::Lower,
            },
        ],
    }
    .into();
    common::assert_golden(&expr, "tensor");
}

#[test]
fn kronecker_delta() {
    let expr: Expression = ExprKind::KroneckerDelta {
        indices: vec![
            TensorIndex {
                name: "i".to_string(),
                index_type: IndexType::Lower,
            },
            TensorIndex {
                name: "j".to_string(),
                index_type: IndexType::Lower,
            },
        ],
    }
    .into();
    common::assert_golden(&expr, "kronecker_delta");
}

#[test]
fn levi_civita() {
    let expr: Expression = ExprKind::LeviCivita {
        indices: vec![
            TensorIndex {
                name: "i".to_string(),
                index_type: IndexType::Lower,
            },
            TensorIndex {
                name: "j".to_string(),
                index_type: IndexType::Lower,
            },
            TensorIndex {
                name: "k".to_string(),
                index_type: IndexType::Lower,
            },
        ],
    }
    .into();
    common::assert_golden(&expr, "levi_civita");
}

// ── Function Theory ──────────────────────────────────────────────────────────

#[test]
fn function_signature() {
    let expr: Expression = ExprKind::FunctionSignature {
        name: "f".to_string(),
        domain: Box::new(Expression::number_set(NumberSet::Real)),
        codomain: Box::new(Expression::number_set(NumberSet::Real)),
    }
    .into();
    common::assert_golden(&expr, "function_signature");
}

#[test]
fn composition() {
    let expr: Expression = ExprKind::Composition {
        outer: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![],
            }
            .into(),
        ),
        inner: Box::new(
            ExprKind::Function {
                name: "g".to_string(),
                args: vec![],
            }
            .into(),
        ),
    }
    .into();
    common::assert_golden(&expr, "composition");
}

// ── Differential Forms ───────────────────────────────────────────────────────

#[test]
fn differential() {
    let expr: Expression = ExprKind::Differential {
        var: "x".to_string(),
    }
    .into();
    common::assert_golden(&expr, "differential");
}

#[test]
fn wedge_product() {
    let expr: Expression = ExprKind::WedgeProduct {
        left: Box::new(
            ExprKind::Differential {
                var: "x".to_string(),
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Differential {
                var: "y".to_string(),
            }
            .into(),
        ),
    }
    .into();
    common::assert_golden(&expr, "wedge_product");
}

// ── Relations ────────────────────────────────────────────────────────────────

#[test]
fn relation() {
    let expr: Expression = ExprKind::Relation {
        op: RelationOp::Approx,
        left: Box::new(Expression::variable("x")),
        right: Box::new(Expression::variable("y")),
    }
    .into();
    common::assert_golden(&expr, "relation");
}

// ── Nested / complex expressions ─────────────────────────────────────────────

#[test]
fn nested_polynomial() {
    // x^2 + 2*x + 1  (nested Binary inside Binary)
    let x_squared: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::variable("x")),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let two_x: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::variable("x")),
    }
    .into();
    let x_squared_plus_2x: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(x_squared),
        right: Box::new(two_x),
    }
    .into();
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(x_squared_plus_2x),
        right: Box::new(Expression::integer(1)),
    }
    .into();
    common::assert_golden(&expr, "nested_polynomial");
}

#[test]
fn nested_trig_identity() {
    // sin(x)^2 + cos(x)^2  (Function inside Binary inside Binary)
    let sin_x: Expression = ExprKind::Function {
        name: "sin".to_string(),
        args: vec![Expression::variable("x")],
    }
    .into();
    let cos_x: Expression = ExprKind::Function {
        name: "cos".to_string(),
        args: vec![Expression::variable("x")],
    }
    .into();
    let sin_sq: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(sin_x),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let cos_sq: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(cos_x),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(sin_sq),
        right: Box::new(cos_sq),
    }
    .into();
    common::assert_golden(&expr, "nested_trig_identity");
}

#[test]
fn nested_derivative_order2() {
    // Second-order derivative of f with respect to x
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(
            ExprKind::Function {
                name: "f".to_string(),
                args: vec![Expression::variable("x")],
            }
            .into(),
        ),
        var: "x".to_string(),
        order: 2,
    }
    .into();
    common::assert_golden(&expr, "nested_derivative_order2");
}

#[test]
fn nested_integral_definite() {
    // Definite integral from 0 to 1 of x dx
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x")),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    common::assert_golden(&expr, "nested_integral_definite");
}

#[test]
fn nested_matrix_2x2() {
    // [[a, b], [c, d]]  — symbolic 2×2 matrix
    let expr = Expression::matrix(vec![
        vec![Expression::variable("a"), Expression::variable("b")],
        vec![Expression::variable("c"), Expression::variable("d")],
    ]);
    common::assert_golden(&expr, "nested_matrix_2x2");
}

#[test]
fn nested_sum_product() {
    // sum from i=1 to n of i^2
    let i_squared: Expression = ExprKind::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::variable("i")),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::variable("n")),
        body: Box::new(i_squared),
    }
    .into();
    common::assert_golden(&expr, "nested_sum_product");
}

#[test]
fn nested_with_annotation() {
    // Variable "x" annotated with unit = "meters"
    use mathlex::ast::AnnotationSet;
    let mut ann = AnnotationSet::default();
    ann.insert("unit".to_string(), "meters".to_string());
    let expr = Expression::with_annotations(ExprKind::Variable("x".to_string()), ann);
    common::assert_golden(&expr, "nested_with_annotation");
}
