//! Expression type system: `ExprKind` (variant catalog) and `Expression` (annotated node).
//!
//! For detailed variant documentation see [`docs/ast-reference.md`](../../../docs/ast-reference.md).

use super::{
    AnnotationSet, BinaryOp, Direction, InequalityOp, IntegralBounds, LogicalOp, MathConstant,
    MathFloat, MultipleBounds, NumberSet, RelationOp, SetOp, SetRelation, TensorIndex, UnaryOp,
    VectorNotation,
};

// ─── ExprKind ────────────────────────────────────────────────────────────────

/// Variant catalog for mathematical expressions.
///
/// Every variant that contains child expressions uses `Box<Expression>` or
/// `Vec<Expression>`, where `Expression` is the annotated wrapper struct.
/// Match on `expr.kind` to inspect variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", content = "value"))]
pub enum ExprKind {
    Integer(i64),
    Float(MathFloat),
    Rational {
        numerator: Box<Expression>,
        denominator: Box<Expression>,
    },
    Complex {
        real: Box<Expression>,
        imaginary: Box<Expression>,
    },
    Quaternion {
        real: Box<Expression>,
        i: Box<Expression>,
        j: Box<Expression>,
        k: Box<Expression>,
    },
    Variable(String),
    Constant(MathConstant),

    // ── Operations ───────────────────────────────────────────────────────
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expression>,
    },
    Function {
        name: String,
        args: Vec<Expression>,
    },

    // ── Calculus ─────────────────────────────────────────────────────────
    Derivative {
        expr: Box<Expression>,
        var: String,
        order: u32,
    },
    PartialDerivative {
        expr: Box<Expression>,
        var: String,
        order: u32,
    },
    Integral {
        integrand: Box<Expression>,
        var: String,
        bounds: Option<IntegralBounds>,
    },
    MultipleIntegral {
        dimension: u8,
        integrand: Box<Expression>,
        bounds: Option<MultipleBounds>,
        vars: Vec<String>,
    },
    ClosedIntegral {
        dimension: u8,
        integrand: Box<Expression>,
        surface: Option<String>,
        var: String,
    },
    Limit {
        expr: Box<Expression>,
        var: String,
        to: Box<Expression>,
        direction: Direction,
    },
    Sum {
        index: String,
        lower: Box<Expression>,
        upper: Box<Expression>,
        body: Box<Expression>,
    },
    Product {
        index: String,
        lower: Box<Expression>,
        upper: Box<Expression>,
        body: Box<Expression>,
    },

    // ── Collections ──────────────────────────────────────────────────────
    Vector(Vec<Expression>),
    Matrix(Vec<Vec<Expression>>),

    // ── Equations & Inequalities ─────────────────────────────────────────
    Equation {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    Inequality {
        op: InequalityOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    // ── Quantifiers & Logic ──────────────────────────────────────────────
    ForAll {
        variable: String,
        domain: Option<Box<Expression>>,
        body: Box<Expression>,
    },
    Exists {
        variable: String,
        domain: Option<Box<Expression>>,
        body: Box<Expression>,
        unique: bool,
    },
    Logical {
        op: LogicalOp,
        operands: Vec<Expression>,
    },

    // ── Vectors & Products ───────────────────────────────────────────────
    MarkedVector {
        name: String,
        notation: VectorNotation,
    },
    DotProduct {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    CrossProduct {
        left: Box<Expression>,
        right: Box<Expression>,
    },
    OuterProduct {
        left: Box<Expression>,
        right: Box<Expression>,
    },

    // ── Vector Calculus ──────────────────────────────────────────────────
    Gradient {
        expr: Box<Expression>,
    },
    Divergence {
        field: Box<Expression>,
    },
    Curl {
        field: Box<Expression>,
    },
    Laplacian {
        expr: Box<Expression>,
    },
    Nabla,

    // ── Linear Algebra ───────────────────────────────────────────────────
    Determinant {
        matrix: Box<Expression>,
    },
    Trace {
        matrix: Box<Expression>,
    },
    Rank {
        matrix: Box<Expression>,
    },
    ConjugateTranspose {
        matrix: Box<Expression>,
    },
    MatrixInverse {
        matrix: Box<Expression>,
    },

    // ── Set Theory ───────────────────────────────────────────────────────
    NumberSetExpr(NumberSet),
    SetOperation {
        op: SetOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    SetRelationExpr {
        relation: SetRelation,
        element: Box<Expression>,
        set: Box<Expression>,
    },
    SetBuilder {
        variable: String,
        domain: Option<Box<Expression>>,
        predicate: Box<Expression>,
    },
    EmptySet,
    PowerSet {
        set: Box<Expression>,
    },

    // ── Tensor Notation ──────────────────────────────────────────────────
    Tensor {
        name: String,
        indices: Vec<TensorIndex>,
    },
    KroneckerDelta {
        indices: Vec<TensorIndex>,
    },
    LeviCivita {
        indices: Vec<TensorIndex>,
    },

    // ── Function Theory ──────────────────────────────────────────────────
    FunctionSignature {
        name: String,
        domain: Box<Expression>,
        codomain: Box<Expression>,
    },
    Composition {
        outer: Box<Expression>,
        inner: Box<Expression>,
    },

    // ── Differential Forms ───────────────────────────────────────────────
    Differential {
        var: String,
    },
    WedgeProduct {
        left: Box<Expression>,
        right: Box<Expression>,
    },

    // ── Relations ────────────────────────────────────────────────────────
    Relation {
        op: RelationOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

// ─── Expression (annotated AST node) ─────────────────────────────────────────

/// An annotated AST node: variant (`kind`) plus optional metadata (`annotations`).
///
/// This is the primary public type. Every node in the tree carries an
/// `AnnotationSet` (empty by default). Construct via helpers like
/// `Expression::integer(42)` or `ExprKind::Binary { ... }.into()`.
/// Pattern-match on `expr.kind`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Expression {
    pub kind: ExprKind,
    pub annotations: AnnotationSet,
}

impl Expression {
    pub fn new(kind: ExprKind) -> Self {
        Self {
            kind,
            annotations: AnnotationSet::default(),
        }
    }

    pub fn with_annotations(kind: ExprKind, annotations: AnnotationSet) -> Self {
        Self { kind, annotations }
    }

    // ── Convenience constructors (no annotations) ────────────────────────

    pub fn integer(v: i64) -> Self {
        Self::new(ExprKind::Integer(v))
    }

    pub fn float(v: MathFloat) -> Self {
        Self::new(ExprKind::Float(v))
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Self::new(ExprKind::Variable(name.into()))
    }

    pub fn constant(c: MathConstant) -> Self {
        Self::new(ExprKind::Constant(c))
    }

    pub fn vector(elements: Vec<Expression>) -> Self {
        Self::new(ExprKind::Vector(elements))
    }

    pub fn matrix(rows: Vec<Vec<Expression>>) -> Self {
        Self::new(ExprKind::Matrix(rows))
    }

    pub fn number_set(s: NumberSet) -> Self {
        Self::new(ExprKind::NumberSetExpr(s))
    }

    pub fn nabla() -> Self {
        Self::new(ExprKind::Nabla)
    }

    pub fn empty_set() -> Self {
        Self::new(ExprKind::EmptySet)
    }
}

impl From<ExprKind> for Expression {
    fn from(kind: ExprKind) -> Self {
        Self::new(kind)
    }
}

// ─── Custom serde ────────────────────────────────────────────────────────────
//
// Wire format per node:
//   { "kind": "<Variant>", "value": <payload>, "annotations": {...} }
// `annotations` omitted when empty. `value` omitted for unit variants.
//
// Relies on serde_json::Value as intermediate because serde's `flatten`
// does not compose with adjacently-tagged enums.

#[cfg(feature = "serde")]
impl serde::Serialize for Expression {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeMap};

        let kind_value = serde_json::to_value(&self.kind).map_err(Error::custom)?;

        let obj = kind_value
            .as_object()
            .ok_or_else(|| Error::custom("ExprKind did not serialize to a JSON object"))?;

        let extra = if self.annotations.is_empty() { 0 } else { 1 };
        let mut map = serializer.serialize_map(Some(obj.len() + extra))?;

        for (k, v) in obj {
            map.serialize_entry(k, v)?;
        }

        if !self.annotations.is_empty() {
            map.serialize_entry("annotations", &self.annotations)?;
        }

        map.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Expression {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let mut value: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;

        let annotations = match value.as_object_mut() {
            Some(obj) => match obj.remove("annotations") {
                Some(ann) => serde_json::from_value(ann).map_err(Error::custom)?,
                None => AnnotationSet::default(),
            },
            None => AnnotationSet::default(),
        };

        let kind: ExprKind = serde_json::from_value(value).map_err(Error::custom)?;

        Ok(Expression { kind, annotations })
    }
}
