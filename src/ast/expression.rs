//! The main Expression AST node type.
//!
//! For detailed variant documentation, examples, and usage notes see
//! [`docs/ast-reference.md`](../../../docs/ast-reference.md).

use super::{
    BinaryOp, Direction, InequalityOp, IntegralBounds, LogicalOp, MathConstant, MathFloat,
    MultipleBounds, NumberSet, RelationOp, SetOp, SetRelation, TensorIndex, UnaryOp,
    VectorNotation,
};

/// The main AST node type representing any mathematical expression.
///
/// Covers scalar values, arithmetic, functions, calculus, linear algebra, set theory,
/// tensor notation, differential forms, and relational expressions.
/// See [`docs/ast-reference.md`](../../../docs/ast-reference.md) for full variant documentation.
///
/// # Example
///
/// ```
/// use mathlex::ast::{Expression, BinaryOp, MathConstant};
///
/// let expr = Expression::Binary {
///     op: BinaryOp::Mul,
///     left: Box::new(Expression::Integer(2)),
///     right: Box::new(Expression::Constant(MathConstant::Pi)),
/// };
/// match expr {
///     Expression::Binary { op: BinaryOp::Mul, .. } => println!("2π"),
///     _ => panic!("unexpected"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expression {
    /// Integer literal (e.g., `42`, `-17`, `0`).
    Integer(i64),

    /// Floating-point literal (e.g., `3.14`, `-2.5`, `1.0e-10`).
    Float(MathFloat),

    /// Rational number as a ratio of two expressions.
    ///
    /// Note: not produced by parsers; available for programmatic construction.
    Rational {
        /// Numerator of the fraction (any expression)
        numerator: Box<Expression>,

        /// Denominator of the fraction (any expression)
        denominator: Box<Expression>,
    },

    /// Complex number in the form `a + bi`.
    ///
    /// Note: not produced by parsers; available for programmatic construction.
    Complex {
        /// Real component (any expression)
        real: Box<Expression>,

        /// Imaginary component - coefficient of i (any expression)
        imaginary: Box<Expression>,
    },

    /// Quaternion in canonical form `a + bi + cj + dk`.
    ///
    /// Note: not produced by parsers; available for programmatic construction.
    Quaternion {
        /// Real (scalar) component
        real: Box<Expression>,

        /// Coefficient of i basis vector
        i: Box<Expression>,

        /// Coefficient of j basis vector
        j: Box<Expression>,

        /// Coefficient of k basis vector
        k: Box<Expression>,
    },

    /// Variable identifier (e.g., `x`, `theta`, `x_1`).
    Variable(String),

    /// Mathematical constant (e.g., π, e, i, ∞).
    Constant(MathConstant),

    /// Binary operation with two operands (e.g., `x + y`, `2 * π`, `a^b`).
    Binary {
        /// The binary operator
        op: BinaryOp,

        /// Left operand
        left: Box<Expression>,

        /// Right operand
        right: Box<Expression>,
    },

    /// Unary operation with a single operand (e.g., `-x`, `n!`, `A'`).
    Unary {
        /// The unary operator
        op: UnaryOp,

        /// The operand
        operand: Box<Expression>,
    },

    /// Function application with zero or more arguments (e.g., `sin(x)`, `max(a, b)`).
    Function {
        /// Function name
        name: String,

        /// Function arguments (may be empty)
        args: Vec<Expression>,
    },

    /// Ordinary derivative of an expression with respect to a variable: `dⁿ/dxⁿ f`.
    Derivative {
        /// The expression being differentiated
        expr: Box<Expression>,

        /// The variable to differentiate with respect to
        var: String,

        /// Order of differentiation (1 for first derivative, 2 for second, etc.)
        order: u32,
    },

    /// Partial derivative of a multivariable expression: `∂ⁿf/∂xⁿ`.
    PartialDerivative {
        /// The expression being partially differentiated
        expr: Box<Expression>,

        /// The variable to partially differentiate with respect to
        var: String,

        /// Order of partial differentiation (1 for first, 2 for second, etc.)
        order: u32,
    },

    /// Definite or indefinite integral: `∫ f(x) dx` or `∫ₐᵇ f(x) dx`.
    Integral {
        /// The integrand (expression being integrated)
        integrand: Box<Expression>,

        /// The variable of integration
        var: String,

        /// Integration bounds (None for indefinite integral, Some for definite)
        bounds: Option<IntegralBounds>,
    },

    /// Multiple integral (double, triple, etc.): `∬ f dA`, `∭ f dV`.
    MultipleIntegral {
        /// Number of integral signs (2=double, 3=triple)
        dimension: u8,

        /// The integrand expression
        integrand: Box<Expression>,

        /// Optional bounds for each variable
        bounds: Option<MultipleBounds>,

        /// Variables of integration in order
        vars: Vec<String>,
    },

    /// Closed/contour integral: `∮` (line), `∯` (surface), `∰` (volume).
    ClosedIntegral {
        /// Dimension: 1=line (∮), 2=surface (∯), 3=volume (∰)
        dimension: u8,

        /// The integrand expression
        integrand: Box<Expression>,

        /// Optional surface/curve name (e.g., "S", "C")
        surface: Option<String>,

        /// Variable of integration
        var: String,
    },

    /// Limit of an expression as a variable approaches a value: `lim_{x→a} f(x)`.
    Limit {
        /// The expression whose limit is being taken
        expr: Box<Expression>,

        /// The variable approaching the limit
        var: String,

        /// The value being approached (can be finite, infinite, or symbolic)
        to: Box<Expression>,

        /// Direction of approach (left, right, or both sides)
        direction: Direction,
    },

    /// Summation over a range using sigma notation: `Σ_{index=lower}^{upper} body`.
    Sum {
        /// The index variable (bound variable in the summation)
        index: String,

        /// Lower bound of summation (inclusive)
        lower: Box<Expression>,

        /// Upper bound of summation (inclusive)
        upper: Box<Expression>,

        /// The expression being summed (can reference index variable)
        body: Box<Expression>,
    },

    /// Product over a range using pi notation: `Π_{index=lower}^{upper} body`.
    Product {
        /// The index variable (bound variable in the product)
        index: String,

        /// Lower bound of product (inclusive)
        lower: Box<Expression>,

        /// Upper bound of product (inclusive)
        upper: Box<Expression>,

        /// The expression being multiplied (can reference index variable)
        body: Box<Expression>,
    },

    /// Ordered collection of expressions as a mathematical vector (e.g., `[1, 2, 3]`).
    Vector(Vec<Expression>),

    /// 2D array of expressions organized in rows and columns.
    ///
    /// Parsers always produce rectangular matrices. All rows should have the same length.
    Matrix(Vec<Vec<Expression>>),

    /// Equality between two expressions (e.g., `x = 5`, `f(x) = x²`).
    Equation {
        /// Left-hand side of the equation
        left: Box<Expression>,

        /// Right-hand side of the equation
        right: Box<Expression>,
    },

    /// Inequality comparison between two expressions (e.g., `x < 5`, `y ≥ 0`).
    Inequality {
        /// The inequality operator
        op: InequalityOp,

        /// Left-hand side of the inequality
        left: Box<Expression>,

        /// Right-hand side of the inequality
        right: Box<Expression>,
    },

    /// Universal quantifier: `∀x ∈ S, P(x)`.
    ForAll {
        /// The bound variable
        variable: String,
        /// Optional domain restriction
        domain: Option<Box<Expression>>,
        /// The body expression
        body: Box<Expression>,
    },

    /// Existential quantifier: `∃x ∈ S, P(x)` or `∃!x` for unique existence.
    Exists {
        /// The bound variable
        variable: String,
        /// Optional domain restriction
        domain: Option<Box<Expression>>,
        /// The body expression
        body: Box<Expression>,
        /// Whether this is unique existence (∃!)
        unique: bool,
    },

    /// Logical expression with a binary or unary logical operator.
    Logical {
        /// The logical operator
        op: LogicalOp,
        /// The operands
        operands: Vec<Expression>,
    },

    /// Vector variable with an explicit visual notation style (bold, arrow, hat, underline).
    MarkedVector {
        /// The vector name
        name: String,
        /// The notation style
        notation: VectorNotation,
    },

    /// Dot product (inner product) of two vectors: `u · v`.
    DotProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    /// Cross product of two vectors: `u × v`.
    CrossProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    /// Outer product (tensor product) of two vectors: `u ⊗ v`.
    OuterProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    // ============================================================
    // Vector Calculus Expressions
    // ============================================================
    /// Gradient of a scalar field: `∇f`.
    Gradient {
        /// The scalar field expression
        expr: Box<Expression>,
    },

    /// Divergence of a vector field: `∇·F`.
    Divergence {
        /// The vector field expression
        field: Box<Expression>,
    },

    /// Curl of a vector field: `∇×F`.
    Curl {
        /// The vector field expression
        field: Box<Expression>,
    },

    /// Laplacian of a scalar field: `∇²f` or `Δf`.
    Laplacian {
        /// The scalar field expression
        expr: Box<Expression>,
    },

    /// Raw nabla/del operator: `∇` (without an operand).
    Nabla,

    // ============================================================
    // Linear Algebra Operations
    // ============================================================
    /// Determinant of a matrix: `det(A)` or `|A|`.
    Determinant {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Trace of a matrix (sum of diagonal elements): `tr(A)`.
    Trace {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Rank of a matrix (dimension of column/row space): `rank(A)`.
    Rank {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Conjugate transpose (Hermitian adjoint): `A†`, `A*`, or `A^H`.
    ConjugateTranspose {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Matrix inverse: `A⁻¹`.
    MatrixInverse {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    // ============================================================
    // Set Theory Expressions
    // ============================================================
    /// A standard number set: ℕ, ℤ, ℚ, ℝ, ℂ, or ℍ.
    NumberSetExpr(NumberSet),

    /// Binary set operation: union (∪), intersection (∩), difference (∖), etc.
    SetOperation {
        /// The set operation
        op: SetOp,
        /// Left operand set
        left: Box<Expression>,
        /// Right operand set
        right: Box<Expression>,
    },

    /// Set membership or subset relation: `x ∈ S`, `A ⊆ B`.
    SetRelationExpr {
        /// The relation type
        relation: SetRelation,
        /// The element (or left set for subset relations)
        element: Box<Expression>,
        /// The set (or right set for subset relations)
        set: Box<Expression>,
    },

    /// Set builder notation: `{x | P(x)}` or `{x ∈ S | P(x)}`.
    SetBuilder {
        /// The bound variable
        variable: String,
        /// Optional domain set
        domain: Option<Box<Expression>>,
        /// The predicate that defines membership
        predicate: Box<Expression>,
    },

    /// The empty set: `∅` or `{}`.
    EmptySet,

    /// Power set of S (the set of all subsets): `𝒫(S)`.
    PowerSet {
        /// The set to take the power set of
        set: Box<Expression>,
    },

    // ============================================================
    // Tensor Notation Expressions
    // ============================================================
    /// Tensor with upper and/or lower indices (Einstein summation convention).
    ///
    /// For example: `T^{ij}`, `T_{ab}`, `T^i_j`.
    Tensor {
        /// The tensor name (e.g., "T", "g", "R", "Γ")
        name: String,

        /// The tensor indices in order of appearance
        indices: Vec<TensorIndex>,
    },

    /// Kronecker delta: `δ^i_j` — equals 1 when indices match, 0 otherwise.
    KroneckerDelta {
        /// The indices (typically two, one upper and one lower)
        indices: Vec<TensorIndex>,
    },

    /// Levi-Civita totally antisymmetric symbol: `ε^{ijk}` or `ε_{ijk}`.
    LeviCivita {
        /// The indices (typically 3 for 3D, n for nD)
        indices: Vec<TensorIndex>,
    },

    // ============================================================
    // Function Theory and Relations
    // ============================================================
    /// Function signature/mapping declaration: `f: A → B`.
    FunctionSignature {
        /// The function name
        name: String,

        /// The domain (input type/set)
        domain: Box<Expression>,

        /// The codomain (output type/set)
        codomain: Box<Expression>,
    },

    /// Function composition: `f ∘ g`, where `(f ∘ g)(x) = f(g(x))`.
    Composition {
        /// The outer function (applied second)
        outer: Box<Expression>,

        /// The inner function (applied first)
        inner: Box<Expression>,
    },

    // ============================================================
    // Differential Forms
    // ============================================================
    /// Differential of a variable: `dx`, `dy`, `dt`.
    ///
    /// Represents a differential 1-form; distinct from derivative notation `d/dx`.
    Differential {
        /// The variable name (without the 'd' prefix)
        var: String,
    },

    /// Wedge product (exterior product) of two differential forms: `dx ∧ dy`.
    ///
    /// Anticommutative: `dx ∧ dy = -(dy ∧ dx)`.
    WedgeProduct {
        /// Left operand (typically a differential or wedge product)
        left: Box<Expression>,
        /// Right operand (typically a differential)
        right: Box<Expression>,
    },

    /// Mathematical relation: similarity (`~`), equivalence (`≡`), congruence (`≅`), approximation (`≈`).
    Relation {
        /// The relation operator
        op: RelationOp,

        /// Left operand
        left: Box<Expression>,

        /// Right operand
        right: Box<Expression>,
    },
}
