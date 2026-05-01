# AST Reference

Detailed documentation for all `Expression` enum variants in mathlex.

For the type definitions and a concise variant summary see
[`src/ast/expression.rs`](../src/ast/expression.rs) and
[`src/ast/mod.rs`](../src/ast/mod.rs).

---

## Table of Contents

- [Scalar Values](#scalar-values)
- [Operations](#operations)
- [Calculus](#calculus)
- [Integral Variants](#integral-variants)
- [Linear Algebra](#linear-algebra)
- [Sets and Logic](#sets-and-logic)
- [Relations](#relations)
- [Tensor and Differential Geometry](#tensor-and-differential-geometry)
- [Function Theory](#function-theory)
- [Differential Forms](#differential-forms)

---

## Scalar Values

### `Integer(i64)`

Represents whole-number literals, both positive and negative.

**Examples:** `42`, `-17`, `0`

---

### `Float(MathFloat)`

Represents decimal number literals. `MathFloat` wraps `OrderedFloat<f64>` to
provide `Hash` and `Eq` implementations. NaN values compare equal to each
other (differs from IEEE 754, required for use in hash collections).

**Examples:** `3.14`, `-2.5`, `1.0e-10`

**Serialization note:** When using JSON (via serde), NaN and Infinity values
serialize to `null`. Use binary formats such as bincode for lossless
serialization of special floats.

---

### `Rational { numerator, denominator }`

Represents a ratio of two arbitrary expressions.

**Important:** Fields are `Expression`, not `i64`. This allows symbolic
rationals such as `x/y` or `(a+b)/(c+d)`, not just numeric fractions.

**Not produced by parsers.** Current parsers represent divisions as
`Binary { op: Div, ... }`. This variant is available for programmatic
construction, typically by symbolic manipulation libraries that want a
simplified rational form.

```rust
use mathlex::ast::Expression;

// Numeric rational: 1/2
let half = Expression::Rational {
    numerator: Box::new(Expression::Integer(1)),
    denominator: Box::new(Expression::Integer(2)),
};

// Symbolic rational: x/y
let symbolic = Expression::Rational {
    numerator: Box::new(Expression::Variable("x".to_string())),
    denominator: Box::new(Expression::Variable("y".to_string())),
};
```

---

### `Complex { real, imaginary }`

Represents a complex number in the form `a + bi`.

**Important:** Fields are `Expression`, not numeric types. This allows
symbolic complex numbers such as `(x+y) + (z+w)i`.

**Not produced by parsers.** Parsers represent complex expressions via
`Binary` operations with the imaginary constant `i`. This variant is for
programmatic construction.

```rust
use mathlex::ast::Expression;

// Numeric: 3 + 4i
let complex = Expression::Complex {
    real: Box::new(Expression::Integer(3)),
    imaginary: Box::new(Expression::Integer(4)),
};

// Pure imaginary: 0 + i
let pure_imaginary = Expression::Complex {
    real: Box::new(Expression::Integer(0)),
    imaginary: Box::new(Expression::Integer(1)),
};
```

---

### `Quaternion { real, i, j, k }`

Represents a quaternion in canonical form `a + bi + cj + dk` using the
standard basis {1, i, j, k}.

**Not produced by parsers.** Available for programmatic construction.

**Quaternion algebra rules:**
- `i² = j² = k² = ijk = -1`
- `ij = k`, `jk = i`, `ki = j`
- `ji = -k`, `kj = -i`, `ik = -j`

```rust
use mathlex::ast::Expression;

// 1 + 2i + 3j + 4k
let quat = Expression::Quaternion {
    real: Box::new(Expression::Integer(1)),
    i: Box::new(Expression::Integer(2)),
    j: Box::new(Expression::Integer(3)),
    k: Box::new(Expression::Integer(4)),
};
```

---

### `Variable(String)`

Represents a symbolic variable name.

**Examples:** `x`, `theta`, `x_1`

Libraries that need additional variable metadata (type, dimension, units)
should maintain a separate metadata map rather than extending this variant:

```rust,ignore
let metadata: HashMap<String, VariableMetadata> = HashMap::new();
for var in expr.find_variables() {
    if let Some(meta) = metadata.get(&var) {
        // use metadata for dimensional analysis, etc.
    }
}
```

---

### `Constant(MathConstant)`

Represents a well-known mathematical constant.

**Values:** π (`Pi`), e (`E`), i (`ImaginaryUnit`), ∞ (`Infinity`),
-∞ (`NegInfinity`)

**Note:** `MathConstant::NegInfinity` is produced by parsers when unary minus
is applied to infinity. Both `-∞` (plain text) and `-\infty` (LaTeX) parse
directly as `Constant(NegInfinity)`.

---

## Operations

### `Binary { op, left, right }`

Represents an infix operation with two operands.

**Operators (`BinaryOp`):** `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Mod`

**Examples:** `x + y`, `2 * π`, `a^b`

---

### `Unary { op, operand }`

Represents a prefix or postfix operation with a single operand.

**Operators (`UnaryOp`):** `Neg` (negation), `Factorial`, `Transpose`

**Examples:** `-x`, `n!`, `A'`

---

### `Function { name, args }`

Represents a named function applied to zero or more argument expressions.

**Examples:** `sin(x)`, `max(a, b, c)`, `f()`

---

## Calculus

### `Derivative { expr, var, order }`

Represents the nth ordinary derivative of an expression with respect to a
single variable.

- `order` must be ≥ 1.
- `var` is the differentiation variable.
- `expr` is the expression being differentiated.

**Notation:**
- First derivative: `d/dx f(x)` or `f'(x)` or `df/dx`
- Second derivative: `d²/dx² f(x)` or `f''(x)`
- nth derivative: `dⁿ/dxⁿ f(x)`

```rust
use mathlex::ast::Expression;

// d/dx(x²)
let first_deriv = Expression::Derivative {
    expr: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Pow,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(2)),
    }),
    var: "x".to_string(),
    order: 1,
};

// d²/dx²(sin(x))
let second_deriv = Expression::Derivative {
    expr: Box::new(Expression::Function {
        name: "sin".to_string(),
        args: vec![Expression::Variable("x".to_string())],
    }),
    var: "x".to_string(),
    order: 2,
};
```

---

### `PartialDerivative { expr, var, order }`

Represents the nth partial derivative of a multivariable expression with
respect to one variable, holding others constant.

**Notation:**
- `∂f/∂x`, `∂/∂x f(x,y,z)`
- `∂²f/∂x²`, `∂ⁿf/∂xⁿ`

**When to use `PartialDerivative` vs `Derivative`:**
- Use `PartialDerivative` for functions of multiple variables, or when
  emphasising that other variables are held constant.
- Use `Derivative` for single-variable calculus.

```rust
use mathlex::ast::Expression;

// ∂/∂x(x²y)
let partial = Expression::PartialDerivative {
    expr: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: mathlex::ast::BinaryOp::Pow,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Variable("y".to_string())),
    }),
    var: "x".to_string(),
    order: 1,
};
```

---

## Integral Variants

### `Integral { integrand, var, bounds }`

Represents both definite and indefinite integrals.

- `bounds = None` → indefinite integral `∫ f(x) dx`
- `bounds = Some(IntegralBounds)` → definite integral `∫ₐᵇ f(x) dx`

Bounds can be numeric, symbolic, infinite (`Constant(Infinity)`), or
complex expressions.

```rust
use mathlex::ast::{Expression, IntegralBounds};

// Indefinite integral: ∫ x dx
let indefinite = Expression::Integral {
    integrand: Box::new(Expression::Variable("x".to_string())),
    var: "x".to_string(),
    bounds: None,
};

// Definite integral: ∫₀¹ x² dx
let definite = Expression::Integral {
    integrand: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Pow,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Integer(2)),
    }),
    var: "x".to_string(),
    bounds: Some(IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    }),
};
```

---

### `MultipleIntegral { dimension, integrand, bounds, vars }`

Represents double, triple, or higher-dimensional integrals.

- `dimension = 2` → double integral (∬), typically over an area
- `dimension = 3` → triple integral (∭), typically over a volume
- `dimension > 3` → higher-dimensional integrals

```rust,ignore
// ∬_R f(x,y) dy dx
Expression::MultipleIntegral {
    dimension: 2,
    integrand: Box::new(f_expr),
    bounds: None,
    vars: vec!["y".to_string(), "x".to_string()],
}
```

---

### `ClosedIntegral { dimension, integrand, surface, var }`

Represents closed path integrals.

- `dimension = 1` → line integral over a closed curve (∮)
- `dimension = 2` → surface integral over a closed surface (∯)
- `dimension = 3` → volume integral over a closed volume (∰)

```rust,ignore
// ∮_C F · dr
Expression::ClosedIntegral {
    dimension: 1,
    integrand: Box::new(f_dot_dr),
    surface: Some("C".to_string()),
    var: "r".to_string(),
}
```

---

### `Limit { expr, var, to, direction }`

Represents the limit of an expression as a variable approaches a value.

**Direction variants:**
- `Direction::Both` — two-sided limit: `lim_{x→a} f(x)`
- `Direction::Left` — left-hand limit: `lim_{x→a⁻} f(x)` (approach from below)
- `Direction::Right` — right-hand limit: `lim_{x→a⁺} f(x)` (approach from above)

The `to` field can be finite, symbolic, `Constant(Infinity)`, or
`Constant(NegInfinity)`.

```rust
use mathlex::ast::{Expression, Direction, MathConstant};

// lim_{x→0} sin(x)/x
let limit_both = Expression::Limit {
    expr: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Div,
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

// lim_{x→∞} 1/x
let limit_infinity = Expression::Limit {
    expr: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Div,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Variable("x".to_string())),
    }),
    var: "x".to_string(),
    to: Box::new(Expression::Constant(MathConstant::Infinity)),
    direction: Direction::Both,
};

// lim_{x→0⁺} 1/x
let limit_right = Expression::Limit {
    expr: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Div,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Variable("x".to_string())),
    }),
    var: "x".to_string(),
    to: Box::new(Expression::Integer(0)),
    direction: Direction::Right,
};
```

---

### `Sum { index, lower, upper, body }`

Represents a summation using sigma notation: `Σ_{index=lower}^{upper} body`.

Evaluates to: `body[index=lower] + body[index=lower+1] + ... + body[index=upper]`

The `index` variable is bound within `body` and takes each integer value from
`lower` to `upper` inclusive.

Bounds can be numeric, symbolic, or infinite.

```rust
use mathlex::ast::Expression;

// Σ_{i=1}^{n} i
let sum = Expression::Sum {
    index: "i".to_string(),
    lower: Box::new(Expression::Integer(1)),
    upper: Box::new(Expression::Variable("n".to_string())),
    body: Box::new(Expression::Variable("i".to_string())),
};

// Σ_{k=1}^{10} k²
let sum_squares = Expression::Sum {
    index: "k".to_string(),
    lower: Box::new(Expression::Integer(1)),
    upper: Box::new(Expression::Integer(10)),
    body: Box::new(Expression::Binary {
        op: mathlex::ast::BinaryOp::Pow,
        left: Box::new(Expression::Variable("k".to_string())),
        right: Box::new(Expression::Integer(2)),
    }),
};
```

---

### `Product { index, lower, upper, body }`

Represents a product using pi notation: `Π_{index=lower}^{upper} body`.

Evaluates to: `body[index=lower] * body[index=lower+1] * ... * body[index=upper]`

```rust
use mathlex::ast::Expression;

// Π_{i=1}^{n} i  (factorial-like)
let factorial = Expression::Product {
    index: "i".to_string(),
    lower: Box::new(Expression::Integer(1)),
    upper: Box::new(Expression::Variable("n".to_string())),
    body: Box::new(Expression::Variable("i".to_string())),
};
```

---

## Linear Algebra

### `Vector(Vec<Expression>)`

An ordered collection of expressions as a mathematical vector.

- Elements can be any expression type.
- Dimension is determined by the number of elements.
- An empty `Vec` represents a zero-dimensional vector.

```rust
use mathlex::ast::Expression;

// [1, 2, 3]
let position = Expression::Vector(vec![
    Expression::Integer(1),
    Expression::Integer(2),
    Expression::Integer(3),
]);

// [x, y, z]
let symbolic = Expression::Vector(vec![
    Expression::Variable("x".to_string()),
    Expression::Variable("y".to_string()),
    Expression::Variable("z".to_string()),
]);
```

---

### `Matrix(Vec<Vec<Expression>>)`

A 2D array of expressions in rows and columns.

- Dimensions are M×N where M is rows and N is columns.
- Parsers always produce rectangular matrices.
- The AST does not enforce uniform row lengths. Use
  `Expression::is_valid_matrix()` or `Expression::matrix_dimensions()` to
  validate manually-constructed matrices.

Special cases:
- Empty: `[]` → 0×0
- Row vector: `[[1, 2, 3]]` → 1×3
- Column vector: `[[1], [2], [3]]` → 3×1
- Scalar: `[[x]]` → 1×1

```rust
use mathlex::ast::Expression;

// 2×2 identity matrix
let identity = Expression::Matrix(vec![
    vec![Expression::Integer(1), Expression::Integer(0)],
    vec![Expression::Integer(0), Expression::Integer(1)],
]);
```

---

### `MarkedVector { name, notation }`

A vector variable with an explicit visual notation style.

**Notations (`VectorNotation`):** `Bold` (`\mathbf{v}`), `Arrow` (`\vec{a}`),
`Hat` (`\hat{n}`), `Underline`, `Plain`

---

### `DotProduct { left, right }`

Dot product (inner product) of two vectors: `u · v` or `\mathbf{a} \cdot \mathbf{b}`.

---

### `CrossProduct { left, right }`

Cross product of two 3D vectors: `u × v` or `\mathbf{a} \times \mathbf{b}`.

---

### `OuterProduct { left, right }`

Outer product (tensor product): `u ⊗ v` or `\mathbf{a} \otimes \mathbf{b}`.

---

### `Gradient { expr }`

Gradient of a scalar field: `∇f`.

Points in the direction of greatest increase of the scalar field.

```rust
use mathlex::ast::Expression;

let grad = Expression::Gradient {
    expr: Box::new(Expression::Variable("f".to_string())),
};
```

---

### `Divergence { field }`

Divergence of a vector field: `∇·F`.

Scalar field measuring the "outflow" of a vector field at each point.

```rust
use mathlex::ast::Expression;

let div = Expression::Divergence {
    field: Box::new(Expression::Variable("F".to_string())),
};
```

---

### `Curl { field }`

Curl of a vector field: `∇×F`.

Vector field measuring the rotation of a vector field at each point.

```rust
use mathlex::ast::Expression;

let curl = Expression::Curl {
    field: Box::new(Expression::Variable("F".to_string())),
};
```

---

### `Laplacian { expr }`

Laplacian of a scalar field: `∇²f` or `Δf`.

Equals the divergence of the gradient: `∇·(∇f)`.

```rust
use mathlex::ast::Expression;

let laplacian = Expression::Laplacian {
    expr: Box::new(Expression::Variable("f".to_string())),
};
```

---

### `Nabla`

The raw nabla/del operator `∇` without an operand. Used when nabla appears
in non-standard combinations or without an immediately following operand.

---

### `Determinant { matrix }`

Determinant of a matrix: `det(A)` or `|A|`.

Returns the signed volume scaling factor of the linear transformation.

```rust
use mathlex::ast::Expression;

let det = Expression::Determinant {
    matrix: Box::new(Expression::Variable("A".to_string())),
};
```

---

### `Trace { matrix }`

Trace of a matrix (sum of diagonal elements): `tr(A)`.

```rust
use mathlex::ast::Expression;

let trace = Expression::Trace {
    matrix: Box::new(Expression::Variable("A".to_string())),
};
```

---

### `Rank { matrix }`

Rank of a matrix (dimension of column space / row space): `rank(A)`.

```rust
use mathlex::ast::Expression;

let rank = Expression::Rank {
    matrix: Box::new(Expression::Variable("A".to_string())),
};
```

---

### `ConjugateTranspose { matrix }`

Conjugate transpose (Hermitian adjoint): `A†`, `A*`, or `A^H`.

Transpose of the complex conjugate. For real matrices, equals the transpose.

```rust
use mathlex::ast::Expression;

let adjoint = Expression::ConjugateTranspose {
    matrix: Box::new(Expression::Variable("A".to_string())),
};
```

---

### `MatrixInverse { matrix }`

Matrix inverse: `A⁻¹`.

The matrix that when multiplied by A gives the identity. Only exists for
square matrices with non-zero determinant.

```rust
use mathlex::ast::Expression;

let inverse = Expression::MatrixInverse {
    matrix: Box::new(Expression::Variable("A".to_string())),
};
```

---

## Sets and Logic

### `NumberSetExpr(NumberSet)`

A standard number set: `ℕ` (Natural), `ℤ` (Integer), `ℚ` (Rational),
`ℝ` (Real), `ℂ` (Complex), `ℍ` (Quaternion).

```rust,ignore
Expression::NumberSetExpr(NumberSet::Real)
```

---

### `SetOperation { op, left, right }`

Binary set operation.

**Operators (`SetOp`):** `Union` (∪), `Intersection` (∩), `Difference` (∖),
`SymmetricDifference` (△), `CartesianProduct` (×)

```rust,ignore
// A ∪ B
Expression::SetOperation {
    op: SetOp::Union,
    left: Box::new(a),
    right: Box::new(b),
}
```

---

### `SetRelationExpr { relation, element, set }`

Set membership or subset relation.

**Relations (`SetRelation`):** `In` (∈), `NotIn` (∉), `Subset` (⊆),
`ProperSubset` (⊊), `Superset` (⊇), `ProperSuperset` (⊋)

```rust,ignore
// x ∈ ℝ
Expression::SetRelationExpr {
    relation: SetRelation::In,
    element: Box::new(Expression::Variable("x".to_string())),
    set: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
}
```

---

### `SetBuilder { variable, domain, predicate }`

Set builder notation: `{x | P(x)}` or `{x ∈ S | P(x)}`.

```rust,ignore
// {x ∈ ℝ | x > 0}
Expression::SetBuilder {
    variable: "x".to_string(),
    domain: Some(Box::new(Expression::NumberSetExpr(NumberSet::Real))),
    predicate: Box::new(x_greater_than_zero),
}
```

---

### `EmptySet`

The empty set: `∅` or `{}`.

---

### `PowerSet { set }`

Power set of S (the set of all subsets): `𝒫(S)`.

```rust,ignore
// 𝒫(A)
Expression::PowerSet {
    set: Box::new(Expression::Variable("A".to_string())),
}
```

---

### `ForAll { variable, domain, body }`

Universal quantifier: `∀x ∈ S, P(x)`.

`domain` is optional; when absent represents `∀x, P(x)`.

---

### `Exists { variable, domain, body, unique }`

Existential quantifier: `∃x ∈ S, P(x)`.

Set `unique = true` for unique existence: `∃!x`.

---

### `Logical { op, operands }`

Logical expression combining operands with a logical operator.

**Operators (`LogicalOp`):** `And` (∧), `Or` (∨), `Not` (¬),
`Implies` (→), `Iff` (↔)

---

## Relations

### `Equation { left, right }`

An equality between two expressions: `x = 5`, `f(x) = x²`.

This type has no identifier field. Libraries that need to track equations
through processing pipelines should wrap it in their own struct:

```rust,ignore
struct TrackedEquation {
    id: String,
    equation: mathlex::Expression,
}
```

---

### `Inequality { op, left, right }`

An inequality comparison: `x < 5`, `y ≥ 0`, `a ≠ b`.

**Operators (`InequalityOp`):** `Lt` (<), `Le` (≤), `Gt` (>), `Ge` (≥),
`Ne` (≠)

---

### `Relation { op, left, right }`

A mathematical relation expressing similarity, equivalence, congruence, or
approximation.

**Operators (`RelationOp`):**
- `Similar` (`~`): `a \sim b`
- `Equivalent` (`≡`): `a \equiv b`
- `Congruent` (`≅`): `a \cong b`
- `Approx` (`≈`): `a \approx b`

```rust
use mathlex::ast::{Expression, RelationOp};

// x ~ y
let similar = Expression::Relation {
    op: RelationOp::Similar,
    left: Box::new(Expression::Variable("x".to_string())),
    right: Box::new(Expression::Variable("y".to_string())),
};

// a ≈ b
let approx = Expression::Relation {
    op: RelationOp::Approx,
    left: Box::new(Expression::Variable("a".to_string())),
    right: Box::new(Expression::Variable("b".to_string())),
};
```

---

## Tensor and Differential Geometry

### `Tensor { name, indices }`

A tensor with upper and/or lower indices, supporting Einstein summation
convention.

**Notation examples:**
- `T^{ij}` — two upper indices
- `T_{ab}` — two lower indices
- `T^i_j` — mixed (one upper, one lower)
- `R^a_{bcd}` — Riemann-like tensor

**Einstein summation convention:** When the same index appears once upper and
once lower in a product, summation is implied: `A^i B_i = Σ_i A^i B_i`.

```rust
use mathlex::ast::{Expression, TensorIndex, IndexType};

// g^{μν}
let metric = Expression::Tensor {
    name: "g".to_string(),
    indices: vec![
        TensorIndex { name: "μ".to_string(), index_type: IndexType::Upper },
        TensorIndex { name: "ν".to_string(), index_type: IndexType::Upper },
    ],
};

// T^i_j
let mixed = Expression::Tensor {
    name: "T".to_string(),
    indices: vec![
        TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
        TensorIndex { name: "j".to_string(), index_type: IndexType::Lower },
    ],
};
```

---

### `KroneckerDelta { indices }`

The Kronecker delta: `δ^i_j` or `δ_{ij}`.

- `δ^i_j = 1` if `i = j`, `0` otherwise
- `δ^i_j A^j = A^i` (index substitution)
- `δ^i_i = n` (trace in n dimensions)

```rust
use mathlex::ast::{Expression, TensorIndex, IndexType};

// δ^i_j
let delta = Expression::KroneckerDelta {
    indices: vec![
        TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
        TensorIndex { name: "j".to_string(), index_type: IndexType::Lower },
    ],
};
```

---

### `LeviCivita { indices }`

The Levi-Civita totally antisymmetric symbol: `ε^{ijk}` or `ε_{ijk}`.

**Properties:**
- `ε^{123} = 1` in 3D (even permutation)
- Changes sign under any index swap (antisymmetric)
- `ε^{ijk} = 0` if any two indices are equal

**Common uses:**
- Cross product: `(a × b)^i = ε^{ijk} a_j b_k`
- Determinant: `det(A) = ε^{i₁...iₙ} A_{1i₁}...A_{niₙ}`
- Exterior algebra and differential forms

```rust
use mathlex::ast::{Expression, TensorIndex, IndexType};

// ε^{ijk}
let epsilon = Expression::LeviCivita {
    indices: vec![
        TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
        TensorIndex { name: "j".to_string(), index_type: IndexType::Upper },
        TensorIndex { name: "k".to_string(), index_type: IndexType::Upper },
    ],
};
```

---

## Function Theory

### `FunctionSignature { name, domain, codomain }`

Function signature/mapping declaration: `f: A → B`.

- LaTeX: `f: A \to B`
- Plain text: `f: A → B`

```rust
use mathlex::ast::{Expression, NumberSet};

// f: ℝ → ℝ
let real_func = Expression::FunctionSignature {
    name: "f".to_string(),
    domain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
    codomain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
};
```

---

### `Composition { outer, inner }`

Function composition: `f ∘ g`, where `(f ∘ g)(x) = f(g(x))`.

The `inner` function is applied first, then `outer`.

- LaTeX: `f \circ g`
- Unicode: `f ∘ g`

```rust
use mathlex::ast::Expression;

let composition = Expression::Composition {
    outer: Box::new(Expression::Variable("f".to_string())),
    inner: Box::new(Expression::Variable("g".to_string())),
};
```

---

## Differential Forms

### `Differential { var }`

Differential of a variable: `dx`, `dy`, `dt`.

Represents a differential 1-form, distinct from derivative notation `d/dx`.
Commonly appears as the integration variable in integrals: `∫ f(x) dx`.
In differential geometry, differentials are 1-forms.

```rust
use mathlex::ast::Expression;

let dx = Expression::Differential { var: "x".to_string() };
let dt = Expression::Differential { var: "t".to_string() };
```

---

### `WedgeProduct { left, right }`

Wedge product (exterior product) of two differential forms: `dx ∧ dy`.

**Properties:**
- Anticommutative: `dx ∧ dy = -(dy ∧ dx)`
- Associative: `(dx ∧ dy) ∧ dz = dx ∧ (dy ∧ dz)`
- Wedge with itself is zero: `dx ∧ dx = 0`

**Common uses:** area/volume elements in integration, exterior calculus,
differential geometry.

```rust
use mathlex::ast::Expression;

// dx ∧ dy
let dx = Expression::Differential { var: "x".to_string() };
let dy = Expression::Differential { var: "y".to_string() };
let wedge = Expression::WedgeProduct {
    left: Box::new(dx),
    right: Box::new(dy),
};

// dx ∧ dy ∧ dz (nested)
let dz = Expression::Differential { var: "z".to_string() };
let wedge_3d = Expression::WedgeProduct {
    left: Box::new(wedge),
    right: Box::new(dz),
};
```
