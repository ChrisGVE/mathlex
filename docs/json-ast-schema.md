# mathlex JSON AST Schema

This document describes the JSON representation of the `Expression` AST produced
by mathlex when compiled with the `serde` feature flag. It is the reference for
consumers — in particular NumericSwift — that decode the JSON output of
`toJSON()` / `serde_json::to_string`.

## Encoding convention

serde's default **externally tagged** format is used throughout. Every
`Expression` value serializes as a single-key JSON object whose key is the
variant name and whose value is the payload:

```json
{ "VariantName": <payload> }
```

Unit variants (those with no fields) serialize as a bare JSON string:

```json
"VariantName"
```

Enum fields that are themselves enums (e.g. `BinaryOp`, `UnaryOp`) follow the
same rules: simple variants become bare strings, struct variants become objects.
Because every operator enum currently contains only unit variants, all operator
values appear as plain strings in JSON.

Optional fields (`Option<T>`) serialize as `null` when absent or as the value
when present.

---

## MathConstant values

All variants serialize as plain strings.

| JSON string   | Meaning                         |
| ------------- | ------------------------------- |
| `"Pi"`        | π ≈ 3.14159…                    |
| `"E"`         | Euler's number e ≈ 2.71828…     |
| `"I"`         | Imaginary unit i (i² = −1); also quaternion basis i |
| `"J"`         | Quaternion basis j (j² = −1)    |
| `"K"`         | Quaternion basis k (k² = −1)    |
| `"Infinity"`  | +∞                              |
| `"NegInfinity"` | −∞                            |
| `"NaN"`       | Not-a-Number (indeterminate)    |

---

## BinaryOp values

| JSON string    | Operator |
| -------------- | -------- |
| `"Add"`        | +        |
| `"Sub"`        | −        |
| `"Mul"`        | ×        |
| `"Div"`        | ÷        |
| `"Pow"`        | ^        |
| `"Mod"`        | %        |
| `"PlusMinus"`  | ±        |
| `"MinusPlus"`  | ∓        |

---

## UnaryOp values

| JSON string   | Operator    | Position |
| ------------- | ----------- | -------- |
| `"Neg"`       | −x          | prefix   |
| `"Pos"`       | +x          | prefix   |
| `"Factorial"` | n!          | postfix  |
| `"Transpose"` | Aᵀ or A'   | postfix  |

---

## InequalityOp values

| JSON string | Relation |
| ----------- | -------- |
| `"Lt"`      | <        |
| `"Le"`      | ≤        |
| `"Gt"`      | >        |
| `"Ge"`      | ≥        |
| `"Ne"`      | ≠        |

---

## LogicalOp values

| JSON string  | Operator |
| ------------ | -------- |
| `"And"`      | ∧        |
| `"Or"`       | ∨        |
| `"Not"`      | ¬        |
| `"Implies"`  | →        |
| `"Iff"`      | ↔        |

---

## RelationOp values

| JSON string    | Relation |
| -------------- | -------- |
| `"Similar"`    | ~        |
| `"Equivalent"` | ≡        |
| `"Congruent"`  | ≅        |
| `"Approx"`     | ≈        |

---

## Direction values

| JSON string | Meaning                   |
| ----------- | ------------------------- |
| `"Left"`    | Approach from below (x⁻)  |
| `"Right"`   | Approach from above (x⁺)  |
| `"Both"`    | Two-sided limit           |

---

## SetOp values

| JSON string      | Operation        |
| ---------------- | ---------------- |
| `"Union"`        | A ∪ B            |
| `"Intersection"` | A ∩ B            |
| `"Difference"`   | A ∖ B            |
| `"SymmetricDiff"`| A △ B            |
| `"CartesianProd"`| A × B            |

---

## SetRelation values

| JSON string    | Relation   |
| -------------- | ---------- |
| `"In"`         | x ∈ S      |
| `"NotIn"`      | x ∉ S      |
| `"Subset"`     | A ⊂ B      |
| `"SubsetEq"`   | A ⊆ B      |
| `"Superset"`   | A ⊃ B      |
| `"SupersetEq"` | A ⊇ B      |

---

## NumberSet values

| JSON string    | Set  |
| -------------- | ---- |
| `"Natural"`    | ℕ    |
| `"Integer"`    | ℤ    |
| `"Rational"`   | ℚ    |
| `"Real"`       | ℝ    |
| `"Complex"`    | ℂ    |
| `"Quaternion"` | ℍ    |

---

## VectorNotation values

| JSON string   | LaTeX notation        |
| ------------- | --------------------- |
| `"Bold"`      | `\mathbf{v}`          |
| `"Arrow"`     | `\vec{v}`             |
| `"Hat"`       | `\hat{v}`             |
| `"Underline"` | `\underline{v}`       |
| `"Plain"`     | no special decoration |

---

## IndexType values

| JSON string | Meaning                      |
| ----------- | ---------------------------- |
| `"Upper"`   | Superscript (contravariant)  |
| `"Lower"`   | Subscript (covariant)        |

---

## Expression variants

### Integer

An integer literal (`i64`).

```json
{ "Integer": 42 }
{ "Integer": -17 }
{ "Integer": 0 }
```

### Float

A finite floating-point literal (`f64`).

```json
{ "Float": 3.14 }
{ "Float": -2.5 }
{ "Float": 1.0e-10 }
```

**Important:** `serde_json` cannot represent non-finite IEEE 754 floats.
`f64::INFINITY`, `f64::NEG_INFINITY`, and `f64::NAN` all serialize to
`{ "Float": null }`. This value cannot round-trip back to the original.
Parsers never emit non-finite floats — use `Constant("Infinity")` or
`Constant("NegInfinity")` instead. If you receive `{ "Float": null }` it
was produced by manual AST construction and should be treated as an error.

### Variable

A symbolic variable name.

```json
{ "Variable": "x" }
{ "Variable": "theta" }
{ "Variable": "x_1" }
```

### Constant

A named mathematical constant.

```json
{ "Constant": "Pi" }
{ "Constant": "E" }
{ "Constant": "Infinity" }
{ "Constant": "NegInfinity" }
```

### Rational

A fraction. Both numerator and denominator are full `Expression` nodes.
Note: parsers emit `Binary { op: "Div", … }` for division. `Rational` is
reserved for symbolic simplification libraries.

```json
{
  "Rational": {
    "numerator": { "Integer": 1 },
    "denominator": { "Integer": 2 }
  }
}
```

### Complex

A complex number `a + bi`. Fields are full `Expression` nodes.
Note: parsers use `Binary` + `Constant("I")`. `Complex` is for simplified form.

```json
{
  "Complex": {
    "real": { "Integer": 3 },
    "imaginary": { "Integer": 4 }
  }
}
```

### Quaternion

A quaternion `a + bi + cj + dk`. Fields are full `Expression` nodes.

```json
{
  "Quaternion": {
    "real": { "Integer": 1 },
    "i": { "Integer": 2 },
    "j": { "Integer": 3 },
    "k": { "Integer": 4 }
  }
}
```

### Binary

A binary operation with an operator and two operands.

```json
{
  "Binary": {
    "op": "Add",
    "left": { "Variable": "x" },
    "right": { "Integer": 1 }
  }
}
```

### Unary

A unary operation with an operator and one operand.

```json
{
  "Unary": {
    "op": "Neg",
    "operand": { "Variable": "x" }
  }
}
```

### Function

A function call with a name and an argument list. The argument list may be
empty.

```json
{
  "Function": {
    "name": "sin",
    "args": [ { "Variable": "x" } ]
  }
}
```

Empty argument list:

```json
{
  "Function": {
    "name": "f",
    "args": []
  }
}
```

### Derivative

An ordinary derivative of order `n` with respect to a variable.

Fields:
- `expr`: the expression being differentiated
- `var`: variable name (string)
- `order`: derivative order (positive integer, `u32`)

```json
{
  "Derivative": {
    "expr": {
      "Function": {
        "name": "sin",
        "args": [ { "Variable": "x" } ]
      }
    },
    "var": "x",
    "order": 1
  }
}
```

### PartialDerivative

A partial derivative. Same fields as `Derivative`.

```json
{
  "PartialDerivative": {
    "expr": { "Variable": "f" },
    "var": "x",
    "order": 1
  }
}
```

### Integral

A definite or indefinite integral.

Fields:
- `integrand`: expression being integrated
- `var`: variable of integration
- `bounds`: `null` for indefinite, or an object with `lower` and `upper`

Indefinite:

```json
{
  "Integral": {
    "integrand": { "Variable": "x" },
    "var": "x",
    "bounds": null
  }
}
```

Definite (∫₀¹ x dx):

```json
{
  "Integral": {
    "integrand": { "Variable": "x" },
    "var": "x",
    "bounds": {
      "lower": { "Integer": 0 },
      "upper": { "Integer": 1 }
    }
  }
}
```

### MultipleIntegral

A double, triple, or higher-dimensional integral.

Fields:
- `dimension`: number of integral signs (`u8`)
- `integrand`: expression being integrated
- `bounds`: `null` or an object `{ "bounds": [ IntegralBounds, … ] }`
- `vars`: array of variable name strings, one per dimension

```json
{
  "MultipleIntegral": {
    "dimension": 2,
    "integrand": { "Variable": "f" },
    "bounds": null,
    "vars": [ "x", "y" ]
  }
}
```

With bounds:

```json
{
  "MultipleIntegral": {
    "dimension": 2,
    "integrand": { "Variable": "f" },
    "bounds": {
      "bounds": [
        { "lower": { "Integer": 0 }, "upper": { "Integer": 1 } },
        { "lower": { "Integer": 0 }, "upper": { "Integer": 2 } }
      ]
    },
    "vars": [ "x", "y" ]
  }
}
```

### ClosedIntegral

A closed contour, surface, or volume integral (∮, ∯, ∰).

Fields:
- `dimension`: 1 = line ∮, 2 = surface ∯, 3 = volume ∰
- `integrand`: expression
- `surface`: optional curve/surface name string, or `null`
- `var`: variable of integration

```json
{
  "ClosedIntegral": {
    "dimension": 1,
    "integrand": { "Variable": "F" },
    "surface": "C",
    "var": "r"
  }
}
```

### Limit

A limit as a variable approaches a value.

Fields:
- `expr`: expression whose limit is taken
- `var`: the approaching variable
- `to`: the value being approached
- `direction`: `"Left"` | `"Right"` | `"Both"`

```json
{
  "Limit": {
    "expr": { "Variable": "f" },
    "var": "x",
    "to": { "Integer": 0 },
    "direction": "Both"
  }
}
```

### Sum

A summation Σ_{index=lower}^{upper} body.

```json
{
  "Sum": {
    "index": "i",
    "lower": { "Integer": 1 },
    "upper": { "Variable": "n" },
    "body": { "Variable": "i" }
  }
}
```

### Product

A product Π_{index=lower}^{upper} body.

```json
{
  "Product": {
    "index": "k",
    "lower": { "Integer": 1 },
    "upper": { "Integer": 10 },
    "body": { "Variable": "k" }
  }
}
```

### Vector

An ordered list of expressions.

```json
{
  "Vector": [
    { "Integer": 1 },
    { "Integer": 2 },
    { "Integer": 3 }
  ]
}
```

Empty vector: `{ "Vector": [] }`

### Matrix

A 2D array of expressions (array of rows, each row an array of expressions).

```json
{
  "Matrix": [
    [ { "Integer": 1 }, { "Integer": 0 } ],
    [ { "Integer": 0 }, { "Integer": 1 } ]
  ]
}
```

### Equation

An equality between two expressions.

```json
{
  "Equation": {
    "left": { "Variable": "x" },
    "right": { "Integer": 5 }
  }
}
```

### Inequality

A relational comparison.

```json
{
  "Inequality": {
    "op": "Lt",
    "left": { "Variable": "x" },
    "right": { "Integer": 0 }
  }
}
```

### ForAll

Universal quantifier ∀x ∈ domain: body.

```json
{
  "ForAll": {
    "variable": "x",
    "domain": { "NumberSetExpr": "Real" },
    "body": { "Variable": "P" }
  }
}
```

Domain-free: `"domain": null`

### Exists

Existential quantifier ∃x: body. `unique: true` means ∃!.

```json
{
  "Exists": {
    "variable": "x",
    "domain": null,
    "body": { "Variable": "P" },
    "unique": false
  }
}
```

### Logical

A logical expression with n-ary operands.

```json
{
  "Logical": {
    "op": "And",
    "operands": [
      { "Variable": "P" },
      { "Variable": "Q" }
    ]
  }
}
```

### MarkedVector

A named vector with an explicit visual notation.

```json
{
  "MarkedVector": {
    "name": "v",
    "notation": "Bold"
  }
}
```

### DotProduct

Inner product u · v.

```json
{
  "DotProduct": {
    "left": { "Variable": "u" },
    "right": { "Variable": "v" }
  }
}
```

### CrossProduct

Cross product u × v.

```json
{
  "CrossProduct": {
    "left": { "Variable": "u" },
    "right": { "Variable": "v" }
  }
}
```

### OuterProduct

Outer (tensor) product u ⊗ v.

```json
{
  "OuterProduct": {
    "left": { "Variable": "u" },
    "right": { "Variable": "v" }
  }
}
```

### Gradient

Gradient of a scalar field: ∇f.

```json
{
  "Gradient": {
    "expr": { "Variable": "f" }
  }
}
```

### Divergence

Divergence of a vector field: ∇·F.

```json
{
  "Divergence": {
    "field": { "Variable": "F" }
  }
}
```

### Curl

Curl of a vector field: ∇×F.

```json
{
  "Curl": {
    "field": { "Variable": "F" }
  }
}
```

### Laplacian

Laplacian of a scalar field: ∇²f.

```json
{
  "Laplacian": {
    "expr": { "Variable": "f" }
  }
}
```

### Nabla

The raw ∇ operator without an operand. Serializes as a bare string.

```json
"Nabla"
```

### Determinant

```json
{
  "Determinant": {
    "matrix": { "Variable": "A" }
  }
}
```

### Trace

```json
{
  "Trace": {
    "matrix": { "Variable": "A" }
  }
}
```

### Rank

```json
{
  "Rank": {
    "matrix": { "Variable": "A" }
  }
}
```

### ConjugateTranspose

Hermitian adjoint A†.

```json
{
  "ConjugateTranspose": {
    "matrix": { "Variable": "A" }
  }
}
```

### MatrixInverse

Matrix inverse A⁻¹.

```json
{
  "MatrixInverse": {
    "matrix": { "Variable": "A" }
  }
}
```

### NumberSetExpr

A standard number set.

```json
{ "NumberSetExpr": "Real" }
{ "NumberSetExpr": "Integer" }
```

### SetOperation

A binary set operation.

```json
{
  "SetOperation": {
    "op": "Union",
    "left": { "Variable": "A" },
    "right": { "Variable": "B" }
  }
}
```

### SetRelationExpr

A set membership or subset relation.

```json
{
  "SetRelationExpr": {
    "relation": "In",
    "element": { "Variable": "x" },
    "set": { "NumberSetExpr": "Real" }
  }
}
```

### SetBuilder

Set-builder notation {x ∈ domain | predicate}.

```json
{
  "SetBuilder": {
    "variable": "x",
    "domain": { "NumberSetExpr": "Real" },
    "predicate": {
      "Inequality": {
        "op": "Gt",
        "left": { "Variable": "x" },
        "right": { "Integer": 0 }
      }
    }
  }
}
```

Domain-free: `"domain": null`

### EmptySet

The empty set ∅. Serializes as a bare string.

```json
"EmptySet"
```

### PowerSet

Power set 𝒫(S).

```json
{
  "PowerSet": {
    "set": { "Variable": "A" }
  }
}
```

### Tensor

A named tensor with indexed notation. Each index has a `name` string and an
`index_type` of `"Upper"` or `"Lower"`.

```json
{
  "Tensor": {
    "name": "T",
    "indices": [
      { "name": "i", "index_type": "Upper" },
      { "name": "j", "index_type": "Lower" }
    ]
  }
}
```

### KroneckerDelta

Kronecker delta δ^i_j.

```json
{
  "KroneckerDelta": {
    "indices": [
      { "name": "i", "index_type": "Upper" },
      { "name": "j", "index_type": "Lower" }
    ]
  }
}
```

### LeviCivita

Levi-Civita symbol ε^{ijk}.

```json
{
  "LeviCivita": {
    "indices": [
      { "name": "i", "index_type": "Upper" },
      { "name": "j", "index_type": "Upper" },
      { "name": "k", "index_type": "Upper" }
    ]
  }
}
```

### FunctionSignature

Function type declaration f: A → B.

```json
{
  "FunctionSignature": {
    "name": "f",
    "domain": { "NumberSetExpr": "Real" },
    "codomain": { "NumberSetExpr": "Real" }
  }
}
```

### Composition

Function composition f ∘ g (outer applied after inner).

```json
{
  "Composition": {
    "outer": { "Variable": "f" },
    "inner": { "Variable": "g" }
  }
}
```

### Differential

A differential form: dx, dy, dt.

```json
{
  "Differential": {
    "var": "x"
  }
}
```

### WedgeProduct

Exterior product of two forms: dx ∧ dy.

```json
{
  "WedgeProduct": {
    "left": { "Differential": { "var": "x" } },
    "right": { "Differential": { "var": "y" } }
  }
}
```

### Relation

A mathematical relation (similarity, equivalence, congruence, approximation).

```json
{
  "Relation": {
    "op": "Similar",
    "left": { "Variable": "a" },
    "right": { "Variable": "b" }
  }
}
```

---

## Edge cases

### Non-finite Float nodes

`serde_json` serializes non-finite `f64` values as `null`. The result is not
round-trippable:

```json
{ "Float": null }
```

This occurs only when an AST is constructed manually with `MathFloat::from(f64::INFINITY)`.
Parsers always use `Constant("Infinity")` / `Constant("NegInfinity")` for
infinity. Treat `{ "Float": null }` as an error in consumer code.

### Empty argument lists (Function)

A zero-argument function is valid and produces an empty JSON array:

```json
{ "Function": { "name": "f", "args": [] } }
```

### Empty Vector and Matrix

```json
{ "Vector": [] }
{ "Matrix": [] }
```

---

## Complete example: sin(x)² + cos(x)²

The expression sin(x)^2 + cos(x)^2 produces:

```json
{
  "Binary": {
    "op": "Add",
    "left": {
      "Binary": {
        "op": "Pow",
        "left": {
          "Function": {
            "name": "sin",
            "args": [ { "Variable": "x" } ]
          }
        },
        "right": { "Integer": 2 }
      }
    },
    "right": {
      "Binary": {
        "op": "Pow",
        "left": {
          "Function": {
            "name": "cos",
            "args": [ { "Variable": "x" } ]
          }
        },
        "right": { "Integer": 2 }
      }
    }
  }
}
```

---

## Swift Codable types for NumericSwift

The following Swift types decode the JSON produced by mathlex. Paste them into
your target and call `JSONDecoder().decode(MathLexExpression.self, from: data)`.

```swift
import Foundation

// MARK: - Operator enums

enum BinaryOp: String, Decodable {
    case add = "Add"
    case sub = "Sub"
    case mul = "Mul"
    case div = "Div"
    case pow = "Pow"
    case mod = "Mod"
    case plusMinus = "PlusMinus"
    case minusPlus = "MinusPlus"
}

enum UnaryOp: String, Decodable {
    case neg = "Neg"
    case pos = "Pos"
    case factorial = "Factorial"
    case transpose = "Transpose"
}

enum InequalityOp: String, Decodable {
    case lt = "Lt"
    case le = "Le"
    case gt = "Gt"
    case ge = "Ge"
    case ne = "Ne"
}

enum LogicalOp: String, Decodable {
    case and = "And"
    case or = "Or"
    case not = "Not"
    case implies = "Implies"
    case iff = "Iff"
}

enum RelationOp: String, Decodable {
    case similar = "Similar"
    case equivalent = "Equivalent"
    case congruent = "Congruent"
    case approx = "Approx"
}

enum MathConstant: String, Decodable {
    case pi = "Pi"
    case e = "E"
    case i = "I"
    case j = "J"
    case k = "K"
    case infinity = "Infinity"
    case negInfinity = "NegInfinity"
    case nan = "NaN"
}

enum Direction: String, Decodable {
    case left = "Left"
    case right = "Right"
    case both = "Both"
}

enum SetOp: String, Decodable {
    case union = "Union"
    case intersection = "Intersection"
    case difference = "Difference"
    case symmetricDiff = "SymmetricDiff"
    case cartesianProd = "CartesianProd"
}

enum SetRelation: String, Decodable {
    case `in` = "In"
    case notIn = "NotIn"
    case subset = "Subset"
    case subsetEq = "SubsetEq"
    case superset = "Superset"
    case supersetEq = "SupersetEq"
}

enum NumberSet: String, Decodable {
    case natural = "Natural"
    case integer = "Integer"
    case rational = "Rational"
    case real = "Real"
    case complex = "Complex"
    case quaternion = "Quaternion"
}

enum VectorNotation: String, Decodable {
    case bold = "Bold"
    case arrow = "Arrow"
    case hat = "Hat"
    case underline = "Underline"
    case plain = "Plain"
}

enum IndexType: String, Decodable {
    case upper = "Upper"
    case lower = "Lower"
}

// MARK: - Supporting types

struct TensorIndex: Decodable {
    let name: String
    let index_type: IndexType
}

struct IntegralBounds: Decodable {
    let lower: MathLexExpression
    let upper: MathLexExpression
}

struct MultipleBounds: Decodable {
    let bounds: [IntegralBounds]
}

// MARK: - Expression

/// Mirrors the Rust `Expression` enum. Decodes the externally-tagged JSON
/// produced by mathlex with the `serde` feature enabled.
indirect enum MathLexExpression: Decodable {

    case integer(Int64)
    case float(Double?)          // null when non-finite; treat as error
    case variable(String)
    case constant(MathConstant)

    case rational(numerator: MathLexExpression, denominator: MathLexExpression)
    case complex(real: MathLexExpression, imaginary: MathLexExpression)
    case quaternion(real: MathLexExpression, i: MathLexExpression,
                    j: MathLexExpression, k: MathLexExpression)

    case binary(op: BinaryOp, left: MathLexExpression, right: MathLexExpression)
    case unary(op: UnaryOp, operand: MathLexExpression)
    case function(name: String, args: [MathLexExpression])

    case derivative(expr: MathLexExpression, var: String, order: UInt32)
    case partialDerivative(expr: MathLexExpression, var: String, order: UInt32)
    case integral(integrand: MathLexExpression, var: String,
                  bounds: IntegralBounds?)
    case multipleIntegral(dimension: UInt8, integrand: MathLexExpression,
                          bounds: MultipleBounds?, vars: [String])
    case closedIntegral(dimension: UInt8, integrand: MathLexExpression,
                        surface: String?, var: String)
    case limit(expr: MathLexExpression, var: String,
               to: MathLexExpression, direction: Direction)
    case sum(index: String, lower: MathLexExpression,
             upper: MathLexExpression, body: MathLexExpression)
    case product(index: String, lower: MathLexExpression,
                 upper: MathLexExpression, body: MathLexExpression)

    case vector([MathLexExpression])
    case matrix([[MathLexExpression]])

    case equation(left: MathLexExpression, right: MathLexExpression)
    case inequality(op: InequalityOp, left: MathLexExpression,
                    right: MathLexExpression)

    case forAll(variable: String, domain: MathLexExpression?,
                body: MathLexExpression)
    case exists(variable: String, domain: MathLexExpression?,
                body: MathLexExpression, unique: Bool)
    case logical(op: LogicalOp, operands: [MathLexExpression])

    case markedVector(name: String, notation: VectorNotation)
    case dotProduct(left: MathLexExpression, right: MathLexExpression)
    case crossProduct(left: MathLexExpression, right: MathLexExpression)
    case outerProduct(left: MathLexExpression, right: MathLexExpression)

    case gradient(expr: MathLexExpression)
    case divergence(field: MathLexExpression)
    case curl(field: MathLexExpression)
    case laplacian(expr: MathLexExpression)
    case nabla

    case determinant(matrix: MathLexExpression)
    case trace(matrix: MathLexExpression)
    case rank(matrix: MathLexExpression)
    case conjugateTranspose(matrix: MathLexExpression)
    case matrixInverse(matrix: MathLexExpression)

    case numberSetExpr(NumberSet)
    case setOperation(op: SetOp, left: MathLexExpression, right: MathLexExpression)
    case setRelationExpr(relation: SetRelation, element: MathLexExpression,
                         set: MathLexExpression)
    case setBuilder(variable: String, domain: MathLexExpression?,
                    predicate: MathLexExpression)
    case emptySet
    case powerSet(set: MathLexExpression)

    case tensor(name: String, indices: [TensorIndex])
    case kroneckerDelta(indices: [TensorIndex])
    case leviCivita(indices: [TensorIndex])

    case functionSignature(name: String, domain: MathLexExpression,
                           codomain: MathLexExpression)
    case composition(outer: MathLexExpression, inner: MathLexExpression)
    case differential(var: String)
    case wedgeProduct(left: MathLexExpression, right: MathLexExpression)
    case relation(op: RelationOp, left: MathLexExpression,
                  right: MathLexExpression)

    // MARK: Decodable

    private enum TopKey: String, CodingKey {
        case Integer, Float, Variable, Constant
        case Rational, Complex, Quaternion
        case Binary, Unary, Function
        case Derivative, PartialDerivative, Integral, MultipleIntegral
        case ClosedIntegral, Limit, Sum, Product
        case Vector, Matrix
        case Equation, Inequality
        case ForAll, Exists, Logical
        case MarkedVector, DotProduct, CrossProduct, OuterProduct
        case Gradient, Divergence, Curl, Laplacian, Nabla
        case Determinant, Trace, Rank, ConjugateTranspose, MatrixInverse
        case NumberSetExpr, SetOperation, SetRelationExpr, SetBuilder
        case EmptySet, PowerSet
        case Tensor, KroneckerDelta, LeviCivita
        case FunctionSignature, Composition, Differential, WedgeProduct
        case Relation
    }

    init(from decoder: Decoder) throws {
        // Unit variants (Nabla, EmptySet) come as a bare string.
        if let str = try? decoder.singleValueContainer().decode(String.self) {
            switch str {
            case "Nabla":    self = .nabla;    return
            case "EmptySet": self = .emptySet; return
            default: break
            }
        }

        let container = try decoder.container(keyedBy: TopKey.self)

        if let v = try container.decodeIfPresent(Int64.self, forKey: .Integer) {
            self = .integer(v); return
        }
        if container.contains(.Float) {
            let v = try container.decodeIfPresent(Double.self, forKey: .Float)
            self = .float(v); return
        }
        if let v = try container.decodeIfPresent(String.self, forKey: .Variable) {
            self = .variable(v); return
        }
        if let v = try container.decodeIfPresent(MathConstant.self, forKey: .Constant) {
            self = .constant(v); return
        }
        if let v = try container.decodeIfPresent(NumberSet.self, forKey: .NumberSetExpr) {
            self = .numberSetExpr(v); return
        }

        // Struct variants — delegate to private helper structs below.
        if container.contains(.Rational) {
            struct P: Decodable { let numerator, denominator: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Rational)
            self = .rational(numerator: p.numerator, denominator: p.denominator)
            return
        }
        if container.contains(.Complex) {
            struct P: Decodable { let real, imaginary: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Complex)
            self = .complex(real: p.real, imaginary: p.imaginary); return
        }
        if container.contains(.Quaternion) {
            struct P: Decodable { let real, i, j, k: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Quaternion)
            self = .quaternion(real: p.real, i: p.i, j: p.j, k: p.k); return
        }
        if container.contains(.Binary) {
            struct P: Decodable { let op: BinaryOp; let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Binary)
            self = .binary(op: p.op, left: p.left, right: p.right); return
        }
        if container.contains(.Unary) {
            struct P: Decodable { let op: UnaryOp; let operand: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Unary)
            self = .unary(op: p.op, operand: p.operand); return
        }
        if container.contains(.Function) {
            struct P: Decodable { let name: String; let args: [MathLexExpression] }
            let p = try container.decode(P.self, forKey: .Function)
            self = .function(name: p.name, args: p.args); return
        }
        if container.contains(.Derivative) {
            struct P: Decodable {
                let expr: MathLexExpression
                let `var`: String
                let order: UInt32
            }
            let p = try container.decode(P.self, forKey: .Derivative)
            self = .derivative(expr: p.expr, var: p.var, order: p.order); return
        }
        if container.contains(.PartialDerivative) {
            struct P: Decodable {
                let expr: MathLexExpression
                let `var`: String
                let order: UInt32
            }
            let p = try container.decode(P.self, forKey: .PartialDerivative)
            self = .partialDerivative(expr: p.expr, var: p.var, order: p.order)
            return
        }
        if container.contains(.Integral) {
            struct P: Decodable {
                let integrand: MathLexExpression
                let `var`: String
                let bounds: IntegralBounds?
            }
            let p = try container.decode(P.self, forKey: .Integral)
            self = .integral(integrand: p.integrand, var: p.var, bounds: p.bounds)
            return
        }
        if container.contains(.MultipleIntegral) {
            struct P: Decodable {
                let dimension: UInt8
                let integrand: MathLexExpression
                let bounds: MultipleBounds?
                let vars: [String]
            }
            let p = try container.decode(P.self, forKey: .MultipleIntegral)
            self = .multipleIntegral(dimension: p.dimension, integrand: p.integrand,
                                     bounds: p.bounds, vars: p.vars); return
        }
        if container.contains(.ClosedIntegral) {
            struct P: Decodable {
                let dimension: UInt8
                let integrand: MathLexExpression
                let surface: String?
                let `var`: String
            }
            let p = try container.decode(P.self, forKey: .ClosedIntegral)
            self = .closedIntegral(dimension: p.dimension, integrand: p.integrand,
                                   surface: p.surface, var: p.var); return
        }
        if container.contains(.Limit) {
            struct P: Decodable {
                let expr: MathLexExpression
                let `var`: String
                let to: MathLexExpression
                let direction: Direction
            }
            let p = try container.decode(P.self, forKey: .Limit)
            self = .limit(expr: p.expr, var: p.var, to: p.to, direction: p.direction)
            return
        }
        if container.contains(.Sum) {
            struct P: Decodable {
                let index: String
                let lower, upper, body: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .Sum)
            self = .sum(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
            return
        }
        if container.contains(.Product) {
            struct P: Decodable {
                let index: String
                let lower, upper, body: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .Product)
            self = .product(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
            return
        }
        if container.contains(.Vector) {
            let v = try container.decode([MathLexExpression].self, forKey: .Vector)
            self = .vector(v); return
        }
        if container.contains(.Matrix) {
            let m = try container.decode([[MathLexExpression]].self, forKey: .Matrix)
            self = .matrix(m); return
        }
        if container.contains(.Equation) {
            struct P: Decodable { let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Equation)
            self = .equation(left: p.left, right: p.right); return
        }
        if container.contains(.Inequality) {
            struct P: Decodable { let op: InequalityOp; let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Inequality)
            self = .inequality(op: p.op, left: p.left, right: p.right); return
        }
        if container.contains(.ForAll) {
            struct P: Decodable {
                let variable: String
                let domain: MathLexExpression?
                let body: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .ForAll)
            self = .forAll(variable: p.variable, domain: p.domain, body: p.body)
            return
        }
        if container.contains(.Exists) {
            struct P: Decodable {
                let variable: String
                let domain: MathLexExpression?
                let body: MathLexExpression
                let unique: Bool
            }
            let p = try container.decode(P.self, forKey: .Exists)
            self = .exists(variable: p.variable, domain: p.domain,
                           body: p.body, unique: p.unique); return
        }
        if container.contains(.Logical) {
            struct P: Decodable { let op: LogicalOp; let operands: [MathLexExpression] }
            let p = try container.decode(P.self, forKey: .Logical)
            self = .logical(op: p.op, operands: p.operands); return
        }
        if container.contains(.MarkedVector) {
            struct P: Decodable { let name: String; let notation: VectorNotation }
            let p = try container.decode(P.self, forKey: .MarkedVector)
            self = .markedVector(name: p.name, notation: p.notation); return
        }
        if container.contains(.DotProduct) {
            struct P: Decodable { let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .DotProduct)
            self = .dotProduct(left: p.left, right: p.right); return
        }
        if container.contains(.CrossProduct) {
            struct P: Decodable { let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .CrossProduct)
            self = .crossProduct(left: p.left, right: p.right); return
        }
        if container.contains(.OuterProduct) {
            struct P: Decodable { let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .OuterProduct)
            self = .outerProduct(left: p.left, right: p.right); return
        }
        if container.contains(.Gradient) {
            struct P: Decodable { let expr: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Gradient)
            self = .gradient(expr: p.expr); return
        }
        if container.contains(.Divergence) {
            struct P: Decodable { let field: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Divergence)
            self = .divergence(field: p.field); return
        }
        if container.contains(.Curl) {
            struct P: Decodable { let field: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Curl)
            self = .curl(field: p.field); return
        }
        if container.contains(.Laplacian) {
            struct P: Decodable { let expr: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Laplacian)
            self = .laplacian(expr: p.expr); return
        }
        if container.contains(.Determinant) {
            struct P: Decodable { let matrix: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Determinant)
            self = .determinant(matrix: p.matrix); return
        }
        if container.contains(.Trace) {
            struct P: Decodable { let matrix: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Trace)
            self = .trace(matrix: p.matrix); return
        }
        if container.contains(.Rank) {
            struct P: Decodable { let matrix: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Rank)
            self = .rank(matrix: p.matrix); return
        }
        if container.contains(.ConjugateTranspose) {
            struct P: Decodable { let matrix: MathLexExpression }
            let p = try container.decode(P.self, forKey: .ConjugateTranspose)
            self = .conjugateTranspose(matrix: p.matrix); return
        }
        if container.contains(.MatrixInverse) {
            struct P: Decodable { let matrix: MathLexExpression }
            let p = try container.decode(P.self, forKey: .MatrixInverse)
            self = .matrixInverse(matrix: p.matrix); return
        }
        if container.contains(.SetOperation) {
            struct P: Decodable { let op: SetOp; let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .SetOperation)
            self = .setOperation(op: p.op, left: p.left, right: p.right); return
        }
        if container.contains(.SetRelationExpr) {
            struct P: Decodable {
                let relation: SetRelation
                let element, set: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .SetRelationExpr)
            self = .setRelationExpr(relation: p.relation, element: p.element,
                                    set: p.set); return
        }
        if container.contains(.SetBuilder) {
            struct P: Decodable {
                let variable: String
                let domain: MathLexExpression?
                let predicate: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .SetBuilder)
            self = .setBuilder(variable: p.variable, domain: p.domain,
                                predicate: p.predicate); return
        }
        if container.contains(.PowerSet) {
            struct P: Decodable { let set: MathLexExpression }
            let p = try container.decode(P.self, forKey: .PowerSet)
            self = .powerSet(set: p.set); return
        }
        if container.contains(.Tensor) {
            struct P: Decodable { let name: String; let indices: [TensorIndex] }
            let p = try container.decode(P.self, forKey: .Tensor)
            self = .tensor(name: p.name, indices: p.indices); return
        }
        if container.contains(.KroneckerDelta) {
            struct P: Decodable { let indices: [TensorIndex] }
            let p = try container.decode(P.self, forKey: .KroneckerDelta)
            self = .kroneckerDelta(indices: p.indices); return
        }
        if container.contains(.LeviCivita) {
            struct P: Decodable { let indices: [TensorIndex] }
            let p = try container.decode(P.self, forKey: .LeviCivita)
            self = .leviCivita(indices: p.indices); return
        }
        if container.contains(.FunctionSignature) {
            struct P: Decodable {
                let name: String
                let domain, codomain: MathLexExpression
            }
            let p = try container.decode(P.self, forKey: .FunctionSignature)
            self = .functionSignature(name: p.name, domain: p.domain,
                                       codomain: p.codomain); return
        }
        if container.contains(.Composition) {
            struct P: Decodable { let outer, inner: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Composition)
            self = .composition(outer: p.outer, inner: p.inner); return
        }
        if container.contains(.Differential) {
            struct P: Decodable { let `var`: String }
            let p = try container.decode(P.self, forKey: .Differential)
            self = .differential(var: p.var); return
        }
        if container.contains(.WedgeProduct) {
            struct P: Decodable { let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .WedgeProduct)
            self = .wedgeProduct(left: p.left, right: p.right); return
        }
        if container.contains(.Relation) {
            struct P: Decodable { let op: RelationOp; let left, right: MathLexExpression }
            let p = try container.decode(P.self, forKey: .Relation)
            self = .relation(op: p.op, left: p.left, right: p.right); return
        }

        throw DecodingError.dataCorrupted(
            DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "Unknown MathLexExpression variant"
            )
        )
    }
}

// MARK: - Basic numeric evaluator skeleton

/// Evaluates a `MathLexExpression` that contains only numeric sub-trees.
/// Returns `nil` for any node that cannot be reduced to a single `Double`
/// (symbolic variables, non-numeric operators, etc.).
struct BasicEvaluator {

    typealias Env = [String: Double]

    func evaluate(_ expr: MathLexExpression, env: Env = [:]) -> Double? {
        switch expr {
        case .integer(let n):
            return Double(n)
        case .float(let v):
            return v        // nil propagates for non-finite floats
        case .variable(let name):
            return env[name]
        case .constant(let c):
            return constantValue(c)

        case .binary(let op, let lhs, let rhs):
            guard let l = evaluate(lhs, env: env),
                  let r = evaluate(rhs, env: env) else { return nil }
            return applyBinary(op, l, r)

        case .unary(let op, let operand):
            guard let v = evaluate(operand, env: env) else { return nil }
            return applyUnary(op, v)

        case .function(let name, let args):
            let vals = args.compactMap { evaluate($0, env: env) }
            guard vals.count == args.count else { return nil }
            return applyFunction(name, vals)

        default:
            return nil      // calculus, sets, tensors, etc. not handled here
        }
    }

    private func constantValue(_ c: MathConstant) -> Double? {
        switch c {
        case .pi:          return Double.pi
        case .e:           return M_E
        case .infinity:    return Double.infinity
        case .negInfinity: return -Double.infinity
        default:           return nil
        }
    }

    private func applyBinary(_ op: BinaryOp, _ l: Double, _ r: Double) -> Double? {
        switch op {
        case .add: return l + r
        case .sub: return l - r
        case .mul: return l * r
        case .div: return l / r
        case .pow: return pow(l, r)
        case .mod: return l.truncatingRemainder(dividingBy: r)
        default:   return nil
        }
    }

    private func applyUnary(_ op: UnaryOp, _ v: Double) -> Double? {
        switch op {
        case .neg:       return -v
        case .pos:       return v
        case .factorial: return v >= 0 ? tgamma(v + 1) : nil
        default:         return nil
        }
    }

    private func applyFunction(_ name: String, _ args: [Double]) -> Double? {
        switch (name, args.count) {
        case ("sin",  1): return sin(args[0])
        case ("cos",  1): return cos(args[0])
        case ("tan",  1): return tan(args[0])
        case ("sqrt", 1): return sqrt(args[0])
        case ("exp",  1): return exp(args[0])
        case ("ln",   1): return log(args[0])
        case ("log",  2): return log(args[0]) / log(args[1])
        case ("abs",  1): return abs(args[0])
        default:          return nil
        }
    }
}
```
