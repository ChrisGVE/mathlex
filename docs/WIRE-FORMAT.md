# mathlex Wire Format v0.4.0

This document is the normative reference for the JSON serialization format
produced and consumed by the `serde` feature of mathlex. It supersedes
`docs/json-ast-schema.md` for all v0.4.0 and later consumers.

All 53 `ExprKind` variants are covered. The golden fixture files under
`tests/fixtures/serde/` are generated from this specification and verified
on every CI run.

---

## Convention

Every serialized node is a JSON object with two mandatory fields and one
optional field:

```
{
  "kind":        "<VariantName>",   // always present
  "value":       <payload>,         // present for all non-unit variants
  "annotations": { ... }            // present only when non-empty
}
```

**Adjacently-tagged enums.** `ExprKind` uses serde's `tag = "kind",
content = "value"` adjacently-tagged format. Both `kind` and `value`
always appear at the top level of the object — never nested.

**Unit variants** carry no `value` field:

```json
{ "kind": "Nabla" }
```

**Nested enums** (`BinaryOp`, `UnaryOp`, `InequalityOp`, etc.) use the
same adjacently-tagged convention. Because all their current variants are
unit variants they serialize as `{ "kind": "Add" }`, `{ "kind": "Neg" }`,
and so on.

**`Expression` vs `ExprKind`.** The public type is `Expression`, which
wraps `ExprKind` together with an `AnnotationSet`. The custom serializer
for `Expression` merges the `ExprKind` fields with the optional
`annotations` field into a single flat object, so the on-wire shape
is always one JSON object per node.

---

## Annotations

`AnnotationSet` is an ordered string-to-string map. The field is omitted
entirely when empty, which is the case for all nodes produced by the
parsers in v0.4.0. Deserializing a missing `annotations` field yields an
empty set (backward compatible).

```json
{
  "kind": "Variable",
  "value": "x",
  "annotations": {
    "unit": "m/s",
    "source": "input"
  }
}
```

Future releases of thales will populate this field. Third-party consumers
must treat unrecognized annotation keys as opaque strings and must not
reject documents that contain them.

---

## Variant Reference

### Numeric Types

#### Integer

Payload: `i64`.

```json
{ "kind": "Integer", "value": 42 }
```

#### Float

Payload: `f64`. Serialized with full double precision.

```json
{ "kind": "Float", "value": 3.14 }
```

#### Rational

Payload: object with `numerator` and `denominator` (both `Expression`).

```json
{
  "kind": "Rational",
  "value": {
    "numerator": { "kind": "Integer", "value": 3 },
    "denominator": { "kind": "Integer", "value": 4 }
  }
}
```

#### Complex

Payload: object with `real` and `imaginary` (both `Expression`).

```json
{
  "kind": "Complex",
  "value": {
    "real": { "kind": "Integer", "value": 2 },
    "imaginary": { "kind": "Integer", "value": 3 }
  }
}
```

#### Quaternion

Payload: object with `real`, `i`, `j`, `k` (all `Expression`).

```json
{
  "kind": "Quaternion",
  "value": {
    "real": { "kind": "Integer", "value": 1 },
    "i":    { "kind": "Integer", "value": 2 },
    "j":    { "kind": "Integer", "value": 3 },
    "k":    { "kind": "Integer", "value": 4 }
  }
}
```

#### Variable

Payload: `String`.

```json
{ "kind": "Variable", "value": "x" }
```

#### Constant

Payload: a `MathConstant` object. All `MathConstant` variants are unit
variants and serialize as `{ "kind": "<Name>" }`.

| `kind` value  | Meaning                         |
|---------------|---------------------------------|
| `"Pi"`        | π ≈ 3.14159…                    |
| `"E"`         | Euler's number e ≈ 2.71828…     |
| `"I"`         | Imaginary unit (i² = −1)        |
| `"J"`         | Quaternion basis j (j² = −1)    |
| `"K"`         | Quaternion basis k (k² = −1)    |
| `"Infinity"`  | +∞                              |
| `"NegInfinity"` | −∞                            |
| `"NaN"`       | Not-a-Number                    |

```json
{ "kind": "Constant", "value": { "kind": "Pi" } }
```

---

### Operations

#### Binary

Payload: `op` (`BinaryOp`), `left` and `right` (`Expression`).

Common `BinaryOp` kind values: `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Mod`,
`PlusMinus`, `MinusPlus`.

```json
{
  "kind": "Binary",
  "value": {
    "op": { "kind": "Add" },
    "left":  { "kind": "Integer", "value": 1 },
    "right": { "kind": "Integer", "value": 2 }
  }
}
```

#### Unary

Payload: `op` (`UnaryOp`), `operand` (`Expression`).

Common `UnaryOp` kind values: `Neg`, `Factorial`, `Abs`, `Sqrt`.

```json
{
  "kind": "Unary",
  "value": {
    "op": { "kind": "Neg" },
    "operand": { "kind": "Variable", "value": "x" }
  }
}
```

#### Function

Payload: `name` (`String`), `args` (`Vec<Expression>`).

```json
{
  "kind": "Function",
  "value": {
    "name": "sin",
    "args": [ { "kind": "Variable", "value": "x" } ]
  }
}
```

---

### Calculus

#### Derivative

Payload: `expr` (`Expression`), `var` (`String`), `order` (`u32`).

```json
{
  "kind": "Derivative",
  "value": {
    "expr": {
      "kind": "Function",
      "value": { "name": "f", "args": [ { "kind": "Variable", "value": "x" } ] }
    },
    "var": "x",
    "order": 1
  }
}
```

#### PartialDerivative

Payload: `expr` (`Expression`), `var` (`String`), `order` (`u32`).

```json
{
  "kind": "PartialDerivative",
  "value": {
    "expr": {
      "kind": "Function",
      "value": { "name": "f", "args": [
        { "kind": "Variable", "value": "x" },
        { "kind": "Variable", "value": "y" }
      ]}
    },
    "var": "x",
    "order": 1
  }
}
```

#### Integral

Payload: `integrand` (`Expression`), `var` (`String`), `bounds`
(`IntegralBounds | null`). `bounds` is `null` for indefinite integrals.
`IntegralBounds` has `lower` and `upper` fields (`Expression`).

```json
{
  "kind": "Integral",
  "value": {
    "integrand": { "kind": "Variable", "value": "x" },
    "var": "x",
    "bounds": {
      "lower": { "kind": "Integer", "value": 0 },
      "upper": { "kind": "Integer", "value": 1 }
    }
  }
}
```

#### MultipleIntegral

Payload: `dimension` (`u8`), `integrand` (`Expression`), `vars`
(`Vec<String>`), `bounds` (`MultipleBounds | null`). `MultipleBounds`
contains a `bounds` array of `IntegralBounds` objects, one per variable.

```json
{
  "kind": "MultipleIntegral",
  "value": {
    "dimension": 2,
    "integrand": {
      "kind": "Function",
      "value": { "name": "f", "args": [
        { "kind": "Variable", "value": "x" },
        { "kind": "Variable", "value": "y" }
      ]}
    },
    "vars": ["x", "y"],
    "bounds": {
      "bounds": [
        { "lower": { "kind": "Integer", "value": 0 }, "upper": { "kind": "Integer", "value": 1 } },
        { "lower": { "kind": "Integer", "value": 0 }, "upper": { "kind": "Integer", "value": 2 } }
      ]
    }
  }
}
```

#### ClosedIntegral

Payload: `dimension` (`u8`; 1 = line, 2 = surface, 3 = volume),
`integrand` (`Expression`), `surface` (`String | null`), `var` (`String`).

```json
{
  "kind": "ClosedIntegral",
  "value": {
    "dimension": 1,
    "integrand": { "kind": "Variable", "value": "f" },
    "surface": "S",
    "var": "s"
  }
}
```

#### Limit

Payload: `expr` (`Expression`), `var` (`String`), `to` (`Expression`),
`direction` (`Direction`).

`Direction` kind values: `"Both"` (two-sided), `"Left"`, `"Right"`.

```json
{
  "kind": "Limit",
  "value": {
    "expr": {
      "kind": "Function",
      "value": { "name": "f", "args": [ { "kind": "Variable", "value": "x" } ] }
    },
    "var": "x",
    "to": { "kind": "Integer", "value": 0 },
    "direction": { "kind": "Both" }
  }
}
```

#### Sum

Payload: `index` (`String`), `lower` (`Expression`), `upper`
(`Expression`), `body` (`Expression`).

```json
{
  "kind": "Sum",
  "value": {
    "index": "i",
    "lower": { "kind": "Integer", "value": 1 },
    "upper": { "kind": "Variable", "value": "n" },
    "body":  { "kind": "Variable", "value": "i" }
  }
}
```

#### Product

Same shape as `Sum`.

```json
{
  "kind": "Product",
  "value": {
    "index": "i",
    "lower": { "kind": "Integer", "value": 1 },
    "upper": { "kind": "Variable", "value": "n" },
    "body":  { "kind": "Variable", "value": "i" }
  }
}
```

---

### Collections

#### Vector

Payload: `Vec<Expression>` (the component list).

```json
{
  "kind": "Vector",
  "value": [
    { "kind": "Integer", "value": 1 },
    { "kind": "Integer", "value": 2 },
    { "kind": "Integer", "value": 3 }
  ]
}
```

#### Matrix

Payload: `Vec<Vec<Expression>>` (row-major; outer array is rows).

```json
{
  "kind": "Matrix",
  "value": [
    [ { "kind": "Integer", "value": 1 }, { "kind": "Integer", "value": 2 } ],
    [ { "kind": "Integer", "value": 3 }, { "kind": "Integer", "value": 4 } ]
  ]
}
```

---

### Equations and Inequalities

#### Equation

Payload: `left` and `right` (`Expression`).

```json
{
  "kind": "Equation",
  "value": {
    "left":  { "kind": "Variable", "value": "x" },
    "right": { "kind": "Integer",  "value": 5 }
  }
}
```

#### Inequality

Payload: `op` (`InequalityOp`), `left` and `right` (`Expression`).

`InequalityOp` kind values: `"Lt"` (<), `"Le"` (≤), `"Gt"` (>),
`"Ge"` (≥), `"Ne"` (≠).

```json
{
  "kind": "Inequality",
  "value": {
    "op": { "kind": "Lt" },
    "left":  { "kind": "Variable", "value": "x" },
    "right": { "kind": "Integer",  "value": 10 }
  }
}
```

---

### Quantifiers and Logic

#### ForAll

Payload: `variable` (`String`), `domain` (`Expression | null`), `body`
(`Expression`).

```json
{
  "kind": "ForAll",
  "value": {
    "variable": "x",
    "domain": { "kind": "NumberSetExpr", "value": { "kind": "Real" } },
    "body": {
      "kind": "Inequality",
      "value": {
        "op": { "kind": "Ge" },
        "left": {
          "kind": "Binary",
          "value": {
            "op": { "kind": "Mul" },
            "left":  { "kind": "Variable", "value": "x" },
            "right": { "kind": "Variable", "value": "x" }
          }
        },
        "right": { "kind": "Integer", "value": 0 }
      }
    }
  }
}
```

#### Exists

Payload: `variable` (`String`), `domain` (`Expression | null`), `body`
(`Expression`), `unique` (`bool`). Set `unique: true` for uniqueness
quantification (∃!).

```json
{
  "kind": "Exists",
  "value": {
    "variable": "x",
    "domain": { "kind": "NumberSetExpr", "value": { "kind": "Real" } },
    "unique": false,
    "body": {
      "kind": "Equation",
      "value": {
        "left": {
          "kind": "Binary",
          "value": {
            "op": { "kind": "Mul" },
            "left":  { "kind": "Variable", "value": "x" },
            "right": { "kind": "Variable", "value": "x" }
          }
        },
        "right": { "kind": "Integer", "value": 2 }
      }
    }
  }
}
```

#### Logical

Payload: `op` (`LogicalOp`), `operands` (`Vec<Expression>`).

`LogicalOp` kind values: `"And"`, `"Or"`, `"Not"`, `"Implies"`, `"Iff"`.

```json
{
  "kind": "Logical",
  "value": {
    "op": { "kind": "And" },
    "operands": [
      { "kind": "Variable", "value": "p" },
      { "kind": "Variable", "value": "q" }
    ]
  }
}
```

---

### Vector Operations

#### MarkedVector

Payload: `name` (`String`), `notation` (`VectorNotation`).

`VectorNotation` kind values: `"Bold"`, `"Arrow"`, `"Hat"`, `"Underline"`.

```json
{
  "kind": "MarkedVector",
  "value": {
    "name": "v",
    "notation": { "kind": "Arrow" }
  }
}
```

#### DotProduct

Payload: `left` and `right` (`Expression`).

```json
{
  "kind": "DotProduct",
  "value": {
    "left":  { "kind": "Variable", "value": "u" },
    "right": { "kind": "Variable", "value": "v" }
  }
}
```

#### CrossProduct

Same shape as `DotProduct`.

```json
{
  "kind": "CrossProduct",
  "value": {
    "left":  { "kind": "Variable", "value": "u" },
    "right": { "kind": "Variable", "value": "v" }
  }
}
```

#### OuterProduct

Same shape as `DotProduct`.

```json
{
  "kind": "OuterProduct",
  "value": {
    "left":  { "kind": "Variable", "value": "u" },
    "right": { "kind": "Variable", "value": "v" }
  }
}
```

---

### Linear Algebra

#### Determinant

Payload: `matrix` (`Expression`).

```json
{
  "kind": "Determinant",
  "value": {
    "matrix": {
      "kind": "Matrix",
      "value": [
        [ { "kind": "Integer", "value": 1 }, { "kind": "Integer", "value": 2 } ],
        [ { "kind": "Integer", "value": 3 }, { "kind": "Integer", "value": 4 } ]
      ]
    }
  }
}
```

#### Trace

Payload: `matrix` (`Expression`).

```json
{
  "kind": "Trace",
  "value": { "matrix": { "kind": "Variable", "value": "A" } }
}
```

#### Rank

Payload: `matrix` (`Expression`).

```json
{
  "kind": "Rank",
  "value": { "matrix": { "kind": "Variable", "value": "A" } }
}
```

#### ConjugateTranspose

Payload: `matrix` (`Expression`).

```json
{
  "kind": "ConjugateTranspose",
  "value": { "matrix": { "kind": "Variable", "value": "A" } }
}
```

#### MatrixInverse

Payload: `matrix` (`Expression`).

```json
{
  "kind": "MatrixInverse",
  "value": { "matrix": { "kind": "Variable", "value": "A" } }
}
```

---

### Set Theory

#### NumberSetExpr

Payload: `NumberSet` object. All `NumberSet` variants are unit variants.

| `kind` value     | Set     |
|------------------|---------|
| `"Natural"`      | ℕ       |
| `"Integer"`      | ℤ       |
| `"Rational"`     | ℚ       |
| `"Real"`         | ℝ       |
| `"Complex"`      | ℂ       |
| `"Quaternion"`   | ℍ       |

```json
{ "kind": "NumberSetExpr", "value": { "kind": "Real" } }
```

#### SetOperation

Payload: `op` (`SetOp`), `left` and `right` (`Expression`).

`SetOp` kind values: `"Union"`, `"Intersection"`, `"Difference"`.

```json
{
  "kind": "SetOperation",
  "value": {
    "op": { "kind": "Union" },
    "left":  { "kind": "Variable", "value": "A" },
    "right": { "kind": "Variable", "value": "B" }
  }
}
```

#### SetRelationExpr

Payload: `relation` (`SetRelation`), `element` (`Expression`), `set`
(`Expression`).

`SetRelation` kind values: `"In"`, `"NotIn"`, `"Subset"`, `"SubsetEq"`,
`"Superset"`, `"SupersetEq"`.

```json
{
  "kind": "SetRelationExpr",
  "value": {
    "relation": { "kind": "In" },
    "element": { "kind": "Variable", "value": "x" },
    "set": { "kind": "NumberSetExpr", "value": { "kind": "Real" } }
  }
}
```

#### SetBuilder

Payload: `variable` (`String`), `domain` (`Expression | null`),
`predicate` (`Expression`).

```json
{
  "kind": "SetBuilder",
  "value": {
    "variable": "x",
    "domain": { "kind": "NumberSetExpr", "value": { "kind": "Real" } },
    "predicate": {
      "kind": "Inequality",
      "value": {
        "op": { "kind": "Gt" },
        "left":  { "kind": "Variable", "value": "x" },
        "right": { "kind": "Integer",  "value": 0 }
      }
    }
  }
}
```

#### EmptySet

Unit variant. No `value` field.

```json
{ "kind": "EmptySet" }
```

#### PowerSet

Payload: `set` (`Expression`).

```json
{
  "kind": "PowerSet",
  "value": { "set": { "kind": "Variable", "value": "S" } }
}
```

---

### Tensor Notation

Each tensor variant uses a `TensorIndex` object with fields `name`
(`String`) and `index_type` (`IndexType`).

`IndexType` kind values: `"Upper"` (contravariant), `"Lower"` (covariant).

#### Tensor

Payload: `name` (`String`), `indices` (`Vec<TensorIndex>`).

```json
{
  "kind": "Tensor",
  "value": {
    "name": "T",
    "indices": [
      { "name": "i", "index_type": { "kind": "Upper" } },
      { "name": "j", "index_type": { "kind": "Lower" } }
    ]
  }
}
```

#### KroneckerDelta

Payload: `indices` (`Vec<TensorIndex>`).

```json
{
  "kind": "KroneckerDelta",
  "value": {
    "indices": [
      { "name": "i", "index_type": { "kind": "Lower" } },
      { "name": "j", "index_type": { "kind": "Lower" } }
    ]
  }
}
```

#### LeviCivita

Payload: `indices` (`Vec<TensorIndex>`).

```json
{
  "kind": "LeviCivita",
  "value": {
    "indices": [
      { "name": "i", "index_type": { "kind": "Lower" } },
      { "name": "j", "index_type": { "kind": "Lower" } },
      { "name": "k", "index_type": { "kind": "Lower" } }
    ]
  }
}
```

---

### Function Theory

#### FunctionSignature

Payload: `name` (`String`), `domain` (`Expression`), `codomain`
(`Expression`).

```json
{
  "kind": "FunctionSignature",
  "value": {
    "name": "f",
    "domain":   { "kind": "NumberSetExpr", "value": { "kind": "Real" } },
    "codomain": { "kind": "NumberSetExpr", "value": { "kind": "Real" } }
  }
}
```

#### Composition

Payload: `outer` (`Expression`), `inner` (`Expression`). Represents
`outer ∘ inner`.

```json
{
  "kind": "Composition",
  "value": {
    "outer": { "kind": "Function", "value": { "name": "f", "args": [] } },
    "inner": { "kind": "Function", "value": { "name": "g", "args": [] } }
  }
}
```

---

### Differential Forms

#### Differential

Payload: `var` (`String`).

```json
{ "kind": "Differential", "value": { "var": "x" } }
```

#### WedgeProduct

Payload: `left` and `right` (`Expression`).

```json
{
  "kind": "WedgeProduct",
  "value": {
    "left":  { "kind": "Differential", "value": { "var": "x" } },
    "right": { "kind": "Differential", "value": { "var": "y" } }
  }
}
```

---

### Vector Calculus

#### Gradient

Payload: `expr` (`Expression`).

```json
{
  "kind": "Gradient",
  "value": {
    "expr": {
      "kind": "Function",
      "value": { "name": "f", "args": [
        { "kind": "Variable", "value": "x" },
        { "kind": "Variable", "value": "y" }
      ]}
    }
  }
}
```

#### Divergence

Payload: `field` (`Expression`).

```json
{
  "kind": "Divergence",
  "value": { "field": { "kind": "Variable", "value": "F" } }
}
```

#### Curl

Payload: `field` (`Expression`).

```json
{
  "kind": "Curl",
  "value": { "field": { "kind": "Variable", "value": "F" } }
}
```

#### Laplacian

Payload: `expr` (`Expression`).

```json
{
  "kind": "Laplacian",
  "value": { "expr": { "kind": "Variable", "value": "f" } }
}
```

#### Nabla

Unit variant. The ∇ operator as a standalone symbol.

```json
{ "kind": "Nabla" }
```

---

### Relations

#### Relation

Payload: `op` (`RelationOp`), `left` and `right` (`Expression`).

`RelationOp` kind values: `"Approx"` (≈), `"Proportional"` (∝),
`"Congruent"` (≅), `"Similar"` (~).

```json
{
  "kind": "Relation",
  "value": {
    "op": { "kind": "Approx" },
    "left":  { "kind": "Variable", "value": "x" },
    "right": { "kind": "Variable", "value": "y" }
  }
}
```

---

## Stability Guarantee

Variant tag names (`kind` strings) are stable from v0.4.0 onward. Changes
to a tag name are breaking and require a coordinated major-version bump
with thales. The golden fixtures in `tests/fixtures/serde/` are machine-
checked on every CI run and serve as the binding contract.

Field names within a variant payload are likewise stable.

New variants may be added in minor releases. Consumers must handle unknown
`kind` values gracefully — typically by returning an error specific to the
consumer's domain rather than panicking.
