# mathlex — Mathcore Integration Specification

**Version:** draft-1 (2026-04-22)
**Status:** draft, pre-freeze
**Target release:** mathlex v0.5.0 (after MX-1..MX-10 land in v0.4.0)
**Companion document:** `thales_v0.9.0_requirements.md` (MX-1..MX-10)

This document specifies the second wave of mathlex work: active consumption
of unit and constant annotations to produce `AnnotatedExpression` (Expression
with unit/constant annotations resolved). These requirements are designated
MI-1..MI-N (Mathlex Integration).

The first wave (MX-1..MX-10) delivered the serde-capable `Expression` AST,
the `AnnotationSet` substrate, and variant-tag stability. That wave is a
prerequisite for every requirement here. Nothing in this document ships
before MX-1..MX-10 are complete.

---

## 1. Scope and Relationship to thales_v0.9.0_requirements.md

### 1.1 What the first wave (MX-1..MX-10) delivered

`thales_v0.9.0_requirements.md` covers the wire-format and substrate layer:

- **MX-1** — serde derives on `Expression`; lossless round-trip.
- **MX-2** — variant tag stability; golden-file regression tests.
- **MX-3** — `AnnotationSet` substrate per RFC §M-R1; opaque to thales v0.9.0.
- **MX-4** — existing parsers unchanged; empty `AnnotationSet` by default.
- **MX-5..MX-9** — derivative/ODE stability, no silent breaking changes,
  fixture files, documentation, version alignment.
- **MX-10** — unit propagation and constant resolution explicitly deferred.

The first wave is complete when mathlex ships v0.4.0 and thales ships v0.9.0.

### 1.2 What this wave (MI-1..MI-N) adds

This document defines how mathlex actively consumes unit and constant
annotations to validate, transform, and annotate the `Expression` tree. The
result is an `AnnotatedExpression` carrying:

- The parsed `Expression` tree (unchanged in shape from MX-1 parse output).
- A populated `AnnotationSet` per node, recording resolved `UnitExpression`
  for unit-bearing sub-expressions and `ConstantId` for named-constant
  symbols.
- A root-level `output_unit` giving the factored result unit of the whole
  expression after all conversions are applied.
- A `warnings` list for non-fatal annotation issues.

### 1.3 What this wave does NOT cover

- Numeric evaluation of conversion factors. That is mathlex-eval's and
  thales's responsibility.
- Substitution of constant numeric values. mathlex keeps symbols symbolic
  per thales Rule 2 (Expression is the contract) and thales Rule 1
  (Arc<Expr> internals). Downstream decides whether to substitute.
- Inline unit syntax. Annotations are always a caller-supplied map, never
  embedded in the source text (see § 3.3).
- Changes to `Expression` shape. The `Expression` AST is frozen per MX-2.
  `AnnotatedExpression` is a wrapper that adds the resolved annotation layer
  on top of the existing AST without modifying it.

---

## 2. ParseContext and API Surface

### 2.1 `ParseContext`

```rust
use mathcore_units::{UnitExpression, ConstantId, System};
use std::collections::HashMap;

/// Caller-supplied context for annotation-aware parsing.
pub struct ParseContext {
    /// Unit annotation for each symbol that is unit-bearing in the expression.
    /// Keys are the exact symbol strings as they appear in the source text.
    /// Values are fully formed `UnitExpression` trees (e.g., m/s, kg·m·s⁻²).
    /// The caller is responsible for constructing these; mathlex resolves tokens
    /// via `unitalg::parse_token` during the annotation-resolution phase (§ 4).
    /// Passing a pre-constructed `UnitExpression` directly is also accepted;
    /// string tokens are resolved at parse time and are an alternative entry path.
    pub unit_annotations: HashMap<String, UnitExpression>,

    /// Constant annotation for each symbol that names a known physical or
    /// mathematical constant. Keys match symbol strings in the source text.
    /// Values are `ConstantId` variants from mathcore-units § 4.
    pub constant_annotations: HashMap<String, ConstantId>,

    /// Preferred output system. If `Some(sys)`, all unit annotations are
    /// converted to `sys` via `unitalg::convert`; a conversion factor is
    /// inserted into the `Expression` tree as a symbolic multiplication.
    /// If `None`, the system is inferred by `unitalg::choose_system`.
    pub target_system: Option<System>,
}
```

`ParseContext` is cheap to clone (all keys are `String`; values are enums or
tree types without heap sharing). The default constructor returns a context
with empty maps and `target_system: None`, which produces behavior identical
to the pre-annotation mathlex parser.

### 2.2 `AnnotatedExpression`

```rust
use mathcore_units::UnitExpression;
use crate::{Expression, AnnotationSet};

/// Output of `parse_with_annotations`.
/// Wraps an `Expression` with the resolved annotation layer.
pub struct AnnotatedExpression {
    /// The parsed AST. Shape is identical to what `parse()` would return.
    /// Conversion factors inserted by target-system enforcement (§ 7) appear
    /// as `Expression::Mul` nodes wrapping the original sub-expressions.
    pub expression: Expression,

    /// Per-node and root-level annotations.
    /// Each node's `AnnotationSet` carries:
    ///   key "unit"     → UnitExpression (serialized per mathcore-units § 7)
    ///   key "constant" → ConstantId  (for symbol nodes mapped to a constant)
    /// Root-level annotations (index = root node id) additionally carry:
    ///   key "output_unit" → UnitExpression (the factored result unit)
    pub annotations: AnnotationSet,

    /// The factored result unit of the whole expression, or `None` when the
    /// expression is dimensionless or all annotations are absent.
    pub output_unit: Option<UnitExpression>,

    /// Non-fatal warnings generated during annotation processing.
    pub warnings: Vec<Warning>,
}
```

### 2.3 `Warning`

```rust
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", content = "value"))]
pub enum Warning {
    /// A symbol appeared in `unit_annotations` or `constant_annotations`
    /// but was not found anywhere in the parsed `Expression` tree.
    UnusedAnnotation { symbol: String },
}
```

Warnings are non-fatal. `parse_with_annotations` returns `Ok(AnnotatedExpression)`
even when warnings are present. Callers inspect `warnings` to surface
annotation hygiene issues to users.

### 2.4 Entry-point function

```rust
/// Parse `source` with annotation-aware resolution.
///
/// Steps (see § 3):
/// 1. Parse source text → Expression  (existing mathlex parse path, unchanged).
/// 2. Resolve unit annotation strings via unitalg::parse_token (§ 4).
/// 3. Resolve constant annotations via mathcore_constants::lookup_constant (§ 5).
/// 4. Run consistency checks (§ 6).
/// 5. Apply target-system enforcement (§ 7).
/// 6. Walk the Expression tree to compute and factor the output unit (§ 8).
/// 7. Assemble AnnotatedExpression (§ 9).
///
/// Returns `Err(ParseError)` only for hard failures: source text parse error,
/// annotation token parse failure, dimension mismatch, transcendental argument
/// error, unknown constant, or system incompatibility. Warnings are soft and
/// appear in `AnnotatedExpression::warnings`.
pub fn parse_with_annotations(
    source: &str,
    ctx: &ParseContext,
) -> Result<AnnotatedExpression, ParseError> { ... }
```

The existing `parse()` and `parse_latex()` entry points remain unchanged and
continue to return `Expression` without annotations. `parse_with_annotations`
is a new, additive entry point.

---

## 3. Parsing Pipeline with Annotations

### 3.1 Stage 1 — Text → Expression

Call the existing mathlex parser (LaTeX or plain text, detected from the
source format flag passed by the caller or auto-detected). This stage is
unchanged from the pre-annotation path (per MX-4). The returned `Expression`
may contain any variant; no annotation logic runs here.

Errors from this stage propagate immediately as `ParseError::Syntax`.

### 3.2 Stage 2 — Annotation token resolution

After Stage 1 succeeds, iterate `ctx.unit_annotations`. For each value:
- If the value is already a `UnitExpression` (the caller constructed it
  directly), accept it as-is.
- If the caller passed a string-form unit token (the alternate entry path
  `unit_annotations_raw: HashMap<String, String>` on `ParseContext`), call
  `unitalg::parse_token(token_str)` for each. Failures surface as
  `ParseError::AnnotationParseError`.

See § 4 for the token-resolution algorithm.

### 3.3 No inline unit syntax

mathlex does not introduce any grammar extension for inline unit annotation.
The caller always supplies annotations as a separate map. This decision
continues the policy established in the mathlex v0.4.0 RFC:

- `5 m/s` is parsed as `5 * m * s^-1`. The symbols `m` and `s` are
  `Expression::Variable` nodes. Their unit semantics are supplied by the
  annotation map, not by the source text.
- `[m/s]` bracket notation is NOT recognized as inline unit syntax. Bracket
  expressions are illegal unless the plain-text parser already supports them
  for another purpose.
- Constants such as `c`, `h`, `ℏ` are parsed as `Expression::Variable` nodes.
  The annotation map (via `constant_annotations`) declares their identity.
  No new grammar is introduced.

Rationale: grammar changes are breaking; annotation maps are additive. A
caller that does not supply a `ParseContext` gets the same parse result as
always (MX-4 backward-compat guarantee). This is the load-bearing design
decision.

### 3.4 Stage 3 — Constant annotation resolution

For each entry in `ctx.constant_annotations`, call
`mathcore_constants::lookup_constant(id)` to obtain the `ConstantSpec`. This
call never fails for valid `ConstantId` values (the mathcore-constants catalog
is exhaustive for v0.1.0, per MC-1). An `id` value that does not match any
`ConstantId` variant is a type error caught at compile time; no runtime
"unknown constant" case exists for well-typed callers.

The `ConstantSpec::unit` field (a `UnitExpression`) participates in
dimension checks alongside user-declared unit annotations (§ 6). See § 5 for
the full resolution algorithm.

### 3.5 Stage 4 — Consistency checks

Call the dimension check and transcendental-argument check algorithms. See § 6.

Failures surface as `ParseError` variants with source position information
drawn from the `Expression` node's position metadata (the position fields
threaded through the Expression AST in the existing parser).

### 3.6 Stage 5 — Target-system enforcement

If `ctx.target_system` is `Some(sys)`, convert all annotated unit values to
`sys`. See § 7.

### 3.7 Stage 6 — Output unit computation

Walk the `Expression` tree bottom-up, propagating unit annotations through
operators to derive the result unit of the root node. Call `unitalg::factor`
on the root unit. See § 8.

### 3.8 Stage 7 — Assembly

Populate `AnnotationSet` per node, set `output_unit`, collect `warnings`,
and return `AnnotatedExpression`. See § 9.

---

## 4. Unit Annotation Resolution

### 4.1 Token parsing

For the `unit_annotations_raw` path, each value string is passed to
`unitalg::parse_token(s)` (unitalg § 3.13). The parse_token algorithm:

1. Rejects known-reserved tokens (`"c"`, `"e"`, `"rem"`) with
   `ParseError::ReservedToken`.
2. Attempts direct alias match via `mathcore_units::alias::lookup_alias`.
3. Falls back to prefix-decomposition via `mathcore_units::alias::lookup_with_prefix`.
4. Checks `PrefixPolicy` admissibility for the resolved `(prefix, id)` pair.

On success, returns a `UnitExpression::Atom { id, prefix }`. Composite unit
strings (e.g., `"m/s"`, `"kg*m*s^-2"`) must be pre-parsed by the caller into
a `UnitExpression` tree using unitalg's expression-level parser (a function
separate from `parse_token`; see MI-12, open issue). For v0.5.0, callers
supply either single-token strings or pre-constructed `UnitExpression` trees.

Failures from `unitalg::parse_token` map to mathlex's
`ParseError::AnnotationParseError { symbol, token, source }`.

### 4.2 Symbol coverage check

After all `unit_annotations` values are resolved, walk the parsed `Expression`
tree collecting all `Expression::Variable` symbol strings. For each symbol `s`
in `ctx.unit_annotations` that does not appear in the collected set, append
`Warning::UnusedAnnotation { symbol: s }` to the warnings list. Do not raise
an error; unused annotations are a user-side hygiene issue, not a correctness
failure.

### 4.3 Annotation attachment

For each symbol in `ctx.unit_annotations` that appears in the `Expression`
tree, locate every `Expression::Variable(name)` node where `name == symbol`
and tag its `AnnotationSet` with key `"unit"` and value equal to the resolved
`UnitExpression` (serialized per mathcore-units § 7 wire format).

When the same symbol appears multiple times in the expression (e.g., `x + x`),
every occurrence receives the same unit annotation.

---

## 5. Constant Annotation Resolution

### 5.1 Lookup

For each `(symbol, id)` in `ctx.constant_annotations`:

1. Call `mathcore_constants::lookup_constant(id)` to obtain the
   `&'static ConstantSpec`. This is the only point in mathlex that imports
   `mathcore-constants`.
2. Locate every `Expression::Variable(name)` node in the tree where
   `name == symbol`.
3. Tag each such node's `AnnotationSet` with:
   - key `"constant"` → `id` (as `ConstantId`, serialized per
     mathcore-units § 4 wire format, i.e., the enum's tag string).
   - key `"unit"` → `spec.unit` (the constant's `UnitExpression`), which
     participates in dimension checks (§ 6) exactly as a user-declared unit
     annotation would.
4. If `symbol` does not appear in the `Expression` tree, append
   `Warning::UnusedAnnotation { symbol }`.

### 5.2 No numeric substitution

mathlex does NOT substitute `spec.value` into the `Expression` tree.
The `Expression::Variable(symbol)` node remains symbolic. The constant's
identity is recorded in the `AnnotationSet`; downstream consumers (thales
for symbolic computation, mathlex-eval for numeric evaluation) decide whether
and when to substitute the value.

This preserves thales Rule 1 (Arc<Expr> internals are `Arc<Expr>` exclusively)
and thales Rule 2 (Expression is the I/O contract): the Expression shape is
not modified by annotation resolution beyond the insertion of conversion-factor
multiplications required by target-system enforcement (§ 7).

### 5.3 Constant unit merging with user-declared units

When a symbol appears in both `ctx.unit_annotations` and
`ctx.constant_annotations`, the explicit user-declared unit takes precedence.
The constant's `spec.unit` is used for dimension checking but the annotation
key `"unit"` records the user-declared value. A warning is emitted if the
two units have different dimensions (this is almost always a user error):
`Warning::ConstantUnitConflict { symbol, declared_unit, constant_unit }` (see
§ 10 for the full error model; this is a warning variant).

---

## 6. Consistency Checks

All checks run after unit and constant annotations are resolved and attached.
Failures are hard errors returned as `ParseError` variants. The checks are
applied in the order listed.

### 6.1 Unused annotation check

Already performed in § 4.2 (unit) and § 5.1 (constant). Produces
`Warning::UnusedAnnotation`, not an error.

### 6.2 Additive site dimension check

Walk the `Expression` tree. For every `Expression::Add(lhs, rhs)` or
`Expression::Sub(lhs, rhs)` node:

1. Determine the effective unit of `lhs` and of `rhs` by looking up their
   resolved annotation in the per-node `AnnotationSet`. If a sub-expression
   has no unit annotation, it is treated as dimensionless.
2. Call `unitalg::combine_additive(u_lhs, u_rhs, ctx.target_system)`.
3. On `Err(DimError::IncompatibleDimensions { lhs: d1, rhs: d2 })`:
   - Surface as `ParseError::DimensionMismatch { op_pos, left_dim, right_dim }`.
   - `op_pos` is the source-position span of the operator node, obtained from
     the position metadata already present in the `Expression` AST.
4. On `Err(DimError::LogarithmicReferenceMismatch { .. })`:
   - Surface as `ParseError::LogarithmicMismatch { op_pos, lhs_ref, rhs_ref }`.
5. On `Ok(AdditiveResult)`:
   - If `factor_for_u1 != 1` or `factor_for_u2 != 1`: the operand(s) need a
     conversion. Wrap them in `Expression::Mul(factor, original_operand)`
     where `factor` is the symbolic conversion factor from `AdditiveResult`.
     This rewrites the tree in-place (the returned `expression` field of
     `AnnotatedExpression` will reflect this rewrite; see § 9.1).
   - Update the addition node's unit annotation to `AdditiveResult::unified_unit`.

This is the only stage where the `Expression` tree is rewritten. All other
stages only annotate; they do not modify node structure.

### 6.3 Transcendental argument check

For every function application node where the function name is one of `sin`,
`cos`, `tan`, `sec`, `csc`, `cot`, `arcsin`, `arccos`, `arctan`, `exp`,
`log`, `log10`, `log2`, `ln`:

1. Obtain the unit of the argument from its `AnnotationSet` key `"unit"`.
   If absent, treat as dimensionless (no check needed).
2. Call `unitalg::check_transcendental_argument(arg_unit)`.
3. On `Err(ArgError::DimensionedArgument { function, found })`:
   - Surface as `ParseError::TranscendentalArgumentError { fn_name, arg_dim, arg_pos }`.
   - `arg_pos` is the source span of the argument node.

Per unitalg § 3.12, `Radian` and `Degree` (dimension `{Angle: 1}`) and all
dimensionless units pass. Any other dimension fails.

### 6.4 Power with symbolic or non-integer exponent

For every `Expression::Pow(base, exponent)` node:

1. Obtain the unit of `base`.
2. If `exponent` is `Expression::Integer(n)`: call `unitalg::power(u_base, n)`.
   No dimension error is possible for integer powers.
3. If `exponent` is `Expression::Rational(num, den)`: call
   `unitalg::power_rational(u_base, num, den)`.
   On `Err(DimError::IncompatibleDimensions)`: surface as
   `ParseError::FractionalDimensionError { op_pos, base_unit }`.
4. If `exponent` is any other variant (a variable, a function call, etc.):
   - Require `u_base` to be dimensionless (empty Dimension). If it is not,
     append `Warning::SymbolicExponentWithUnit { op_pos, base_unit }`.
     This is a warning, not a hard error, because the CAS may still handle
     it symbolically in thales.

### 6.5 Incompleteness warning for addition sites

If a sub-expression appearing as an operand in `+` or `-` has no unit
annotation and the other operand does carry a unit annotation, emit
`Warning::UnannotatedAddend { op_pos, unannotated_side }`. This warns the
user that dimension checking was skipped for one arm of an additive expression,
which is a common source of latent errors.

---

## 7. Target-System Enforcement

Target-system enforcement runs after all consistency checks pass.

### 7.1 System selection

Call `unitalg::choose_system(all_unit_annotations, ctx.target_system)`:

- If `ctx.target_system = Some(sys)` and any annotation unit is incompatible
  with `sys`, return
  `ParseError::SystemIncompatible { target: sys, unresolvable_units: ... }`.
- If `ctx.target_system = None`, `choose_system` returns the inferred system
  (first-wins heuristic per unitalg § 5). If `NoCompatibleSystem`, that error
  bubbles up as `ParseError::SystemIncompatible`.

### 7.2 Conversion of annotation units

For each annotation unit `u` in the resolved `ctx.unit_annotations` (and the
constant-side `spec.unit` values), if `u`'s system differs from the selected
system `sys`:

1. Find the target unit `u_target` for `u`'s dimension in `sys` via
   `unitalg::choose_system(&[u], Some(sys))`.
2. Call `unitalg::convert(Expression::Integer(1), &u, &u_target)` to emit the
   symbolic conversion factor `f`.
3. For every `Expression::Variable(symbol)` node annotated with `u`, wrap it
   as `Expression::Mul(f.clone(), Expression::Variable(symbol))`.
4. Update the node's `AnnotationSet` key `"unit"` to `u_target`.

The returned `AnnotatedExpression::expression` field contains the rewritten
tree. This is the second (and last) stage that may rewrite the tree (the first
being additive site unification in § 6.2).

### 7.3 Constants and target system

Constants expressed in SI (as all `ConstantSpec::value` fields are) do not
have their numeric values converted by mathlex. mathlex tags the
`AnnotationSet` with a key `"target_system"` carrying the selected `System`
value, so downstream consumers (thales, mathlex-eval) know which system was
requested and can apply the conversion at evaluation time if needed. This
avoids numeric evaluation inside mathlex (Rule 1 of mathlex CLAUDE.md:
parser produces AST, never evaluates).

---

## 8. Output Unit Factoring

### 8.1 Bottom-up unit propagation

After all consistency checks and target-system conversions are applied, walk
the `Expression` tree bottom-up. For each node, compute its effective unit by
applying the following rules:

| Node type | Effective unit |
|---|---|
| `Variable(s)` | Look up `AnnotationSet["unit"]`; if absent, dimensionless. |
| `Integer`, `Float`, `Rational` | Dimensionless. |
| `Mul(a, b)` | `unitalg::multiply(unit(a), unit(b))` |
| `Div(a, b)` | `unitalg::divide(unit(a), unit(b))` |
| `Pow(base, Integer(n))` | `unitalg::power(unit(base), n)` |
| `Pow(base, Rational(p,q))` | `unitalg::power_rational(unit(base), p, q)` |
| `Add(a, b)`, `Sub(a, b)` | `unified_unit` from the additive result (§ 6.2); both sides must carry the same unit after unification. |
| `Neg(a)` | `unit(a)` |
| `sin`, `cos`, `tan`, `exp`, `ln`, `log` | Dimensionless (transcendental functions return dimensionless results). |
| `sqrt(a)` | `unitalg::power_rational(unit(a), 1, 2)` |
| `Derivative(expr, var, _)` | `unitalg::divide(unit(expr), unit(var))` |
| `Integral(expr, var, _, _)` | `unitalg::multiply(unit(expr), unit(var))` |
| All other nodes | Dimensionless (conservative default; annotate with a warning if a unit-bearing sub-expression is present). |

### 8.2 Root unit factoring

After propagation, obtain the unit of the root node. Call
`unitalg::factor(root_unit)` to fold the base-unit product to the most
specific named unit available in the catalog (per unitalg § 3.7). Assign the
result to `AnnotatedExpression::output_unit`.

If the root unit is dimensionless (empty `Dimension`), set `output_unit` to
`None`.

### 8.3 Per-node annotation population

Store the effective unit of each node in its `AnnotationSet` under key
`"unit"`. This completes the annotation population pass. Nodes that were
already annotated by § 4 (Variable nodes with direct unit annotations) already
have this key; the propagation pass fills in all remaining nodes that derive
their unit from their operands.

---

## 9. Wire Format

### 9.1 Expression tree rewrite policy

The `expression` field of `AnnotatedExpression` is the primary `Expression`
AST. Its shape is based on the output of Stage 1 (text parse), with two
possible categories of structural addition:

1. **Conversion multiplications** inserted at additive sites where unit
   conversion is needed (§ 6.2, `AdditiveResult`). These appear as
   `Expression::Mul(conversion_factor, original_operand)` nodes wrapping the
   operand that required scale adjustment.
2. **Target-system conversion multiplications** inserted when a variable's
   annotation unit does not match the selected system (§ 7.2). These also
   appear as `Expression::Mul(conversion_factor, Expression::Variable(sym))`
   nodes.

The `Expression` type is not modified; `Expression::Mul` is an existing
variant. The tree is structurally valid and round-trips through serde
unchanged (MX-1 guarantee applies).

### 9.2 AnnotationSet payload conventions

The `AnnotationSet` on each node is a map from string keys to typed payloads.
The keys defined by this specification:

| Key | Payload type | Scope | Notes |
|---|---|---|---|
| `"unit"` | `UnitExpression` (mathcore-units § 2.8 wire format) | Per node | Set on Variable nodes by § 4.3; propagated to all nodes by § 8.3 |
| `"constant"` | String (serialized `ConstantId` tag) | Per node | Set on Variable nodes by § 5.1 |
| `"output_unit"` | `UnitExpression` | Root node only | Set by § 8.2; same value as `AnnotatedExpression::output_unit` |
| `"target_system"` | String (serialized `System` tag) | Root node only | Set by § 7.3 |

All annotation payloads are stored as JSON-compatible values. `UnitExpression`
serializes per mathcore-units § 7 wire format (tag-and-content convention).
`ConstantId` and `System` serialize as their variant tag strings.

### 9.3 JSON example

Expression `m * a` with `m: Kilogram`, `a: Meter/Second²`. The root `Mul`
node's propagated unit folds to `Newton` via `unitalg::factor`.

```json
{
  "expression": {
    "kind": "Mul",
    "value": {
      "left":  {"kind": "Variable", "value": {"name": "m"},
                "annotations": {"unit": {"kind": "Atom", "value": {"id": "Kilogram", "prefix": null}}}},
      "right": {"kind": "Variable", "value": {"name": "a"},
                "annotations": {"unit": {"kind": "Binary", "value": {
                  "op": {"kind": "Div"},
                  "left":  {"kind": "Atom", "value": {"id": "Meter", "prefix": null}},
                  "right": {"kind": "Binary", "value": {
                    "op":    {"kind": "Pow"},
                    "left":  {"kind": "Atom",    "value": {"id": "Second", "prefix": null}},
                    "right": {"kind": "Literal", "value": "2"}}}}}}},
      "annotations": {
        "unit":        {"kind": "Atom", "value": {"id": "Newton", "prefix": null}},
        "output_unit": {"kind": "Atom", "value": {"id": "Newton", "prefix": null}}
      }
    }
  },
  "output_unit": {"kind": "Atom", "value": {"id": "Newton", "prefix": null}},
  "warnings": []
}
```

### 9.4 Serde feature gating

`AnnotatedExpression` serde derives are gated behind the `serde` feature,
consistent with MX-1. All types referenced by `AnnotatedExpression`
(`UnitExpression`, `ConstantId`, `System`) require `mathcore-units/serde`
to be enabled. The feature dependency chain:

```toml
[features]
serde = [
    "dep:serde",
    "mathcore-units/serde",
    "mathcore-constants/serde",
    "unitalg/serde",
]
```

Enabling `mathlex/serde` without also enabling `mathcore-units/serde` is a
compile error (the annotation payload types are not independently serde-capable).

---

## 10. Error Model

All errors from annotation processing are variants of mathlex's existing
`ParseError` enum. The following new variants are added in v0.5.0. All
variants are `#[non_exhaustive]`; future minor versions may add variants.

```rust
// Additions to the existing ParseError enum:

/// A unit token in `ParseContext::unit_annotations_raw` could not be resolved
/// by `unitalg::parse_token`.
AnnotationParseError {
    /// The symbol for which the annotation was declared.
    symbol: String,
    /// The token string that failed to parse.
    token: String,
    /// The underlying `unitalg::ParseError`.
    source: unitalg::ParseError,
},

/// An additive operator (`+` or `-`) has operands with incompatible dimensions.
DimensionMismatch {
    /// Source span of the operator node.
    op_pos: SourceSpan,
    /// Dimension of the left operand.
    left_dim: mathcore_units::Dimension,
    /// Dimension of the right operand.
    right_dim: mathcore_units::Dimension,
},

/// An additive operator mixes a logarithmic unit (dB family) with a linear
/// unit of the same underlying dimension (e.g., dBm + Watt).
LogarithmicMismatch {
    op_pos: SourceSpan,
    lhs_ref: Option<mathcore_units::UnitId>,
    rhs_ref: Option<mathcore_units::UnitId>,
},

/// A transcendental function received an argument with a non-dimensionless,
/// non-angle unit (e.g., sin(x) where x has unit Meter).
TranscendentalArgumentError {
    fn_name: String,
    arg_dim: mathcore_units::Dimension,
    arg_pos: SourceSpan,
},

/// A `Pow` node has a rational exponent that would produce fractional
/// dimension exponents (e.g., sqrt(Newton) → {L:0.5, M:0.5, T:-1}).
FractionalDimensionError {
    op_pos: SourceSpan,
    base_unit: mathcore_units::UnitExpression,
},

/// The requested `target_system` cannot accommodate one or more of the
/// annotation units (e.g., target = Imperial, unit = Tesla).
SystemIncompatible {
    target: mathcore_units::System,
    unresolvable_units: Vec<mathcore_units::UnitExpression>,
},
```

The following are `Warning` variants (non-fatal, returned in
`AnnotatedExpression::warnings`):

```rust
/// A symbol in `unit_annotations` or `constant_annotations` did not appear
/// in the parsed `Expression` tree.
Warning::UnusedAnnotation { symbol: String },

/// An addend in `+` or `-` had no unit annotation while the other addend did.
Warning::UnannotatedAddend { op_pos: SourceSpan, unannotated_side: Side },

/// A `Pow` node had a symbolic (non-literal) exponent and the base had a
/// non-dimensionless unit. The check was skipped; thales will handle this.
Warning::SymbolicExponentWithUnit {
    op_pos: SourceSpan,
    base_unit: mathcore_units::UnitExpression,
},

/// A symbol appeared in both `unit_annotations` and `constant_annotations`;
/// the explicit user-declared unit took precedence, but the constant's
/// cataloged unit has a different dimension.
Warning::ConstantUnitConflict {
    symbol: String,
    declared_unit: mathcore_units::UnitExpression,
    constant_unit: mathcore_units::UnitExpression,
},
```

---

## 11. Backward Compatibility

### 11.1 Existing parse paths unchanged

The existing `parse(source: &str) -> Result<Expression, ParseError>` and
`parse_latex(source: &str) -> Result<Expression, ParseError>` entry points
are unchanged. They return a plain `Expression` with an empty `AnnotationSet`
on every node, exactly as in v0.4.0. No caller that uses the pre-annotation
API is affected by this wave of work.

This guarantee is the continuation of the MX-4 commitment. Specifically:

- No existing test fixture in mathlex's test suite requires modification.
- The `Expression` enum gains no new variants in v0.5.0 for the annotation
  integration (the existing `Variable`, `Mul`, etc. are sufficient).
- `AnnotationSet` remains present on every `Expression` node as an empty set
  when `parse()` is called, exactly as established in v0.4.0.

### 11.2 `AnnotatedExpression` is a new type

`AnnotatedExpression` is additive. It does not replace `Expression`; it wraps
it. Consumers that have not migrated to `parse_with_annotations` continue to
work without any changes.

### 11.3 MI-N note

Every requirement tagged `Blocker` in § 13 gates mathlex v0.5.0. Requirements
tagged `Required` must ship in v0.5.0 alongside the Blocker items.
Requirements tagged `Deferred` are explicitly out of scope for v0.5.0 and
tracked for v0.6.0 or later.

---

## 12. Test Strategy

### 12.1 Unit annotation resolution tests

For each `unitalg::ParseError` variant, a test in mathlex's test suite calls
`parse_with_annotations` with a deliberately malformed unit annotation and
asserts the correct `ParseError::AnnotationParseError` is returned. Cover at
minimum: `UnknownToken`, `PrefixWithoutAtom`, `ReservedToken`,
`PrefixNotAllowed`.

### 12.2 Dimension check tests

For each dimension-check path (additive site, transcendental argument,
rational power):

- One positive test: annotation-consistent expression returns `Ok`.
- One negative test: annotation-inconsistent expression returns the
  correct `ParseError` variant with accurate `op_pos` or `arg_pos`.

Representative cases:

```rust
// Dimension mismatch: adding meters to seconds
parse_with_annotations(
    "x + y",
    &ParseContext {
        unit_annotations: [("x", meter()), ("y", second())],
        ..Default::default()
    },
) == Err(ParseError::DimensionMismatch { left_dim: {Length:1}, right_dim: {Time:1}, .. })

// Compatible additive: meters and feet → conversion factor inserted
parse_with_annotations(
    "x + y",
    &ParseContext {
        unit_annotations: [("x", meter()), ("y", foot())],
        target_system: Some(System::SI),
        ..Default::default()
    },
) == Ok(AnnotatedExpression { output_unit: Some(Meter), warnings: [], .. })
// expression.right is wrapped: Mul(Rational(3048, 10000), Variable("y"))

// Transcendental: sin(x_meters) fails
parse_with_annotations(
    "sin(x)",
    &ParseContext {
        unit_annotations: [("x", meter())],
        ..Default::default()
    },
) == Err(ParseError::TranscendentalArgumentError { fn_name: "sin", arg_dim: {Length:1}, .. })

// Transcendental: sin(theta_rad) passes
parse_with_annotations(
    "sin(theta)",
    &ParseContext {
        unit_annotations: [("theta", radian())],
        ..Default::default()
    },
) == Ok(..)
```

### 12.3 Constant annotation tests

- Positive: `c` annotated as `ConstantId::SpeedOfLight` produces a
  `AnnotationSet` on the Variable node with `"constant": "SpeedOfLight"` and
  `"unit"` matching the `ConstantSpec::unit` for `SpeedOfLight`.
- Negative: symbol in `constant_annotations` not present in expression
  produces `Warning::UnusedAnnotation`.
- No-substitute: verify that `Expression::Variable("c")` is NOT replaced by
  the numeric value `299792458`; the Variable node is preserved.

### 12.4 Target-system enforcement tests

- Mixed SI + Imperial in an additive expression with `target_system: Some(SI)`:
  verify the Imperial operand is wrapped in a conversion factor.
- `target_system: Some(Imperial)` with a Tesla annotation:
  verify `ParseError::SystemIncompatible`.
- No target system with all-SI annotations:
  verify `choose_system` selects SI and no conversions are emitted.

### 12.5 Output unit tests

- `F = m * a` with `m: Kilogram`, `a: Meter/Second²`:
  `output_unit = Some(Newton)` (factored by unitalg).
- `v = d / t` with `d: Meter`, `t: Second`:
  `output_unit = Some(Atom { id: Meter, prefix: None }) / Atom { id: Second, prefix: None }`
  — no named unit match, stays as base-unit product.
- `E = m * c^2` with `m: Kilogram`, `c` annotated as `SpeedOfLight`
  (constant, `unit: m/s`):
  effective unit of `c^2` = `m²/s²`; `m * c^2` → `kg·m²·s⁻²` = `Joule`.

### 12.6 Integration tests (cross-crate)

Integration tests live under `mathlex/tests/integration/` with
`dev-dependencies` on `unitalg`, `mathcore-units`, and `mathcore-constants`.

Required cases:

- **Solar mass additive**: `1.5 * M_sun + 0.3 * M_earth` (both annotated as
  constants and mass units). Dimension check passes; `FromConstant` conversion
  factors emitted; `output_unit = Kilogram`.
- **Speed-of-light energy**: `m * c^2`, `m: Kilogram`, `c: SpeedOfLight`
  constant. `output_unit = Joule`.
- **dBm + Watt**: `x + y`, `x: DecibelMilliwatt`, `y: Watt`.
  `ParseError::LogarithmicMismatch`.
- **Temperature conversion**: `T1 + T2`, `T1: DegreeCelsius`,
  `T2: DegreeFahrenheit`, `target_system: Some(SI)`. Affine factors emitted;
  `output_unit = Kelvin`.

### 12.7 Backward compatibility regression tests

Run the full existing mathlex test suite (all tests that existed before
v0.5.0) against the v0.5.0 build. Zero failures permitted. This is a CI gate.
Additionally, run the MX-7 golden fixture round-trip tests to confirm the
`Expression` serde format is unchanged.

### 12.8 Unused annotation warning tests

- Annotate a symbol that does not appear in the expression: verify exactly one
  `Warning::UnusedAnnotation` entry.
- Annotate all symbols that appear in the expression: verify zero warnings.

---

## 13. Requirements (MI-1..MI-N)

| ID | Requirement | Severity |
|---|---|---|
| MI-1 | `parse_with_annotations(source, ctx)` is a new public entry point returning `Result<AnnotatedExpression, ParseError>`; the existing `parse()` and `parse_latex()` entry points are unchanged | Blocker |
| MI-2 | `ParseContext` carries `unit_annotations: HashMap<String, UnitExpression>`, `constant_annotations: HashMap<String, ConstantId>`, and `target_system: Option<System>` | Blocker |
| MI-3 | Unit annotation tokens are resolved via `unitalg::parse_token`; failures surface as `ParseError::AnnotationParseError` carrying the symbol, the token, and the underlying `unitalg::ParseError` | Blocker |
| MI-4 | Symbols in `unit_annotations` that do not appear in the `Expression` tree emit `Warning::UnusedAnnotation` (not an error) | Required |
| MI-5 | Every `Expression::Variable(s)` node where `s` appears in `unit_annotations` receives a `AnnotationSet["unit"]` entry with the resolved `UnitExpression` | Blocker |
| MI-6 | `mathcore_constants::lookup_constant(id)` is called for each entry in `constant_annotations`; the resulting `ConstantSpec::unit` participates in dimension checks | Blocker |
| MI-7 | Constant-annotated Variable nodes receive `AnnotationSet["constant"]` with the serialized `ConstantId` tag; their `AnnotationSet["unit"]` is set from `ConstantSpec::unit` | Blocker |
| MI-8 | mathlex does NOT substitute `ConstantSpec::value` into the `Expression` tree; Variable nodes remain symbolic | Blocker |
| MI-9 | Additive (`+`/`-`) sites call `unitalg::combine_additive`; incompatible dimensions produce `ParseError::DimensionMismatch`; compatible but different-unit cases wrap the out-of-system operand in a symbolic `Expression::Mul` conversion | Blocker |
| MI-10 | Transcendental function sites (`sin`, `cos`, `tan`, `exp`, `log`, `ln`, and equivalents) call `unitalg::check_transcendental_argument` on the argument's resolved unit; non-dimensionless non-angle units produce `ParseError::TranscendentalArgumentError` | Blocker |
| MI-11 | Rational-exponent `Pow` sites call `unitalg::power_rational`; fractional dimension results produce `ParseError::FractionalDimensionError`; symbolic exponents with unit-bearing bases produce `Warning::SymbolicExponentWithUnit` | Required |
| MI-12 | `ctx.target_system = Some(sys)` causes all annotation units incompatible with `sys` to be converted via `unitalg::convert`; the conversion factor is inserted as a symbolic `Expression::Mul` node and the annotation is updated to the target-system unit | Blocker |
| MI-13 | `ParseError::SystemIncompatible` is returned when `target_system` is set and at least one annotation unit cannot be converted to that system | Blocker |
| MI-14 | Bottom-up unit propagation walks the entire `Expression` tree after checks and conversions; each node receives an `AnnotationSet["unit"]` entry derived from its operator and its children's units | Required |
| MI-15 | `unitalg::factor` is called on the root node's propagated unit; the result is stored in `AnnotatedExpression::output_unit` and in the root node's `AnnotationSet["output_unit"]` | Required |
| MI-16 | `AnnotatedExpression` carries `expression: Expression`, `annotations: AnnotationSet`, `output_unit: Option<UnitExpression>`, and `warnings: Vec<Warning>` | Blocker |
| MI-17 | All new `ParseError` variants use `#[non_exhaustive]` and `#[serde(tag = "kind", content = "value")]` when the `serde` feature is enabled | Required |
| MI-18 | The `serde` feature for `AnnotatedExpression` requires `mathcore-units/serde`, `mathcore-constants/serde`, and `unitalg/serde` to be co-enabled; enabling `mathlex/serde` alone without them is a compile error | Required |
| MI-19 | All existing parse paths (`parse()`, `parse_latex()`, etc.) return `Expression` with an empty `AnnotationSet` on every node; no existing test fixture requires modification for v0.5.0 | Blocker |
| MI-20 | CI integration tests cover: (a) dimension mismatch at additive site, (b) transcendental argument with dimensional unit, (c) constant annotation with no substitution, (d) target-system conversion factor insertion, (e) output unit factored to a named unit, (f) backward-compat golden fixture round-trip | Blocker |

---

## 14. Resolved Decisions and Open Issues

### 14.1 Resolved decisions

**D-1: No inline unit grammar.** The decision from the v0.4.0 RFC period is
confirmed: annotations are always caller-supplied maps; mathlex introduces
no bracket syntax or other inline annotation grammar. Rationale: grammar
additions are breaking changes; the map API is purely additive and keeps
existing parse inputs working without modification.

**D-2: No numeric substitution at parse time.** mathlex keeps constants
symbolic. The `Expression::Variable(sym)` node for a constant survives into
the output AST unchanged. thales Rule 1 and mathlex Rule 1 (parser produces
AST, never evaluates) both require this. Downstream consumers substitute when
they decide to evaluate.

**D-3: `AnnotatedExpression` is a wrapper, not a replacement.** `Expression`
shape is frozen by MX-2. `AnnotatedExpression` is a new struct that holds
`Expression` and adds the resolution layer. No new `Expression` variants are
needed for v0.5.0.

**D-4: Conversion factors use `Expression::Mul` (existing variant).** Tree
rewrites for unit conversion insert standard `Expression::Mul` nodes carrying
a `Rational` or `Variable` literal as the conversion factor. No new
`Expression` variant is needed.

**D-5: Additive site with `LogarithmicReferenceMismatch` is a hard error.**
Mixing `dBm + Watt` is rejected at parse time. It is not a warning because
the CAS cannot represent the result symbolically without unwrapping the
logarithmic relationship, which is thales's responsibility, not mathlex's. If
the caller intends a purely symbolic expression without dimensional semantics,
they should not supply unit annotations for those symbols.

**D-6: `Warning::UnannotatedAddend` is a warning, not an error.** An
unannotated addend might legitimately be a dimensionless constant. Forcing
annotation on all addends would break expressions like `E = mc^2 + 0` where
the `0` is unannotated. A warning surfaces the issue without hard-failing
legitimate expressions.

**D-7: `mathcore-constants` is a direct dependency of mathlex starting in
v0.5.0.** The `lookup_constant` call in § 5.1 requires the dependency.
In v0.4.0, mathlex had no dependency on mathcore-constants (constant
annotation was deferred per MX-10). Adding this dependency in v0.5.0 is a
`[dependencies]` addition; it is not a breaking change to the public API.

**D-8: `unitalg` is a direct dependency of mathlex starting in v0.5.0.**
Same rationale as D-7. `parse_token`, `combine_additive`,
`check_transcendental_argument`, `convert`, `factor`, and `choose_system` are
all called by the annotation pipeline. In v0.4.0 mathlex had no dependency on
unitalg.

### 14.2 Open issues

**MI-ISSUE-1: Composite unit token parsing API in unitalg.**
`unitalg::parse_token` handles single atomic tokens. Composite unit strings
such as `"m/s"` or `"kg*m*s^-2"` are not handled by `parse_token`. The
`unit_annotations_raw` path in `ParseContext` (§ 4.1) promises that callers
can pass string tokens for resolution, but for composite strings this requires
an expression-level unit parser in unitalg that does not yet exist in the
unitalg specification. Options: (a) restrict `unit_annotations_raw` to single
tokens only and require callers to pre-construct `UnitExpression` trees for
composite units; (b) add a `parse_unit_expression(s: &str)` function to
unitalg that tokenizes and builds a `UnitExpression` tree. Option (a) is
conservative; option (b) is the right long-term answer but requires a unitalg
spec amendment. This issue must be resolved before the v0.5.0 API is frozen.
**Blocking for MI-3 acceptance criteria.**

**MI-ISSUE-2: `AnnotationSet` key type for `UnitExpression` payload.**
The `AnnotationSet` substrate from MX-3 carries opaque `serde_json::Value`
payloads keyed by strings. When mathlex v0.5.0 populates `"unit"` keys with
`UnitExpression` values, the payload is a strongly typed `UnitExpression`
on the Rust side but a JSON object on the wire. The question is whether
`AnnotationSet` should carry typed variant payloads (via an enum) or continue
as a string-keyed `serde_json::Value` map. A typed enum would allow compile-time
correctness checking; a value map is more flexible for future extension.
The decision affects the `AnnotationSet` type shape defined in MX-3, which is
already shipped in v0.4.0. Changing the shape would be a breaking change.
This must be resolved in coordination with thales before v0.5.0 scope is
locked. **Non-blocking for v0.5.0 implementation start but must be resolved
before the v0.5.0-rc wire format is considered provisional-final.**

**MI-ISSUE-3: `SourceSpan` type in new `ParseError` variants.**
The new `ParseError` variants (`DimensionMismatch`, `TranscendentalArgumentError`,
etc.) carry a `SourceSpan` field encoding the source position of the error
site. The current mathlex `ParseError` type uses a position representation
that may need to be exposed as a public type alias or struct for the new
variants. If the existing position type is internal, it must be made public
in v0.5.0. This is a minor API addition but must be confirmed with the
mathlex CLAUDE.md backward-compat policy. **Non-blocking; resolved during
v0.5.0 PRD authoring.**
