# Thales v0.9.0 Requirements for mathlex

This document enumerates the capabilities thales v0.9.0 requires from
mathlex. The mathlex team will author its own PRD against this
requirements list; thales does not specify mathlex's internal design,
only the contract it must deliver and the stability guarantees it must
commit to.

This file is a sibling of `thales_requirements.md`, which captured the
v0.7.0-era parser gap list. Those items remain valid and are not
superseded here. This file scopes strictly to what thales v0.9.0
needs in addition, which is primarily serde wire-format work and
annotation substrate plumbing.

## Context

thales v0.8.1 closed the shape of its public API: a single entry
point (`execute`), a single output shape (`Response` with a map of
Expression results, each packaged with resolution steps, narratives,
and difficulty levels), and an FFI transport (`execute_ffi`) that
round-trips through JSON. Internally thales uses `Arc<Expr>` as its
computational representation and `Expression` (the mathlex type) at
the I/O boundary.

thales v0.9.0 promotes every NotImplemented stub in the dispatch
table to real engine wiring, collapses the remaining ~45 legacy
`*_ffi` functions down to the single `execute_ffi`, and makes the
Request/Response JSON transport round-trip through serde directly
instead of through a string-encoded Expression adapter.

The final step above is where mathlex comes in. thales currently
serializes Expression by emitting it as a LaTeX string and parsing
that string back on the far side. This works but loses information
and cannot carry annotations. v0.9.0 retires that adapter and
consumes `Expression` as a serde-capable type directly. That
requires mathlex to ship serde derives on `Expression` and every
type transitively referenced inside it, with a stable wire format.

Concurrently, the RFC at `thales/docs/RFC-mathlex-annotations.md`
specifies an `AnnotationSet` substrate (M-R1) that thales wants
threaded through the AST in v0.9.0 even though thales does not
consume annotations until v0.10.0. Landing the substrate shape early
means v0.10.0 is a pure extension, not a breaking wire-format change.

## Severity

- **Blocker**: thales v0.9.0 cannot ship without this.
- **Required**: Must ship with mathlex v0.4.0 alongside thales v0.9.0.
- **Deferred**: Explicitly out of scope for mathlex v0.4.0; tracked
  for the v0.10.0 window.

---

## MX-1 — Serde-capable Expression AST [Blocker]

mathlex must derive `serde::Serialize` and `serde::Deserialize` on
its public `Expression` type and every transitively referenced type
required to encode a complete Expression tree. At minimum this
covers:

- `Expression` (the sum type itself)
- `Variable`
- `Function` (including both named builtins and user-defined
  variants if applicable)
- `BinaryOp` and `UnaryOp` (the operator enums)
- Any constant / literal carrier types (rational, integer, real,
  complex, symbolic constant id)
- Any matrix, vector, tuple, list, or set container types exposed
  inside Expression
- Any position or metadata wrappers that are part of the public
  Expression shape

The derived representation must round-trip losslessly:
`from_json(to_json(x)) == x` holds for every inhabitant of the AST.
If mathlex uses non-serde-native types internally (for example `Rc`,
custom arena indices, interned-string handles), the serde
implementation must project them onto a serializable form at the
public boundary without requiring the consumer to know the internal
representation.

### Enum tag style

Every public enum in the AST must use the same tag-and-content style
uniformly. The concrete choice is mathlex's, but it must be either
`#[serde(tag = "op", content = "args")]` or
`#[serde(tag = "kind", content = "value")]` applied everywhere.
Untagged enums and per-variant ad-hoc shapes are not acceptable:
thales decoders (Rust on the receive side, Swift on the cross-FFI
side) must see one predictable JSON shape per enum.

### Numeric literal precision

Numeric literals must serialize in a form that preserves exactness.
Floats serialize as a string when finite precision would lose
information. Rationals serialize as a structured object (for
example `{ "num": "…", "den": "…" }`) with string payloads so
arbitrary-precision integers survive. Complex numbers serialize
with explicit real and imaginary components, each following the
rational or float rules above.

The concrete schema is mathlex's choice but must be documented
(see MX-8) and must not lose precision on any round-trip.

---

## MX-2 — Variant tag stability [Blocker]

Once thales v0.9.0 ships with mathlex as a serde producer and
consumer, every serialized Expression JSON document becomes a
wire-format artifact. Swift callers retain these documents in app
state, and downstream storage (caches, notebooks, server-side
persistence) may hold them indefinitely. mathlex must therefore
freeze the string tag for every public enum variant at the mathlex
v0.4.0 release.

Specifically:

- Adding new variants is permitted. This is a minor version bump on
  mathlex and does not force a thales bump unless thales needs to
  consume the new variant.
- Renaming existing variants is a breaking change and requires a
  major version bump on mathlex, coordinated with a thales major
  bump. No unilateral renames.
- Reordering variants in the source enum must not change the
  serialized tag. Serde uses the variant name, not the
  discriminant, so this is free in practice. mathlex must assert
  it with a golden-file test (see MX-7) to prevent accidental
  regressions from discriminant-based code elsewhere in mathlex.

A single `#[test]` in mathlex that asserts the JSON shape of every
variant via a golden fixture file is sufficient. thales will add a
symmetric decoder test on its consumer side. The two tests together
guarantee round-trip parity.

---

## MX-3 — AnnotationSet substrate (M-R1 from RFC) [Required]

thales does not consume units, constants as runtime values, or
domain narrowing from annotations in v0.9.0. M-R2 through M-R4 of
the annotations RFC stay deferred to the v0.10.0 window. thales
does, however, want the substrate shape ready in v0.9.0 so that
v0.10.0 work is a pure extension, not a breaking wire-format change.

mathlex must land the `AnnotationSet` substrate as specified in
`docs/RFC-mathlex-annotations.md` §M-R1. Concretely:

- Every `Expression` node carries an `AnnotationSet` (possibly
  empty) as a sidecar field. The field name and its position in
  the serialized JSON must be reserved in v0.9.0-compatible
  releases even when the set is empty.
- `AnnotationSet` is itself serde-capable. This requirement
  inherits from MX-1.
- The set is opaque to thales v0.9.0. thales reads it, preserves
  it across Arc<Expr> round-trips, and writes it back unchanged on
  decompile. No dispatch arm in thales v0.9.0 consumes annotations.
- Deserializing a missing `AnnotationSet` field must default to the
  empty set. This keeps pre-substrate payloads decoding correctly
  and is the backward-compat hinge with MX-4.

A concrete consequence: thales v0.9.0 extends its `compile` and
`decompile` passes in `src/numeric/compile.rs` (or its split
successor) to thread the `AnnotationSet` through as an attribute on
the `Arc<Expr>` node equivalent without yet using it. That thales
work depends on MX-3 landing upstream first.

### Substrate-only scope

What MX-3 does NOT require in v0.4.0:

- Populating concrete unit instances (M-R2).
- Tagging constants via annotations (M-R3 beyond variant-carrier
  substrate).
- Domain-aware parsing that affects Expression shape (M-R4 beyond
  annotation carriage).

Those items are explicitly deferred. See MX-10.

---

## MX-4 — Backward compatibility on existing parsers [Required]

Every existing parse input (LaTeX, plain text, mathlex native
syntax) must continue to succeed in mathlex v0.4.0 without any
annotations present, producing an `Expression` with an empty
`AnnotationSet` on every node. Existing test fixtures, existing
downstream integrations, and the thales v0.8.x-era call sites
(before thales completes its own v0.9.0 migration) must continue to
work unchanged.

In particular: no existing fixture in mathlex's own test suite may
need modification to pass on mathlex v0.4.0. Any fixture that
requires modification indicates a wire-format regression that must
be fixed in mathlex before release.

---

## MX-5 — Derivative and ODE primitives remain stable [Required]

mathlex v0.3.3 already parses derivative notation and detects ODEs
from text. thales v0.9.0 requires these primitives to remain stable
in both shape and behavior. Specifically:

- The derivative variant (whatever its current public name in
  mathlex, for example `Expression::Derivative { expr, var, order }`
  or equivalent) must retain the same variant tag after MX-1 serde
  derives land.
- ODE classification in mathlex output must continue to produce the
  same Expression shape that thales's `src/mathlex_bridge.rs`
  consumes today. No field additions that change the tag layout,
  no removals.

If mathlex needs to reshape derivative or ODE variants for internal
cleanup (for example to simplify `mathlex_bridge.rs::convert_expression`
or `collect_ode_terms` on thales's side), that reshape must happen
**before** MX-1 freezes the wire format. Once v0.4.0 ships, these
variants are frozen along with all others.

---

## MX-6 — No silent Expression layout changes during the v0.9.0 window [Required]

Between the start of thales v0.9.0 work (immediately after this
document is accepted) and thales v0.9.0's release, mathlex may ship
bugfix releases and additive features but must not:

- Remove any public Expression variant.
- Change any variant's serialized field names or field order in a
  way that changes JSON shape.
- Change any numeric literal serialization format.
- Change AnnotationSet shape or field name on the Expression node.

If mathlex needs to land a breaking change during this window — for
a compelling reason such as a security fix or a blocking correctness
issue — it must notify the thales team with at least one release's
warning so thales can synchronize. The default posture is: no
breaking changes to Expression shape in this window.

---

## MX-7 — Test fixtures mathlex must publish [Required]

For each of the following categories, mathlex must publish a small
JSON fixture file under `tests/fixtures/serde/` (or an equivalent
path agreed with the thales team). thales will consume these
fixtures verbatim as round-trip test inputs in its own CI, so drift
between mathlex's serde implementation and thales's decoder
expectations is caught on every mathlex release.

Minimum fixture coverage:

- Every public Expression variant, at least one instance each. One
  fixture file per variant is acceptable and preferred for
  diff-ability.
- Nested expressions covering: rationals, transcendentals (sin, cos,
  exp, log), matrices of at least 2x2, piecewise expressions, sums
  and products (both as reducers and as collections), derivatives
  of order 1 and order 2, indefinite and definite integrals, and
  at least one ODE instance.
- At least one Expression with a non-empty `AnnotationSet`. The
  AnnotationSet content is a placeholder per M-R1 substrate — no
  units or domains are populated in v0.9.0. The fixture only needs
  to exercise the serialization path, not meaningful annotation
  content.

thales will add a CI job that for each fixture decodes, re-encodes,
and asserts textual equality with the original file. Any drift
fails the thales build and surfaces the regression immediately.

---

## MX-8 — Documentation deliverables [Required]

mathlex must publish, at the time MX-1 through MX-4 ship in v0.4.0:

- An updated README section describing the serde schema, with a
  short example of a serialized Expression.
- A `docs/WIRE-FORMAT.md` (or equivalent) enumerating every public
  variant's JSON shape with one example per variant, and
  documenting the tag-and-content style chosen in MX-1.
- A migration note in the v0.4.0 CHANGELOG for any consumer who
  was using the pre-serde mathlex API, listing the breaking
  changes and pointing to the new schema doc.

thales will reference `docs/WIRE-FORMAT.md` from its own v0.9.0
release notes for downstream Swift consumers, so the doc must be
publicly hosted (on crates.io's docs.rs page or the mathlex GitHub
repository) before thales v0.9.0 ships.

---

## MX-9 — Version alignment [Required]

mathlex v0.4.0 is the target release for MX-1 through MX-8. The
mathlex v0.4.0 tag must precede thales v0.9.0 so thales's
`Cargo.toml` can pin `mathlex = "0.4"` cleanly without relying on
a pre-release tag.

A pre-release (`mathlex = "0.4.0-rc.1"`) is acceptable as an
intermediate step and allows thales Phase 3 (serde-on-AST
migration) to begin before the mathlex final tag. thales will not,
however, consume a pre-release that still has open questions about
tag naming or AnnotationSet placement. Once the rc goes out, its
wire format is considered provisional-final, and any change before
the final tag must be flagged explicitly.

The thales Swift package follows thales version lockstep. Swift
package v0.9.0 ships at the same commit as thales crate v0.9.0.
This cadence is internal to thales and has no direct mathlex
action, but it's documented here so mathlex release-coordination
can plan around it.

---

## MX-10 — Out of scope for mathlex v0.4.0 [Deferred]

The following items are explicitly deferred to a later mathlex
release and are not required for thales v0.9.0. If mathlex ships
any of them earlier as additive capability, thales ignores them
until v0.10.0.

- **Unit propagation logic** (M-R2 consumption, not just
  annotation carriage). Requires a shared `mathcore-units` crate
  with `Unit`, `Dimension`, and related types, referenced by both
  mathlex and thales. See RFC §M-R2.
- **Runtime constant identification** (M-R3 beyond substrate).
  v0.4.0 may reserve the annotation slot; it must not resolve
  constants to numerical values from annotations.
- **Domain-aware parsing that affects Expression shape** (M-R4
  beyond annotation carriage). The parser may accept domain
  syntax as annotations but must not branch on domain when
  producing the Expression tree.
- **Multi-locale parser messages.** thales v0.9.0 remains
  English-only for error and narrative text; mathlex's parser
  error locale does not block thales.
- **The v0.7.0-era parser gaps** listed in the sibling
  `thales_requirements.md` (integral plain-text parsing, special
  function recognition, LaTeX Leibniz notation, prime notation).
  Those remain open and are tracked separately. They do not
  gate thales v0.9.0 — thales works around them today — but
  closing them is welcome whenever mathlex has bandwidth.

---

## Parallelization plan

mathlex and thales teams work in parallel as follows. The plan
assumes both teams are active throughout the v0.9.0 window; if
either team pauses, the dependency edges below show which phases
can still advance.

- **Day 0.** mathlex team writes its own PRD against MX-1 through
  MX-10 and begins MX-1 (serde derives). thales team begins
  thales Phase 1 (dispatch split, narrative render pass) and
  Phase 2 F1a–F1f (NotImplemented promotions). None of these
  depend on mathlex changes.

- **Midway.** mathlex ships v0.4.0-rc with MX-1 and MX-2
  complete. thales begins Phase 3 (F3 serde-on-AST migration)
  against the rc. thales Phase 2 continues in parallel.

- **Late.** mathlex ships MX-3 (AnnotationSet substrate).
  thales threads it through compile and decompile without
  consuming. mathlex also ships MX-7 fixtures; thales wires them
  into CI.

- **Release gate.** Both crates tag simultaneously — mathlex
  v0.4.0 final and thales v0.9.0 — with the Swift package v0.9.0
  matching thales lockstep. The simultaneous tag is a single
  coordination step between the two teams; neither side tags
  first.

### Slip handling

If mathlex MX-1 slips past the end of thales Phase 2, thales's
Phase 3 is the only blocked thales phase. thales Phases 1, 2, and
4 stay on their own critical path. thales v0.9.0 release simply
waits on mathlex v0.4.0 and does not degrade scope to accommodate
the slip. Thales will not ship a v0.9.0 with partial Phase 3
(no Rule 5 exception for shipping with a hybrid string-adapter
plus partial-serde FFI surface).

If a new mathlex breaking requirement emerges mid-window, the
thales team is notified per MX-6, and the two teams re-coordinate
the release gate.

---

## Summary: what mathlex must deliver in v0.4.0

Blockers for thales v0.9.0:

1. **MX-1** — serde derives on the full public Expression AST,
   lossless round-trip, uniform tag-and-content enum style,
   precision-preserving numeric literal serialization.
2. **MX-2** — variant tag stability guarantee plus a golden-file
   regression test.

Required alongside v0.4.0:

3. **MX-3** — AnnotationSet substrate per RFC §M-R1, opaque to
   thales v0.9.0.
4. **MX-4** — existing parsers unchanged, empty AnnotationSet on
   every node by default.
5. **MX-5** — derivative and ODE variants frozen in their current
   shape under MX-1 serde derives.
6. **MX-6** — no silent breaking changes during the v0.9.0 window.
7. **MX-7** — serde fixture files for every public variant plus
   representative nested shapes.
8. **MX-8** — updated README, `docs/WIRE-FORMAT.md`, and v0.4.0
   CHANGELOG migration note.
9. **MX-9** — v0.4.0 tag lands before thales v0.9.0; rc acceptable
   as an intermediate.

Explicitly deferred to later releases:

10. **MX-10** — unit propagation, runtime constant resolution,
    domain-aware parsing, multi-locale messages, and the v0.7.0-era
    parser gap list.
