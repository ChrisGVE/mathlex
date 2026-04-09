# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4] - 2026-04-09

### Added

- **Plain text integral notation**: `integrate(expr, var)` and `integrate(expr, var, lower, upper)` producing `Expression::Integral`. Aliases: `integral`, `int`. Variable accepts `dx` or bare `x` form.
- **Plain text summation**: `sum(expr, var, lower, upper)` producing `Expression::Sum`. Aliases: `summation`, `Sum`.
- **Plain text product**: `product(expr, var, lower, upper)` producing `Expression::Product`. Aliases: `prod`, `Product`.
- **Plain text limit**: `limit(expr, var, point)` with optional direction (`+`, `-`, `left`, `right`) producing `Expression::Limit`. Aliases: `lim`, `Limit`.
- **Operator-form derivative**: `d(expr)/dx` and `d(expr)/d(var)` producing `Expression::Derivative`.
- **Special function recognition**: `gamma`, `beta`, `erf`, `erfc`, `zeta`, `bessel_j`, `bessel_y`, `bessel_i`, `bessel_k` added to known-functions list for typo suggestions.
- **Transform functions**: `laplace`, `fourier`, `ilaplace`, `ifourier` recognized as known functions.
- **Unicode logic/set tokens**: `∧` (and), `∨` (or), `¬` (not), `→` (implies), `↔` (iff), `∪` (union), `∩` (intersect), `∈` (in), `∉` (notin) recognized by text tokenizer.
- **Cross-parser round-trip tests**: 47 tests verifying text→LaTeX→text and LaTeX→text→LaTeX paths.
- **Plain text syntax reference**: Comprehensive `docs/plain-text-syntax.md` documenting all supported notation with LaTeX equivalents.

## [0.3.3] - 2026-04-09

### Added

- **Gradient plain text notation**: `nabla(f)` as alias for `grad(f)`, and Unicode `∇f` for gradient expressions
- **Parenthesis-optional vector calculus**: `grad f`, `∇f`, `div f`, `curl f`, `laplacian f` now work without parentheses (parsing the next primary expression as the argument)

### Documentation

- Updated README to document all plain text derivative, partial derivative, and gradient notations

## [0.3.2] - 2026-04-09

### Added

- **Plain text derivative parsing**: Leibniz notation (`dy/dx`, `d2y/dx2`, `d3y/dx3`), prime notation (`y'`, `y''`, `y'''`), and functional notation (`diff(expr, var)`, `diff(expr, var, order)`) all produce `Expression::Derivative`
- **Plain text partial derivative parsing**: functional notation `partial(f, x)`, `partial(f, x, 2)` for higher-order, and `partial(f, x, y)` for mixed partials producing nested `Expression::PartialDerivative`
- **Apostrophe token**: tokenizer recognizes `'` for prime derivative notation

## [0.3.1] - 2026-04-06

### Added

- **New functions**: `trunc`, `clamp`, `lerp`, `rad`, `deg` added to both plain text and LaTeX parsers
- **NaN constant**: `MathConstant::NaN` variant; parseable from `nan`/`NaN` in plain text and `\text{NaN}`/`\mathrm{NaN}` in LaTeX
- **JSON serialization**: `toJSON()` and `toJSONPretty()` methods on `MathExpression` in both the Rust FFI layer and the Swift wrapper; requires the `serde` feature
- **JSON AST schema**: `docs/json-ast-schema.md` documents the complete JSON representation of every AST node
- **NumericSwift integration example**: `examples/numericswift-integration/` demonstrates decoding the JSON AST into Swift `Decodable` types and evaluating expressions numerically
- **Swift integration tests**: 14 new XCTest cases covering JSON serialization round-trips
- **FFI test coverage**: expanded FFI test suite from 12 to 45 tests, covering all JSON serialization paths

### Fixed

- **Swift CI**: `serde` feature flag added to the Swift CI workflow and XCFramework build so JSON serialization is available in packaged builds

## [0.3.0] - 2026-04-06

### Added

- **Equation system parsing**: `parse_equation_system()` and `parse_latex_equation_system()` for parsing semicolon-delimited equations into `Vec<Expression>`
- **Function name aliases**: plain text parser recognizes `asin`/`acos`/`atan` as `arcsin`/`arccos`/`arctan`, `sign` as `sgn`, `log2` as `lg`
- **Additional functions**: `atan2`, `cbrt`, `round`, `pow` parse as named functions; added to known-functions list for typo suggestions
- **Tree traversal**: `Expression::map()` for bottom-up tree transformation
- **Tree reduction**: `Expression::fold()` for generic tree accumulation
- **Variable lookup**: `Expression::contains_variable()` with early-exit semantics
- **FFI**: equation system parsing wrappers for Swift bindings
- **Serde**: comprehensive round-trip tests for all Expression variants

### Documentation

- Added guidance for extending `Equation` with tracking IDs
- Added guidance for attaching variable metadata (dimensions, units)
- Updated utility module documentation to list all available methods

## [0.2.0] - 2026-04-03

### Added

- **Vector calculus**: gradient (`\nabla f`), divergence (`\nabla \cdot F`), curl (`\nabla \times F`), Laplacian (`\nabla^2 f`) in both AST and LaTeX parser
- **Vector notation**: bold (`\mathbf{v}`), arrow (`\vec{v}`), hat (`\hat{n}`), underline parsing; dot, cross, and outer product operators
- **Multiple integrals**: double (`\iint`), triple (`\iiint`), and quadruple (`\iiiint`) integrals with optional region bounds and multiple differential variables
- **Closed integrals**: line (`\oint`), surface (`\oiint`), and volume (`\oiiint`) integrals
- **Set theory**: union (`\cup`), intersection (`\cap`), difference (`\setminus`); membership (`\in`, `\notin`); subset/superset relations; empty set (`\emptyset`); power set (`\mathcal{P}`); number sets (`\mathbb{N}` through `\mathbb{H}`)
- **Quantifiers**: universal (`\forall`) and existential (`\exists`) with optional domains
- **Logical connectives**: `\land`, `\lor`, `\lnot`, `\implies`, `\iff` with correct precedence hierarchy
- **Quaternions**: `Expression::Quaternion` variant; basis vectors `J` and `K` in `MathConstant`; parsing via `\mathrm{j}`, `\mathrm{k}`, `\mathbf{j}`, `\mathbf{k}`
- **Tensor notation**: Einstein index notation (`T^{ij}_{kl}`), Kronecker delta (`\delta^i_j`), Levi-Civita symbol (`\varepsilon_{ijk}`)
- **Linear algebra operations**: `\det`, `\tr`, `\rank`, conjugate transpose (`A^\dagger`), matrix inverse (`A^{-1}`), transpose (`A^T`, `A^\top`)
- **Context-aware constants**: `e` and `i` are parsed as `MathConstant::E` and `MathConstant::I` by default in LaTeX, with scope-aware fallback to variables when bound in `\sum`/`\prod`; explicit markers via `\mathrm{e}`, `\mathrm{i}`, `\imath`, `\jmath`
- **NegInfinity folding**: `-\infty` (LaTeX) and `-inf`/`-∞` (plain text) parse directly as `Constant(NegInfinity)` instead of `Unary(Neg, Constant(Infinity))`
- **Expression subscripts**: `x_{i+1}`, `a_{n-1}` in LaTeX parser (flattened to variable names)
- **Plain text subscripts**: `x_1`, `x_i` in plain text parser
- **Plain text extensions**: vector operations (`dot`, `cross`), vector calculus (`grad`, `div`, `curl`, `laplacian`), quantifiers (`forall`, `exists`), set operations (`union`, `intersect`, `in`, `notin`), logical operators (`and`, `or`, `not`, `implies`, `iff`)
- **Expression context engine**: `ExpressionContext` for tracking variable types, binding scopes, and number systems across multiple expressions; `parse_system()` for multi-expression parsing
- **Expression metadata**: type inference foundation for expression analysis
- **Function signatures**: `f: A \to B` notation and relation parsing in LaTeX
- **Error suggestions**: helpful hints for common parse mistakes
- **Comprehensive benchmarks**: Criterion benchmarks for all parser features
- **Fuzz testing**: cargo-fuzz infrastructure for both text and LaTeX parsers
- **Property-based tests**: LaTeX parser property tests; serialization precedence tests
- **Integration tests**: vector calculus, set theory, and real-world expression test suites
- **Swift CI**: GitHub Actions workflow for Swift package testing

### Fixed

- **ToLatex precedence**: parentheses around unary prefix operators in power expressions and additive expressions in product operators, ensuring round-trip safety
- **Round-trip tests**: LaTeX round-trip tests now fail on unparsable output instead of printing warnings
- **Swift bindings**: generated bindings are copied into `Sources/MathLexRust`; simplified import structure
- **Tokenizer**: removed duplicate code from malformed merge

### Changed

- Removed unused `chumsky` dependency

## [0.1.1] - 2026-01-17

### Fixed

- Move DocC catalog into MathLex target for Swift Package Index compatibility
- Resolve clippy warnings and formatting issues in CI
- Add write permissions for release workflows

## [0.1.0] - 2026-01-17

### Added

- Initial release
- LaTeX parser: arithmetic, fractions, roots, trigonometric/logarithmic functions, derivatives, integrals, limits, sums, products, matrices, vectors
- Plain text parser: arithmetic, functions, implicit multiplication, configurable options
- AST with 19+ expression variants covering algebra, calculus, and linear algebra
- `Display` trait for human-readable output
- `ToLatex` trait for LaTeX serialization
- AST utilities: `find_variables`, `find_functions`, `find_constants`, `depth`, `node_count`, `substitute`, `substitute_all`
- Optional `serde` feature for AST serialization
- Swift bindings via `swift-bridge` (behind `ffi` feature flag)
- XCFramework build script for iOS/macOS distribution
- GitHub Actions CI/CD with release automation
- Comprehensive unit test suite (700+ tests)

[Unreleased]: https://github.com/ChrisGVE/mathlex/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/ChrisGVE/mathlex/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/ChrisGVE/mathlex/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ChrisGVE/mathlex/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ChrisGVE/mathlex/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ChrisGVE/mathlex/releases/tag/v0.1.0
