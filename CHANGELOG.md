# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ChrisGVE/mathlex/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ChrisGVE/mathlex/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ChrisGVE/mathlex/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ChrisGVE/mathlex/releases/tag/v0.1.0
