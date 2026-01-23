# mathlex

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/mathlex)](https://github.com/ChrisGVE/mathlex/releases)
[![CI](https://github.com/ChrisGVE/mathlex/actions/workflows/ci.yml/badge.svg)](https://github.com/ChrisGVE/mathlex/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/mathlex.svg)](https://crates.io/crates/mathlex)
[![Rust Version](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![docs.rs](https://docs.rs/mathlex/badge.svg)](https://docs.rs/mathlex)
[![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FChrisGVE%2Fmathlex%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/ChrisGVE/mathlex)
[![Swift Platform](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FChrisGVE%2Fmathlex%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/ChrisGVE/mathlex)

A mathematical expression parser for LaTeX and plain text notation, producing a language-agnostic Abstract Syntax Tree (AST).

## Features

- **LaTeX Parsing** - Parse mathematical LaTeX notation (`\frac{1}{2}`, `\int_0^1`, `\sum_{i=1}^n`)
- **Plain Text Parsing** - Parse standard math notation (`2*x + 3`, `sin(x)`)
- **Rich AST** - Comprehensive AST supporting:
  - Algebra and calculus expressions
  - Linear algebra (vectors and matrices)
  - Vector calculus (gradient, divergence, curl, Laplacian)
  - Set theory (unions, intersections, quantifiers)
  - Logic (connectives, quantifiers)
  - Multiple integrals (double, triple, closed path)
  - Quaternion algebra
- **Vector Notation** - Multiple styles: bold (`\mathbf{v}`), arrow (`\vec{v}`), hat (`\hat{n}`)
- **Context-Aware** - Smart handling of `e`, `i`, `j`, `k` based on number system context
- **No Evaluation** - Pure parsing library, interpretation is client responsibility
- **Cross-Platform** - Native support for Rust and Swift (iOS/macOS)

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
mathlex = "0.1.1"
```

### Swift

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/mathlex.git", from: "0.1.1")
]
```

Or in Xcode: File → Add Package Dependencies → Enter `https://github.com/ChrisGVE/mathlex.git`

## Quick Start

### Rust

#### Parse Plain Text

```rust
use mathlex::{parse, Expression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = parse("2*x + sin(y)")?;

    // Find all variables
    let vars = expr.find_variables();
    println!("Variables: {:?}", vars); // {"x", "y"}

    // Convert back to string
    println!("{}", expr); // "2 * x + sin(y)"

    Ok(())
}
```

#### Parse LaTeX

```rust
use mathlex::{parse_latex, Expression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = parse_latex(r"\frac{1}{2} + \sqrt{x}")?;

    // Convert to LaTeX string
    println!("{}", expr.to_latex()); // "\frac{1}{2} + \sqrt{x}"

    Ok(())
}
```

### Swift

```swift
import MathLex

do {
    // Parse plain text
    let expr = try MathExpression.parse("2*x + sin(y)")
    print(expr.variables) // ["x", "y"]
    print(expr.description) // "2 * x + sin(y)"

    // Parse LaTeX
    let latex = try MathExpression.parseLatex(#"\frac{1}{2}"#)
    print(latex.latex) // "\frac{1}{2}"
} catch {
    print("Parse error: \(error)")
}
```

## Supported Notation

### Literals
- Integers: `42`, `-17`
- Floats: `3.14`, `2.5e-3`
- Rationals: LaTeX `\frac{1}{2}`
- Complex: via construction
- Quaternions: via construction with basis vectors `i`, `j`, `k`

### Symbols
- Variables: `x`, `y`, `theta`
- Greek letters: `\alpha`, `\beta`, `\Gamma`
- Constants: `\pi` (π), `e`, `\infty` (∞)
- Imaginary/quaternion units: `i`, `j`, `k` (context-aware)

### Operations
- Binary: `+`, `-`, `*`, `/`, `^`, `**`, `%`, `\pm`, `\mp`
- Unary: `-x`, `x!`
- Functions: `sin`, `cos`, `tan`, `log`, `ln`, `exp`, `sqrt`, `abs`, `floor`, `ceil`, `det`
- Vector products: `\cdot` (dot), `\times` (cross)

### Calculus (Representation Only)
- Derivatives:
  - Standard notation: `\frac{d}{dx}f`, `\frac{\partial}{\partial x}f` (operator followed by expression)
  - With explicit multiplication: `\frac{d}{d*x}f`, `\frac{\partial}{\partial*x}f` (when variable needs marker)
  - Higher order: `\frac{d^2}{dx^2}f`, `\frac{\partial^2}{\partial x^2}f`
- Integrals: `\int`, `\int_a^b`
- Multiple integrals: `\iint` (double), `\iiint` (triple)
- Closed integrals: `\oint` (line), `\oiint` (surface)
- Limits: `\lim_{x \to a}`
- Sums: `\sum_{i=1}^{n}`
- Products: `\prod_{i=1}^{n}`
- Subscripts: Supports expression subscripts like `x_{i+1}`, `a_{n-1}` (flattened to variable names)

### Vector Calculus
- Gradient: `\nabla f`
- Divergence: `\nabla \cdot \mathbf{F}`
- Curl: `\nabla \times \mathbf{F}`
- Laplacian: `\nabla^2 f`
- Vector notation styles:
  - Bold: `\mathbf{v}`
  - Arrow: `\vec{v}`
  - Hat (unit vectors): `\hat{n}`

### Set Theory
- Operations: `\cup` (union), `\cap` (intersection), `\setminus` (difference)
- Relations: `\in`, `\notin`, `\subset`, `\subseteq`, `\supset`, `\supseteq`
- Special sets: `\emptyset` or `\varnothing` (empty set)
- Number sets: `\mathbb{N}`, `\mathbb{Z}`, `\mathbb{Q}`, `\mathbb{R}`, `\mathbb{C}`, `\mathbb{H}` (quaternions)
- Power set: `\mathcal{P}(A)`
- Quantifiers: `\forall` (for all), `\exists` (there exists)

### Logic
- Connectives: `\land` (and), `\lor` (or), `\lnot` (not)
- Implications: `\implies`, `\iff` (if and only if)

### Structures
- Vectors: `\begin{pmatrix} a \\ b \end{pmatrix}`
- Matrices: `\begin{bmatrix} a & b \\ c & d \end{bmatrix}`
- Equations: `x = y`
- Inequalities: `x < y`, `\leq`, `\geq`, `\neq`

## Advanced Examples

### Vector Calculus

```rust
use mathlex::parse_latex;

// Maxwell's equations components
let gauss = parse_latex(r"\nabla \cdot \mathbf{E}").unwrap();
let faraday = parse_latex(r"\nabla \times \mathbf{E}").unwrap();

// Laplacian operator
let poisson = parse_latex(r"\nabla^2 \phi").unwrap();

// Vector identities
let div_curl = parse_latex(r"\nabla \cdot (\nabla \times \mathbf{F})").unwrap();
let curl_grad = parse_latex(r"\nabla \times (\nabla f)").unwrap();
```

### Multiple Integrals

```rust
use mathlex::parse_latex;

// Surface area integral
let surface = parse_latex(r"\iint_R f(x,y) dA").unwrap();

// Volume integral
let volume = parse_latex(r"\iiint_V \rho dV").unwrap();

// Closed line integral (circulation)
let circulation = parse_latex(r"\oint_C \mathbf{F} \cdot d\mathbf{r}").unwrap();
```

### Set Theory and Logic

```rust
use mathlex::parse_latex;

// Set operations
let union = parse_latex(r"A \cup B \cap C").unwrap();
let membership = parse_latex(r"x \in A \cup B").unwrap();

// Logical statements
let forall = parse_latex(r"\forall x \in \mathbb{R} (x^2 \geq 0)").unwrap();
let exists = parse_latex(r"\exists x (P(x) \land Q(x))").unwrap();
let implication = parse_latex(r"P \implies Q").unwrap();
```

### Quaternions

```rust
use mathlex::{Expression, MathConstant};

// Quaternion basis vectors satisfy:
// i² = j² = k² = ijk = -1
// ij = k, jk = i, ki = j
// ji = -k, kj = -i, ik = -j

let quaternion = Expression::Quaternion {
    real: Box::new(Expression::Integer(1)),
    i: Box::new(Expression::Integer(2)),
    j: Box::new(Expression::Integer(3)),
    k: Box::new(Expression::Integer(4)),
};
```

### Context-Aware Parsing

```rust
use mathlex::{parse_latex, Expression, MathConstant};

// 'e' is parsed as Euler's constant
let euler = parse_latex(r"e^x").unwrap();

// 'i' requires explicit marking to be treated as imaginary unit
let complex = parse_latex(r"3 + 4\mathrm{i}").unwrap();

// Quaternion basis vectors (when marked)
let quat = parse_latex(r"\mathbf{i} + \mathbf{j} + \mathbf{k}").unwrap();
```

## Design Philosophy

mathlex is a **pure parsing library**. It converts text to AST and back - nothing more.

- **No evaluation** - mathlex does not compute values
- **No simplification** - mathlex does not transform expressions
- **No dependencies on consumers** - can be used by any library

This design allows different libraries to interpret the AST according to their capabilities:
- A CAS library can perform symbolic differentiation on `Derivative` nodes
- A numerical library can evaluate `Function` nodes numerically
- An educational tool can render step-by-step explanations

## Optional Features

### Rust

```toml
[dependencies]
mathlex = { version = "0.1.1", features = ["serde"] }
```

- `serde` - Enable serialization/deserialization of AST types
- `ffi` - Enable Swift FFI bindings (for building XCFramework)

## Documentation

- **Rust**: [docs.rs/mathlex](https://docs.rs/mathlex)
- **Swift**: [Swift Package Index](https://swiftpackageindex.com/ChrisGVE/mathlex/documentation)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Crates.io**: [crates.io/crates/mathlex](https://crates.io/crates/mathlex)
- **Swift Package Index**: [swiftpackageindex.com/ChrisGVE/mathlex](https://swiftpackageindex.com/ChrisGVE/mathlex)
- **Documentation**: [docs.rs/mathlex](https://docs.rs/mathlex)
- **Repository**: [github.com/ChrisGVE/mathlex](https://github.com/ChrisGVE/mathlex)
- **Issues**: [Report bugs](https://github.com/ChrisGVE/mathlex/issues)
