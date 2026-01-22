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
- **Rich AST** - Comprehensive AST supporting algebra, calculus, and linear algebra
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

### Symbols
- Variables: `x`, `y`, `theta`, `e`, `i`
- Greek letters: `\alpha`, `\beta`, `\Gamma`
- Constants: `\pi`, `\infty`

### Operations
- Binary: `+`, `-`, `*`, `/`, `^`, `**`, `%`, `\pm`, `\mp`
- Unary: `-x`, `x!`
- Functions: `sin`, `cos`, `tan`, `log`, `ln`, `exp`, `sqrt`, `abs`, `floor`, `ceil`, `det`

### Calculus (Representation Only)
- Derivatives: `\frac{d}{d*x}`, `\frac{\partial}{\partial*x}` (use `d*x` syntax)
- Integrals: `\int`, `\int_a^b`
- Limits: `\lim_{x \to a}`
- Sums: `\sum_{i=1}^{n}`
- Products: `\prod_{i=1}^{n}`

### Structures
- Vectors: `\begin{pmatrix} a \\ b \end{pmatrix}`
- Matrices: `\begin{bmatrix} a & b \\ c & d \end{bmatrix}`
- Equations: `x = y`
- Inequalities: `x < y`, `\leq`, `\geq`, `\neq`

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
