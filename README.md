# mathlex

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/mathlex.svg)](https://crates.io/crates/mathlex)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)
[![Documentation](https://docs.rs/mathlex/badge.svg)](https://docs.rs/mathlex)

A mathematical expression parser for LaTeX and plain text notation, producing a language-agnostic Abstract Syntax Tree (AST).

**[Full Documentation on docs.rs](https://docs.rs/mathlex)**

## Features

- **LaTeX Parsing** - Parse mathematical LaTeX notation (`\frac{1}{2}`, `\int_0^1`, `\sum_{i=1}^n`)
- **Plain Text Parsing** - Parse standard math notation (`2*x + 3`, `sin(x)`)
- **Rich AST** - Comprehensive AST supporting algebra, calculus, and linear algebra
- **No Evaluation** - Pure parsing library, interpretation is client responsibility
- **Swift Support** - Native Swift bindings for iOS/macOS

## Installation

```toml
[dependencies]
mathlex = "0.1.0"
```

## Quick Start

### Parse Plain Text

```rust
use mathlex::{parse, Expression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = parse("2*x + sin(y)")?;

    // Find all variables
    let vars = expr.find_variables();
    println!("Variables: {:?}", vars); // {"x", "y"}

    // Convert back to string
    println!("{}", expr.to_string()); // "2 * x + sin(y)"

    Ok(())
}
```

### Parse LaTeX

```rust
use mathlex::{parse_latex, Expression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expr = parse_latex(r"\frac{d}{dx}(x^2 + 3x)")?;

    // Convert to LaTeX string
    println!("{}", expr.to_latex()); // "\frac{d}{dx}(x^{2} + 3x)"

    Ok(())
}
```

## AST Structure

mathlex produces a rich AST that can represent:

- **Literals**: integers, floats, rationals, complex numbers
- **Symbols**: variables, Greek letters, mathematical constants (pi, e, i)
- **Operations**: binary (+, -, *, /, ^), unary (-, factorial)
- **Functions**: sin, cos, log, sqrt, and user-defined
- **Calculus**: derivatives, integrals, limits (representation only)
- **Structures**: vectors, matrices, equations, inequalities

## Design Philosophy

mathlex is a **pure parsing library**. It converts text to AST and back - nothing more.

- **No evaluation** - mathlex does not compute values
- **No simplification** - mathlex does not transform expressions
- **No dependencies on consumers** - can be used by any library

This design allows different libraries to interpret the AST according to their capabilities:
- A CAS library can perform symbolic differentiation on `Derivative` nodes
- A numerical library can evaluate `Function` nodes numerically
- An educational tool can render step-by-step explanations

## Documentation

- **Rust**: [docs.rs/mathlex](https://docs.rs/mathlex)
- **Swift**: [Swift Package Index](https://swiftpackageindex.com/ChrisGVE/mathlex/documentation)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Crate**: [crates.io/crates/mathlex](https://crates.io/crates/mathlex)
- **Documentation**: [docs.rs/mathlex](https://docs.rs/mathlex)
- **Repository**: [github.com/ChrisGVE/mathlex](https://github.com/ChrisGVE/mathlex)
- **Issues**: [Report bugs](https://github.com/ChrisGVE/mathlex/issues)
