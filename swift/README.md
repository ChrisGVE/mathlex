# MathLex Swift Package

Swift wrapper for the mathlex mathematical expression parser library.

## Overview

MathLex provides a Swift-friendly API for parsing mathematical expressions in both plain text and LaTeX formats. The library converts mathematical notation into an Abstract Syntax Tree (AST) that can be queried and analyzed.

**Important**: MathLex is a pure parsing library. It does NOT evaluate expressions or perform mathematical computations.

## Installation

### Swift Package Manager

Add MathLex to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/ChrisGVE/mathlex", from: "0.1.0")
]
```

### Xcode

1. File > Add Packages
2. Enter the repository URL: `https://github.com/ChrisGVE/mathlex`
3. Select the version and add to your target

## Requirements

- iOS 15.0+ / macOS 12.0+
- Swift 5.9+
- Xcode 15.0+

## Quick Start

```swift
import MathLex

// Parse plain text expression
let expr = try MathExpression.parse("2*x + sin(y)")

// Parse LaTeX expression
let latexExpr = try MathExpression.parseLatex(#"\frac{1}{2}"#)

// Query the expression
let variables = expr.variables    // {"x", "y"}
let functions = expr.functions    // {"sin"}

// Convert to different formats
let plainText = expr.description  // "2 * x + sin(y)"
let latex = expr.latex           // LaTeX representation
```

## Features

### Parsing

#### Plain Text Parser

Supports standard mathematical notation:

```swift
// Basic arithmetic
let expr1 = try MathExpression.parse("2 + 3 * 4")

// Functions
let expr2 = try MathExpression.parse("sin(x) + cos(y)")

// Variables and constants
let expr3 = try MathExpression.parse("2*pi*r")

// Complex expressions
let expr4 = try MathExpression.parse("(x^2 + y^2)^(1/2)")
```

#### LaTeX Parser

Supports common LaTeX mathematical notation:

```swift
// Fractions
let frac = try MathExpression.parseLatex(#"\frac{a}{b}"#)

// Powers and roots
let power = try MathExpression.parseLatex(#"x^{2n}"#)
let root = try MathExpression.parseLatex(#"\sqrt[3]{x}"#)

// Calculus
let deriv = try MathExpression.parseLatex(#"\frac{d}{dx} f(x)"#)
let integral = try MathExpression.parseLatex(#"\int_0^1 x^2 dx"#)

// Summations
let sum = try MathExpression.parseLatex(#"\sum_{i=1}^{n} i^2"#)

// Matrices
let matrix = try MathExpression.parseLatex(#"""
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
"""#)
```

### Querying

Extract information about expressions:

```swift
let expr = try MathExpression.parse("sin(x) + cos(y) + pi")

// Get all variables
let vars = expr.variables       // {"x", "y"}

// Get all functions
let funcs = expr.functions      // {"sin", "cos"}

// Get mathematical constants
let consts = expr.constants     // {"pi"}

// Get tree metrics
let depth = expr.depth          // Maximum depth of AST
let count = expr.nodeCount      // Total number of nodes
```

### Conversion

Convert between formats:

```swift
// Parse LaTeX and convert to plain text
let expr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
print(expr.description)  // "1 / 2"

// Parse plain text and convert to LaTeX
let expr2 = try MathExpression.parse("1/2")
print(expr2.latex)  // "\frac{1}{2}"
```

## Error Handling

MathLex uses Swift's error handling for parsing failures:

```swift
do {
    let expr = try MathExpression.parse("invalid syntax )")
    // Use expr
} catch MathLexError.parseError(let message) {
    print("Parse error: \(message)")
} catch {
    print("Unexpected error: \(error)")
}
```

## Supported Operations

### Arithmetic
- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Exponentiation: `^`
- Modulo: `%`

### Functions
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
- Logarithmic: `log`, `ln`, `log10`
- Other: `sqrt`, `abs`, `exp`, etc.

### Calculus
- Derivatives: `d/dx`
- Partial derivatives: `∂/∂x`
- Integrals: `∫`
- Limits: `lim`
- Summations: `Σ`
- Products: `Π`

### Linear Algebra
- Vectors: `[a, b, c]`
- Matrices: `[[a, b], [c, d]]`
- Transpose

### Constants
- Pi: `pi` or `π`
- Euler's number: `e`
- Imaginary unit: `i`
- Infinity: `inf` or `∞`

## Building the XCFramework

The Swift package relies on an XCFramework built from the Rust library:

```bash
cd swift
./build.sh
```

This will:
1. Build the Rust library for all target platforms
2. Generate Swift bindings using swift-bridge
3. Create a universal XCFramework
4. Update the Swift package to use the framework

## Architecture

```
MathLex (Swift Package)
├── MathLexRust (Generated bindings from Rust via swift-bridge)
│   └── Generated Swift/C code for FFI
└── MathLex (Swift wrapper)
    └── Idiomatic Swift API
```

The Swift package wraps the Rust-based mathlex library:
- **Rust Core**: Fast, memory-safe parser implementation
- **swift-bridge**: Automatic FFI code generation
- **Swift Wrapper**: Idiomatic Swift API with proper error handling

## Examples

### Calculator App

```swift
import MathLex

struct CalculatorView: View {
    @State private var input = ""
    @State private var result = ""

    var body: some View {
        VStack {
            TextField("Enter expression", text: $input)
            Button("Parse") {
                parseExpression()
            }
            Text(result)
        }
    }

    func parseExpression() {
        do {
            let expr = try MathExpression.parse(input)
            result = """
                Variables: \(expr.variables)
                Functions: \(expr.functions)
                LaTeX: \(expr.latex)
                """
        } catch {
            result = "Error: \(error.localizedDescription)"
        }
    }
}
```

### Equation Analyzer

```swift
import MathLex

func analyzeEquation(_ input: String) throws -> EquationAnalysis {
    let expr = try MathExpression.parse(input)

    return EquationAnalysis(
        variables: Array(expr.variables).sorted(),
        functions: Array(expr.functions).sorted(),
        constants: Array(expr.constants).sorted(),
        complexity: expr.nodeCount,
        depth: expr.depth,
        latex: expr.latex
    )
}

struct EquationAnalysis {
    let variables: [String]
    let functions: [String]
    let constants: [String]
    let complexity: Int
    let depth: Int
    let latex: String
}
```

### LaTeX Renderer

```swift
import MathLex

func convertToLatex(_ plainText: String) throws -> String {
    let expr = try MathExpression.parse(plainText)
    return expr.latex
}

// Usage
let latex = try convertToLatex("sqrt(x^2 + y^2)")
// Renders as: \sqrt{x^2 + y^2}
```

## Testing

Run tests using Swift Package Manager:

```bash
swift test
```

Or in Xcode:
- Open `Package.swift` in Xcode
- Product > Test (⌘U)

## Performance

MathLex is designed for performance:
- Written in Rust for speed and memory safety
- Zero-copy FFI where possible
- Efficient AST representation
- Optimized for both parsing and querying

Typical performance on modern hardware:
- Simple expressions: < 1μs
- Complex expressions: < 10μs
- Very large expressions: < 100μs

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## License

MIT License - See LICENSE file in the repository root.

## Links

- [Repository](https://github.com/ChrisGVE/mathlex)
- [Documentation](https://docs.rs/mathlex)
- [Issue Tracker](https://github.com/ChrisGVE/mathlex/issues)

## Consumers

MathLex is designed to serve as a foundation for:
- **NumericSwift**: Numerical computing library
- **thales**: Symbolic computer algebra system
- Other mathematical software needing expression parsing
