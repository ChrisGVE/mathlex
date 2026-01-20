# ``MathLex``

A mathematical expression parser for LaTeX and plain text notation.

## Overview

MathLex is a pure parsing library that converts mathematical expressions in LaTeX or plain text format into a well-defined Abstract Syntax Tree (AST). The library does **not** perform any evaluation or mathematical operations—interpretation of the AST is entirely the responsibility of consuming applications.

### Key Features

- **LaTeX Parsing**: Parse mathematical LaTeX notation (`\frac{1}{2}`, `\int_0^1`, `\sum_{i=1}^n`)
- **Plain Text Parsing**: Parse standard math notation (`2*x + 3`, `sin(x)`)
- **Rich AST**: Comprehensive AST supporting algebra, calculus, and linear algebra
- **Query Utilities**: Extract variables, functions, and constants from expressions
- **Bidirectional Conversion**: Convert between plain text and LaTeX representations

### Quick Start

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
let latex = expr.latex            // LaTeX representation
```

### Design Philosophy

MathLex is designed to be a foundation for mathematical software:

- **No Evaluation**: MathLex does not compute values
- **No Simplification**: MathLex does not transform expressions
- **Consumer Independence**: Can be used by any library needing expression parsing

This design allows different libraries to interpret the AST according to their capabilities:
- A CAS library can perform symbolic differentiation
- A numerical library can evaluate expressions numerically
- An educational tool can render step-by-step explanations

## Topics

### Essentials

- ``MathExpression``
- ``MathExpression/parse(_:)``
- ``MathExpression/parseLatex(_:)``

### Querying Expressions

- ``MathExpression/variables``
- ``MathExpression/functions``
- ``MathExpression/constants``
- ``MathExpression/depth``
- ``MathExpression/nodeCount``

### Converting Expressions

- ``MathExpression/description``
- ``MathExpression/latex``

### Error Handling

- ``MathLexError``
- ``MathLexError/parseError(_:)``
- ``MathLexError/internalError(_:)``

### Reference

- <doc:SupportedNotation>
