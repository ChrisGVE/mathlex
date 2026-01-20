# Supported Notation

Learn about the mathematical notation that MathLex can parse.

## Overview

MathLex supports a wide range of mathematical notation in both plain text and LaTeX formats. This article provides a comprehensive reference of supported constructs.

## Literals

### Integers
Plain text and LaTeX: `42`, `-17`, `0`

### Floating Point
Plain text and LaTeX: `3.14`, `2.5e-3`, `-0.001`

### Rationals
LaTeX only: `\frac{1}{2}`, `\frac{a}{b}`

## Variables and Symbols

### Variables
Plain text: `x`, `y`, `theta`, `myVar`

LaTeX: `x`, `\theta`, `\alpha`, `\beta`

### Greek Letters (LaTeX)
- Lowercase: `\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\zeta`, `\eta`, `\theta`, `\iota`, `\kappa`, `\lambda`, `\mu`, `\nu`, `\xi`, `\pi`, `\rho`, `\sigma`, `\tau`, `\upsilon`, `\phi`, `\chi`, `\psi`, `\omega`
- Uppercase: `\Gamma`, `\Delta`, `\Theta`, `\Lambda`, `\Xi`, `\Pi`, `\Sigma`, `\Upsilon`, `\Phi`, `\Psi`, `\Omega`

### Mathematical Constants
- Pi: `pi` (plain text), `\pi` (LaTeX)
- Euler's number: `e`
- Imaginary unit: `i`
- Infinity: `inf` (plain text), `\infty` (LaTeX)

## Operators

### Binary Operators
| Operation | Plain Text | LaTeX |
|-----------|------------|-------|
| Addition | `a + b` | `a + b` |
| Subtraction | `a - b` | `a - b` |
| Multiplication | `a * b` | `a \cdot b` or `a \times b` |
| Division | `a / b` | `\frac{a}{b}` |
| Exponentiation | `a ^ b` | `a^b` or `a^{expr}` |
| Modulo | `a % b` | — |

### Unary Operators
| Operation | Plain Text | LaTeX |
|-----------|------------|-------|
| Negation | `-x` | `-x` |
| Factorial | `n!` | `n!` |

## Functions

### Standard Functions
Both plain text and LaTeX support these functions:
- Trigonometric: `sin`, `cos`, `tan`, `cot`, `sec`, `csc`
- Inverse trigonometric: `asin`, `acos`, `atan`
- Hyperbolic: `sinh`, `cosh`, `tanh`
- Logarithmic: `log`, `ln`, `log10`
- Other: `sqrt`, `abs`, `exp`, `max`, `min`

### LaTeX Function Syntax
```latex
\sin{x}
\cos{x}
\sqrt{x}
\sqrt[n]{x}  % nth root
\log_{10}{x} % log base 10
```

## Calculus (Representation Only)

MathLex parses calculus notation into AST nodes but does not evaluate them.

### Derivatives
LaTeX only:
```latex
\frac{d}{dx} f(x)
\frac{d^2}{dx^2} f(x)
```

### Partial Derivatives
LaTeX only:
```latex
\frac{\partial}{\partial x} f(x,y)
```

### Integrals
LaTeX only:
```latex
\int f(x) dx           % indefinite
\int_a^b f(x) dx       % definite
```

### Limits
LaTeX only:
```latex
\lim_{x \to a} f(x)
\lim_{x \to \infty} f(x)
```

### Summations and Products
LaTeX only:
```latex
\sum_{i=1}^{n} a_i
\prod_{i=1}^{n} a_i
```

## Structures

### Equations and Inequalities
```latex
x = y
x < y
x \leq y
x > y
x \geq y
x \neq y
```

### Vectors (LaTeX)
```latex
\begin{pmatrix} a \\ b \\ c \end{pmatrix}
```

### Matrices (LaTeX)
```latex
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}

\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
```

## Examples

### Simple Expressions
```swift
// Plain text
try MathExpression.parse("2*x + 3")
try MathExpression.parse("sin(x)^2 + cos(x)^2")

// LaTeX
try MathExpression.parseLatex(#"\frac{1}{2} + \sqrt{x}"#)
```

### Complex Expressions
```swift
// Quadratic formula in LaTeX
try MathExpression.parseLatex(#"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"#)

// Integration
try MathExpression.parseLatex(#"\int_0^1 x^2 dx"#)

// Summation
try MathExpression.parseLatex(#"\sum_{i=1}^{n} i^2"#)
```
