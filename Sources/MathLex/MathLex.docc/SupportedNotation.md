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
- Euler's number: `e` (both formats), `\mathrm{e}` (LaTeX explicit)
- Imaginary unit: `i` (both formats), `\mathrm{i}`, `\imath` (LaTeX explicit)
- Infinity: `inf` or `∞` (plain text), `\infty` (LaTeX)
- Negative infinity: `-inf` or `-∞` (plain text), `-\infty` (LaTeX) — parsed as `Constant(NegInfinity)`

> Note: In LaTeX, `e` and `i` are context-aware. They default to constants but become variables when bound in scopes like `\sum_{i=1}^n` or `\prod_{e=1}^n`. Use `\mathrm{e}` or `\mathrm{i}` to force constant interpretation.

## Operators

### Binary Operators
| Operation | Plain Text | LaTeX |
|-----------|------------|-------|
| Addition | `a + b` | `a + b` |
| Subtraction | `a - b` | `a - b` |
| Multiplication | `a * b` | `a \cdot b` or `a \times b` |
| Division | `a / b` | `\frac{a}{b}` |
| Exponentiation | `a ^ b` or `a ** b` | `a^b` or `a^{expr}` |
| Modulo | `a % b` | — |
| Plus-Minus | — | `a \pm b` |
| Minus-Plus | — | `a \mp b` |

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
- Other: `sqrt`, `abs`, `exp`, `max`, `min`, `floor`, `ceil`, `gcd`, `lcm`, `sgn`, `det`

### LaTeX Function Syntax
```latex
\sin{x}
\cos{x}
\sqrt{x}
\sqrt[n]{x}       % nth root
\log_{10}{x}      % log base 10
\abs{x}           % absolute value
|x|               % absolute value (alternative)
\lfloor x \rfloor % floor function
\lceil x \rceil   % ceiling function
```

## Calculus (Representation Only)

MathLex parses calculus notation into AST nodes but does not evaluate them.

### Derivatives
LaTeX only:
```latex
\frac{d}{dx} f(x)          % first order (standard)
\frac{d^2}{dx^2} f(x)      % second order
\frac{d}{d*x} f(x)         % with explicit multiplication marker
```

### Partial Derivatives
LaTeX only:
```latex
\frac{\partial}{\partial x} f(x,y)
\frac{\partial^2}{\partial x^2} f(x,y)
```

### Integrals
LaTeX only:
```latex
\int f(x) dx           % indefinite
\int_a^b f(x) dx       % definite
\iint_R f dA            % double integral
\iiint_V f dV           % triple integral
\oint_C F dr            % closed line integral
\oiint_S F dS           % closed surface integral
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

## Vector Calculus

LaTeX only:
```latex
\nabla f                          % gradient
\nabla \cdot \mathbf{F}           % divergence
\nabla \times \mathbf{F}          % curl
\nabla^2 f                        % Laplacian
```

### Vector Notation Styles
```latex
\mathbf{v}            % bold
\vec{v}               % arrow
\hat{n}               % hat (unit vector)
\overrightarrow{AB}   % arrow over name
```

### Vector Products
```latex
\mathbf{a} \cdot \mathbf{b}      % dot product (via \bullet)
\mathbf{a} \times \mathbf{b}     % cross product
\mathbf{a} \otimes \mathbf{b}    % outer product
```

Plain text equivalents: `dot(a, b)`, `cross(a, b)`, `grad(f)`, `div(F)`, `curl(F)`, `laplacian(f)`

## Linear Algebra

LaTeX only:
```latex
\det(A)               % determinant
\tr(A)                % trace
\rank(A)              % rank
A^{-1}                % matrix inverse
A^T                   % transpose
A^\top                % transpose (alternative)
A^\dagger             % conjugate transpose
```

## Set Theory

### Set Operations (LaTeX)
```latex
A \cup B              % union
A \cap B              % intersection
A \setminus B         % difference
```

### Set Relations
```latex
x \in A               % element of
x \notin A            % not element of
A \subset B           % proper subset
A \subseteq B         % subset or equal
A \supset B           % proper superset
A \supseteq B         % superset or equal
```

### Special Sets
```latex
\emptyset             % empty set
\varnothing           % empty set (alternative)
\mathcal{P}(A)        % power set
\mathbb{N}            % natural numbers
\mathbb{Z}            % integers
\mathbb{Q}            % rationals
\mathbb{R}            % reals
\mathbb{C}            % complex numbers
\mathbb{H}            % quaternions
```

Plain text equivalents: `union`, `intersect`, `in`, `notin`

## Quantifiers and Logic

### Quantifiers
```latex
\forall x (P(x))                 % universal
\forall x \in S (P(x))           % with domain
\exists x (P(x))                 % existential
```

### Logical Connectives
```latex
P \land Q             % and
P \lor Q              % or
\lnot P               % not (also \neg)
P \implies Q          % implication
P \iff Q              % biconditional
```

Plain text equivalents: `and`, `or`, `not`, `implies`, `iff`, `forall`, `exists`

## Tensor Notation

LaTeX only:
```latex
T^{ij}_{kl}           % tensor with indices
\delta^i_j            % Kronecker delta
\varepsilon_{ijk}     % Levi-Civita symbol
```

## Quaternions

Quaternion basis vectors require explicit markers in LaTeX:
```latex
\mathrm{i}            % imaginary unit i
\mathrm{j}            % quaternion j (also \mathbf{j})
\mathrm{k}            % quaternion k (also \mathbf{k})
```

## Structures

### Subscripts
```latex
x_i                   % simple subscript
x_{i+1}              % expression subscript (flattened to variable name)
a_{n-1}              % expression subscript
```

Plain text: `x_1`, `x_i`

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
