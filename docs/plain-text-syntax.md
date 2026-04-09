# mathlex Plain Text Syntax Reference

This document is the authoritative specification for mathlex's plain text input format. Every supported construct is shown with its plain text syntax, the literal LaTeX equivalent, and a description of the rendered form.

Conventions adopted here follow a cross-CAS survey of Mathematica, SymPy, Maple, MATLAB, Maxima, SageMath, GeoGebra, and AsciiMath. The guiding principles are:

- **Lowercase function names** (matching Maple, MATLAB, Maxima, SageMath)
- **Positional arguments** with consistent `(expr, var, ...)` ordering
- **Explicit functional notation** over ambiguous shorthand

---

## 1. Literals

| Plain Text | LaTeX | Rendered | AST |
|---|---|---|---|
| `42` | `42` | 42 | `Integer(42)` |
| `-17` | `-17` | -17 | `Unary(Neg, Integer(17))` |
| `3.14` | `3.14` | 3.14 | `Float(3.14)` |
| `2.5e-3` | `2.5e-3` | 0.0025 | `Float(0.0025)` |

## 2. Variables and Constants

| Plain Text | LaTeX | Rendered | AST |
|---|---|---|---|
| `x` | `x` | *x* | `Variable("x")` |
| `theta` | `theta` | theta | `Variable("theta")` |
| `x_1` | `x_1` | x&#x2081; | `Variable("x_1")` |
| `pi` | `\pi` | &pi; | `Constant(Pi)` |
| `e` | `e` | *e* | `Constant(E)` |
| `i` | `i` | *i* | `Constant(I)` |
| `inf` | `\infty` | &infin; | `Constant(Infinity)` |
| `-inf` | `-\infty` | -&infin; | `Constant(NegInfinity)` |
| `nan` | `\text{NaN}` | NaN | `Constant(NaN)` |

## 3. Arithmetic Operators

| Plain Text | LaTeX | Rendered | Notes |
|---|---|---|---|
| `a + b` | `a + b` | *a* + *b* | Addition |
| `a - b` | `a - b` | *a* - *b* | Subtraction |
| `a * b` | `a \cdot b` | *a* &middot; *b* | Multiplication |
| `a / b` | `\frac{a}{b}` | *a*/*b* | Division |
| `a ^ b` | `a^{b}` | *a*&#x1D47; | Exponentiation (right-associative) |
| `a ** b` | `a^{b}` | *a*&#x1D47; | Exponentiation (alternative) |
| `a % b` | `a \bmod b` | *a* mod *b* | Modulo |
| `-x` | `-x` | -*x* | Unary negation |
| `+x` | `+x` | +*x* | Unary positive |
| `n!` | `n!` | *n*! | Factorial |

### Operator Precedence (lowest to highest)

1. Logical: `iff`, `implies`, `or`, `and`, `not`
2. Quantifiers: `forall`, `exists`
3. Set operations: `union`, `intersect`, `in`, `notin`
4. Relations: `=`, `<`, `>`, `<=`, `>=`, `!=`
5. Addition/Subtraction: `+`, `-`
6. Multiplication/Division/Modulo: `*`, `/`, `%`
7. Unary prefix: `-x`, `+x`
8. Power: `^`, `**` (right-associative)
9. Postfix: `!`
10. Function calls and atoms

## 4. Relations

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `x = y` | `x = y` | *x* = *y* |
| `x < y` | `x < y` | *x* < *y* |
| `x > y` | `x > y` | *x* > *y* |
| `x <= y` | `x \leq y` | *x* &le; *y* |
| `x >= y` | `x \geq y` | *x* &ge; *y* |
| `x != y` | `x \neq y` | *x* &ne; *y* |

Unicode alternatives: `x ≤ y`, `x ≥ y`, `x ≠ y`

## 5. Reserved Function Names

### 5.1 Trigonometric

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `sin(x)` | `\sin\left(x\right)` | sin(*x*) |
| `cos(x)` | `\cos\left(x\right)` | cos(*x*) |
| `tan(x)` | `\tan\left(x\right)` | tan(*x*) |
| `cot(x)` | `\cot\left(x\right)` | cot(*x*) |
| `sec(x)` | `\sec\left(x\right)` | sec(*x*) |
| `csc(x)` | `\csc\left(x\right)` | csc(*x*) |

### 5.2 Inverse Trigonometric

| Plain Text | Alias | LaTeX |
|---|---|---|
| `arcsin(x)` | `asin(x)` | `\arcsin\left(x\right)` |
| `arccos(x)` | `acos(x)` | `\arccos\left(x\right)` |
| `arctan(x)` | `atan(x)` | `\arctan\left(x\right)` |

### 5.3 Hyperbolic

| Plain Text | LaTeX |
|---|---|
| `sinh(x)` | `\sinh\left(x\right)` |
| `cosh(x)` | `\cosh\left(x\right)` |
| `tanh(x)` | `\tanh\left(x\right)` |

### 5.4 Exponential and Logarithmic

| Plain Text | LaTeX | Notes |
|---|---|---|
| `exp(x)` | `\exp\left(x\right)` | Exponential |
| `ln(x)` | `\ln\left(x\right)` | Natural logarithm |
| `log(x)` | `\log\left(x\right)` | Natural logarithm (CAS convention) |
| `log(b, x)` | `\log_{b}\left(x\right)` | Logarithm base *b* |
| `lg(x)` | `\log_2\left(x\right)` | Binary logarithm (alias: `log2`) |

### 5.5 Other Standard Functions

| Plain Text | LaTeX | Notes |
|---|---|---|
| `sqrt(x)` | `\sqrt{x}` | Square root |
| `abs(x)` | `\lvert x \rvert` | Absolute value |
| `sgn(x)` | `\operatorname{sgn}\left(x\right)` | Sign function (alias: `sign`) |
| `floor(x)` | `\lfloor x \rfloor` | Floor |
| `ceil(x)` | `\lceil x \rceil` | Ceiling |
| `max(a, b)` | `\max\left(a, b\right)` | Maximum |
| `min(a, b)` | `\min\left(a, b\right)` | Minimum |
| `atan2(y, x)` | `\operatorname{atan2}\left(y, x\right)` | Two-argument arctangent |
| `cbrt(x)` | `\sqrt[3]{x}` | Cube root |
| `round(x)` | `\operatorname{round}\left(x\right)` | Rounding |
| `pow(x, n)` | `x^{n}` | Power (functional form) |
| `trunc(x)` | `\operatorname{trunc}\left(x\right)` | Truncation |
| `clamp(x, a, b)` | `\operatorname{clamp}\left(x, a, b\right)` | Clamp to range |
| `det(A)` | `\det\left(A\right)` | Determinant |
| `tr(A)` | `\operatorname{tr}\left(A\right)` | Trace |
| `rank(A)` | `\operatorname{rank}\left(A\right)` | Matrix rank |

### 5.6 Special Functions (planned)

| Plain Text | LaTeX | Description |
|---|---|---|
| `gamma(z)` | `\Gamma\left(z\right)` | Gamma function |
| `beta(a, b)` | `B\left(a, b\right)` | Beta function |
| `erf(x)` | `\operatorname{erf}\left(x\right)` | Error function |
| `zeta(s)` | `\zeta\left(s\right)` | Riemann zeta |
| `bessel_j(n, x)` | `J_n\left(x\right)` | Bessel function first kind |

## 6. Calculus

### 6.1 Derivatives

| Plain Text | LaTeX | Rendered | Notes |
|---|---|---|---|
| `diff(y, x)` | `\frac{d}{dx}y` | dy/dx | First derivative (CAS standard) |
| `diff(y, x, 2)` | `\frac{d^2}{dx^2}y` | d&sup2;y/dx&sup2; | Second derivative |
| `diff(y, x, n)` | `\frac{d^n}{dx^n}y` | d&#x207F;y/dx&#x207F; | nth derivative |
| `dy/dx` | `\frac{dy}{dx}` | dy/dx | Leibniz shorthand |
| `d2y/dx2` | `\frac{d^2y}{dx^2}` | d&sup2;y/dx&sup2; | Higher-order Leibniz |
| `y'` | — | y&prime; | Prime notation (var implicit) |
| `y''` | — | y&Prime; | Second derivative |
| `y'''` | — | y&prime;&prime;&prime; | Third derivative |

The **recommended form** is `diff(expr, var[, order])` — it is unambiguous and matches SymPy, Maple, MATLAB, Maxima, and SageMath.

### 6.2 Partial Derivatives

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `partial(f, x)` | `\frac{\partial f}{\partial x}` | &part;f/&part;x |
| `partial(f, x, 2)` | `\frac{\partial^2 f}{\partial x^2}` | &part;&sup2;f/&part;x&sup2; |
| `partial(f, x, y)` | `\frac{\partial^2 f}{\partial x \partial y}` | &part;&sup2;f/&part;x&part;y |

Note: `diff(expr, var)` also works for partial derivatives when the expression contains multiple free variables — the interpretation is context-dependent, matching CAS conventions.

### 6.3 Integrals (planned)

Based on CAS consensus, the following forms will be supported:

| Plain Text | LaTeX | Rendered | Notes |
|---|---|---|---|
| `integrate(f, x)` | `\int f \, dx` | &int;f dx | Indefinite integral |
| `integrate(f, x, a, b)` | `\int_a^b f \, dx` | &int;&#x2090;&#x1D47; f dx | Definite integral |

Aliases: `integral`, `int`

### 6.4 Summation (planned)

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `sum(f, i, 1, n)` | `\sum_{i=1}^{n} f` | &Sigma;&#x2071;&#x3D;&#x2081;&#x207F; f |

Aliases: `summation`, `Sum`

### 6.5 Products (planned)

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `product(f, k, 1, n)` | `\prod_{k=1}^{n} f` | &Pi;&#x2096;&#x3D;&#x2081;&#x207F; f |

Aliases: `prod`, `Product`

### 6.6 Limits (planned)

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `limit(f, x, a)` | `\lim_{x \to a} f` | lim&#x2093;&#x2192;&#x2090; f |
| `limit(f, x, a, "+")` | `\lim_{x \to a^+} f` | Right-hand limit |
| `limit(f, x, a, "-")` | `\lim_{x \to a^-} f` | Left-hand limit |

Aliases: `lim`, `Limit`

## 7. Vector Calculus

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `grad(f)` | `\nabla f` | &nabla;f |
| `nabla(f)` | `\nabla f` | &nabla;f |
| `div(F)` | `\nabla \cdot F` | &nabla;&middot;F |
| `curl(F)` | `\nabla \times F` | &nabla;&times;F |
| `laplacian(f)` | `\nabla^2 f` | &nabla;&sup2;f |
| `dot(a, b)` | `a \cdot b` | a&middot;b |
| `cross(a, b)` | `a \times b` | a&times;b |

Unicode alternative: `∇f` is equivalent to `grad(f)`. All vector calculus operators work with or without parentheses: `grad f` = `grad(f)`.

## 8. Logic

| Plain Text | Unicode | LaTeX | Rendered |
|---|---|---|---|
| `x and y` | `x ∧ y` | `x \land y` | x &and; y |
| `x or y` | `x ∨ y` | `x \lor y` | x &or; y |
| `not x` | `¬x` | `\lnot x` | &not;x |
| `x implies y` | `x → y` | `x \implies y` | x &rArr; y |
| `x iff y` | `x ↔ y` | `x \iff y` | x &hArr; y |

## 9. Quantifiers

| Plain Text | LaTeX | Rendered |
|---|---|---|
| `forall x, P(x)` | `\forall x, P(x)` | &forall;x, P(x) |
| `forall x in S, P(x)` | `\forall x \in S, P(x)` | &forall;x &in; S, P(x) |
| `exists x, P(x)` | `\exists x, P(x)` | &exist;x, P(x) |
| `exists x in S, P(x)` | `\exists x \in S, P(x)` | &exist;x &in; S, P(x) |

## 10. Set Operations

| Plain Text | Unicode | LaTeX | Rendered |
|---|---|---|---|
| `x union y` | `x ∪ y` | `x \cup y` | x &cup; y |
| `x intersect y` | `x ∩ y` | `x \cap y` | x &cap; y |
| `x in S` | `x ∈ S` | `x \in S` | x &in; S |
| `x notin S` | `x ∉ S` | `x \notin S` | x &notin; S |

## 11. Transforms (planned)

| Plain Text | LaTeX | Description |
|---|---|---|
| `laplace(f, t, s)` | `\mathcal{L}\{f\}` | Laplace transform |
| `fourier(f, t, omega)` | `\mathcal{F}\{f\}` | Fourier transform |
| `ilaplace(F, s, t)` | `\mathcal{L}^{-1}\{F\}` | Inverse Laplace |
| `ifourier(F, omega, t)` | `\mathcal{F}^{-1}\{F\}` | Inverse Fourier |

---

## Appendix: Alias Table

| Canonical | Aliases |
|---|---|
| `arcsin` | `asin` |
| `arccos` | `acos` |
| `arctan` | `atan` |
| `sgn` | `sign` |
| `lg` | `log2` |
| `grad` | `nabla` |
| `integrate` | `integral`, `int` |
| `sum` | `summation`, `Sum` |
| `product` | `prod`, `Product` |
| `limit` | `lim`, `Limit` |
| `laplace` | `laplace_transform` |
| `fourier` | `fourier_transform` |
