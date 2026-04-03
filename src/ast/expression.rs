//! The main Expression AST node type.

use super::{
    BinaryOp, Direction, InequalityOp, IntegralBounds, LogicalOp, MathConstant, MathFloat,
    MultipleBounds, NumberSet, RelationOp, SetOp, SetRelation, TensorIndex, UnaryOp,
    VectorNotation,
};

/// The main AST node type representing mathematical expressions.
///
/// This enum covers the full range of mathematical expressions that mathlex can parse,
/// including basic arithmetic, functions, calculus operations, linear algebra, and equations.
///
/// # Variants
///
/// ## Basic Values
/// - [`Integer`](Expression::Integer): Integer literals (e.g., 42)
/// - [`Float`](Expression::Float): Floating-point literals (e.g., 3.14)
/// - [`Rational`](Expression::Rational): Rational numbers (e.g., 1/2)
/// - [`Complex`](Expression::Complex): Complex numbers (e.g., 3+4i)
/// - [`Variable`](Expression::Variable): Variables (e.g., x, y)
/// - [`Constant`](Expression::Constant): Mathematical constants (π, e, i)
///
/// ## Operations
/// - [`Binary`](Expression::Binary): Binary operations (e.g., x + y)
/// - [`Unary`](Expression::Unary): Unary operations (e.g., -x, x!)
/// - [`Function`](Expression::Function): Function calls (e.g., sin(x))
///
/// ## Calculus
/// - [`Derivative`](Expression::Derivative): Ordinary derivatives (dx/dt)
/// - [`PartialDerivative`](Expression::PartialDerivative): Partial derivatives (∂f/∂x)
/// - [`Integral`](Expression::Integral): Integrals (∫ f(x) dx)
/// - [`Limit`](Expression::Limit): Limits (lim x→a f(x))
/// - [`Sum`](Expression::Sum): Summations (Σ)
/// - [`Product`](Expression::Product): Products (Π)
///
/// ## Linear Algebra
/// - [`Vector`](Expression::Vector): Vectors ([1, 2, 3])
/// - [`Matrix`](Expression::Matrix): Matrices ([[1, 2], [3, 4]])
///
/// ## Equations
/// - [`Equation`](Expression::Equation): Equations (x = y)
/// - [`Inequality`](Expression::Inequality): Inequalities (x < y)
///
/// # Examples
///
/// ```
/// use mathlex::ast::{Expression, BinaryOp, MathConstant};
///
/// // 2 * π
/// let expr = Expression::Binary {
///     op: BinaryOp::Mul,
///     left: Box::new(Expression::Integer(2)),
///     right: Box::new(Expression::Constant(MathConstant::Pi)),
/// };
///
/// // Pattern match to verify structure
/// match expr {
///     Expression::Binary { op: BinaryOp::Mul, .. } => println!("Multiplication expression"),
///     _ => panic!("Unexpected expression"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expression {
    /// Integer literal.
    ///
    /// Represents whole numbers, both positive and negative.
    ///
    /// # Examples
    /// - `42`
    /// - `-17`
    /// - `0`
    Integer(i64),

    /// Floating-point literal.
    ///
    /// Represents decimal numbers.
    ///
    /// # Examples
    /// - `3.14`
    /// - `-2.5`
    /// - `1.0e-10`
    Float(MathFloat),

    /// Rational number (fraction).
    ///
    /// Represents a ratio of two expressions as numerator/denominator.
    ///
    /// ## Important Notes
    ///
    /// - **Fields are `Expression`, not `i64`**: This allows symbolic rationals like `x/y`
    ///   or `(a+b)/(c+d)`, not just numeric fractions.
    /// - **Not produced by parsers**: Current parsers represent divisions as
    ///   `Binary { op: Div, ... }`. This variant is available for programmatic construction,
    ///   typically by symbolic manipulation libraries that want to represent simplified
    ///   rational forms.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // Numeric rational: 1/2
    /// let half = Expression::Rational {
    ///     numerator: Box::new(Expression::Integer(1)),
    ///     denominator: Box::new(Expression::Integer(2)),
    /// };
    ///
    /// // Symbolic rational: x/y
    /// let symbolic = Expression::Rational {
    ///     numerator: Box::new(Expression::Variable("x".to_string())),
    ///     denominator: Box::new(Expression::Variable("y".to_string())),
    /// };
    /// ```
    Rational {
        /// Numerator of the fraction (any expression)
        numerator: Box<Expression>,

        /// Denominator of the fraction (any expression)
        denominator: Box<Expression>,
    },

    /// Complex number.
    ///
    /// Represents a number with real and imaginary components in the form `a + bi`.
    ///
    /// ## Important Notes
    ///
    /// - **Fields are `Expression`, not numeric types**: This allows symbolic complex numbers
    ///   like `(x+y) + (z+w)i`, not just numeric values.
    /// - **Not produced by parsers**: Current parsers represent complex expressions using
    ///   `Binary` operations with the imaginary constant `i`. This variant is available
    ///   for programmatic construction by libraries that want to represent simplified
    ///   complex number forms.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // Numeric complex: 3 + 4i
    /// let complex = Expression::Complex {
    ///     real: Box::new(Expression::Integer(3)),
    ///     imaginary: Box::new(Expression::Integer(4)),
    /// };
    ///
    /// // Symbolic complex: (a+b) + (c+d)i
    /// let symbolic = Expression::Complex {
    ///     real: Box::new(Expression::Variable("a".to_string())),
    ///     imaginary: Box::new(Expression::Variable("c".to_string())),
    /// };
    ///
    /// // Pure imaginary: 0 + i
    /// let pure_imaginary = Expression::Complex {
    ///     real: Box::new(Expression::Integer(0)),
    ///     imaginary: Box::new(Expression::Integer(1)),
    /// };
    /// ```
    Complex {
        /// Real component (any expression)
        real: Box<Expression>,

        /// Imaginary component - coefficient of i (any expression)
        imaginary: Box<Expression>,
    },

    /// Quaternion number in canonical form a + bi + cj + dk.
    ///
    /// Represents a quaternion with four components using the standard basis {1, i, j, k}.
    ///
    /// ## Important Notes
    ///
    /// - **Fields are `Expression`, not numeric types**: This allows symbolic quaternions
    ///   like `(a+b) + (c+d)i + ej + fk`, not just numeric values.
    /// - **Not produced by parsers**: Current parsers represent quaternion expressions using
    ///   `Binary` operations with the quaternion basis constants. This variant is available
    ///   for programmatic construction by libraries that want to represent simplified
    ///   quaternion forms.
    ///
    /// ## Quaternion Algebra
    ///
    /// The basis elements satisfy:
    /// - i² = j² = k² = ijk = -1
    /// - ij = k, jk = i, ki = j
    /// - ji = -k, kj = -i, ik = -j
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // Numeric quaternion: 1 + 2i + 3j + 4k
    /// let quat = Expression::Quaternion {
    ///     real: Box::new(Expression::Integer(1)),
    ///     i: Box::new(Expression::Integer(2)),
    ///     j: Box::new(Expression::Integer(3)),
    ///     k: Box::new(Expression::Integer(4)),
    /// };
    ///
    /// // Pure quaternion: i + j + k (no real part)
    /// let pure = Expression::Quaternion {
    ///     real: Box::new(Expression::Integer(0)),
    ///     i: Box::new(Expression::Integer(1)),
    ///     j: Box::new(Expression::Integer(1)),
    ///     k: Box::new(Expression::Integer(1)),
    /// };
    ///
    /// // Symbolic quaternion
    /// let symbolic = Expression::Quaternion {
    ///     real: Box::new(Expression::Variable("a".to_string())),
    ///     i: Box::new(Expression::Variable("b".to_string())),
    ///     j: Box::new(Expression::Variable("c".to_string())),
    ///     k: Box::new(Expression::Variable("d".to_string())),
    /// };
    /// ```
    Quaternion {
        /// Real (scalar) component
        real: Box<Expression>,

        /// Coefficient of i basis vector
        i: Box<Expression>,

        /// Coefficient of j basis vector
        j: Box<Expression>,

        /// Coefficient of k basis vector
        k: Box<Expression>,
    },

    /// Variable identifier.
    ///
    /// Represents a symbolic variable name.
    ///
    /// # Examples
    /// - `x`
    /// - `theta`
    /// - `x_1`
    Variable(String),

    /// Mathematical constant.
    ///
    /// Represents well-known mathematical constants (π, e, i, ∞).
    ///
    /// # Examples
    /// - `π` (pi)
    /// - `e` (Euler's number)
    /// - `i` (imaginary unit)
    Constant(MathConstant),

    /// Binary operation.
    ///
    /// Represents an operation with two operands (left op right).
    ///
    /// # Examples
    /// - `x + y`
    /// - `2 * π`
    /// - `a^b`
    Binary {
        /// The binary operator
        op: BinaryOp,

        /// Left operand
        left: Box<Expression>,

        /// Right operand
        right: Box<Expression>,
    },

    /// Unary operation.
    ///
    /// Represents an operation with a single operand.
    ///
    /// # Examples
    /// - `-x` (negation)
    /// - `n!` (factorial)
    /// - `A'` (transpose)
    Unary {
        /// The unary operator
        op: UnaryOp,

        /// The operand
        operand: Box<Expression>,
    },

    /// Function call.
    ///
    /// Represents a function application with zero or more arguments.
    ///
    /// # Examples
    /// - `sin(x)`
    /// - `max(a, b, c)`
    /// - `f()`
    Function {
        /// Function name
        name: String,

        /// Function arguments (may be empty)
        args: Vec<Expression>,
    },

    /// Ordinary derivative.
    ///
    /// Represents the nth derivative of an expression with respect to a single variable.
    /// Used for derivatives of functions of one variable.
    ///
    /// ## Notation
    ///
    /// - First derivative: `d/dx f(x)` or `f'(x)` or `df/dx`
    /// - Second derivative: `d²/dx² f(x)` or `f''(x)` or `d²f/dx²`
    /// - nth derivative: `dⁿ/dxⁿ f(x)`
    ///
    /// ## Usage
    ///
    /// - **`order`**: Specifies the number of times to differentiate. Must be ≥ 1.
    /// - **`var`**: The variable with respect to which differentiation occurs.
    /// - **`expr`**: The expression being differentiated (can be any expression).
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // First derivative: d/dx(x²)
    /// let first_deriv = Expression::Derivative {
    ///     expr: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Pow,
    ///         left: Box::new(Expression::Variable("x".to_string())),
    ///         right: Box::new(Expression::Integer(2)),
    ///     }),
    ///     var: "x".to_string(),
    ///     order: 1,
    /// };
    ///
    /// // Second derivative: d²/dx²(sin(x))
    /// let second_deriv = Expression::Derivative {
    ///     expr: Box::new(Expression::Function {
    ///         name: "sin".to_string(),
    ///         args: vec![Expression::Variable("x".to_string())],
    ///     }),
    ///     var: "x".to_string(),
    ///     order: 2,
    /// };
    /// ```
    Derivative {
        /// The expression being differentiated
        expr: Box<Expression>,

        /// The variable to differentiate with respect to
        var: String,

        /// Order of differentiation (1 for first derivative, 2 for second, etc.)
        order: u32,
    },

    /// Partial derivative.
    ///
    /// Represents the nth partial derivative of a multivariable expression with respect
    /// to one variable, holding others constant.
    ///
    /// ## Notation
    ///
    /// - First partial: `∂f/∂x` or `∂/∂x f(x,y,z)`
    /// - Second partial: `∂²f/∂x²` or `∂²/∂x² f(x,y,z)`
    /// - nth partial: `∂ⁿf/∂xⁿ`
    ///
    /// ## Distinction from Derivative
    ///
    /// Use `PartialDerivative` when:
    /// - The expression is a function of multiple variables
    /// - You want to emphasize that other variables are held constant
    /// - Following standard multivariable calculus notation
    ///
    /// Use `Derivative` when:
    /// - The expression is a function of a single variable
    /// - Following ordinary differential calculus notation
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // First partial derivative: ∂/∂x(x²y)
    /// let partial = Expression::PartialDerivative {
    ///     expr: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Mul,
    ///         left: Box::new(Expression::Binary {
    ///             op: mathlex::ast::BinaryOp::Pow,
    ///             left: Box::new(Expression::Variable("x".to_string())),
    ///             right: Box::new(Expression::Integer(2)),
    ///         }),
    ///         right: Box::new(Expression::Variable("y".to_string())),
    ///     }),
    ///     var: "x".to_string(),
    ///     order: 1,
    /// };
    /// ```
    PartialDerivative {
        /// The expression being partially differentiated
        expr: Box<Expression>,

        /// The variable to partially differentiate with respect to
        var: String,

        /// Order of partial differentiation (1 for first, 2 for second, etc.)
        order: u32,
    },

    /// Integral.
    ///
    /// Represents both definite and indefinite integrals.
    ///
    /// ## Integral Types
    ///
    /// - **Indefinite integral**: `bounds = None`, represents `∫ f(x) dx`
    ///   - Result is a family of functions (antiderivative + C)
    ///   - No specific bounds of integration
    ///
    /// - **Definite integral**: `bounds = Some(...)`, represents `∫ₐᵇ f(x) dx`
    ///   - Evaluates the integral from lower bound `a` to upper bound `b`
    ///   - Result is a number or expression (not a function)
    ///
    /// ## Bounds
    ///
    /// When `bounds` is `Some(IntegralBounds)`, the bounds can be:
    /// - Numeric: `∫₀¹ f(x) dx`
    /// - Symbolic: `∫ₐᵇ f(x) dx`
    /// - Infinite: `∫₀^∞ f(x) dx` (use `Constant(Infinity)`)
    /// - Complex expressions: `∫_{a+b}^{c+d} f(x) dx`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, IntegralBounds};
    ///
    /// // Indefinite integral: ∫ x dx
    /// let indefinite = Expression::Integral {
    ///     integrand: Box::new(Expression::Variable("x".to_string())),
    ///     var: "x".to_string(),
    ///     bounds: None,
    /// };
    ///
    /// // Definite integral: ∫₀¹ x² dx
    /// let definite = Expression::Integral {
    ///     integrand: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Pow,
    ///         left: Box::new(Expression::Variable("x".to_string())),
    ///         right: Box::new(Expression::Integer(2)),
    ///     }),
    ///     var: "x".to_string(),
    ///     bounds: Some(IntegralBounds {
    ///         lower: Box::new(Expression::Integer(0)),
    ///         upper: Box::new(Expression::Integer(1)),
    ///     }),
    /// };
    /// ```
    Integral {
        /// The integrand (expression being integrated)
        integrand: Box<Expression>,

        /// The variable of integration
        var: String,

        /// Integration bounds (None for indefinite integral, Some for definite)
        bounds: Option<IntegralBounds>,
    },

    /// Multiple integral (double, triple, etc.).
    ///
    /// Represents integrals like ∬ f dA (double) or ∭ f dV (triple).
    /// The dimension indicates the number of integral signs.
    ///
    /// ## Dimension Semantics
    ///
    /// - **dimension = 2**: Double integral (∬), typically over an area
    /// - **dimension = 3**: Triple integral (∭), typically over a volume
    /// - **dimension > 3**: Higher-dimensional integrals
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // Double integral: ∬_R f(x,y) dy dx
    /// Expression::MultipleIntegral {
    ///     dimension: 2,
    ///     integrand: Box::new(f_expr),
    ///     bounds: None,
    ///     vars: vec!["y".to_string(), "x".to_string()],
    /// }
    /// ```
    MultipleIntegral {
        /// Number of integral signs (2=double, 3=triple)
        dimension: u8,

        /// The integrand expression
        integrand: Box<Expression>,

        /// Optional bounds for each variable
        bounds: Option<MultipleBounds>,

        /// Variables of integration in order
        vars: Vec<String>,
    },

    /// Closed/contour integral (line, surface, volume).
    ///
    /// Represents closed path integrals like ∮ (line), ∯ (surface), ∰ (volume).
    /// These indicate integration over a closed curve, surface, or volume.
    ///
    /// ## Dimension Semantics
    ///
    /// - **dimension = 1**: Line integral over closed curve (∮)
    /// - **dimension = 2**: Surface integral over closed surface (∯)
    /// - **dimension = 3**: Volume integral over closed volume (∰)
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // Line integral: ∮_C F · dr
    /// Expression::ClosedIntegral {
    ///     dimension: 1,
    ///     integrand: Box::new(f_dot_dr),
    ///     surface: Some("C".to_string()),
    ///     var: "r".to_string(),
    /// }
    /// ```
    ClosedIntegral {
        /// Dimension: 1=line (∮), 2=surface (∯), 3=volume (∰)
        dimension: u8,

        /// The integrand expression
        integrand: Box<Expression>,

        /// Optional surface/curve name (e.g., "S", "C")
        surface: Option<String>,

        /// Variable of integration
        var: String,
    },

    /// Limit.
    ///
    /// Represents the limit of an expression as a variable approaches a value.
    ///
    /// ## Direction of Approach
    ///
    /// - **`Direction::Both`**: Two-sided limit, standard notation `lim_{x→a} f(x)`
    ///   - The limit exists only if left and right limits agree
    /// - **`Direction::Left`**: Left-hand limit, `lim_{x→a⁻} f(x)`
    ///   - Approaches from values less than `a`
    /// - **`Direction::Right`**: Right-hand limit, `lim_{x→a⁺} f(x)`
    ///   - Approaches from values greater than `a`
    ///
    /// ## Limit Points
    ///
    /// The `to` field can be any expression:
    /// - **Finite values**: `lim_{x→0} f(x)`, `lim_{x→a} f(x)`
    /// - **Infinity**: `lim_{x→∞} f(x)` using `Constant(Infinity)`
    /// - **Negative infinity**: `lim_{x→-∞} f(x)` using `Constant(NegInfinity)`
    ///   or `Unary { op: Neg, operand: Constant(Infinity) }`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, Direction, MathConstant};
    ///
    /// // Two-sided limit: lim_{x→0} sin(x)/x
    /// let limit_both = Expression::Limit {
    ///     expr: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Div,
    ///         left: Box::new(Expression::Function {
    ///             name: "sin".to_string(),
    ///             args: vec![Expression::Variable("x".to_string())],
    ///         }),
    ///         right: Box::new(Expression::Variable("x".to_string())),
    ///     }),
    ///     var: "x".to_string(),
    ///     to: Box::new(Expression::Integer(0)),
    ///     direction: Direction::Both,
    /// };
    ///
    /// // Limit to infinity: lim_{x→∞} 1/x
    /// let limit_infinity = Expression::Limit {
    ///     expr: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Div,
    ///         left: Box::new(Expression::Integer(1)),
    ///         right: Box::new(Expression::Variable("x".to_string())),
    ///     }),
    ///     var: "x".to_string(),
    ///     to: Box::new(Expression::Constant(MathConstant::Infinity)),
    ///     direction: Direction::Both,
    /// };
    ///
    /// // Right-hand limit: lim_{x→0⁺} 1/x
    /// let limit_right = Expression::Limit {
    ///     expr: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Div,
    ///         left: Box::new(Expression::Integer(1)),
    ///         right: Box::new(Expression::Variable("x".to_string())),
    ///     }),
    ///     var: "x".to_string(),
    ///     to: Box::new(Expression::Integer(0)),
    ///     direction: Direction::Right,
    /// };
    /// ```
    Limit {
        /// The expression whose limit is being taken
        expr: Box<Expression>,

        /// The variable approaching the limit
        var: String,

        /// The value being approached (can be finite, infinite, or symbolic)
        to: Box<Expression>,

        /// Direction of approach (left, right, or both sides)
        direction: Direction,
    },

    /// Summation.
    ///
    /// Represents a sum over a range of values, using sigma notation: `Σ_{index=lower}^{upper} body`.
    ///
    /// ## Semantics
    ///
    /// Evaluates to: `body[index=lower] + body[index=lower+1] + ... + body[index=upper]`
    ///
    /// The `index` variable is bound within the `body` expression and takes on each integer
    /// value from `lower` to `upper` (inclusive).
    ///
    /// ## Bounds
    ///
    /// - **Numeric bounds**: `Σ_{i=1}^{10} i²` - explicit numeric start and end
    /// - **Symbolic bounds**: `Σ_{i=1}^{n} i` - upper bound is a variable
    /// - **Infinite bounds**: `Σ_{i=0}^{∞} 1/2^i` - upper bound is infinity
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // Sum of first n integers: Σ_{i=1}^{n} i
    /// let sum = Expression::Sum {
    ///     index: "i".to_string(),
    ///     lower: Box::new(Expression::Integer(1)),
    ///     upper: Box::new(Expression::Variable("n".to_string())),
    ///     body: Box::new(Expression::Variable("i".to_string())),
    /// };
    ///
    /// // Sum of squares: Σ_{k=1}^{10} k²
    /// let sum_squares = Expression::Sum {
    ///     index: "k".to_string(),
    ///     lower: Box::new(Expression::Integer(1)),
    ///     upper: Box::new(Expression::Integer(10)),
    ///     body: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Pow,
    ///         left: Box::new(Expression::Variable("k".to_string())),
    ///         right: Box::new(Expression::Integer(2)),
    ///     }),
    /// };
    /// ```
    Sum {
        /// The index variable (bound variable in the summation)
        index: String,

        /// Lower bound of summation (inclusive)
        lower: Box<Expression>,

        /// Upper bound of summation (inclusive)
        upper: Box<Expression>,

        /// The expression being summed (can reference index variable)
        body: Box<Expression>,
    },

    /// Product.
    ///
    /// Represents a product over a range of values, using pi notation: `Π_{index=lower}^{upper} body`.
    ///
    /// ## Semantics
    ///
    /// Evaluates to: `body[index=lower] * body[index=lower+1] * ... * body[index=upper]`
    ///
    /// The `index` variable is bound within the `body` expression and takes on each integer
    /// value from `lower` to `upper` (inclusive).
    ///
    /// ## Bounds
    ///
    /// - **Numeric bounds**: `Π_{i=1}^{n} i` - factorial-like product
    /// - **Symbolic bounds**: `Π_{k=1}^{m} (1 + x/k)` - upper bound is a variable
    /// - **Infinite bounds**: `Π_{n=1}^{∞} (1 - 1/n²)` - infinite product
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // Factorial: Π_{i=1}^{n} i
    /// let factorial = Expression::Product {
    ///     index: "i".to_string(),
    ///     lower: Box::new(Expression::Integer(1)),
    ///     upper: Box::new(Expression::Variable("n".to_string())),
    ///     body: Box::new(Expression::Variable("i".to_string())),
    /// };
    ///
    /// // Product of (1 + k): Π_{k=0}^{5} (1 + k)
    /// let product = Expression::Product {
    ///     index: "k".to_string(),
    ///     lower: Box::new(Expression::Integer(0)),
    ///     upper: Box::new(Expression::Integer(5)),
    ///     body: Box::new(Expression::Binary {
    ///         op: mathlex::ast::BinaryOp::Add,
    ///         left: Box::new(Expression::Integer(1)),
    ///         right: Box::new(Expression::Variable("k".to_string())),
    ///     }),
    /// };
    /// ```
    Product {
        /// The index variable (bound variable in the product)
        index: String,

        /// Lower bound of product (inclusive)
        lower: Box<Expression>,

        /// Upper bound of product (inclusive)
        upper: Box<Expression>,

        /// The expression being multiplied (can reference index variable)
        body: Box<Expression>,
    },

    /// Vector.
    ///
    /// Represents an ordered collection of expressions as a mathematical vector.
    ///
    /// ## Properties
    ///
    /// - **Elements**: Can be any expression type (integers, floats, variables, etc.)
    /// - **Dimension**: Determined by the number of elements (can be 0 for empty vector)
    /// - **Notation**: Typically displayed as `[a, b, c]` or as a column vector
    ///
    /// ## Use Cases
    ///
    /// - Position vectors: `[x, y, z]`
    /// - Numeric vectors: `[1, 2, 3]`
    /// - Mixed expressions: `[x, 2y, z+1]`
    /// - Empty vectors: `[]` (edge case, dimension 0)
    ///
    /// ## Operations
    ///
    /// Vectors can be operands in:
    /// - Binary operations (component-wise addition, scalar multiplication)
    /// - Unary operations (transpose to convert to row vector)
    /// - Function calls (norm, dot product, cross product)
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // 3D position vector: [1, 2, 3]
    /// let position = Expression::Vector(vec![
    ///     Expression::Integer(1),
    ///     Expression::Integer(2),
    ///     Expression::Integer(3),
    /// ]);
    ///
    /// // Symbolic vector: [x, y, z]
    /// let symbolic = Expression::Vector(vec![
    ///     Expression::Variable("x".to_string()),
    ///     Expression::Variable("y".to_string()),
    ///     Expression::Variable("z".to_string()),
    /// ]);
    ///
    /// // Empty vector (edge case)
    /// let empty = Expression::Vector(vec![]);
    /// ```
    Vector(Vec<Expression>),

    /// Matrix.
    ///
    /// Represents a 2D array of expressions organized in rows and columns.
    ///
    /// ## Properties
    ///
    /// - **Dimensions**: M×N where M is number of rows, N is number of columns
    /// - **Uniformity**: All rows should have the same length for a valid matrix
    /// - **Elements**: Each element can be any expression type
    /// - **Notation**: Displayed using brackets or parentheses: `[[a, b], [c, d]]`
    ///
    /// ## Special Cases
    ///
    /// - **Empty matrix**: `[]` - zero rows, dimension 0×0
    /// - **Row vector**: Single row like `[[1, 2, 3]]` - dimension 1×3
    /// - **Column vector**: Single column like `[[1], [2], [3]]` - dimension 3×1
    /// - **Scalar**: `[[x]]` - dimension 1×1
    ///
    /// ## Validation Note
    ///
    /// The AST itself does not enforce uniform row lengths. Consumers should validate
    /// that all rows have the same number of elements before performing matrix operations.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // 2×2 identity matrix
    /// let identity = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(0)],
    ///     vec![Expression::Integer(0), Expression::Integer(1)],
    /// ]);
    ///
    /// // 2×3 matrix with variables
    /// let matrix = Expression::Matrix(vec![
    ///     vec![
    ///         Expression::Variable("a".to_string()),
    ///         Expression::Variable("b".to_string()),
    ///         Expression::Variable("c".to_string()),
    ///     ],
    ///     vec![
    ///         Expression::Variable("d".to_string()),
    ///         Expression::Variable("e".to_string()),
    ///         Expression::Variable("f".to_string()),
    ///     ],
    /// ]);
    ///
    /// // 1×1 matrix (scalar-like)
    /// let scalar_matrix = Expression::Matrix(vec![
    ///     vec![Expression::Integer(42)],
    /// ]);
    /// ```
    ///
    /// **Note:** Parsers always produce rectangular matrices. Manual construction can create
    /// ragged matrices; use [`Expression::is_valid_matrix()`] or
    /// [`Expression::matrix_dimensions()`] to validate.
    Matrix(Vec<Vec<Expression>>),

    /// Equation.
    ///
    /// Represents an equality between two expressions.
    ///
    /// # Examples
    /// - `x = 5`
    /// - `y = 2x + 1`
    /// - `f(x) = x²`
    Equation {
        /// Left-hand side of the equation
        left: Box<Expression>,

        /// Right-hand side of the equation
        right: Box<Expression>,
    },

    /// Inequality.
    ///
    /// Represents an inequality comparison between two expressions.
    ///
    /// # Examples
    /// - `x < 5`
    /// - `y ≥ 0`
    /// - `a ≠ b`
    Inequality {
        /// The inequality operator
        op: InequalityOp,

        /// Left-hand side of the inequality
        left: Box<Expression>,

        /// Right-hand side of the inequality
        right: Box<Expression>,
    },

    /// Universal quantifier.
    ForAll {
        /// The bound variable
        variable: String,
        /// Optional domain restriction
        domain: Option<Box<Expression>>,
        /// The body expression
        body: Box<Expression>,
    },

    /// Existential quantifier.
    Exists {
        /// The bound variable
        variable: String,
        /// Optional domain restriction
        domain: Option<Box<Expression>>,
        /// The body expression
        body: Box<Expression>,
        /// Whether this is unique existence (∃!)
        unique: bool,
    },

    /// Logical expression.
    Logical {
        /// The logical operator
        op: LogicalOp,
        /// The operands
        operands: Vec<Expression>,
    },

    /// Marked vector with explicit notation style.
    ///
    /// Represents a vector variable with a specific visual notation
    /// (bold, arrow, hat, underline, or plain).
    ///
    /// # Examples
    /// - `\mathbf{v}` → Bold notation
    /// - `\vec{a}` → Arrow notation
    /// - `\hat{n}` → Hat notation (typically for unit vectors)
    MarkedVector {
        /// The vector name
        name: String,
        /// The notation style
        notation: VectorNotation,
    },

    /// Dot product (inner product) of two vectors.
    ///
    /// # Examples
    /// - `u · v`
    /// - `\mathbf{a} \cdot \mathbf{b}`
    DotProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    /// Cross product of two vectors.
    ///
    /// # Examples
    /// - `u × v`
    /// - `\mathbf{a} \times \mathbf{b}`
    CrossProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    /// Outer product (tensor product) of two vectors.
    ///
    /// # Examples
    /// - `u ⊗ v`
    /// - `\mathbf{a} \otimes \mathbf{b}`
    OuterProduct {
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },

    // ============================================================
    // Vector Calculus Expressions
    // ============================================================
    /// Gradient of a scalar field: ∇f.
    ///
    /// The gradient is a vector field pointing in the direction of
    /// greatest increase of the scalar field.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // ∇f (gradient of f)
    /// let grad = Expression::Gradient {
    ///     expr: Box::new(Expression::Variable("f".to_string())),
    /// };
    /// ```
    Gradient {
        /// The scalar field expression
        expr: Box<Expression>,
    },

    /// Divergence of a vector field: ∇·F.
    ///
    /// The divergence is a scalar field measuring the "outflow" of
    /// a vector field at each point.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // ∇·F (divergence of F)
    /// let div = Expression::Divergence {
    ///     field: Box::new(Expression::Variable("F".to_string())),
    /// };
    /// ```
    Divergence {
        /// The vector field expression
        field: Box<Expression>,
    },

    /// Curl of a vector field: ∇×F.
    ///
    /// The curl is a vector field measuring the rotation of a vector
    /// field at each point.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // ∇×F (curl of F)
    /// let curl = Expression::Curl {
    ///     field: Box::new(Expression::Variable("F".to_string())),
    /// };
    /// ```
    Curl {
        /// The vector field expression
        field: Box<Expression>,
    },

    /// Laplacian of a scalar field: ∇²f or Δf.
    ///
    /// The Laplacian is a scalar field equal to the divergence of the gradient.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // ∇²f (Laplacian of f)
    /// let laplacian = Expression::Laplacian {
    ///     expr: Box::new(Expression::Variable("f".to_string())),
    /// };
    /// ```
    Laplacian {
        /// The scalar field expression
        expr: Box<Expression>,
    },

    /// Raw nabla/del operator: ∇.
    ///
    /// Used when the nabla appears without an operand or in
    /// non-standard combinations.
    Nabla,

    // ============================================================
    // Linear Algebra Operations
    // ============================================================
    /// Determinant of a matrix: det(A) or |A|.
    ///
    /// Returns a scalar value representing the signed volume scaling factor
    /// of the linear transformation represented by the matrix.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // det(A)
    /// let det = Expression::Determinant {
    ///     matrix: Box::new(Expression::Variable("A".to_string())),
    /// };
    /// ```
    Determinant {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Trace of a matrix: tr(A).
    ///
    /// Returns the sum of the diagonal elements of a square matrix.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // tr(A)
    /// let trace = Expression::Trace {
    ///     matrix: Box::new(Expression::Variable("A".to_string())),
    /// };
    /// ```
    Trace {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Rank of a matrix: rank(A).
    ///
    /// Returns the dimension of the column space (or row space) of the matrix.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // rank(A)
    /// let rank = Expression::Rank {
    ///     matrix: Box::new(Expression::Variable("A".to_string())),
    /// };
    /// ```
    Rank {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Conjugate transpose (Hermitian adjoint): A†, A*, or A^H.
    ///
    /// The transpose of the complex conjugate of the matrix.
    /// For real matrices, this is just the transpose.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // A† (conjugate transpose)
    /// let adjoint = Expression::ConjugateTranspose {
    ///     matrix: Box::new(Expression::Variable("A".to_string())),
    /// };
    /// ```
    ConjugateTranspose {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    /// Matrix inverse: A⁻¹.
    ///
    /// The matrix that when multiplied by A gives the identity matrix.
    /// Only exists for square matrices with non-zero determinant.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // A⁻¹
    /// let inverse = Expression::MatrixInverse {
    ///     matrix: Box::new(Expression::Variable("A".to_string())),
    /// };
    /// ```
    MatrixInverse {
        /// The matrix expression
        matrix: Box<Expression>,
    },

    // ============================================================
    // Set Theory Expressions
    // ============================================================
    /// A standard number set (ℕ, ℤ, ℚ, ℝ, ℂ, ℍ).
    ///
    /// Represents one of the standard mathematical number sets.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // The real numbers ℝ
    /// Expression::NumberSetExpr(NumberSet::Real)
    /// ```
    NumberSetExpr(NumberSet),

    /// Binary set operation.
    ///
    /// Represents operations like union (∪), intersection (∩), etc.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // A ∪ B (union)
    /// Expression::SetOperation {
    ///     op: SetOp::Union,
    ///     left: Box::new(a),
    ///     right: Box::new(b),
    /// }
    /// ```
    SetOperation {
        /// The set operation
        op: SetOp,
        /// Left operand set
        left: Box<Expression>,
        /// Right operand set
        right: Box<Expression>,
    },

    /// Set membership or relation expression.
    ///
    /// Represents relations like x ∈ S, A ⊆ B, etc.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // x ∈ ℝ
    /// Expression::SetRelationExpr {
    ///     relation: SetRelation::In,
    ///     element: Box::new(Expression::Variable("x".to_string())),
    ///     set: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
    /// }
    /// ```
    SetRelationExpr {
        /// The relation type
        relation: SetRelation,
        /// The element (or left set for subset relations)
        element: Box<Expression>,
        /// The set (or right set for subset relations)
        set: Box<Expression>,
    },

    /// Set builder notation: {x | P(x)} or {x ∈ S | P(x)}.
    ///
    /// Defines a set by a predicate on its elements.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // {x ∈ ℝ | x > 0}
    /// Expression::SetBuilder {
    ///     variable: "x".to_string(),
    ///     domain: Some(Box::new(Expression::NumberSetExpr(NumberSet::Real))),
    ///     predicate: Box::new(x_greater_than_zero),
    /// }
    /// ```
    SetBuilder {
        /// The bound variable
        variable: String,
        /// Optional domain set
        domain: Option<Box<Expression>>,
        /// The predicate that defines membership
        predicate: Box<Expression>,
    },

    /// The empty set: ∅ or {}.
    EmptySet,

    /// Power set: 𝒫(S) - the set of all subsets of S.
    ///
    /// ## Examples
    ///
    /// ```ignore
    /// // 𝒫(A)
    /// Expression::PowerSet {
    ///     set: Box::new(Expression::Variable("A".to_string())),
    /// }
    /// ```
    PowerSet {
        /// The set to take the power set of
        set: Box<Expression>,
    },

    // ============================================================
    // Tensor Notation Expressions
    // ============================================================
    /// Tensor with indexed notation.
    ///
    /// Represents a tensor with upper and/or lower indices, supporting
    /// Einstein summation convention notation.
    ///
    /// ## Notation
    ///
    /// - `T^{ij}` - Tensor T with two upper indices
    /// - `T_{ab}` - Tensor T with two lower indices
    /// - `T^i_j` - Mixed tensor with one upper and one lower index
    /// - `R^a_{bcd}` - Riemann-like tensor with mixed indices
    ///
    /// ## Einstein Summation Convention
    ///
    /// When the same index appears once upper and once lower in a product,
    /// summation is implied. For example: `A^i B_i = Σ_i A^i B_i`.
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, TensorIndex, IndexType};
    ///
    /// // Metric tensor g^{μν}
    /// let metric = Expression::Tensor {
    ///     name: "g".to_string(),
    ///     indices: vec![
    ///         TensorIndex { name: "μ".to_string(), index_type: IndexType::Upper },
    ///         TensorIndex { name: "ν".to_string(), index_type: IndexType::Upper },
    ///     ],
    /// };
    ///
    /// // Mixed tensor T^i_j
    /// let mixed = Expression::Tensor {
    ///     name: "T".to_string(),
    ///     indices: vec![
    ///         TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
    ///         TensorIndex { name: "j".to_string(), index_type: IndexType::Lower },
    ///     ],
    /// };
    /// ```
    Tensor {
        /// The tensor name (e.g., "T", "g", "R", "Γ")
        name: String,

        /// The tensor indices in order of appearance
        indices: Vec<TensorIndex>,
    },

    /// Kronecker delta: δ^i_j or δ_{ij}.
    ///
    /// The Kronecker delta is 1 when indices are equal, 0 otherwise.
    /// It acts as the identity in index manipulation.
    ///
    /// ## Properties
    ///
    /// - δ^i_j = 1 if i = j, 0 otherwise
    /// - δ^i_j A^j = A^i (index substitution)
    /// - δ^i_i = n (trace in n dimensions)
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, TensorIndex, IndexType};
    ///
    /// // δ^i_j (mixed Kronecker delta)
    /// let delta = Expression::KroneckerDelta {
    ///     indices: vec![
    ///         TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
    ///         TensorIndex { name: "j".to_string(), index_type: IndexType::Lower },
    ///     ],
    /// };
    /// ```
    KroneckerDelta {
        /// The indices (typically two, one upper and one lower)
        indices: Vec<TensorIndex>,
    },

    /// Levi-Civita symbol: ε^{ijk} or ε_{ijk}.
    ///
    /// The totally antisymmetric symbol used in cross products,
    /// determinants, and differential forms.
    ///
    /// ## Properties
    ///
    /// - ε^{123} = 1 in 3D (even permutation)
    /// - Changes sign under index swap (antisymmetric)
    /// - ε^{ijk} = 0 if any two indices are equal
    ///
    /// ## Usage
    ///
    /// - Cross product: (a × b)^i = ε^{ijk} a_j b_k
    /// - Determinant: det(A) = ε^{i_1...i_n} A_{1i_1}...A_{ni_n}
    /// - Exterior algebra and differential forms
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, TensorIndex, IndexType};
    ///
    /// // ε^{ijk} (3D Levi-Civita with upper indices)
    /// let epsilon = Expression::LeviCivita {
    ///     indices: vec![
    ///         TensorIndex { name: "i".to_string(), index_type: IndexType::Upper },
    ///         TensorIndex { name: "j".to_string(), index_type: IndexType::Upper },
    ///         TensorIndex { name: "k".to_string(), index_type: IndexType::Upper },
    ///     ],
    /// };
    /// ```
    LeviCivita {
        /// The indices (typically 3 for 3D, n for nD)
        indices: Vec<TensorIndex>,
    },

    // ============================================================
    // Function Theory and Relations
    // ============================================================
    /// Function signature/mapping declaration: f: A → B
    ///
    /// Represents a function with its domain and codomain, commonly used in
    /// mathematical notation to declare the type of a function.
    ///
    /// ## Notation
    ///
    /// - LaTeX: `f: A \to B`
    /// - Plain text: `f: A → B`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, NumberSet};
    ///
    /// // f: ℝ → ℝ
    /// let real_func = Expression::FunctionSignature {
    ///     name: "f".to_string(),
    ///     domain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
    ///     codomain: Box::new(Expression::NumberSetExpr(NumberSet::Real)),
    /// };
    /// ```
    FunctionSignature {
        /// The function name
        name: String,

        /// The domain (input type/set)
        domain: Box<Expression>,

        /// The codomain (output type/set)
        codomain: Box<Expression>,
    },

    /// Function composition: f ∘ g
    ///
    /// Represents the composition of two functions where (f ∘ g)(x) = f(g(x)).
    /// The inner function g is applied first, followed by the outer function f.
    ///
    /// ## Notation
    ///
    /// - LaTeX: `f \circ g`
    /// - Unicode: `f ∘ g`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // f ∘ g
    /// let composition = Expression::Composition {
    ///     outer: Box::new(Expression::Variable("f".to_string())),
    ///     inner: Box::new(Expression::Variable("g".to_string())),
    /// };
    /// ```
    Composition {
        /// The outer function (applied second)
        outer: Box<Expression>,

        /// The inner function (applied first)
        inner: Box<Expression>,
    },

    // ============================================================
    // Differential Forms
    // ============================================================
    /// Differential of a variable: dx, dy, dt
    ///
    /// Represents the differential form of a single variable.
    /// Used in integration and differential geometry contexts.
    ///
    /// ## Important Notes
    ///
    /// - Represents a differential form, distinct from a derivative
    /// - Commonly appears in integrals: `∫ f(x) dx`
    /// - In differential geometry, differentials are 1-forms
    /// - Not to be confused with derivative notation `d/dx`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // dx
    /// let dx = Expression::Differential {
    ///     var: "x".to_string(),
    /// };
    ///
    /// // dt
    /// let dt = Expression::Differential {
    ///     var: "t".to_string(),
    /// };
    /// ```
    Differential {
        /// The variable name (without the 'd' prefix)
        var: String,
    },

    /// Wedge product for differential forms: dx ∧ dy
    ///
    /// The wedge product (exterior product) of two differential forms.
    /// Used in differential geometry and multivariable calculus.
    ///
    /// ## Properties
    ///
    /// - Anticommutative: `dx ∧ dy = -(dy ∧ dx)`
    /// - Associative: `(dx ∧ dy) ∧ dz = dx ∧ (dy ∧ dz)`
    /// - Wedge with itself is zero: `dx ∧ dx = 0`
    ///
    /// ## Common Uses
    ///
    /// - Area/volume elements in integration: `dx ∧ dy`, `dx ∧ dy ∧ dz`
    /// - Exterior calculus and differential forms
    /// - Differential geometry
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// // dx ∧ dy
    /// let dx = Expression::Differential { var: "x".to_string() };
    /// let dy = Expression::Differential { var: "y".to_string() };
    /// let wedge = Expression::WedgeProduct {
    ///     left: Box::new(dx),
    ///     right: Box::new(dy),
    /// };
    ///
    /// // dx ∧ dy ∧ dz (nested)
    /// let dz = Expression::Differential { var: "z".to_string() };
    /// let wedge_3d = Expression::WedgeProduct {
    ///     left: Box::new(wedge),
    ///     right: Box::new(dz),
    /// };
    /// ```
    WedgeProduct {
        /// Left operand (typically a differential or wedge product)
        left: Box<Expression>,
        /// Right operand (typically a differential)
        right: Box<Expression>,
    },

    /// Relation expression: a ~ b, a ≡ b, a ≅ b, a ≈ b
    ///
    /// Represents a mathematical relation between two expressions, such as
    /// similarity, equivalence, congruence, or approximation.
    ///
    /// ## Notation
    ///
    /// - Similar (~): `a \sim b`
    /// - Equivalent (≡): `a \equiv b`
    /// - Congruent (≅): `a \cong b`
    /// - Approximate (≈): `a \approx b`
    ///
    /// ## Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, RelationOp};
    ///
    /// // x ~ y (similarity)
    /// let similar = Expression::Relation {
    ///     op: RelationOp::Similar,
    ///     left: Box::new(Expression::Variable("x".to_string())),
    ///     right: Box::new(Expression::Variable("y".to_string())),
    /// };
    ///
    /// // a ≈ b (approximation)
    /// let approx = Expression::Relation {
    ///     op: RelationOp::Approx,
    ///     left: Box::new(Expression::Variable("a".to_string())),
    ///     right: Box::new(Expression::Variable("b".to_string())),
    /// };
    /// ```
    Relation {
        /// The relation operator
        op: RelationOp,

        /// Left operand
        left: Box<Expression>,

        /// Right operand
        right: Box<Expression>,
    },
}
