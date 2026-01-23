//! # Abstract Syntax Tree (AST) Types
//!
//! This module defines the core AST types used to represent mathematical expressions.
//! The AST is the contract between mathlex parsers and consuming libraries.
//!
//! ## Design Philosophy
//!
//! - **Format Agnostic**: The same mathematical concept produces the same AST regardless
//!   of input format (LaTeX or plain text)
//! - **Structural Representation**: AST nodes represent syntax, not evaluated values
//! - **Complete Coverage**: Supports algebra, calculus, linear algebra, and equations
//!
//! ## Key Types
//!
//! - [`Expression`]: The main AST node type representing any mathematical expression
//! - [`MathConstant`]: Mathematical constants (π, e, i, ∞)
//! - [`BinaryOp`]: Binary operators (+, -, *, /, ^, %)
//! - [`UnaryOp`]: Unary operators (negation, factorial, transpose)
//! - [`MathFloat`]: Wrapper for f64 with proper equality and hashing semantics
//!
//! ## AST Semantics and Conventions
//!
//! ### Expression Types
//!
//! - **`Rational`**: Contains `Expression` fields (not `i64`), allowing symbolic rationals
//!   like `x/y`. This enables representation of unevaluated rational expressions where
//!   numerator and denominator are arbitrary expressions.
//! - **`Complex`**: Contains `Expression` fields for real and imaginary parts, enabling
//!   symbolic complex numbers like `(a+b)+(c+d)i` rather than just numeric values.
//! - **`MathFloat`**: Wraps `OrderedFloat<f64>` to provide proper `Hash` and `Eq` implementations.
//!   NaN values are comparable (NaN == NaN), which differs from standard IEEE 754 semantics
//!   but is necessary for use in hash-based collections.
//!
//! ### Known Limitations
//!
//! - **`Rational` and `Complex` variants are not produced by parsers** in the current implementation.
//!   These variants are available for programmatic construction by consumers of the AST,
//!   allowing symbolic manipulation libraries to build complex expressions.
//! - **Some ASTs don't round-trip perfectly** due to precedence and formatting choices.
//!   For example, `(2 + 3) * 4` and `2 + 3 * 4` have different ASTs but the first may display
//!   without parentheses depending on context.
//! - **`MathConstant::NegInfinity` requires explicit parsing** as a distinct constant.
//!   The expression `-∞` is parsed as `Unary { op: Neg, operand: Constant(Infinity) }`,
//!   not as `Constant(NegInfinity)`.
//!
//! ### Serialization Notes
//!
//! - **`Display` trait**: Uses minimal parentheses based on operator precedence. The output
//!   is human-readable but may omit parentheses where they can be inferred from precedence rules.
//! - **`ToLatex` trait**: Produces standard LaTeX notation that can be re-parsed by the LaTeX
//!   parser. This is the recommended format for round-trip serialization.
//! - **Special float values**: When using JSON serialization (via serde), NaN and Infinity
//!   values serialize to `null` per JSON specification. For lossless serialization of special
//!   floats, use binary formats like bincode.
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Representing: 2 * π
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Integer(2)),
//!     right: Box::new(Expression::Constant(MathConstant::Pi)),
//! };
//!
//! // Verify structure
//! match expr {
//!     Expression::Binary { op: BinaryOp::Mul, .. } => println!("It's multiplication!"),
//!     _ => panic!("Unexpected expression type"),
//! }
//! ```

use ordered_float::OrderedFloat;
use std::fmt;

/// Wrapper type for f64 that provides proper equality and hashing semantics.
///
/// This type wraps `ordered_float::OrderedFloat<f64>` to enable `f64` values
/// to be used in `Expression` variants while implementing `PartialEq`, `Eq`, and `Hash`.
///
/// ## Semantics
///
/// - **NaN equality**: Unlike standard IEEE 754, `NaN == NaN` returns `true` for `MathFloat`.
///   This is necessary for use in hash-based collections like `HashSet` and `HashMap`.
/// - **Ordering**: Values are totally ordered, with NaN considered greater than infinity.
/// - **Hash stability**: Identical float values produce identical hashes, including special
///   values like NaN, Infinity, and -Infinity.
///
/// ## Use Cases
///
/// Use this type when you need to store floating-point values in collections that require
/// `Eq` and `Hash`, or when building AST nodes that will be compared for equality.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::MathFloat;
/// use std::collections::HashSet;
///
/// let f1 = MathFloat::from(3.14);
/// let f2 = MathFloat::from(3.14);
/// assert_eq!(f1, f2);
///
/// let value: f64 = f1.into();
/// assert_eq!(value, 3.14);
///
/// // NaN values are equal to themselves
/// let nan1 = MathFloat::from(f64::NAN);
/// let nan2 = MathFloat::from(f64::NAN);
/// assert_eq!(nan1, nan2);
///
/// // Can be used in HashSet
/// let mut set = HashSet::new();
/// set.insert(MathFloat::from(1.0));
/// set.insert(MathFloat::from(2.0));
/// assert_eq!(set.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MathFloat(OrderedFloat<f64>);

impl MathFloat {
    /// Creates a new MathFloat from an f64 value.
    #[inline]
    pub fn new(value: f64) -> Self {
        Self(OrderedFloat(value))
    }

    /// Returns the inner f64 value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.0.into_inner()
    }
}

impl From<f64> for MathFloat {
    #[inline]
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl From<MathFloat> for f64 {
    #[inline]
    fn from(math_float: MathFloat) -> Self {
        math_float.value()
    }
}

impl fmt::Display for MathFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Mathematical constants used in expressions.
///
/// These represent well-known mathematical constants with precise mathematical meaning.
///
/// ## Parsing Notes
///
/// - **`Pi`**: Parsed from `π` (Unicode) or `\pi` (LaTeX)
/// - **`E`**: Parsed from `e` (plain text) or `e` (LaTeX)
/// - **`I`**: Parsed from `i` (plain text) or `i` (LaTeX), represents the imaginary unit
/// - **`Infinity`**: Parsed from `∞` (Unicode) or `\infty` (LaTeX)
/// - **`NegInfinity`**: Not directly produced by parsers. The input `-∞` is parsed as
///   `Unary { op: Neg, operand: Constant(Infinity) }`. This variant exists for programmatic
///   construction and simplification by consumers.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::MathConstant;
///
/// let pi = MathConstant::Pi;
/// let euler = MathConstant::E;
/// assert_ne!(pi, euler);
///
/// // Note: NegInfinity is for programmatic use
/// let neg_inf = MathConstant::NegInfinity;
/// assert_ne!(neg_inf, MathConstant::Infinity);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathConstant {
    /// The mathematical constant π (pi), approximately 3.14159...
    Pi,

    /// Euler's number e, approximately 2.71828...
    E,

    /// The imaginary unit i, where i² = -1
    I,

    /// Positive infinity (∞)
    Infinity,

    /// Negative infinity (-∞)
    NegInfinity,
}

/// Binary operators for mathematical expressions.
///
/// Represents operators that take two operands (left and right).
///
/// ## Operator Precedence
///
/// When displaying or parsing expressions, operators follow standard mathematical precedence:
/// 1. **`Pow`** (highest) - Exponentiation: `^`
/// 2. **`Mul`, `Div`, `Mod`** - Multiplication, division, modulo: `*`, `/`, `%`
/// 3. **`Add`, `Sub`** - Addition, subtraction: `+`, `-`
/// 4. **`PlusMinus`, `MinusPlus`** (lowest) - Combined operators: `±`, `∓`
///
/// ## Usage Notes
///
/// - **Associativity**: Most operators are left-associative except `Pow`, which is right-associative.
///   For example, `2^3^4` is parsed as `2^(3^4)`, not `(2^3)^4`.
/// - **PlusMinus and MinusPlus**: These represent the special combined operators `±` and `∓`,
///   commonly used in mathematics to indicate dual solutions (e.g., `x = 1 ± 2`).
///
/// ## Examples
///
/// ```
/// use mathlex::ast::{BinaryOp, Expression};
///
/// let add = BinaryOp::Add;  // +
/// let pow = BinaryOp::Pow;  // ^
/// assert_ne!(add, pow);
///
/// // Right-associative power: 2^3^4 is 2^(3^4)
/// let expr = Expression::Binary {
///     op: BinaryOp::Pow,
///     left: Box::new(Expression::Integer(2)),
///     right: Box::new(Expression::Binary {
///         op: BinaryOp::Pow,
///         left: Box::new(Expression::Integer(3)),
///         right: Box::new(Expression::Integer(4)),
///     }),
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinaryOp {
    /// Addition operator (+)
    Add,

    /// Subtraction operator (-)
    Sub,

    /// Multiplication operator (*)
    Mul,

    /// Division operator (/)
    Div,

    /// Exponentiation operator (^)
    Pow,

    /// Modulo operator (%)
    Mod,

    /// Plus-minus operator (±)
    PlusMinus,

    /// Minus-plus operator (∓)
    MinusPlus,
}

/// Unary operators for mathematical expressions.
///
/// Represents operators that take a single operand.
///
/// ## Operator Semantics
///
/// - **`Neg`**: Arithmetic negation (`-x`). Applied as a prefix operator.
/// - **`Pos`**: Unary plus (`+x`). Applied as a prefix operator. Usually redundant but
///   can be explicitly represented in the AST.
/// - **`Factorial`**: Factorial operator (`n!`). Applied as a postfix operator.
///   Typically used with non-negative integers.
/// - **`Transpose`**: Matrix or vector transpose (`Aᵀ` or `A'`). Applied as a postfix
///   operator. Used in linear algebra contexts.
///
/// ## Position
///
/// - **Prefix operators**: `Neg`, `Pos` - appear before the operand
/// - **Postfix operators**: `Factorial`, `Transpose` - appear after the operand
///
/// ## Examples
///
/// ```
/// use mathlex::ast::{UnaryOp, Expression};
///
/// // Negation: -5
/// let neg_expr = Expression::Unary {
///     op: UnaryOp::Neg,
///     operand: Box::new(Expression::Integer(5)),
/// };
///
/// // Factorial: n!
/// let fact_expr = Expression::Unary {
///     op: UnaryOp::Factorial,
///     operand: Box::new(Expression::Variable("n".to_string())),
/// };
///
/// // Transpose: A'
/// let transpose_expr = Expression::Unary {
///     op: UnaryOp::Transpose,
///     operand: Box::new(Expression::Variable("A".to_string())),
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnaryOp {
    /// Negation operator (-)
    Neg,

    /// Positive sign operator (+)
    Pos,

    /// Factorial operator (!)
    Factorial,

    /// Matrix/vector transpose operator (ᵀ or ')
    Transpose,
}

/// Direction for limit evaluation.
///
/// Specifies the direction from which a limit approaches a value.
///
/// # Examples
///
/// ```
/// use mathlex::ast::Direction;
///
/// let from_left = Direction::Left;   // lim x→a⁻
/// let from_right = Direction::Right; // lim x→a⁺
/// let both = Direction::Both;        // lim x→a
/// assert_ne!(from_left, both);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    /// Approach from the left (values less than the limit point)
    Left,

    /// Approach from the right (values greater than the limit point)
    Right,

    /// Approach from both sides (standard two-sided limit)
    Both,
}

/// Inequality operators for comparisons.
///
/// Represents relational operators used in inequalities.
///
/// # Examples
///
/// ```
/// use mathlex::ast::InequalityOp;
///
/// let less_than = InequalityOp::Lt;     // <
/// let less_equal = InequalityOp::Le;    // ≤
/// let not_equal = InequalityOp::Ne;     // ≠
/// assert_ne!(less_than, less_equal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InequalityOp {
    /// Less than (<)
    Lt,

    /// Less than or equal (≤)
    Le,

    /// Greater than (>)
    Gt,

    /// Greater than or equal (≥)
    Ge,

    /// Not equal (≠)
    Ne,
}


/// Logical operators for propositional logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LogicalOp {
    /// Logical conjunction (∧)
    And,
    /// Logical disjunction (∨)
    Or,
    /// Logical negation (¬)
    Not,
    /// Logical implication (→)
    Implies,
    /// Logical biconditional/equivalence (↔)
    Iff,
}

/// Bounds for definite integrals.
///
/// Represents the lower and upper bounds of integration.
///
/// # Examples
///
/// ```
/// use mathlex::ast::{IntegralBounds, Expression};
///
/// // Integral from 0 to 1
/// let bounds = IntegralBounds {
///     lower: Box::new(Expression::Integer(0)),
///     upper: Box::new(Expression::Integer(1)),
/// };
///
/// match (*bounds.lower, *bounds.upper) {
///     (Expression::Integer(0), Expression::Integer(1)) => println!("Bounds are 0 to 1"),
///     _ => panic!("Unexpected bounds"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntegralBounds {
    /// Lower bound of integration
    pub lower: Box<Expression>,

    /// Upper bound of integration
    pub upper: Box<Expression>,
}

/// Notation style for marking vectors.
///
/// Specifies how a vector is visually distinguished in mathematical notation.
/// Different fields use different conventions for marking vectors.
///
/// ## Usage in LaTeX
///
/// - **Bold**: `\mathbf{v}` - Common in physics and engineering
/// - **Arrow**: `\vec{v}` - Traditional notation, common in introductory texts
/// - **Hat**: `\hat{n}` - Typically used for unit vectors
/// - **Underline**: `\underline{v}` - Less common, sometimes used in handwriting
/// - **Plain**: No special notation - relies on context
///
/// ## Examples
///
/// ```
/// use mathlex::ast::VectorNotation;
///
/// let bold = VectorNotation::Bold;      // \mathbf{v}
/// let arrow = VectorNotation::Arrow;    // \vec{v}
/// let hat = VectorNotation::Hat;        // \hat{n}
/// let underline = VectorNotation::Underline;  // \underline{v}
/// let plain = VectorNotation::Plain;    // v
///
/// assert_ne!(bold, arrow);
/// assert_ne!(arrow, hat);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VectorNotation {
    /// Bold notation: **v** or `\mathbf{v}`
    Bold,

    /// Arrow notation: v⃗ or `\vec{v}`
    Arrow,

    /// Hat notation: v̂ or `\hat{v}` (typically for unit vectors)
    Hat,

    /// Underline notation: v̲ or `\underline{v}`
    Underline,

    /// Plain notation: v (no special marking)
    Plain,
}
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
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    // Tests for MathConstant
    #[test]
    fn test_math_constant_variants() {
        let pi = MathConstant::Pi;
        let e = MathConstant::E;
        let i = MathConstant::I;
        let inf = MathConstant::Infinity;
        let neg_inf = MathConstant::NegInfinity;

        // Verify they are all distinct
        assert_ne!(pi, e);
        assert_ne!(e, i);
        assert_ne!(i, inf);
        assert_ne!(inf, neg_inf);
    }

    #[test]
    fn test_math_constant_copy() {
        let pi = MathConstant::Pi;
        let pi_copy = pi;
        assert_eq!(pi, pi_copy);
    }

    #[test]
    fn test_math_constant_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MathConstant::Pi);
        set.insert(MathConstant::E);
        set.insert(MathConstant::Pi); // Duplicate

        assert_eq!(set.len(), 2); // Only Pi and E
    }

    // Tests for BinaryOp
    #[test]
    fn test_binary_op_variants() {
        let add = BinaryOp::Add;
        let sub = BinaryOp::Sub;
        let mul = BinaryOp::Mul;
        let div = BinaryOp::Div;
        let pow = BinaryOp::Pow;
        let modulo = BinaryOp::Mod;

        assert_ne!(add, sub);
        assert_ne!(mul, div);
        assert_ne!(pow, modulo);
    }

    #[test]
    fn test_binary_op_copy() {
        let add = BinaryOp::Add;
        let add_copy = add;
        assert_eq!(add, add_copy);
    }

    #[test]
    fn test_binary_op_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(BinaryOp::Add, "addition");
        map.insert(BinaryOp::Mul, "multiplication");

        assert_eq!(map.get(&BinaryOp::Add), Some(&"addition"));
        assert_eq!(map.get(&BinaryOp::Mul), Some(&"multiplication"));
    }

    // Tests for UnaryOp
    #[test]
    fn test_unary_op_variants() {
        let neg = UnaryOp::Neg;
        let pos = UnaryOp::Pos;
        let fact = UnaryOp::Factorial;
        let transpose = UnaryOp::Transpose;

        assert_ne!(neg, pos);
        assert_ne!(fact, transpose);
    }

    #[test]
    fn test_unary_op_copy() {
        let neg = UnaryOp::Neg;
        let neg_copy = neg;
        assert_eq!(neg, neg_copy);
    }

    // Tests for Direction
    #[test]
    fn test_direction_variants() {
        let left = Direction::Left;
        let right = Direction::Right;
        let both = Direction::Both;

        assert_ne!(left, right);
        assert_ne!(right, both);
        assert_ne!(left, both);
    }

    #[test]
    fn test_direction_copy() {
        let left = Direction::Left;
        let left_copy = left;
        assert_eq!(left, left_copy);
    }

    // Tests for InequalityOp
    #[test]
    fn test_inequality_op_variants() {
        let lt = InequalityOp::Lt;
        let le = InequalityOp::Le;
        let gt = InequalityOp::Gt;
        let ge = InequalityOp::Ge;
        let ne = InequalityOp::Ne;

        assert_ne!(lt, le);
        assert_ne!(gt, ge);
        assert_ne!(lt, gt);
        assert_ne!(ne, lt);
    }

    #[test]
    fn test_inequality_op_copy() {
        let lt = InequalityOp::Lt;
        let lt_copy = lt;
        assert_eq!(lt, lt_copy);
    }

    // Tests for IntegralBounds
    #[test]
    fn test_integral_bounds_creation() {
        let bounds = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        match (*bounds.lower, *bounds.upper) {
            (Expression::Integer(l), Expression::Integer(u)) => {
                assert_eq!(l, 0);
                assert_eq!(u, 1);
            }
            _ => panic!("Expected integer bounds"),
        }
    }

    #[test]
    fn test_integral_bounds_clone() {
        let bounds = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        let bounds_clone = bounds.clone();

        match (*bounds_clone.lower, *bounds_clone.upper) {
            (Expression::Integer(l), Expression::Integer(u)) => {
                assert_eq!(l, 0);
                assert_eq!(u, 1);
            }
            _ => panic!("Expected integer bounds"),
        }
    }

    // Tests for Expression - Integer
    #[test]
    fn test_expression_integer() {
        let expr = Expression::Integer(42);
        match expr {
            Expression::Integer(n) => assert_eq!(n, 42),
            _ => panic!("Expected Integer variant"),
        }
    }

    #[test]
    fn test_expression_integer_negative() {
        let expr = Expression::Integer(-17);
        match expr {
            Expression::Integer(n) => assert_eq!(n, -17),
            _ => panic!("Expected Integer variant"),
        }
    }

    #[test]
    fn test_expression_integer_clone() {
        let expr = Expression::Integer(42);
        let expr_clone = expr.clone();

        match (expr, expr_clone) {
            (Expression::Integer(a), Expression::Integer(b)) => assert_eq!(a, b),
            _ => panic!("Expected Integer variants"),
        }
    }

    // Tests for Expression - Float
    #[test]
    fn test_expression_float() {
        let expr = Expression::Float(MathFloat::from(42.5));
        match expr {
            Expression::Float(f) => {
                let value: f64 = f.into();
                assert!((value - 42.5).abs() < 1e-10);
            }
            _ => panic!("Expected Float variant"),
        }
    }

    #[test]
    fn test_expression_float_negative() {
        let expr = Expression::Float(MathFloat::from(-2.5));
        match expr {
            Expression::Float(f) => {
                let value: f64 = f.into();
                assert!((value + 2.5).abs() < 1e-10);
            }
            _ => panic!("Expected Float variant"),
        }
    }

    // Tests for Expression - Rational
    #[test]
    fn test_expression_rational() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(1)),
            denominator: Box::new(Expression::Integer(2)),
        };

        match expr {
            Expression::Rational {
                numerator,
                denominator,
            } => {
                assert!(matches!(*numerator, Expression::Integer(1)));
                assert!(matches!(*denominator, Expression::Integer(2)));
            }
            _ => panic!("Expected Rational variant"),
        }
    }

    #[test]
    fn test_expression_rational_clone() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(3)),
            denominator: Box::new(Expression::Integer(4)),
        };

        let expr_clone = expr.clone();

        match expr_clone {
            Expression::Rational {
                numerator,
                denominator,
            } => {
                assert!(matches!(*numerator, Expression::Integer(3)));
                assert!(matches!(*denominator, Expression::Integer(4)));
            }
            _ => panic!("Expected Rational variant"),
        }
    }

    // Tests for Expression - Complex
    #[test]
    fn test_expression_complex() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(3)),
            imaginary: Box::new(Expression::Integer(4)),
        };

        match expr {
            Expression::Complex { real, imaginary } => {
                assert!(matches!(*real, Expression::Integer(3)));
                assert!(matches!(*imaginary, Expression::Integer(4)));
            }
            _ => panic!("Expected Complex variant"),
        }
    }

    #[test]
    fn test_expression_complex_pure_imaginary() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(0)),
            imaginary: Box::new(Expression::Integer(1)),
        };

        match expr {
            Expression::Complex { real, imaginary } => {
                assert!(matches!(*real, Expression::Integer(0)));
                assert!(matches!(*imaginary, Expression::Integer(1)));
            }
            _ => panic!("Expected Complex variant"),
        }
    }

    // Tests for Expression - Variable
    #[test]
    fn test_expression_variable() {
        let expr = Expression::Variable("x".to_string());
        match expr {
            Expression::Variable(name) => assert_eq!(name, "x"),
            _ => panic!("Expected Variable variant"),
        }
    }

    #[test]
    fn test_expression_variable_greek() {
        let expr = Expression::Variable("theta".to_string());
        match expr {
            Expression::Variable(name) => assert_eq!(name, "theta"),
            _ => panic!("Expected Variable variant"),
        }
    }

    #[test]
    fn test_expression_variable_subscript() {
        let expr = Expression::Variable("x_1".to_string());
        match expr {
            Expression::Variable(name) => assert_eq!(name, "x_1"),
            _ => panic!("Expected Variable variant"),
        }
    }

    // Tests for Expression - Constant
    #[test]
    fn test_expression_constant_pi() {
        let expr = Expression::Constant(MathConstant::Pi);
        match expr {
            Expression::Constant(c) => assert_eq!(c, MathConstant::Pi),
            _ => panic!("Expected Constant variant"),
        }
    }

    #[test]
    fn test_expression_constant_e() {
        let expr = Expression::Constant(MathConstant::E);
        match expr {
            Expression::Constant(c) => assert_eq!(c, MathConstant::E),
            _ => panic!("Expected Constant variant"),
        }
    }

    // Tests for Expression - Binary
    #[test]
    fn test_expression_binary_add() {
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(3)),
        };

        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(*right, Expression::Integer(3)));
            }
            _ => panic!("Expected Binary variant"),
        }
    }

    #[test]
    fn test_expression_binary_nested() {
        // (2 + 3) * 4
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Integer(3)),
            }),
            right: Box::new(Expression::Integer(4)),
        };

        match expr {
            Expression::Binary { op, left, .. } => {
                assert_eq!(op, BinaryOp::Mul);
                match *left {
                    Expression::Binary { op, .. } => assert_eq!(op, BinaryOp::Add),
                    _ => panic!("Expected nested Binary"),
                }
            }
            _ => panic!("Expected Binary variant"),
        }
    }

    // Tests for Expression - Unary
    #[test]
    fn test_expression_unary_neg() {
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Integer(5)),
        };

        match expr {
            Expression::Unary { op, operand } => {
                assert_eq!(op, UnaryOp::Neg);
                assert!(matches!(*operand, Expression::Integer(5)));
            }
            _ => panic!("Expected Unary variant"),
        }
    }

    #[test]
    fn test_expression_unary_factorial() {
        let expr = Expression::Unary {
            op: UnaryOp::Factorial,
            operand: Box::new(Expression::Variable("n".to_string())),
        };

        match expr {
            Expression::Unary { op, operand } => {
                assert_eq!(op, UnaryOp::Factorial);
                match *operand {
                    Expression::Variable(ref name) => assert_eq!(name, "n"),
                    _ => panic!("Expected Variable operand"),
                }
            }
            _ => panic!("Expected Unary variant"),
        }
    }

    // Tests for Expression - Function
    #[test]
    fn test_expression_function_no_args() {
        let expr = Expression::Function {
            name: "f".to_string(),
            args: vec![],
        };

        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 0);
            }
            _ => panic!("Expected Function variant"),
        }
    }

    #[test]
    fn test_expression_function_one_arg() {
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };

        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    Expression::Variable(v) => assert_eq!(v, "x"),
                    _ => panic!("Expected Variable argument"),
                }
            }
            _ => panic!("Expected Function variant"),
        }
    }

    #[test]
    fn test_expression_function_multiple_args() {
        let expr = Expression::Function {
            name: "max".to_string(),
            args: vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
            ],
        };

        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected Function variant"),
        }
    }

    // Tests for Expression - Derivative
    #[test]
    fn test_expression_derivative_first_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };

        match expr {
            Expression::Derivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Variable(_)));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            _ => panic!("Expected Derivative variant"),
        }
    }

    #[test]
    fn test_expression_derivative_second_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        };

        match expr {
            Expression::Derivative { order, .. } => assert_eq!(order, 2),
            _ => panic!("Expected Derivative variant"),
        }
    }

    // Tests for Expression - PartialDerivative
    #[test]
    fn test_expression_partial_derivative() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };

        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Variable(_)));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            _ => panic!("Expected PartialDerivative variant"),
        }
    }

    #[test]
    fn test_expression_partial_derivative_higher_order() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "y".to_string(),
            order: 3,
        };

        match expr {
            Expression::PartialDerivative { var, order, .. } => {
                assert_eq!(var, "y");
                assert_eq!(order, 3);
            }
            _ => panic!("Expected PartialDerivative variant"),
        }
    }

    // Tests for Expression - Integral
    #[test]
    fn test_expression_integral_indefinite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };

        match expr {
            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                assert!(matches!(*integrand, Expression::Variable(_)));
                assert_eq!(var, "x");
                assert!(bounds.is_none());
            }
            _ => panic!("Expected Integral variant"),
        }
    }

    #[test]
    fn test_expression_integral_definite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };

        match expr {
            Expression::Integral { bounds, .. } => {
                assert!(bounds.is_some());
                let bounds = bounds.unwrap();
                assert!(matches!(*bounds.lower, Expression::Integer(0)));
                assert!(matches!(*bounds.upper, Expression::Integer(1)));
            }
            _ => panic!("Expected Integral variant"),
        }
    }

    // Tests for Expression - Limit
    #[test]
    fn test_expression_limit_both_sides() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };

        match expr {
            Expression::Limit {
                expr,
                var,
                to,
                direction,
            } => {
                assert!(matches!(*expr, Expression::Variable(_)));
                assert_eq!(var, "x");
                assert!(matches!(*to, Expression::Integer(0)));
                assert_eq!(direction, Direction::Both);
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    #[test]
    fn test_expression_limit_from_left() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Left,
        };

        match expr {
            Expression::Limit { direction, .. } => assert_eq!(direction, Direction::Left),
            _ => panic!("Expected Limit variant"),
        }
    }

    #[test]
    fn test_expression_limit_to_infinity() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: Direction::Both,
        };

        match expr {
            Expression::Limit { to, .. } => {
                assert!(matches!(*to, Expression::Constant(MathConstant::Infinity)));
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    // Tests for Expression - Sum
    #[test]
    fn test_expression_sum() {
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };

        match expr {
            Expression::Sum {
                index,
                lower,
                upper,
                body,
            } => {
                assert_eq!(index, "i");
                assert!(matches!(*lower, Expression::Integer(1)));
                assert!(matches!(*upper, Expression::Variable(_)));
                assert!(matches!(*body, Expression::Variable(_)));
            }
            _ => panic!("Expected Sum variant"),
        }
    }

    #[test]
    fn test_expression_sum_complex_body() {
        let expr = Expression::Sum {
            index: "k".to_string(),
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(10)),
            body: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Variable("k".to_string())),
                right: Box::new(Expression::Integer(2)),
            }),
        };

        match expr {
            Expression::Sum { body, .. } => {
                assert!(matches!(*body, Expression::Binary { .. }));
            }
            _ => panic!("Expected Sum variant"),
        }
    }

    // Tests for Expression - Product
    #[test]
    fn test_expression_product() {
        let expr = Expression::Product {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };

        match expr {
            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                assert_eq!(index, "i");
                assert!(matches!(*lower, Expression::Integer(1)));
                assert!(matches!(*upper, Expression::Variable(_)));
                assert!(matches!(*body, Expression::Variable(_)));
            }
            _ => panic!("Expected Product variant"),
        }
    }

    // Tests for Expression - Vector
    #[test]
    fn test_expression_vector_empty() {
        let expr = Expression::Vector(vec![]);
        match expr {
            Expression::Vector(elements) => assert_eq!(elements.len(), 0),
            _ => panic!("Expected Vector variant"),
        }
    }

    #[test]
    fn test_expression_vector_single() {
        let expr = Expression::Vector(vec![Expression::Integer(1)]);
        match expr {
            Expression::Vector(elements) => {
                assert_eq!(elements.len(), 1);
                assert!(matches!(elements[0], Expression::Integer(1)));
            }
            _ => panic!("Expected Vector variant"),
        }
    }

    #[test]
    fn test_expression_vector_multiple() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ]);

        match expr {
            Expression::Vector(elements) => {
                assert_eq!(elements.len(), 3);
                assert!(matches!(elements[0], Expression::Integer(1)));
                assert!(matches!(elements[1], Expression::Integer(2)));
                assert!(matches!(elements[2], Expression::Integer(3)));
            }
            _ => panic!("Expected Vector variant"),
        }
    }

    #[test]
    fn test_expression_vector_mixed_types() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Variable("x".to_string()),
            Expression::Float(MathFloat::from(2.5)),
        ]);

        match expr {
            Expression::Vector(elements) => assert_eq!(elements.len(), 3),
            _ => panic!("Expected Vector variant"),
        }
    }

    // Tests for Expression - Matrix
    #[test]
    fn test_expression_matrix_empty() {
        let expr = Expression::Matrix(vec![]);
        match expr {
            Expression::Matrix(rows) => assert_eq!(rows.len(), 0),
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_expression_matrix_single_element() {
        let expr = Expression::Matrix(vec![vec![Expression::Integer(1)]]);

        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 1);
                assert!(matches!(rows[0][0], Expression::Integer(1)));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_expression_matrix_2x2() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);

        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
                assert_eq!(rows[1].len(), 2);
                assert!(matches!(rows[0][0], Expression::Integer(1)));
                assert!(matches!(rows[1][1], Expression::Integer(4)));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_expression_matrix_rectangular() {
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
            ],
            vec![
                Expression::Integer(4),
                Expression::Integer(5),
                Expression::Integer(6),
            ],
        ]);

        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 3);
                assert_eq!(rows[1].len(), 3);
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    // Tests for Expression - Equation
    #[test]
    fn test_expression_equation_simple() {
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };

        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(*left, Expression::Variable(_)));
                assert!(matches!(*right, Expression::Integer(5)));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_expression_equation_complex() {
        // y = 2x + 1
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expression::Integer(2)),
                    right: Box::new(Expression::Variable("x".to_string())),
                }),
                right: Box::new(Expression::Integer(1)),
            }),
        };

        match expr {
            Expression::Equation { right, .. } => {
                assert!(matches!(*right, Expression::Binary { .. }));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    // Tests for Expression - Inequality
    #[test]
    fn test_expression_inequality_less_than() {
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };

        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Lt);
                assert!(matches!(*left, Expression::Variable(_)));
                assert!(matches!(*right, Expression::Integer(5)));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_expression_inequality_greater_equal() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ge,
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Integer(0)),
        };

        match expr {
            Expression::Inequality { op, .. } => assert_eq!(op, InequalityOp::Ge),
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_expression_inequality_not_equal() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ne,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        };

        match expr {
            Expression::Inequality { op, .. } => assert_eq!(op, InequalityOp::Ne),
            _ => panic!("Expected Inequality variant"),
        }
    }

    // Test Expression::Clone
    #[test]
    fn test_expression_clone_deep() {
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Variable("x".to_string())),
        };

        let expr_clone = expr.clone();

        match (expr, expr_clone) {
            (Expression::Binary { op: op1, .. }, Expression::Binary { op: op2, .. }) => {
                assert_eq!(op1, op2);
            }
            _ => panic!("Clone failed"),
        }
    }

    // Test Debug trait
    #[test]
    fn test_expression_debug() {
        let expr = Expression::Integer(42);
        let debug_str = format!("{:?}", expr);
        assert!(debug_str.contains("Integer"));
        assert!(debug_str.contains("42"));
    }

    // Tests for MathFloat
    #[test]
    fn test_math_float_creation() {
        let f1 = MathFloat::new(3.14);
        let f2 = MathFloat::from(3.14);
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_math_float_value_extraction() {
        let f = MathFloat::from(2.718);
        assert_eq!(f.value(), 2.718);

        let val: f64 = f.into();
        assert_eq!(val, 2.718);
    }

    #[test]
    fn test_math_float_equality() {
        let f1 = MathFloat::from(1.5);
        let f2 = MathFloat::from(1.5);
        let f3 = MathFloat::from(2.5);

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
    }

    #[test]
    fn test_math_float_copy() {
        let f1 = MathFloat::from(3.14);
        let f2 = f1;
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_math_float_display() {
        let f = MathFloat::from(3.14159);
        let display_str = format!("{}", f);
        assert_eq!(display_str, "3.14159");
    }

    #[test]
    fn test_math_float_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MathFloat::from(1.0));
        set.insert(MathFloat::from(2.0));
        set.insert(MathFloat::from(1.0)); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&MathFloat::from(1.0)));
        assert!(set.contains(&MathFloat::from(2.0)));
    }

    #[test]
    fn test_math_float_nan_equality() {
        // NaN values should compare equal to themselves in MathFloat
        let nan1 = MathFloat::from(f64::NAN);
        let nan2 = MathFloat::from(f64::NAN);
        assert_eq!(nan1, nan2);
    }

    #[test]
    fn test_math_float_infinity() {
        let inf = MathFloat::from(f64::INFINITY);
        let neg_inf = MathFloat::from(f64::NEG_INFINITY);

        assert_ne!(inf, neg_inf);
        assert_eq!(inf, MathFloat::from(f64::INFINITY));
    }

    // Tests for IntegralBounds equality and hashing
    #[test]
    fn test_integral_bounds_equality() {
        let bounds1 = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        let bounds2 = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        let bounds3 = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(2)),
        };

        assert_eq!(bounds1, bounds2);
        assert_ne!(bounds1, bounds3);
    }

    #[test]
    fn test_integral_bounds_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        let bounds1 = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        let bounds2 = IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        };

        set.insert(bounds1);
        set.insert(bounds2); // Should be considered duplicate

        assert_eq!(set.len(), 1);
    }

    // Tests for Expression equality
    #[test]
    fn test_expression_integer_equality() {
        let e1 = Expression::Integer(42);
        let e2 = Expression::Integer(42);
        let e3 = Expression::Integer(43);

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_float_equality() {
        let e1 = Expression::Float(MathFloat::from(3.14));
        let e2 = Expression::Float(MathFloat::from(3.14));
        let e3 = Expression::Float(MathFloat::from(2.71));

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_variable_equality() {
        let e1 = Expression::Variable("x".to_string());
        let e2 = Expression::Variable("x".to_string());
        let e3 = Expression::Variable("y".to_string());

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_binary_equality() {
        let e1 = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };

        let e2 = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };

        let e3 = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_nested_equality() {
        // (1 + 2) * 3
        let e1 = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(1)),
                right: Box::new(Expression::Integer(2)),
            }),
            right: Box::new(Expression::Integer(3)),
        };

        let e2 = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(1)),
                right: Box::new(Expression::Integer(2)),
            }),
            right: Box::new(Expression::Integer(3)),
        };

        assert_eq!(e1, e2);
    }

    #[test]
    fn test_expression_vector_equality() {
        let e1 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(2)]);

        let e2 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(2)]);

        let e3 = Expression::Vector(vec![Expression::Integer(1), Expression::Integer(3)]);

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_matrix_equality() {
        let e1 = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);

        let e2 = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);

        let e3 = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(5)],
        ]);

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    // Tests for Expression hashing and use in collections
    #[test]
    fn test_expression_hash_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        set.insert(Expression::Integer(1));
        set.insert(Expression::Integer(2));
        set.insert(Expression::Integer(1)); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&Expression::Integer(1)));
        assert!(set.contains(&Expression::Integer(2)));
    }

    #[test]
    fn test_expression_hash_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        map.insert(Expression::Variable("x".to_string()), 42);
        map.insert(Expression::Variable("y".to_string()), 17);
        map.insert(Expression::Variable("x".to_string()), 99); // Update

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&Expression::Variable("x".to_string())), Some(&99));
        assert_eq!(map.get(&Expression::Variable("y".to_string())), Some(&17));
    }

    #[test]
    fn test_expression_complex_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        let expr1 = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };

        let expr2 = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };

        set.insert(expr1);
        set.insert(expr2); // Should be considered duplicate

        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_expression_float_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();

        set.insert(Expression::Float(MathFloat::from(3.14)));
        set.insert(Expression::Float(MathFloat::from(2.71)));
        set.insert(Expression::Float(MathFloat::from(3.14))); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_expression_function_equality() {
        let e1 = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };

        let e2 = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };

        let e3 = Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_derivative_equality() {
        let e1 = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };

        let e2 = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };

        let e3 = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_expression_integral_equality() {
        let e1 = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };

        let e2 = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };

        let e3 = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };

        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    // Tests for serde serialization/deserialization
    #[cfg(feature = "serde")]
    mod serde_tests {
        use super::*;

        #[test]
        fn test_serialize_deserialize_integer() {
            let expr = Expression::Integer(42);
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_float() {
            let expr = Expression::Float(MathFloat::from(3.14159));
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_variable() {
            let expr = Expression::Variable("x".to_string());
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_constant() {
            let expr = Expression::Constant(MathConstant::Pi);
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_binary() {
            let expr = Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Variable("x".to_string())),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_unary() {
            let expr = Expression::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Expression::Integer(5)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_function() {
            let expr = Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_rational() {
            let expr = Expression::Rational {
                numerator: Box::new(Expression::Integer(1)),
                denominator: Box::new(Expression::Integer(2)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_complex() {
            let expr = Expression::Complex {
                real: Box::new(Expression::Integer(3)),
                imaginary: Box::new(Expression::Integer(4)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_derivative() {
            let expr = Expression::Derivative {
                expr: Box::new(Expression::Variable("f".to_string())),
                var: "x".to_string(),
                order: 2,
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_partial_derivative() {
            let expr = Expression::PartialDerivative {
                expr: Box::new(Expression::Variable("f".to_string())),
                var: "x".to_string(),
                order: 1,
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_integral_indefinite() {
            let expr = Expression::Integral {
                integrand: Box::new(Expression::Variable("x".to_string())),
                var: "x".to_string(),
                bounds: None,
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_integral_definite() {
            let expr = Expression::Integral {
                integrand: Box::new(Expression::Variable("x".to_string())),
                var: "x".to_string(),
                bounds: Some(IntegralBounds {
                    lower: Box::new(Expression::Integer(0)),
                    upper: Box::new(Expression::Integer(1)),
                }),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_limit() {
            let expr = Expression::Limit {
                expr: Box::new(Expression::Variable("f".to_string())),
                var: "x".to_string(),
                to: Box::new(Expression::Integer(0)),
                direction: Direction::Both,
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_sum() {
            let expr = Expression::Sum {
                index: "i".to_string(),
                lower: Box::new(Expression::Integer(1)),
                upper: Box::new(Expression::Variable("n".to_string())),
                body: Box::new(Expression::Variable("i".to_string())),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_product() {
            let expr = Expression::Product {
                index: "i".to_string(),
                lower: Box::new(Expression::Integer(1)),
                upper: Box::new(Expression::Variable("n".to_string())),
                body: Box::new(Expression::Variable("i".to_string())),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_vector() {
            let expr = Expression::Vector(vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
            ]);
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_matrix() {
            let expr = Expression::Matrix(vec![
                vec![Expression::Integer(1), Expression::Integer(2)],
                vec![Expression::Integer(3), Expression::Integer(4)],
            ]);
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_equation() {
            let expr = Expression::Equation {
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Integer(5)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_inequality() {
            let expr = Expression::Inequality {
                op: InequalityOp::Lt,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Integer(5)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_nested_expression() {
            // (2 + x) * 3
            let expr = Expression::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::Integer(2)),
                    right: Box::new(Expression::Variable("x".to_string())),
                }),
                right: Box::new(Expression::Integer(3)),
            };
            let json = serde_json::to_string(&expr).unwrap();
            let parsed: Expression = serde_json::from_str(&json).unwrap();
            assert_eq!(expr, parsed);
        }

        #[test]
        fn test_serialize_deserialize_math_float() {
            let float = MathFloat::from(3.14159);
            let json = serde_json::to_string(&float).unwrap();
            let parsed: MathFloat = serde_json::from_str(&json).unwrap();
            assert_eq!(float, parsed);
        }

        #[test]
        fn test_serialize_deserialize_all_constants() {
            let constants = vec![
                MathConstant::Pi,
                MathConstant::E,
                MathConstant::I,
                MathConstant::Infinity,
                MathConstant::NegInfinity,
            ];
            for constant in constants {
                let json = serde_json::to_string(&constant).unwrap();
                let parsed: MathConstant = serde_json::from_str(&json).unwrap();
                assert_eq!(constant, parsed);
            }
        }

        #[test]
        fn test_serialize_deserialize_all_binary_ops() {
            let ops = vec![
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Pow,
                BinaryOp::Mod,
            ];
            for op in ops {
                let json = serde_json::to_string(&op).unwrap();
                let parsed: BinaryOp = serde_json::from_str(&json).unwrap();
                assert_eq!(op, parsed);
            }
        }

        #[test]
        fn test_serialize_deserialize_all_unary_ops() {
            let ops = vec![
                UnaryOp::Neg,
                UnaryOp::Pos,
                UnaryOp::Factorial,
                UnaryOp::Transpose,
            ];
            for op in ops {
                let json = serde_json::to_string(&op).unwrap();
                let parsed: UnaryOp = serde_json::from_str(&json).unwrap();
                assert_eq!(op, parsed);
            }
        }

        #[test]
        fn test_serialize_deserialize_all_directions() {
            let directions = vec![Direction::Left, Direction::Right, Direction::Both];
            for direction in directions {
                let json = serde_json::to_string(&direction).unwrap();
                let parsed: Direction = serde_json::from_str(&json).unwrap();
                assert_eq!(direction, parsed);
            }
        }

        #[test]
        fn test_serialize_deserialize_all_inequality_ops() {
            let ops = vec![
                InequalityOp::Lt,
                InequalityOp::Le,
                InequalityOp::Gt,
                InequalityOp::Ge,
                InequalityOp::Ne,
            ];
            for op in ops {
                let json = serde_json::to_string(&op).unwrap();
                let parsed: InequalityOp = serde_json::from_str(&json).unwrap();
                assert_eq!(op, parsed);
            }
        }

        #[test]
        fn test_serialize_deserialize_integral_bounds() {
            let bounds = IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(10)),
            };
            let json = serde_json::to_string(&bounds).unwrap();
            let parsed: IntegralBounds = serde_json::from_str(&json).unwrap();
            assert_eq!(bounds, parsed);
        }

        #[test]
        fn test_math_float_nan_serialization() {
            // Note: JSON doesn't natively support NaN, it serializes to null
            // This is expected behavior from ordered-float's serde implementation
            let nan = MathFloat::from(f64::NAN);
            let json = serde_json::to_string(&nan).unwrap();
            assert_eq!(json, "null");

            // For actual round-trip preservation of NaN, use binary formats like bincode
            // JSON explicitly doesn't support NaN per spec
        }

        #[test]
        fn test_math_float_infinity_serialization() {
            // Note: JSON doesn't natively support Infinity, it serializes to null
            // This is expected behavior from ordered-float's serde implementation
            let inf = MathFloat::from(f64::INFINITY);
            let json = serde_json::to_string(&inf).unwrap();
            assert_eq!(json, "null");

            let neg_inf = MathFloat::from(f64::NEG_INFINITY);
            let json = serde_json::to_string(&neg_inf).unwrap();
            assert_eq!(json, "null");

            // For actual round-trip preservation of special floats, use binary formats
        }
    }
}
