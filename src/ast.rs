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
//!
//! ## Examples
//!
//! ```ignore
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Representing: 2 * π
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Integer(2)),
//!     right: Box::new(Expression::Constant(MathConstant::Pi)),
//! };
//! ```

/// Mathematical constants used in expressions.
///
/// These represent well-known mathematical constants with precise mathematical meaning.
///
/// # Examples
///
/// ```ignore
/// use mathlex::ast::MathConstant;
///
/// let pi = MathConstant::Pi;
/// let euler = MathConstant::E;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// # Examples
///
/// ```ignore
/// use mathlex::ast::BinaryOp;
///
/// let add = BinaryOp::Add;  // +
/// let pow = BinaryOp::Pow;  // ^
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
}

/// Unary operators for mathematical expressions.
///
/// Represents operators that take a single operand.
///
/// # Examples
///
/// ```ignore
/// use mathlex::ast::UnaryOp;
///
/// let neg = UnaryOp::Neg;        // Negation
/// let fact = UnaryOp::Factorial; // Factorial (!)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// ```ignore
/// use mathlex::ast::Direction;
///
/// let from_left = Direction::Left;   // lim x→a⁻
/// let from_right = Direction::Right; // lim x→a⁺
/// let both = Direction::Both;        // lim x→a
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
/// ```ignore
/// use mathlex::ast::InequalityOp;
///
/// let less_than = InequalityOp::Lt;     // <
/// let less_equal = InequalityOp::Le;    // ≤
/// let not_equal = InequalityOp::Ne;     // ≠
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Bounds for definite integrals.
///
/// Represents the lower and upper bounds of integration.
///
/// # Examples
///
/// ```ignore
/// use mathlex::ast::{IntegralBounds, Expression};
///
/// // Integral from 0 to 1
/// let bounds = IntegralBounds {
///     lower: Box::new(Expression::Integer(0)),
///     upper: Box::new(Expression::Integer(1)),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct IntegralBounds {
    /// Lower bound of integration
    pub lower: Box<Expression>,

    /// Upper bound of integration
    pub upper: Box<Expression>,
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
/// ```ignore
/// use mathlex::ast::{Expression, BinaryOp, MathConstant};
///
/// // 2 * π
/// let expr = Expression::Binary {
///     op: BinaryOp::Mul,
///     left: Box::new(Expression::Integer(2)),
///     right: Box::new(Expression::Constant(MathConstant::Pi)),
/// };
/// ```
#[derive(Debug, Clone)]
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
    Float(f64),

    /// Rational number (fraction).
    ///
    /// Represents a ratio of two integers (numerator/denominator).
    ///
    /// # Examples
    /// - `1/2`
    /// - `-3/4`
    /// - `22/7` (approximation of π)
    Rational {
        /// Numerator of the fraction
        numerator: Box<Expression>,

        /// Denominator of the fraction
        denominator: Box<Expression>,
    },

    /// Complex number.
    ///
    /// Represents a number with real and imaginary components (a + bi).
    ///
    /// # Examples
    /// - `3 + 4i`
    /// - `-2 - 5i`
    /// - `0 + i` (pure imaginary)
    Complex {
        /// Real component
        real: Box<Expression>,

        /// Imaginary component (coefficient of i)
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
    /// Represents the nth derivative of an expression with respect to a variable.
    ///
    /// # Examples
    /// - `d/dx(f)` (first derivative, order=1)
    /// - `d²/dx²(f)` (second derivative, order=2)
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
    /// Represents the nth partial derivative of an expression with respect to a variable.
    ///
    /// # Examples
    /// - `∂f/∂x`
    /// - `∂²f/∂x²`
    PartialDerivative {
        /// The expression being partially differentiated
        expr: Box<Expression>,

        /// The variable to partially differentiate with respect to
        var: String,

        /// Order of partial differentiation
        order: u32,
    },

    /// Integral.
    ///
    /// Represents both definite and indefinite integrals.
    ///
    /// # Examples
    /// - `∫ f(x) dx` (indefinite, bounds=None)
    /// - `∫₀¹ f(x) dx` (definite, bounds=Some(...))
    Integral {
        /// The integrand (expression being integrated)
        integrand: Box<Expression>,

        /// The variable of integration
        var: String,

        /// Integration bounds (None for indefinite integral)
        bounds: Option<IntegralBounds>,
    },

    /// Limit.
    ///
    /// Represents the limit of an expression as a variable approaches a value.
    ///
    /// # Examples
    /// - `lim x→0 (sin(x)/x)`
    /// - `lim x→∞ (1/x)`
    /// - `lim x→a⁺ f(x)` (from right)
    Limit {
        /// The expression whose limit is being taken
        expr: Box<Expression>,

        /// The variable approaching the limit
        var: String,

        /// The value being approached
        to: Box<Expression>,

        /// Direction of approach (left, right, or both)
        direction: Direction,
    },

    /// Summation.
    ///
    /// Represents a sum over a range of values.
    ///
    /// # Examples
    /// - `Σᵢ₌₁ⁿ i²`
    /// - `Σₖ f(k)`
    Sum {
        /// The index variable
        index: String,

        /// Lower bound of summation
        lower: Box<Expression>,

        /// Upper bound of summation
        upper: Box<Expression>,

        /// The expression being summed
        body: Box<Expression>,
    },

    /// Product.
    ///
    /// Represents a product over a range of values.
    ///
    /// # Examples
    /// - `Πᵢ₌₁ⁿ i`
    /// - `Πₖ f(k)`
    Product {
        /// The index variable
        index: String,

        /// Lower bound of product
        lower: Box<Expression>,

        /// Upper bound of product
        upper: Box<Expression>,

        /// The expression being multiplied
        body: Box<Expression>,
    },

    /// Vector.
    ///
    /// Represents an ordered collection of expressions as a vector.
    ///
    /// # Examples
    /// - `[1, 2, 3]`
    /// - `[x, y, z]`
    /// - `[]` (empty vector)
    Vector(Vec<Expression>),

    /// Matrix.
    ///
    /// Represents a 2D array of expressions. All rows must have the same length.
    ///
    /// # Examples
    /// - `[[1, 2], [3, 4]]` (2×2 matrix)
    /// - `[[x]]` (1×1 matrix)
    /// - `[[]]` (0×0 matrix - edge case)
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
}

#[cfg(test)]
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
        let expr = Expression::Float(42.5);
        match expr {
            Expression::Float(f) => assert!((f - 42.5).abs() < 1e-10),
            _ => panic!("Expected Float variant"),
        }
    }

    #[test]
    fn test_expression_float_negative() {
        let expr = Expression::Float(-2.5);
        match expr {
            Expression::Float(f) => assert!((f + 2.5).abs() < 1e-10),
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
            Expression::Rational { numerator, denominator } => {
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
            Expression::Rational { numerator, denominator } => {
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
            Expression::Integral { integrand, var, bounds } => {
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
            Expression::Limit { expr, var, to, direction } => {
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
            Expression::Sum { index, lower, upper, body } => {
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
            Expression::Product { index, lower, upper, body } => {
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
            Expression::Float(2.5),
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
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1)],
        ]);

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
            vec![Expression::Integer(1), Expression::Integer(2), Expression::Integer(3)],
            vec![Expression::Integer(4), Expression::Integer(5), Expression::Integer(6)],
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
}
