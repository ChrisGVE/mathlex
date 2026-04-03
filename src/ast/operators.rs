//! Operator types for mathematical expressions.

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

/// Relation operators for mathematical relations.
///
/// Represents relations between mathematical objects such as similarity,
/// equivalence, congruence, and approximation.
///
/// # Examples
///
/// ```
/// use mathlex::ast::RelationOp;
///
/// let similar = RelationOp::Similar;     // ~
/// let equiv = RelationOp::Equivalent;    // ≡
/// let approx = RelationOp::Approx;       // ≈
/// assert_ne!(similar, equiv);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RelationOp {
    /// Similarity relation (~)
    Similar,

    /// Equivalence relation (≡)
    Equivalent,

    /// Congruence relation (≅)
    Congruent,

    /// Approximation relation (≈)
    Approx,
}
