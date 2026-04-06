//! Mathematical constants and float wrapper types.

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
/// ## Parser Behavior
///
/// Parsers (`parse` and `parse_latex`) never produce NaN values. Only finite numbers and
/// infinities appear in parsed ASTs. NaN values can only arise through manual AST
/// construction via `MathFloat::new(f64::NAN)` or `MathFloat::from(f64::NAN)`.
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
///   (also serves as quaternion basis vector i)
/// - **`J`**: Quaternion basis vector j, parsed from `\mathbf{j}` or in quaternion context
/// - **`K`**: Quaternion basis vector k, parsed from `\mathbf{k}` or in quaternion context
/// - **`Infinity`**: Parsed from `∞` (Unicode) or `\infty` (LaTeX)
/// - **`NegInfinity`**: Produced by parsers when unary minus is applied to infinity.
///   Both `-∞` / `-inf` (plain text) and `-\infty` (LaTeX) parse directly as
///   `Constant(NegInfinity)`.
/// - **`NaN`**: Not-a-Number. Parsed from `nan` or `NaN` (plain text) or
///   `\text{NaN}`, `\text{nan}`, `\mathrm{NaN}` (LaTeX). Represents an
///   indeterminate or undefined numeric result.
///
/// ## Quaternion Context
///
/// The presence of `J` or `K` constants in an expression implies quaternion context.
/// The existing `I` constant serves dual purpose: complex imaginary unit and quaternion
/// basis vector i. In quaternion expressions (a + bi + cj + dk), the multiplication
/// rules are: i² = j² = k² = ijk = -1.
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
/// // Quaternion basis vectors
/// let i = MathConstant::I;
/// let j = MathConstant::J;
/// let k = MathConstant::K;
/// assert_ne!(i, j);
/// assert_ne!(j, k);
///
/// // Note: NegInfinity is for programmatic use
/// let neg_inf = MathConstant::NegInfinity;
/// assert_ne!(neg_inf, MathConstant::Infinity);
///
/// // NaN is distinct from all other constants
/// let nan = MathConstant::NaN;
/// assert_ne!(nan, MathConstant::Infinity);
/// assert_ne!(nan, MathConstant::NegInfinity);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathConstant {
    /// The mathematical constant π (pi), approximately 3.14159...
    Pi,

    /// Euler's number e, approximately 2.71828...
    E,

    /// The imaginary unit i, where i² = -1
    /// Also serves as quaternion basis vector i.
    I,

    /// Quaternion basis vector j.
    /// Satisfies j² = -1 and ij = k, ji = -k.
    J,

    /// Quaternion basis vector k.
    /// Satisfies k² = -1 and jk = i, kj = -i.
    K,

    /// Positive infinity (∞)
    Infinity,

    /// Negative infinity (-∞)
    NegInfinity,

    /// Not-a-Number (NaN): an indeterminate or undefined numeric value.
    ///
    /// Parsed from `nan` or `NaN` (plain text), and from `\text{NaN}`,
    /// `\text{nan}`, or `\mathrm{NaN}` (LaTeX).
    NaN,
}
