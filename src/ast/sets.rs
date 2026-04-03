//! Set theory types for mathematical expressions.

/// Set operation type.
///
/// Represents binary operations on sets in set theory notation.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::SetOp;
///
/// let union = SetOp::Union;          // A ∪ B
/// let intersection = SetOp::Intersection;  // A ∩ B
/// let difference = SetOp::Difference;      // A ∖ B
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SetOp {
    /// Union of two sets: A ∪ B
    Union,

    /// Intersection of two sets: A ∩ B
    Intersection,

    /// Set difference: A ∖ B (elements in A but not in B)
    Difference,

    /// Symmetric difference: A △ B (elements in exactly one set)
    SymmetricDiff,

    /// Cartesian product: A × B
    CartesianProd,
}

/// Set membership and subset relations.
///
/// Represents relations between elements and sets, or between sets.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::SetRelation;
///
/// let member = SetRelation::In;       // x ∈ S
/// let subset = SetRelation::SubsetEq; // A ⊆ B
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SetRelation {
    /// Element membership: x ∈ S
    In,

    /// Non-membership: x ∉ S
    NotIn,

    /// Proper subset: A ⊂ B
    Subset,

    /// Subset or equal: A ⊆ B
    SubsetEq,

    /// Proper superset: A ⊃ B
    Superset,

    /// Superset or equal: A ⊇ B
    SupersetEq,
}

/// Standard number sets in mathematics.
///
/// These are the commonly used number sets denoted with blackboard bold letters.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::NumberSet;
///
/// let naturals = NumberSet::Natural;   // ℕ
/// let reals = NumberSet::Real;         // ℝ
/// let complex = NumberSet::Complex;    // ℂ
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumberSet {
    /// Natural numbers: ℕ = {0, 1, 2, 3, ...} or {1, 2, 3, ...}
    Natural,

    /// Integers: ℤ = {..., -2, -1, 0, 1, 2, ...}
    Integer,

    /// Rational numbers: ℚ (fractions p/q where p,q ∈ ℤ, q ≠ 0)
    Rational,

    /// Real numbers: ℝ
    Real,

    /// Complex numbers: ℂ
    Complex,

    /// Quaternions: ℍ
    Quaternion,
}
