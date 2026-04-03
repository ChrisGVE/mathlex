//! Linear algebra types for mathematical expressions.

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

/// Index position type for tensor notation.
///
/// In Einstein summation convention, indices can be either upper (contravariant)
/// or lower (covariant). The position determines how the index transforms under
/// coordinate changes.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::IndexType;
///
/// let upper = IndexType::Upper;  // T^i (superscript)
/// let lower = IndexType::Lower;  // T_j (subscript)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IndexType {
    /// Upper index (contravariant) - superscript position
    Upper,

    /// Lower index (covariant) - subscript position
    Lower,
}

/// A single tensor index with name and position.
///
/// Represents an index in tensor notation, specifying both the index name
/// (typically a single letter like i, j, k) and whether it appears as an
/// upper (contravariant) or lower (covariant) index.
///
/// ## Einstein Summation Convention
///
/// When the same index name appears once as upper and once as lower in a term,
/// summation over that index is implied. For example, `T^i_j v^j` implies
/// `Σ_j T^i_j v^j`.
///
/// ## Examples
///
/// ```
/// use mathlex::ast::{TensorIndex, IndexType};
///
/// // Upper index i (T^i)
/// let upper_i = TensorIndex {
///     name: "i".to_string(),
///     index_type: IndexType::Upper,
/// };
///
/// // Lower index j (T_j)
/// let lower_j = TensorIndex {
///     name: "j".to_string(),
///     index_type: IndexType::Lower,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorIndex {
    /// The index name (e.g., "i", "j", "k", "μ", "ν")
    pub name: String,

    /// Whether this is an upper or lower index
    pub index_type: IndexType,
}
