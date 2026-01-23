//! Expression metadata and type inference types.
//!
//! These types provide the foundation for tracking mathematical types
//! and context information associated with expressions.
//!
//! ## Overview
//!
//! The metadata system allows consumers of the mathlex AST to annotate expressions
//! with inferred or declared type information. This is particularly useful for:
//!
//! - Type checking in symbolic mathematics systems
//! - Disambiguation of operations (e.g., scalar vs. vector multiplication)
//! - Tracking the provenance of type information
//! - Building more sophisticated mathematical analysis tools
//!
//! ## Design Philosophy
//!
//! mathlex is a parsing library and does not perform type inference itself.
//! These types are provided as a foundation for consumers to build their own
//! type inference systems on top of the AST.
//!
//! ## Examples
//!
//! ```
//! use mathlex::metadata::{MathType, ExpressionMetadata, ContextSource};
//!
//! // Create metadata for an explicitly declared vector
//! let vector_meta = ExpressionMetadata::explicit(MathType::Vector(Some(3)));
//! assert_eq!(vector_meta.confidence, 1.0);
//! assert_eq!(vector_meta.context_source, ContextSource::Explicit);
//!
//! // Create metadata inferred from structure
//! let inferred_meta = ExpressionMetadata::structural(
//!     MathType::Scalar,
//!     0.9
//! );
//! assert_eq!(inferred_meta.confidence, 0.9);
//! assert_eq!(inferred_meta.context_source, ContextSource::Structural);
//! ```

use crate::ast::IndexType;

/// Mathematical type of an expression.
///
/// Used to track whether an expression represents a scalar, vector,
/// matrix, or other mathematical object.
///
/// ## Examples
///
/// ```
/// use mathlex::metadata::MathType;
///
/// // A scalar value
/// let scalar = MathType::Scalar;
///
/// // A 3-dimensional vector
/// let vec3 = MathType::Vector(Some(3));
///
/// // A vector of unknown dimension
/// let vec = MathType::Vector(None);
///
/// // A 3x3 matrix
/// let mat = MathType::Matrix(Some(3), Some(3));
///
/// // A function from scalars to scalars
/// let func = MathType::Function {
///     domain: Box::new(MathType::Scalar),
///     codomain: Box::new(MathType::Scalar),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MathType {
    /// A scalar value (real or complex number).
    ///
    /// Represents a single numerical value in any number system.
    Scalar,

    /// A vector with optional known dimension.
    ///
    /// - `Vector(None)` indicates a vector of unknown dimension
    /// - `Vector(Some(n))` indicates a vector in n-dimensional space
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::MathType;
    ///
    /// let vec3 = MathType::Vector(Some(3));  // 3D vector
    /// let vec = MathType::Vector(None);      // Unknown dimension
    /// ```
    Vector(Option<usize>),

    /// A matrix with optional known dimensions (rows, cols).
    ///
    /// - `Matrix(None, None)` indicates a matrix of unknown dimensions
    /// - `Matrix(Some(m), Some(n))` indicates an m×n matrix
    /// - Mixed `Some`/`None` indicates partially known dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::MathType;
    ///
    /// let mat = MathType::Matrix(Some(3), Some(3));  // 3×3 matrix
    /// let unknown = MathType::Matrix(None, None);    // Unknown dimensions
    /// ```
    Matrix(Option<usize>, Option<usize>),

    /// A tensor with known index structure.
    ///
    /// The vector contains the type of each index (upper or lower).
    /// The length of the vector indicates the tensor's rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::MathType;
    /// use mathlex::ast::IndexType;
    ///
    /// // Rank-2 tensor: T^i_j
    /// let tensor = MathType::Tensor(vec![IndexType::Upper, IndexType::Lower]);
    /// ```
    Tensor(Vec<IndexType>),

    /// A set.
    ///
    /// Represents a mathematical set of elements.
    Set,

    /// A function type with domain and codomain.
    ///
    /// Represents the type of a mathematical function mapping from
    /// the domain type to the codomain type.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::MathType;
    ///
    /// // f: ℝ → ℝ (scalar to scalar)
    /// let real_func = MathType::Function {
    ///     domain: Box::new(MathType::Scalar),
    ///     codomain: Box::new(MathType::Scalar),
    /// };
    ///
    /// // F: ℝ³ → ℝ³ (vector field)
    /// let vector_field = MathType::Function {
    ///     domain: Box::new(MathType::Vector(Some(3))),
    ///     codomain: Box::new(MathType::Vector(Some(3))),
    /// };
    /// ```
    Function {
        /// The domain type (input type)
        domain: Box<MathType>,
        /// The codomain type (output type)
        codomain: Box<MathType>,
    },

    /// Type is not yet determined.
    ///
    /// This is the default state before type inference has been performed.
    Unknown,
}

/// Source of type/context information.
///
/// Tracks how the type information was determined, which is useful for:
/// - Prioritizing conflicting type information
/// - Debugging type inference issues
/// - Reporting warnings about assumptions
///
/// # Priority Order
///
/// When multiple sources of type information conflict, the priority
/// order is typically:
/// 1. Explicit (highest)
/// 2. Declaration
/// 3. Structural
/// 4. Convention
/// 5. Default (lowest)
///
/// # Examples
///
/// ```
/// use mathlex::metadata::ContextSource;
///
/// let explicit = ContextSource::Explicit;
/// let inferred = ContextSource::Structural;
/// let convention = ContextSource::Convention;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContextSource {
    /// Explicitly annotated by user.
    ///
    /// The type was directly specified by the user, either through
    /// type annotations or explicit declarations.
    ///
    /// This is the highest priority source.
    Explicit,

    /// Inferred from a declaration (e.g., ∀x ∈ ℝ).
    ///
    /// The type was determined from a quantifier, set membership,
    /// or other explicit declaration in the expression.
    Declaration,

    /// Inferred from expression structure.
    ///
    /// The type was determined by analyzing the structure of the
    /// expression (e.g., a dot product implies vectors).
    Structural,

    /// Inferred from convention (e.g., Einstein summation).
    ///
    /// The type was determined based on notational conventions
    /// or common usage patterns.
    Convention,

    /// Default assumption.
    ///
    /// The type was assigned based on default assumptions when
    /// no other information was available.
    ///
    /// This is the lowest priority source.
    Default,
}

/// Metadata that can be attached to expressions.
///
/// This type combines a mathematical type with information about how
/// that type was determined and the confidence level.
///
/// ## Confidence Levels
///
/// The confidence field ranges from 0.0 to 1.0:
/// - `1.0`: Certain (e.g., explicit declaration)
/// - `0.9`: High confidence (e.g., structural inference with clear evidence)
/// - `0.7`: Moderate confidence (e.g., contextual inference)
/// - `0.5`: Low confidence (e.g., weak heuristic)
/// - `0.0`: Unknown (default)
///
/// ## Examples
///
/// ```
/// use mathlex::metadata::{ExpressionMetadata, MathType, ContextSource};
///
/// // Create metadata with default values
/// let default_meta = ExpressionMetadata::default();
/// assert_eq!(default_meta.inferred_type, None);
/// assert_eq!(default_meta.confidence, 0.0);
///
/// // Create explicit metadata
/// let explicit = ExpressionMetadata::explicit(MathType::Scalar);
/// assert_eq!(explicit.confidence, 1.0);
/// assert_eq!(explicit.context_source, ContextSource::Explicit);
///
/// // Create structural metadata with confidence
/// let inferred = ExpressionMetadata::structural(MathType::Vector(Some(3)), 0.95);
/// assert_eq!(inferred.confidence, 0.95);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionMetadata {
    /// The inferred mathematical type.
    ///
    /// `None` indicates that no type has been inferred yet.
    pub inferred_type: Option<MathType>,

    /// How the type was determined.
    pub context_source: ContextSource,

    /// Confidence level (0.0 to 1.0).
    ///
    /// Higher values indicate greater confidence in the type inference.
    /// A value of 1.0 indicates certainty (e.g., explicit declaration).
    pub confidence: f64,
}

impl Default for ExpressionMetadata {
    /// Creates metadata with no inferred type and zero confidence.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::{ExpressionMetadata, ContextSource};
    ///
    /// let meta = ExpressionMetadata::default();
    /// assert_eq!(meta.inferred_type, None);
    /// assert_eq!(meta.context_source, ContextSource::Default);
    /// assert_eq!(meta.confidence, 0.0);
    /// ```
    fn default() -> Self {
        Self {
            inferred_type: None,
            context_source: ContextSource::Default,
            confidence: 0.0,
        }
    }
}

impl ExpressionMetadata {
    /// Create new metadata with explicit type information.
    ///
    /// Sets confidence to 1.0 and source to `ContextSource::Explicit`.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::{ExpressionMetadata, MathType, ContextSource};
    ///
    /// let meta = ExpressionMetadata::explicit(MathType::Scalar);
    /// assert_eq!(meta.inferred_type, Some(MathType::Scalar));
    /// assert_eq!(meta.context_source, ContextSource::Explicit);
    /// assert_eq!(meta.confidence, 1.0);
    /// ```
    pub fn explicit(math_type: MathType) -> Self {
        Self {
            inferred_type: Some(math_type),
            context_source: ContextSource::Explicit,
            confidence: 1.0,
        }
    }

    /// Create metadata inferred from structure.
    ///
    /// Sets source to `ContextSource::Structural` with the given confidence level.
    ///
    /// # Arguments
    ///
    /// * `math_type` - The inferred mathematical type
    /// * `confidence` - Confidence level (should be between 0.0 and 1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::metadata::{ExpressionMetadata, MathType, ContextSource};
    ///
    /// let meta = ExpressionMetadata::structural(MathType::Vector(Some(3)), 0.9);
    /// assert_eq!(meta.inferred_type, Some(MathType::Vector(Some(3))));
    /// assert_eq!(meta.context_source, ContextSource::Structural);
    /// assert_eq!(meta.confidence, 0.9);
    /// ```
    pub fn structural(math_type: MathType, confidence: f64) -> Self {
        Self {
            inferred_type: Some(math_type),
            context_source: ContextSource::Structural,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_type_scalar() {
        let scalar = MathType::Scalar;
        assert_eq!(scalar, MathType::Scalar);
    }

    #[test]
    fn test_math_type_vector_with_dimension() {
        let vec3 = MathType::Vector(Some(3));
        assert_eq!(vec3, MathType::Vector(Some(3)));
    }

    #[test]
    fn test_math_type_vector_without_dimension() {
        let vec = MathType::Vector(None);
        assert_eq!(vec, MathType::Vector(None));
    }

    #[test]
    fn test_math_type_vector_dimensions_not_equal() {
        let vec3 = MathType::Vector(Some(3));
        let vec4 = MathType::Vector(Some(4));
        assert_ne!(vec3, vec4);
    }

    #[test]
    fn test_math_type_matrix_with_dimensions() {
        let mat = MathType::Matrix(Some(3), Some(3));
        assert_eq!(mat, MathType::Matrix(Some(3), Some(3)));
    }

    #[test]
    fn test_math_type_matrix_without_dimensions() {
        let mat = MathType::Matrix(None, None);
        assert_eq!(mat, MathType::Matrix(None, None));
    }

    #[test]
    fn test_math_type_matrix_partial_dimensions() {
        let mat = MathType::Matrix(Some(3), None);
        assert_eq!(mat, MathType::Matrix(Some(3), None));
    }

    #[test]
    fn test_math_type_tensor() {
        let tensor = MathType::Tensor(vec![IndexType::Upper, IndexType::Lower]);
        assert_eq!(
            tensor,
            MathType::Tensor(vec![IndexType::Upper, IndexType::Lower])
        );
    }

    #[test]
    fn test_math_type_tensor_empty() {
        let tensor = MathType::Tensor(vec![]);
        assert_eq!(tensor, MathType::Tensor(vec![]));
    }

    #[test]
    fn test_math_type_set() {
        let set = MathType::Set;
        assert_eq!(set, MathType::Set);
    }

    #[test]
    fn test_math_type_function_scalar_to_scalar() {
        let func = MathType::Function {
            domain: Box::new(MathType::Scalar),
            codomain: Box::new(MathType::Scalar),
        };
        assert_eq!(
            func,
            MathType::Function {
                domain: Box::new(MathType::Scalar),
                codomain: Box::new(MathType::Scalar),
            }
        );
    }

    #[test]
    fn test_math_type_function_vector_field() {
        let func = MathType::Function {
            domain: Box::new(MathType::Vector(Some(3))),
            codomain: Box::new(MathType::Vector(Some(3))),
        };
        assert_eq!(
            func,
            MathType::Function {
                domain: Box::new(MathType::Vector(Some(3))),
                codomain: Box::new(MathType::Vector(Some(3))),
            }
        );
    }

    #[test]
    fn test_math_type_unknown() {
        let unknown = MathType::Unknown;
        assert_eq!(unknown, MathType::Unknown);
    }

    #[test]
    fn test_math_type_clone() {
        let scalar = MathType::Scalar;
        let cloned = scalar.clone();
        assert_eq!(scalar, cloned);
    }

    #[test]
    fn test_math_type_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(MathType::Scalar, "scalar");
        map.insert(MathType::Vector(Some(3)), "vec3");
        assert_eq!(map.get(&MathType::Scalar), Some(&"scalar"));
        assert_eq!(map.get(&MathType::Vector(Some(3))), Some(&"vec3"));
    }

    #[test]
    fn test_context_source_explicit() {
        let source = ContextSource::Explicit;
        assert_eq!(source, ContextSource::Explicit);
    }

    #[test]
    fn test_context_source_declaration() {
        let source = ContextSource::Declaration;
        assert_eq!(source, ContextSource::Declaration);
    }

    #[test]
    fn test_context_source_structural() {
        let source = ContextSource::Structural;
        assert_eq!(source, ContextSource::Structural);
    }

    #[test]
    fn test_context_source_convention() {
        let source = ContextSource::Convention;
        assert_eq!(source, ContextSource::Convention);
    }

    #[test]
    fn test_context_source_default() {
        let source = ContextSource::Default;
        assert_eq!(source, ContextSource::Default);
    }

    #[test]
    fn test_context_source_not_equal() {
        assert_ne!(ContextSource::Explicit, ContextSource::Declaration);
        assert_ne!(ContextSource::Structural, ContextSource::Convention);
    }

    #[test]
    fn test_context_source_clone() {
        let source = ContextSource::Explicit;
        let cloned = source;
        assert_eq!(source, cloned);
    }

    #[test]
    fn test_context_source_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(ContextSource::Explicit, "explicit");
        map.insert(ContextSource::Structural, "structural");
        assert_eq!(map.get(&ContextSource::Explicit), Some(&"explicit"));
        assert_eq!(map.get(&ContextSource::Structural), Some(&"structural"));
    }

    #[test]
    fn test_expression_metadata_default() {
        let meta = ExpressionMetadata::default();
        assert_eq!(meta.inferred_type, None);
        assert_eq!(meta.context_source, ContextSource::Default);
        assert_eq!(meta.confidence, 0.0);
    }

    #[test]
    fn test_expression_metadata_explicit() {
        let meta = ExpressionMetadata::explicit(MathType::Scalar);
        assert_eq!(meta.inferred_type, Some(MathType::Scalar));
        assert_eq!(meta.context_source, ContextSource::Explicit);
        assert_eq!(meta.confidence, 1.0);
    }

    #[test]
    fn test_expression_metadata_explicit_vector() {
        let meta = ExpressionMetadata::explicit(MathType::Vector(Some(3)));
        assert_eq!(meta.inferred_type, Some(MathType::Vector(Some(3))));
        assert_eq!(meta.context_source, ContextSource::Explicit);
        assert_eq!(meta.confidence, 1.0);
    }

    #[test]
    fn test_expression_metadata_structural() {
        let meta = ExpressionMetadata::structural(MathType::Scalar, 0.9);
        assert_eq!(meta.inferred_type, Some(MathType::Scalar));
        assert_eq!(meta.context_source, ContextSource::Structural);
        assert_eq!(meta.confidence, 0.9);
    }

    #[test]
    fn test_expression_metadata_structural_low_confidence() {
        let meta = ExpressionMetadata::structural(MathType::Unknown, 0.3);
        assert_eq!(meta.inferred_type, Some(MathType::Unknown));
        assert_eq!(meta.context_source, ContextSource::Structural);
        assert_eq!(meta.confidence, 0.3);
    }

    #[test]
    fn test_expression_metadata_clone() {
        let meta = ExpressionMetadata::explicit(MathType::Scalar);
        let cloned = meta.clone();
        assert_eq!(meta, cloned);
    }

    #[test]
    fn test_expression_metadata_partial_eq() {
        let meta1 = ExpressionMetadata::explicit(MathType::Scalar);
        let meta2 = ExpressionMetadata::explicit(MathType::Scalar);
        let meta3 = ExpressionMetadata::structural(MathType::Scalar, 0.9);
        assert_eq!(meta1, meta2);
        assert_ne!(meta1, meta3);
    }

    #[test]
    fn test_expression_metadata_different_confidence() {
        let meta1 = ExpressionMetadata::structural(MathType::Scalar, 0.9);
        let meta2 = ExpressionMetadata::structural(MathType::Scalar, 0.8);
        assert_ne!(meta1, meta2);
    }

    #[test]
    fn test_expression_metadata_different_types() {
        let meta1 = ExpressionMetadata::explicit(MathType::Scalar);
        let meta2 = ExpressionMetadata::explicit(MathType::Vector(Some(3)));
        assert_ne!(meta1, meta2);
    }

    #[test]
    fn test_math_type_function_nested() {
        // (ℝ → ℝ) → ℝ (higher-order function)
        let func = MathType::Function {
            domain: Box::new(MathType::Function {
                domain: Box::new(MathType::Scalar),
                codomain: Box::new(MathType::Scalar),
            }),
            codomain: Box::new(MathType::Scalar),
        };

        if let MathType::Function { domain, .. } = func {
            assert!(matches!(*domain, MathType::Function { .. }));
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_math_type_tensor_rank() {
        let rank0 = MathType::Tensor(vec![]);
        let rank1 = MathType::Tensor(vec![IndexType::Upper]);
        let rank2 = MathType::Tensor(vec![IndexType::Upper, IndexType::Lower]);

        assert_ne!(rank0, rank1);
        assert_ne!(rank1, rank2);
    }

    #[test]
    fn test_math_type_different_variants() {
        let scalar = MathType::Scalar;
        let vector = MathType::Vector(None);
        let matrix = MathType::Matrix(None, None);
        let set = MathType::Set;
        let unknown = MathType::Unknown;

        assert_ne!(scalar, vector);
        assert_ne!(vector, matrix);
        assert_ne!(matrix, set);
        assert_ne!(set, unknown);
    }
}
