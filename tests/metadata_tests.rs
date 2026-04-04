use mathlex::ast::IndexType;
use mathlex::metadata::{ContextSource, ExpressionMetadata, MathType};
use std::collections::HashMap;

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
