//! Tests for tensor notation parsing in LaTeX.

use crate::ast::{Expression, IndexType};
use crate::parser::parse_latex;

#[test]
fn test_kronecker_delta_lower_indices() {
    let expr = parse_latex(r"\delta_{ij}").unwrap();
    match expr {
        Expression::KroneckerDelta { indices } => {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, "i");
            assert_eq!(indices[0].index_type, IndexType::Lower);
            assert_eq!(indices[1].name, "j");
            assert_eq!(indices[1].index_type, IndexType::Lower);
        }
        _ => panic!("Expected KroneckerDelta, got {:?}", expr),
    }
}

#[test]
fn test_kronecker_delta_mixed_indices() {
    let expr = parse_latex(r"\delta^i_j").unwrap();
    match expr {
        Expression::KroneckerDelta { indices } => {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, "i");
            assert_eq!(indices[0].index_type, IndexType::Upper);
            assert_eq!(indices[1].name, "j");
            assert_eq!(indices[1].index_type, IndexType::Lower);
        }
        _ => panic!("Expected KroneckerDelta, got {:?}", expr),
    }
}

#[test]
fn test_kronecker_delta_upper_indices() {
    let expr = parse_latex(r"\delta^{ij}").unwrap();
    match expr {
        Expression::KroneckerDelta { indices } => {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, "i");
            assert_eq!(indices[0].index_type, IndexType::Upper);
            assert_eq!(indices[1].name, "j");
            assert_eq!(indices[1].index_type, IndexType::Upper);
        }
        _ => panic!("Expected KroneckerDelta, got {:?}", expr),
    }
}

#[test]
fn test_levi_civita_lower_indices() {
    let expr = parse_latex(r"\varepsilon_{ijk}").unwrap();
    match expr {
        Expression::LeviCivita { indices } => {
            assert_eq!(indices.len(), 3);
            assert_eq!(indices[0].name, "i");
            assert_eq!(indices[1].name, "j");
            assert_eq!(indices[2].name, "k");
            for idx in &indices {
                assert_eq!(idx.index_type, IndexType::Lower);
            }
        }
        _ => panic!("Expected LeviCivita, got {:?}", expr),
    }
}

#[test]
fn test_levi_civita_epsilon_alias() {
    let expr = parse_latex(r"\epsilon_{abc}").unwrap();
    match expr {
        Expression::LeviCivita { indices } => {
            assert_eq!(indices.len(), 3);
            assert_eq!(indices[0].name, "a");
            assert_eq!(indices[1].name, "b");
            assert_eq!(indices[2].name, "c");
        }
        _ => panic!("Expected LeviCivita, got {:?}", expr),
    }
}

#[test]
fn test_levi_civita_upper_indices() {
    let expr = parse_latex(r"\varepsilon^{ijk}").unwrap();
    match expr {
        Expression::LeviCivita { indices } => {
            assert_eq!(indices.len(), 3);
            for idx in &indices {
                assert_eq!(idx.index_type, IndexType::Upper);
            }
        }
        _ => panic!("Expected LeviCivita, got {:?}", expr),
    }
}

#[test]
fn test_delta_without_indices_is_variable() {
    let expr = parse_latex(r"\delta + 1").unwrap();
    match expr {
        Expression::Binary { left, .. } => match *left {
            Expression::Variable(name) => assert_eq!(name, "delta"),
            _ => panic!("Expected Variable(delta), got {:?}", left),
        },
        _ => panic!("Expected Binary expression, got {:?}", expr),
    }
}

#[test]
fn test_delta_power_is_not_tensor() {
    // \delta^2 should be delta squared, not a tensor
    let expr = parse_latex(r"\delta^2").unwrap();
    match expr {
        Expression::Binary {
            op: crate::ast::BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("delta".to_string()));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected Power expression, got {:?}", expr),
    }
}

#[test]
fn test_epsilon_without_indices_is_variable() {
    let expr = parse_latex(r"\epsilon").unwrap();
    assert_eq!(expr, Expression::Variable("epsilon".to_string()));
}

#[test]
fn test_varepsilon_without_indices_is_variable() {
    let expr = parse_latex(r"\varepsilon").unwrap();
    assert_eq!(expr, Expression::Variable("varepsilon".to_string()));
}

#[test]
fn test_delta_single_index() {
    let expr = parse_latex(r"\delta_i").unwrap();
    match expr {
        Expression::KroneckerDelta { indices } => {
            assert_eq!(indices.len(), 1);
            assert_eq!(indices[0].name, "i");
            assert_eq!(indices[0].index_type, IndexType::Lower);
        }
        _ => panic!("Expected KroneckerDelta, got {:?}", expr),
    }
}

#[test]
fn test_tensor_with_greek_indices() {
    let expr = parse_latex(r"\delta^{\mu}_{\nu}").unwrap();
    match expr {
        Expression::KroneckerDelta { indices } => {
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, "mu");
            assert_eq!(indices[0].index_type, IndexType::Upper);
            assert_eq!(indices[1].name, "nu");
            assert_eq!(indices[1].index_type, IndexType::Lower);
        }
        _ => panic!("Expected KroneckerDelta, got {:?}", expr),
    }
}
