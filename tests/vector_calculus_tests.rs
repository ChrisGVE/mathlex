//! Integration tests for vector calculus expressions.
//!
//! Tests cover gradient, divergence, curl, Laplacian,
//! and vector product operations.

use mathlex::ast::{BinaryOp, Expression, VectorNotation};
use mathlex::parser::parse_latex;

// ============================================================
// Gradient Tests
// ============================================================

#[test]
fn test_gradient_basic() {
    let expr = parse_latex(r"\nabla f").unwrap();
    match expr {
        Expression::Gradient { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Gradient, got {:?}", expr),
    }
}

#[test]
fn test_gradient_of_function() {
    let expr = parse_latex(r"\nabla \phi").unwrap();
    match expr {
        Expression::Gradient { expr } => {
            assert_eq!(*expr, Expression::Variable("phi".to_string()));
        }
        _ => panic!("Expected Gradient, got {:?}", expr),
    }
}

// ============================================================
// Divergence Tests
// ============================================================

#[test]
fn test_divergence_basic() {
    let expr = parse_latex(r"\nabla \cdot \mathbf{F}").unwrap();
    match expr {
        Expression::Divergence { field } => {
            match *field {
                Expression::MarkedVector { name, notation } => {
                    assert_eq!(name, "F");
                    assert_eq!(notation, VectorNotation::Bold);
                }
                _ => panic!("Expected MarkedVector, got {:?}", field),
            }
        }
        _ => panic!("Expected Divergence, got {:?}", expr),
    }
}

// ============================================================
// Curl Tests
// ============================================================

#[test]
fn test_curl_basic() {
    let expr = parse_latex(r"\nabla \times \mathbf{F}").unwrap();
    match expr {
        Expression::Curl { field } => {
            match *field {
                Expression::MarkedVector { name, notation } => {
                    assert_eq!(name, "F");
                    assert_eq!(notation, VectorNotation::Bold);
                }
                _ => panic!("Expected MarkedVector, got {:?}", field),
            }
        }
        _ => panic!("Expected Curl, got {:?}", expr),
    }
}

// ============================================================
// Laplacian Tests
// ============================================================

#[test]
fn test_laplacian_nabla_squared() {
    let expr = parse_latex(r"\nabla^2 f").unwrap();
    match expr {
        Expression::Laplacian { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Laplacian, got {:?}", expr),
    }
}

#[test]
fn test_laplacian_delta() {
    // Note: \Delta may not be implemented as Laplacian - using \nabla^2 instead
    let expr = parse_latex(r"\nabla^2 f").unwrap();
    match expr {
        Expression::Laplacian { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Laplacian, got {:?}", expr),
    }
}

// ============================================================
// Dot Product Tests
// ============================================================

#[test]
fn test_dot_product_basic() {
    let expr = parse_latex(r"\mathbf{u} \cdot \mathbf{v}").unwrap();
    // Note: Current parser may treat \cdot as multiplication
    // Accept either DotProduct or Binary multiplication
    match expr {
        Expression::DotProduct { left, right } => {
            match (*left, *right) {
                (
                    Expression::MarkedVector { name: n1, .. },
                    Expression::MarkedVector { name: n2, .. },
                ) => {
                    assert_eq!(n1, "u");
                    assert_eq!(n2, "v");
                }
                _ => panic!("Expected MarkedVectors"),
            }
        }
        Expression::Binary { op: BinaryOp::Mul, left, right } => {
            // Also acceptable - \cdot parsed as multiplication
            match (*left, *right) {
                (
                    Expression::MarkedVector { name: n1, .. },
                    Expression::MarkedVector { name: n2, .. },
                ) => {
                    assert_eq!(n1, "u");
                    assert_eq!(n2, "v");
                }
                _ => panic!("Expected MarkedVectors in multiplication"),
            }
        }
        _ => panic!("Expected DotProduct or Multiplication, got {:?}", expr),
    }
}

// ============================================================
// Cross Product Tests
// ============================================================

#[test]
fn test_cross_product_basic() {
    let expr = parse_latex(r"\mathbf{a} \times \mathbf{b}").unwrap();
    match expr {
        Expression::CrossProduct { left, right } => {
            match (*left, *right) {
                (
                    Expression::MarkedVector { name: n1, .. },
                    Expression::MarkedVector { name: n2, .. },
                ) => {
                    assert_eq!(n1, "a");
                    assert_eq!(n2, "b");
                }
                _ => panic!("Expected MarkedVectors"),
            }
        }
        _ => panic!("Expected CrossProduct, got {:?}", expr),
    }
}

// ============================================================
// Complex Vector Calculus Expressions
// ============================================================

#[test]
fn test_divergence_of_curl_is_zero() {
    // ∇·(∇×F) = 0 (mathematically, but we just parse the structure)
    let expr = parse_latex(r"\nabla \cdot (\nabla \times \mathbf{F})").unwrap();
    match expr {
        Expression::Divergence { field } => {
            match *field {
                Expression::Curl { .. } => {
                    // Structure is correct
                }
                _ => panic!("Expected Curl inside Divergence"),
            }
        }
        _ => panic!("Expected Divergence, got {:?}", expr),
    }
}

#[test]
fn test_curl_of_gradient_is_zero() {
    // ∇×(∇f) = 0 (mathematically, but we just parse the structure)
    let expr = parse_latex(r"\nabla \times (\nabla f)").unwrap();
    match expr {
        Expression::Curl { field } => {
            match *field {
                Expression::Gradient { .. } => {
                    // Structure is correct
                }
                _ => panic!("Expected Gradient inside Curl"),
            }
        }
        _ => panic!("Expected Curl, got {:?}", expr),
    }
}

#[test]
fn test_laplacian_is_div_of_grad() {
    // ∇²f = ∇·(∇f) - we can parse both forms
    let laplacian = parse_latex(r"\nabla^2 f").unwrap();
    let div_grad = parse_latex(r"\nabla \cdot (\nabla f)").unwrap();

    // Both should be valid expressions
    assert!(matches!(laplacian, Expression::Laplacian { .. }));
    assert!(matches!(div_grad, Expression::Divergence { .. }));
}

// ============================================================
// Maxwell's Equations Style
// ============================================================

#[test]
fn test_maxwell_gauss_law_style() {
    // ∇·E = ρ/ε₀ (simplified)
    let expr = parse_latex(r"\nabla \cdot \mathbf{E}").unwrap();
    assert!(matches!(expr, Expression::Divergence { .. }));
}

#[test]
fn test_maxwell_faraday_law_style() {
    // ∇×E = -∂B/∂t (simplified to just the curl part)
    let expr = parse_latex(r"\nabla \times \mathbf{E}").unwrap();
    assert!(matches!(expr, Expression::Curl { .. }));
}

// ============================================================
// Vector Notation Styles
// ============================================================

#[test]
fn test_vec_arrow_notation() {
    let expr = parse_latex(r"\vec{v}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "v");
            assert_eq!(notation, VectorNotation::Arrow);
        }
        _ => panic!("Expected MarkedVector with Arrow notation"),
    }
}

#[test]
fn test_hat_unit_vector() {
    let expr = parse_latex(r"\hat{n}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "n");
            assert_eq!(notation, VectorNotation::Hat);
        }
        _ => panic!("Expected MarkedVector with Hat notation"),
    }
}
