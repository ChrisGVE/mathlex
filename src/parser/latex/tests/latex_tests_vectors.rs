//! Tests for vector notation parsing in LaTeX.

use crate::ast::{Expression, VectorNotation};
use crate::parser::parse_latex;

// ============================================================
// Marked Vector Tests
// ============================================================

#[test]
fn test_mathbf_braced() {
    let expr = parse_latex(r"\mathbf{v}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "v");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_mathbf_greek() {
    let expr = parse_latex(r"\mathbf{\alpha}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "alpha");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_boldsymbol() {
    let expr = parse_latex(r"\boldsymbol{F}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "F");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_vec_braced() {
    let expr = parse_latex(r"\vec{a}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "a");
            assert_eq!(notation, VectorNotation::Arrow);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_vec_unbraced() {
    let expr = parse_latex(r"\vec a").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "a");
            assert_eq!(notation, VectorNotation::Arrow);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_overrightarrow() {
    let expr = parse_latex(r"\overrightarrow{AB}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "AB");
            assert_eq!(notation, VectorNotation::Arrow);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_hat_braced() {
    let expr = parse_latex(r"\hat{n}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "n");
            assert_eq!(notation, VectorNotation::Hat);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_hat_unbraced() {
    let expr = parse_latex(r"\hat x").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "x");
            assert_eq!(notation, VectorNotation::Hat);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
fn test_underline() {
    let expr = parse_latex(r"\underline{v}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "v");
            assert_eq!(notation, VectorNotation::Underline);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

// ============================================================
// Nabla / Vector Calculus Tests
// ============================================================

#[test]
fn test_gradient() {
    let expr = parse_latex(r"\nabla f").unwrap();
    match expr {
        Expression::Gradient { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Gradient, got {:?}", expr),
    }
}

#[test]
fn test_gradient_braced() {
    let expr = parse_latex(r"\nabla{f}").unwrap();
    match expr {
        Expression::Gradient { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Gradient, got {:?}", expr),
    }
}

#[test]
fn test_divergence_cdot() {
    let expr = parse_latex(r"\nabla \cdot F").unwrap();
    match expr {
        Expression::Divergence { field } => {
            assert_eq!(*field, Expression::Variable("F".to_string()));
        }
        _ => panic!("Expected Divergence, got {:?}", expr),
    }
}

#[test]
fn test_divergence_bullet() {
    let expr = parse_latex(r"\nabla \bullet F").unwrap();
    match expr {
        Expression::Divergence { field } => {
            assert_eq!(*field, Expression::Variable("F".to_string()));
        }
        _ => panic!("Expected Divergence, got {:?}", expr),
    }
}

#[test]
fn test_curl() {
    let expr = parse_latex(r"\nabla \times F").unwrap();
    match expr {
        Expression::Curl { field } => {
            assert_eq!(*field, Expression::Variable("F".to_string()));
        }
        _ => panic!("Expected Curl, got {:?}", expr),
    }
}

#[test]
fn test_laplacian() {
    let expr = parse_latex(r"\nabla^2 f").unwrap();
    match expr {
        Expression::Laplacian { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Laplacian, got {:?}", expr),
    }
}

#[test]
fn test_laplacian_braced() {
    let expr = parse_latex(r"\nabla^{2} f").unwrap();
    match expr {
        Expression::Laplacian { expr } => {
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Laplacian, got {:?}", expr),
    }
}

// ============================================================
// Vector Product Tests
// ============================================================

#[test]
fn test_dot_product_bullet() {
    let expr = parse_latex(r"a \bullet b").unwrap();
    match expr {
        Expression::DotProduct { left, right } => {
            assert_eq!(*left, Expression::Variable("a".to_string()));
            assert_eq!(*right, Expression::Variable("b".to_string()));
        }
        _ => panic!("Expected DotProduct, got {:?}", expr),
    }
}

#[test]
fn test_outer_product() {
    let expr = parse_latex(r"u \otimes v").unwrap();
    match expr {
        Expression::OuterProduct { left, right } => {
            assert_eq!(*left, Expression::Variable("u".to_string()));
            assert_eq!(*right, Expression::Variable("v".to_string()));
        }
        _ => panic!("Expected OuterProduct, got {:?}", expr),
    }
}

#[test]
fn test_vector_product_with_marked_vectors() {
    let expr = parse_latex(r"\mathbf{a} \bullet \mathbf{b}").unwrap();
    match expr {
        Expression::DotProduct { left, right } => {
            match (*left, *right) {
                (
                    Expression::MarkedVector { name: n1, notation: VectorNotation::Bold },
                    Expression::MarkedVector { name: n2, notation: VectorNotation::Bold },
                ) => {
                    assert_eq!(n1, "a");
                    assert_eq!(n2, "b");
                }
                _ => panic!("Expected MarkedVectors in dot product"),
            }
        }
        _ => panic!("Expected DotProduct, got {:?}", expr),
    }
}

// ============================================================
// Complex Expression Tests
// ============================================================

#[test]
fn test_gradient_of_scalar_product() {
    // \nabla (f * g) should parse as gradient of (f * g)
    let expr = parse_latex(r"\nabla f").unwrap();
    match expr {
        Expression::Gradient { .. } => {}
        _ => panic!("Expected Gradient, got {:?}", expr),
    }
}

#[test]
fn test_vector_in_expression() {
    let expr = parse_latex(r"\mathbf{v} + \mathbf{u}").unwrap();
    match expr {
        Expression::Binary { op: crate::ast::BinaryOp::Add, left, right } => {
            match (*left, *right) {
                (
                    Expression::MarkedVector { name: n1, notation: VectorNotation::Bold },
                    Expression::MarkedVector { name: n2, notation: VectorNotation::Bold },
                ) => {
                    assert_eq!(n1, "v");
                    assert_eq!(n2, "u");
                }
                _ => panic!("Expected MarkedVectors in addition"),
            }
        }
        _ => panic!("Expected Binary Add, got {:?}", expr),
    }
}
