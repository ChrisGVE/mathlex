//! Tests for multiple integral parsing in LaTeX.

use crate::ast::Expression;
use crate::parser::parse_latex;

// ============================================================
// Double Integral Tests
// ============================================================

#[test]
fn test_double_integral_basic() {
    let expr = parse_latex(r"\iint f dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(*integrand, Expression::Variable("f".to_string()));
            assert!(bounds.is_none());
            assert_eq!(vars, vec!["y".to_string(), "x".to_string()]);
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}

#[test]
fn test_double_integral_with_region() {
    let expr = parse_latex(r"\iint_R f dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            vars,
            ..
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(*integrand, Expression::Variable("f".to_string()));
            assert_eq!(vars, vec!["y".to_string(), "x".to_string()]);
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}

// ============================================================
// Triple Integral Tests
// ============================================================

#[test]
fn test_triple_integral_basic() {
    let expr = parse_latex(r"\iiint f dz dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            assert_eq!(dimension, 3);
            assert_eq!(*integrand, Expression::Variable("f".to_string()));
            assert!(bounds.is_none());
            assert_eq!(
                vars,
                vec!["z".to_string(), "y".to_string(), "x".to_string()]
            );
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}

#[test]
fn test_triple_integral_with_volume() {
    let expr = parse_latex(r"\iiint_V f dz dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension, vars, ..
        } => {
            assert_eq!(dimension, 3);
            assert_eq!(
                vars,
                vec!["z".to_string(), "y".to_string(), "x".to_string()]
            );
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}

// ============================================================
// Quadruple Integral Tests
// ============================================================

#[test]
fn test_quad_integral_basic() {
    let expr = parse_latex(r"\iiiint f dw dz dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            vars,
            ..
        } => {
            assert_eq!(dimension, 4);
            assert_eq!(*integrand, Expression::Variable("f".to_string()));
            assert_eq!(
                vars,
                vec![
                    "w".to_string(),
                    "z".to_string(),
                    "y".to_string(),
                    "x".to_string()
                ]
            );
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}

// ============================================================
// Closed Integral Tests
// ============================================================

#[test]
fn test_closed_line_integral() {
    let expr = parse_latex(r"\oint F dr").unwrap();
    match expr {
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => {
            assert_eq!(dimension, 1);
            assert_eq!(*integrand, Expression::Variable("F".to_string()));
            assert!(surface.is_none());
            assert_eq!(var, "r");
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

#[test]
fn test_closed_line_integral_with_curve() {
    let expr = parse_latex(r"\oint_C F dr").unwrap();
    match expr {
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => {
            assert_eq!(dimension, 1);
            assert_eq!(*integrand, Expression::Variable("F".to_string()));
            assert_eq!(surface, Some("C".to_string()));
            assert_eq!(var, "r");
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

#[test]
fn test_closed_surface_integral() {
    let expr = parse_latex(r"\oiint F dA").unwrap();
    match expr {
        Expression::ClosedIntegral {
            dimension,
            integrand,
            var,
            ..
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(*integrand, Expression::Variable("F".to_string()));
            assert_eq!(var, "A");
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

#[test]
fn test_closed_surface_integral_with_surface() {
    let expr = parse_latex(r"\oiint_S F dA").unwrap();
    match expr {
        Expression::ClosedIntegral {
            dimension,
            surface,
            var,
            ..
        } => {
            assert_eq!(dimension, 2);
            assert_eq!(surface, Some("S".to_string()));
            assert_eq!(var, "A");
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

#[test]
fn test_closed_volume_integral() {
    let expr = parse_latex(r"\oiiint F dV").unwrap();
    match expr {
        Expression::ClosedIntegral {
            dimension,
            integrand,
            var,
            ..
        } => {
            assert_eq!(dimension, 3);
            assert_eq!(*integrand, Expression::Variable("F".to_string()));
            assert_eq!(var, "V");
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

// ============================================================
// Complex Integrand Tests
// ============================================================

#[test]
fn test_double_integral_complex_integrand() {
    let expr = parse_latex(r"\iint x y dy dx").unwrap();
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            vars,
            ..
        } => {
            assert_eq!(dimension, 2);
            // x * y (implicit multiplication)
            match *integrand {
                Expression::Binary {
                    op: crate::ast::BinaryOp::Mul,
                    ..
                } => {}
                _ => panic!("Expected multiplication in integrand, got {:?}", integrand),
            }
            assert_eq!(vars, vec!["y".to_string(), "x".to_string()]);
        }
        _ => panic!("Expected MultipleIntegral, got {:?}", expr),
    }
}
