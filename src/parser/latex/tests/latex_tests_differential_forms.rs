//! Tests for differential forms notation (differentials and wedge products).

use crate::ast::ExprKind;
use crate::latex::ToLatex;
use crate::parser::parse_latex;

#[test]
fn test_simple_differential_dx() {
    let expr = parse_latex("dx").unwrap();
    match &expr.kind {
        ExprKind::Differential { var } => {
            assert_eq!(var, "x");
        }
        _ => panic!("Expected Differential, got {:?}", expr),
    }
}

#[test]
fn test_simple_differential_dy() {
    let expr = parse_latex("dy").unwrap();
    match &expr.kind {
        ExprKind::Differential { var } => {
            assert_eq!(var, "y");
        }
        _ => panic!("Expected Differential, got {:?}", expr),
    }
}

#[test]
fn test_simple_differential_dt() {
    let expr = parse_latex("dt").unwrap();
    match &expr.kind {
        ExprKind::Differential { var } => {
            assert_eq!(var, "t");
        }
        _ => panic!("Expected Differential, got {:?}", expr),
    }
}

#[test]
fn test_wedge_product_dx_dy() {
    let expr = parse_latex(r"dx \wedge dy").unwrap();
    match &expr.kind {
        ExprKind::WedgeProduct { left, right } => {
            match &left.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "x"),
                _ => panic!("Expected Differential for left, got {:?}", left),
            }
            match &right.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "y"),
                _ => panic!("Expected Differential for right, got {:?}", right),
            }
        }
        _ => panic!("Expected WedgeProduct, got {:?}", expr),
    }
}

#[test]
fn test_wedge_product_nested() {
    // dx ∧ dy ∧ dz
    let expr = parse_latex(r"dx \wedge dy \wedge dz").unwrap();
    match &expr.kind {
        ExprKind::WedgeProduct { left, right } => {
            // Left should be dx ∧ dy
            match &left.kind {
                ExprKind::WedgeProduct {
                    left: ref dx,
                    right: ref dy,
                } => {
                    match &dx.kind {
                        ExprKind::Differential { var: ref v } => assert_eq!(v, "x"),
                        _ => panic!("Expected Differential for dx"),
                    }
                    match &dy.kind {
                        ExprKind::Differential { var: ref v } => assert_eq!(v, "y"),
                        _ => panic!("Expected Differential for dy"),
                    }
                }
                _ => panic!("Expected WedgeProduct for left, got {:?}", left),
            }
            // Right should be dz
            match &right.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "z"),
                _ => panic!("Expected Differential for right, got {:?}", right),
            }
        }
        _ => panic!("Expected WedgeProduct, got {:?}", expr),
    }
}

#[test]
fn test_differential_with_coefficient() {
    // 2 * dx
    let expr = parse_latex("2 dx").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            use crate::ast::BinaryOp;
            assert_eq!(*op, BinaryOp::Mul);
            match &left.kind {
                ExprKind::Integer(n) => assert_eq!(*n, 2),
                _ => panic!("Expected Integer for left"),
            }
            match &right.kind {
                ExprKind::Differential { var } => assert_eq!(var, "x"),
                _ => panic!("Expected Differential for right, got {:?}", right),
            }
        }
        _ => panic!("Expected Binary multiplication, got {:?}", expr),
    }
}

#[test]
fn test_wedge_product_with_expressions() {
    // (x + y) ∧ dz
    let expr = parse_latex(r"(x + y) \wedge dz").unwrap();
    match &expr.kind {
        ExprKind::WedgeProduct { left, right } => {
            match &left.kind {
                ExprKind::Binary { .. } => {} // x + y
                _ => panic!("Expected Binary for left, got {:?}", left),
            }
            match &right.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "z"),
                _ => panic!("Expected Differential for right"),
            }
        }
        _ => panic!("Expected WedgeProduct, got {:?}", expr),
    }
}

#[test]
fn test_differential_display() {
    let expr = ExprKind::Differential {
        var: "x".to_string(),
    };
    assert_eq!(format!("{}", expr), "dx");
}

#[test]
fn test_wedge_product_display() {
    let dx = ExprKind::Differential {
        var: "x".to_string(),
    };
    let dy = ExprKind::Differential {
        var: "y".to_string(),
    };
    let wedge = ExprKind::WedgeProduct {
        left: Box::new(dx.into()),
        right: Box::new(dy.into()),
    };
    assert_eq!(format!("{}", wedge), "dx ∧ dy");
}

#[test]
fn test_differential_to_latex() {
    let expr = ExprKind::Differential {
        var: "x".to_string(),
    };
    assert_eq!(expr.to_latex(), "dx");
}

#[test]
fn test_wedge_product_to_latex() {
    let dx = ExprKind::Differential {
        var: "x".to_string(),
    };
    let dy = ExprKind::Differential {
        var: "y".to_string(),
    };
    let wedge = ExprKind::WedgeProduct {
        left: Box::new(dx.into()),
        right: Box::new(dy.into()),
    };
    assert_eq!(wedge.to_latex(), r"dx \wedge dy");
}

#[test]
fn test_differential_roundtrip() {
    let input = "dx";
    let expr = parse_latex(input).unwrap();
    let latex = expr.to_latex();
    let expr2 = parse_latex(&latex).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_wedge_product_roundtrip() {
    let input = r"dx \wedge dy";
    let expr = parse_latex(input).unwrap();
    let latex = expr.to_latex();
    let expr2 = parse_latex(&latex).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_differential_vs_derivative() {
    // Test that standalone dx is parsed as differential
    let expr = parse_latex("dx").unwrap();
    match &expr.kind {
        ExprKind::Differential { var } => {
            assert_eq!(var, "x");
        }
        _ => panic!("Expected Differential, got {:?}", expr),
    }

    // Note: \frac{d}{dx} without a following expression is incomplete
    // and handled by the derivative parser tests, not differential tests
}

#[test]
fn test_standalone_d_is_variable() {
    // 'd' by itself should be treated as a variable
    let expr = parse_latex("d").unwrap();
    match &expr.kind {
        ExprKind::Variable(v) => {
            assert_eq!(v, "d");
        }
        _ => panic!("Expected Variable, got {:?}", expr),
    }
}

#[test]
fn test_differential_in_expression() {
    // Test differential in a mathematical expression (not in integral)
    // f * dx should parse as multiplication of function by differential
    let expr = parse_latex("f dx").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            use crate::ast::BinaryOp;
            assert_eq!(*op, BinaryOp::Mul);
            match &left.kind {
                ExprKind::Variable(ref v) => assert_eq!(v, "f"),
                _ => panic!("Expected Variable for left"),
            }
            match &right.kind {
                ExprKind::Differential { ref var } => assert_eq!(var, "x"),
                _ => panic!("Expected Differential for right"),
            }
        }
        _ => panic!("Expected Binary multiplication, got {:?}", expr),
    }
}

#[test]
fn test_wedge_precedence() {
    // Test that wedge has correct precedence (similar to multiplication)
    let expr = parse_latex(r"a + dx \wedge dy").unwrap();
    // Should parse as a + (dx ∧ dy), not (a + dx) ∧ dy
    match &expr.kind {
        ExprKind::Binary {
            op,
            left,
            right: wedge,
        } => {
            use crate::ast::BinaryOp;
            assert_eq!(*op, BinaryOp::Add);
            match &left.kind {
                ExprKind::Variable(ref v) => assert_eq!(v, "a"),
                _ => panic!("Expected Variable for left"),
            }
            match &wedge.kind {
                ExprKind::WedgeProduct { .. } => {}
                _ => panic!("Expected WedgeProduct for right"),
            }
        }
        _ => panic!("Expected Binary addition, got {:?}", expr),
    }
}

#[test]
fn test_multiple_differentials_in_series() {
    // dx dy should parse as dx * dy (implicit multiplication)
    let expr = parse_latex("dx dy").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            use crate::ast::BinaryOp;
            assert_eq!(*op, BinaryOp::Mul);
            match &left.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "x"),
                _ => panic!("Expected Differential for left"),
            }
            match &right.kind {
                ExprKind::Differential { var: ref v } => assert_eq!(v, "y"),
                _ => panic!("Expected Differential for right"),
            }
        }
        _ => panic!("Expected Binary multiplication, got {:?}", expr),
    }
}
