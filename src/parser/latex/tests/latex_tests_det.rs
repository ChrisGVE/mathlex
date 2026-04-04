// Determinant function tests for LaTeX parser
use super::*;

#[test]
fn test_det_braced_arg() {
    let expr = parse_latex(r"\det{A}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "det");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Function(det, [A]), got {:?}", expr),
    }
}

#[test]
fn test_det_paren_arg() {
    let expr = parse_latex(r"\det(A)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "det");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Function(det, [A]), got {:?}", expr),
    }
}

#[test]
fn test_det_unbraced_arg() {
    let expr = parse_latex(r"\det A").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "det");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Function(det, [A]), got {:?}", expr),
    }
}

// Round-trip: \det is a KNOWN_FUNCTION so to_latex emits \det\left(A\right)

#[test]
fn test_det_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\det{A}").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}
