// Floor and ceil delimiter tests for LaTeX parser
use super::*;

#[test]
fn test_floor_simple() {
    let expr = parse_latex(r"\lfloor x \rfloor").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "floor");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(floor, [x]), got {:?}", expr),
    }
}

#[test]
fn test_ceil_simple() {
    let expr = parse_latex(r"\lceil x \rceil").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ceil");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(ceil, [x]), got {:?}", expr),
    }
}

#[test]
fn test_floor_expression() {
    let expr = parse_latex(r"\lfloor x + 1 \rfloor").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "floor");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ),
                "Expected Add expression, got {:?}",
                args[0]
            );
        }
        _ => panic!("Expected Function(floor, [x+1]), got {:?}", expr),
    }
}

#[test]
fn test_ceil_expression() {
    let expr = parse_latex(r"\lceil \frac{n}{2} \rceil").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ceil");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ),
                "Expected Div expression, got {:?}",
                args[0]
            );
        }
        _ => panic!("Expected Function(ceil, [n/2]), got {:?}", expr),
    }
}

#[test]
fn test_floor_missing_close_errors() {
    let result = parse_latex(r"\lfloor x \rceil");
    assert!(
        result.is_err(),
        "Expected error for mismatched floor delimiters"
    );
}

#[test]
fn test_ceil_missing_close_errors() {
    let result = parse_latex(r"\lceil x \rfloor");
    assert!(
        result.is_err(),
        "Expected error for mismatched ceil delimiters"
    );
}

// Round-trip: parse -> to_latex -> parse

#[test]
fn test_floor_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\lfloor x \rfloor").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}

#[test]
fn test_ceil_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\lceil x \rceil").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}
