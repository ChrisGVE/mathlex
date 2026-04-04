// Abs and sgn function tests for LaTeX parser
use super::*;

// |x| via bare pipe (already tested elsewhere, confirm here)

#[test]
fn test_abs_bare_pipe() {
    let expr = parse_latex(r"|x|").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "abs");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(abs, [x]), got {:?}", expr),
    }
}

// \left| x \right| should also parse as abs

#[test]
fn test_abs_left_right_pipe() {
    let expr = parse_latex(r"\left| x \right|").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "abs");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(abs, [x]), got {:?}", expr),
    }
}

#[test]
fn test_abs_expression() {
    let expr = parse_latex(r"\left| x - y \right|").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "abs");
            assert_eq!(args.len(), 1);
            assert!(
                matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ),
                "Expected Sub expression, got {:?}",
                args[0]
            );
        }
        _ => panic!("Expected Function(abs, [x-y]), got {:?}", expr),
    }
}

// \operatorname{sgn}

#[test]
fn test_sgn_operatorname() {
    let expr = parse_latex(r"\operatorname{sgn}{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sgn");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(sgn, [x]), got {:?}", expr),
    }
}

#[test]
fn test_sgn_operatorname_parens() {
    let expr = parse_latex(r"\operatorname{sgn}(x)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sgn");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(sgn, [x]), got {:?}", expr),
    }
}

// \operatorname can name arbitrary functions

#[test]
fn test_operatorname_arbitrary() {
    let expr = parse_latex(r"\operatorname{myf}(x)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "myf");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(myf, [x]), got {:?}", expr),
    }
}

// Round-trip tests

#[test]
fn test_abs_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"|x|").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}

#[test]
fn test_sgn_roundtrip() {
    use crate::latex::ToLatex;
    // \sgn is not a known LaTeX command so to_latex emits \operatorname{sgn}(x)
    let expr = parse_latex(r"\operatorname{sgn}(x)").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}
