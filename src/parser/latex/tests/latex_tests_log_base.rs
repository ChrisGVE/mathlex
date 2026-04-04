// Log base tests for LaTeX parser
use super::*;

// \log with explicit base (2-arg form: args[0]=arg, args[1]=base)

#[test]
fn test_log_base_10_braced() {
    let expr = parse_latex(r"\log_{10}{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "log");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Integer(10));
        }
        _ => panic!("Expected Function(log, [x, 10]), got {:?}", expr),
    }
}

#[test]
fn test_log_base_2_braced() {
    let expr = parse_latex(r"\log_{2}{n}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "log");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("n".to_string()));
            assert_eq!(args[1], Expression::Integer(2));
        }
        _ => panic!("Expected Function(log, [n, 2]), got {:?}", expr),
    }
}

#[test]
fn test_log_no_base() {
    let expr = parse_latex(r"\log{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "log");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(log, [x]), got {:?}", expr),
    }
}

#[test]
fn test_ln_no_base() {
    let expr = parse_latex(r"\ln{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function(ln, [x]), got {:?}", expr),
    }
}

#[test]
fn test_log_base_variable() {
    let expr = parse_latex(r"\log_{b}{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "log");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Variable("b".to_string()));
        }
        _ => panic!("Expected Function(log, [x, b]), got {:?}", expr),
    }
}

// Round-trip: parse -> to_latex -> parse

#[test]
fn test_log_base_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\log_{10}{x}").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}

#[test]
fn test_log_no_base_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\log{x}").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}

#[test]
fn test_ln_roundtrip() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"\ln{x}").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}
