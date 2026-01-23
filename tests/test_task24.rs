use mathlex::parser::parse_latex;
use mathlex::ast::{Expression, RelationOp, NumberSet};
use mathlex::latex::ToLatex;

#[test]
fn test_parse_similar_relation() {
    let expr = parse_latex(r"a \sim b").unwrap();
    match expr {
        Expression::Relation { op, left, right } => {
            assert_eq!(op, RelationOp::Similar);
            assert!(matches!(*left, Expression::Variable(_)));
            assert!(matches!(*right, Expression::Variable(_)));
        }
        _ => panic!("Expected Relation expression"),
    }
}

#[test]
fn test_parse_equivalent_relation() {
    let expr = parse_latex(r"x \equiv y").unwrap();
    match expr {
        Expression::Relation { op, .. } => {
            assert_eq!(op, RelationOp::Equivalent);
        }
        _ => panic!("Expected Relation expression"),
    }
}

#[test]
fn test_parse_congruent_relation() {
    let expr = parse_latex(r"a \cong b").unwrap();
    match expr {
        Expression::Relation { op, .. } => {
            assert_eq!(op, RelationOp::Congruent);
        }
        _ => panic!("Expected Relation expression"),
    }
}

#[test]
fn test_parse_approx_relation() {
    let expr = parse_latex(r"3.14 \approx \pi").unwrap();
    match expr {
        Expression::Relation { op, .. } => {
            assert_eq!(op, RelationOp::Approx);
        }
        _ => panic!("Expected Relation expression"),
    }
}

#[test]
fn test_parse_function_composition() {
    let expr = parse_latex(r"f \circ g").unwrap();
    match expr {
        Expression::Composition { outer, inner } => {
            assert!(matches!(*outer, Expression::Variable(_)));
            assert!(matches!(*inner, Expression::Variable(_)));
        }
        _ => panic!("Expected Composition expression"),
    }
}

#[test]
fn test_parse_function_signature_reals() {
    let expr = parse_latex(r"f: \mathbb{R} \to \mathbb{R}").unwrap();
    match expr {
        Expression::FunctionSignature { name, domain, codomain } => {
            assert_eq!(name, "f");
            assert!(matches!(*domain, Expression::NumberSetExpr(NumberSet::Real)));
            assert!(matches!(*codomain, Expression::NumberSetExpr(NumberSet::Real)));
        }
        _ => panic!("Expected FunctionSignature expression"),
    }
}

#[test]
fn test_parse_function_signature_complex() {
    let expr = parse_latex(r"g: \mathbb{C} \to \mathbb{R}").unwrap();
    match expr {
        Expression::FunctionSignature { name, domain, codomain } => {
            assert_eq!(name, "g");
            assert!(matches!(*domain, Expression::NumberSetExpr(NumberSet::Complex)));
            assert!(matches!(*codomain, Expression::NumberSetExpr(NumberSet::Real)));
        }
        _ => panic!("Expected FunctionSignature expression"),
    }
}

#[test]
fn test_round_trip_similar() {
    let input = r"a \sim b";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"a \sim b");
}

#[test]
fn test_round_trip_equiv() {
    let input = r"x \equiv y";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"x \equiv y");
}

#[test]
fn test_round_trip_cong() {
    let input = r"a \cong b";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"a \cong b");
}

#[test]
fn test_round_trip_approx() {
    let input = r"x \approx y";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"x \approx y");
}

#[test]
fn test_round_trip_composition() {
    let input = r"f \circ g";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"f \circ g");
}

#[test]
fn test_round_trip_function_signature() {
    let input = r"f: \mathbb{R} \to \mathbb{R}";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"f: \mathbb{R} \to \mathbb{R}");
}

#[test]
fn test_display_similar() {
    let expr = parse_latex(r"a \sim b").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "a ~ b");
}

#[test]
fn test_display_equivalent() {
    let expr = parse_latex(r"x \equiv y").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "x ≡ y");
}

#[test]
fn test_display_congruent() {
    let expr = parse_latex(r"a \cong b").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "a ≅ b");
}

#[test]
fn test_display_approx() {
    let expr = parse_latex(r"x \approx y").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "x ≈ y");
}

#[test]
fn test_display_composition() {
    let expr = parse_latex(r"f \circ g").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "f ∘ g");
}

#[test]
fn test_display_function_signature() {
    let expr = parse_latex(r"f: \mathbb{R} \to \mathbb{R}").unwrap();
    let output = format!("{}", expr);
    assert_eq!(output, "f: ℝ → ℝ");
}
