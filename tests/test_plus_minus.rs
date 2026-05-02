//! Tests for plus-minus (\pm) and minus-plus (\mp) operators

use mathlex::ast::{BinaryOp, ExprKind, Expression};
use mathlex::parser::parse_latex;

#[test]
fn test_parse_pm_simple() {
    let expr = parse_latex(r"x \pm y").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            assert_eq!(*op, BinaryOp::PlusMinus);
            assert_eq!(**left, Expression::variable("x".to_string()));
            assert_eq!(**right, Expression::variable("y".to_string()));
        }
        _ => panic!("Expected binary expression with PlusMinus"),
    }
}

#[test]
fn test_parse_mp_simple() {
    let expr = parse_latex(r"a \mp b").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            assert_eq!(*op, BinaryOp::MinusPlus);
            assert_eq!(**left, Expression::variable("a".to_string()));
            assert_eq!(**right, Expression::variable("b".to_string()));
        }
        _ => panic!("Expected binary expression with MinusPlus"),
    }
}

#[test]
fn test_parse_pm_with_numbers() {
    let expr = parse_latex(r"5 \pm 2").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            assert_eq!(*op, BinaryOp::PlusMinus);
            assert_eq!(**left, Expression::integer(5));
            assert_eq!(**right, Expression::integer(2));
        }
        _ => panic!("Expected binary expression with PlusMinus"),
    }
}

#[test]
fn test_parse_mp_with_numbers() {
    let expr = parse_latex(r"10 \mp 3").unwrap();
    match &expr.kind {
        ExprKind::Binary { op, left, right } => {
            assert_eq!(*op, BinaryOp::MinusPlus);
            assert_eq!(**left, Expression::integer(10));
            assert_eq!(**right, Expression::integer(3));
        }
        _ => panic!("Expected binary expression with MinusPlus"),
    }
}

#[test]
fn test_parse_quadratic_formula() {
    // x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    // Simplified: x = -b \pm \sqrt{b}
    let expr = parse_latex(r"x = -b \pm \sqrt{b}").unwrap();
    match &expr.kind {
        ExprKind::Equation { left, right } => {
            assert_eq!(**left, Expression::variable("x".to_string()));
            // Right side should be: -b ± sqrt(b)
            match &right.kind {
                ExprKind::Binary {
                    op: BinaryOp::PlusMinus,
                    ..
                } => {} // Expected
                _ => panic!("Expected PlusMinus in quadratic formula"),
            }
        }
        _ => panic!("Expected equation"),
    }
}

#[test]
fn test_parse_pm_precedence_with_multiplication() {
    // a * b ± c should parse as (a * b) ± c
    let expr = parse_latex(r"a * b \pm c").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::PlusMinus,
            left,
            right,
        } => {
            // Left should be multiplication
            assert!(matches!(
                left.kind,
                ExprKind::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            ));
            assert_eq!(**right, Expression::variable("c".to_string()));
        }
        _ => panic!("Expected PlusMinus with proper precedence"),
    }
}

#[test]
fn test_parse_pm_precedence_with_addition() {
    // a + b ± c should parse as (a + b) ± c (left-to-right for same precedence)
    let expr = parse_latex(r"a + b \pm c").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::PlusMinus,
            left,
            right,
        } => {
            // Left should be addition
            assert!(matches!(
                left.kind,
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert_eq!(**right, Expression::variable("c".to_string()));
        }
        _ => panic!("Expected PlusMinus with proper precedence"),
    }
}

#[test]
fn test_parse_mp_in_expression() {
    // Test ∓ in a more complex expression
    let expr = parse_latex(r"2x \mp y").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::MinusPlus,
            left,
            right,
        } => {
            // Left should be implicit multiplication: 2*x
            assert!(matches!(
                left.kind,
                ExprKind::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            ));
            assert_eq!(**right, Expression::variable("y".to_string()));
        }
        _ => panic!("Expected MinusPlus with implicit multiplication"),
    }
}

#[test]
fn test_display_pm() {
    use std::fmt::Write as _;

    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::PlusMinus,
        left: Box::new(Expression::variable("a".to_string())),
        right: Box::new(Expression::variable("b".to_string())),
    }
    .into();

    let mut output = String::new();
    write!(&mut output, "{}", expr).unwrap();
    assert_eq!(output, "a ± b");
}

#[test]
fn test_display_mp() {
    use std::fmt::Write as _;

    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::MinusPlus,
        left: Box::new(Expression::integer(5)),
        right: Box::new(Expression::integer(2)),
    }
    .into();

    let mut output = String::new();
    write!(&mut output, "{}", expr).unwrap();
    assert_eq!(output, "5 ∓ 2");
}

#[test]
fn test_to_latex_pm() {
    use mathlex::latex::ToLatex;

    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::PlusMinus,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();

    assert_eq!(expr.to_latex(), r"x \pm y");
}

#[test]
fn test_to_latex_mp() {
    use mathlex::latex::ToLatex;

    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::MinusPlus,
        left: Box::new(Expression::integer(3)),
        right: Box::new(Expression::integer(1)),
    }
    .into();

    assert_eq!(expr.to_latex(), r"3 \mp 1");
}

#[test]
fn test_round_trip_pm() {
    use mathlex::latex::ToLatex;

    let original = r"a \pm b";
    let parsed = parse_latex(original).unwrap();
    let latex_output = parsed.to_latex();

    // Should round-trip successfully
    let reparsed = parse_latex(&latex_output).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn test_round_trip_mp() {
    use mathlex::latex::ToLatex;

    let original = r"x \mp y";
    let parsed = parse_latex(original).unwrap();
    let latex_output = parsed.to_latex();

    // Should round-trip successfully
    let reparsed = parse_latex(&latex_output).unwrap();
    assert_eq!(parsed, reparsed);
}

#[test]
fn test_pm_with_parentheses() {
    let expr = parse_latex(r"(a + b) \pm (c - d)").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::PlusMinus,
            left,
            right,
        } => {
            assert!(matches!(
                left.kind,
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert!(matches!(
                right.kind,
                ExprKind::Binary {
                    op: BinaryOp::Sub,
                    ..
                }
            ));
        }
        _ => panic!("Expected PlusMinus with parenthesized operands"),
    }
}

#[test]
fn test_binary_op_display() {
    assert_eq!(format!("{}", BinaryOp::PlusMinus), "±");
    assert_eq!(format!("{}", BinaryOp::MinusPlus), "∓");
}

#[test]
fn test_binary_op_to_latex() {
    use mathlex::latex::ToLatex;

    assert_eq!(BinaryOp::PlusMinus.to_latex(), r"\pm");
    assert_eq!(BinaryOp::MinusPlus.to_latex(), r"\mp");
}
