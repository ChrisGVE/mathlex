// Greek letter tests for LaTeX parser
use super::*;
use crate::latex::ToLatex;

#[test]
fn test_lowercase_greek_letters() {
    // Test all lowercase Greek letters (except pi which is a constant)
    let greek_letters = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "rho", "sigma", "tau", "upsilon", "phi", "chi",
        "psi", "omega",
    ];

    for letter in &greek_letters {
        let input = format!(r"\{}", letter);
        let expr = parse_latex(&input).unwrap();
        assert_eq!(
            expr,
            Expression::Variable(letter.to_string()),
            "Failed to parse \\{}",
            letter
        );
    }
}

#[test]
fn test_uppercase_greek_letters() {
    let greek_letters = [
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    ];

    for letter in &greek_letters {
        let input = format!(r"\{}", letter);
        let expr = parse_latex(&input).unwrap();
        assert_eq!(
            expr,
            Expression::Variable(letter.to_string()),
            "Failed to parse \\{}",
            letter
        );
    }
}

#[test]
fn test_greek_letter_with_single_digit_subscript() {
    let expr = parse_latex(r"\alpha_1").unwrap();
    assert_eq!(expr, Expression::Variable("alpha_1".to_string()));

    let expr = parse_latex(r"\beta_2").unwrap();
    assert_eq!(expr, Expression::Variable("beta_2".to_string()));

    let expr = parse_latex(r"\gamma_i").unwrap();
    assert_eq!(expr, Expression::Variable("gamma_i".to_string()));
}

#[test]
fn test_greek_letter_with_multi_digit_subscript() {
    let expr = parse_latex(r"\alpha_{12}").unwrap();
    assert_eq!(expr, Expression::Variable("alpha_12".to_string()));

    let expr = parse_latex(r"\beta_{123}").unwrap();
    assert_eq!(expr, Expression::Variable("beta_123".to_string()));

    // Note: "max" as a subscript parses as a variable, not separate letters
    // This test verifies current behavior - subscripts can only be simple integers or single variables
    let expr = parse_latex(r"\gamma_m").unwrap();
    assert_eq!(expr, Expression::Variable("gamma_m".to_string()));
}

#[test]
fn test_uppercase_greek_with_subscript() {
    let expr = parse_latex(r"\Gamma_1").unwrap();
    assert_eq!(expr, Expression::Variable("Gamma_1".to_string()));

    let expr = parse_latex(r"\Delta_{12}").unwrap();
    assert_eq!(expr, Expression::Variable("Delta_12".to_string()));
}

#[test]
fn test_greek_letters_in_expression() {
    // α + β
    let expr = parse_latex(r"\alpha + \beta").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Variable("alpha".to_string()));
            assert_eq!(*right, Expression::Variable("beta".to_string()));
        }
        _ => panic!("Expected binary expression"),
    }

    // γ * δ^2
    let expr = parse_latex(r"\gamma * \delta^2").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("gamma".to_string()));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Pow, ..
                } => {}
                _ => panic!("Expected power"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_pi_is_constant_not_variable() {
    let expr = parse_latex(r"\pi").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::Pi));

    // pi in expressions
    let expr = parse_latex(r"2 * \pi").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Constant(MathConstant::Pi));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_infinity_constant() {
    let expr = parse_latex(r"\infty").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::Infinity));

    // Test in expressions
    let expr = parse_latex(r"x + \infty").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Constant(MathConstant::Infinity));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_greek_letter_round_trip() {
    // Single Greek letter
    let input = r"\alpha";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\alpha");

    // Parse again to verify round-trip
    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);

    // Greek letter with subscript
    let input = r"\beta_1";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\beta_1");
    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);

    // Greek letter with multi-char subscript
    let input = r"\gamma_{12}";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\gamma_{12}");
    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_uppercase_greek_round_trip() {
    let input = r"\Gamma";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\Gamma");

    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_pi_constant_round_trip() {
    let input = r"\pi";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\pi");

    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_infinity_constant_round_trip() {
    let input = r"\infty";
    let expr = parse_latex(input).unwrap();
    let output = expr.to_latex();
    assert_eq!(output, r"\infty");

    let expr2 = parse_latex(&output).unwrap();
    assert_eq!(expr, expr2);
}

#[test]
fn test_complex_expression_with_greek_letters() {
    // Test: α + β * γ^2
    // Note: to_latex() outputs \cdot for multiplication which parser doesn't support yet
    // So we test parsing and structure, not full round-trip
    let input = r"\alpha + \beta * \gamma^2";
    let expr = parse_latex(input).unwrap();

    // Verify structure
    match &expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(**left, Expression::Variable("alpha".to_string()));
            match &**right {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left,
                    right,
                } => {
                    assert_eq!(**left, Expression::Variable("beta".to_string()));
                    match &**right {
                        Expression::Binary {
                            op: BinaryOp::Pow, ..
                        } => {}
                        _ => panic!("Expected power"),
                    }
                }
                _ => panic!("Expected multiplication"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_all_lowercase_greek_round_trip() {
    let greek_letters = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
        "lambda", "mu", "nu", "xi", "omicron", "rho", "sigma", "tau", "upsilon", "phi", "chi",
        "psi", "omega",
    ];

    for letter in &greek_letters {
        let input = format!(r"\{}", letter);
        let expr = parse_latex(&input).unwrap();
        let output = expr.to_latex();
        assert_eq!(output, input, "Round-trip failed for {}", letter);

        let expr2 = parse_latex(&output).unwrap();
        assert_eq!(
            expr, expr2,
            "Parsing round-trip output failed for {}",
            letter
        );
    }
}

#[test]
fn test_all_uppercase_greek_round_trip() {
    let greek_letters = [
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    ];

    for letter in &greek_letters {
        let input = format!(r"\{}", letter);
        let expr = parse_latex(&input).unwrap();
        let output = expr.to_latex();
        assert_eq!(output, input, "Round-trip failed for {}", letter);

        let expr2 = parse_latex(&output).unwrap();
        assert_eq!(
            expr, expr2,
            "Parsing round-trip output failed for {}",
            letter
        );
    }
}

#[test]
fn test_greek_with_various_subscripts() {
    let test_cases = vec![
        (r"\alpha_1", r"\alpha_1"),
        (r"\alpha_{12}", r"\alpha_{12}"),
        (r"\beta_i", r"\beta_i"),
        (r"\Gamma_n", r"\Gamma_n"),
    ];

    for (input, expected_output) in test_cases {
        let expr = parse_latex(input).unwrap();
        let output = expr.to_latex();
        assert_eq!(output, expected_output, "to_latex failed for {}", input);

        let expr2 = parse_latex(&output).unwrap();
        assert_eq!(expr, expr2, "Round-trip failed for {}", input);
    }
}
