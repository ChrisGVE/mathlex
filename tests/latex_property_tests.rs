#![allow(clippy::approx_constant)]
//! Property-based tests for LaTeX parser using proptest.
//!
//! This module uses proptest to generate arbitrary valid LaTeX expressions
//! and verify important properties of the LaTeX parser.

use mathlex::ast::{BinaryOp, Expression, MathConstant, UnaryOp};
use mathlex::{parse_latex, ToLatex};
use proptest::prelude::*;

/// Strategy to generate simple LaTeX numbers.
///
/// Generates integers and floats in LaTeX format.
fn arb_latex_number() -> impl Strategy<Value = String> {
    prop_oneof![
        // Positive integers
        (1i64..1000).prop_map(|n| n.to_string()),
        // Negative integers
        (-1000i64..-1).prop_map(|n| n.to_string()),
        // Positive floats
        (1.0f64..100.0).prop_map(|f| format!("{:.2}", f)),
        // Negative floats
        (-100.0f64..-1.0).prop_map(|f| format!("{:.2}", f)),
    ]
}

/// Strategy to generate LaTeX variable names.
///
/// Generates single-letter variables and Greek letters.
fn arb_latex_variable() -> impl Strategy<Value = String> {
    prop_oneof![
        // Single letter variables
        "[a-z]".prop_map(|s| s),
        // Greek letters
        Just(r"\alpha".to_string()),
        Just(r"\beta".to_string()),
        Just(r"\gamma".to_string()),
        Just(r"\theta".to_string()),
        Just(r"\phi".to_string()),
        Just(r"\psi".to_string()),
        Just(r"\omega".to_string()),
    ]
}

/// Strategy to generate LaTeX constants.
fn arb_latex_constant() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(r"\pi".to_string()),
        Just("e".to_string()),
        Just("i".to_string()),
        Just(r"\infty".to_string()),
    ]
}

/// Strategy to generate simple LaTeX expressions (leaf nodes).
fn arb_simple_latex() -> impl Strategy<Value = String> {
    prop_oneof![
        arb_latex_number(),
        arb_latex_variable(),
        arb_latex_constant(),
    ]
}

/// Strategy to generate LaTeX binary operator symbols.
fn arb_latex_binop() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("+"),
        Just("-"),
        Just(r"\cdot"),
        Just(r"\times"),
        Just("^"),
    ]
}

/// Strategy to generate LaTeX function names.
fn arb_latex_function() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just(r"\sin"),
        Just(r"\cos"),
        Just(r"\tan"),
        Just(r"\log"),
        Just(r"\ln"),
        Just(r"\exp"),
        Just(r"\sqrt"),
    ]
}

/// Strategy to generate complex LaTeX expressions with controlled recursion.
///
/// This generates expressions with:
/// - Binary operations (addition, multiplication, power)
/// - Fractions
/// - Functions (sin, cos, etc.)
/// - Square roots
/// - Parenthesized expressions
fn arb_complex_latex() -> impl Strategy<Value = String> {
    let leaf = arb_simple_latex();

    leaf.prop_recursive(
        3,  // max depth: 3 levels (shallower than AST tests to avoid parser complexity)
        32, // max total nodes: 32
        8,  // expected branch size: 8
        |inner| {
            prop_oneof![
                // Binary operations: a + b, a - b, a * b
                (inner.clone(), arb_latex_binop(), inner.clone()).prop_map(|(left, op, right)| {
                    if op == "^" {
                        // Power needs braces for multi-character exponents
                        format!("{}^{{{}}}", left, right)
                    } else {
                        format!("{} {} {}", left, op, right)
                    }
                }),
                // Fractions: \frac{a}{b}
                (inner.clone(), inner.clone()).prop_map(|(num, den)| {
                    format!(r"\frac{{{}}}{{{}}}", num, den)
                }),
                // Functions with single argument: \sin(x)
                (arb_latex_function(), inner.clone()).prop_map(|(func, arg)| {
                    if func == r"\sqrt" {
                        format!(r"\sqrt{{{}}}", arg)
                    } else {
                        format!(r"{}({})", func, arg)
                    }
                }),
                // Parenthesized expressions: (a)
                inner.clone().prop_map(|expr| format!("({})", expr)),
                // Negation: -a
                inner.clone().prop_map(|expr| format!("-{}", expr)),
            ]
        },
    )
}

proptest! {
    /// Property: Parsing simple numbers doesn't panic
    ///
    /// Any valid LaTeX number should parse without panicking.
    #[test]
    fn prop_parse_number_no_panic(latex in arb_latex_number()) {
        let _ = parse_latex(&latex);
    }

    /// Property: Parsing simple variables doesn't panic
    ///
    /// Any valid LaTeX variable should parse without panicking.
    #[test]
    fn prop_parse_variable_no_panic(latex in arb_latex_variable()) {
        let _ = parse_latex(&latex);
    }

    /// Property: Parsing constants doesn't panic
    ///
    /// Any valid LaTeX constant should parse without panicking.
    #[test]
    fn prop_parse_constant_no_panic(latex in arb_latex_constant()) {
        let _ = parse_latex(&latex);
    }

    /// Property: Parsing complex expressions doesn't panic
    ///
    /// Any generated LaTeX expression should parse without panicking.
    #[test]
    fn prop_parse_complex_no_panic(latex in arb_complex_latex()) {
        let _ = parse_latex(&latex);
    }

    /// Property: Parsing is deterministic
    ///
    /// Parsing the same LaTeX string twice should produce identical results.
    #[test]
    fn prop_parse_deterministic(latex in arb_complex_latex()) {
        let result1 = parse_latex(&latex);
        let result2 = parse_latex(&latex);

        match (result1, result2) {
            (Ok(expr1), Ok(expr2)) => prop_assert_eq!(expr1, expr2),
            (Err(_), Err(_)) => {
                // Both failed - this is consistent
            }
            _ => {
                // One succeeded, one failed - this is not deterministic
                return Err(TestCaseError::fail("Parsing is not deterministic"));
            }
        }
    }

    /// Property: Valid expressions round-trip through to_latex
    ///
    /// For successfully parsed expressions, converting back to LaTeX
    /// and parsing again should produce an equivalent expression.
    #[test]
    fn prop_roundtrip_latex(latex in arb_complex_latex()) {
        if let Ok(expr) = parse_latex(&latex) {
            let latex2 = expr.to_latex();
            match parse_latex(&latex2) {
                Ok(expr2) => {
                    // The expressions should be equal
                    // (though the LaTeX representation may differ)
                    prop_assert_eq!(expr, expr2);
                }
                Err(_) => {
                    // If to_latex produces invalid LaTeX, that's a bug
                    return Err(TestCaseError::fail(
                        format!("Round-trip failed: to_latex produced unparseable LaTeX: {}", latex2)
                    ));
                }
            }
        }
    }

    /// Property: to_latex produces non-empty strings
    ///
    /// Every expression should have a non-empty LaTeX representation.
    #[test]
    fn prop_to_latex_non_empty(latex in arb_complex_latex()) {
        if let Ok(expr) = parse_latex(&latex) {
            let latex_str = expr.to_latex();
            prop_assert!(!latex_str.is_empty());
        }
    }

    /// Property: Parsing preserves numeric values
    ///
    /// When parsing a number, the resulting AST should contain that number.
    #[test]
    fn prop_parse_number_preserves_value(n in -1000i64..1000) {
        let latex = n.to_string();
        if let Ok(expr) = parse_latex(&latex) {
            match expr {
                Expression::Integer(val) => prop_assert_eq!(val, n),
                Expression::Unary { op: UnaryOp::Neg, operand } => {
                    // Negative numbers might be parsed as negation of positive
                    if n < 0 {
                        if let Expression::Integer(val) = *operand {
                            prop_assert_eq!(val, -n);
                        }
                    }
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected Integer, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Parsing fractions produces division
    ///
    /// LaTeX fractions should parse to division operations.
    #[test]
    fn prop_parse_frac_is_division(
        num in arb_latex_number(),
        den in arb_latex_number()
    ) {
        let latex = format!(r"\frac{{{}}}{{{}}}", num, den);
        if let Ok(expr) = parse_latex(&latex) {
            match expr {
                Expression::Binary { op: BinaryOp::Div, .. } |
                Expression::Rational { .. } => {
                    // Both division and rational representation are valid
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected division or rational, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Parsing sqrt produces appropriate structure
    ///
    /// Square roots should parse to function calls or power operations.
    #[test]
    fn prop_parse_sqrt(arg in arb_latex_number()) {
        let latex = format!(r"\sqrt{{{}}}", arg);
        if let Ok(expr) = parse_latex(&latex) {
            match expr {
                Expression::Function { name, .. } if name == "sqrt" => {
                    // Function call representation
                }
                Expression::Binary { op: BinaryOp::Pow, .. } => {
                    // Power representation (x^0.5)
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected sqrt function or power, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Functions are parsed with correct names
    ///
    /// LaTeX function commands should preserve the function name.
    #[test]
    fn prop_parse_function_names(func in arb_latex_function(), arg in arb_latex_number()) {
        if func == r"\sqrt" {
            // Skip sqrt as it has special handling
            return Ok(());
        }

        let latex = format!(r"{}({})", func, arg);
        if let Ok(expr) = parse_latex(&latex) {
            match expr {
                Expression::Function { name, .. } => {
                    // Remove backslash from function name for comparison
                    let expected_name = func.trim_start_matches('\\');
                    prop_assert_eq!(name, expected_name);
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected function, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Parentheses don't change the parsed expression
    ///
    /// (x) should parse to the same thing as x for simple expressions.
    #[test]
    fn prop_parentheses_transparent(latex in arb_simple_latex()) {
        let with_parens = format!("({})", latex);

        let result1 = parse_latex(&latex);
        let result2 = parse_latex(&with_parens);

        match (result1, result2) {
            (Ok(expr1), Ok(expr2)) => {
                // Parentheses might be preserved in some cases, so we check structural equivalence
                // For simple expressions, they should be identical
                prop_assert_eq!(expr1, expr2);
            }
            _ => {
                // If either fails to parse, we can't verify the property
            }
        }
    }

    /// Property: Parsing Greek letters produces variables
    ///
    /// Greek letter commands should parse to variable nodes.
    #[test]
    fn prop_parse_greek_is_variable(greek in prop_oneof![
        Just(r"\alpha"),
        Just(r"\beta"),
        Just(r"\gamma"),
    ]) {
        if let Ok(expr) = parse_latex(greek) {
            match expr {
                Expression::Variable(_) => {
                    // Greek letters are variables
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected variable for Greek letter, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Constants parse to constant nodes
    ///
    /// Mathematical constants should parse to Constant enum variants.
    #[test]
    fn prop_parse_constants(constant in arb_latex_constant()) {
        if let Ok(expr) = parse_latex(&constant) {
            match expr {
                Expression::Constant(_) => {
                    // Successfully parsed as a constant
                }
                Expression::Variable(_) => {
                    // Some constants (like "e" and "i") might be parsed as variables
                    // depending on context, which is acceptable
                    if constant == "e" || constant == "i" {
                        // This is acceptable
                    } else {
                        return Err(TestCaseError::fail(
                            format!("Expected constant, got variable for: {}", constant)
                        ));
                    }
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected constant or variable, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Negation is parsed correctly
    ///
    /// Negative expressions should either be unary negation or negative literals.
    #[test]
    fn prop_parse_negation(latex in arb_latex_number()) {
        let neg_latex = format!("-{}", latex);
        if let Ok(expr) = parse_latex(&neg_latex) {
            match expr {
                Expression::Unary { op: UnaryOp::Neg, .. } => {
                    // Parsed as unary negation
                }
                Expression::Integer(n) if n < 0 => {
                    // Parsed as negative integer literal
                }
                Expression::Float(f) if f.value() < 0.0 => {
                    // Parsed as negative float literal
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected negation, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Addition is parsed with correct operator
    ///
    /// a + b should parse to a binary operation with Add operator.
    #[test]
    fn prop_parse_addition(a in arb_latex_number(), b in arb_latex_number()) {
        let latex = format!("{} + {}", a, b);
        if let Ok(expr) = parse_latex(&latex) {
            match expr {
                Expression::Binary { op: BinaryOp::Add, .. } => {
                    // Correctly parsed as addition
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected addition, got: {:?}", expr)
                    ));
                }
            }
        }
    }

    /// Property: Multiplication is parsed with correct operator
    ///
    /// a * b or a \cdot b should parse to multiplication.
    #[test]
    fn prop_parse_multiplication(a in arb_latex_number(), b in arb_latex_number()) {
        let latex1 = format!(r"{} \cdot {}", a, b);
        let latex2 = format!(r"{} \times {}", a, b);

        for latex in [latex1, latex2] {
            if let Ok(expr) = parse_latex(&latex) {
                match expr {
                    Expression::Binary { op: BinaryOp::Mul, .. } => {
                        // Correctly parsed as multiplication
                    }
                    _ => {
                        return Err(TestCaseError::fail(
                            format!("Expected multiplication, got: {:?}", expr)
                        ));
                    }
                }
            }
        }
    }

    /// Property: Power is parsed with correct operator
    ///
    /// a^b should parse to power operation (using positive base to avoid negation wrapping).
    #[test]
    fn prop_parse_power(a in 1i64..100, b in 1i64..10) {
        let latex = format!("{}^{}", a, b);
        if let Ok(expr) = parse_latex(&latex) {
            match &expr {
                Expression::Binary { op: BinaryOp::Pow, .. } => {
                    // Correctly parsed as power
                }
                Expression::Unary { op: UnaryOp::Neg, operand } => {
                    // If there's a negation, check if the operand is a power
                    match operand.as_ref() {
                        Expression::Binary { op: BinaryOp::Pow, .. } => {
                            // This is acceptable for negative bases
                        }
                        _ => {
                            return Err(TestCaseError::fail(
                                format!("Expected power (possibly negated), got: {:?}", expr)
                            ));
                        }
                    }
                }
                _ => {
                    return Err(TestCaseError::fail(
                        format!("Expected power, got: {:?}", expr)
                    ));
                }
            }
        }
    }
}

// Regular unit tests for specific LaTeX parsing cases

#[test]
fn test_parse_latex_pi() {
    let expr = parse_latex(r"\pi").unwrap();
    assert!(matches!(expr, Expression::Constant(MathConstant::Pi)));
}

#[test]
fn test_parse_latex_frac() {
    let expr = parse_latex(r"\frac{1}{2}").unwrap();
    match expr {
        Expression::Binary { op: BinaryOp::Div, .. } | Expression::Rational { .. } => {}
        _ => panic!("Expected division or rational"),
    }
}

#[test]
fn test_parse_latex_sin() {
    let expr = parse_latex(r"\sin(x)").unwrap();
    match expr {
        Expression::Function { name, .. } => {
            assert_eq!(name, "sin");
        }
        _ => panic!("Expected function"),
    }
}

#[test]
fn test_parse_latex_sqrt() {
    let expr = parse_latex(r"\sqrt{4}").unwrap();
    match expr {
        Expression::Function { name, .. } if name == "sqrt" => {}
        Expression::Binary { op: BinaryOp::Pow, .. } => {}
        _ => panic!("Expected sqrt function or power"),
    }
}

#[test]
fn test_parse_latex_power() {
    let expr = parse_latex(r"x^2").unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Pow,
            ..
        }
    ));
}

#[test]
fn test_parse_latex_greek() {
    let expr = parse_latex(r"\alpha").unwrap();
    match expr {
        Expression::Variable(name) => {
            assert!(name.contains("α") || name == "alpha");
        }
        _ => panic!("Expected variable"),
    }
}

#[test]
fn test_roundtrip_simple_add() {
    let original = parse_latex(r"1 + 2").unwrap();
    let latex = original.to_latex();
    let reparsed = parse_latex(&latex).unwrap();
    assert_eq!(original, reparsed);
}

#[test]
fn test_roundtrip_frac() {
    let original = parse_latex(r"\frac{3}{4}").unwrap();
    let latex = original.to_latex();
    let reparsed = parse_latex(&latex).unwrap();
    assert_eq!(original, reparsed);
}

#[test]
fn test_roundtrip_power() {
    let original = parse_latex(r"x^2").unwrap();
    let latex = original.to_latex();
    let reparsed = parse_latex(&latex).unwrap();
    assert_eq!(original, reparsed);
}

#[test]
fn test_to_latex_non_empty_constant() {
    let expr = Expression::Constant(MathConstant::E);
    let latex = expr.to_latex();
    assert!(!latex.is_empty());
}

#[test]
fn test_to_latex_non_empty_integer() {
    let expr = Expression::Integer(42);
    let latex = expr.to_latex();
    assert!(!latex.is_empty());
    assert_eq!(latex, "42");
}
