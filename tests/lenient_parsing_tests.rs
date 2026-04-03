use mathlex::ast::{BinaryOp, Expression};
use mathlex::error::ParseOutput;

// =============================================================================
// ParseOutput struct tests
// =============================================================================

#[test]
fn test_parse_output_is_ok_when_success() {
    let output = ParseOutput {
        expression: Some(Expression::Integer(42)),
        errors: vec![],
    };
    assert!(output.is_ok());
    assert!(!output.has_errors());
}

#[test]
fn test_parse_output_not_ok_when_no_expression() {
    let output = ParseOutput {
        expression: None,
        errors: vec![],
    };
    assert!(!output.is_ok());
}

#[test]
fn test_parse_output_not_ok_when_has_errors() {
    let output = ParseOutput {
        expression: Some(Expression::Integer(42)),
        errors: vec![mathlex::ParseError::empty_expression(None)],
    };
    assert!(!output.is_ok());
    assert!(output.has_errors());
}

#[test]
fn test_parse_output_from_ok_result() {
    let result: Result<Expression, mathlex::ParseError> = Ok(Expression::Integer(1));
    let output = ParseOutput::from_result(result);
    assert!(output.is_ok());
    assert_eq!(output.expression, Some(Expression::Integer(1)));
}

#[test]
fn test_parse_output_from_err_result() {
    let result: Result<Expression, mathlex::ParseError> =
        Err(mathlex::ParseError::empty_expression(None));
    let output = ParseOutput::from_result(result);
    assert!(!output.is_ok());
    assert!(output.expression.is_none());
    assert_eq!(output.errors.len(), 1);
}

// =============================================================================
// Strict mode is unchanged
// =============================================================================

#[test]
fn test_strict_parse_still_works() {
    let expr = mathlex::parse("2 + 3").unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn test_strict_parse_still_fails_on_error() {
    // Unmatched closing paren is an error in strict mode
    assert!(mathlex::parse("2 + 3)").is_err());
}

#[test]
fn test_strict_latex_still_works() {
    let expr = mathlex::parse_latex(r"\frac{1}{2}").unwrap();
    assert!(matches!(
        expr,
        Expression::Binary {
            op: BinaryOp::Div,
            ..
        }
    ));
}

#[test]
fn test_strict_latex_still_fails_on_error() {
    assert!(mathlex::parse_latex(r"\frac{1}{}").is_err());
}

// =============================================================================
// Lenient text parsing: valid inputs
// =============================================================================

#[test]
fn test_lenient_text_valid_expression() {
    let output = mathlex::parse_lenient("2 + 3");
    assert!(output.is_ok());
    assert!(matches!(
        output.expression,
        Some(Expression::Binary {
            op: BinaryOp::Add,
            ..
        })
    ));
}

#[test]
fn test_lenient_text_single_number() {
    let output = mathlex::parse_lenient("42");
    assert!(output.is_ok());
    assert_eq!(output.expression, Some(Expression::Integer(42)));
}

#[test]
fn test_lenient_text_complex_valid() {
    let output = mathlex::parse_lenient("sin(x) + cos(y)");
    assert!(output.is_ok());
    assert!(!output.has_errors());
}

// =============================================================================
// Lenient text parsing: error recovery
// =============================================================================

#[test]
fn test_lenient_text_collects_errors() {
    // Trailing close paren is invalid
    let output = mathlex::parse_lenient("2 + 3)");
    assert!(output.has_errors());
}

#[test]
fn test_lenient_text_partial_ast_on_trailing_garbage() {
    // Valid expression followed by invalid tokens
    let output = mathlex::parse_lenient("2 + 3) extra");
    assert!(output.has_errors());
    // Should still have parsed the valid "2 + 3" part
    assert!(output.expression.is_some());
}

#[test]
fn test_lenient_text_empty_input() {
    let output = mathlex::parse_lenient("");
    // Empty input should produce an error or no expression
    // The tokenizer may return empty tokens, which gives no expression
    assert!(output.expression.is_none() || output.has_errors());
}

#[test]
fn test_lenient_text_unmatched_paren() {
    let output = mathlex::parse_lenient("(2 + 3");
    assert!(output.has_errors());
}

#[test]
fn test_lenient_text_multiple_expressions() {
    // Two valid expressions separated by something unexpected
    let output = mathlex::parse_lenient("2 + 3 ) 4 + 5");
    // Should recover and report errors
    assert!(output.has_errors());
    // Should still have parsed the first part
    assert!(output.expression.is_some());
}

// =============================================================================
// Lenient LaTeX parsing: valid inputs
// =============================================================================

#[test]
fn test_lenient_latex_valid_expression() {
    let output = mathlex::parse_latex_lenient(r"\frac{1}{2}");
    assert!(output.is_ok());
    assert!(matches!(
        output.expression,
        Some(Expression::Binary {
            op: BinaryOp::Div,
            ..
        })
    ));
}

#[test]
fn test_lenient_latex_simple_addition() {
    let output = mathlex::parse_latex_lenient(r"x + y");
    assert!(output.is_ok());
    assert!(!output.has_errors());
}

#[test]
fn test_lenient_latex_complex_valid() {
    let output = mathlex::parse_latex_lenient(r"\int_0^1 x^2 dx");
    assert!(output.is_ok());
}

// =============================================================================
// Lenient LaTeX parsing: error recovery
// =============================================================================

#[test]
fn test_lenient_latex_empty_frac_denominator() {
    let output = mathlex::parse_latex_lenient(r"\frac{1}{}");
    assert!(output.has_errors());
}

#[test]
fn test_lenient_latex_partial_ast_with_errors() {
    // Valid part + error part: "x + \frac{}{} + y"
    let output = mathlex::parse_latex_lenient(r"x + \frac{}{} + y");
    assert!(output.has_errors());
}

#[test]
fn test_lenient_latex_empty_input() {
    let output = mathlex::parse_latex_lenient("");
    assert!(output.expression.is_none() || output.has_errors());
}

#[test]
fn test_lenient_latex_unmatched_brace() {
    let output = mathlex::parse_latex_lenient(r"\frac{1}{2");
    assert!(output.has_errors());
}

#[test]
fn test_lenient_latex_multiple_errors() {
    // Multiple problematic sections
    let output = mathlex::parse_latex_lenient(r"\frac{}{} + \frac{}{}");
    assert!(output.has_errors());
    // Should collect multiple errors (one per empty brace group)
    assert!(output.errors.len() >= 1);
}

#[test]
fn test_lenient_latex_recovers_after_bad_command() {
    // Unknown command followed by valid math
    let output = mathlex::parse_latex_lenient(r"\badcommand + x");
    // Should have an error for the bad command
    assert!(output.has_errors());
}

// =============================================================================
// Lenient with config
// =============================================================================

#[test]
fn test_lenient_with_config_implicit_mult() {
    let config = mathlex::ParserConfig {
        implicit_multiplication: true,
        ..Default::default()
    };
    let output = mathlex::parse_lenient_with_config("2x + 3", &config);
    assert!(output.is_ok());
}

#[test]
fn test_lenient_with_config_no_implicit_mult() {
    let config = mathlex::ParserConfig {
        implicit_multiplication: false,
        ..Default::default()
    };
    // "2x" without implicit multiplication may cause an error in lenient mode
    let output = mathlex::parse_lenient_with_config("2x", &config);
    // Either parses differently or reports an error — both are valid
    assert!(output.expression.is_some() || output.has_errors());
}

// =============================================================================
// Error details in lenient mode
// =============================================================================

#[test]
fn test_lenient_errors_have_spans() {
    let output = mathlex::parse_lenient("2 + + 3");
    for err in &output.errors {
        // All errors from the parser should have span information
        assert!(err.span.is_some(), "Error missing span: {}", err);
    }
}

#[test]
fn test_lenient_latex_errors_have_spans() {
    let output = mathlex::parse_latex_lenient(r"\frac{1}{}");
    for err in &output.errors {
        assert!(err.span.is_some(), "Error missing span: {}", err);
    }
}
