// Error case tests for LaTeX parser
use super::*;

// Fraction errors

#[test]
fn test_frac_missing_numerator() {
    let result = parse_latex(r"\frac{}");
    assert!(result.is_err());
}

#[test]
fn test_frac_missing_denominator() {
    let result = parse_latex(r"\frac{1}");
    assert!(result.is_err());
}

#[test]
fn test_frac_missing_both_braces() {
    let result = parse_latex(r"\frac");
    assert!(result.is_err());
}

#[test]
fn test_frac_unclosed_numerator() {
    let result = parse_latex(r"\frac{1{2}");
    assert!(result.is_err());
}

#[test]
fn test_frac_unclosed_denominator() {
    let result = parse_latex(r"\frac{1}{2");
    assert!(result.is_err());
}

// Root errors

#[test]
fn test_sqrt_missing_argument() {
    let result = parse_latex(r"\sqrt");
    assert!(result.is_err());
}

#[test]
fn test_sqrt_unclosed_brace() {
    let result = parse_latex(r"\sqrt{x");
    assert!(result.is_err());
}

#[test]
fn test_sqrt_empty() {
    let result = parse_latex(r"\sqrt{}");
    assert!(result.is_err());
}

#[test]
fn test_root_missing_index() {
    let result = parse_latex(r"\sqrt[]{x}");
    assert!(result.is_err());
}

#[test]
fn test_root_missing_radicand() {
    let result = parse_latex(r"\sqrt[3]");
    assert!(result.is_err());
}

#[test]
fn test_root_unclosed_index() {
    let result = parse_latex(r"\sqrt[3{x}");
    assert!(result.is_err());
}

// Brace mismatch errors

#[test]
fn test_unclosed_brace() {
    let result = parse_latex("{x");
    assert!(result.is_err());
}

#[test]
fn test_unopened_brace() {
    let result = parse_latex("x}");
    assert!(result.is_err());
}

#[test]
fn test_mismatched_braces() {
    let result = parse_latex("{x}}");
    assert!(result.is_err());
}

#[test]
fn test_unclosed_parenthesis() {
    let result = parse_latex("(x+1");
    assert!(result.is_err());
}

#[test]
fn test_unopened_parenthesis() {
    let result = parse_latex("x+1)");
    assert!(result.is_err());
}

// Invalid command errors

#[test]
fn test_invalid_command() {
    let result = parse_latex(r"\notacommand");
    assert!(result.is_err());
}

#[test]
fn test_backslash_only() {
    let result = parse_latex(r"\");
    assert!(result.is_err());
}

// Relation errors

#[test]
fn test_chained_relations_less_less() {
    let result = parse_latex("a < b < c");
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("chained relations"));
    }
}

#[test]
fn test_chained_relations_equals_equals() {
    let result = parse_latex("a = b = c");
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("chained relations"));
    }
}

#[test]
fn test_chained_relations_mixed() {
    let result = parse_latex("a < b = c");
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("chained relations"));
    }
}

#[test]
fn test_chained_relations_leq() {
    let result = parse_latex(r"a \leq b \leq c");
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("chained relations"));
    }
}

// Matrix errors

#[test]
fn test_matrix_ragged() {
    let result = parse_latex(r"\begin{matrix}1 & 2 \\ 3\end{matrix}");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("inconsistent matrix row lengths"));
    }
}

#[test]
fn test_matrix_mismatched_environment() {
    let result = parse_latex(r"\begin{matrix}1\end{bmatrix}");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("mismatched environment"));
    }
}

#[test]
fn test_matrix_invalid_environment() {
    let result = parse_latex(r"\begin{invalid}1\end{invalid}");
    assert!(result.is_err());
}

#[test]
fn test_matrix_unclosed() {
    let result = parse_latex(r"\begin{matrix}1 & 2");
    assert!(result.is_err());
}

#[test]
fn test_matrix_unopened() {
    let result = parse_latex(r"1 & 2\end{matrix}");
    assert!(result.is_err());
}

#[test]
fn test_begin_without_brace() {
    let result = parse_latex(r"\begin");
    assert!(result.is_err());
}

#[test]
fn test_begin_unclosed_brace() {
    let result = parse_latex(r"\begin{matrix");
    assert!(result.is_err());
}

// Subscript errors

#[test]
fn test_subscript_on_number() {
    // Subscript can only be applied to variables
    let result = parse_latex("5_1");
    assert!(result.is_err());
}

#[test]
fn test_subscript_on_expression() {
    // Subscript can only be applied to variables
    let result = parse_latex("(x+1)_2");
    assert!(result.is_err());
}

// Derivative errors

#[test]
fn test_derivative_mismatched_types() {
    // d in numerator but \partial in denominator
    // The parser tries to match derivative pattern but fails, so it errors
    let result = parse_latex(r"\frac{d}{\partial * x}f");
    assert!(result.is_err());
}

#[test]
fn test_derivative_mismatched_orders() {
    // d^2 in numerator but dx in denominator (not dx^2)
    // The parser tries to match derivative pattern but fails, so it errors
    let result = parse_latex(r"\frac{d^2}{d*x}f");
    assert!(result.is_err());
}

// Integral errors

#[test]
fn test_integral_missing_variable() {
    let result = parse_latex(r"\int x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("expected 'd'"));
    }
}

#[test]
fn test_integral_missing_d() {
    let result = parse_latex(r"\int x x");
    assert!(result.is_err());
}

#[test]
fn test_integral_upper_bound_without_lower() {
    let result = parse_latex(r"\int^1 x dx");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("upper bound must also have lower bound"));
    }
}

#[test]
fn test_integral_lower_bound_without_upper() {
    let result = parse_latex(r"\int_0 x dx");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("lower bound must also have upper bound"));
    }
}

// Limit errors

#[test]
fn test_limit_missing_subscript() {
    let result = parse_latex(r"\lim x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("limit must have subscript"));
    }
}

#[test]
fn test_limit_missing_to() {
    let result = parse_latex(r"\lim_{x} x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("expected \\to"));
    }
}

#[test]
fn test_limit_invalid_direction() {
    let result = parse_latex(r"\lim_{x \to 0^*} x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("expected + or -"));
    }
}

// Sum/Product errors

#[test]
fn test_sum_missing_subscript() {
    let result = parse_latex(r"\sum x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("iterator must have subscript"));
    }
}

#[test]
fn test_sum_missing_equals() {
    let result = parse_latex(r"\sum_{i 1}^{n} i");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("expected ="));
    }
}

#[test]
fn test_sum_missing_superscript() {
    let result = parse_latex(r"\sum_{i=1} i");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("iterator must have superscript"));
    }
}

#[test]
fn test_prod_missing_subscript() {
    let result = parse_latex(r"\prod x");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("iterator must have subscript"));
    }
}

#[test]
fn test_prod_missing_equals() {
    let result = parse_latex(r"\prod_{i 1}^{n} i");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("expected ="));
    }
}

#[test]
fn test_prod_missing_superscript() {
    let result = parse_latex(r"\prod_{i=1} i");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("iterator must have superscript"));
    }
}

// Unexpected end of input

#[test]
fn test_unexpected_eof_after_operator() {
    let result = parse_latex("x +");
    assert!(result.is_err());
}

#[test]
fn test_unexpected_eof_in_power() {
    let result = parse_latex("x^");
    assert!(result.is_err());
}

#[test]
fn test_unexpected_eof_in_subscript() {
    let result = parse_latex("x_");
    assert!(result.is_err());
}

// Absolute value errors

#[test]
fn test_absolute_value_unclosed() {
    let result = parse_latex("|x");
    assert!(result.is_err());
}

#[test]
fn test_absolute_value_unopened() {
    let result = parse_latex("x|");
    assert!(result.is_err());
}

// Empty expressions

#[test]
fn test_empty_parentheses() {
    let result = parse_latex("()");
    assert!(result.is_err());
}

#[test]
fn test_empty_braces() {
    let result = parse_latex("{}");
    assert!(result.is_err());
}

// Trailing operators

#[test]
fn test_trailing_plus() {
    let result = parse_latex("x + y +");
    assert!(result.is_err());
}

#[test]
fn test_trailing_times() {
    let result = parse_latex("x * y *");
    assert!(result.is_err());
}

#[test]
fn test_trailing_equals() {
    let result = parse_latex("x =");
    assert!(result.is_err());
}

// Invalid number formats

#[test]
fn test_multiple_decimal_points() {
    // The tokenizer should handle this - verify behavior
    let result = parse_latex("3.14.15");
    // This might tokenize as 3.14 followed by .15 (error) or as an error
    // Either way, it should not succeed as a valid expression
    assert!(result.is_err() || {
        // If it parses, check what we got
        if let Ok(expr) = result {
            // Should not be a single float with value 3.14.15
            !matches!(expr, Expression::Float(_))
        } else {
            true
        }
    });
}
