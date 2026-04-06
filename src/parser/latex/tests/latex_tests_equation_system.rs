//! Tests for `parse_latex_equation_system`.

use super::*;

#[test]
fn test_two_latex_equations() {
    let exprs = parse_latex_equation_system(r"x = 1; \frac{y}{2} = 3").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_single_latex_expression() {
    let exprs = parse_latex_equation_system(r"x + 1").unwrap();
    assert_eq!(exprs.len(), 1);
}

#[test]
fn test_empty_segments_filtered() {
    let exprs = parse_latex_equation_system(r"x = 1;; y = 2").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_whitespace_trimmed() {
    let exprs = parse_latex_equation_system(r"  x = 1  ;  y = 2  ").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_non_equation_latex_expressions() {
    let exprs = parse_latex_equation_system(r"x + 1; y + 2").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_invalid_segment_returns_error() {
    // An unterminated brace is a LaTeX parse error.
    let result = parse_latex_equation_system(r"x = 1; \frac{}{");
    assert!(result.is_err());
}

#[test]
fn test_trailing_semicolon_ignored() {
    let exprs = parse_latex_equation_system(r"x = 1;").unwrap();
    assert_eq!(exprs.len(), 1);
}

#[test]
fn test_empty_input() {
    let exprs = parse_latex_equation_system("").unwrap();
    assert!(exprs.is_empty());
}

#[test]
fn test_only_semicolons() {
    let exprs = parse_latex_equation_system(";;;").unwrap();
    assert!(exprs.is_empty());
}
