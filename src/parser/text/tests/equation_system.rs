//! Tests for `parse_equation_system` and `parse_equation_system_with_config`.

use super::*;

#[test]
fn test_two_equations() {
    let exprs = parse_equation_system("x + y = 5; 2*x - y = 1").unwrap();
    assert_eq!(exprs.len(), 2);
    assert!(matches!(exprs[0], Expression::Equation { .. }));
    assert!(matches!(exprs[1], Expression::Equation { .. }));
}

#[test]
fn test_empty_segments_filtered() {
    // Double semicolon produces an empty segment that should be ignored.
    let exprs = parse_equation_system("x = 1;; y = 2").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_single_expression_no_delimiter() {
    let exprs = parse_equation_system("x = 1").unwrap();
    assert_eq!(exprs.len(), 1);
}

#[test]
fn test_whitespace_trimmed() {
    let exprs = parse_equation_system("  x = 1  ;  y = 2  ").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_non_equation_expressions() {
    // Segments don't have to be equations; any valid expression is accepted.
    let exprs = parse_equation_system("x + 1; y + 2").unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_invalid_segment_returns_error() {
    let result = parse_equation_system("x + 1; invalid @@@");
    assert!(result.is_err());
}

#[test]
fn test_trailing_semicolon_ignored() {
    let exprs = parse_equation_system("x = 1;").unwrap();
    assert_eq!(exprs.len(), 1);
}

#[test]
fn test_with_config() {
    let config = ParserConfig::default();
    let exprs = parse_equation_system_with_config("x = 1; y = 2", &config).unwrap();
    assert_eq!(exprs.len(), 2);
}

#[test]
fn test_empty_input() {
    let exprs = parse_equation_system("").unwrap();
    assert!(exprs.is_empty());
}

#[test]
fn test_only_semicolons() {
    let exprs = parse_equation_system(";;;").unwrap();
    assert!(exprs.is_empty());
}
