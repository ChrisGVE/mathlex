#![cfg(feature = "ffi")]

use super::*;

#[test]
fn test_parse_text_success() {
    let result = parse_text("2 + 3");
    assert!(result.is_ok());
}

#[test]
fn test_parse_text_error() {
    let result = parse_text("2 +");
    assert!(result.is_err());
}

#[test]
fn test_parse_latex_success() {
    let result = parse_latex_ffi(r"\frac{1}{2}");
    assert!(result.is_ok());
}

#[test]
fn test_parse_latex_error() {
    let result = parse_latex_ffi(r"\frac{1}");
    assert!(result.is_err());
}

#[test]
fn test_expression_to_string() {
    let expr = parse_text("2 + 3").unwrap();
    let s = expression_to_string(&expr);
    assert!(s.contains("2") && s.contains("3"));
}

#[test]
fn test_expression_to_latex() {
    let expr = parse_text("1/2").unwrap();
    let latex = expression_to_latex(&expr);
    assert!(latex.contains("frac"));
}

#[test]
fn test_expression_find_variables() {
    let expr = parse_text("x + y").unwrap();
    let vars = expression_find_variables(&expr);
    assert_eq!(vars.len(), 2);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));
}

#[test]
fn test_expression_find_functions() {
    let expr = parse_text("sin(x) + cos(y)").unwrap();
    let funcs = expression_find_functions(&expr);
    assert_eq!(funcs.len(), 2);
    assert!(funcs.contains(&"sin".to_string()));
    assert!(funcs.contains(&"cos".to_string()));
}

#[test]
fn test_expression_depth() {
    let expr = parse_text("2 + 3").unwrap();
    assert!(expression_depth(&expr) > 0);
}

#[test]
fn test_expression_node_count() {
    let expr = parse_text("2 + 3").unwrap();
    assert!(expression_node_count(&expr) >= 3);
}
