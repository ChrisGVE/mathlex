//! FFI bridge for Swift bindings using swift-bridge.
//!
//! This module provides a C-compatible FFI layer that allows Swift code
//! to parse mathematical expressions and interact with the AST.
//!
//! The bridge is only compiled when the `ffi` feature is enabled.

#[cfg(feature = "ffi")]
#[allow(clippy::unnecessary_cast)]
#[swift_bridge::bridge]
mod ffi {
    extern "Rust" {
        type Expression;

        #[swift_bridge(swift_name = "parseText")]
        fn parse_text(input: &str) -> Result<Expression, String>;

        #[swift_bridge(swift_name = "parseLatex")]
        fn parse_latex_ffi(input: &str) -> Result<Expression, String>;

        #[swift_bridge(swift_name = "toString")]
        fn expression_to_string(expr: &Expression) -> String;

        #[swift_bridge(swift_name = "toLatex")]
        fn expression_to_latex(expr: &Expression) -> String;

        #[swift_bridge(swift_name = "findVariables")]
        fn expression_find_variables(expr: &Expression) -> Vec<String>;

        #[swift_bridge(swift_name = "findFunctions")]
        fn expression_find_functions(expr: &Expression) -> Vec<String>;

        #[swift_bridge(swift_name = "depth")]
        fn expression_depth(expr: &Expression) -> usize;

        #[swift_bridge(swift_name = "nodeCount")]
        fn expression_node_count(expr: &Expression) -> usize;
    }
}

#[cfg(feature = "ffi")]
use crate::{parse, parser::parse_latex as parse_latex_internal, Expression};

/// FFI wrapper for parsing plain text expressions.
///
/// Converts ParseError into a String for FFI compatibility.
#[cfg(feature = "ffi")]
pub fn parse_text(input: &str) -> Result<Expression, String> {
    parse(input).map_err(|e| e.to_string())
}

/// FFI wrapper for parsing LaTeX expressions.
///
/// Converts ParseError into a String for FFI compatibility.
#[cfg(feature = "ffi")]
pub fn parse_latex_ffi(input: &str) -> Result<Expression, String> {
    parse_latex_internal(input).map_err(|e| e.to_string())
}

/// FFI wrapper for converting expression to string.
///
/// Uses the Display implementation internally.
#[cfg(feature = "ffi")]
pub fn expression_to_string(expr: &Expression) -> String {
    format!("{}", expr)
}

/// FFI wrapper for converting expression to LaTeX.
///
/// Uses the ToLatex trait implementation internally.
#[cfg(feature = "ffi")]
pub fn expression_to_latex(expr: &Expression) -> String {
    use crate::latex::ToLatex;
    expr.to_latex()
}

/// FFI wrapper for finding variables.
///
/// Converts HashSet to Vec for FFI compatibility.
#[cfg(feature = "ffi")]
pub fn expression_find_variables(expr: &Expression) -> Vec<String> {
    expr.find_variables().into_iter().collect()
}

/// FFI wrapper for finding functions.
///
/// Converts HashSet to Vec for FFI compatibility.
#[cfg(feature = "ffi")]
pub fn expression_find_functions(expr: &Expression) -> Vec<String> {
    expr.find_functions().into_iter().collect()
}

/// FFI wrapper for calculating depth.
#[cfg(feature = "ffi")]
pub fn expression_depth(expr: &Expression) -> usize {
    expr.depth()
}

/// FFI wrapper for counting nodes.
#[cfg(feature = "ffi")]
pub fn expression_node_count(expr: &Expression) -> usize {
    expr.node_count()
}

#[cfg(all(test, feature = "ffi"))]
mod tests {
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
}
