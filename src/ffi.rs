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

        #[swift_bridge(swift_name = "findConstants")]
        fn expression_find_constants(expr: &Expression) -> Vec<String>;

        #[swift_bridge(swift_name = "depth")]
        fn expression_depth(expr: &Expression) -> usize;

        #[swift_bridge(swift_name = "nodeCount")]
        fn expression_node_count(expr: &Expression) -> usize;

        #[swift_bridge(swift_name = "toJSON")]
        fn expression_to_json(expr: &Expression) -> Result<String, String>;

        #[swift_bridge(swift_name = "toJSONPretty")]
        fn expression_to_json_pretty(expr: &Expression) -> Result<String, String>;
    }
}

#[cfg(feature = "ffi")]
use crate::{parse, parser::parse_latex as parse_latex_internal, Expression};

#[cfg(feature = "ffi")]
use crate::parser::{
    latex::parse_latex_equation_system, text::parse_equation_system as parse_text_equation_system,
};

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

/// FFI wrapper for finding constants.
///
/// Returns constant names as strings for FFI compatibility.
#[cfg(feature = "ffi")]
pub fn expression_find_constants(expr: &Expression) -> Vec<String> {
    expr.find_constants()
        .into_iter()
        .map(|c| format!("{}", c))
        .collect()
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

/// FFI wrapper for serializing an expression to compact JSON.
///
/// Returns the JSON string on success, or an error string if serialization fails
/// or the `serde` feature is not enabled.
#[cfg(feature = "ffi")]
pub fn expression_to_json(expr: &Expression) -> Result<String, String> {
    #[cfg(feature = "serde")]
    {
        serde_json::to_string(expr).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "serde"))]
    {
        let _ = expr;
        Err("serde feature is not enabled".to_string())
    }
}

/// FFI wrapper for serializing an expression to pretty-printed JSON.
///
/// Returns the indented JSON string on success, or an error string if
/// serialization fails or the `serde` feature is not enabled.
#[cfg(feature = "ffi")]
pub fn expression_to_json_pretty(expr: &Expression) -> Result<String, String> {
    #[cfg(feature = "serde")]
    {
        serde_json::to_string_pretty(expr).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "serde"))]
    {
        let _ = expr;
        Err("serde feature is not enabled".to_string())
    }
}

/// FFI wrapper for parsing semicolon-delimited equation systems (plain text).
///
/// Returns a JSON-serialized array of expressions for FFI compatibility,
/// since swift-bridge does not support `Vec<OpaqueRustType>` directly.
#[cfg(feature = "ffi")]
pub fn parse_equation_system_ffi(input: &str) -> Result<Vec<Expression>, String> {
    parse_text_equation_system(input).map_err(|e| e.to_string())
}

/// FFI wrapper for parsing semicolon-delimited equation systems (LaTeX).
#[cfg(feature = "ffi")]
pub fn parse_latex_equation_system_ffi(input: &str) -> Result<Vec<Expression>, String> {
    parse_latex_equation_system(input).map_err(|e| e.to_string())
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

#[cfg(all(test, feature = "ffi", feature = "serde"))]
mod json_tests {
    use super::*;

    #[test]
    fn test_json_integer() {
        let expr = parse_text("42").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert_eq!(json, r#"{"Integer":42}"#);
    }

    #[test]
    fn test_json_float() {
        let expr = parse_text("3.14").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("Float"),
            "expected Float variant, got: {json}"
        );
    }

    #[test]
    fn test_json_variable() {
        let expr = parse_text("x").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert_eq!(json, r#"{"Variable":"x"}"#);
    }

    #[test]
    fn test_json_constant_pi() {
        let expr = parse_text("pi").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("Constant"),
            "expected Constant variant, got: {json}"
        );
        assert!(json.contains("Pi"), "expected Pi, got: {json}");
    }

    #[test]
    fn test_json_binary_add() {
        let expr = parse_text("x + y").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("Binary"),
            "expected Binary variant, got: {json}"
        );
        assert!(json.contains("Add"), "expected Add op, got: {json}");
    }

    #[test]
    fn test_json_unary_neg() {
        let expr = parse_text("-x").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("Neg") || json.contains("Unary"),
            "expected negation, got: {json}"
        );
    }

    #[test]
    fn test_json_function_call() {
        let expr = parse_text("sin(x)").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("Function"),
            "expected Function variant, got: {json}"
        );
        assert!(json.contains("sin"), "expected sin, got: {json}");
    }

    #[test]
    fn test_json_nested_expression() {
        let expr = parse_text("sin(x)^2 + cos(x)^2").unwrap();
        let json = expression_to_json(&expr).unwrap();
        assert!(
            json.contains("sin"),
            "expected sin in nested expr, got: {json}"
        );
        assert!(
            json.contains("cos"),
            "expected cos in nested expr, got: {json}"
        );
    }

    #[test]
    fn test_json_pretty_is_multiline() {
        let expr = parse_text("x + y").unwrap();
        let pretty = expression_to_json_pretty(&expr).unwrap();
        assert!(pretty.contains('\n'), "pretty JSON should be multi-line");
    }

    #[test]
    fn test_json_pretty_contains_same_data() {
        let expr = parse_text("x + y").unwrap();
        let compact = expression_to_json(&expr).unwrap();
        let pretty = expression_to_json_pretty(&expr).unwrap();
        // Both should parse to the same structure by comparing deserialized values
        let compact_val: serde_json::Value = serde_json::from_str(&compact).unwrap();
        let pretty_val: serde_json::Value = serde_json::from_str(&pretty).unwrap();
        assert_eq!(compact_val, pretty_val);
    }

    #[test]
    fn test_json_round_trip() {
        use crate::Expression;
        let expr = parse_text("2 * x + 3").unwrap();
        let json = expression_to_json(&expr).unwrap();
        let restored: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, restored);
    }

    #[test]
    fn test_json_round_trip_function() {
        use crate::Expression;
        let expr = parse_text("sin(x)").unwrap();
        let json = expression_to_json(&expr).unwrap();
        let restored: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, restored);
    }

    #[test]
    fn test_json_round_trip_nested() {
        use crate::Expression;
        let expr = parse_text("sin(x)^2 + cos(x)^2").unwrap();
        let json = expression_to_json(&expr).unwrap();
        let restored: Expression = serde_json::from_str(&json).unwrap();
        assert_eq!(expr, restored);
    }
}
