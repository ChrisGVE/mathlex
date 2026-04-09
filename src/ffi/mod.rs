//! FFI bridge for Swift bindings using swift-bridge.
//!
//! This module provides a C-compatible FFI layer that allows Swift code
//! to parse mathematical expressions and interact with the AST.
//!
//! The bridge is only compiled when the `ffi` feature is enabled.
//!
//! ## JSON serialization
//!
//! When both the `ffi` and `serde` features are enabled, [`expression_to_json`]
//! and [`expression_to_json_pretty`] serialize any `Expression` to a JSON string
//! following the schema documented in `docs/json-ast-schema.md`. Swift consumers
//! can decode this JSON into `Decodable` types for evaluation or further
//! processing without needing direct access to the opaque Rust type.

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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_json;
