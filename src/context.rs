//! Context engine for multi-expression parsing.
//!
//! Provides tracking of semantic context across multiple related
//! mathematical expressions.

use crate::ast::Expression;
use crate::error::ParseResult;
use crate::metadata::MathType;
use crate::{NumberSystem, ParserConfig};
use std::collections::{HashMap, HashSet};

/// Context for parsing multiple related expressions.
///
/// Tracks type information, declared symbols, and semantic context
/// that spans across multiple expressions in a system.
#[derive(Debug, Clone, Default)]
pub struct ExpressionContext {
    /// Known variable types
    variable_types: HashMap<String, MathType>,
    /// Number system in use
    number_system: NumberSystem,
    /// Declared vector symbols
    vectors: HashSet<String>,
    /// Declared matrix symbols
    matrices: HashSet<String>,
    /// Declared scalar symbols
    scalars: HashSet<String>,
    /// Active variable bindings from quantifiers
    bindings: Vec<(String, Option<MathType>)>,
}

impl ExpressionContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create context with a specific number system.
    pub fn with_number_system(number_system: NumberSystem) -> Self {
        Self {
            number_system,
            ..Default::default()
        }
    }

    /// Declare a variable with a specific type.
    pub fn declare_variable(&mut self, name: impl Into<String>, math_type: MathType) {
        self.variable_types.insert(name.into(), math_type);
    }

    /// Declare a symbol as a vector.
    pub fn declare_vector(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.vectors.insert(name.clone());
        self.variable_types.insert(name, MathType::Vector(None));
    }

    /// Declare a symbol as a matrix.
    pub fn declare_matrix(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.matrices.insert(name.clone());
        self.variable_types
            .insert(name, MathType::Matrix(None, None));
    }

    /// Declare a symbol as a scalar.
    pub fn declare_scalar(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.scalars.insert(name.clone());
        self.variable_types.insert(name, MathType::Scalar);
    }

    /// Check if a symbol is declared as a vector.
    pub fn is_vector(&self, name: &str) -> bool {
        self.vectors.contains(name)
    }

    /// Check if a symbol is declared as a matrix.
    pub fn is_matrix(&self, name: &str) -> bool {
        self.matrices.contains(name)
    }

    /// Get the inferred type for a symbol.
    pub fn get_type(&self, name: &str) -> Option<&MathType> {
        self.variable_types.get(name)
    }

    /// Get the number system in use.
    pub fn number_system(&self) -> NumberSystem {
        self.number_system
    }

    /// Push a binding scope (for quantifiers).
    pub fn push_binding(&mut self, variable: String, math_type: Option<MathType>) {
        self.bindings.push((variable, math_type));
    }

    /// Pop the most recent binding scope.
    pub fn pop_binding(&mut self) -> Option<(String, Option<MathType>)> {
        self.bindings.pop()
    }

    /// Check if a variable is currently bound.
    pub fn is_bound(&self, name: &str) -> bool {
        self.bindings.iter().any(|(n, _)| n == name)
    }

    /// Update context by analyzing an expression.
    /// Extracts type information from structural patterns.
    pub fn analyze_expression(&mut self, expr: &Expression) {
        // Analyze expression to extract type hints
        self.infer_types_from_expression(expr);
    }

    fn infer_types_from_expression(&mut self, expr: &Expression) {
        match expr {
            // Vector calculus operators indicate vector/scalar relationships
            Expression::Gradient { expr } => {
                // Input is scalar, output is vector
                if let Expression::Variable(name) = expr.as_ref() {
                    self.declare_scalar(name.clone());
                }
            }
            Expression::Divergence { field } | Expression::Curl { field } => {
                // Input is vector
                if let Expression::Variable(name) = field.as_ref() {
                    self.declare_vector(name.clone());
                }
            }
            Expression::MarkedVector { name, .. } => {
                self.declare_vector(name.clone());
            }
            Expression::Matrix(_) => {
                // Already a matrix literal
            }
            // Recursively analyze sub-expressions
            Expression::Binary { left, right, .. } => {
                self.infer_types_from_expression(left);
                self.infer_types_from_expression(right);
            }
            Expression::Unary { operand, .. } => {
                self.infer_types_from_expression(operand);
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.infer_types_from_expression(arg);
                }
            }
            _ => {}
        }
    }

    /// Convert context to a ParserConfig.
    pub fn to_parser_config(&self) -> ParserConfig {
        ParserConfig {
            number_system: self.number_system,
            declared_vectors: self.vectors.clone(),
            declared_matrices: self.matrices.clone(),
            declared_scalars: self.scalars.clone(),
            ..ParserConfig::default()
        }
    }
}

/// Parse multiple expressions with shared context.
#[allow(clippy::result_large_err)]
pub fn parse_system(inputs: &[&str], config: &ParserConfig) -> ParseResult<Vec<Expression>> {
    use crate::parse_latex;

    let mut ctx = ExpressionContext::new();
    ctx.number_system = config.number_system;
    ctx.vectors = config.declared_vectors.clone();
    ctx.matrices = config.declared_matrices.clone();
    ctx.scalars = config.declared_scalars.clone();

    let mut results = Vec::new();

    for input in inputs {
        let expr = parse_latex(input)?;
        ctx.analyze_expression(&expr);
        results.push(expr);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, VectorNotation};

    #[test]
    fn test_context_creation() {
        let ctx = ExpressionContext::new();
        assert_eq!(ctx.number_system(), NumberSystem::Real);
        assert!(ctx.variable_types.is_empty());
        assert!(ctx.vectors.is_empty());
        assert!(ctx.matrices.is_empty());
        assert!(ctx.scalars.is_empty());
    }

    #[test]
    fn test_context_with_number_system() {
        let ctx = ExpressionContext::with_number_system(NumberSystem::Complex);
        assert_eq!(ctx.number_system(), NumberSystem::Complex);
    }

    #[test]
    fn test_declare_vector() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_vector("v");

        assert!(ctx.is_vector("v"));
        assert!(!ctx.is_matrix("v"));
        assert_eq!(ctx.get_type("v"), Some(&MathType::Vector(None)));
    }

    #[test]
    fn test_declare_matrix() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_matrix("M");

        assert!(ctx.is_matrix("M"));
        assert!(!ctx.is_vector("M"));
        assert_eq!(ctx.get_type("M"), Some(&MathType::Matrix(None, None)));
    }

    #[test]
    fn test_declare_scalar() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_scalar("x");

        assert!(!ctx.is_vector("x"));
        assert!(!ctx.is_matrix("x"));
        assert_eq!(ctx.get_type("x"), Some(&MathType::Scalar));
    }

    #[test]
    fn test_declare_variable() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_variable("v", MathType::Vector(Some(3)));

        assert_eq!(ctx.get_type("v"), Some(&MathType::Vector(Some(3))));
    }

    #[test]
    fn test_binding_scope() {
        let mut ctx = ExpressionContext::new();

        ctx.push_binding("x".to_string(), Some(MathType::Scalar));
        assert!(ctx.is_bound("x"));
        assert!(!ctx.is_bound("y"));

        ctx.push_binding("y".to_string(), None);
        assert!(ctx.is_bound("y"));

        let popped = ctx.pop_binding();
        assert_eq!(popped, Some(("y".to_string(), None)));
        assert!(!ctx.is_bound("y"));
        assert!(ctx.is_bound("x"));

        let popped = ctx.pop_binding();
        assert_eq!(popped, Some(("x".to_string(), Some(MathType::Scalar))));
        assert!(!ctx.is_bound("x"));
    }

    #[test]
    fn test_infer_types_from_gradient() {
        let mut ctx = ExpressionContext::new();
        let expr = Expression::Gradient {
            expr: Box::new(Expression::Variable("f".to_string())),
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("f"), Some(&MathType::Scalar));
    }

    #[test]
    fn test_infer_types_from_divergence() {
        let mut ctx = ExpressionContext::new();
        let expr = Expression::Divergence {
            field: Box::new(Expression::Variable("F".to_string())),
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("F"), Some(&MathType::Vector(None)));
        assert!(ctx.is_vector("F"));
    }

    #[test]
    fn test_infer_types_from_curl() {
        let mut ctx = ExpressionContext::new();
        let expr = Expression::Curl {
            field: Box::new(Expression::Variable("F".to_string())),
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("F"), Some(&MathType::Vector(None)));
        assert!(ctx.is_vector("F"));
    }

    #[test]
    fn test_infer_types_from_marked_vector() {
        let mut ctx = ExpressionContext::new();
        let expr = Expression::MarkedVector {
            name: "v".to_string(),
            notation: VectorNotation::Bold,
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("v"), Some(&MathType::Vector(None)));
        assert!(ctx.is_vector("v"));
    }

    #[test]
    fn test_infer_types_from_binary_expression() {
        let mut ctx = ExpressionContext::new();
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Gradient {
                expr: Box::new(Expression::Variable("f".to_string())),
            }),
            right: Box::new(Expression::Divergence {
                field: Box::new(Expression::Variable("F".to_string())),
            }),
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("f"), Some(&MathType::Scalar));
        assert_eq!(ctx.get_type("F"), Some(&MathType::Vector(None)));
    }

    #[test]
    fn test_to_parser_config() {
        let mut ctx = ExpressionContext::with_number_system(NumberSystem::Complex);
        ctx.declare_vector("v");
        ctx.declare_matrix("M");
        ctx.declare_scalar("x");

        let config = ctx.to_parser_config();
        assert_eq!(config.number_system, NumberSystem::Complex);
        assert!(config.declared_vectors.contains("v"));
        assert!(config.declared_matrices.contains("M"));
        assert!(config.declared_scalars.contains("x"));
    }

    #[test]
    fn test_parse_system_basic() {
        let config = ParserConfig::default();
        let inputs = vec!["x + y", "2*x"];

        let exprs = parse_system(&inputs, &config).unwrap();
        assert_eq!(exprs.len(), 2);
    }

    #[test]
    fn test_parse_system_with_context() {
        let mut config = ParserConfig::default();
        config.declared_vectors.insert("v".to_string());

        let inputs = vec![r"\nabla f", r"\mathbf{v}"];

        let exprs = parse_system(&inputs, &config).unwrap();
        assert_eq!(exprs.len(), 2);
    }

    #[test]
    fn test_parse_system_error_propagation() {
        let config = ParserConfig::default();
        let inputs = vec!["x + y", "invalid $$$ syntax"];

        let result = parse_system(&inputs, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_context_clone() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_vector("v");
        ctx.declare_scalar("x");

        let cloned = ctx.clone();
        assert!(cloned.is_vector("v"));
        assert_eq!(cloned.get_type("x"), Some(&MathType::Scalar));
    }

    #[test]
    fn test_multiple_declarations() {
        let mut ctx = ExpressionContext::new();
        ctx.declare_vector("v1");
        ctx.declare_vector("v2");
        ctx.declare_matrix("M1");
        ctx.declare_matrix("M2");
        ctx.declare_scalar("x");
        ctx.declare_scalar("y");

        assert!(ctx.is_vector("v1"));
        assert!(ctx.is_vector("v2"));
        assert!(ctx.is_matrix("M1"));
        assert!(ctx.is_matrix("M2"));
        assert_eq!(ctx.get_type("x"), Some(&MathType::Scalar));
        assert_eq!(ctx.get_type("y"), Some(&MathType::Scalar));
    }

    #[test]
    fn test_undeclared_symbol() {
        let ctx = ExpressionContext::new();
        assert!(!ctx.is_vector("unknown"));
        assert!(!ctx.is_matrix("unknown"));
        assert_eq!(ctx.get_type("unknown"), None);
    }

    #[test]
    fn test_nested_bindings() {
        let mut ctx = ExpressionContext::new();

        ctx.push_binding("x".to_string(), Some(MathType::Scalar));
        ctx.push_binding("y".to_string(), Some(MathType::Vector(Some(3))));
        ctx.push_binding("z".to_string(), None);

        assert!(ctx.is_bound("x"));
        assert!(ctx.is_bound("y"));
        assert!(ctx.is_bound("z"));

        ctx.pop_binding();
        assert!(!ctx.is_bound("z"));
        assert!(ctx.is_bound("x"));
        assert!(ctx.is_bound("y"));
    }

    #[test]
    fn test_analyze_expression_recursion() {
        let mut ctx = ExpressionContext::new();

        // Create nested expression: (∇f) + (∇·F)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Gradient {
                expr: Box::new(Expression::Variable("f".to_string())),
            }),
            right: Box::new(Expression::Divergence {
                field: Box::new(Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expression::Variable("F".to_string())),
                    right: Box::new(Expression::Variable("G".to_string())),
                }),
            }),
        };

        ctx.analyze_expression(&expr);
        assert_eq!(ctx.get_type("f"), Some(&MathType::Scalar));
        // F and G aren't simple variables in the positions analyzed
        assert_eq!(ctx.get_type("F"), None);
    }

    #[test]
    fn test_parse_system_accumulates_context() {
        let config = ParserConfig::default();
        let inputs = vec![r"\nabla f", r"\nabla \cdot \mathbf{F}", r"\mathbf{v}"];

        let exprs = parse_system(&inputs, &config).unwrap();
        assert_eq!(exprs.len(), 3);

        // Verify expressions are parsed correctly
        assert!(matches!(exprs[0], Expression::Gradient { .. }));
        assert!(matches!(exprs[1], Expression::Divergence { .. }));
        assert!(matches!(exprs[2], Expression::MarkedVector { .. }));
    }
}
