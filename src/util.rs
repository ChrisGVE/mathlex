//! # AST Utility Functions
//!
//! This module provides utility methods for querying and analyzing the AST.
//! These methods allow consumers to extract information about expressions
//! without manually traversing the tree structure.
//!
//! ## Key Features
//!
//! - **Variable Discovery**: Find all variables in an expression
//! - **Function Discovery**: Find all function calls in an expression
//! - **Constant Discovery**: Find all mathematical constants used
//! - **Tree Metrics**: Calculate depth and node count
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Create expression: 2 * π * x
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Binary {
//!         op: BinaryOp::Mul,
//!         left: Box::new(Expression::Integer(2)),
//!         right: Box::new(Expression::Constant(MathConstant::Pi)),
//!     }),
//!     right: Box::new(Expression::Variable("x".to_string())),
//! };
//!
//! // Query the expression
//! assert_eq!(expr.find_variables().len(), 1); // {x}
//! assert_eq!(expr.find_constants().len(), 1); // {π}
//! assert_eq!(expr.depth(), 3);
//! assert_eq!(expr.node_count(), 5);
//! ```

use crate::ast::{Expression, MathConstant};
use std::collections::HashSet;

impl Expression {
    /// Finds all unique variable names in the expression.
    ///
    /// Recursively traverses the AST and collects all `Variable` nodes,
    /// returning their names as a set. Index variables from summations
    /// and products are also included.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // x + y
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Variable("x".to_string())),
    ///     right: Box::new(Expression::Variable("y".to_string())),
    /// };
    ///
    /// let vars = expr.find_variables();
    /// assert_eq!(vars.len(), 2);
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// ```
    pub fn find_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();
        self.collect_variables(&mut variables);
        variables
    }

    /// Internal helper for recursively collecting variables.
    fn collect_variables(&self, variables: &mut HashSet<String>) {
        match self {
            // Leaf nodes - no recursion needed
            Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => {}

            // Variable node - add to set
            Expression::Variable(name) => {
                variables.insert(name.clone());
            }

            // Binary operations - recurse on both operands
            Expression::Rational {
                numerator,
                denominator,
            }
            | Expression::Complex {
                real: numerator,
                imaginary: denominator,
            }
            | Expression::Equation {
                left: numerator,
                right: denominator,
            }
            | Expression::Binary {
                left: numerator,
                right: denominator,
                ..
            }
            | Expression::Inequality {
                left: numerator,
                right: denominator,
                ..
            } => {
                numerator.collect_variables(variables);
                denominator.collect_variables(variables);
            }

            // Quaternion - recurse on all four components
            Expression::Quaternion {
                real,
                i,
                j,
                k,
            } => {
                real.collect_variables(variables);
                i.collect_variables(variables);
                j.collect_variables(variables);
                k.collect_variables(variables);
            }

            // Unary operations - recurse on operand
            Expression::Unary { operand, .. } => {
                operand.collect_variables(variables);
            }

            // Functions - recurse on all arguments
            Expression::Function { args, .. } => {
                for arg in args {
                    arg.collect_variables(variables);
                }
            }

            // Calculus operations with single expression
            Expression::Derivative { expr, var, .. }
            | Expression::PartialDerivative { expr, var, .. } => {
                expr.collect_variables(variables);
                // Note: var is the differentiation variable, also a variable
                variables.insert(var.clone());
            }

            // Integral - recurse on integrand and bounds
            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                integrand.collect_variables(variables);
                variables.insert(var.clone());
                if let Some(bounds) = bounds {
                    bounds.lower.collect_variables(variables);
                    bounds.upper.collect_variables(variables);
                }
            }

            // Multiple integral - recurse on integrand and bounds
            Expression::MultipleIntegral {
                integrand,
                vars,
                bounds,
                ..
            } => {
                integrand.collect_variables(variables);
                for v in vars {
                    variables.insert(v.clone());
                }
                if let Some(b) = bounds {
                    for ib in &b.bounds {
                        ib.lower.collect_variables(variables);
                        ib.upper.collect_variables(variables);
                    }
                }
            }

            // Closed integral - recurse on integrand
            Expression::ClosedIntegral {
                integrand,
                var,
                ..
            } => {
                integrand.collect_variables(variables);
                variables.insert(var.clone());
            }

            // Limit - recurse on expression and target value
            Expression::Limit { expr, var, to, .. } => {
                expr.collect_variables(variables);
                variables.insert(var.clone());
                to.collect_variables(variables);
            }

            // Sum and Product - recurse on bounds and body
            Expression::Sum {
                index,
                lower,
                upper,
                body,
            }
            | Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                variables.insert(index.clone());
                lower.collect_variables(variables);
                upper.collect_variables(variables);
                body.collect_variables(variables);
            }

            // Vector - recurse on all elements
            Expression::Vector(elements) => {
                for element in elements {
                    element.collect_variables(variables);
                }
            }

            // Matrix - recurse on all elements in all rows
            Expression::Matrix(rows) => {
                for row in rows {
                    for element in row {
                        element.collect_variables(variables);
                    }
                }
            }

            // Quantifiers - recurse on domain and body, include bound variable
            Expression::ForAll {
                variable,
                domain,
                body,
            }
            | Expression::Exists {
                variable,
                domain,
                body,
                ..
            } => {
                variables.insert(variable.clone());
                if let Some(d) = domain {
                    d.collect_variables(variables);
                }
                body.collect_variables(variables);
            }

            // Logical operations - recurse on all operands
            Expression::Logical { operands, .. } => {
                for operand in operands {
                    operand.collect_variables(variables);
                }
            }

            // MarkedVector - no variables to collect (name is not a variable reference)
            Expression::MarkedVector { .. } => {}

            // Vector products - recurse on both operands
            Expression::DotProduct { left, right }
            | Expression::CrossProduct { left, right }
            | Expression::OuterProduct { left, right } => {
                left.collect_variables(variables);
                right.collect_variables(variables);
            }

            // Set theory expressions
            Expression::NumberSetExpr(_) | Expression::EmptySet => {}

            Expression::SetOperation { left, right, .. } => {
                left.collect_variables(variables);
                right.collect_variables(variables);
            }

            Expression::SetRelationExpr { element, set, .. } => {
                element.collect_variables(variables);
                set.collect_variables(variables);
            }

            Expression::SetBuilder {
                variable,
                domain,
                predicate,
            } => {
                variables.insert(variable.clone());
                if let Some(d) = domain {
                    d.collect_variables(variables);
                }
                predicate.collect_variables(variables);
            }

            Expression::PowerSet { set } => {
                set.collect_variables(variables);
            }

            // Tensor notation - tensor index names are variables
            Expression::Tensor { indices, .. }
            | Expression::KroneckerDelta { indices }
            | Expression::LeviCivita { indices } => {
                for idx in indices {
                    variables.insert(idx.name.clone());
                }
            }
        }
    }

    /// Finds all unique function names in the expression.
    ///
    /// Recursively traverses the AST and collects all `Function` node names,
    /// returning them as a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // sin(x) + cos(y)
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Function {
    ///         name: "sin".to_string(),
    ///         args: vec![Expression::Variable("x".to_string())],
    ///     }),
    ///     right: Box::new(Expression::Function {
    ///         name: "cos".to_string(),
    ///         args: vec![Expression::Variable("y".to_string())],
    ///     }),
    /// };
    ///
    /// let funcs = expr.find_functions();
    /// assert_eq!(funcs.len(), 2);
    /// assert!(funcs.contains("sin"));
    /// assert!(funcs.contains("cos"));
    /// ```
    pub fn find_functions(&self) -> HashSet<String> {
        let mut functions = HashSet::new();
        self.collect_functions(&mut functions);
        functions
    }

    /// Internal helper for recursively collecting function names.
    fn collect_functions(&self, functions: &mut HashSet<String>) {
        match self {
            // Leaf nodes - no recursion needed
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Variable(_)
            | Expression::Constant(_) => {}

            // Binary operations - recurse on both operands
            Expression::Rational {
                numerator,
                denominator,
            }
            | Expression::Complex {
                real: numerator,
                imaginary: denominator,
            }
            | Expression::Equation {
                left: numerator,
                right: denominator,
            }
            | Expression::Binary {
                left: numerator,
                right: denominator,
                ..
            }
            | Expression::Inequality {
                left: numerator,
                right: denominator,
                ..
            } => {
                numerator.collect_functions(functions);
                denominator.collect_functions(functions);
            }

            // Quaternion - recurse on all four components
            Expression::Quaternion { real, i, j, k } => {
                real.collect_functions(functions);
                i.collect_functions(functions);
                j.collect_functions(functions);
                k.collect_functions(functions);
            }

            // Unary operations - recurse on operand
            Expression::Unary { operand, .. } => {
                operand.collect_functions(functions);
            }

            // Function - add name and recurse on arguments
            Expression::Function { name, args } => {
                functions.insert(name.clone());
                for arg in args {
                    arg.collect_functions(functions);
                }
            }

            // Calculus operations with single expression
            Expression::Derivative { expr, .. } | Expression::PartialDerivative { expr, .. } => {
                expr.collect_functions(functions);
            }

            // Integral - recurse on integrand and bounds
            Expression::Integral {
                integrand, bounds, ..
            } => {
                integrand.collect_functions(functions);
                if let Some(bounds) = bounds {
                    bounds.lower.collect_functions(functions);
                    bounds.upper.collect_functions(functions);
                }
            }

            // Multiple integral - recurse on integrand and bounds
            Expression::MultipleIntegral {
                integrand, bounds, ..
            } => {
                integrand.collect_functions(functions);
                if let Some(b) = bounds {
                    for ib in &b.bounds {
                        ib.lower.collect_functions(functions);
                        ib.upper.collect_functions(functions);
                    }
                }
            }

            // Closed integral - recurse on integrand
            Expression::ClosedIntegral { integrand, .. } => {
                integrand.collect_functions(functions);
            }

            // Limit - recurse on expression and target value
            Expression::Limit { expr, to, .. } => {
                expr.collect_functions(functions);
                to.collect_functions(functions);
            }

            // Sum and Product - recurse on bounds and body
            Expression::Sum {
                lower, upper, body, ..
            }
            | Expression::Product {
                lower, upper, body, ..
            } => {
                lower.collect_functions(functions);
                upper.collect_functions(functions);
                body.collect_functions(functions);
            }

            // Vector - recurse on all elements
            Expression::Vector(elements) => {
                for element in elements {
                    element.collect_functions(functions);
                }
            }

            // Matrix - recurse on all elements in all rows
            Expression::Matrix(rows) => {
                for row in rows {
                    for element in row {
                        element.collect_functions(functions);
                    }
                }
            }
       
            // Quantifiers
            Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
                if let Some(d) = domain { d.collect_functions(functions); }
                body.collect_functions(functions);
            }
            Expression::Logical { operands, .. } => {
                for op in operands { op.collect_functions(functions); }
            }

            // MarkedVector - no functions to collect
            Expression::MarkedVector { .. } => {}

            // Vector products - recurse on both operands
            Expression::DotProduct { left, right }
            | Expression::CrossProduct { left, right }
            | Expression::OuterProduct { left, right } => {
                left.collect_functions(functions);
                right.collect_functions(functions);
            }

            // Set theory expressions - no functions to collect from NumberSetExpr/EmptySet
            Expression::NumberSetExpr(_) | Expression::EmptySet => {}

            Expression::SetOperation { left, right, .. } => {
                left.collect_functions(functions);
                right.collect_functions(functions);
            }

            Expression::SetRelationExpr { element, set, .. } => {
                element.collect_functions(functions);
                set.collect_functions(functions);
            }

            Expression::SetBuilder {
                domain, predicate, ..
            } => {
                if let Some(d) = domain {
                    d.collect_functions(functions);
                }
                predicate.collect_functions(functions);
            }

            Expression::PowerSet { set } => {
                set.collect_functions(functions);
            }

            // Tensor notation - no functions to collect
            Expression::Tensor { .. }
            | Expression::KroneckerDelta { .. }
            | Expression::LeviCivita { .. } => {}
        }
    }

    /// Finds all unique mathematical constants in the expression.
    ///
    /// Recursively traverses the AST and collects all `Constant` nodes,
    /// returning them as a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, MathConstant, BinaryOp};
    ///
    /// // 2 * π + e
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Binary {
    ///         op: BinaryOp::Mul,
    ///         left: Box::new(Expression::Integer(2)),
    ///         right: Box::new(Expression::Constant(MathConstant::Pi)),
    ///     }),
    ///     right: Box::new(Expression::Constant(MathConstant::E)),
    /// };
    ///
    /// let consts = expr.find_constants();
    /// assert_eq!(consts.len(), 2);
    /// assert!(consts.contains(&MathConstant::Pi));
    /// assert!(consts.contains(&MathConstant::E));
    /// ```
    pub fn find_constants(&self) -> HashSet<MathConstant> {
        let mut constants = HashSet::new();
        self.collect_constants(&mut constants);
        constants
    }

    /// Internal helper for recursively collecting constants.
    fn collect_constants(&self, constants: &mut HashSet<MathConstant>) {
        match self {
            // Leaf nodes - check for constant
            Expression::Integer(_) | Expression::Float(_) | Expression::Variable(_) => {}

            Expression::Constant(c) => {
                constants.insert(*c);
            }

            // Binary operations - recurse on both operands
            Expression::Rational {
                numerator,
                denominator,
            }
            | Expression::Complex {
                real: numerator,
                imaginary: denominator,
            }
            | Expression::Equation {
                left: numerator,
                right: denominator,
            }
            | Expression::Binary {
                left: numerator,
                right: denominator,
                ..
            }
            | Expression::Inequality {
                left: numerator,
                right: denominator,
                ..
            } => {
                numerator.collect_constants(constants);
                denominator.collect_constants(constants);
            }

            // Quaternion - recurse on all four components
            Expression::Quaternion { real, i, j, k } => {
                real.collect_constants(constants);
                i.collect_constants(constants);
                j.collect_constants(constants);
                k.collect_constants(constants);
            }

            // Unary operations - recurse on operand
            Expression::Unary { operand, .. } => {
                operand.collect_constants(constants);
            }

            // Functions - recurse on all arguments
            Expression::Function { args, .. } => {
                for arg in args {
                    arg.collect_constants(constants);
                }
            }

            // Calculus operations with single expression
            Expression::Derivative { expr, .. } | Expression::PartialDerivative { expr, .. } => {
                expr.collect_constants(constants);
            }

            // Integral - recurse on integrand and bounds
            Expression::Integral {
                integrand, bounds, ..
            } => {
                integrand.collect_constants(constants);
                if let Some(bounds) = bounds {
                    bounds.lower.collect_constants(constants);
                    bounds.upper.collect_constants(constants);
                }
            }

            // Multiple integral - recurse on integrand and bounds
            Expression::MultipleIntegral {
                integrand, bounds, ..
            } => {
                integrand.collect_constants(constants);
                if let Some(b) = bounds {
                    for ib in &b.bounds {
                        ib.lower.collect_constants(constants);
                        ib.upper.collect_constants(constants);
                    }
                }
            }

            // Closed integral - recurse on integrand
            Expression::ClosedIntegral { integrand, .. } => {
                integrand.collect_constants(constants);
            }

            // Limit - recurse on expression and target value
            Expression::Limit { expr, to, .. } => {
                expr.collect_constants(constants);
                to.collect_constants(constants);
            }

            // Sum and Product - recurse on bounds and body
            Expression::Sum {
                lower, upper, body, ..
            }
            | Expression::Product {
                lower, upper, body, ..
            } => {
                lower.collect_constants(constants);
                upper.collect_constants(constants);
                body.collect_constants(constants);
            }

            // Vector - recurse on all elements
            Expression::Vector(elements) => {
                for element in elements {
                    element.collect_constants(constants);
                }
            }

            // Matrix - recurse on all elements in all rows
            Expression::Matrix(rows) => {
                for row in rows {
                    for element in row {
                        element.collect_constants(constants);
                    }
                }
            }
       
            // Quantifiers
            Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
                if let Some(d) = domain { d.collect_constants(constants); }
                body.collect_constants(constants);
            }
            Expression::Logical { operands, .. } => {
                for op in operands { op.collect_constants(constants); }
            }

            // MarkedVector - no constants to collect
            Expression::MarkedVector { .. } => {}

            // Vector products - recurse on both operands
            Expression::DotProduct { left, right }
            | Expression::CrossProduct { left, right }
            | Expression::OuterProduct { left, right } => {
                left.collect_constants(constants);
                right.collect_constants(constants);
            }

            // Set theory expressions - no constants to collect from NumberSetExpr/EmptySet
            Expression::NumberSetExpr(_) | Expression::EmptySet => {}

            Expression::SetOperation { left, right, .. } => {
                left.collect_constants(constants);
                right.collect_constants(constants);
            }

            Expression::SetRelationExpr { element, set, .. } => {
                element.collect_constants(constants);
                set.collect_constants(constants);
            }

            Expression::SetBuilder {
                domain, predicate, ..
            } => {
                if let Some(d) = domain {
                    d.collect_constants(constants);
                }
                predicate.collect_constants(constants);
            }

            Expression::PowerSet { set } => {
                set.collect_constants(constants);
            }

            // Tensor notation - no constants to collect
            Expression::Tensor { .. }
            | Expression::KroneckerDelta { .. }
            | Expression::LeviCivita { .. } => {}
        }
    }

    /// Calculates the maximum depth of the expression tree.
    ///
    /// The depth is defined as the longest path from the root to a leaf node.
    /// Leaf nodes (integers, floats, variables, constants) have depth 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // Simple leaf: x
    /// let leaf = Expression::Variable("x".to_string());
    /// assert_eq!(leaf.depth(), 1);
    ///
    /// // Binary expression: (x + y)
    /// let binary = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Variable("x".to_string())),
    ///     right: Box::new(Expression::Variable("y".to_string())),
    /// };
    /// assert_eq!(binary.depth(), 2);
    ///
    /// // Nested: ((x + y) * z)
    /// let nested = Expression::Binary {
    ///     op: BinaryOp::Mul,
    ///     left: Box::new(binary),
    ///     right: Box::new(Expression::Variable("z".to_string())),
    /// };
    /// assert_eq!(nested.depth(), 3);
    /// ```
    pub fn depth(&self) -> usize {
        match self {
            // Leaf nodes - depth 1
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Variable(_)
            | Expression::Constant(_) => 1,

            // Binary operations - 1 + max depth of children
            Expression::Rational {
                numerator,
                denominator,
            }
            | Expression::Complex {
                real: numerator,
                imaginary: denominator,
            }
            | Expression::Equation {
                left: numerator,
                right: denominator,
            }
            | Expression::Binary {
                left: numerator,
                right: denominator,
                ..
            }
            | Expression::Inequality {
                left: numerator,
                right: denominator,
                ..
            } => 1 + numerator.depth().max(denominator.depth()),

            // Quaternion - 1 + max depth of four components
            Expression::Quaternion { real, i, j, k } => {
                1 + real
                    .depth()
                    .max(i.depth())
                    .max(j.depth())
                    .max(k.depth())
            }

            // Unary operations - 1 + depth of operand
            Expression::Unary { operand, .. } => 1 + operand.depth(),

            // Functions - 1 + max depth of arguments
            Expression::Function { args, .. } => {
                if args.is_empty() {
                    1
                } else {
                    1 + args.iter().map(|arg| arg.depth()).max().unwrap_or(0)
                }
            }

            // Derivative and partial derivative - 1 + depth of expression
            Expression::Derivative { expr, .. } | Expression::PartialDerivative { expr, .. } => {
                1 + expr.depth()
            }

            // Integral - 1 + max depth among integrand and bounds
            Expression::Integral {
                integrand, bounds, ..
            } => {
                let integrand_depth = integrand.depth();
                let bounds_depth = bounds
                    .as_ref()
                    .map_or(0, |b| b.lower.depth().max(b.upper.depth()));
                1 + integrand_depth.max(bounds_depth)
            }

            // Multiple integral - 1 + max depth among integrand and all bounds
            Expression::MultipleIntegral {
                integrand, bounds, ..
            } => {
                let integrand_depth = integrand.depth();
                let bounds_depth = bounds.as_ref().map_or(0, |b| {
                    b.bounds
                        .iter()
                        .map(|ib| ib.lower.depth().max(ib.upper.depth()))
                        .max()
                        .unwrap_or(0)
                });
                1 + integrand_depth.max(bounds_depth)
            }

            // Closed integral - 1 + integrand depth
            Expression::ClosedIntegral { integrand, .. } => 1 + integrand.depth(),

            // Limit - 1 + max depth of expression and target
            Expression::Limit { expr, to, .. } => 1 + expr.depth().max(to.depth()),

            // Sum and Product - 1 + max depth of all components
            Expression::Sum {
                lower, upper, body, ..
            }
            | Expression::Product {
                lower, upper, body, ..
            } => 1 + lower.depth().max(upper.depth()).max(body.depth()),

            // Vector - 1 + max depth of elements
            Expression::Vector(elements) => {
                if elements.is_empty() {
                    1
                } else {
                    1 + elements.iter().map(|e| e.depth()).max().unwrap_or(0)
                }
            }

            // Matrix - 1 + max depth of all elements
            Expression::Matrix(rows) => {
                if rows.is_empty() {
                    1
                } else {
                    let max_element_depth = rows
                        .iter()
                        .flat_map(|row| row.iter())
                        .map(|e| e.depth())
                        .max()
                        .unwrap_or(0);
                    1 + max_element_depth
                }
            }

            Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
                let domain_depth = domain.as_ref().map_or(0, |d| d.depth());
                1 + domain_depth.max(body.depth())
            }

            Expression::Logical { operands, .. } => {
                1 + operands.iter().map(|e| e.depth()).max().unwrap_or(0)
            }

            // MarkedVector - depth 1 (leaf node)
            Expression::MarkedVector { .. } => 1,

            // Vector products - 1 + max depth of operands
            Expression::DotProduct { left, right }
            | Expression::CrossProduct { left, right }
            | Expression::OuterProduct { left, right } => 1 + left.depth().max(right.depth()),

            // Set theory expressions
            Expression::NumberSetExpr(_) | Expression::EmptySet => 1,

            Expression::SetOperation { left, right, .. } => 1 + left.depth().max(right.depth()),

            Expression::SetRelationExpr { element, set, .. } => {
                1 + element.depth().max(set.depth())
            }

            Expression::SetBuilder {
                domain, predicate, ..
            } => {
                let domain_depth = domain.as_ref().map_or(0, |d| d.depth());
                1 + domain_depth.max(predicate.depth())
            }

            Expression::PowerSet { set } => 1 + set.depth(),

            // Tensor notation - leaf nodes with depth 1
            Expression::Tensor { .. }
            | Expression::KroneckerDelta { .. }
            | Expression::LeviCivita { .. } => 1,
        }
    }

    /// Counts the total number of nodes in the expression tree.
    ///
    /// Every AST node is counted, including the root node. This provides
    /// a measure of expression complexity.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // Simple leaf: x
    /// let leaf = Expression::Variable("x".to_string());
    /// assert_eq!(leaf.node_count(), 1);
    ///
    /// // Binary expression: (x + y)
    /// let binary = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Variable("x".to_string())),
    ///     right: Box::new(Expression::Variable("y".to_string())),
    /// };
    /// assert_eq!(binary.node_count(), 3); // Add node + x + y
    ///
    /// // Nested: ((x + y) * z)
    /// let nested = Expression::Binary {
    ///     op: BinaryOp::Mul,
    ///     left: Box::new(binary),
    ///     right: Box::new(Expression::Variable("z".to_string())),
    /// };
    /// assert_eq!(nested.node_count(), 5); // Mul + (Add + x + y) + z
    /// ```
    pub fn node_count(&self) -> usize {
        match self {
            // Leaf nodes - count 1
            Expression::Integer(_)
            | Expression::Float(_)
            | Expression::Variable(_)
            | Expression::Constant(_) => 1,

            // Binary operations - 1 + count of both children
            Expression::Rational {
                numerator,
                denominator,
            }
            | Expression::Complex {
                real: numerator,
                imaginary: denominator,
            }
            | Expression::Equation {
                left: numerator,
                right: denominator,
            }
            | Expression::Binary {
                left: numerator,
                right: denominator,
                ..
            }
            | Expression::Inequality {
                left: numerator,
                right: denominator,
                ..
            } => 1 + numerator.node_count() + denominator.node_count(),

            // Quaternion - 1 + sum of all four component counts
            Expression::Quaternion { real, i, j, k } => {
                1 + real.node_count() + i.node_count() + j.node_count() + k.node_count()
            }

            // Unary operations - 1 + count of operand
            Expression::Unary { operand, .. } => 1 + operand.node_count(),

            // Functions - 1 + sum of argument counts
            Expression::Function { args, .. } => {
                1 + args.iter().map(|arg| arg.node_count()).sum::<usize>()
            }

            // Derivative and partial derivative - 1 + expression count
            Expression::Derivative { expr, .. } | Expression::PartialDerivative { expr, .. } => {
                1 + expr.node_count()
            }

            // Integral - 1 + integrand count + bounds count
            Expression::Integral {
                integrand, bounds, ..
            } => {
                let bounds_count = bounds
                    .as_ref()
                    .map_or(0, |b| b.lower.node_count() + b.upper.node_count());
                1 + integrand.node_count() + bounds_count
            }

            // Multiple integral - 1 + integrand count + all bounds count
            Expression::MultipleIntegral {
                integrand, bounds, ..
            } => {
                let bounds_count = bounds.as_ref().map_or(0, |b| {
                    b.bounds
                        .iter()
                        .map(|ib| ib.lower.node_count() + ib.upper.node_count())
                        .sum::<usize>()
                });
                1 + integrand.node_count() + bounds_count
            }

            // Closed integral - 1 + integrand count
            Expression::ClosedIntegral { integrand, .. } => 1 + integrand.node_count(),

            // Limit - 1 + expression count + target count
            Expression::Limit { expr, to, .. } => 1 + expr.node_count() + to.node_count(),

            // Sum and Product - 1 + counts of all components
            Expression::Sum {
                lower, upper, body, ..
            }
            | Expression::Product {
                lower, upper, body, ..
            } => 1 + lower.node_count() + upper.node_count() + body.node_count(),

            // Vector - 1 + sum of element counts
            Expression::Vector(elements) => {
                1 + elements.iter().map(|e| e.node_count()).sum::<usize>()
            }

            // Matrix - 1 + sum of all element counts
            Expression::Matrix(rows) => {
                1 + rows
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|e| e.node_count())
                    .sum::<usize>()
            }

            // ForAll - 1 + domain count (if any) + body count
            Expression::ForAll { domain, body, .. } => {
                let domain_count = domain.as_ref().map_or(0, |d| d.node_count());
                1 + domain_count + body.node_count()
            }

            // Exists - 1 + domain count (if any) + body count
            Expression::Exists { domain, body, .. } => {
                let domain_count = domain.as_ref().map_or(0, |d| d.node_count());
                1 + domain_count + body.node_count()
            }

            // Logical - 1 + sum of operand counts
            Expression::Logical { operands, .. } => {
                1 + operands.iter().map(|e| e.node_count()).sum::<usize>()
            }

            // MarkedVector - count 1 (leaf node)
            Expression::MarkedVector { .. } => 1,

            // Vector products - 1 + count of both operands
            Expression::DotProduct { left, right }
            | Expression::CrossProduct { left, right }
            | Expression::OuterProduct { left, right } => {
                1 + left.node_count() + right.node_count()
            }

            // Set theory expressions
            Expression::NumberSetExpr(_) | Expression::EmptySet => 1,

            Expression::SetOperation { left, right, .. } => {
                1 + left.node_count() + right.node_count()
            }

            Expression::SetRelationExpr { element, set, .. } => {
                1 + element.node_count() + set.node_count()
            }

            Expression::SetBuilder {
                domain, predicate, ..
            } => {
                let domain_count = domain.as_ref().map_or(0, |d| d.node_count());
                1 + domain_count + predicate.node_count()
            }

            Expression::PowerSet { set } => 1 + set.node_count(),

            // Tensor notation - 1 node each
            Expression::Tensor { .. }
            | Expression::KroneckerDelta { .. }
            | Expression::LeviCivita { .. } => 1,
        }
    }

    /// Substitutes all occurrences of a variable with a replacement expression.
    ///
    /// This method performs variable substitution throughout the expression tree,
    /// respecting bound variable scoping rules. Bound variables in calculus and
    /// iterator constructs are not substituted within their scope.
    ///
    /// # Bound Variables
    ///
    /// The following constructs introduce bound variables that are NOT substituted
    /// within their scope:
    ///
    /// - `Sum { index, ... }` - `index` is bound in `body`
    /// - `Product { index, ... }` - `index` is bound in `body`
    /// - `Integral { var, ... }` - `var` is bound in `integrand`
    /// - `Limit { var, ... }` - `var` is bound in `expr`
    /// - `Derivative { var, ... }` - `var` is bound in `expr`
    /// - `PartialDerivative { var, ... }` - `var` is bound in `expr`
    ///
    /// However, variables in bounds and limits (e.g., `lower`, `upper`, `to`)
    /// are still substituted since they are outside the binding scope.
    pub fn substitute(&self, var: &str, replacement: &Expression) -> Expression {
        match self {
            // Leaf nodes - check for variable match
            Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => self.clone(),

            Expression::Variable(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }

            // Binary operations - recurse on both operands
            Expression::Rational {
                numerator,
                denominator,
            } => Expression::Rational {
                numerator: Box::new(numerator.substitute(var, replacement)),
                denominator: Box::new(denominator.substitute(var, replacement)),
            },

            Expression::Complex { real, imaginary } => Expression::Complex {
                real: Box::new(real.substitute(var, replacement)),
                imaginary: Box::new(imaginary.substitute(var, replacement)),
            },

            Expression::Quaternion {
                real,
                i: qi,
                j: qj,
                k: qk,
            } => Expression::Quaternion {
                real: Box::new(real.substitute(var, replacement)),
                i: Box::new(qi.substitute(var, replacement)),
                j: Box::new(qj.substitute(var, replacement)),
                k: Box::new(qk.substitute(var, replacement)),
            },

            Expression::Binary { op, left, right } => Expression::Binary {
                op: *op,
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            Expression::Equation { left, right } => Expression::Equation {
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            Expression::Inequality { op, left, right } => Expression::Inequality {
                op: *op,
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            // Unary operations - recurse on operand
            Expression::Unary { op, operand } => Expression::Unary {
                op: *op,
                operand: Box::new(operand.substitute(var, replacement)),
            },

            // Functions - recurse on all arguments
            Expression::Function { name, args } => Expression::Function {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|arg| arg.substitute(var, replacement))
                    .collect(),
            },

            // Derivative - var is bound in expr
            Expression::Derivative {
                expr,
                var: diff_var,
                order,
            } => {
                if diff_var == var {
                    // var is bound in expr, don't substitute there
                    self.clone()
                } else {
                    Expression::Derivative {
                        expr: Box::new(expr.substitute(var, replacement)),
                        var: diff_var.clone(),
                        order: *order,
                    }
                }
            }

            // PartialDerivative - var is bound in expr
            Expression::PartialDerivative {
                expr,
                var: diff_var,
                order,
            } => {
                if diff_var == var {
                    // var is bound in expr, don't substitute there
                    self.clone()
                } else {
                    Expression::PartialDerivative {
                        expr: Box::new(expr.substitute(var, replacement)),
                        var: diff_var.clone(),
                        order: *order,
                    }
                }
            }

            // Integral - var is bound in integrand but not in bounds
            Expression::Integral {
                integrand,
                var: int_var,
                bounds,
            } => {
                if int_var == var {
                    // var is bound in integrand, don't substitute there
                    // but still substitute in bounds
                    Expression::Integral {
                        integrand: integrand.clone(),
                        var: int_var.clone(),
                        bounds: bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                            lower: Box::new(b.lower.substitute(var, replacement)),
                            upper: Box::new(b.upper.substitute(var, replacement)),
                        }),
                    }
                } else {
                    Expression::Integral {
                        integrand: Box::new(integrand.substitute(var, replacement)),
                        var: int_var.clone(),
                        bounds: bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                            lower: Box::new(b.lower.substitute(var, replacement)),
                            upper: Box::new(b.upper.substitute(var, replacement)),
                        }),
                    }
                }
            }

            // Multiple integral - vars are bound in integrand
            Expression::MultipleIntegral {
                dimension,
                integrand,
                bounds,
                vars,
            } => {
                // Check if any of the integration variables match
                let is_bound = vars.contains(&var.to_string());
                Expression::MultipleIntegral {
                    dimension: *dimension,
                    integrand: if is_bound {
                        integrand.clone()
                    } else {
                        Box::new(integrand.substitute(var, replacement))
                    },
                    bounds: bounds.as_ref().map(|b| crate::ast::MultipleBounds {
                        bounds: b.bounds.iter().map(|ib| crate::ast::IntegralBounds {
                            lower: Box::new(ib.lower.substitute(var, replacement)),
                            upper: Box::new(ib.upper.substitute(var, replacement)),
                        }).collect(),
                    }),
                    vars: vars.clone(),
                }
            }

            // Closed integral - var is bound in integrand
            Expression::ClosedIntegral {
                dimension,
                integrand,
                surface,
                var: int_var,
            } => {
                if int_var == var {
                    Expression::ClosedIntegral {
                        dimension: *dimension,
                        integrand: integrand.clone(),
                        surface: surface.clone(),
                        var: int_var.clone(),
                    }
                } else {
                    Expression::ClosedIntegral {
                        dimension: *dimension,
                        integrand: Box::new(integrand.substitute(var, replacement)),
                        surface: surface.clone(),
                        var: int_var.clone(),
                    }
                }
            }

            // Limit - var is bound in expr but not in to
            Expression::Limit {
                expr,
                var: limit_var,
                to,
                direction,
            } => {
                if limit_var == var {
                    // var is bound in expr, don't substitute there
                    // but still substitute in to
                    Expression::Limit {
                        expr: expr.clone(),
                        var: limit_var.clone(),
                        to: Box::new(to.substitute(var, replacement)),
                        direction: *direction,
                    }
                } else {
                    Expression::Limit {
                        expr: Box::new(expr.substitute(var, replacement)),
                        var: limit_var.clone(),
                        to: Box::new(to.substitute(var, replacement)),
                        direction: *direction,
                    }
                }
            }

            // Sum - index is bound in body but not in lower/upper
            Expression::Sum {
                index,
                lower,
                upper,
                body,
            } => {
                if index == var {
                    // index is bound in body, don't substitute there
                    // but still substitute in bounds
                    Expression::Sum {
                        index: index.clone(),
                        lower: Box::new(lower.substitute(var, replacement)),
                        upper: Box::new(upper.substitute(var, replacement)),
                        body: body.clone(),
                    }
                } else {
                    Expression::Sum {
                        index: index.clone(),
                        lower: Box::new(lower.substitute(var, replacement)),
                        upper: Box::new(upper.substitute(var, replacement)),
                        body: Box::new(body.substitute(var, replacement)),
                    }
                }
            }

            // Product - index is bound in body but not in lower/upper
            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                if index == var {
                    // index is bound in body, don't substitute there
                    // but still substitute in bounds
                    Expression::Product {
                        index: index.clone(),
                        lower: Box::new(lower.substitute(var, replacement)),
                        upper: Box::new(upper.substitute(var, replacement)),
                        body: body.clone(),
                    }
                } else {
                    Expression::Product {
                        index: index.clone(),
                        lower: Box::new(lower.substitute(var, replacement)),
                        upper: Box::new(upper.substitute(var, replacement)),
                        body: Box::new(body.substitute(var, replacement)),
                    }
                }
            }

            // Vector - recurse on all elements
            Expression::Vector(elements) => Expression::Vector(
                elements
                    .iter()
                    .map(|e| e.substitute(var, replacement))
                    .collect(),
            ),

            // Matrix - recurse on all elements in all rows
            Expression::Matrix(rows) => Expression::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|e| e.substitute(var, replacement)).collect())
                    .collect(),
            ),

            // ForAll - variable is bound in body
            Expression::ForAll {
                variable: bound_var,
                domain,
                body,
            } => {
                if bound_var == var {
                    // var is bound in body, don't substitute there
                    Expression::ForAll {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute(var, replacement))),
                        body: body.clone(),
                    }
                } else {
                    Expression::ForAll {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute(var, replacement))),
                        body: Box::new(body.substitute(var, replacement)),
                    }
                }
            }

            // Exists - variable is bound in body
            Expression::Exists {
                variable: bound_var,
                domain,
                body,
                unique,
            } => {
                if bound_var == var {
                    // var is bound in body, don't substitute there
                    Expression::Exists {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute(var, replacement))),
                        body: body.clone(),
                        unique: *unique,
                    }
                } else {
                    Expression::Exists {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute(var, replacement))),
                        body: Box::new(body.substitute(var, replacement)),
                        unique: *unique,
                    }
                }
            }

            // Logical - recurse on all operands
            Expression::Logical { op, operands } => Expression::Logical {
                op: *op,
                operands: operands
                    .iter()
                    .map(|e| e.substitute(var, replacement))
                    .collect(),
            },

            // MarkedVector - no variables to substitute (name is literal, not a variable reference)
            Expression::MarkedVector { .. } => self.clone(),

            // Vector products - recurse on both operands
            Expression::DotProduct { left, right } => Expression::DotProduct {
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            Expression::CrossProduct { left, right } => Expression::CrossProduct {
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            Expression::OuterProduct { left, right } => Expression::OuterProduct {
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            // Set theory expressions
            Expression::NumberSetExpr(_) | Expression::EmptySet => self.clone(),

            Expression::SetOperation { op, left, right } => Expression::SetOperation {
                op: *op,
                left: Box::new(left.substitute(var, replacement)),
                right: Box::new(right.substitute(var, replacement)),
            },

            Expression::SetRelationExpr {
                relation,
                element,
                set,
            } => Expression::SetRelationExpr {
                relation: *relation,
                element: Box::new(element.substitute(var, replacement)),
                set: Box::new(set.substitute(var, replacement)),
            },

            // SetBuilder - variable is bound in predicate
            Expression::SetBuilder {
                variable: bound_var,
                domain,
                predicate,
            } => {
                if bound_var == var {
                    // var is bound, don't substitute in predicate
                    Expression::SetBuilder {
                        variable: bound_var.clone(),
                        domain: domain
                            .as_ref()
                            .map(|d| Box::new(d.substitute(var, replacement))),
                        predicate: predicate.clone(),
                    }
                } else {
                    Expression::SetBuilder {
                        variable: bound_var.clone(),
                        domain: domain
                            .as_ref()
                            .map(|d| Box::new(d.substitute(var, replacement))),
                        predicate: Box::new(predicate.substitute(var, replacement)),
                    }
                }
            }

            Expression::PowerSet { set } => Expression::PowerSet {
                set: Box::new(set.substitute(var, replacement)),
            },

            // Tensor notation - substitute in index names (if index name matches var)
            Expression::Tensor { name, indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if idx.name == var {
                            // If replacement is a variable, use its name; otherwise keep original
                            if let Expression::Variable(new_name) = replacement {
                                crate::ast::TensorIndex {
                                    name: new_name.clone(),
                                    index_type: idx.index_type,
                                }
                            } else {
                                idx.clone()
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::Tensor {
                    name: name.clone(),
                    indices: new_indices,
                }
            }

            Expression::KroneckerDelta { indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if idx.name == var {
                            if let Expression::Variable(new_name) = replacement {
                                crate::ast::TensorIndex {
                                    name: new_name.clone(),
                                    index_type: idx.index_type,
                                }
                            } else {
                                idx.clone()
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::KroneckerDelta { indices: new_indices }
            }

            Expression::LeviCivita { indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if idx.name == var {
                            if let Expression::Variable(new_name) = replacement {
                                crate::ast::TensorIndex {
                                    name: new_name.clone(),
                                    index_type: idx.index_type,
                                }
                            } else {
                                idx.clone()
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::LeviCivita { indices: new_indices }
            }
        }
    }

    /// Substitutes multiple variables simultaneously with replacement expressions.
    ///
    /// This method performs simultaneous substitution of multiple variables,
    /// respecting bound variable scoping rules. The substitutions are applied
    /// simultaneously, meaning that replacements don't affect each other.
    ///
    /// # Bound Variables
    ///
    /// Same scoping rules as [`substitute`](Expression::substitute) apply.
    /// See that method's documentation for details on bound variables.
    pub fn substitute_all(
        &self,
        subs: &std::collections::HashMap<String, Expression>,
    ) -> Expression {
        match self {
            // Leaf nodes - check for variable match
            Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => self.clone(),

            Expression::Variable(name) => {
                if let Some(replacement) = subs.get(name) {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }

            // Binary operations - recurse on both operands
            Expression::Rational {
                numerator,
                denominator,
            } => Expression::Rational {
                numerator: Box::new(numerator.substitute_all(subs)),
                denominator: Box::new(denominator.substitute_all(subs)),
            },

            Expression::Complex { real, imaginary } => Expression::Complex {
                real: Box::new(real.substitute_all(subs)),
                imaginary: Box::new(imaginary.substitute_all(subs)),
            },

            Expression::Quaternion {
                real,
                i: qi,
                j: qj,
                k: qk,
            } => Expression::Quaternion {
                real: Box::new(real.substitute_all(subs)),
                i: Box::new(qi.substitute_all(subs)),
                j: Box::new(qj.substitute_all(subs)),
                k: Box::new(qk.substitute_all(subs)),
            },

            Expression::Binary { op, left, right } => Expression::Binary {
                op: *op,
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            Expression::Equation { left, right } => Expression::Equation {
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            Expression::Inequality { op, left, right } => Expression::Inequality {
                op: *op,
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            // Unary operations - recurse on operand
            Expression::Unary { op, operand } => Expression::Unary {
                op: *op,
                operand: Box::new(operand.substitute_all(subs)),
            },

            // Functions - recurse on all arguments
            Expression::Function { name, args } => Expression::Function {
                name: name.clone(),
                args: args.iter().map(|arg| arg.substitute_all(subs)).collect(),
            },

            // Derivative - var is bound in expr
            Expression::Derivative {
                expr,
                var: diff_var,
                order,
            } => {
                if subs.contains_key(diff_var) {
                    // var is bound in expr, don't substitute there
                    self.clone()
                } else {
                    Expression::Derivative {
                        expr: Box::new(expr.substitute_all(subs)),
                        var: diff_var.clone(),
                        order: *order,
                    }
                }
            }

            // PartialDerivative - var is bound in expr
            Expression::PartialDerivative {
                expr,
                var: diff_var,
                order,
            } => {
                if subs.contains_key(diff_var) {
                    // var is bound in expr, don't substitute there
                    self.clone()
                } else {
                    Expression::PartialDerivative {
                        expr: Box::new(expr.substitute_all(subs)),
                        var: diff_var.clone(),
                        order: *order,
                    }
                }
            }

            // Integral - var is bound in integrand but not in bounds
            Expression::Integral {
                integrand,
                var: int_var,
                bounds,
            } => {
                if subs.contains_key(int_var) {
                    // var is bound in integrand, don't substitute there
                    // but still substitute in bounds
                    Expression::Integral {
                        integrand: integrand.clone(),
                        var: int_var.clone(),
                        bounds: bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                            lower: Box::new(b.lower.substitute_all(subs)),
                            upper: Box::new(b.upper.substitute_all(subs)),
                        }),
                    }
                } else {
                    Expression::Integral {
                        integrand: Box::new(integrand.substitute_all(subs)),
                        var: int_var.clone(),
                        bounds: bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                            lower: Box::new(b.lower.substitute_all(subs)),
                            upper: Box::new(b.upper.substitute_all(subs)),
                        }),
                    }
                }
            }

            // Multiple integral - vars are bound in integrand
            Expression::MultipleIntegral {
                dimension,
                integrand,
                bounds,
                vars,
            } => {
                // Check if any of the integration variables are in subs
                let any_bound = vars.iter().any(|v| subs.contains_key(v));
                Expression::MultipleIntegral {
                    dimension: *dimension,
                    integrand: if any_bound {
                        integrand.clone()
                    } else {
                        Box::new(integrand.substitute_all(subs))
                    },
                    bounds: bounds.as_ref().map(|b| crate::ast::MultipleBounds {
                        bounds: b.bounds.iter().map(|ib| crate::ast::IntegralBounds {
                            lower: Box::new(ib.lower.substitute_all(subs)),
                            upper: Box::new(ib.upper.substitute_all(subs)),
                        }).collect(),
                    }),
                    vars: vars.clone(),
                }
            }

            // Closed integral - var is bound in integrand
            Expression::ClosedIntegral {
                dimension,
                integrand,
                surface,
                var: int_var,
            } => {
                if subs.contains_key(int_var) {
                    Expression::ClosedIntegral {
                        dimension: *dimension,
                        integrand: integrand.clone(),
                        surface: surface.clone(),
                        var: int_var.clone(),
                    }
                } else {
                    Expression::ClosedIntegral {
                        dimension: *dimension,
                        integrand: Box::new(integrand.substitute_all(subs)),
                        surface: surface.clone(),
                        var: int_var.clone(),
                    }
                }
            }

            // Limit - var is bound in expr but not in to
            Expression::Limit {
                expr,
                var: limit_var,
                to,
                direction,
            } => {
                if subs.contains_key(limit_var) {
                    // var is bound in expr, don't substitute there
                    // but still substitute in to
                    Expression::Limit {
                        expr: expr.clone(),
                        var: limit_var.clone(),
                        to: Box::new(to.substitute_all(subs)),
                        direction: *direction,
                    }
                } else {
                    Expression::Limit {
                        expr: Box::new(expr.substitute_all(subs)),
                        var: limit_var.clone(),
                        to: Box::new(to.substitute_all(subs)),
                        direction: *direction,
                    }
                }
            }

            // Sum - index is bound in body but not in lower/upper
            Expression::Sum {
                index,
                lower,
                upper,
                body,
            } => {
                if subs.contains_key(index) {
                    // index is bound in body, don't substitute there
                    // but still substitute in bounds
                    Expression::Sum {
                        index: index.clone(),
                        lower: Box::new(lower.substitute_all(subs)),
                        upper: Box::new(upper.substitute_all(subs)),
                        body: body.clone(),
                    }
                } else {
                    Expression::Sum {
                        index: index.clone(),
                        lower: Box::new(lower.substitute_all(subs)),
                        upper: Box::new(upper.substitute_all(subs)),
                        body: Box::new(body.substitute_all(subs)),
                    }
                }
            }

            // Product - index is bound in body but not in lower/upper
            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                if subs.contains_key(index) {
                    // index is bound in body, don't substitute there
                    // but still substitute in bounds
                    Expression::Product {
                        index: index.clone(),
                        lower: Box::new(lower.substitute_all(subs)),
                        upper: Box::new(upper.substitute_all(subs)),
                        body: body.clone(),
                    }
                } else {
                    Expression::Product {
                        index: index.clone(),
                        lower: Box::new(lower.substitute_all(subs)),
                        upper: Box::new(upper.substitute_all(subs)),
                        body: Box::new(body.substitute_all(subs)),
                    }
                }
            }

            // Vector - recurse on all elements
            Expression::Vector(elements) => {
                Expression::Vector(elements.iter().map(|e| e.substitute_all(subs)).collect())
            }

            // Matrix - recurse on all elements in all rows
            Expression::Matrix(rows) => Expression::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|e| e.substitute_all(subs)).collect())
                    .collect(),
            ),

            // ForAll - variable is bound in body
            Expression::ForAll {
                variable: bound_var,
                domain,
                body,
            } => {
                if subs.contains_key(bound_var) {
                    // var is bound in body, don't substitute there
                    Expression::ForAll {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        body: body.clone(),
                    }
                } else {
                    Expression::ForAll {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        body: Box::new(body.substitute_all(subs)),
                    }
                }
            }

            // Exists - variable is bound in body
            Expression::Exists {
                variable: bound_var,
                domain,
                body,
                unique,
            } => {
                if subs.contains_key(bound_var) {
                    // var is bound in body, don't substitute there
                    Expression::Exists {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        body: body.clone(),
                        unique: *unique,
                    }
                } else {
                    Expression::Exists {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        body: Box::new(body.substitute_all(subs)),
                        unique: *unique,
                    }
                }
            }

            // Logical - recurse on all operands
            Expression::Logical { op, operands } => Expression::Logical {
                op: *op,
                operands: operands.iter().map(|e| e.substitute_all(subs)).collect(),
            },

            // MarkedVector - no variables to substitute (name is literal, not a variable reference)
            Expression::MarkedVector { .. } => self.clone(),

            // Vector products - recurse on both operands
            Expression::DotProduct { left, right } => Expression::DotProduct {
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            Expression::CrossProduct { left, right } => Expression::CrossProduct {
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            Expression::OuterProduct { left, right } => Expression::OuterProduct {
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            // Set theory expressions
            Expression::NumberSetExpr(_) | Expression::EmptySet => self.clone(),

            Expression::SetOperation { op, left, right } => Expression::SetOperation {
                op: *op,
                left: Box::new(left.substitute_all(subs)),
                right: Box::new(right.substitute_all(subs)),
            },

            Expression::SetRelationExpr {
                relation,
                element,
                set,
            } => Expression::SetRelationExpr {
                relation: *relation,
                element: Box::new(element.substitute_all(subs)),
                set: Box::new(set.substitute_all(subs)),
            },

            // SetBuilder - variable is bound in predicate
            Expression::SetBuilder {
                variable: bound_var,
                domain,
                predicate,
            } => {
                if subs.contains_key(bound_var) {
                    // var is bound, don't substitute in predicate
                    Expression::SetBuilder {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        predicate: predicate.clone(),
                    }
                } else {
                    Expression::SetBuilder {
                        variable: bound_var.clone(),
                        domain: domain.as_ref().map(|d| Box::new(d.substitute_all(subs))),
                        predicate: Box::new(predicate.substitute_all(subs)),
                    }
                }
            }

            Expression::PowerSet { set } => Expression::PowerSet {
                set: Box::new(set.substitute_all(subs)),
            },

            // Tensor notation - substitute in index names
            Expression::Tensor { name, indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if let Some(Expression::Variable(new_name)) = subs.get(&idx.name) {
                            crate::ast::TensorIndex {
                                name: new_name.clone(),
                                index_type: idx.index_type,
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::Tensor {
                    name: name.clone(),
                    indices: new_indices,
                }
            }

            Expression::KroneckerDelta { indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if let Some(Expression::Variable(new_name)) = subs.get(&idx.name) {
                            crate::ast::TensorIndex {
                                name: new_name.clone(),
                                index_type: idx.index_type,
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::KroneckerDelta { indices: new_indices }
            }

            Expression::LeviCivita { indices } => {
                let new_indices = indices
                    .iter()
                    .map(|idx| {
                        if let Some(Expression::Variable(new_name)) = subs.get(&idx.name) {
                            crate::ast::TensorIndex {
                                name: new_name.clone(),
                                index_type: idx.index_type,
                            }
                        } else {
                            idx.clone()
                        }
                    })
                    .collect();
                Expression::LeviCivita { indices: new_indices }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Direction, InequalityOp, IntegralBounds, MathConstant, UnaryOp};

    // Tests for find_variables

    #[test]
    fn test_find_variables_leaf_nodes() {
        // Integer - no variables
        let expr = Expression::Integer(42);
        assert_eq!(expr.find_variables().len(), 0);

        // Float - no variables
        let expr = Expression::Float(crate::ast::MathFloat::from(3.14));
        assert_eq!(expr.find_variables().len(), 0);

        // Constant - no variables
        let expr = Expression::Constant(MathConstant::Pi);
        assert_eq!(expr.find_variables().len(), 0);

        // Variable - one variable
        let expr = Expression::Variable("x".to_string());
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_binary() {
        // x + y
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_duplicate() {
        // x + x
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("x".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1); // Set deduplicates
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_rational() {
        // x / y
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Variable("x".to_string())),
            denominator: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_complex() {
        // a + bi
        let expr = Expression::Complex {
            real: Box::new(Expression::Variable("a".to_string())),
            imaginary: Box::new(Expression::Variable("b".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_find_variables_unary() {
        // -x
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_function() {
        // sin(x, y)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_derivative() {
        // d/dx(f)
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("f"));
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_integral() {
        // ∫₀¹ x dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_integral_with_variable_bounds() {
        // ∫ₐᵇ x dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Variable("a".to_string())),
                upper: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_find_variables_limit() {
        // lim x→0 f
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("f"));
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_sum() {
        // Σᵢ₌₁ⁿ i
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("i"));
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_find_variables_vector() {
        // [x, y, z]
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
            Expression::Variable("z".to_string()),
        ]);
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }

    #[test]
    fn test_find_variables_matrix() {
        // [[a, b], [c, d]]
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("a".to_string()),
                Expression::Variable("b".to_string()),
            ],
            vec![
                Expression::Variable("c".to_string()),
                Expression::Variable("d".to_string()),
            ],
        ]);
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 4);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("c"));
        assert!(vars.contains("d"));
    }

    #[test]
    fn test_find_variables_equation() {
        // x = y
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    // Tests for find_functions

    #[test]
    fn test_find_functions_leaf_nodes() {
        // No functions in leaf nodes
        let expr = Expression::Variable("x".to_string());
        assert_eq!(expr.find_functions().len(), 0);
    }

    #[test]
    fn test_find_functions_simple() {
        // sin(x)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1);
        assert!(funcs.contains("sin"));
    }

    #[test]
    fn test_find_functions_multiple() {
        // sin(x) + cos(y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            right: Box::new(Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("y".to_string())],
            }),
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 2);
        assert!(funcs.contains("sin"));
        assert!(funcs.contains("cos"));
    }

    #[test]
    fn test_find_functions_nested() {
        // sin(cos(x))
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }],
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 2);
        assert!(funcs.contains("sin"));
        assert!(funcs.contains("cos"));
    }

    #[test]
    fn test_find_functions_duplicate() {
        // sin(x) + sin(y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            right: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("y".to_string())],
            }),
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1); // Set deduplicates
        assert!(funcs.contains("sin"));
    }

    #[test]
    fn test_find_functions_in_integral() {
        // ∫ sin(x) dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            var: "x".to_string(),
            bounds: None,
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1);
        assert!(funcs.contains("sin"));
    }

    // Tests for find_constants

    #[test]
    fn test_find_constants_none() {
        // x + 1
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(1)),
        };
        assert_eq!(expr.find_constants().len(), 0);
    }

    #[test]
    fn test_find_constants_single() {
        // π
        let expr = Expression::Constant(MathConstant::Pi);
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1);
        assert!(consts.contains(&MathConstant::Pi));
    }

    #[test]
    fn test_find_constants_multiple() {
        // π + e
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Constant(MathConstant::Pi)),
            right: Box::new(Expression::Constant(MathConstant::E)),
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 2);
        assert!(consts.contains(&MathConstant::Pi));
        assert!(consts.contains(&MathConstant::E));
    }

    #[test]
    fn test_find_constants_duplicate() {
        // π + π
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Constant(MathConstant::Pi)),
            right: Box::new(Expression::Constant(MathConstant::Pi)),
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1); // Set deduplicates
        assert!(consts.contains(&MathConstant::Pi));
    }

    #[test]
    fn test_find_constants_all_types() {
        // [π, e, i, ∞, -∞]
        let expr = Expression::Vector(vec![
            Expression::Constant(MathConstant::Pi),
            Expression::Constant(MathConstant::E),
            Expression::Constant(MathConstant::I),
            Expression::Constant(MathConstant::Infinity),
            Expression::Constant(MathConstant::NegInfinity),
        ]);
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 5);
        assert!(consts.contains(&MathConstant::Pi));
        assert!(consts.contains(&MathConstant::E));
        assert!(consts.contains(&MathConstant::I));
        assert!(consts.contains(&MathConstant::Infinity));
        assert!(consts.contains(&MathConstant::NegInfinity));
    }

    #[test]
    fn test_find_constants_in_limit() {
        // lim x→∞ f
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: Direction::Both,
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1);
        assert!(consts.contains(&MathConstant::Infinity));
    }

    // Tests for depth

    #[test]
    fn test_depth_leaf_nodes() {
        // All leaf nodes have depth 1
        assert_eq!(Expression::Integer(42).depth(), 1);
        assert_eq!(
            Expression::Float(crate::ast::MathFloat::from(3.14)).depth(),
            1
        );
        assert_eq!(Expression::Variable("x".to_string()).depth(), 1);
        assert_eq!(Expression::Constant(MathConstant::Pi).depth(), 1);
    }

    #[test]
    fn test_depth_unary() {
        // -x has depth 2
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_binary() {
        // x + y has depth 2
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_nested() {
        // (x + y) * z has depth 3
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            right: Box::new(Expression::Variable("z".to_string())),
        };
        assert_eq!(expr.depth(), 3);
    }

    #[test]
    fn test_depth_asymmetric() {
        // ((x + y) + z) + w has depth 4 on left, 1 on right
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::Variable("x".to_string())),
                    right: Box::new(Expression::Variable("y".to_string())),
                }),
                right: Box::new(Expression::Variable("z".to_string())),
            }),
            right: Box::new(Expression::Variable("w".to_string())),
        };
        assert_eq!(expr.depth(), 4);
    }

    #[test]
    fn test_depth_function() {
        // sin(x) has depth 2
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.depth(), 2);

        // sin(cos(x)) has depth 3
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }],
        };
        assert_eq!(expr.depth(), 3);
    }

    #[test]
    fn test_depth_vector() {
        // [x, y] has depth 2
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
        ]);
        assert_eq!(expr.depth(), 2);

        // Empty vector has depth 1
        let expr = Expression::Vector(vec![]);
        assert_eq!(expr.depth(), 1);
    }

    #[test]
    fn test_depth_matrix() {
        // [[x, y], [z, w]] has depth 2
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
            vec![
                Expression::Variable("z".to_string()),
                Expression::Variable("w".to_string()),
            ],
        ]);
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_integral() {
        // ∫₀¹ x dx has depth 2 (integrand is leaf, bounds are leaves)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.depth(), 2);
    }

    // Tests for node_count

    #[test]
    fn test_node_count_leaf_nodes() {
        // All leaf nodes have count 1
        assert_eq!(Expression::Integer(42).node_count(), 1);
        assert_eq!(
            Expression::Float(crate::ast::MathFloat::from(3.14)).node_count(),
            1
        );
        assert_eq!(Expression::Variable("x".to_string()).node_count(), 1);
        assert_eq!(Expression::Constant(MathConstant::Pi).node_count(), 1);
    }

    #[test]
    fn test_node_count_unary() {
        // -x has 2 nodes (Neg + x)
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.node_count(), 2);
    }

    #[test]
    fn test_node_count_binary() {
        // x + y has 3 nodes (Add + x + y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_nested() {
        // (x + y) * z has 5 nodes (Mul + (Add + x + y) + z)
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            right: Box::new(Expression::Variable("z".to_string())),
        };
        assert_eq!(expr.node_count(), 5);
    }

    #[test]
    fn test_node_count_function() {
        // sin() has 1 node
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![],
        };
        assert_eq!(expr.node_count(), 1);

        // sin(x) has 2 nodes (Function + x)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.node_count(), 2);

        // max(x, y, z) has 4 nodes (Function + x + y + z)
        let expr = Expression::Function {
            name: "max".to_string(),
            args: vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
                Expression::Variable("z".to_string()),
            ],
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_vector() {
        // [] has 1 node
        let expr = Expression::Vector(vec![]);
        assert_eq!(expr.node_count(), 1);

        // [x, y, z] has 4 nodes (Vector + x + y + z)
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
            Expression::Variable("z".to_string()),
        ]);
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_matrix() {
        // [[x, y], [z, w]] has 5 nodes (Matrix + x + y + z + w)
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
            vec![
                Expression::Variable("z".to_string()),
                Expression::Variable("w".to_string()),
            ],
        ]);
        assert_eq!(expr.node_count(), 5);
    }

    #[test]
    fn test_node_count_integral() {
        // ∫ x dx has 2 nodes (Integral + x)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };
        assert_eq!(expr.node_count(), 2);

        // ∫₀¹ x dx has 4 nodes (Integral + x + 0 + 1)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_sum() {
        // Σᵢ₌₁ⁿ i has 4 nodes (Sum + 1 + n + i)
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_equation() {
        // x = y has 3 nodes (Equation + x + y)
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_inequality() {
        // x < y has 3 nodes (Inequality + x + y)
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_complex_expression() {
        // 2 * π * x has 5 nodes
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Constant(MathConstant::Pi)),
            }),
            right: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.node_count(), 5);
    }
}
