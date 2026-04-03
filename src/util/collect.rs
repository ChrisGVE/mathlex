//! Variable, function, and constant collection — `impl Expression` methods.

use crate::ast::{Expression, MathConstant};
use std::collections::HashSet;

use super::collect_consts::cc_core;
use super::collect_fns::cf_core;
use super::collect_vars::cv_core;

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

    pub(crate) fn collect_variables(&self, variables: &mut HashSet<String>) {
        cv_core(self, variables);
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

    pub(crate) fn collect_functions(&self, functions: &mut HashSet<String>) {
        cf_core(self, functions);
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

    pub(crate) fn collect_constants(&self, constants: &mut HashSet<MathConstant>) {
        cc_core(self, constants);
    }
}
