//! Tree metrics: depth and node count.

use crate::ast::Expression;

use super::walker::for_each_child;

fn depth_core(expr: &Expression) -> usize {
    let mut max_child = 0usize;
    for_each_child(expr, |child| {
        max_child = max_child.max(depth_core(child));
    });
    1 + max_child
}

fn nc_core(expr: &Expression) -> usize {
    let mut total = 0usize;
    for_each_child(expr, |child| {
        total += nc_core(child);
    });
    1 + total
}

impl Expression {
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
        depth_core(self)
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
        nc_core(self)
    }
}
