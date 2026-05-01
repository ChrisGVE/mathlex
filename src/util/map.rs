//! Bottom-up generic map traversal for the expression tree.

use crate::ast::Expression;

use super::walker::map_children;

fn map_tensor_index<F>(indices: &[crate::ast::TensorIndex], f: &F) -> Vec<crate::ast::TensorIndex>
where
    F: Fn(Expression) -> Expression,
{
    indices
        .iter()
        .map(|idx| {
            let mapped = f(Expression::Variable(idx.name.clone()));
            let new_name = match mapped {
                Expression::Variable(n) => n,
                _ => idx.name.clone(),
            };
            crate::ast::TensorIndex {
                name: new_name,
                index_type: idx.index_type,
            }
        })
        .collect()
}

fn map_core<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    // Special case: tensor indices need name mapping through f
    match expr {
        Expression::Tensor { name, indices } => {
            return Expression::Tensor {
                name: name.clone(),
                indices: map_tensor_index(indices, f),
            };
        }
        Expression::KroneckerDelta { indices } => {
            return Expression::KroneckerDelta {
                indices: map_tensor_index(indices, f),
            };
        }
        Expression::LeviCivita { indices } => {
            return Expression::LeviCivita {
                indices: map_tensor_index(indices, f),
            };
        }
        _ => {}
    }

    // All other variants: structurally map children
    map_children(expr, &mut |child| child.map_impl(f))
}

impl Expression {
    /// Applies a bottom-up transformation to every node in the expression tree.
    ///
    /// The closure `f` is called on each node after its children have been
    /// transformed. This means the deepest nodes are transformed first
    /// (leaves → root), so `f` always receives fully-transformed subtrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // Double every integer in the tree
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Integer(2)),
    ///     right: Box::new(Expression::Integer(3)),
    /// };
    ///
    /// let doubled = expr.map(|e| match e {
    ///     Expression::Integer(n) => Expression::Integer(n * 2),
    ///     other => other,
    /// });
    ///
    /// // Verify the result: (4 + 6)
    /// match doubled {
    ///     Expression::Binary { left, right, .. } => {
    ///         assert_eq!(*left, Expression::Integer(4));
    ///         assert_eq!(*right, Expression::Integer(6));
    ///     }
    ///     _ => panic!("expected binary"),
    /// }
    /// ```
    pub fn map<F>(&self, f: F) -> Expression
    where
        F: Fn(Expression) -> Expression,
    {
        self.map_impl(&f)
    }

    pub(crate) fn map_impl<F>(&self, f: &F) -> Expression
    where
        F: Fn(Expression) -> Expression,
    {
        let mapped = map_core(self, f);
        f(mapped)
    }
}
