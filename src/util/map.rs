//! Bottom-up generic map traversal for the expression tree.

use crate::ast::{ExprKind, Expression};

use super::walker::map_children;

fn map_tensor_index<F>(indices: &[crate::ast::TensorIndex], f: &F) -> Vec<crate::ast::TensorIndex>
where
    F: Fn(Expression) -> Expression,
{
    indices
        .iter()
        .map(|idx| {
            let mapped = f(Expression::variable(idx.name.clone()).into());
            let new_name = match &mapped.kind {
                ExprKind::Variable(n) => n.clone(),
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
    match &expr.kind {
        ExprKind::Tensor { name, indices } => {
            return ExprKind::Tensor {
                name: name.clone(),
                indices: map_tensor_index(indices, f),
            }
            .into();
        }
        ExprKind::KroneckerDelta { indices } => {
            return ExprKind::KroneckerDelta {
                indices: map_tensor_index(indices, f),
            }
            .into();
        }
        ExprKind::LeviCivita { indices } => {
            return ExprKind::LeviCivita {
                indices: map_tensor_index(indices, f),
            }
            .into();
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
    /// use mathlex::ast::{ExprKind, Expression, BinaryOp};
    ///
    /// // Double every integer in the tree
    /// let expr: Expression = ExprKind::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::integer(2)),
    ///     right: Box::new(Expression::integer(3)),
    /// }.into();
    ///
    /// let doubled = expr.map(|e| match e.kind {
    ///     ExprKind::Integer(n) => ExprKind::Integer(n * 2).into(),
    ///     _ => e,
    /// });
    ///
    /// // Verify the result: (4 + 6)
    /// match &doubled.kind {
    ///     ExprKind::Binary { left, right, .. } => {
    ///         assert_eq!(**left, Expression::integer(4));
    ///         assert_eq!(**right, Expression::integer(6));
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
