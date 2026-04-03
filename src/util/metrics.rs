//! Tree metrics: depth and node count.

use crate::ast::Expression;

// ── depth helpers ────────────────────────────────────────────────────────────

fn depth_core(expr: &Expression) -> usize {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_) => 1,
        Expression::Rational {
            numerator: l,
            denominator: r,
        }
        | Expression::Complex {
            real: l,
            imaginary: r,
        }
        | Expression::Equation { left: l, right: r }
        | Expression::Binary {
            left: l, right: r, ..
        }
        | Expression::Inequality {
            left: l, right: r, ..
        } => 1 + l.depth().max(r.depth()),
        Expression::Quaternion { real, i, j, k } => {
            1 + real.depth().max(i.depth()).max(j.depth()).max(k.depth())
        }
        Expression::Unary { operand, .. } => 1 + operand.depth(),
        Expression::Function { args, .. } => 1 + args.iter().map(|a| a.depth()).max().unwrap_or(0),
        _ => depth_calculus_and_rest(expr),
    }
}

fn depth_calculus_and_rest(expr: &Expression) -> usize {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            1 + e.depth()
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            let bd = bounds
                .as_ref()
                .map_or(0, |b| b.lower.depth().max(b.upper.depth()));
            1 + integrand.depth().max(bd)
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            let bd = bounds.as_ref().map_or(0, |b| {
                b.bounds
                    .iter()
                    .map(|ib| ib.lower.depth().max(ib.upper.depth()))
                    .max()
                    .unwrap_or(0)
            });
            1 + integrand.depth().max(bd)
        }
        Expression::ClosedIntegral { integrand, .. } => 1 + integrand.depth(),
        Expression::Limit { expr: e, to, .. } => 1 + e.depth().max(to.depth()),
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => 1 + lower.depth().max(upper.depth()).max(body.depth()),
        Expression::Vector(elems) => 1 + elems.iter().map(|e| e.depth()).max().unwrap_or(0),
        Expression::Matrix(rows) => {
            1 + rows
                .iter()
                .flat_map(|r| r.iter())
                .map(|e| e.depth())
                .max()
                .unwrap_or(0)
        }
        _ => depth_logic_sets_and_rest(expr),
    }
}

fn depth_logic_sets_and_rest(expr: &Expression) -> usize {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            1 + domain.as_ref().map_or(0, |d| d.depth()).max(body.depth())
        }
        Expression::Logical { operands, .. } => {
            1 + operands.iter().map(|e| e.depth()).max().unwrap_or(0)
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => 1,
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::SetRelationExpr {
            element: left,
            set: right,
            ..
        }
        | Expression::WedgeProduct { left, right } => 1 + left.depth().max(right.depth()),
        Expression::Gradient { expr } | Expression::Laplacian { expr } => 1 + expr.depth(),
        Expression::Divergence { field } | Expression::Curl { field } => 1 + field.depth(),
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => 1 + matrix.depth(),
        _ => depth_sets_theory(expr),
    }
}

fn depth_sets_theory(expr: &Expression) -> usize {
    match expr {
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            1 + domain
                .as_ref()
                .map_or(0, |d| d.depth())
                .max(predicate.depth())
        }
        Expression::PowerSet { set } => 1 + set.depth(),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => 1 + domain.depth().max(codomain.depth()),
        Expression::Composition { outer, inner } => 1 + outer.depth().max(inner.depth()),
        Expression::Relation { left, right, .. } => 1 + left.depth().max(right.depth()),
        _ => 1,
    }
}

// ── node_count helpers ───────────────────────────────────────────────────────

fn nc_core(expr: &Expression) -> usize {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_) => 1,
        Expression::Rational {
            numerator: l,
            denominator: r,
        }
        | Expression::Complex {
            real: l,
            imaginary: r,
        }
        | Expression::Equation { left: l, right: r }
        | Expression::Binary {
            left: l, right: r, ..
        }
        | Expression::Inequality {
            left: l, right: r, ..
        } => 1 + l.node_count() + r.node_count(),
        Expression::Quaternion { real, i, j, k } => {
            1 + real.node_count() + i.node_count() + j.node_count() + k.node_count()
        }
        Expression::Unary { operand, .. } => 1 + operand.node_count(),
        Expression::Function { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
        _ => nc_calculus_and_rest(expr),
    }
}

fn nc_calculus_and_rest(expr: &Expression) -> usize {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            1 + e.node_count()
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            let bc = bounds
                .as_ref()
                .map_or(0, |b| b.lower.node_count() + b.upper.node_count());
            1 + integrand.node_count() + bc
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            let bc = bounds.as_ref().map_or(0, |b| {
                b.bounds
                    .iter()
                    .map(|ib| ib.lower.node_count() + ib.upper.node_count())
                    .sum()
            });
            1 + integrand.node_count() + bc
        }
        Expression::ClosedIntegral { integrand, .. } => 1 + integrand.node_count(),
        Expression::Limit { expr: e, to, .. } => 1 + e.node_count() + to.node_count(),
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => 1 + lower.node_count() + upper.node_count() + body.node_count(),
        Expression::Vector(elems) => 1 + elems.iter().map(|e| e.node_count()).sum::<usize>(),
        Expression::Matrix(rows) => {
            1 + rows
                .iter()
                .flat_map(|r| r.iter())
                .map(|e| e.node_count())
                .sum::<usize>()
        }
        _ => nc_logic_sets_and_rest(expr),
    }
}

fn nc_logic_sets_and_rest(expr: &Expression) -> usize {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            1 + domain.as_ref().map_or(0, |d| d.node_count()) + body.node_count()
        }
        Expression::Logical { operands, .. } => {
            1 + operands.iter().map(|e| e.node_count()).sum::<usize>()
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => 1,
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::SetRelationExpr {
            element: left,
            set: right,
            ..
        }
        | Expression::WedgeProduct { left, right } => 1 + left.node_count() + right.node_count(),
        Expression::Gradient { expr } | Expression::Laplacian { expr } => 1 + expr.node_count(),
        Expression::Divergence { field } | Expression::Curl { field } => 1 + field.node_count(),
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => 1 + matrix.node_count(),
        _ => nc_sets_theory(expr),
    }
}

fn nc_sets_theory(expr: &Expression) -> usize {
    match expr {
        Expression::SetBuilder {
            domain, predicate, ..
        } => 1 + domain.as_ref().map_or(0, |d| d.node_count()) + predicate.node_count(),
        Expression::PowerSet { set } => 1 + set.node_count(),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => 1 + domain.node_count() + codomain.node_count(),
        Expression::Composition { outer, inner } => 1 + outer.node_count() + inner.node_count(),
        Expression::Relation { left, right, .. } => 1 + left.node_count() + right.node_count(),
        _ => 1,
    }
}

// ── impl Expression ──────────────────────────────────────────────────────────

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
