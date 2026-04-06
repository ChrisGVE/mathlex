//! Generic fold/reduce traversal for the expression tree.

use crate::ast::Expression;

// ── fold helpers ─────────────────────────────────────────────────────────────

fn fold_core<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Constant(_)
        | Expression::Variable(_) => acc,
        Expression::Rational {
            numerator,
            denominator,
        } => {
            let acc = numerator.fold_impl(acc, f);
            denominator.fold_impl(acc, f)
        }
        Expression::Complex { real, imaginary } => {
            let acc = real.fold_impl(acc, f);
            imaginary.fold_impl(acc, f)
        }
        Expression::Quaternion { real, i, j, k } => {
            let acc = real.fold_impl(acc, f);
            let acc = i.fold_impl(acc, f);
            let acc = j.fold_impl(acc, f);
            k.fold_impl(acc, f)
        }
        Expression::Binary { left, right, .. } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        Expression::Equation { left, right } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        Expression::Inequality { left, right, .. } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        Expression::Unary { operand, .. } => operand.fold_impl(acc, f),
        Expression::Function { args, .. } => args.iter().fold(acc, |a, arg| arg.fold_impl(a, f)),
        _ => fold_calculus(expr, acc, f),
    }
}

fn fold_calculus<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            e.fold_impl(acc, f)
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            let acc = integrand.fold_impl(acc, f);
            if let Some(b) = bounds.as_ref() {
                let acc = b.lower.fold_impl(acc, f);
                b.upper.fold_impl(acc, f)
            } else {
                acc
            }
        }
        _ => fold_integrals(expr, acc, f),
    }
}

fn fold_integrals<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            let acc = integrand.fold_impl(acc, f);
            if let Some(b) = bounds.as_ref() {
                b.bounds.iter().fold(acc, |a, ib| {
                    let a = ib.lower.fold_impl(a, f);
                    ib.upper.fold_impl(a, f)
                })
            } else {
                acc
            }
        }
        Expression::ClosedIntegral { integrand, .. } => integrand.fold_impl(acc, f),
        Expression::Limit { expr: e, to, .. } => {
            let acc = e.fold_impl(acc, f);
            to.fold_impl(acc, f)
        }
        _ => fold_iter_containers(expr, acc, f),
    }
}

fn fold_iter_containers<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            let acc = lower.fold_impl(acc, f);
            let acc = upper.fold_impl(acc, f);
            body.fold_impl(acc, f)
        }
        Expression::Vector(elems) => elems.iter().fold(acc, |a, e| e.fold_impl(a, f)),
        Expression::Matrix(rows) => rows
            .iter()
            .flat_map(|r| r.iter())
            .fold(acc, |a, e| e.fold_impl(a, f)),
        _ => fold_logic_sets(expr, acc, f),
    }
}

fn fold_logic_sets<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            let acc = if let Some(d) = domain.as_ref() {
                d.fold_impl(acc, f)
            } else {
                acc
            };
            body.fold_impl(acc, f)
        }
        Expression::Logical { operands, .. } => operands.iter().fold(acc, |a, e| e.fold_impl(a, f)),
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla => acc,
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::WedgeProduct { left, right } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        Expression::Gradient { expr: e } | Expression::Laplacian { expr: e } => e.fold_impl(acc, f),
        Expression::Divergence { field } | Expression::Curl { field } => field.fold_impl(acc, f),
        _ => fold_linalg_sets_theory(expr, acc, f),
    }
}

fn fold_linalg_sets_theory<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => matrix.fold_impl(acc, f),
        Expression::SetOperation { left, right, .. }
        | Expression::SetRelationExpr {
            element: left,
            set: right,
            ..
        } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            let acc = if let Some(d) = domain.as_ref() {
                d.fold_impl(acc, f)
            } else {
                acc
            };
            predicate.fold_impl(acc, f)
        }
        Expression::PowerSet { set } => set.fold_impl(acc, f),
        _ => fold_tensors_forms_theory(expr, acc, f),
    }
}

fn fold_tensors_forms_theory<T, F>(expr: &Expression, acc: T, f: &F) -> T
where
    F: Fn(T, &Expression) -> T,
    T: Clone,
{
    match expr {
        Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => acc,
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            let acc = domain.fold_impl(acc, f);
            codomain.fold_impl(acc, f)
        }
        Expression::Composition { outer, inner } => {
            let acc = outer.fold_impl(acc, f);
            inner.fold_impl(acc, f)
        }
        Expression::Relation { left, right, .. } => {
            let acc = left.fold_impl(acc, f);
            right.fold_impl(acc, f)
        }
        _ => acc,
    }
}

// ── impl Expression ──────────────────────────────────────────────────────────

impl Expression {
    /// Folds the expression tree into a single value using a bottom-up traversal.
    ///
    /// Children are folded first (left-to-right), then `f` is applied to
    /// the accumulated result and the current node. This means `f` is called
    /// for every node in the tree, leaves first.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // Count all nodes: should equal node_count()
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Integer(1)),
    ///     right: Box::new(Expression::Integer(2)),
    /// };
    ///
    /// let count = expr.fold(0usize, |acc, _| acc + 1);
    /// assert_eq!(count, expr.node_count());
    /// ```
    pub fn fold<T, F>(&self, init: T, f: F) -> T
    where
        F: Fn(T, &Expression) -> T,
        T: Clone,
    {
        self.fold_impl(init, &f)
    }

    pub(crate) fn fold_impl<T, F>(&self, acc: T, f: &F) -> T
    where
        F: Fn(T, &Expression) -> T,
        T: Clone,
    {
        let child_acc = fold_core(self, acc, f);
        f(child_acc, self)
    }
}
