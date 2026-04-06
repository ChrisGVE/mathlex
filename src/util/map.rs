//! Bottom-up generic map traversal for the expression tree.

use crate::ast::Expression;

// ── map helpers ──────────────────────────────────────────────────────────────

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
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => expr.clone(),
        Expression::Variable(_) => expr.clone(),
        Expression::Rational {
            numerator,
            denominator,
        } => Expression::Rational {
            numerator: Box::new(numerator.map_impl(f)),
            denominator: Box::new(denominator.map_impl(f)),
        },
        Expression::Complex { real, imaginary } => Expression::Complex {
            real: Box::new(real.map_impl(f)),
            imaginary: Box::new(imaginary.map_impl(f)),
        },
        Expression::Quaternion {
            real,
            i: qi,
            j: qj,
            k: qk,
        } => Expression::Quaternion {
            real: Box::new(real.map_impl(f)),
            i: Box::new(qi.map_impl(f)),
            j: Box::new(qj.map_impl(f)),
            k: Box::new(qk.map_impl(f)),
        },
        Expression::Binary { op, left, right } => Expression::Binary {
            op: *op,
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::Equation { left, right } => Expression::Equation {
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::Inequality { op, left, right } => Expression::Inequality {
            op: *op,
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::Unary { op, operand } => Expression::Unary {
            op: *op,
            operand: Box::new(operand.map_impl(f)),
        },
        Expression::Function { name, args } => Expression::Function {
            name: name.clone(),
            args: args.iter().map(|a| a.map_impl(f)).collect(),
        },
        _ => map_calculus(expr, f),
    }
}

fn map_calculus<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::Derivative {
            expr: e,
            var,
            order,
        } => Expression::Derivative {
            expr: Box::new(e.map_impl(f)),
            var: var.clone(),
            order: *order,
        },
        Expression::PartialDerivative {
            expr: e,
            var,
            order,
        } => Expression::PartialDerivative {
            expr: Box::new(e.map_impl(f)),
            var: var.clone(),
            order: *order,
        },
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            let new_bounds = bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                lower: Box::new(b.lower.map_impl(f)),
                upper: Box::new(b.upper.map_impl(f)),
            });
            Expression::Integral {
                integrand: Box::new(integrand.map_impl(f)),
                var: var.clone(),
                bounds: new_bounds,
            }
        }
        _ => map_integrals(expr, f),
    }
}

fn map_integrals<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            let new_bounds = bounds.as_ref().map(|b| crate::ast::MultipleBounds {
                bounds: b
                    .bounds
                    .iter()
                    .map(|ib| crate::ast::IntegralBounds {
                        lower: Box::new(ib.lower.map_impl(f)),
                        upper: Box::new(ib.upper.map_impl(f)),
                    })
                    .collect(),
            });
            Expression::MultipleIntegral {
                dimension: *dimension,
                integrand: Box::new(integrand.map_impl(f)),
                bounds: new_bounds,
                vars: vars.clone(),
            }
        }
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => Expression::ClosedIntegral {
            dimension: *dimension,
            integrand: Box::new(integrand.map_impl(f)),
            surface: surface.clone(),
            var: var.clone(),
        },
        Expression::Limit {
            expr: e,
            var,
            to,
            direction,
        } => Expression::Limit {
            expr: Box::new(e.map_impl(f)),
            var: var.clone(),
            to: Box::new(to.map_impl(f)),
            direction: *direction,
        },
        _ => map_iter_containers(expr, f),
    }
}

fn map_iter_containers<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => Expression::Sum {
            index: index.clone(),
            lower: Box::new(lower.map_impl(f)),
            upper: Box::new(upper.map_impl(f)),
            body: Box::new(body.map_impl(f)),
        },
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => Expression::Product {
            index: index.clone(),
            lower: Box::new(lower.map_impl(f)),
            upper: Box::new(upper.map_impl(f)),
            body: Box::new(body.map_impl(f)),
        },
        Expression::Vector(elems) => {
            Expression::Vector(elems.iter().map(|e| e.map_impl(f)).collect())
        }
        Expression::Matrix(rows) => Expression::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|e| e.map_impl(f)).collect())
                .collect(),
        ),
        _ => map_logic_sets(expr, f),
    }
}

fn map_logic_sets<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::ForAll {
            variable,
            domain,
            body,
        } => Expression::ForAll {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.map_impl(f))),
            body: Box::new(body.map_impl(f)),
        },
        Expression::Exists {
            variable,
            domain,
            body,
            unique,
        } => Expression::Exists {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.map_impl(f))),
            body: Box::new(body.map_impl(f)),
            unique: *unique,
        },
        Expression::Logical { op, operands } => Expression::Logical {
            op: *op,
            operands: operands.iter().map(|e| e.map_impl(f)).collect(),
        },
        Expression::MarkedVector { .. } => expr.clone(),
        Expression::DotProduct { left, right } => Expression::DotProduct {
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::CrossProduct { left, right } => Expression::CrossProduct {
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::OuterProduct { left, right } => Expression::OuterProduct {
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::Gradient { expr: e } => Expression::Gradient {
            expr: Box::new(e.map_impl(f)),
        },
        Expression::Divergence { field } => Expression::Divergence {
            field: Box::new(field.map_impl(f)),
        },
        Expression::Curl { field } => Expression::Curl {
            field: Box::new(field.map_impl(f)),
        },
        Expression::Laplacian { expr: e } => Expression::Laplacian {
            expr: Box::new(e.map_impl(f)),
        },
        Expression::Nabla => Expression::Nabla,
        _ => map_linalg_sets_theory(expr, f),
    }
}

fn map_linalg_sets_theory<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::Determinant { matrix } => Expression::Determinant {
            matrix: Box::new(matrix.map_impl(f)),
        },
        Expression::Trace { matrix } => Expression::Trace {
            matrix: Box::new(matrix.map_impl(f)),
        },
        Expression::Rank { matrix } => Expression::Rank {
            matrix: Box::new(matrix.map_impl(f)),
        },
        Expression::ConjugateTranspose { matrix } => Expression::ConjugateTranspose {
            matrix: Box::new(matrix.map_impl(f)),
        },
        Expression::MatrixInverse { matrix } => Expression::MatrixInverse {
            matrix: Box::new(matrix.map_impl(f)),
        },
        Expression::NumberSetExpr(_) | Expression::EmptySet => expr.clone(),
        Expression::SetOperation { op, left, right } => Expression::SetOperation {
            op: *op,
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => Expression::SetRelationExpr {
            relation: *relation,
            element: Box::new(element.map_impl(f)),
            set: Box::new(set.map_impl(f)),
        },
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => Expression::SetBuilder {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.map_impl(f))),
            predicate: Box::new(predicate.map_impl(f)),
        },
        Expression::PowerSet { set } => Expression::PowerSet {
            set: Box::new(set.map_impl(f)),
        },
        _ => map_tensors_forms_theory(expr, f),
    }
}

fn map_tensors_forms_theory<F>(expr: &Expression, f: &F) -> Expression
where
    F: Fn(Expression) -> Expression,
{
    match expr {
        Expression::Tensor { name, indices } => Expression::Tensor {
            name: name.clone(),
            indices: map_tensor_index(indices, f),
        },
        Expression::KroneckerDelta { indices } => Expression::KroneckerDelta {
            indices: map_tensor_index(indices, f),
        },
        Expression::LeviCivita { indices } => Expression::LeviCivita {
            indices: map_tensor_index(indices, f),
        },
        Expression::Differential { var } => Expression::Differential { var: var.clone() },
        Expression::WedgeProduct { left, right } => Expression::WedgeProduct {
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        Expression::FunctionSignature {
            name,
            domain,
            codomain,
        } => Expression::FunctionSignature {
            name: name.clone(),
            domain: Box::new(domain.map_impl(f)),
            codomain: Box::new(codomain.map_impl(f)),
        },
        Expression::Composition { outer, inner } => Expression::Composition {
            outer: Box::new(outer.map_impl(f)),
            inner: Box::new(inner.map_impl(f)),
        },
        Expression::Relation { op, left, right } => Expression::Relation {
            op: *op,
            left: Box::new(left.map_impl(f)),
            right: Box::new(right.map_impl(f)),
        },
        _ => expr.clone(),
    }
}

// ── impl Expression ──────────────────────────────────────────────────────────

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
