//! Generic expression tree traversal and mapping.
//!
//! Provides `for_each_child` (read-only visitor) and `map_children`
//! (structural reconstruction) over all ~50 Expression variants,
//! eliminating the need for each utility to implement its own
//! exhaustive match scaffold.

use crate::ast::{Expression, IntegralBounds, MultipleBounds};

/// Call `f` on every immediate child `Expression` of `expr`.
///
/// This is the single canonical traversal function — all tree-walking
/// utilities (collect_vars, collect_fns, collect_consts, depth, node_count)
/// delegate to it instead of reimplementing the match scaffold.
pub(crate) fn for_each_child(expr: &Expression, mut f: impl FnMut(&Expression)) {
    for_each_child_ref(expr, &mut f);
}

/// Internal helper using `&mut` to avoid closure size blowup in recursion.
fn for_each_child_ref(expr: &Expression, f: &mut impl FnMut(&Expression)) {
    match expr {
        // ── Leaf nodes ──────────────────────────────────────────────────
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_)
        | Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Differential { .. } => {}

        // Tensor-like leaves: indices are strings, not Expressions
        Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. } => {}

        // ── One child ───────────────────────────────────────────────────
        Expression::Unary { operand, .. } => f(operand),
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            f(e)
        }
        Expression::ClosedIntegral { integrand, .. } => f(integrand),
        Expression::Gradient { expr: e } | Expression::Laplacian { expr: e } => f(e),
        Expression::Divergence { field } | Expression::Curl { field } => f(field),
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => f(matrix),
        Expression::PowerSet { set } => f(set),

        // ── Two children ────────────────────────────────────────────────
        Expression::Rational {
            numerator,
            denominator,
        } => {
            f(numerator);
            f(denominator);
        }
        Expression::Complex { real, imaginary } => {
            f(real);
            f(imaginary);
        }
        Expression::Binary { left, right, .. }
        | Expression::Equation { left, right }
        | Expression::Inequality { left, right, .. }
        | Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            f(left);
            f(right);
        }
        Expression::SetRelationExpr { element, set, .. } => {
            f(element);
            f(set);
        }
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            f(domain);
            f(codomain);
        }
        Expression::Composition { outer, inner } => {
            f(outer);
            f(inner);
        }
        Expression::Relation { left, right, .. } => {
            f(left);
            f(right);
        }

        // ── Four children ───────────────────────────────────────────────
        Expression::Quaternion { real, i, j, k } => {
            f(real);
            f(i);
            f(j);
            f(k);
        }

        // ── Variable-length children ────────────────────────────────────
        Expression::Function { args, .. } => {
            for a in args {
                f(a);
            }
        }
        Expression::Vector(elems) => {
            for e in elems {
                f(e);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    f(e);
                }
            }
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                f(o);
            }
        }

        // ── Special: children + optional bounds ─────────────────────────
        Expression::Integral {
            integrand, bounds, ..
        } => {
            f(integrand);
            if let Some(b) = bounds {
                f(&b.lower);
                f(&b.upper);
            }
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            f(integrand);
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    f(&ib.lower);
                    f(&ib.upper);
                }
            }
        }
        Expression::Limit { expr: e, to, .. } => {
            f(e);
            f(to);
        }
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            f(lower);
            f(upper);
            f(body);
        }
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                f(d);
            }
            f(body);
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            if let Some(d) = domain {
                f(d);
            }
            f(predicate);
        }
    }
}

/// Return a new `Expression` identical to `expr` but with every child
/// `Expression` replaced by `f(child)`. Non-expression fields (operators,
/// variable names, flags) are cloned unchanged.
pub(crate) fn map_children(
    expr: &Expression,
    f: &mut impl FnMut(&Expression) -> Expression,
) -> Expression {
    match expr {
        // ── Leaf nodes (no children to map) ─────────────────────────────
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_)
        | Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Differential { .. }
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. } => expr.clone(),

        // ── One child ───────────────────────────────────────────────────
        Expression::Unary { op, operand } => Expression::Unary {
            op: *op,
            operand: Box::new(f(operand)),
        },
        Expression::Derivative {
            expr: e,
            var,
            order,
        } => Expression::Derivative {
            expr: Box::new(f(e)),
            var: var.clone(),
            order: *order,
        },
        Expression::PartialDerivative {
            expr: e,
            var,
            order,
        } => Expression::PartialDerivative {
            expr: Box::new(f(e)),
            var: var.clone(),
            order: *order,
        },
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => Expression::ClosedIntegral {
            dimension: *dimension,
            integrand: Box::new(f(integrand)),
            surface: surface.clone(),
            var: var.clone(),
        },
        Expression::Gradient { expr: e } => Expression::Gradient {
            expr: Box::new(f(e)),
        },
        Expression::Laplacian { expr: e } => Expression::Laplacian {
            expr: Box::new(f(e)),
        },
        Expression::Divergence { field } => Expression::Divergence {
            field: Box::new(f(field)),
        },
        Expression::Curl { field } => Expression::Curl {
            field: Box::new(f(field)),
        },
        Expression::Determinant { matrix } => Expression::Determinant {
            matrix: Box::new(f(matrix)),
        },
        Expression::Trace { matrix } => Expression::Trace {
            matrix: Box::new(f(matrix)),
        },
        Expression::Rank { matrix } => Expression::Rank {
            matrix: Box::new(f(matrix)),
        },
        Expression::ConjugateTranspose { matrix } => Expression::ConjugateTranspose {
            matrix: Box::new(f(matrix)),
        },
        Expression::MatrixInverse { matrix } => Expression::MatrixInverse {
            matrix: Box::new(f(matrix)),
        },
        Expression::PowerSet { set } => Expression::PowerSet {
            set: Box::new(f(set)),
        },

        // ── Two children ────────────────────────────────────────────────
        Expression::Rational {
            numerator,
            denominator,
        } => Expression::Rational {
            numerator: Box::new(f(numerator)),
            denominator: Box::new(f(denominator)),
        },
        Expression::Complex { real, imaginary } => Expression::Complex {
            real: Box::new(f(real)),
            imaginary: Box::new(f(imaginary)),
        },
        Expression::Binary { op, left, right } => Expression::Binary {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::Equation { left, right } => Expression::Equation {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::Inequality { op, left, right } => Expression::Inequality {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::DotProduct { left, right } => Expression::DotProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::CrossProduct { left, right } => Expression::CrossProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::OuterProduct { left, right } => Expression::OuterProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::SetOperation { op, left, right } => Expression::SetOperation {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::WedgeProduct { left, right } => Expression::WedgeProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => Expression::SetRelationExpr {
            relation: *relation,
            element: Box::new(f(element)),
            set: Box::new(f(set)),
        },
        Expression::FunctionSignature {
            name,
            domain,
            codomain,
        } => Expression::FunctionSignature {
            name: name.clone(),
            domain: Box::new(f(domain)),
            codomain: Box::new(f(codomain)),
        },
        Expression::Composition { outer, inner } => Expression::Composition {
            outer: Box::new(f(outer)),
            inner: Box::new(f(inner)),
        },
        Expression::Relation { op, left, right } => Expression::Relation {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },

        // ── Four children ───────────────────────────────────────────────
        Expression::Quaternion { real, i, j, k } => Expression::Quaternion {
            real: Box::new(f(real)),
            i: Box::new(f(i)),
            j: Box::new(f(j)),
            k: Box::new(f(k)),
        },

        // ── Variable-length children ────────────────────────────────────
        Expression::Function { name, args } => Expression::Function {
            name: name.clone(),
            args: args.iter().map(|a| f(a)).collect(),
        },
        Expression::Vector(elems) => Expression::Vector(elems.iter().map(|e| f(e)).collect()),
        Expression::Matrix(rows) => Expression::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|e| f(e)).collect())
                .collect(),
        ),
        Expression::Logical { op, operands } => Expression::Logical {
            op: *op,
            operands: operands.iter().map(|o| f(o)).collect(),
        },

        // ── Special: children + optional bounds ─────────────────────────
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => Expression::Integral {
            integrand: Box::new(f(integrand)),
            var: var.clone(),
            bounds: bounds.as_ref().map(|b| IntegralBounds {
                lower: Box::new(f(&b.lower)),
                upper: Box::new(f(&b.upper)),
            }),
        },
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => Expression::MultipleIntegral {
            dimension: *dimension,
            integrand: Box::new(f(integrand)),
            bounds: bounds.as_ref().map(|b| MultipleBounds {
                bounds: b
                    .bounds
                    .iter()
                    .map(|ib| IntegralBounds {
                        lower: Box::new(f(&ib.lower)),
                        upper: Box::new(f(&ib.upper)),
                    })
                    .collect(),
            }),
            vars: vars.clone(),
        },
        Expression::Limit {
            expr: e,
            var,
            to,
            direction,
        } => Expression::Limit {
            expr: Box::new(f(e)),
            var: var.clone(),
            to: Box::new(f(to)),
            direction: *direction,
        },
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => Expression::Sum {
            index: index.clone(),
            lower: Box::new(f(lower)),
            upper: Box::new(f(upper)),
            body: Box::new(f(body)),
        },
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => Expression::Product {
            index: index.clone(),
            lower: Box::new(f(lower)),
            upper: Box::new(f(upper)),
            body: Box::new(f(body)),
        },
        Expression::ForAll {
            variable,
            domain,
            body,
        } => Expression::ForAll {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            body: Box::new(f(body)),
        },
        Expression::Exists {
            variable,
            domain,
            body,
            unique,
        } => Expression::Exists {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            body: Box::new(f(body)),
            unique: *unique,
        },
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => Expression::SetBuilder {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            predicate: Box::new(f(predicate)),
        },
    }
}
