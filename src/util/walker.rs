//! Generic expression tree traversal and mapping.
//!
//! Provides `for_each_child` (read-only visitor) and `map_children`
//! (structural reconstruction) over all ~50 Expression variants,
//! eliminating the need for each utility to implement its own
//! exhaustive match scaffold.

use crate::ast::{ExprKind, Expression, IntegralBounds, MultipleBounds};

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
    match &expr.kind {
        // ── Leaf nodes ──────────────────────────────────────────────────
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Variable(_)
        | ExprKind::Constant(_)
        | ExprKind::MarkedVector { .. }
        | ExprKind::NumberSetExpr(_)
        | ExprKind::EmptySet
        | ExprKind::Nabla
        | ExprKind::Differential { .. } => {}

        // Tensor-like leaves: indices are strings, not Expressions
        ExprKind::Tensor { .. } | ExprKind::KroneckerDelta { .. } | ExprKind::LeviCivita { .. } => {
        }

        // ── One child ───────────────────────────────────────────────────
        ExprKind::Unary { operand, .. } => f(operand),
        ExprKind::Derivative { expr: e, .. } | ExprKind::PartialDerivative { expr: e, .. } => f(e),
        ExprKind::ClosedIntegral { integrand, .. } => f(integrand),
        ExprKind::Gradient { expr: e } | ExprKind::Laplacian { expr: e } => f(e),
        ExprKind::Divergence { field } | ExprKind::Curl { field } => f(field),
        ExprKind::Determinant { matrix }
        | ExprKind::Trace { matrix }
        | ExprKind::Rank { matrix }
        | ExprKind::ConjugateTranspose { matrix }
        | ExprKind::MatrixInverse { matrix } => f(matrix),
        ExprKind::PowerSet { set } => f(set),

        // ── Two children ────────────────────────────────────────────────
        ExprKind::Rational {
            numerator,
            denominator,
        } => {
            f(numerator);
            f(denominator);
        }
        ExprKind::Complex { real, imaginary } => {
            f(real);
            f(imaginary);
        }
        ExprKind::Binary { left, right, .. }
        | ExprKind::Equation { left, right }
        | ExprKind::Inequality { left, right, .. }
        | ExprKind::DotProduct { left, right }
        | ExprKind::CrossProduct { left, right }
        | ExprKind::OuterProduct { left, right }
        | ExprKind::SetOperation { left, right, .. }
        | ExprKind::WedgeProduct { left, right } => {
            f(left);
            f(right);
        }
        ExprKind::SetRelationExpr { element, set, .. } => {
            f(element);
            f(set);
        }
        ExprKind::FunctionSignature {
            domain, codomain, ..
        } => {
            f(domain);
            f(codomain);
        }
        ExprKind::Composition { outer, inner } => {
            f(outer);
            f(inner);
        }
        ExprKind::Relation { left, right, .. } => {
            f(left);
            f(right);
        }

        // ── Four children ───────────────────────────────────────────────
        ExprKind::Quaternion { real, i, j, k } => {
            f(real);
            f(i);
            f(j);
            f(k);
        }

        // ── Variable-length children ────────────────────────────────────
        ExprKind::Function { args, .. } => {
            for a in args {
                f(a);
            }
        }
        ExprKind::Vector(elems) => {
            for e in elems {
                f(e);
            }
        }
        ExprKind::Matrix(rows) => {
            for row in rows {
                for e in row {
                    f(e);
                }
            }
        }
        ExprKind::Logical { operands, .. } => {
            for o in operands {
                f(o);
            }
        }

        // ── Special: children + optional bounds ─────────────────────────
        ExprKind::Integral {
            integrand, bounds, ..
        } => {
            f(integrand);
            if let Some(b) = bounds {
                f(&b.lower);
                f(&b.upper);
            }
        }
        ExprKind::MultipleIntegral {
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
        ExprKind::Limit { expr: e, to, .. } => {
            f(e);
            f(to);
        }
        ExprKind::Sum {
            lower, upper, body, ..
        }
        | ExprKind::Product {
            lower, upper, body, ..
        } => {
            f(lower);
            f(upper);
            f(body);
        }
        ExprKind::ForAll { domain, body, .. } | ExprKind::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                f(d);
            }
            f(body);
        }
        ExprKind::SetBuilder {
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
    let kind: ExprKind = match &expr.kind {
        // ── Leaf nodes (no children to map) ─────────────────────────────
        ExprKind::Integer(_)
        | ExprKind::Float(_)
        | ExprKind::Variable(_)
        | ExprKind::Constant(_)
        | ExprKind::MarkedVector { .. }
        | ExprKind::NumberSetExpr(_)
        | ExprKind::EmptySet
        | ExprKind::Nabla
        | ExprKind::Differential { .. }
        | ExprKind::Tensor { .. }
        | ExprKind::KroneckerDelta { .. }
        | ExprKind::LeviCivita { .. } => expr.kind.clone(),

        // ── One child ───────────────────────────────────────────────────
        ExprKind::Unary { op, operand } => ExprKind::Unary {
            op: *op,
            operand: Box::new(f(operand)),
        },
        ExprKind::Derivative {
            expr: e,
            var,
            order,
        } => ExprKind::Derivative {
            expr: Box::new(f(e)),
            var: var.clone(),
            order: *order,
        },
        ExprKind::PartialDerivative {
            expr: e,
            var,
            order,
        } => ExprKind::PartialDerivative {
            expr: Box::new(f(e)),
            var: var.clone(),
            order: *order,
        },
        ExprKind::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => ExprKind::ClosedIntegral {
            dimension: *dimension,
            integrand: Box::new(f(integrand)),
            surface: surface.clone(),
            var: var.clone(),
        },
        ExprKind::Gradient { expr: e } => ExprKind::Gradient {
            expr: Box::new(f(e)),
        },
        ExprKind::Laplacian { expr: e } => ExprKind::Laplacian {
            expr: Box::new(f(e)),
        },
        ExprKind::Divergence { field } => ExprKind::Divergence {
            field: Box::new(f(field)),
        },
        ExprKind::Curl { field } => ExprKind::Curl {
            field: Box::new(f(field)),
        },
        ExprKind::Determinant { matrix } => ExprKind::Determinant {
            matrix: Box::new(f(matrix)),
        },
        ExprKind::Trace { matrix } => ExprKind::Trace {
            matrix: Box::new(f(matrix)),
        },
        ExprKind::Rank { matrix } => ExprKind::Rank {
            matrix: Box::new(f(matrix)),
        },
        ExprKind::ConjugateTranspose { matrix } => ExprKind::ConjugateTranspose {
            matrix: Box::new(f(matrix)),
        },
        ExprKind::MatrixInverse { matrix } => ExprKind::MatrixInverse {
            matrix: Box::new(f(matrix)),
        },
        ExprKind::PowerSet { set } => ExprKind::PowerSet {
            set: Box::new(f(set)),
        },

        // ── Two children ────────────────────────────────────────────────
        ExprKind::Rational {
            numerator,
            denominator,
        } => ExprKind::Rational {
            numerator: Box::new(f(numerator)),
            denominator: Box::new(f(denominator)),
        },
        ExprKind::Complex { real, imaginary } => ExprKind::Complex {
            real: Box::new(f(real)),
            imaginary: Box::new(f(imaginary)),
        },
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::Equation { left, right } => ExprKind::Equation {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::Inequality { op, left, right } => ExprKind::Inequality {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::DotProduct { left, right } => ExprKind::DotProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::CrossProduct { left, right } => ExprKind::CrossProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::OuterProduct { left, right } => ExprKind::OuterProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::SetOperation { op, left, right } => ExprKind::SetOperation {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::WedgeProduct { left, right } => ExprKind::WedgeProduct {
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },
        ExprKind::SetRelationExpr {
            relation,
            element,
            set,
        } => ExprKind::SetRelationExpr {
            relation: *relation,
            element: Box::new(f(element)),
            set: Box::new(f(set)),
        },
        ExprKind::FunctionSignature {
            name,
            domain,
            codomain,
        } => ExprKind::FunctionSignature {
            name: name.clone(),
            domain: Box::new(f(domain)),
            codomain: Box::new(f(codomain)),
        },
        ExprKind::Composition { outer, inner } => ExprKind::Composition {
            outer: Box::new(f(outer)),
            inner: Box::new(f(inner)),
        },
        ExprKind::Relation { op, left, right } => ExprKind::Relation {
            op: *op,
            left: Box::new(f(left)),
            right: Box::new(f(right)),
        },

        // ── Four children ───────────────────────────────────────────────
        ExprKind::Quaternion { real, i, j, k } => ExprKind::Quaternion {
            real: Box::new(f(real)),
            i: Box::new(f(i)),
            j: Box::new(f(j)),
            k: Box::new(f(k)),
        },

        // ── Variable-length children ────────────────────────────────────
        ExprKind::Function { name, args } => ExprKind::Function {
            name: name.clone(),
            args: args.iter().map(|a| f(a)).collect(),
        },
        ExprKind::Vector(elems) => ExprKind::Vector(elems.iter().map(|e| f(e)).collect()),
        ExprKind::Matrix(rows) => ExprKind::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|e| f(e)).collect())
                .collect(),
        ),
        ExprKind::Logical { op, operands } => ExprKind::Logical {
            op: *op,
            operands: operands.iter().map(|o| f(o)).collect(),
        },

        // ── Special: children + optional bounds ─────────────────────────
        ExprKind::Integral {
            integrand,
            var,
            bounds,
        } => ExprKind::Integral {
            integrand: Box::new(f(integrand)),
            var: var.clone(),
            bounds: bounds.as_ref().map(|b| IntegralBounds {
                lower: Box::new(f(&b.lower)),
                upper: Box::new(f(&b.upper)),
            }),
        },
        ExprKind::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => ExprKind::MultipleIntegral {
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
        ExprKind::Limit {
            expr: e,
            var,
            to,
            direction,
        } => ExprKind::Limit {
            expr: Box::new(f(e)),
            var: var.clone(),
            to: Box::new(f(to)),
            direction: *direction,
        },
        ExprKind::Sum {
            index,
            lower,
            upper,
            body,
        } => ExprKind::Sum {
            index: index.clone(),
            lower: Box::new(f(lower)),
            upper: Box::new(f(upper)),
            body: Box::new(f(body)),
        },
        ExprKind::Product {
            index,
            lower,
            upper,
            body,
        } => ExprKind::Product {
            index: index.clone(),
            lower: Box::new(f(lower)),
            upper: Box::new(f(upper)),
            body: Box::new(f(body)),
        },
        ExprKind::ForAll {
            variable,
            domain,
            body,
        } => ExprKind::ForAll {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            body: Box::new(f(body)),
        },
        ExprKind::Exists {
            variable,
            domain,
            body,
            unique,
        } => ExprKind::Exists {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            body: Box::new(f(body)),
            unique: *unique,
        },
        ExprKind::SetBuilder {
            variable,
            domain,
            predicate,
        } => ExprKind::SetBuilder {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(f(d))),
            predicate: Box::new(f(predicate)),
        },
    };
    kind.into()
}
