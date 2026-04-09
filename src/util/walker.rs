//! Generic expression tree traversal.
//!
//! Provides `for_each_child` which visits all child `Expression` nodes of a
//! given expression variant, eliminating the need for each utility to
//! implement its own exhaustive match over all ~50 variants.

use crate::ast::Expression;

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
