//! Variable substitution helpers.

use crate::ast::{ExprKind, Expression, IntegralBounds, MultipleBounds};

use super::walker::map_children;

// ── substitute_with helpers ──────────────────────────────────────────────────

fn sub_tensor_index(
    indices: &[crate::ast::TensorIndex],
    lookup: &impl Fn(&str) -> Option<Expression>,
) -> Vec<crate::ast::TensorIndex> {
    indices
        .iter()
        .map(|idx| match lookup(&idx.name).map(|e| e.kind) {
            Some(ExprKind::Variable(new_name)) => crate::ast::TensorIndex {
                name: new_name,
                index_type: idx.index_type,
            },
            _ => idx.clone(),
        })
        .collect()
}

/// Recursively substitute through `expr`, respecting bound-variable scoping.
///
/// Variants that bind a variable (Derivative, Integral, Sum, etc.) skip
/// substitution in the scope of that binding. All other variants delegate
/// to `map_children` for structural recursion.
fn sw_core(expr: &Expression, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
    let recurse = |e: &Expression| e.substitute_with(lookup);

    match &expr.kind {
        // ── Variable: the substitution point ────────────────────────────
        ExprKind::Variable(name) => lookup(name).unwrap_or_else(|| expr.clone()),

        // ── Bound-variable scoping: skip body when var is being substituted ──
        ExprKind::Derivative {
            expr: e,
            var,
            order,
        } => ExprKind::Derivative {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            order: *order,
        }
        .into(),
        ExprKind::PartialDerivative {
            expr: e,
            var,
            order,
        } => ExprKind::PartialDerivative {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            order: *order,
        }
        .into(),
        ExprKind::Integral {
            integrand,
            var,
            bounds,
        } => ExprKind::Integral {
            integrand: if lookup(var).is_some() {
                integrand.clone()
            } else {
                Box::new(recurse(integrand))
            },
            var: var.clone(),
            bounds: bounds.as_ref().map(|b| IntegralBounds {
                lower: Box::new(recurse(&b.lower)),
                upper: Box::new(recurse(&b.upper)),
            }),
        }
        .into(),
        ExprKind::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            let is_bound = vars.iter().any(|v| lookup(v).is_some());
            ExprKind::MultipleIntegral {
                dimension: *dimension,
                integrand: if is_bound {
                    integrand.clone()
                } else {
                    Box::new(recurse(integrand))
                },
                bounds: bounds.as_ref().map(|b| MultipleBounds {
                    bounds: b
                        .bounds
                        .iter()
                        .map(|ib| IntegralBounds {
                            lower: Box::new(recurse(&ib.lower)),
                            upper: Box::new(recurse(&ib.upper)),
                        })
                        .collect(),
                }),
                vars: vars.clone(),
            }
            .into()
        }
        ExprKind::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => ExprKind::ClosedIntegral {
            dimension: *dimension,
            integrand: if lookup(var).is_some() {
                integrand.clone()
            } else {
                Box::new(recurse(integrand))
            },
            surface: surface.clone(),
            var: var.clone(),
        }
        .into(),
        ExprKind::Limit {
            expr: e,
            var,
            to,
            direction,
        } => ExprKind::Limit {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            to: Box::new(recurse(to)),
            direction: *direction,
        }
        .into(),
        ExprKind::Sum {
            index,
            lower,
            upper,
            body,
        } => ExprKind::Sum {
            index: index.clone(),
            lower: Box::new(recurse(lower)),
            upper: Box::new(recurse(upper)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        }
        .into(),
        ExprKind::Product {
            index,
            lower,
            upper,
            body,
        } => ExprKind::Product {
            index: index.clone(),
            lower: Box::new(recurse(lower)),
            upper: Box::new(recurse(upper)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        }
        .into(),
        ExprKind::ForAll {
            variable,
            domain,
            body,
        } => ExprKind::ForAll {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            body: if lookup(variable).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        }
        .into(),
        ExprKind::Exists {
            variable,
            domain,
            body,
            unique,
        } => ExprKind::Exists {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            body: if lookup(variable).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
            unique: *unique,
        }
        .into(),
        ExprKind::SetBuilder {
            variable,
            domain,
            predicate,
        } => ExprKind::SetBuilder {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            predicate: if lookup(variable).is_some() {
                predicate.clone()
            } else {
                Box::new(recurse(predicate))
            },
        }
        .into(),

        // ── Tensor indices: substitute names, not child expressions ──────
        ExprKind::Tensor { name, indices } => ExprKind::Tensor {
            name: name.clone(),
            indices: sub_tensor_index(indices, lookup),
        }
        .into(),
        ExprKind::KroneckerDelta { indices } => ExprKind::KroneckerDelta {
            indices: sub_tensor_index(indices, lookup),
        }
        .into(),
        ExprKind::LeviCivita { indices } => ExprKind::LeviCivita {
            indices: sub_tensor_index(indices, lookup),
        }
        .into(),
        ExprKind::Differential { var } => match lookup(var).map(|e| e.kind) {
            Some(ExprKind::Variable(new_name)) => ExprKind::Differential { var: new_name }.into(),
            _ => expr.clone(),
        },

        // ── Everything else: structural recursion via map_children ───────
        _ => map_children(expr, &mut |e| e.substitute_with(lookup)),
    }
}

// ── impl Expression ──────────────────────────────────────────────────────────

impl Expression {
    /// Substitutes all occurrences of a variable with a replacement expression.
    ///
    /// Respects bound variable scoping rules. Bound variables in calculus and
    /// iterator constructs are not substituted within their scope.
    pub fn substitute(&self, var: &str, replacement: &Expression) -> Expression {
        self.substitute_with(&|name| {
            if name == var {
                Some(replacement.clone())
            } else {
                None
            }
        })
    }

    /// Substitutes multiple variables simultaneously with replacement expressions.
    ///
    /// Same scoping rules as [`substitute`](Expression::substitute) apply.
    pub fn substitute_all(
        &self,
        subs: &std::collections::HashMap<String, Expression>,
    ) -> Expression {
        self.substitute_with(&|name| subs.get(name).cloned())
    }

    pub(crate) fn substitute_with(
        &self,
        lookup: &impl Fn(&str) -> Option<Expression>,
    ) -> Expression {
        sw_core(self, lookup)
    }
}
