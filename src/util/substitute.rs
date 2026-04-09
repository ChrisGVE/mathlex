//! Variable substitution helpers.

use crate::ast::{Expression, IntegralBounds, MultipleBounds};

use super::walker::map_children;

// ── substitute_with helpers ──────────────────────────────────────────────────

fn sub_tensor_index(
    indices: &[crate::ast::TensorIndex],
    lookup: &impl Fn(&str) -> Option<Expression>,
) -> Vec<crate::ast::TensorIndex> {
    indices
        .iter()
        .map(|idx| match lookup(&idx.name) {
            Some(Expression::Variable(new_name)) => crate::ast::TensorIndex {
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

    match expr {
        // ── Variable: the substitution point ────────────────────────────
        Expression::Variable(name) => lookup(name).unwrap_or_else(|| expr.clone()),

        // ── Bound-variable scoping: skip body when var is being substituted ──
        Expression::Derivative {
            expr: e,
            var,
            order,
        } => Expression::Derivative {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            order: *order,
        },
        Expression::PartialDerivative {
            expr: e,
            var,
            order,
        } => Expression::PartialDerivative {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            order: *order,
        },
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => Expression::Integral {
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
        },
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            let is_bound = vars.iter().any(|v| lookup(v).is_some());
            Expression::MultipleIntegral {
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
        }
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => Expression::ClosedIntegral {
            dimension: *dimension,
            integrand: if lookup(var).is_some() {
                integrand.clone()
            } else {
                Box::new(recurse(integrand))
            },
            surface: surface.clone(),
            var: var.clone(),
        },
        Expression::Limit {
            expr: e,
            var,
            to,
            direction,
        } => Expression::Limit {
            expr: if lookup(var).is_some() {
                e.clone()
            } else {
                Box::new(recurse(e))
            },
            var: var.clone(),
            to: Box::new(recurse(to)),
            direction: *direction,
        },
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => Expression::Sum {
            index: index.clone(),
            lower: Box::new(recurse(lower)),
            upper: Box::new(recurse(upper)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        },
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => Expression::Product {
            index: index.clone(),
            lower: Box::new(recurse(lower)),
            upper: Box::new(recurse(upper)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        },
        Expression::ForAll {
            variable,
            domain,
            body,
        } => Expression::ForAll {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            body: if lookup(variable).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
        },
        Expression::Exists {
            variable,
            domain,
            body,
            unique,
        } => Expression::Exists {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            body: if lookup(variable).is_some() {
                body.clone()
            } else {
                Box::new(recurse(body))
            },
            unique: *unique,
        },
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => Expression::SetBuilder {
            variable: variable.clone(),
            domain: domain.as_ref().map(|d| Box::new(recurse(d))),
            predicate: if lookup(variable).is_some() {
                predicate.clone()
            } else {
                Box::new(recurse(predicate))
            },
        },

        // ── Tensor indices: substitute names, not child expressions ──────
        Expression::Tensor { name, indices } => Expression::Tensor {
            name: name.clone(),
            indices: sub_tensor_index(indices, lookup),
        },
        Expression::KroneckerDelta { indices } => Expression::KroneckerDelta {
            indices: sub_tensor_index(indices, lookup),
        },
        Expression::LeviCivita { indices } => Expression::LeviCivita {
            indices: sub_tensor_index(indices, lookup),
        },
        Expression::Differential { var } => match lookup(var) {
            Some(Expression::Variable(new_name)) => Expression::Differential { var: new_name },
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
