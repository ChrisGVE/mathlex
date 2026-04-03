//! Variable substitution helpers.

use crate::ast::Expression;

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

fn sw_core(expr: &Expression, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => expr.clone(),
        Expression::Variable(name) => lookup(name).unwrap_or_else(|| expr.clone()),
        Expression::Rational {
            numerator,
            denominator,
        } => Expression::Rational {
            numerator: Box::new(numerator.substitute_with(lookup)),
            denominator: Box::new(denominator.substitute_with(lookup)),
        },
        Expression::Complex { real, imaginary } => Expression::Complex {
            real: Box::new(real.substitute_with(lookup)),
            imaginary: Box::new(imaginary.substitute_with(lookup)),
        },
        Expression::Quaternion {
            real,
            i: qi,
            j: qj,
            k: qk,
        } => Expression::Quaternion {
            real: Box::new(real.substitute_with(lookup)),
            i: Box::new(qi.substitute_with(lookup)),
            j: Box::new(qj.substitute_with(lookup)),
            k: Box::new(qk.substitute_with(lookup)),
        },
        Expression::Binary { op, left, right } => Expression::Binary {
            op: *op,
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::Equation { left, right } => Expression::Equation {
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::Inequality { op, left, right } => Expression::Inequality {
            op: *op,
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::Unary { op, operand } => Expression::Unary {
            op: *op,
            operand: Box::new(operand.substitute_with(lookup)),
        },
        Expression::Function { name, args } => Expression::Function {
            name: name.clone(),
            args: args.iter().map(|a| a.substitute_with(lookup)).collect(),
        },
        _ => sw_calculus(expr, lookup),
    }
}

fn sw_calculus(expr: &Expression, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
    match expr {
        Expression::Derivative {
            expr: e,
            var: dv,
            order,
        } => {
            if lookup(dv).is_some() {
                expr.clone()
            } else {
                Expression::Derivative {
                    expr: Box::new(e.substitute_with(lookup)),
                    var: dv.clone(),
                    order: *order,
                }
            }
        }
        Expression::PartialDerivative {
            expr: e,
            var: dv,
            order,
        } => {
            if lookup(dv).is_some() {
                expr.clone()
            } else {
                Expression::PartialDerivative {
                    expr: Box::new(e.substitute_with(lookup)),
                    var: dv.clone(),
                    order: *order,
                }
            }
        }
        Expression::Integral {
            integrand,
            var: iv,
            bounds,
        } => {
            let new_bounds = bounds.as_ref().map(|b| crate::ast::IntegralBounds {
                lower: Box::new(b.lower.substitute_with(lookup)),
                upper: Box::new(b.upper.substitute_with(lookup)),
            });
            Expression::Integral {
                integrand: if lookup(iv).is_some() {
                    integrand.clone()
                } else {
                    Box::new(integrand.substitute_with(lookup))
                },
                var: iv.clone(),
                bounds: new_bounds,
            }
        }
        _ => sw_integrals(expr, lookup),
    }
}

fn sw_integrals(expr: &Expression, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
    match expr {
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => {
            let is_bound = vars.iter().any(|v| lookup(v).is_some());
            let new_bounds = bounds.as_ref().map(|b| crate::ast::MultipleBounds {
                bounds: b
                    .bounds
                    .iter()
                    .map(|ib| crate::ast::IntegralBounds {
                        lower: Box::new(ib.lower.substitute_with(lookup)),
                        upper: Box::new(ib.upper.substitute_with(lookup)),
                    })
                    .collect(),
            });
            Expression::MultipleIntegral {
                dimension: *dimension,
                integrand: if is_bound {
                    integrand.clone()
                } else {
                    Box::new(integrand.substitute_with(lookup))
                },
                bounds: new_bounds,
                vars: vars.clone(),
            }
        }
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var: iv,
        } => Expression::ClosedIntegral {
            dimension: *dimension,
            integrand: if lookup(iv).is_some() {
                integrand.clone()
            } else {
                Box::new(integrand.substitute_with(lookup))
            },
            surface: surface.clone(),
            var: iv.clone(),
        },
        Expression::Limit {
            expr: e,
            var: lv,
            to,
            direction,
        } => Expression::Limit {
            expr: if lookup(lv).is_some() {
                e.clone()
            } else {
                Box::new(e.substitute_with(lookup))
            },
            var: lv.clone(),
            to: Box::new(to.substitute_with(lookup)),
            direction: *direction,
        },
        _ => sw_iter_containers(expr, lookup),
    }
}

fn sw_iter_containers(
    expr: &Expression,
    lookup: &impl Fn(&str) -> Option<Expression>,
) -> Expression {
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => Expression::Sum {
            index: index.clone(),
            lower: Box::new(lower.substitute_with(lookup)),
            upper: Box::new(upper.substitute_with(lookup)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(body.substitute_with(lookup))
            },
        },
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => Expression::Product {
            index: index.clone(),
            lower: Box::new(lower.substitute_with(lookup)),
            upper: Box::new(upper.substitute_with(lookup)),
            body: if lookup(index).is_some() {
                body.clone()
            } else {
                Box::new(body.substitute_with(lookup))
            },
        },
        Expression::Vector(elems) => {
            Expression::Vector(elems.iter().map(|e| e.substitute_with(lookup)).collect())
        }
        Expression::Matrix(rows) => Expression::Matrix(
            rows.iter()
                .map(|row| row.iter().map(|e| e.substitute_with(lookup)).collect())
                .collect(),
        ),
        _ => sw_logic_sets(expr, lookup),
    }
}

fn sw_logic_sets(expr: &Expression, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
    match expr {
        Expression::ForAll {
            variable: bv,
            domain,
            body,
        } => Expression::ForAll {
            variable: bv.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.substitute_with(lookup))),
            body: if lookup(bv).is_some() {
                body.clone()
            } else {
                Box::new(body.substitute_with(lookup))
            },
        },
        Expression::Exists {
            variable: bv,
            domain,
            body,
            unique,
        } => Expression::Exists {
            variable: bv.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.substitute_with(lookup))),
            body: if lookup(bv).is_some() {
                body.clone()
            } else {
                Box::new(body.substitute_with(lookup))
            },
            unique: *unique,
        },
        Expression::Logical { op, operands } => Expression::Logical {
            op: *op,
            operands: operands.iter().map(|e| e.substitute_with(lookup)).collect(),
        },
        Expression::MarkedVector { .. } => expr.clone(),
        Expression::DotProduct { left, right } => Expression::DotProduct {
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::CrossProduct { left, right } => Expression::CrossProduct {
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::OuterProduct { left, right } => Expression::OuterProduct {
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::Gradient { expr: e } => Expression::Gradient {
            expr: Box::new(e.substitute_with(lookup)),
        },
        Expression::Divergence { field } => Expression::Divergence {
            field: Box::new(field.substitute_with(lookup)),
        },
        Expression::Curl { field } => Expression::Curl {
            field: Box::new(field.substitute_with(lookup)),
        },
        Expression::Laplacian { expr: e } => Expression::Laplacian {
            expr: Box::new(e.substitute_with(lookup)),
        },
        Expression::Nabla => Expression::Nabla,
        _ => sw_linalg_sets_theory(expr, lookup),
    }
}

fn sw_linalg_sets_theory(
    expr: &Expression,
    lookup: &impl Fn(&str) -> Option<Expression>,
) -> Expression {
    match expr {
        Expression::Determinant { matrix } => Expression::Determinant {
            matrix: Box::new(matrix.substitute_with(lookup)),
        },
        Expression::Trace { matrix } => Expression::Trace {
            matrix: Box::new(matrix.substitute_with(lookup)),
        },
        Expression::Rank { matrix } => Expression::Rank {
            matrix: Box::new(matrix.substitute_with(lookup)),
        },
        Expression::ConjugateTranspose { matrix } => Expression::ConjugateTranspose {
            matrix: Box::new(matrix.substitute_with(lookup)),
        },
        Expression::MatrixInverse { matrix } => Expression::MatrixInverse {
            matrix: Box::new(matrix.substitute_with(lookup)),
        },
        Expression::NumberSetExpr(_) | Expression::EmptySet => expr.clone(),
        Expression::SetOperation { op, left, right } => Expression::SetOperation {
            op: *op,
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => Expression::SetRelationExpr {
            relation: *relation,
            element: Box::new(element.substitute_with(lookup)),
            set: Box::new(set.substitute_with(lookup)),
        },
        Expression::SetBuilder {
            variable: bv,
            domain,
            predicate,
        } => Expression::SetBuilder {
            variable: bv.clone(),
            domain: domain.as_ref().map(|d| Box::new(d.substitute_with(lookup))),
            predicate: if lookup(bv).is_some() {
                predicate.clone()
            } else {
                Box::new(predicate.substitute_with(lookup))
            },
        },
        Expression::PowerSet { set } => Expression::PowerSet {
            set: Box::new(set.substitute_with(lookup)),
        },
        _ => sw_tensors_forms_theory(expr, lookup),
    }
}

fn sw_tensors_forms_theory(
    expr: &Expression,
    lookup: &impl Fn(&str) -> Option<Expression>,
) -> Expression {
    match expr {
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
        Expression::WedgeProduct { left, right } => Expression::WedgeProduct {
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        Expression::FunctionSignature {
            name,
            domain,
            codomain,
        } => Expression::FunctionSignature {
            name: name.clone(),
            domain: Box::new(domain.substitute_with(lookup)),
            codomain: Box::new(codomain.substitute_with(lookup)),
        },
        Expression::Composition { outer, inner } => Expression::Composition {
            outer: Box::new(outer.substitute_with(lookup)),
            inner: Box::new(inner.substitute_with(lookup)),
        },
        Expression::Relation { op, left, right } => Expression::Relation {
            op: *op,
            left: Box::new(left.substitute_with(lookup)),
            right: Box::new(right.substitute_with(lookup)),
        },
        _ => expr.clone(),
    }
}

// ── impl Expression ──────────────────────────────────────────────────────────

impl Expression {
    /// Substitutes all occurrences of a variable with a replacement expression.
    ///
    /// This method performs variable substitution throughout the expression tree,
    /// respecting bound variable scoping rules. Bound variables in calculus and
    /// iterator constructs are not substituted within their scope.
    ///
    /// # Bound Variables
    ///
    /// The following constructs introduce bound variables that are NOT substituted
    /// within their scope:
    ///
    /// - `Sum { index, ... }` - `index` is bound in `body`
    /// - `Product { index, ... }` - `index` is bound in `body`
    /// - `Integral { var, ... }` - `var` is bound in `integrand`
    /// - `Limit { var, ... }` - `var` is bound in `expr`
    /// - `Derivative { var, ... }` - `var` is bound in `expr`
    /// - `PartialDerivative { var, ... }` - `var` is bound in `expr`
    ///
    /// However, variables in bounds and limits (e.g., `lower`, `upper`, `to`)
    /// are still substituted since they are outside the binding scope.
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
    /// This method performs simultaneous substitution of multiple variables,
    /// respecting bound variable scoping rules. The substitutions are applied
    /// simultaneously, meaning that replacements don't affect each other.
    ///
    /// # Bound Variables
    ///
    /// Same scoping rules as [`substitute`](Expression::substitute) apply.
    /// See that method's documentation for details on bound variables.
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
