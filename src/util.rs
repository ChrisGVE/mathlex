//! # AST Utility Functions
//!
//! This module provides utility methods for querying and analyzing the AST.
//! These methods allow consumers to extract information about expressions
//! without manually traversing the tree structure.
//!
//! ## Key Features
//!
//! - **Variable Discovery**: Find all variables in an expression
//! - **Function Discovery**: Find all function calls in an expression
//! - **Constant Discovery**: Find all mathematical constants used
//! - **Tree Metrics**: Calculate depth and node count
//!
//! ## Examples
//!
//! ```
//! use mathlex::ast::{Expression, BinaryOp, MathConstant};
//!
//! // Create expression: 2 * π * x
//! let expr = Expression::Binary {
//!     op: BinaryOp::Mul,
//!     left: Box::new(Expression::Binary {
//!         op: BinaryOp::Mul,
//!         left: Box::new(Expression::Integer(2)),
//!         right: Box::new(Expression::Constant(MathConstant::Pi)),
//!     }),
//!     right: Box::new(Expression::Variable("x".to_string())),
//! };
//!
//! // Query the expression
//! assert_eq!(expr.find_variables().len(), 1); // {x}
//! assert_eq!(expr.find_constants().len(), 1); // {π}
//! assert_eq!(expr.depth(), 3);
//! assert_eq!(expr.node_count(), 5);
//! ```

use crate::ast::{Expression, MathConstant};
use std::collections::HashSet;

// ── collect_variables helpers ────────────────────────────────────────────────

fn cv_core(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Constant(_) => {}
        Expression::Variable(name) => {
            vars.insert(name.clone());
        }
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
        } => {
            l.collect_variables(vars);
            r.collect_variables(vars);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_variables(vars);
            i.collect_variables(vars);
            j.collect_variables(vars);
            k.collect_variables(vars);
        }
        Expression::Unary { operand, .. } => operand.collect_variables(vars),
        Expression::Function { args, .. } => {
            for a in args {
                a.collect_variables(vars);
            }
        }
        _ => cv_calculus_and_rest(expr, vars),
    }
}

fn cv_calculus_and_rest(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::Derivative { expr: e, var, .. }
        | Expression::PartialDerivative { expr: e, var, .. } => {
            e.collect_variables(vars);
            vars.insert(var.clone());
        }
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            integrand.collect_variables(vars);
            vars.insert(var.clone());
            if let Some(b) = bounds {
                b.lower.collect_variables(vars);
                b.upper.collect_variables(vars);
            }
        }
        Expression::MultipleIntegral {
            integrand,
            vars: ivars,
            bounds,
            ..
        } => {
            integrand.collect_variables(vars);
            for v in ivars {
                vars.insert(v.clone());
            }
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_variables(vars);
                    ib.upper.collect_variables(vars);
                }
            }
        }
        Expression::ClosedIntegral { integrand, var, .. } => {
            integrand.collect_variables(vars);
            vars.insert(var.clone());
        }
        Expression::Limit {
            expr: e, var, to, ..
        } => {
            e.collect_variables(vars);
            vars.insert(var.clone());
            to.collect_variables(vars);
        }
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        }
        | Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            vars.insert(index.clone());
            lower.collect_variables(vars);
            upper.collect_variables(vars);
            body.collect_variables(vars);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_variables(vars);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_variables(vars);
                }
            }
        }
        _ => cv_logic_sets_and_rest(expr, vars),
    }
}

fn cv_logic_sets_and_rest(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::ForAll {
            variable,
            domain,
            body,
        }
        | Expression::Exists {
            variable,
            domain,
            body,
            ..
        } => {
            vars.insert(variable.clone());
            if let Some(d) = domain {
                d.collect_variables(vars);
            }
            body.collect_variables(vars);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_variables(vars);
            }
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla => {}
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            left.collect_variables(vars);
            right.collect_variables(vars);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_variables(vars);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_variables(vars);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_variables(vars);
        }
        _ => cv_sets_tensors_theory(expr, vars),
    }
}

fn cv_sets_tensors_theory(expr: &Expression, vars: &mut HashSet<String>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_variables(vars);
            set.collect_variables(vars);
        }
        Expression::SetBuilder {
            variable,
            domain,
            predicate,
        } => {
            vars.insert(variable.clone());
            if let Some(d) = domain {
                d.collect_variables(vars);
            }
            predicate.collect_variables(vars);
        }
        Expression::PowerSet { set } => set.collect_variables(vars),
        Expression::Tensor { indices, .. }
        | Expression::KroneckerDelta { indices }
        | Expression::LeviCivita { indices } => {
            for idx in indices {
                vars.insert(idx.name.clone());
            }
        }
        Expression::Differential { var } => {
            vars.insert(var.clone());
        }
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_variables(vars);
            codomain.collect_variables(vars);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_variables(vars);
            inner.collect_variables(vars);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_variables(vars);
            right.collect_variables(vars);
        }
        _ => {}
    }
}

// ── collect_functions helpers ────────────────────────────────────────────────

fn cf_core(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Variable(_)
        | Expression::Constant(_) => {}
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
        } => {
            l.collect_functions(fns);
            r.collect_functions(fns);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_functions(fns);
            i.collect_functions(fns);
            j.collect_functions(fns);
            k.collect_functions(fns);
        }
        Expression::Unary { operand, .. } => operand.collect_functions(fns),
        Expression::Function { name, args } => {
            fns.insert(name.clone());
            for a in args {
                a.collect_functions(fns);
            }
        }
        _ => cf_calculus_and_rest(expr, fns),
    }
}

fn cf_calculus_and_rest(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            e.collect_functions(fns)
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            integrand.collect_functions(fns);
            if let Some(b) = bounds {
                b.lower.collect_functions(fns);
                b.upper.collect_functions(fns);
            }
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            integrand.collect_functions(fns);
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_functions(fns);
                    ib.upper.collect_functions(fns);
                }
            }
        }
        Expression::ClosedIntegral { integrand, .. } => integrand.collect_functions(fns),
        Expression::Limit { expr: e, to, .. } => {
            e.collect_functions(fns);
            to.collect_functions(fns);
        }
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            lower.collect_functions(fns);
            upper.collect_functions(fns);
            body.collect_functions(fns);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_functions(fns);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_functions(fns);
                }
            }
        }
        _ => cf_logic_sets_and_rest(expr, fns),
    }
}

fn cf_logic_sets_and_rest(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                d.collect_functions(fns);
            }
            body.collect_functions(fns);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_functions(fns);
            }
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => {}
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            left.collect_functions(fns);
            right.collect_functions(fns);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_functions(fns);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_functions(fns);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_functions(fns);
        }
        _ => cf_sets_theory(expr, fns),
    }
}

fn cf_sets_theory(expr: &Expression, fns: &mut HashSet<String>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_functions(fns);
            set.collect_functions(fns);
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            if let Some(d) = domain {
                d.collect_functions(fns);
            }
            predicate.collect_functions(fns);
        }
        Expression::PowerSet { set } => set.collect_functions(fns),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_functions(fns);
            codomain.collect_functions(fns);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_functions(fns);
            inner.collect_functions(fns);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_functions(fns);
            right.collect_functions(fns);
        }
        _ => {}
    }
}

// ── collect_constants helpers ────────────────────────────────────────────────

fn cc_core(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::Integer(_) | Expression::Float(_) | Expression::Variable(_) => {}
        Expression::Constant(c) => {
            cs.insert(*c);
        }
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
        } => {
            l.collect_constants(cs);
            r.collect_constants(cs);
        }
        Expression::Quaternion { real, i, j, k } => {
            real.collect_constants(cs);
            i.collect_constants(cs);
            j.collect_constants(cs);
            k.collect_constants(cs);
        }
        Expression::Unary { operand, .. } => operand.collect_constants(cs),
        Expression::Function { args, .. } => {
            for a in args {
                a.collect_constants(cs);
            }
        }
        _ => cc_calculus_and_rest(expr, cs),
    }
}

fn cc_calculus_and_rest(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::Derivative { expr: e, .. } | Expression::PartialDerivative { expr: e, .. } => {
            e.collect_constants(cs)
        }
        Expression::Integral {
            integrand, bounds, ..
        } => {
            integrand.collect_constants(cs);
            if let Some(b) = bounds {
                b.lower.collect_constants(cs);
                b.upper.collect_constants(cs);
            }
        }
        Expression::MultipleIntegral {
            integrand, bounds, ..
        } => {
            integrand.collect_constants(cs);
            if let Some(b) = bounds {
                for ib in &b.bounds {
                    ib.lower.collect_constants(cs);
                    ib.upper.collect_constants(cs);
                }
            }
        }
        Expression::ClosedIntegral { integrand, .. } => integrand.collect_constants(cs),
        Expression::Limit { expr: e, to, .. } => {
            e.collect_constants(cs);
            to.collect_constants(cs);
        }
        Expression::Sum {
            lower, upper, body, ..
        }
        | Expression::Product {
            lower, upper, body, ..
        } => {
            lower.collect_constants(cs);
            upper.collect_constants(cs);
            body.collect_constants(cs);
        }
        Expression::Vector(elems) => {
            for e in elems {
                e.collect_constants(cs);
            }
        }
        Expression::Matrix(rows) => {
            for row in rows {
                for e in row {
                    e.collect_constants(cs);
                }
            }
        }
        _ => cc_logic_sets_and_rest(expr, cs),
    }
}

fn cc_logic_sets_and_rest(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::ForAll { domain, body, .. } | Expression::Exists { domain, body, .. } => {
            if let Some(d) = domain {
                d.collect_constants(cs);
            }
            body.collect_constants(cs);
        }
        Expression::Logical { operands, .. } => {
            for o in operands {
                o.collect_constants(cs);
            }
        }
        Expression::MarkedVector { .. }
        | Expression::NumberSetExpr(_)
        | Expression::EmptySet
        | Expression::Nabla
        | Expression::Tensor { .. }
        | Expression::KroneckerDelta { .. }
        | Expression::LeviCivita { .. }
        | Expression::Differential { .. } => {}
        Expression::DotProduct { left, right }
        | Expression::CrossProduct { left, right }
        | Expression::OuterProduct { left, right }
        | Expression::SetOperation { left, right, .. }
        | Expression::WedgeProduct { left, right } => {
            left.collect_constants(cs);
            right.collect_constants(cs);
        }
        Expression::Gradient { expr } | Expression::Laplacian { expr } => {
            expr.collect_constants(cs);
        }
        Expression::Divergence { field } | Expression::Curl { field } => {
            field.collect_constants(cs);
        }
        Expression::Determinant { matrix }
        | Expression::Trace { matrix }
        | Expression::Rank { matrix }
        | Expression::ConjugateTranspose { matrix }
        | Expression::MatrixInverse { matrix } => {
            matrix.collect_constants(cs);
        }
        _ => cc_sets_theory(expr, cs),
    }
}

fn cc_sets_theory(expr: &Expression, cs: &mut HashSet<MathConstant>) {
    match expr {
        Expression::SetRelationExpr { element, set, .. } => {
            element.collect_constants(cs);
            set.collect_constants(cs);
        }
        Expression::SetBuilder {
            domain, predicate, ..
        } => {
            if let Some(d) = domain {
                d.collect_constants(cs);
            }
            predicate.collect_constants(cs);
        }
        Expression::PowerSet { set } => set.collect_constants(cs),
        Expression::FunctionSignature {
            domain, codomain, ..
        } => {
            domain.collect_constants(cs);
            codomain.collect_constants(cs);
        }
        Expression::Composition { outer, inner } => {
            outer.collect_constants(cs);
            inner.collect_constants(cs);
        }
        Expression::Relation { left, right, .. } => {
            left.collect_constants(cs);
            right.collect_constants(cs);
        }
        _ => {}
    }
}

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
    /// Finds all unique variable names in the expression.
    ///
    /// Recursively traverses the AST and collects all `Variable` nodes,
    /// returning their names as a set. Index variables from summations
    /// and products are also included.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // x + y
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Variable("x".to_string())),
    ///     right: Box::new(Expression::Variable("y".to_string())),
    /// };
    ///
    /// let vars = expr.find_variables();
    /// assert_eq!(vars.len(), 2);
    /// assert!(vars.contains("x"));
    /// assert!(vars.contains("y"));
    /// ```
    pub fn find_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();
        self.collect_variables(&mut variables);
        variables
    }

    fn collect_variables(&self, variables: &mut HashSet<String>) {
        cv_core(self, variables);
    }

    /// Finds all unique function names in the expression.
    ///
    /// Recursively traverses the AST and collects all `Function` node names,
    /// returning them as a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, BinaryOp};
    ///
    /// // sin(x) + cos(y)
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Function {
    ///         name: "sin".to_string(),
    ///         args: vec![Expression::Variable("x".to_string())],
    ///     }),
    ///     right: Box::new(Expression::Function {
    ///         name: "cos".to_string(),
    ///         args: vec![Expression::Variable("y".to_string())],
    ///     }),
    /// };
    ///
    /// let funcs = expr.find_functions();
    /// assert_eq!(funcs.len(), 2);
    /// assert!(funcs.contains("sin"));
    /// assert!(funcs.contains("cos"));
    /// ```
    pub fn find_functions(&self) -> HashSet<String> {
        let mut functions = HashSet::new();
        self.collect_functions(&mut functions);
        functions
    }

    fn collect_functions(&self, functions: &mut HashSet<String>) {
        cf_core(self, functions);
    }

    /// Finds all unique mathematical constants in the expression.
    ///
    /// Recursively traverses the AST and collects all `Constant` nodes,
    /// returning them as a set.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, MathConstant, BinaryOp};
    ///
    /// // 2 * π + e
    /// let expr = Expression::Binary {
    ///     op: BinaryOp::Add,
    ///     left: Box::new(Expression::Binary {
    ///         op: BinaryOp::Mul,
    ///         left: Box::new(Expression::Integer(2)),
    ///         right: Box::new(Expression::Constant(MathConstant::Pi)),
    ///     }),
    ///     right: Box::new(Expression::Constant(MathConstant::E)),
    /// };
    ///
    /// let consts = expr.find_constants();
    /// assert_eq!(consts.len(), 2);
    /// assert!(consts.contains(&MathConstant::Pi));
    /// assert!(consts.contains(&MathConstant::E));
    /// ```
    pub fn find_constants(&self) -> HashSet<MathConstant> {
        let mut constants = HashSet::new();
        self.collect_constants(&mut constants);
        constants
    }

    fn collect_constants(&self, constants: &mut HashSet<MathConstant>) {
        cc_core(self, constants);
    }

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

    fn substitute_with(&self, lookup: &impl Fn(&str) -> Option<Expression>) -> Expression {
        sw_core(self, lookup)
    }

    /// Returns whether this expression is a well-formed rectangular matrix.
    ///
    /// A valid matrix has at least one row, and all rows have the same length.
    /// Returns `false` for non-Matrix expressions.
    ///
    /// Parsers always produce valid matrices. This method is useful for
    /// validating manually constructed matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// let valid = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]);
    /// assert!(valid.is_valid_matrix());
    ///
    /// let ragged = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3)],
    /// ]);
    /// assert!(!ragged.is_valid_matrix());
    ///
    /// let not_matrix = Expression::Integer(42);
    /// assert!(!not_matrix.is_valid_matrix());
    /// ```
    pub fn is_valid_matrix(&self) -> bool {
        match self {
            Expression::Matrix(rows) => {
                if rows.is_empty() {
                    return false;
                }
                let cols = rows[0].len();
                cols > 0 && rows.iter().all(|row| row.len() == cols)
            }
            _ => false,
        }
    }

    /// Returns the dimensions `(rows, cols)` of a valid rectangular matrix.
    ///
    /// Returns `None` if this is not a Matrix expression or if the matrix is
    /// empty or ragged (non-rectangular).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::Expression;
    ///
    /// let matrix = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2), Expression::Integer(3)],
    ///     vec![Expression::Integer(4), Expression::Integer(5), Expression::Integer(6)],
    /// ]);
    /// assert_eq!(matrix.matrix_dimensions(), Some((2, 3)));
    ///
    /// let not_matrix = Expression::Integer(42);
    /// assert_eq!(not_matrix.matrix_dimensions(), None);
    /// ```
    pub fn matrix_dimensions(&self) -> Option<(usize, usize)> {
        if self.is_valid_matrix() {
            if let Expression::Matrix(rows) = self {
                return Some((rows.len(), rows[0].len()));
            }
        }
        None
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, Direction, InequalityOp, IntegralBounds, MathConstant, UnaryOp};

    // Tests for find_variables

    #[test]
    fn test_find_variables_leaf_nodes() {
        // Integer - no variables
        let expr = Expression::Integer(42);
        assert_eq!(expr.find_variables().len(), 0);

        // Float - no variables
        let expr = Expression::Float(crate::ast::MathFloat::from(3.14));
        assert_eq!(expr.find_variables().len(), 0);

        // Constant - no variables
        let expr = Expression::Constant(MathConstant::Pi);
        assert_eq!(expr.find_variables().len(), 0);

        // Variable - one variable
        let expr = Expression::Variable("x".to_string());
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_binary() {
        // x + y
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_duplicate() {
        // x + x
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("x".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1); // Set deduplicates
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_rational() {
        // x / y
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Variable("x".to_string())),
            denominator: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_complex() {
        // a + bi
        let expr = Expression::Complex {
            real: Box::new(Expression::Variable("a".to_string())),
            imaginary: Box::new(Expression::Variable("b".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_find_variables_unary() {
        // -x
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_function() {
        // sin(x, y)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_find_variables_derivative() {
        // d/dx(f)
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("f"));
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_integral() {
        // ∫₀¹ x dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_integral_with_variable_bounds() {
        // ∫ₐᵇ x dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Variable("a".to_string())),
                upper: Box::new(Expression::Variable("b".to_string())),
            }),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
    }

    #[test]
    fn test_find_variables_limit() {
        // lim x→0 f
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("f"));
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_find_variables_sum() {
        // Σᵢ₌₁ⁿ i
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("i"));
        assert!(vars.contains("n"));
    }

    #[test]
    fn test_find_variables_vector() {
        // [x, y, z]
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
            Expression::Variable("z".to_string()),
        ]);
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
    }

    #[test]
    fn test_find_variables_matrix() {
        // [[a, b], [c, d]]
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("a".to_string()),
                Expression::Variable("b".to_string()),
            ],
            vec![
                Expression::Variable("c".to_string()),
                Expression::Variable("d".to_string()),
            ],
        ]);
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 4);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("c"));
        assert!(vars.contains("d"));
    }

    #[test]
    fn test_find_variables_equation() {
        // x = y
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        let vars = expr.find_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    // Tests for find_functions

    #[test]
    fn test_find_functions_leaf_nodes() {
        // No functions in leaf nodes
        let expr = Expression::Variable("x".to_string());
        assert_eq!(expr.find_functions().len(), 0);
    }

    #[test]
    fn test_find_functions_simple() {
        // sin(x)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1);
        assert!(funcs.contains("sin"));
    }

    #[test]
    fn test_find_functions_multiple() {
        // sin(x) + cos(y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            right: Box::new(Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("y".to_string())],
            }),
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 2);
        assert!(funcs.contains("sin"));
        assert!(funcs.contains("cos"));
    }

    #[test]
    fn test_find_functions_nested() {
        // sin(cos(x))
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }],
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 2);
        assert!(funcs.contains("sin"));
        assert!(funcs.contains("cos"));
    }

    #[test]
    fn test_find_functions_duplicate() {
        // sin(x) + sin(y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            right: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("y".to_string())],
            }),
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1); // Set deduplicates
        assert!(funcs.contains("sin"));
    }

    #[test]
    fn test_find_functions_in_integral() {
        // ∫ sin(x) dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            var: "x".to_string(),
            bounds: None,
        };
        let funcs = expr.find_functions();
        assert_eq!(funcs.len(), 1);
        assert!(funcs.contains("sin"));
    }

    // Tests for find_constants

    #[test]
    fn test_find_constants_none() {
        // x + 1
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(1)),
        };
        assert_eq!(expr.find_constants().len(), 0);
    }

    #[test]
    fn test_find_constants_single() {
        // π
        let expr = Expression::Constant(MathConstant::Pi);
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1);
        assert!(consts.contains(&MathConstant::Pi));
    }

    #[test]
    fn test_find_constants_multiple() {
        // π + e
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Constant(MathConstant::Pi)),
            right: Box::new(Expression::Constant(MathConstant::E)),
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 2);
        assert!(consts.contains(&MathConstant::Pi));
        assert!(consts.contains(&MathConstant::E));
    }

    #[test]
    fn test_find_constants_duplicate() {
        // π + π
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Constant(MathConstant::Pi)),
            right: Box::new(Expression::Constant(MathConstant::Pi)),
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1); // Set deduplicates
        assert!(consts.contains(&MathConstant::Pi));
    }

    #[test]
    fn test_find_constants_all_types() {
        // [π, e, i, ∞, -∞]
        let expr = Expression::Vector(vec![
            Expression::Constant(MathConstant::Pi),
            Expression::Constant(MathConstant::E),
            Expression::Constant(MathConstant::I),
            Expression::Constant(MathConstant::Infinity),
            Expression::Constant(MathConstant::NegInfinity),
        ]);
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 5);
        assert!(consts.contains(&MathConstant::Pi));
        assert!(consts.contains(&MathConstant::E));
        assert!(consts.contains(&MathConstant::I));
        assert!(consts.contains(&MathConstant::Infinity));
        assert!(consts.contains(&MathConstant::NegInfinity));
    }

    #[test]
    fn test_find_constants_in_limit() {
        // lim x→∞ f
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: Direction::Both,
        };
        let consts = expr.find_constants();
        assert_eq!(consts.len(), 1);
        assert!(consts.contains(&MathConstant::Infinity));
    }

    // Tests for depth

    #[test]
    fn test_depth_leaf_nodes() {
        // All leaf nodes have depth 1
        assert_eq!(Expression::Integer(42).depth(), 1);
        assert_eq!(
            Expression::Float(crate::ast::MathFloat::from(3.14)).depth(),
            1
        );
        assert_eq!(Expression::Variable("x".to_string()).depth(), 1);
        assert_eq!(Expression::Constant(MathConstant::Pi).depth(), 1);
    }

    #[test]
    fn test_depth_unary() {
        // -x has depth 2
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_binary() {
        // x + y has depth 2
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_nested() {
        // (x + y) * z has depth 3
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            right: Box::new(Expression::Variable("z".to_string())),
        };
        assert_eq!(expr.depth(), 3);
    }

    #[test]
    fn test_depth_asymmetric() {
        // ((x + y) + z) + w has depth 4 on left, 1 on right
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expression::Variable("x".to_string())),
                    right: Box::new(Expression::Variable("y".to_string())),
                }),
                right: Box::new(Expression::Variable("z".to_string())),
            }),
            right: Box::new(Expression::Variable("w".to_string())),
        };
        assert_eq!(expr.depth(), 4);
    }

    #[test]
    fn test_depth_function() {
        // sin(x) has depth 2
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.depth(), 2);

        // sin(cos(x)) has depth 3
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Function {
                name: "cos".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }],
        };
        assert_eq!(expr.depth(), 3);
    }

    #[test]
    fn test_depth_vector() {
        // [x, y] has depth 2
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
        ]);
        assert_eq!(expr.depth(), 2);

        // Empty vector has depth 1
        let expr = Expression::Vector(vec![]);
        assert_eq!(expr.depth(), 1);
    }

    #[test]
    fn test_depth_matrix() {
        // [[x, y], [z, w]] has depth 2
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
            vec![
                Expression::Variable("z".to_string()),
                Expression::Variable("w".to_string()),
            ],
        ]);
        assert_eq!(expr.depth(), 2);
    }

    #[test]
    fn test_depth_integral() {
        // ∫₀¹ x dx has depth 2 (integrand is leaf, bounds are leaves)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.depth(), 2);
    }

    // Tests for node_count

    #[test]
    fn test_node_count_leaf_nodes() {
        // All leaf nodes have count 1
        assert_eq!(Expression::Integer(42).node_count(), 1);
        assert_eq!(
            Expression::Float(crate::ast::MathFloat::from(3.14)).node_count(),
            1
        );
        assert_eq!(Expression::Variable("x".to_string()).node_count(), 1);
        assert_eq!(Expression::Constant(MathConstant::Pi).node_count(), 1);
    }

    #[test]
    fn test_node_count_unary() {
        // -x has 2 nodes (Neg + x)
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.node_count(), 2);
    }

    #[test]
    fn test_node_count_binary() {
        // x + y has 3 nodes (Add + x + y)
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_nested() {
        // (x + y) * z has 5 nodes (Mul + (Add + x + y) + z)
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Variable("y".to_string())),
            }),
            right: Box::new(Expression::Variable("z".to_string())),
        };
        assert_eq!(expr.node_count(), 5);
    }

    #[test]
    fn test_node_count_function() {
        // sin() has 1 node
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![],
        };
        assert_eq!(expr.node_count(), 1);

        // sin(x) has 2 nodes (Function + x)
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.node_count(), 2);

        // max(x, y, z) has 4 nodes (Function + x + y + z)
        let expr = Expression::Function {
            name: "max".to_string(),
            args: vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
                Expression::Variable("z".to_string()),
            ],
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_vector() {
        // [] has 1 node
        let expr = Expression::Vector(vec![]);
        assert_eq!(expr.node_count(), 1);

        // [x, y, z] has 4 nodes (Vector + x + y + z)
        let expr = Expression::Vector(vec![
            Expression::Variable("x".to_string()),
            Expression::Variable("y".to_string()),
            Expression::Variable("z".to_string()),
        ]);
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_matrix() {
        // [[x, y], [z, w]] has 5 nodes (Matrix + x + y + z + w)
        let expr = Expression::Matrix(vec![
            vec![
                Expression::Variable("x".to_string()),
                Expression::Variable("y".to_string()),
            ],
            vec![
                Expression::Variable("z".to_string()),
                Expression::Variable("w".to_string()),
            ],
        ]);
        assert_eq!(expr.node_count(), 5);
    }

    #[test]
    fn test_node_count_integral() {
        // ∫ x dx has 2 nodes (Integral + x)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };
        assert_eq!(expr.node_count(), 2);

        // ∫₀¹ x dx has 4 nodes (Integral + x + 0 + 1)
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_sum() {
        // Σᵢ₌₁ⁿ i has 4 nodes (Sum + 1 + n + i)
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(expr.node_count(), 4);
    }

    #[test]
    fn test_node_count_equation() {
        // x = y has 3 nodes (Equation + x + y)
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_inequality() {
        // x < y has 3 nodes (Inequality + x + y)
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        };
        assert_eq!(expr.node_count(), 3);
    }

    #[test]
    fn test_node_count_complex_expression() {
        // 2 * π * x has 5 nodes
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expression::Integer(2)),
                right: Box::new(Expression::Constant(MathConstant::Pi)),
            }),
            right: Box::new(Expression::Variable("x".to_string())),
        };
        assert_eq!(expr.node_count(), 5);
    }
}
