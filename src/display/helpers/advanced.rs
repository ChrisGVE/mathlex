//! Advanced formatting helpers: calculus, linear algebra, logic/sets, relations.

use crate::ast::{
    Direction, ExprKind, Expression, LogicalOp, NumberSet, SetOp, SetRelation, TensorIndex,
    VectorNotation,
};
use std::fmt;

/// Format multiple-integral bodies.  Used only by `fmt_calculus`.
fn fmt_multiple_integral(
    dimension: &u8,
    integrand: &Expression,
    bounds: &Option<crate::ast::MultipleBounds>,
    vars: &[String],
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let symbol = match dimension {
        2 => "∬",
        3 => "∭",
        _ => "∫∫...",
    };
    let vars_str = vars
        .iter()
        .map(|v| format!("d{}", v))
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(b) = bounds {
        let bounds_str = b
            .bounds
            .iter()
            .map(|ib| format!("{}", ib))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "{} {} {} [{}]", symbol, integrand, vars_str, bounds_str)
    } else {
        write!(f, "{} {} {}", symbol, integrand, vars_str)
    }
}

/// Format a closed integral expression.  Used only by `fmt_calculus`.
fn fmt_closed_integral(
    dimension: &u8,
    integrand: &Expression,
    surface: &Option<String>,
    var: &str,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let symbol = match dimension {
        1 => "∮",
        2 => "∯",
        3 => "∰",
        _ => "∮",
    };
    if let Some(s) = surface {
        write!(f, "{}_{} {} d{}", symbol, s, integrand, var)
    } else {
        write!(f, "{} {} d{}", symbol, integrand, var)
    }
}

/// Format calculus expressions: Derivative, PartialDerivative, Integral,
/// MultipleIntegral, ClosedIntegral, Limit, Sum, Product.
pub(crate) fn fmt_calculus(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::Derivative { expr, var, order } => {
            if *order == 1 {
                write!(f, "d/d{}({})", var, expr)
            } else {
                write!(f, "d^{}/d{}^{}({})", order, var, order, expr)
            }
        }
        ExprKind::PartialDerivative { expr, var, order } => {
            if *order == 1 {
                write!(f, "∂/∂{}({})", var, expr)
            } else {
                write!(f, "∂^{}/∂{}^{}({})", order, var, order, expr)
            }
        }
        ExprKind::Integral {
            integrand,
            var,
            bounds,
        } => {
            if let Some(bounds) = bounds {
                write!(f, "int({}, d{}, {})", integrand, var, bounds)
            } else {
                write!(f, "int({}, d{})", integrand, var)
            }
        }
        ExprKind::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => fmt_multiple_integral(dimension, integrand, bounds, vars, f),
        ExprKind::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => fmt_closed_integral(dimension, integrand, surface, var, f),
        ExprKind::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            let dir_str = match direction {
                Direction::Both => String::new(),
                Direction::Left => "-".to_string(),
                Direction::Right => "+".to_string(),
            };
            write!(f, "lim({}->{}{})({})", var, to, dir_str, expr)
        }
        ExprKind::Sum {
            index,
            lower,
            upper,
            body,
        } => write!(f, "sum({}={}, {}, {})", index, lower, upper, body),
        ExprKind::Product {
            index,
            lower,
            upper,
            body,
        } => write!(f, "prod({}={}, {}, {})", index, lower, upper, body),
        _ => unreachable!("fmt_calculus called on non-calculus"),
    }
}

/// Write indexed notation (^{...} and _{...}) for a slice of tensor indices.
fn fmt_tensor_indices(indices: &[TensorIndex], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let (upper, lower) = crate::ast::linear_algebra::format_tensor_indices(indices);
    write!(f, "{upper}{lower}")
}

/// Format linear-algebra expressions: Vector, Matrix, MarkedVector, DotProduct,
/// CrossProduct, OuterProduct, Gradient, Divergence, Curl, Laplacian, Nabla,
/// Determinant, Trace, Rank, ConjugateTranspose, MatrixInverse,
/// Tensor, KroneckerDelta, LeviCivita.
pub(crate) fn fmt_linear_algebra(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::Vector(elements) => {
            write!(f, "[")?;
            for (i, elem) in elements.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "]")
        }
        ExprKind::Matrix(rows) => {
            write!(f, "[")?;
            for (i, row) in rows.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "[")?;
                for (j, elem) in row.iter().enumerate() {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")?;
            }
            write!(f, "]")
        }
        ExprKind::MarkedVector { name, notation } => {
            let prefix = match notation {
                VectorNotation::Bold => "[bold]",
                VectorNotation::Arrow | VectorNotation::Hat | VectorNotation::Plain => "",
                VectorNotation::Underline => "[underline]",
            };
            let suffix = match notation {
                VectorNotation::Arrow => "⃗",
                VectorNotation::Hat => "̂",
                _ => "",
            };
            write!(f, "{}{}{}", prefix, name, suffix)
        }
        ExprKind::DotProduct { left, right } => write!(f, "{} · {}", left, right),
        ExprKind::CrossProduct { left, right } => write!(f, "{} × {}", left, right),
        ExprKind::OuterProduct { left, right } => write!(f, "{} ⊗ {}", left, right),
        ExprKind::Gradient { expr } => write!(f, "∇{}", expr),
        ExprKind::Divergence { field } => write!(f, "∇·{}", field),
        ExprKind::Curl { field } => write!(f, "∇×{}", field),
        ExprKind::Laplacian { expr } => write!(f, "∇²{}", expr),
        ExprKind::Nabla => write!(f, "∇"),
        ExprKind::Determinant { matrix } => write!(f, "det({})", matrix),
        ExprKind::Trace { matrix } => write!(f, "tr({})", matrix),
        ExprKind::Rank { matrix } => write!(f, "rank({})", matrix),
        ExprKind::ConjugateTranspose { matrix } => write!(f, "{}†", matrix),
        ExprKind::MatrixInverse { matrix } => write!(f, "{}⁻¹", matrix),
        ExprKind::Tensor { name, indices } => {
            write!(f, "{}", name)?;
            fmt_tensor_indices(indices, f)
        }
        ExprKind::KroneckerDelta { indices } => {
            write!(f, "δ")?;
            fmt_tensor_indices(indices, f)
        }
        ExprKind::LeviCivita { indices } => {
            write!(f, "ε")?;
            fmt_tensor_indices(indices, f)
        }
        _ => unreachable!("fmt_linear_algebra called on non-linear-algebra"),
    }
}

/// Format quantifier expressions (ForAll, Exists).  Used only by `fmt_logic_sets`.
fn fmt_quantifiers(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::ForAll {
            variable,
            domain,
            body,
        } => {
            if let Some(d) = domain {
                write!(f, "∀{} ∈ {}: {}", variable, d, body)
            } else {
                write!(f, "∀{}: {}", variable, body)
            }
        }
        ExprKind::Exists {
            variable,
            domain,
            body,
            unique,
        } => {
            let quantifier = if *unique { "∃!" } else { "∃" };
            if let Some(d) = domain {
                write!(f, "{}{} ∈ {}: {}", quantifier, variable, d, body)
            } else {
                write!(f, "{}{}: {}", quantifier, variable, body)
            }
        }
        _ => unreachable!("fmt_quantifiers called on non-quantifier"),
    }
}

/// Format set-membership and set-operation expressions.  Used only by
/// `fmt_logic_sets`.
fn fmt_set_ops(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::NumberSetExpr(set) => {
            let symbol = match set {
                NumberSet::Natural => "ℕ",
                NumberSet::Integer => "ℤ",
                NumberSet::Rational => "ℚ",
                NumberSet::Real => "ℝ",
                NumberSet::Complex => "ℂ",
                NumberSet::Quaternion => "ℍ",
            };
            write!(f, "{}", symbol)
        }
        ExprKind::SetOperation { op, left, right } => {
            let symbol = match op {
                SetOp::Union => "∪",
                SetOp::Intersection => "∩",
                SetOp::Difference => "∖",
                SetOp::SymmetricDiff => "△",
                SetOp::CartesianProd => "×",
            };
            write!(f, "{} {} {}", left, symbol, right)
        }
        ExprKind::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            let symbol = match relation {
                SetRelation::In => "∈",
                SetRelation::NotIn => "∉",
                SetRelation::Subset => "⊂",
                SetRelation::SubsetEq => "⊆",
                SetRelation::Superset => "⊃",
                SetRelation::SupersetEq => "⊇",
            };
            write!(f, "{} {} {}", element, symbol, set)
        }
        ExprKind::SetBuilder {
            variable,
            domain,
            predicate,
        } => {
            if let Some(d) = domain {
                write!(f, "{{{} ∈ {} | {}}}", variable, d, predicate)
            } else {
                write!(f, "{{{} | {}}}", variable, predicate)
            }
        }
        ExprKind::EmptySet => write!(f, "∅"),
        ExprKind::PowerSet { set } => write!(f, "𝒫({})", set),
        _ => unreachable!("fmt_set_ops called on non-set-op"),
    }
}

/// Format logic and set expressions: Equation, Inequality, ForAll, Exists,
/// Logical, SetOperation, SetRelationExpr, SetBuilder, NumberSetExpr,
/// EmptySet, PowerSet.
pub(crate) fn fmt_logic_sets(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::Equation { left, right } => write!(f, "{} = {}", left, right),
        ExprKind::Inequality { op, left, right } => write!(f, "{} {} {}", left, op, right),
        ExprKind::ForAll { .. } | ExprKind::Exists { .. } => fmt_quantifiers(expr, f),
        ExprKind::Logical { op, operands } => match op {
            LogicalOp::Not => {
                if operands.len() == 1 {
                    write!(f, "{}{}", op, operands[0])
                } else {
                    write!(f, "{}({})", op, operands[0])
                }
            }
            _ => {
                for (i, operand) in operands.iter().enumerate() {
                    if i > 0 {
                        write!(f, " {} ", op)?;
                    }
                    write!(f, "{}", operand)?;
                }
                Ok(())
            }
        },
        ExprKind::NumberSetExpr(_)
        | ExprKind::SetOperation { .. }
        | ExprKind::SetRelationExpr { .. }
        | ExprKind::SetBuilder { .. }
        | ExprKind::EmptySet
        | ExprKind::PowerSet { .. } => fmt_set_ops(expr, f),
        _ => unreachable!("fmt_logic_sets called on non-logic-sets"),
    }
}

/// Format relational/misc expressions: FunctionSignature, Composition,
/// Differential, WedgeProduct, Relation.
pub(crate) fn fmt_relations(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &expr.kind {
        ExprKind::FunctionSignature {
            name,
            domain,
            codomain,
        } => write!(f, "{}: {} → {}", name, domain, codomain),
        ExprKind::Composition { outer, inner } => write!(f, "{} ∘ {}", outer, inner),
        ExprKind::Differential { var } => write!(f, "d{}", var),
        ExprKind::WedgeProduct { left, right } => write!(f, "{} ∧ {}", left, right),
        ExprKind::Relation { op, left, right } => write!(f, "{} {} {}", left, op, right),
        _ => unreachable!("fmt_relations called on non-relation"),
    }
}
