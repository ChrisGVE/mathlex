//! Advanced formatting helpers: calculus, linear algebra, logic/sets, relations.

use crate::ast::{
    Direction, Expression, IndexType, LogicalOp, NumberSet, SetOp, SetRelation, TensorIndex,
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
    match expr {
        Expression::Derivative { expr, var, order } => {
            if *order == 1 {
                write!(f, "d/d{}({})", var, expr)
            } else {
                write!(f, "d^{}/d{}^{}({})", order, var, order, expr)
            }
        }
        Expression::PartialDerivative { expr, var, order } => {
            if *order == 1 {
                write!(f, "∂/∂{}({})", var, expr)
            } else {
                write!(f, "∂^{}/∂{}^{}({})", order, var, order, expr)
            }
        }
        Expression::Integral {
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
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => fmt_multiple_integral(dimension, integrand, bounds, vars, f),
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => fmt_closed_integral(dimension, integrand, surface, var, f),
        Expression::Limit {
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
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => write!(f, "sum({}={}, {}, {})", index, lower, upper, body),
        Expression::Product {
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
    let upper: Vec<_> = indices
        .iter()
        .filter(|i| i.index_type == IndexType::Upper)
        .collect();
    let lower: Vec<_> = indices
        .iter()
        .filter(|i| i.index_type == IndexType::Lower)
        .collect();
    if !upper.is_empty() {
        write!(
            f,
            "^{{{}}}",
            upper
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>()
                .join("")
        )?;
    }
    if !lower.is_empty() {
        write!(
            f,
            "_{{{}}}",
            lower
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>()
                .join("")
        )?;
    }
    Ok(())
}

/// Format linear-algebra expressions: Vector, Matrix, MarkedVector, DotProduct,
/// CrossProduct, OuterProduct, Gradient, Divergence, Curl, Laplacian, Nabla,
/// Determinant, Trace, Rank, ConjugateTranspose, MatrixInverse,
/// Tensor, KroneckerDelta, LeviCivita.
pub(crate) fn fmt_linear_algebra(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        Expression::Vector(elements) => {
            write!(f, "[")?;
            for (i, elem) in elements.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", elem)?;
            }
            write!(f, "]")
        }
        Expression::Matrix(rows) => {
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
        Expression::MarkedVector { name, notation } => {
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
        Expression::DotProduct { left, right } => write!(f, "{} · {}", left, right),
        Expression::CrossProduct { left, right } => write!(f, "{} × {}", left, right),
        Expression::OuterProduct { left, right } => write!(f, "{} ⊗ {}", left, right),
        Expression::Gradient { expr } => write!(f, "∇{}", expr),
        Expression::Divergence { field } => write!(f, "∇·{}", field),
        Expression::Curl { field } => write!(f, "∇×{}", field),
        Expression::Laplacian { expr } => write!(f, "∇²{}", expr),
        Expression::Nabla => write!(f, "∇"),
        Expression::Determinant { matrix } => write!(f, "det({})", matrix),
        Expression::Trace { matrix } => write!(f, "tr({})", matrix),
        Expression::Rank { matrix } => write!(f, "rank({})", matrix),
        Expression::ConjugateTranspose { matrix } => write!(f, "{}†", matrix),
        Expression::MatrixInverse { matrix } => write!(f, "{}⁻¹", matrix),
        Expression::Tensor { name, indices } => {
            write!(f, "{}", name)?;
            fmt_tensor_indices(indices, f)
        }
        Expression::KroneckerDelta { indices } => {
            write!(f, "δ")?;
            fmt_tensor_indices(indices, f)
        }
        Expression::LeviCivita { indices } => {
            write!(f, "ε")?;
            fmt_tensor_indices(indices, f)
        }
        _ => unreachable!("fmt_linear_algebra called on non-linear-algebra"),
    }
}

/// Format quantifier expressions (ForAll, Exists).  Used only by `fmt_logic_sets`.
fn fmt_quantifiers(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        Expression::ForAll {
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
        Expression::Exists {
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
    match expr {
        Expression::NumberSetExpr(set) => {
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
        Expression::SetOperation { op, left, right } => {
            let symbol = match op {
                SetOp::Union => "∪",
                SetOp::Intersection => "∩",
                SetOp::Difference => "∖",
                SetOp::SymmetricDiff => "△",
                SetOp::CartesianProd => "×",
            };
            write!(f, "{} {} {}", left, symbol, right)
        }
        Expression::SetRelationExpr {
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
        Expression::SetBuilder {
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
        Expression::EmptySet => write!(f, "∅"),
        Expression::PowerSet { set } => write!(f, "𝒫({})", set),
        _ => unreachable!("fmt_set_ops called on non-set-op"),
    }
}

/// Format logic and set expressions: Equation, Inequality, ForAll, Exists,
/// Logical, SetOperation, SetRelationExpr, SetBuilder, NumberSetExpr,
/// EmptySet, PowerSet.
pub(crate) fn fmt_logic_sets(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        Expression::Equation { left, right } => write!(f, "{} = {}", left, right),
        Expression::Inequality { op, left, right } => write!(f, "{} {} {}", left, op, right),
        Expression::ForAll { .. } | Expression::Exists { .. } => fmt_quantifiers(expr, f),
        Expression::Logical { op, operands } => match op {
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
        Expression::NumberSetExpr(_)
        | Expression::SetOperation { .. }
        | Expression::SetRelationExpr { .. }
        | Expression::SetBuilder { .. }
        | Expression::EmptySet
        | Expression::PowerSet { .. } => fmt_set_ops(expr, f),
        _ => unreachable!("fmt_logic_sets called on non-logic-sets"),
    }
}

/// Format relational/misc expressions: FunctionSignature, Composition,
/// Differential, WedgeProduct, Relation.
pub(crate) fn fmt_relations(expr: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match expr {
        Expression::FunctionSignature {
            name,
            domain,
            codomain,
        } => write!(f, "{}: {} → {}", name, domain, codomain),
        Expression::Composition { outer, inner } => write!(f, "{} ∘ {}", outer, inner),
        Expression::Differential { var } => write!(f, "d{}", var),
        Expression::WedgeProduct { left, right } => write!(f, "{} ∧ {}", left, right),
        Expression::Relation { op, left, right } => write!(f, "{} {} {}", left, op, right),
        _ => unreachable!("fmt_relations called on non-relation"),
    }
}
