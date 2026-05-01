use super::trait_def::{wrap_if_additive, ToLatex};
use crate::ast::linear_algebra::format_tensor_indices;
use crate::ast::{
    ExprKind, Expression, LogicalOp, NumberSet, SetOp, SetRelation, TensorIndex, VectorNotation,
};

fn indexed_symbol_to_latex(prefix: &str, indices: &[TensorIndex]) -> String {
    let (upper, lower) = format_tensor_indices(indices);
    format!("{prefix}{upper}{lower}")
}

pub(super) fn to_latex_linear_algebra(expr: &Expression) -> String {
    match &expr.kind {
        ExprKind::Vector(elements) => {
            let s = elements
                .iter()
                .map(|e| e.to_latex())
                .collect::<Vec<_>>()
                .join(r" \\ ");
            format!(r"\begin{{pmatrix}} {} \end{{pmatrix}}", s)
        }
        ExprKind::Matrix(rows) => {
            let s = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|e| e.to_latex())
                        .collect::<Vec<_>>()
                        .join(" & ")
                })
                .collect::<Vec<_>>()
                .join(r" \\ ");
            format!(r"\begin{{pmatrix}} {} \end{{pmatrix}}", s)
        }
        ExprKind::MarkedVector { name, notation } => match notation {
            VectorNotation::Bold => format!(r"\mathbf{{{}}}", name),
            VectorNotation::Arrow => format!(r"\vec{{{}}}", name),
            VectorNotation::Hat => format!(r"\hat{{{}}}", name),
            VectorNotation::Underline => format!(r"\underline{{{}}}", name),
            VectorNotation::Plain => name.clone(),
        },
        ExprKind::DotProduct { left, right } => {
            format!(
                r"{} \cdot {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        ExprKind::CrossProduct { left, right } => {
            format!(
                r"{} \times {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        ExprKind::OuterProduct { left, right } => {
            format!(
                r"{} \otimes {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        ExprKind::Gradient { expr } => format!(r"\nabla {}", expr.to_latex()),
        ExprKind::Divergence { field } => format!(r"\nabla \cdot {}", field.to_latex()),
        ExprKind::Curl { field } => format!(r"\nabla \times {}", field.to_latex()),
        ExprKind::Laplacian { expr } => format!(r"\nabla^2 {}", expr.to_latex()),
        ExprKind::Nabla => r"\nabla".to_string(),
        ExprKind::Determinant { matrix } => format!(r"\det({})", matrix.to_latex()),
        ExprKind::Trace { matrix } => format!(r"\text{{tr}}({})", matrix.to_latex()),
        ExprKind::Rank { matrix } => format!(r"\text{{rank}}({})", matrix.to_latex()),
        ExprKind::ConjugateTranspose { matrix } => format!(r"{}^\dagger", matrix.to_latex()),
        ExprKind::MatrixInverse { matrix } => format!(r"{}^{{-1}}", matrix.to_latex()),
        ExprKind::Tensor { name, indices } => indexed_symbol_to_latex(name, indices),
        ExprKind::KroneckerDelta { indices } => indexed_symbol_to_latex(r"\delta", indices),
        ExprKind::LeviCivita { indices } => indexed_symbol_to_latex(r"\varepsilon", indices),
        _ => unreachable!("to_latex_linear_algebra called on non-linear-algebra expression"),
    }
}

fn to_latex_quantifiers(expr: &Expression) -> String {
    match &expr.kind {
        ExprKind::ForAll {
            variable,
            domain,
            body,
        } => {
            if let Some(d) = domain {
                format!(
                    r"\forall {} \in {}: {}",
                    variable,
                    d.to_latex(),
                    body.to_latex()
                )
            } else {
                format!(r"\forall {}: {}", variable, body.to_latex())
            }
        }
        ExprKind::Exists {
            variable,
            domain,
            body,
            unique,
        } => {
            let q = if *unique { r"\exists!" } else { r"\exists" };
            if let Some(d) = domain {
                format!(
                    r"{} {} \in {}: {}",
                    q,
                    variable,
                    d.to_latex(),
                    body.to_latex()
                )
            } else {
                format!(r"{} {}: {}", q, variable, body.to_latex())
            }
        }
        ExprKind::Logical { op, operands } => match op {
            LogicalOp::Not => {
                if operands.len() == 1 {
                    format!(r"{} {}", op.to_latex(), operands[0].to_latex())
                } else {
                    format!(r"{} ({})", op.to_latex(), operands[0].to_latex())
                }
            }
            _ => operands
                .iter()
                .map(|e| e.to_latex())
                .collect::<Vec<_>>()
                .join(&format!(" {} ", op.to_latex())),
        },
        _ => unreachable!("to_latex_quantifiers called on non-quantifier expression"),
    }
}

fn to_latex_set_ops(expr: &Expression) -> String {
    match &expr.kind {
        ExprKind::SetOperation { op, left, right } => {
            let latex_op = match op {
                SetOp::Union => r"\cup",
                SetOp::Intersection => r"\cap",
                SetOp::Difference => r"\setminus",
                SetOp::SymmetricDiff => r"\triangle",
                SetOp::CartesianProd => r"\times",
            };
            format!("{} {} {}", left.to_latex(), latex_op, right.to_latex())
        }
        ExprKind::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            let latex_rel = match relation {
                SetRelation::In => r"\in",
                SetRelation::NotIn => r"\notin",
                SetRelation::Subset => r"\subset",
                SetRelation::SubsetEq => r"\subseteq",
                SetRelation::Superset => r"\supset",
                SetRelation::SupersetEq => r"\supseteq",
            };
            format!("{} {} {}", element.to_latex(), latex_rel, set.to_latex())
        }
        ExprKind::SetBuilder {
            variable,
            domain,
            predicate,
        } => {
            if let Some(d) = domain {
                format!(
                    r"\{{{} \in {} \mid {}\}}",
                    variable,
                    d.to_latex(),
                    predicate.to_latex()
                )
            } else {
                format!(r"\{{{} \mid {}\}}", variable, predicate.to_latex())
            }
        }
        ExprKind::EmptySet => r"\emptyset".to_string(),
        ExprKind::PowerSet { set } => format!(r"\mathcal{{P}}({})", set.to_latex()),
        ExprKind::NumberSetExpr(set) => match set {
            NumberSet::Natural => r"\mathbb{N}",
            NumberSet::Integer => r"\mathbb{Z}",
            NumberSet::Rational => r"\mathbb{Q}",
            NumberSet::Real => r"\mathbb{R}",
            NumberSet::Complex => r"\mathbb{C}",
            NumberSet::Quaternion => r"\mathbb{H}",
        }
        .to_string(),
        _ => unreachable!("to_latex_set_ops called on non-set expression"),
    }
}

pub(super) fn to_latex_logic_sets(expr: &Expression) -> String {
    match &expr.kind {
        ExprKind::Equation { left, right } => {
            format!("{} = {}", left.to_latex(), right.to_latex())
        }
        ExprKind::Inequality { op, left, right } => {
            format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
        }
        ExprKind::ForAll { .. } | ExprKind::Exists { .. } | ExprKind::Logical { .. } => {
            to_latex_quantifiers(expr)
        }
        ExprKind::SetOperation { .. }
        | ExprKind::SetRelationExpr { .. }
        | ExprKind::SetBuilder { .. }
        | ExprKind::EmptySet
        | ExprKind::PowerSet { .. }
        | ExprKind::NumberSetExpr(_) => to_latex_set_ops(expr),
        _ => unreachable!("to_latex_logic_sets called on non-logic/set expression"),
    }
}

pub(super) fn to_latex_relations(expr: &Expression) -> String {
    match &expr.kind {
        ExprKind::FunctionSignature {
            name,
            domain,
            codomain,
        } => {
            format!(
                "{}: {} \\to {}",
                name,
                domain.to_latex(),
                codomain.to_latex()
            )
        }
        ExprKind::Composition { outer, inner } => {
            format!("{} \\circ {}", outer.to_latex(), inner.to_latex())
        }
        ExprKind::Differential { var } => format!("d{}", var),
        ExprKind::WedgeProduct { left, right } => {
            format!(r"{} \wedge {}", left.to_latex(), right.to_latex())
        }
        ExprKind::Relation { op, left, right } => {
            format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
        }
        _ => unreachable!("to_latex_relations called on non-relation expression"),
    }
}
