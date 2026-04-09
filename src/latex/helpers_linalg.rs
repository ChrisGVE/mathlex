use super::trait_def::{wrap_if_additive, ToLatex};
use crate::ast::linear_algebra::format_tensor_indices;
use crate::ast::{
    Expression, LogicalOp, NumberSet, SetOp, SetRelation, TensorIndex, VectorNotation,
};

fn indexed_symbol_to_latex(prefix: &str, indices: &[TensorIndex]) -> String {
    let (upper, lower) = format_tensor_indices(indices);
    format!("{prefix}{upper}{lower}")
}

pub(super) fn to_latex_linear_algebra(expr: &Expression) -> String {
    match expr {
        Expression::Vector(elements) => {
            let s = elements
                .iter()
                .map(|e| e.to_latex())
                .collect::<Vec<_>>()
                .join(r" \\ ");
            format!(r"\begin{{pmatrix}} {} \end{{pmatrix}}", s)
        }
        Expression::Matrix(rows) => {
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
        Expression::MarkedVector { name, notation } => match notation {
            VectorNotation::Bold => format!(r"\mathbf{{{}}}", name),
            VectorNotation::Arrow => format!(r"\vec{{{}}}", name),
            VectorNotation::Hat => format!(r"\hat{{{}}}", name),
            VectorNotation::Underline => format!(r"\underline{{{}}}", name),
            VectorNotation::Plain => name.clone(),
        },
        Expression::DotProduct { left, right } => {
            format!(
                r"{} \cdot {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        Expression::CrossProduct { left, right } => {
            format!(
                r"{} \times {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        Expression::OuterProduct { left, right } => {
            format!(
                r"{} \otimes {}",
                wrap_if_additive(left),
                wrap_if_additive(right)
            )
        }
        Expression::Gradient { expr } => format!(r"\nabla {}", expr.to_latex()),
        Expression::Divergence { field } => format!(r"\nabla \cdot {}", field.to_latex()),
        Expression::Curl { field } => format!(r"\nabla \times {}", field.to_latex()),
        Expression::Laplacian { expr } => format!(r"\nabla^2 {}", expr.to_latex()),
        Expression::Nabla => r"\nabla".to_string(),
        Expression::Determinant { matrix } => format!(r"\det({})", matrix.to_latex()),
        Expression::Trace { matrix } => format!(r"\text{{tr}}({})", matrix.to_latex()),
        Expression::Rank { matrix } => format!(r"\text{{rank}}({})", matrix.to_latex()),
        Expression::ConjugateTranspose { matrix } => format!(r"{}^\dagger", matrix.to_latex()),
        Expression::MatrixInverse { matrix } => format!(r"{}^{{-1}}", matrix.to_latex()),
        Expression::Tensor { name, indices } => indexed_symbol_to_latex(name, indices),
        Expression::KroneckerDelta { indices } => indexed_symbol_to_latex(r"\delta", indices),
        Expression::LeviCivita { indices } => indexed_symbol_to_latex(r"\varepsilon", indices),
        _ => unreachable!("to_latex_linear_algebra called on non-linear-algebra expression"),
    }
}

fn to_latex_quantifiers(expr: &Expression) -> String {
    match expr {
        Expression::ForAll {
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
        Expression::Exists {
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
        Expression::Logical { op, operands } => match op {
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
    match expr {
        Expression::SetOperation { op, left, right } => {
            let latex_op = match op {
                SetOp::Union => r"\cup",
                SetOp::Intersection => r"\cap",
                SetOp::Difference => r"\setminus",
                SetOp::SymmetricDiff => r"\triangle",
                SetOp::CartesianProd => r"\times",
            };
            format!("{} {} {}", left.to_latex(), latex_op, right.to_latex())
        }
        Expression::SetRelationExpr {
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
        Expression::SetBuilder {
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
        Expression::EmptySet => r"\emptyset".to_string(),
        Expression::PowerSet { set } => format!(r"\mathcal{{P}}({})", set.to_latex()),
        Expression::NumberSetExpr(set) => match set {
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
    match expr {
        Expression::Equation { left, right } => {
            format!("{} = {}", left.to_latex(), right.to_latex())
        }
        Expression::Inequality { op, left, right } => {
            format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
        }
        Expression::ForAll { .. } | Expression::Exists { .. } | Expression::Logical { .. } => {
            to_latex_quantifiers(expr)
        }
        Expression::SetOperation { .. }
        | Expression::SetRelationExpr { .. }
        | Expression::SetBuilder { .. }
        | Expression::EmptySet
        | Expression::PowerSet { .. }
        | Expression::NumberSetExpr(_) => to_latex_set_ops(expr),
        _ => unreachable!("to_latex_logic_sets called on non-logic/set expression"),
    }
}

pub(super) fn to_latex_relations(expr: &Expression) -> String {
    match expr {
        Expression::FunctionSignature {
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
        Expression::Composition { outer, inner } => {
            format!("{} \\circ {}", outer.to_latex(), inner.to_latex())
        }
        Expression::Differential { var } => format!("d{}", var),
        Expression::WedgeProduct { left, right } => {
            format!(r"{} \wedge {}", left.to_latex(), right.to_latex())
        }
        Expression::Relation { op, left, right } => {
            format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
        }
        _ => unreachable!("to_latex_relations called on non-relation expression"),
    }
}
