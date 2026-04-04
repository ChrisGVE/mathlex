use super::trait_def::{needs_parens, ToLatex, KNOWN_FUNCTIONS};
use crate::ast::Expression;

const GREEK_LETTERS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi",
    "Psi", "Omega",
];

pub(super) fn variable_to_latex(name: &str) -> String {
    if let Some(underscore_pos) = name.find('_') {
        let (base, subscript_with_underscore) = name.split_at(underscore_pos);
        let subscript = &subscript_with_underscore[1..];
        let base_latex = if GREEK_LETTERS.contains(&base) {
            format!(r"\{}", base)
        } else {
            base.to_string()
        };
        if subscript.len() == 1 {
            format!("{}_{}", base_latex, subscript)
        } else {
            format!("{}_{{{}}}", base_latex, subscript)
        }
    } else if GREEK_LETTERS.contains(&name) {
        format!(r"\{}", name)
    } else {
        name.to_string()
    }
}

pub(super) fn to_latex_literal(expr: &Expression) -> String {
    match expr {
        Expression::Integer(n) => format!("{}", n),
        Expression::Float(x) => format!("{}", x),
        Expression::Rational {
            numerator,
            denominator,
        } => format!(
            r"\frac{{{}}}{{{}}}",
            numerator.to_latex(),
            denominator.to_latex()
        ),
        Expression::Complex { real, imaginary } => {
            format!("{} + {}i", real.to_latex(), imaginary.to_latex())
        }
        Expression::Quaternion { real, i, j, k } => format!(
            "{} + {}\\mathbf{{i}} + {}\\mathbf{{j}} + {}\\mathbf{{k}}",
            real.to_latex(),
            i.to_latex(),
            j.to_latex(),
            k.to_latex()
        ),
        Expression::Variable(name) => variable_to_latex(name),
        Expression::Constant(c) => c.to_latex(),
        _ => unreachable!("to_latex_literal called on non-literal"),
    }
}

pub(super) fn to_latex_binary(
    op: &crate::ast::BinaryOp,
    left: &Expression,
    right: &Expression,
) -> String {
    match op {
        crate::ast::BinaryOp::Div => {
            format!(r"\frac{{{}}}{{{}}}", left.to_latex(), right.to_latex())
        }
        crate::ast::BinaryOp::Pow => {
            let left_str = if needs_parens(left, *op, false) {
                format!(r"\left({}\right)", left.to_latex())
            } else {
                left.to_latex()
            };
            let right_str = if needs_parens(right, *op, true) {
                format!(r"\left({}\right)", right.to_latex())
            } else {
                right.to_latex()
            };
            format!("{}^{{{}}}", left_str, right_str)
        }
        crate::ast::BinaryOp::Mod => {
            format!("{} \\bmod {}", left.to_latex(), right.to_latex())
        }
        _ => {
            let left_str = if needs_parens(left, *op, false) {
                format!(r"\left({}\right)", left.to_latex())
            } else {
                left.to_latex()
            };
            let right_str = if needs_parens(right, *op, true) {
                format!(r"\left({}\right)", right.to_latex())
            } else {
                right.to_latex()
            };
            format!("{} {} {}", left_str, op.to_latex(), right_str)
        }
    }
}

pub(super) fn to_latex_unary(op: &crate::ast::UnaryOp, operand: &Expression) -> String {
    let is_binary = matches!(operand, Expression::Binary { .. });
    match op {
        crate::ast::UnaryOp::Neg => {
            if is_binary {
                format!(r"-\left({}\right)", operand.to_latex())
            } else {
                format!("-{}", operand.to_latex())
            }
        }
        crate::ast::UnaryOp::Pos => {
            if is_binary {
                format!(r"+\left({}\right)", operand.to_latex())
            } else {
                format!("+{}", operand.to_latex())
            }
        }
        crate::ast::UnaryOp::Factorial => {
            if is_binary {
                format!(r"\left({}\right)!", operand.to_latex())
            } else {
                format!("{}!", operand.to_latex())
            }
        }
        crate::ast::UnaryOp::Transpose => {
            if is_binary {
                format!(r"\left({}\right)^T", operand.to_latex())
            } else {
                format!("{}^T", operand.to_latex())
            }
        }
    }
}

pub(super) fn to_latex_function(name: &str, args: &[Expression]) -> String {
    if name == "sqrt" {
        return match args.len() {
            1 => format!(r"\sqrt{{{}}}", args[0].to_latex()),
            2 => format!(r"\sqrt[{}]{{{}}}", args[0].to_latex(), args[1].to_latex()),
            _ => format!(r"\operatorname{{{}}}", name),
        };
    }
    // log with base: args[0] = argument, args[1] = base → \log_{base}{arg}
    if name == "log" && args.len() == 2 {
        return format!(r"\log_{{{}}}{{{}}}", args[1].to_latex(), args[0].to_latex());
    }
    // floor/ceil with delimiters
    if name == "floor" && args.len() == 1 {
        return format!(r"\lfloor {} \rfloor", args[0].to_latex());
    }
    if name == "ceil" && args.len() == 1 {
        return format!(r"\lceil {} \rceil", args[0].to_latex());
    }
    let func_prefix = if KNOWN_FUNCTIONS.contains(&name) {
        format!(r"\{}", name)
    } else {
        format!(r"\operatorname{{{}}}", name)
    };
    if args.is_empty() {
        func_prefix
    } else {
        let args_str = args
            .iter()
            .map(|arg| arg.to_latex())
            .collect::<Vec<_>>()
            .join(", ");
        format!(r"{}\left({}\right)", func_prefix, args_str)
    }
}

fn to_latex_derivative(expr: &Expression) -> String {
    match expr {
        Expression::Derivative { expr, var, order } => {
            if *order == 1 {
                format!(r"\frac{{d}}{{d{}}}{}", var, expr.to_latex())
            } else {
                format!(
                    r"\frac{{d^{{{}}}}}{{d{}^{{{}}}}}{}",
                    order,
                    var,
                    order,
                    expr.to_latex()
                )
            }
        }
        Expression::PartialDerivative { expr, var, order } => {
            if *order == 1 {
                format!(r"\frac{{\partial}}{{\partial {}}}{}", var, expr.to_latex())
            } else {
                format!(
                    r"\frac{{\partial^{{{}}}}}{{\partial {}^{{{}}}}}{}",
                    order,
                    var,
                    order,
                    expr.to_latex()
                )
            }
        }
        _ => unreachable!("to_latex_derivative called on non-derivative expression"),
    }
}

fn to_latex_multiple_integral(
    dimension: u8,
    integrand: &Expression,
    bounds: Option<&crate::ast::MultipleBounds>,
    vars: &[String],
) -> String {
    let int_cmd = match dimension {
        2 => r"\iint",
        3 => r"\iiint",
        4 => r"\iiiint",
        _ => r"\int\cdots\int",
    };
    let vars_str = vars
        .iter()
        .map(|v| format!("d{}", v))
        .collect::<Vec<_>>()
        .join(" \\, ");
    if let Some(b) = bounds {
        let bounds_latex: Vec<String> = b
            .bounds
            .iter()
            .map(|ib| format!("_{{{}}}^{{{}}}", ib.lower.to_latex(), ib.upper.to_latex()))
            .collect();
        format!(
            "{}{} {} \\, {}",
            int_cmd,
            bounds_latex.join(""),
            integrand.to_latex(),
            vars_str
        )
    } else {
        format!("{} {} \\, {}", int_cmd, integrand.to_latex(), vars_str)
    }
}

fn to_latex_integral(expr: &Expression) -> String {
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            if let Some(b) = bounds {
                format!(
                    r"\int_{{{}}}^{{{}}} {} d{}",
                    b.lower.to_latex(),
                    b.upper.to_latex(),
                    integrand.to_latex(),
                    var
                )
            } else {
                format!(r"\int {} d{}", integrand.to_latex(), var)
            }
        }
        Expression::MultipleIntegral {
            dimension,
            integrand,
            bounds,
            vars,
        } => to_latex_multiple_integral(*dimension, integrand, bounds.as_ref(), vars),
        Expression::ClosedIntegral {
            dimension,
            integrand,
            surface,
            var,
        } => {
            let int_cmd = match dimension {
                1 => r"\oint",
                2 => r"\oiint",
                3 => r"\oiiint",
                _ => r"\oint",
            };
            if let Some(s) = surface {
                format!(
                    "{}_{{{}}} {} \\, d{}",
                    int_cmd,
                    s,
                    integrand.to_latex(),
                    var
                )
            } else {
                format!("{} {} \\, d{}", int_cmd, integrand.to_latex(), var)
            }
        }
        _ => unreachable!("to_latex_integral called on non-integral expression"),
    }
}

pub(super) fn to_latex_calculus(expr: &Expression) -> String {
    match expr {
        Expression::Derivative { .. } | Expression::PartialDerivative { .. } => {
            to_latex_derivative(expr)
        }
        Expression::Integral { .. }
        | Expression::MultipleIntegral { .. }
        | Expression::ClosedIntegral { .. } => to_latex_integral(expr),
        Expression::Limit {
            expr,
            var,
            to,
            direction,
        } => format!(
            r"\lim_{{{} \to {}{}}}{}",
            var,
            to.to_latex(),
            direction.to_latex(),
            expr.to_latex()
        ),
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => format!(
            r"\sum_{{{}={}}}^{{{}}}{}",
            index,
            lower.to_latex(),
            upper.to_latex(),
            body.to_latex()
        ),
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => format!(
            r"\prod_{{{}={}}}^{{{}}}{}",
            index,
            lower.to_latex(),
            upper.to_latex(),
            body.to_latex()
        ),
        _ => unreachable!("to_latex_calculus called on non-calculus expression"),
    }
}
