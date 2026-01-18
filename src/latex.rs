//! # LaTeX Conversion for AST
//!
//! This module provides the `ToLatex` trait for converting AST types back to LaTeX notation.
//!
//! ## Design Philosophy
//!
//! - **Standard LaTeX**: Generate valid LaTeX mathematical notation
//! - **Readability**: Produce clean, readable LaTeX code
//! - **Round-trip Capable**: Output can be parsed back (though not guaranteed identical AST)
//!
//! ## LaTeX Mappings
//!
//! - Integers/Floats: Direct string representation
//! - Rational: `\frac{numerator}{denominator}`
//! - Binary Division: `\frac{left}{right}`
//! - Functions: `\sin`, `\cos`, etc. for known functions, `\operatorname{name}` for others
//! - Square Root: `\sqrt{x}` or `\sqrt[n]{x}`
//! - Derivatives: `\frac{d}{dx}` or `\frac{d^n}{dx^n}`
//! - Partial Derivatives: `\frac{\partial}{\partial x}`
//! - Integrals: `\int_{lower}^{upper} expr \, dx`
//! - Limits: `\lim_{x \to value^{direction}}`
//! - Sum/Product: `\sum_{i=lower}^{upper}`, `\prod_{i=lower}^{upper}`
//! - Vectors/Matrices: `\begin{pmatrix}...\end{pmatrix}`
//!
//! ## Examples
//!
//! ```ignore
//! use mathlex::ast::{Expression, BinaryOp};
//! use mathlex::latex::ToLatex;
//!
//! // 1/2 → "\frac{1}{2}"
//! let expr = Expression::Rational {
//!     numerator: Box::new(Expression::Integer(1)),
//!     denominator: Box::new(Expression::Integer(2)),
//! };
//! assert_eq!(expr.to_latex(), r"\frac{1}{2}");
//! ```

use crate::ast::*;

/// Trait for converting AST types to LaTeX notation.
///
/// This trait provides a method to convert mathematical expressions and
/// operators into valid LaTeX strings.
///
/// # Examples
///
/// ```ignore
/// use mathlex::ast::{Expression, MathConstant};
/// use mathlex::latex::ToLatex;
///
/// let pi = Expression::Constant(MathConstant::Pi);
/// assert_eq!(pi.to_latex(), r"\pi");
/// ```
pub trait ToLatex {
    /// Converts the value to a LaTeX string.
    fn to_latex(&self) -> String;
}

// Known trigonometric and mathematical functions that have LaTeX commands
const KNOWN_FUNCTIONS: &[&str] = &[
    "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "arccot", "arcsec", "arccsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "ln", "log", "exp", "lg",
    "det", "dim", "ker", "hom", "arg",
    "deg", "gcd", "lcm", "max", "min", "sup", "inf",
    "lim", "limsup", "liminf",
];

impl ToLatex for MathConstant {
    fn to_latex(&self) -> String {
        match self {
            MathConstant::Pi => r"\pi".to_string(),
            MathConstant::E => "e".to_string(),
            MathConstant::I => "i".to_string(),
            MathConstant::Infinity => r"\infty".to_string(),
            MathConstant::NegInfinity => r"-\infty".to_string(),
        }
    }
}

impl ToLatex for BinaryOp {
    fn to_latex(&self) -> String {
        match self {
            BinaryOp::Add => "+".to_string(),
            BinaryOp::Sub => "-".to_string(),
            BinaryOp::Mul => r"\cdot".to_string(),
            BinaryOp::Div => "/".to_string(), // Division is handled specially in Expression
            BinaryOp::Pow => "^".to_string(),
            BinaryOp::Mod => r"\bmod".to_string(),
        }
    }
}

impl ToLatex for InequalityOp {
    fn to_latex(&self) -> String {
        match self {
            InequalityOp::Lt => "<".to_string(),
            InequalityOp::Le => r"\leq".to_string(),
            InequalityOp::Gt => ">".to_string(),
            InequalityOp::Ge => r"\geq".to_string(),
            InequalityOp::Ne => r"\neq".to_string(),
        }
    }
}

impl ToLatex for Direction {
    fn to_latex(&self) -> String {
        match self {
            Direction::Left => "^-".to_string(),
            Direction::Right => "^+".to_string(),
            Direction::Both => "".to_string(),
        }
    }
}

impl ToLatex for Expression {
    fn to_latex(&self) -> String {
        match self {
            Expression::Integer(n) => format!("{}", n),

            Expression::Float(x) => format!("{}", x),

            Expression::Rational { numerator, denominator } => {
                format!(r"\frac{{{}}}{{{}}}", numerator.to_latex(), denominator.to_latex())
            }

            Expression::Complex { real, imaginary } => {
                format!("{} + {}i", real.to_latex(), imaginary.to_latex())
            }

            Expression::Variable(name) => name.clone(),

            Expression::Constant(c) => c.to_latex(),

            Expression::Binary { op, left, right } => {
                match op {
                    BinaryOp::Div => {
                        // Use \frac for division
                        format!(r"\frac{{{}}}{{{}}}", left.to_latex(), right.to_latex())
                    }
                    BinaryOp::Pow => {
                        // Use superscript notation
                        format!("{}^{{{}}}", left.to_latex(), right.to_latex())
                    }
                    BinaryOp::Mod => {
                        // Use \bmod with spaces
                        format!("{} \\bmod {}", left.to_latex(), right.to_latex())
                    }
                    _ => {
                        // Regular binary operations
                        format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
                    }
                }
            }

            Expression::Unary { op, operand } => {
                match op {
                    UnaryOp::Neg => format!("-{}", operand.to_latex()),
                    UnaryOp::Pos => format!("+{}", operand.to_latex()),
                    UnaryOp::Factorial => format!("{}!", operand.to_latex()),
                    UnaryOp::Transpose => format!("{}^T", operand.to_latex()),
                }
            }

            Expression::Function { name, args } => {
                // Check if it's a known function
                let func_prefix = if KNOWN_FUNCTIONS.contains(&name.as_str()) {
                    format!(r"\{}", name)
                } else if name == "sqrt" {
                    // Special handling for sqrt with optional index
                    return if args.len() == 1 {
                        format!(r"\sqrt{{{}}}", args[0].to_latex())
                    } else if args.len() == 2 {
                        // sqrt with index: sqrt[n]{x}
                        format!(r"\sqrt[{}]{{{}}}", args[0].to_latex(), args[1].to_latex())
                    } else {
                        format!(r"\operatorname{{{}}}", name)
                    };
                } else {
                    format!(r"\operatorname{{{}}}", name)
                };

                // Format arguments
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

            Expression::Derivative { expr, var, order } => {
                if *order == 1 {
                    format!(r"\frac{{d}}{{d{}}}{}", var, expr.to_latex())
                } else {
                    format!(
                        r"\frac{{d^{{{}}}}}{{d{}^{{{}}}}}{}",
                        order, var, order, expr.to_latex()
                    )
                }
            }

            Expression::PartialDerivative { expr, var, order } => {
                if *order == 1 {
                    format!(r"\frac{{\partial}}{{\partial {}}}{}", var, expr.to_latex())
                } else {
                    format!(
                        r"\frac{{\partial^{{{}}}}}{{\partial {}^{{{}}}}}{}",
                        order, var, order, expr.to_latex()
                    )
                }
            }

            Expression::Integral { integrand, var, bounds } => {
                if let Some(bounds) = bounds {
                    format!(
                        r"\int_{{{}}}^{{{}}} {} \, d{}",
                        bounds.lower.to_latex(),
                        bounds.upper.to_latex(),
                        integrand.to_latex(),
                        var
                    )
                } else {
                    format!(r"\int {} \, d{}", integrand.to_latex(), var)
                }
            }

            Expression::Limit { expr, var, to, direction } => {
                let dir_str = direction.to_latex();
                format!(
                    r"\lim_{{{} \to {}{}}}{}",
                    var, to.to_latex(), dir_str, expr.to_latex()
                )
            }

            Expression::Sum { index, lower, upper, body } => {
                format!(
                    r"\sum_{{{}={}}}^{{{}}}{}",
                    index,
                    lower.to_latex(),
                    upper.to_latex(),
                    body.to_latex()
                )
            }

            Expression::Product { index, lower, upper, body } => {
                format!(
                    r"\prod_{{{}={}}}^{{{}}}{}",
                    index,
                    lower.to_latex(),
                    upper.to_latex(),
                    body.to_latex()
                )
            }

            Expression::Vector(elements) => {
                let elements_str = elements
                    .iter()
                    .map(|e| e.to_latex())
                    .collect::<Vec<_>>()
                    .join(r" \\ ");
                format!(r"\begin{{pmatrix}} {} \end{{pmatrix}}", elements_str)
            }

            Expression::Matrix(rows) => {
                let rows_str = rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|e| e.to_latex())
                            .collect::<Vec<_>>()
                            .join(" & ")
                    })
                    .collect::<Vec<_>>()
                    .join(r" \\ ");
                format!(r"\begin{{pmatrix}} {} \end{{pmatrix}}", rows_str)
            }

            Expression::Equation { left, right } => {
                format!("{} = {}", left.to_latex(), right.to_latex())
            }

            Expression::Inequality { op, left, right } => {
                format!("{} {} {}", left.to_latex(), op.to_latex(), right.to_latex())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for MathConstant
    #[test]
    fn test_math_constant_pi() {
        assert_eq!(MathConstant::Pi.to_latex(), r"\pi");
    }

    #[test]
    fn test_math_constant_e() {
        assert_eq!(MathConstant::E.to_latex(), "e");
    }

    #[test]
    fn test_math_constant_i() {
        assert_eq!(MathConstant::I.to_latex(), "i");
    }

    #[test]
    fn test_math_constant_infinity() {
        assert_eq!(MathConstant::Infinity.to_latex(), r"\infty");
    }

    #[test]
    fn test_math_constant_neg_infinity() {
        assert_eq!(MathConstant::NegInfinity.to_latex(), r"-\infty");
    }

    // Tests for BinaryOp
    #[test]
    fn test_binary_op_add() {
        assert_eq!(BinaryOp::Add.to_latex(), "+");
    }

    #[test]
    fn test_binary_op_sub() {
        assert_eq!(BinaryOp::Sub.to_latex(), "-");
    }

    #[test]
    fn test_binary_op_mul() {
        assert_eq!(BinaryOp::Mul.to_latex(), r"\cdot");
    }

    #[test]
    fn test_binary_op_pow() {
        assert_eq!(BinaryOp::Pow.to_latex(), "^");
    }

    #[test]
    fn test_binary_op_mod() {
        assert_eq!(BinaryOp::Mod.to_latex(), r"\bmod");
    }

    // Tests for InequalityOp
    #[test]
    fn test_inequality_op_lt() {
        assert_eq!(InequalityOp::Lt.to_latex(), "<");
    }

    #[test]
    fn test_inequality_op_le() {
        assert_eq!(InequalityOp::Le.to_latex(), r"\leq");
    }

    #[test]
    fn test_inequality_op_gt() {
        assert_eq!(InequalityOp::Gt.to_latex(), ">");
    }

    #[test]
    fn test_inequality_op_ge() {
        assert_eq!(InequalityOp::Ge.to_latex(), r"\geq");
    }

    #[test]
    fn test_inequality_op_ne() {
        assert_eq!(InequalityOp::Ne.to_latex(), r"\neq");
    }

    // Tests for Direction
    #[test]
    fn test_direction_left() {
        assert_eq!(Direction::Left.to_latex(), "^-");
    }

    #[test]
    fn test_direction_right() {
        assert_eq!(Direction::Right.to_latex(), "^+");
    }

    #[test]
    fn test_direction_both() {
        assert_eq!(Direction::Both.to_latex(), "");
    }

    // Tests for Expression - Basic Values
    #[test]
    fn test_integer() {
        let expr = Expression::Integer(42);
        assert_eq!(expr.to_latex(), "42");
    }

    #[test]
    fn test_integer_negative() {
        let expr = Expression::Integer(-17);
        assert_eq!(expr.to_latex(), "-17");
    }

    #[test]
    fn test_float() {
        let expr = Expression::Float(3.14.into());
        assert_eq!(expr.to_latex(), "3.14");
    }

    #[test]
    fn test_rational() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Integer(1)),
            denominator: Box::new(Expression::Integer(2)),
        };
        assert_eq!(expr.to_latex(), r"\frac{1}{2}");
    }

    #[test]
    fn test_rational_complex() {
        let expr = Expression::Rational {
            numerator: Box::new(Expression::Variable("a".to_string())),
            denominator: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(expr.to_latex(), r"\frac{a}{b}");
    }

    #[test]
    fn test_complex() {
        let expr = Expression::Complex {
            real: Box::new(Expression::Integer(3)),
            imaginary: Box::new(Expression::Integer(4)),
        };
        assert_eq!(expr.to_latex(), "3 + 4i");
    }

    #[test]
    fn test_variable() {
        let expr = Expression::Variable("x".to_string());
        assert_eq!(expr.to_latex(), "x");
    }

    #[test]
    fn test_constant_pi() {
        let expr = Expression::Constant(MathConstant::Pi);
        assert_eq!(expr.to_latex(), r"\pi");
    }

    // Tests for Binary Operations
    #[test]
    fn test_binary_add() {
        let expr = Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(2)),
            right: Box::new(Expression::Integer(3)),
        };
        assert_eq!(expr.to_latex(), "2 + 3");
    }

    #[test]
    fn test_binary_mul() {
        let expr = Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(expr.to_latex(), r"a \cdot b");
    }

    #[test]
    fn test_binary_div() {
        let expr = Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        };
        assert_eq!(expr.to_latex(), r"\frac{1}{2}");
    }

    #[test]
    fn test_binary_pow() {
        let expr = Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(2)),
        };
        assert_eq!(expr.to_latex(), "x^{2}");
    }

    #[test]
    fn test_binary_mod() {
        let expr = Expression::Binary {
            op: BinaryOp::Mod,
            left: Box::new(Expression::Integer(7)),
            right: Box::new(Expression::Integer(3)),
        };
        assert_eq!(expr.to_latex(), r"7 \bmod 3");
    }

    // Tests for Unary Operations
    #[test]
    fn test_unary_neg() {
        let expr = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Integer(5)),
        };
        assert_eq!(expr.to_latex(), "-5");
    }

    #[test]
    fn test_unary_factorial() {
        let expr = Expression::Unary {
            op: UnaryOp::Factorial,
            operand: Box::new(Expression::Variable("n".to_string())),
        };
        assert_eq!(expr.to_latex(), "n!");
    }

    #[test]
    fn test_unary_transpose() {
        let expr = Expression::Unary {
            op: UnaryOp::Transpose,
            operand: Box::new(Expression::Variable("A".to_string())),
        };
        assert_eq!(expr.to_latex(), "A^T");
    }

    // Tests for Functions
    #[test]
    fn test_function_sin() {
        let expr = Expression::Function {
            name: "sin".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.to_latex(), r"\sin\left(x\right)");
    }

    #[test]
    fn test_function_cos() {
        let expr = Expression::Function {
            name: "cos".to_string(),
            args: vec![Expression::Variable("theta".to_string())],
        };
        assert_eq!(expr.to_latex(), r"\cos\left(theta\right)");
    }

    #[test]
    fn test_function_unknown() {
        let expr = Expression::Function {
            name: "myfunction".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.to_latex(), r"\operatorname{myfunction}\left(x\right)");
    }

    #[test]
    fn test_function_sqrt_single_arg() {
        let expr = Expression::Function {
            name: "sqrt".to_string(),
            args: vec![Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.to_latex(), r"\sqrt{x}");
    }

    #[test]
    fn test_function_sqrt_two_args() {
        let expr = Expression::Function {
            name: "sqrt".to_string(),
            args: vec![Expression::Integer(3), Expression::Variable("x".to_string())],
        };
        assert_eq!(expr.to_latex(), r"\sqrt[3]{x}");
    }

    #[test]
    fn test_function_multiple_args() {
        let expr = Expression::Function {
            name: "max".to_string(),
            args: vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3),
            ],
        };
        assert_eq!(expr.to_latex(), r"\max\left(1, 2, 3\right)");
    }

    // Tests for Derivative
    #[test]
    fn test_derivative_first_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        assert_eq!(expr.to_latex(), r"\frac{d}{dx}f");
    }

    #[test]
    fn test_derivative_second_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 2,
        };
        assert_eq!(expr.to_latex(), r"\frac{d^{2}}{dx^{2}}f");
    }

    #[test]
    fn test_derivative_third_order() {
        let expr = Expression::Derivative {
            expr: Box::new(Expression::Variable("y".to_string())),
            var: "t".to_string(),
            order: 3,
        };
        assert_eq!(expr.to_latex(), r"\frac{d^{3}}{dt^{3}}y");
    }

    // Tests for PartialDerivative
    #[test]
    fn test_partial_derivative_first_order() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            order: 1,
        };
        assert_eq!(expr.to_latex(), r"\frac{\partial}{\partial x}f");
    }

    #[test]
    fn test_partial_derivative_second_order() {
        let expr = Expression::PartialDerivative {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "y".to_string(),
            order: 2,
        };
        assert_eq!(expr.to_latex(), r"\frac{\partial^{2}}{\partial y^{2}}f");
    }

    // Tests for Integral
    #[test]
    fn test_integral_indefinite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: None,
        };
        assert_eq!(expr.to_latex(), r"\int x \, dx");
    }

    #[test]
    fn test_integral_definite() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Variable("x".to_string())),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.to_latex(), r"\int_{0}^{1} x \, dx");
    }

    #[test]
    fn test_integral_complex_bounds() {
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("t".to_string())],
            }),
            var: "t".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Constant(MathConstant::Pi)),
            }),
        };
        assert_eq!(expr.to_latex(), r"\int_{0}^{\pi} \sin\left(t\right) \, dt");
    }

    // Tests for Limit
    #[test]
    fn test_limit_both() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Both,
        };
        assert_eq!(expr.to_latex(), r"\lim_{x \to 0}f");
    }

    #[test]
    fn test_limit_left() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Left,
        };
        assert_eq!(expr.to_latex(), r"\lim_{x \to 0^-}f");
    }

    #[test]
    fn test_limit_right() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Variable("f".to_string())),
            var: "x".to_string(),
            to: Box::new(Expression::Integer(0)),
            direction: Direction::Right,
        };
        assert_eq!(expr.to_latex(), r"\lim_{x \to 0^+}f");
    }

    #[test]
    fn test_limit_to_infinity() {
        let expr = Expression::Limit {
            expr: Box::new(Expression::Binary {
                op: BinaryOp::Div,
                left: Box::new(Expression::Integer(1)),
                right: Box::new(Expression::Variable("x".to_string())),
            }),
            var: "x".to_string(),
            to: Box::new(Expression::Constant(MathConstant::Infinity)),
            direction: Direction::Both,
        };
        assert_eq!(expr.to_latex(), r"\lim_{x \to \infty}\frac{1}{x}");
    }

    // Tests for Sum
    #[test]
    fn test_sum_simple() {
        let expr = Expression::Sum {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(expr.to_latex(), r"\sum_{i=1}^{n}i");
    }

    #[test]
    fn test_sum_complex_body() {
        let expr = Expression::Sum {
            index: "k".to_string(),
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(10)),
            body: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(Expression::Variable("k".to_string())),
                right: Box::new(Expression::Integer(2)),
            }),
        };
        assert_eq!(expr.to_latex(), r"\sum_{k=0}^{10}k^{2}");
    }

    // Tests for Product
    #[test]
    fn test_product_simple() {
        let expr = Expression::Product {
            index: "i".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Variable("n".to_string())),
            body: Box::new(Expression::Variable("i".to_string())),
        };
        assert_eq!(expr.to_latex(), r"\prod_{i=1}^{n}i");
    }

    #[test]
    fn test_product_complex() {
        let expr = Expression::Product {
            index: "j".to_string(),
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Integer(5)),
            body: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("j".to_string())),
                right: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.to_latex(), r"\prod_{j=1}^{5}j + 1");
    }

    // Tests for Vector
    #[test]
    fn test_vector_empty() {
        let expr = Expression::Vector(vec![]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix}  \end{pmatrix}");
    }

    #[test]
    fn test_vector_single() {
        let expr = Expression::Vector(vec![Expression::Integer(1)]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix} 1 \end{pmatrix}");
    }

    #[test]
    fn test_vector_multiple() {
        let expr = Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}");
    }

    // Tests for Matrix
    #[test]
    fn test_matrix_empty() {
        let expr = Expression::Matrix(vec![]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix}  \end{pmatrix}");
    }

    #[test]
    fn test_matrix_1x1() {
        let expr = Expression::Matrix(vec![vec![Expression::Integer(1)]]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix} 1 \end{pmatrix}");
    }

    #[test]
    fn test_matrix_2x2() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
        ]);
        assert_eq!(expr.to_latex(), r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}");
    }

    #[test]
    fn test_matrix_3x2() {
        let expr = Expression::Matrix(vec![
            vec![Expression::Integer(1), Expression::Integer(2)],
            vec![Expression::Integer(3), Expression::Integer(4)],
            vec![Expression::Integer(5), Expression::Integer(6)],
        ]);
        assert_eq!(
            expr.to_latex(),
            r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}"
        );
    }

    // Tests for Equation
    #[test]
    fn test_equation_simple() {
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        assert_eq!(expr.to_latex(), "x = 5");
    }

    #[test]
    fn test_equation_complex() {
        let expr = Expression::Equation {
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expression::Integer(2)),
                    right: Box::new(Expression::Variable("x".to_string())),
                }),
                right: Box::new(Expression::Integer(1)),
            }),
        };
        assert_eq!(expr.to_latex(), r"y = 2 \cdot x + 1");
    }

    // Tests for Inequality
    #[test]
    fn test_inequality_lt() {
        let expr = Expression::Inequality {
            op: InequalityOp::Lt,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(5)),
        };
        assert_eq!(expr.to_latex(), "x < 5");
    }

    #[test]
    fn test_inequality_le() {
        let expr = Expression::Inequality {
            op: InequalityOp::Le,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Integer(10)),
        };
        assert_eq!(expr.to_latex(), r"x \leq 10");
    }

    #[test]
    fn test_inequality_ge() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ge,
            left: Box::new(Expression::Variable("y".to_string())),
            right: Box::new(Expression::Integer(0)),
        };
        assert_eq!(expr.to_latex(), r"y \geq 0");
    }

    #[test]
    fn test_inequality_ne() {
        let expr = Expression::Inequality {
            op: InequalityOp::Ne,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        };
        assert_eq!(expr.to_latex(), r"a \neq b");
    }

    // Complex nested expressions
    #[test]
    fn test_nested_expression() {
        // (a + b) / (c - d)
        let expr = Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Sub,
                left: Box::new(Expression::Variable("c".to_string())),
                right: Box::new(Expression::Variable("d".to_string())),
            }),
        };
        assert_eq!(expr.to_latex(), r"\frac{a + b}{c - d}");
    }

    #[test]
    fn test_complex_calculus_expression() {
        // ∫₀^π sin(x) dx
        let expr = Expression::Integral {
            integrand: Box::new(Expression::Function {
                name: "sin".to_string(),
                args: vec![Expression::Variable("x".to_string())],
            }),
            var: "x".to_string(),
            bounds: Some(IntegralBounds {
                lower: Box::new(Expression::Integer(0)),
                upper: Box::new(Expression::Constant(MathConstant::Pi)),
            }),
        };
        assert_eq!(expr.to_latex(), r"\int_{0}^{\pi} \sin\left(x\right) \, dx");
    }
}
