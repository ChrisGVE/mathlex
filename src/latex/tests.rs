#![allow(clippy::approx_constant)]

use super::trait_def::ToLatex;
use crate::ast::{
    BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, MathConstant, UnaryOp,
};

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
    assert_eq!(expr.to_latex(), r"\cos\left(\theta\right)");
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
        args: vec![
            Expression::Integer(3),
            Expression::Variable("x".to_string()),
        ],
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
    assert_eq!(expr.to_latex(), r"\int x dx");
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
    assert_eq!(expr.to_latex(), r"\int_{0}^{1} x dx");
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
    assert_eq!(expr.to_latex(), r"\int_{0}^{\pi} \sin\left(t\right) dt");
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
    assert_eq!(
        expr.to_latex(),
        r"\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}"
    );
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
    assert_eq!(
        expr.to_latex(),
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
    );
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
    assert_eq!(expr.to_latex(), r"\int_{0}^{\pi} \sin\left(x\right) dx");
}

// Tests for precedence-safe parentheses

#[test]
fn test_latex_unary_neg_with_binary_operand() {
    // -(a + b) should output as "-\left(a + b\right)"
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"-\left(a + b\right)");
}

#[test]
fn test_latex_unary_pos_with_binary_operand() {
    // +(a * b) should output as "+\left(a \cdot b\right)"
    let expr = Expression::Unary {
        op: UnaryOp::Pos,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"+\left(a \cdot b\right)");
}

#[test]
fn test_latex_factorial_with_binary_operand() {
    // (a + b)! should output as "\left(a + b\right)!"
    let expr = Expression::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"\left(a + b\right)!");
}

#[test]
fn test_latex_transpose_with_binary_operand() {
    // (A + B)' should output as "\left(A + B\right)^T"
    let expr = Expression::Unary {
        op: UnaryOp::Transpose,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("A".to_string())),
            right: Box::new(Expression::Variable("B".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"\left(A + B\right)^T");
}

#[test]
fn test_latex_power_left_associative() {
    // (a^b)^c should output with parens on left: "\left(a^{b}\right)^{c}"
    let expr = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
        right: Box::new(Expression::Variable("c".to_string())),
    };
    assert_eq!(expr.to_latex(), r"\left(a^{b}\right)^{c}");
}

#[test]
fn test_latex_power_right_associative() {
    // a^(b^c) - in LaTeX, parens are added inside the braces for clarity
    let expr = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Variable("a".to_string())),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(Expression::Variable("b".to_string())),
            right: Box::new(Expression::Variable("c".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"a^{\left(b^{c}\right)}");
}

#[test]
fn test_latex_precedence_add_mul() {
    // (a + b) * c should output with parens: "\left(a + b\right) \cdot c"
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
        right: Box::new(Expression::Variable("c".to_string())),
    };
    assert_eq!(expr.to_latex(), r"\left(a + b\right) \cdot c");
}

#[test]
fn test_latex_precedence_sub_sub_right() {
    // a - (b - c) should output with parens: "a - \left(b - c\right)"
    let expr = Expression::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::Variable("a".to_string())),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Variable("b".to_string())),
            right: Box::new(Expression::Variable("c".to_string())),
        }),
    };
    assert_eq!(expr.to_latex(), r"a - \left(b - c\right)");
}

#[test]
fn test_latex_precedence_sub_sub_left() {
    // (a - b) - c should output without parens: "a - b - c"
    let expr = Expression::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
        right: Box::new(Expression::Variable("c".to_string())),
    };
    assert_eq!(expr.to_latex(), r"a - b - c");
}

#[test]
fn test_latex_unary_with_non_binary_operand() {
    // -x should output as "-x" (no parens needed)
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Variable("x".to_string())),
    };
    assert_eq!(expr.to_latex(), "-x");
}

#[test]
fn test_latex_factorial_with_non_binary_operand() {
    // n! should output as "n!" (no parens needed)
    let expr = Expression::Unary {
        op: UnaryOp::Factorial,
        operand: Box::new(Expression::Variable("n".to_string())),
    };
    assert_eq!(expr.to_latex(), "n!");
}

#[test]
fn test_latex_complex_precedence_example() {
    // -(a + b) * c should output as "-\left(a + b\right) \cdot c"
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Expression::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::Variable("a".to_string())),
                right: Box::new(Expression::Variable("b".to_string())),
            }),
        }),
        right: Box::new(Expression::Variable("c".to_string())),
    };
    assert_eq!(expr.to_latex(), r"-\left(a + b\right) \cdot c");
}
