use super::*;

#[test]
fn test_parse_simple_number() {
    let expr = parse_latex("42").unwrap();
    assert_eq!(expr, Expression::Integer(42));
}

#[test]
fn test_parse_float() {
    let expr = parse_latex("3.14").unwrap();
    match expr {
        Expression::Float(f) => {
            assert!((f.value() - 3.14).abs() < 1e-10);
        }
        _ => panic!("Expected float"),
    }
}

#[test]
fn test_parse_variable() {
    let expr = parse_latex("x").unwrap();
    assert_eq!(expr, Expression::Variable("x".to_string()));
}

#[test]
fn test_parse_addition() {
    let expr = parse_latex("1 + 2").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Integer(1));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_subtraction() {
    let expr = parse_latex("5 - 3").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Sub);
            assert_eq!(*left, Expression::Integer(5));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_multiplication() {
    let expr = parse_latex("2 * 3").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_division() {
    let expr = parse_latex("6 / 2").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Div);
            assert_eq!(*left, Expression::Integer(6));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_power() {
    let expr = parse_latex("x^2").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Pow);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_power_braced() {
    let expr = parse_latex("x^{2+3}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in exponent"),
            }
        }
        _ => panic!("Expected power expression"),
    }
}

#[test]
fn test_parse_frac() {
    let expr = parse_latex(r"\frac{1}{2}").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Div);
            assert_eq!(*left, Expression::Integer(1));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary division"),
    }
}

#[test]
fn test_parse_sqrt() {
    let expr = parse_latex(r"\sqrt{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_sqrt_nth() {
    let expr = parse_latex(r"\sqrt[3]{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Integer(3));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_subscript() {
    let expr = parse_latex("x_1").unwrap();
    assert_eq!(expr, Expression::Variable("x_1".to_string()));
}

#[test]
fn test_parse_subscript_braced() {
    let expr = parse_latex("x_{12}").unwrap();
    assert_eq!(expr, Expression::Variable("x_12".to_string()));
}

#[test]
fn test_parse_greek_letter() {
    let expr = parse_latex(r"\alpha").unwrap();
    assert_eq!(expr, Expression::Variable("alpha".to_string()));
}

#[test]
fn test_parse_pi_constant() {
    let expr = parse_latex(r"\pi").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::Pi));
}

#[test]
fn test_parse_infinity() {
    let expr = parse_latex(r"\infty").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
}

#[test]
fn test_parse_sin() {
    let expr = parse_latex(r"\sin{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_sin_unbraced() {
    let expr = parse_latex(r"\sin x").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_sin_parentheses() {
    let expr = parse_latex(r"\sin(x)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_parentheses() {
    let expr = parse_latex("(1 + 2)").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Integer(1));
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_parse_absolute_value() {
    let expr = parse_latex("|x|").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "abs");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_parse_unary_minus() {
    let expr = parse_latex("-x").unwrap();
    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, crate::ast::UnaryOp::Neg);
            assert_eq!(*operand, Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected unary expression"),
    }
}

#[test]
fn test_parse_complex_expression() {
    // (2 + 3) * 4
    let expr = parse_latex("(2 + 3) * 4").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            match *left {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in left"),
            }
            assert_eq!(*right, Expression::Integer(4));
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_operator_precedence() {
    // 2 + 3 * 4 should be 2 + (3 * 4)
    let expr = parse_latex("2 + 3 * 4").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication in right"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_power_precedence() {
    // 2 * x^3 should be 2 * (x^3)
    let expr = parse_latex("2 * x^3").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            match *right {
                Expression::Binary {
                    op: BinaryOp::Pow, ..
                } => {}
                _ => panic!("Expected power in right"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_nested_frac() {
    // \frac{\frac{1}{2}}{3}
    let expr = parse_latex(r"\frac{\frac{1}{2}}{3}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            match *left {
                Expression::Binary {
                    op: BinaryOp::Div, ..
                } => {}
                _ => panic!("Expected nested division"),
            }
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected division"),
    }
}

// Relation tests

#[test]
fn test_latex_simple_equation() {
    let expr = parse_latex("x = 5").unwrap();
    match expr {
        Expression::Equation { left, right } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(5));
        }
        _ => panic!("Expected Equation variant"),
    }
}

#[test]
fn test_latex_inequality_less() {
    let expr = parse_latex("x < 5").unwrap();
    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Lt);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(5));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_less_command() {
    let expr = parse_latex(r"x \lt 5").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Lt);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_greater() {
    let expr = parse_latex("x > 0").unwrap();
    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Gt);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(0));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_greater_command() {
    let expr = parse_latex(r"x \gt 0").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Gt);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_leq() {
    let expr = parse_latex(r"x \leq 3").unwrap();
    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Le);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_le() {
    let expr = parse_latex(r"x \le 3").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Le);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_geq() {
    let expr = parse_latex(r"x \geq -1").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Ge);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_ge() {
    let expr = parse_latex(r"x \ge -1").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Ge);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_neq() {
    let expr = parse_latex(r"x \neq 0").unwrap();
    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Ne);
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Integer(0));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_inequality_ne() {
    let expr = parse_latex(r"a \ne b").unwrap();
    match expr {
        Expression::Inequality { op, .. } => {
            assert_eq!(op, InequalityOp::Ne);
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_complex_equation() {
    // \frac{x}{2} = 3
    let expr = parse_latex(r"\frac{x}{2} = 3").unwrap();
    match expr {
        Expression::Equation { left, right } => {
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected Equation variant"),
    }
}

#[test]
fn test_latex_complex_inequality() {
    // a + b < c + d
    let expr = parse_latex("a + b < c + d").unwrap();
    match expr {
        Expression::Inequality { op, left, right } => {
            assert_eq!(op, InequalityOp::Lt);
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
        }
        _ => panic!("Expected Inequality variant"),
    }
}

#[test]
fn test_latex_chained_relation_error() {
    // a < b < c should error
    let result = parse_latex("a < b < c");
    assert!(result.is_err());
    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("chained relations"));
    }
}

#[test]
fn test_latex_relation_precedence() {
    // 2 + 3 = 5 should parse as (2 + 3) = 5
    let expr = parse_latex("2 + 3 = 5").unwrap();
    match expr {
        Expression::Equation { left, right } => {
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert_eq!(*right, Expression::Integer(5));
        }
        _ => panic!("Expected Equation variant"),
    }
}

// Matrix and vector tests
#[test]
fn test_parse_empty_matrix() {
    let expr = parse_latex(r"\begin{matrix}\end{matrix}").unwrap();
    assert_eq!(expr, Expression::Matrix(vec![]));
}

#[test]
fn test_parse_1x1_matrix() {
    let expr = parse_latex(r"\begin{matrix}1\end{matrix}").unwrap();
    assert_eq!(expr, Expression::Vector(vec![Expression::Integer(1)]));
}

#[test]
fn test_parse_column_vector() {
    let expr = parse_latex(r"\begin{matrix}1 \\ 2 \\ 3\end{matrix}").unwrap();
    assert_eq!(
        expr,
        Expression::Vector(vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3)
        ])
    );
}

#[test]
fn test_parse_row_matrix() {
    let expr = parse_latex(r"\begin{matrix}1 & 2 & 3\end{matrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[0][0], Expression::Integer(1));
            assert_eq!(rows[0][1], Expression::Integer(2));
            assert_eq!(rows[0][2], Expression::Integer(3));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_2x2_matrix() {
    let expr = parse_latex(r"\begin{matrix}1 & 2 \\ 3 & 4\end{matrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
            assert_eq!(rows[1].len(), 2);
            assert_eq!(rows[0][0], Expression::Integer(1));
            assert_eq!(rows[0][1], Expression::Integer(2));
            assert_eq!(rows[1][0], Expression::Integer(3));
            assert_eq!(rows[1][1], Expression::Integer(4));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_bmatrix() {
    let expr = parse_latex(r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_pmatrix() {
    let expr = parse_latex(r"\begin{pmatrix}x & y \\ z & w\end{pmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0][0], Expression::Variable("x".to_string()));
            assert_eq!(rows[0][1], Expression::Variable("y".to_string()));
            assert_eq!(rows[1][0], Expression::Variable("z".to_string()));
            assert_eq!(rows[1][1], Expression::Variable("w".to_string()));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_vmatrix() {
    let expr = parse_latex(r"\begin{vmatrix}a & b \\ c & d\end{vmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_big_bmatrix() {
    let expr = parse_latex(r"\begin{Bmatrix}1 & 2\end{Bmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_big_vmatrix() {
    let expr = parse_latex(r"\begin{Vmatrix}1\end{Vmatrix}").unwrap();
    assert_eq!(expr, Expression::Vector(vec![Expression::Integer(1)]));
}

#[test]
fn test_parse_matrix_with_expressions() {
    let expr = parse_latex(r"\begin{matrix}x+1 & 2 \\ 3 & y^2\end{matrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
            // First element should be x+1
            match &rows[0][0] {
                Expression::Binary { op, .. } => assert_eq!(*op, BinaryOp::Add),
                _ => panic!("Expected binary expression"),
            }
            // Last element should be y^2
            match &rows[1][1] {
                Expression::Binary { op, .. } => assert_eq!(*op, BinaryOp::Pow),
                _ => panic!("Expected binary expression"),
            }
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_3x3_matrix() {
    let expr =
        parse_latex(r"\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 3);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[1].len(), 3);
            assert_eq!(rows[2].len(), 3);
            assert_eq!(rows[2][2], Expression::Integer(9));
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_matrix_trailing_backslash() {
    // Matrix with trailing \\ should not add empty row
    let expr = parse_latex(r"\begin{matrix}1 \\ 2 \\\end{matrix}").unwrap();
    match expr {
        Expression::Vector(elements) => {
            assert_eq!(elements.len(), 2);
            assert_eq!(elements[0], Expression::Integer(1));
            assert_eq!(elements[1], Expression::Integer(2));
        }
        _ => panic!("Expected Vector variant"),
    }
}

#[test]
fn test_parse_ragged_matrix_error() {
    let result = parse_latex(r"\begin{matrix}1 & 2 \\ 3\end{matrix}");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("inconsistent matrix row lengths"));
    }
}

#[test]
fn test_parse_mismatched_environment_error() {
    let result = parse_latex(r"\begin{matrix}1\end{bmatrix}");
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("mismatched environment"));
    }
}

#[test]
fn test_parse_invalid_matrix_environment() {
    let result = parse_latex(r"\begin{invalid}1\end{invalid}");
    assert!(result.is_err());
}

#[test]
fn test_parse_matrix_with_floats() {
    let expr = parse_latex(r"\begin{matrix}1.5 & 2.7 \\ 3.2 & 4.9\end{matrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            match &rows[0][0] {
                Expression::Float(f) => assert!((f.value() - 1.5).abs() < 1e-10),
                _ => panic!("Expected float"),
            }
        }
        _ => panic!("Expected Matrix variant"),
    }
}

#[test]
fn test_parse_matrix_mixed_types() {
    let expr = parse_latex(r"\begin{pmatrix}1 & x & 2.5 \\ y & 3 & z\end{pmatrix}").unwrap();
    match expr {
        Expression::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 3);
            assert_eq!(rows[0][0], Expression::Integer(1));
            assert_eq!(rows[0][1], Expression::Variable("x".to_string()));
            match &rows[0][2] {
                Expression::Float(f) => assert!((f.value() - 2.5).abs() < 1e-10),
                _ => panic!("Expected float"),
            }
        }
        _ => panic!("Expected Matrix variant"),
    }
}

// Calculus tests

// Derivative tests
#[test]
fn test_parse_derivative_first_order() {
    let expr = parse_latex(r"\frac{d}{d*x}x").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            assert_eq!(*expr, Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_parse_derivative_second_order() {
    let expr = parse_latex(r"\frac{d^2}{d*x^2}f").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 2);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Derivative variant"),
    }
}

// Partial derivative tests
#[test]
fn test_parse_partial_derivative_first_order() {
    let expr = parse_latex(r"\frac{\partial}{\partial * x}f").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_parse_partial_derivative_second_order() {
    let expr = parse_latex(r"\frac{\partial^2}{\partial * x^2}f").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 2);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

// Test that regular fractions still work
#[test]
fn test_parse_frac_not_derivative() {
    let expr = parse_latex(r"\frac{x+1}{y-2}").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Div);
            assert!(matches!(*left, Expression::Binary { .. }));
            assert!(matches!(*right, Expression::Binary { .. }));
        }
        _ => panic!("Expected Binary division"),
    }
}

// Integral tests
#[test]
fn test_parse_integral_indefinite() {
    let expr = parse_latex(r"\int x dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*integrand, Expression::Variable("x".to_string()));
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_parse_integral_definite() {
    let expr = parse_latex(r"\int_0^1 x dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*integrand, Expression::Variable("x".to_string()));
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            assert_eq!(*bounds.lower, Expression::Integer(0));
            assert_eq!(*bounds.upper, Expression::Integer(1));
        }
        _ => panic!("Expected Integral variant"),
    }
}

// Limit tests
#[test]
fn test_parse_limit_both_sides() {
    let expr = parse_latex(r"\lim_{x \to 0} x").unwrap();
    match expr {
        Expression::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*to, Expression::Integer(0));
            assert_eq!(direction, Direction::Both);
            assert_eq!(*expr, Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_parse_limit_from_right() {
    let expr = parse_latex(r"\lim_{x \to 0^+} x").unwrap();
    match expr {
        Expression::Limit {
            expr: _,
            var,
            to,
            direction,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*to, Expression::Integer(0));
            assert_eq!(direction, Direction::Right);
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_parse_limit_from_left() {
    let expr = parse_latex(r"\lim_{x \to 0^-} x").unwrap();
    match expr {
        Expression::Limit {
            expr: _,
            var,
            to,
            direction,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*to, Expression::Integer(0));
            assert_eq!(direction, Direction::Left);
        }
        _ => panic!("Expected Limit variant"),
    }
}

// Sum tests
#[test]
fn test_parse_sum_simple() {
    let expr = parse_latex(r"\sum_{i=1}^{n} i").unwrap();
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert_eq!(*lower, Expression::Integer(1));
            assert_eq!(*upper, Expression::Variable("n".to_string()));
            assert_eq!(*body, Expression::Variable("i".to_string()));
        }
        _ => panic!("Expected Sum variant"),
    }
}

// Product tests
#[test]
fn test_parse_product_simple() {
    let expr = parse_latex(r"\prod_{i=1}^{n} i").unwrap();
    match expr {
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert_eq!(*lower, Expression::Integer(1));
            assert_eq!(*upper, Expression::Variable("n".to_string()));
            assert_eq!(*body, Expression::Variable("i".to_string()));
        }
        _ => panic!("Expected Product variant"),
    }
}

// Multiplication command tests
#[test]
fn test_parse_cdot_multiplication() {
    let expr = parse_latex(r"a \cdot b").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Variable("a".to_string()));
            assert_eq!(*right, Expression::Variable("b".to_string()));
        }
        _ => panic!("Expected binary multiplication"),
    }
}

#[test]
fn test_parse_times_cross_product() {
    // \times is parsed as CrossProduct in mathematical notation
    // Consuming libraries can interpret this as scalar multiplication when operands are scalars
    let expr = parse_latex(r"2 \times 3").unwrap();
    match expr {
        Expression::CrossProduct { left, right } => {
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected cross product, got {:?}", expr),
    }
}

#[test]
fn test_parse_cdot_complex_expression() {
    let expr = parse_latex(r"2 \cdot x + 3").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            match *left {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left: ref l,
                    right: ref r,
                } => {
                    assert_eq!(**l, Expression::Integer(2));
                    assert_eq!(**r, Expression::Variable("x".to_string()));
                }
                _ => panic!("Expected multiplication in left operand"),
            }
            assert_eq!(*right, Expression::Integer(3));
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_parse_times_with_parentheses() {
    // \times is parsed as CrossProduct
    let expr = parse_latex(r"(a + b) \times (c - d)").unwrap();
    match expr {
        Expression::CrossProduct { left, right } => {
            assert!(matches!(
                *left,
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert!(matches!(
                *right,
                Expression::Binary {
                    op: BinaryOp::Sub,
                    ..
                }
            ));
        }
        _ => panic!("Expected cross product, got {:?}", expr),
    }
}

#[test]
fn test_parse_mixed_multiplication_operators() {
    // Test that * and \cdot both work
    let expr1 = parse_latex(r"a * b").unwrap();
    let expr2 = parse_latex(r"a \cdot b").unwrap();

    assert_eq!(expr1, expr2);
}
