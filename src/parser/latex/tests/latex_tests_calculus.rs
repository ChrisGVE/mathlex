// Calculus tests for LaTeX parser
use super::*;

// Derivative tests

#[test]
fn test_derivative_first_order() {
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
fn test_derivative_second_order() {
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

#[test]
fn test_derivative_third_order() {
    let expr = parse_latex(r"\frac{d^3}{d*x^3}f").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 3);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_derivative_of_expression() {
    let expr = parse_latex(r"\frac{d}{d*x}(x^2+1)").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            match *expr {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in derivative"),
            }
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_derivative_of_function() {
    let expr = parse_latex(r"\frac{d}{d*x}\sin{x}").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            match *expr {
                Expression::Function { name, .. } => assert_eq!(name, "sin"),
                _ => panic!("Expected function in derivative"),
            }
        }
        _ => panic!("Expected Derivative variant"),
    }
}

#[test]
fn test_derivative_different_variable() {
    let expr = parse_latex(r"\frac{d}{d*t}f").unwrap();
    match expr {
        Expression::Derivative { expr, var, order } => {
            assert_eq!(var, "t");
            assert_eq!(order, 1);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected Derivative variant"),
    }
}

// Partial derivative tests

#[test]
fn test_partial_derivative_first_order() {
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
fn test_partial_derivative_second_order() {
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

#[test]
fn test_partial_derivative_second_order_standard_notation() {
    // Standard notation without explicit multiplication: \frac{\partial^2}{\partial x^2}
    let expr = parse_latex(r"\frac{\partial^2}{\partial x^2}f").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 2);
            assert_eq!(*expr, Expression::Variable("f".to_string()));
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_partial_derivative_third_order() {
    let expr = parse_latex(r"\frac{\partial^3}{\partial * y^3}g").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "y");
            assert_eq!(order, 3);
            assert_eq!(*expr, Expression::Variable("g".to_string()));
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_partial_derivative_third_order_standard_notation() {
    // Standard notation without explicit multiplication: \frac{\partial^3}{\partial y^3}
    let expr = parse_latex(r"\frac{\partial^3}{\partial y^3}g").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "y");
            assert_eq!(order, 3);
            assert_eq!(*expr, Expression::Variable("g".to_string()));
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_partial_derivative_complex_expression() {
    let expr = parse_latex(r"\frac{\partial}{\partial x}(x*y + y^2)").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            match *expr {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition"),
            }
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

#[test]
fn test_partial_derivative_of_expression() {
    let expr = parse_latex(r"\frac{\partial}{\partial * x}(x*y)").unwrap();
    match expr {
        Expression::PartialDerivative { expr, var, order } => {
            assert_eq!(var, "x");
            assert_eq!(order, 1);
            match *expr {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication"),
            }
        }
        _ => panic!("Expected PartialDerivative variant"),
    }
}

// Integral tests

#[test]
fn test_integral_indefinite_simple() {
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
fn test_integral_indefinite_function() {
    let expr = parse_latex(r"\int \sin{x} dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            match *integrand {
                Expression::Function { name, .. } => assert_eq!(name, "sin"),
                _ => panic!("Expected function"),
            }
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_indefinite_expression() {
    let expr = parse_latex(r"\int (x^2 + 1) dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            match *integrand {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition"),
            }
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_definite_simple() {
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

#[test]
fn test_integral_definite_negative_bounds() {
    let expr = parse_latex(r"\int_{-1}^{1} x dx").unwrap();
    match expr {
        Expression::Integral {
            integrand: _,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            match *bounds.lower {
                Expression::Unary {
                    op: crate::ast::UnaryOp::Neg,
                    operand,
                } => {
                    assert_eq!(*operand, Expression::Integer(1));
                }
                _ => panic!("Expected unary negation"),
            }
            assert_eq!(*bounds.upper, Expression::Integer(1));
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_to_infinity() {
    let expr = parse_latex(r"\int_0^\infty x dx").unwrap();
    match expr {
        Expression::Integral {
            integrand: _,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            assert_eq!(*bounds.lower, Expression::Integer(0));
            assert_eq!(*bounds.upper, Expression::Constant(MathConstant::Infinity));
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_variable_bounds() {
    let expr = parse_latex(r"\int_a^b f dx").unwrap();
    match expr {
        Expression::Integral {
            integrand: _,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            assert_eq!(*bounds.lower, Expression::Variable("a".to_string()));
            assert_eq!(*bounds.upper, Expression::Variable("b".to_string()));
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_complex_integrand() {
    let expr = parse_latex(r"\int_0^1 (2*x + 3) dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            match *integrand {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition"),
            }
            assert!(bounds.is_some());
            let bounds = bounds.unwrap();
            assert_eq!(*bounds.lower, Expression::Integer(0));
            assert_eq!(*bounds.upper, Expression::Integer(1));
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_different_variable() {
    let expr = parse_latex(r"\int t dt").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "t");
            assert_eq!(*integrand, Expression::Variable("t".to_string()));
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_with_addition() {
    // \int x dx + 1 parses as (\int x dx) + 1
    // To include addition in integrand, use parentheses: \int (x + 1) dx
    let expr = parse_latex(r"\int x dx + 1").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            // Left side should be the integral
            match *left {
                Expression::Integral {
                    integrand,
                    var,
                    bounds,
                } => {
                    assert_eq!(var, "x");
                    assert_eq!(*integrand, Expression::Variable("x".to_string()));
                    assert!(bounds.is_none());
                }
                _ => panic!("Expected Integral on left side"),
            }
            // Right side should be 1
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected Binary Add variant"),
    }
}

#[test]
fn test_integral_with_multiplication() {
    // \int x * y dx should parse correctly
    let expr = parse_latex(r"\int x * y dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            match *integrand {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication in integrand"),
            }
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_with_explicit_parentheses() {
    // \int (x + 1) dx should still work correctly
    let expr = parse_latex(r"\int (x + 1) dx").unwrap();
    match expr {
        Expression::Integral {
            integrand,
            var,
            bounds,
        } => {
            assert_eq!(var, "x");
            match *integrand {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Variable("x".to_string()));
                    assert_eq!(*right, Expression::Integer(1));
                }
                _ => panic!("Expected addition in integrand"),
            }
            assert!(bounds.is_none());
        }
        _ => panic!("Expected Integral variant"),
    }
}

#[test]
fn test_integral_definite_with_addition() {
    // \int_0^1 x dx + 1 parses as (\int_0^1 x dx) + 1
    // To include addition in integrand, use parentheses: \int_0^1 (x + 1) dx
    let expr = parse_latex(r"\int_0^1 x dx + 1").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            // Left side should be the definite integral
            match *left {
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
                _ => panic!("Expected Integral on left side"),
            }
            // Right side should be 1
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected Binary Add variant"),
    }
}

// Limit tests

#[test]
fn test_limit_both_sides() {
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
fn test_limit_from_right() {
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
fn test_limit_from_left() {
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

#[test]
fn test_limit_to_infinity() {
    let expr = parse_latex(r"\lim_{x \to \infty} x").unwrap();
    match expr {
        Expression::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            assert_eq!(var, "x");
            assert_eq!(*to, Expression::Constant(MathConstant::Infinity));
            assert_eq!(direction, Direction::Both);
            assert_eq!(*expr, Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_limit_of_expression() {
    let expr = parse_latex(r"\lim_{x \to 0} (x^2 + 1)").unwrap();
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
            match *expr {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition"),
            }
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_limit_of_function() {
    let expr = parse_latex(r"\lim_{x \to 0} \sin{x}").unwrap();
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
            match *expr {
                Expression::Function { name, .. } => assert_eq!(name, "sin"),
                _ => panic!("Expected function"),
            }
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_limit_of_fraction() {
    let expr = parse_latex(r"\lim_{x \to 0} \frac{1}{x}").unwrap();
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
            match *expr {
                Expression::Binary {
                    op: BinaryOp::Div, ..
                } => {}
                _ => panic!("Expected division"),
            }
        }
        _ => panic!("Expected Limit variant"),
    }
}

#[test]
fn test_limit_different_variable() {
    let expr = parse_latex(r"\lim_{t \to 1} t").unwrap();
    match expr {
        Expression::Limit {
            expr,
            var,
            to,
            direction,
        } => {
            assert_eq!(var, "t");
            assert_eq!(*to, Expression::Integer(1));
            assert_eq!(direction, Direction::Both);
            assert_eq!(*expr, Expression::Variable("t".to_string()));
        }
        _ => panic!("Expected Limit variant"),
    }
}

// Sum tests

#[test]
fn test_sum_simple() {
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

#[test]
fn test_sum_zero_to_n() {
    let expr = parse_latex(r"\sum_{i=0}^{n} i").unwrap();
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert_eq!(*lower, Expression::Integer(0));
            assert_eq!(*upper, Expression::Variable("n".to_string()));
            assert_eq!(*body, Expression::Variable("i".to_string()));
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_sum_complex_body() {
    let expr = parse_latex(r"\sum_{i=1}^{n} (i^2)").unwrap();
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
            match *body {
                Expression::Binary {
                    op: BinaryOp::Pow, ..
                } => {}
                _ => panic!("Expected power"),
            }
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_sum_expression_bounds() {
    let expr = parse_latex(r"\sum_{i=1}^{n+1} i").unwrap();
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert_eq!(*lower, Expression::Integer(1));
            match *upper {
                Expression::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                _ => panic!("Expected addition in upper bound"),
            }
            assert_eq!(*body, Expression::Variable("i".to_string()));
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_sum_different_index() {
    let expr = parse_latex(r"\sum_{k=0}^{m} k").unwrap();
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "k");
            assert_eq!(*lower, Expression::Integer(0));
            assert_eq!(*upper, Expression::Variable("m".to_string()));
            assert_eq!(*body, Expression::Variable("k".to_string()));
        }
        _ => panic!("Expected Sum variant"),
    }
}

// Product tests

#[test]
fn test_product_simple() {
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

#[test]
fn test_product_complex_body() {
    let expr = parse_latex(r"\prod_{i=1}^{n} (2*i)").unwrap();
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
            match *body {
                Expression::Binary {
                    op: BinaryOp::Mul, ..
                } => {}
                _ => panic!("Expected multiplication"),
            }
        }
        _ => panic!("Expected Product variant"),
    }
}

#[test]
fn test_product_expression_bounds() {
    let expr = parse_latex(r"\prod_{k=2}^{n-1} k").unwrap();
    match expr {
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "k");
            assert_eq!(*lower, Expression::Integer(2));
            match *upper {
                Expression::Binary {
                    op: BinaryOp::Sub, ..
                } => {}
                _ => panic!("Expected subtraction in upper bound"),
            }
            assert_eq!(*body, Expression::Variable("k".to_string()));
        }
        _ => panic!("Expected Product variant"),
    }
}

#[test]
fn test_product_different_index() {
    let expr = parse_latex(r"\prod_{j=0}^{m} j").unwrap();
    match expr {
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "j");
            assert_eq!(*lower, Expression::Integer(0));
            assert_eq!(*upper, Expression::Variable("m".to_string()));
            assert_eq!(*body, Expression::Variable("j".to_string()));
        }
        _ => panic!("Expected Product variant"),
    }
}

// Tests for sum/product body parsing - capturing powers without parentheses

#[test]
fn test_sum_power_without_parens() {
    // \sum_{i=1}^{n} i^2 should parse as sum of i^2, not (sum i)^2
    let expr = parse_latex(r"\sum_{i=1}^{n} i^2").unwrap();
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
            // Body should be i^2
            match *body {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Variable("i".to_string()));
                    assert_eq!(*right, Expression::Integer(2));
                }
                _ => panic!("Expected power expression in body"),
            }
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_sum_addition_outside() {
    // \sum_{i=1}^{n} i + 1 should parse as (sum i) + 1, not sum(i+1)
    let expr = parse_latex(r"\sum_{i=1}^{n} i + 1").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            // Left should be the sum
            match *left {
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
                _ => panic!("Expected Sum in left operand"),
            }
            // Right should be 1
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected Binary addition"),
    }
}

#[test]
fn test_product_power_without_parens() {
    // \prod_{i=1}^{n} i^2 should parse as product of i^2
    let expr = parse_latex(r"\prod_{i=1}^{n} i^2").unwrap();
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
            // Body should be i^2
            match *body {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Variable("i".to_string()));
                    assert_eq!(*right, Expression::Integer(2));
                }
                _ => panic!("Expected power expression in body"),
            }
        }
        _ => panic!("Expected Product variant"),
    }
}

#[test]
fn test_product_multiplication_in_body() {
    // \prod_{i=1}^{n} 2*i should parse as product of (2*i)
    let expr = parse_latex(r"\prod_{i=1}^{n} 2*i").unwrap();
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
            // Body should be 2*i
            match *body {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Integer(2));
                    assert_eq!(*right, Expression::Variable("i".to_string()));
                }
                _ => panic!("Expected multiplication expression in body"),
            }
        }
        _ => panic!("Expected Product variant"),
    }
}

#[test]
fn test_sum_multiplication_power_in_body() {
    // \sum_{i=1}^{n} 2*i^3 should parse as sum of (2*(i^3))
    let expr = parse_latex(r"\sum_{i=1}^{n} 2*i^3").unwrap();
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
            // Body should be 2*i^3
            match *body {
                Expression::Binary {
                    op: BinaryOp::Mul,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Integer(2));
                    match *right {
                        Expression::Binary {
                            op: BinaryOp::Pow,
                            left: power_left,
                            right: power_right,
                        } => {
                            assert_eq!(*power_left, Expression::Variable("i".to_string()));
                            assert_eq!(*power_right, Expression::Integer(3));
                        }
                        _ => panic!("Expected power in right operand"),
                    }
                }
                _ => panic!("Expected multiplication expression in body"),
            }
        }
        _ => panic!("Expected Sum variant"),
    }
}
