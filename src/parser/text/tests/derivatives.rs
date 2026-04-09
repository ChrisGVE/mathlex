//! Tests for plain text derivative and partial derivative parsing.

use super::*;

mod leibniz_notation {
    use super::*;

    #[test]
    fn first_derivative_dy_dx() {
        let expr = parse("dy/dx").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn first_derivative_df_dx() {
        let expr = parse("df/dx").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn first_derivative_dz_dt() {
        let expr = parse("dz/dt").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("z".to_string()));
                assert_eq!(var, "t");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn second_derivative_d2y_dx2() {
        let expr = parse("d2y/dx2").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 2);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn third_derivative_d3y_dx3() {
        let expr = parse("d3y/dx3").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 3);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn multi_letter_function_dtheta_dt() {
        let expr = parse("dtheta/dt").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("theta".to_string()));
                assert_eq!(var, "t");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn derivative_in_equation() {
        let exprs = parse_equation_system("dy/dx = x*y").unwrap();
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expression::Equation { left, .. } => {
                assert!(matches!(**left, Expression::Derivative { .. }));
            }
            other => panic!("Expected Relation, got {:?}", other),
        }
    }

    #[test]
    fn second_derivative_in_ode() {
        // d2y/dx2 + 3*dy/dx + 2*y = 0
        let expr = parse("d2y/dx2 + 3*dy/dx + 2*y = 0").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }

    #[test]
    fn not_a_derivative_plain_division() {
        // 'da' by itself followed by / and a non-d identifier is normal division
        let expr = parse("da/b").unwrap();
        // This should be division, not a derivative (denominator doesn't start with d)
        assert!(matches!(expr, Expression::Binary { .. }));
    }
}

mod prime_notation {
    use super::*;

    #[test]
    fn first_derivative_y_prime() {
        let expr = parse("y'").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn second_derivative_y_double_prime() {
        let expr = parse("y''").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "");
                assert_eq!(order, 2);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn third_derivative_y_triple_prime() {
        let expr = parse("y'''").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "");
                assert_eq!(order, 3);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn prime_in_equation() {
        let expr = parse("y' = -y").unwrap();
        match &expr {
            Expression::Equation { left, .. } => {
                assert!(matches!(**left, Expression::Derivative { .. }));
            }
            other => panic!("Expected Relation, got {:?}", other),
        }
    }

    #[test]
    fn second_order_ode_prime() {
        // y'' + y = 0
        let expr = parse("y'' + y = 0").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }

    #[test]
    fn prime_on_different_variable() {
        let expr = parse("f'").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
                assert_eq!(var, "");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn prime_with_rhs_expression() {
        // y' = -2*y + 3*x
        let expr = parse("y' = -2*y + 3*x").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }
}

mod diff_function {
    use super::*;

    #[test]
    fn diff_first_derivative() {
        let expr = parse("diff(y, x)").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn diff_second_derivative() {
        let expr = parse("diff(y, x, 2)").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 2);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn diff_third_derivative() {
        let expr = parse("diff(y, x, 3)").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("y".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 3);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn diff_with_complex_expression() {
        let expr = parse("diff(x^2 + y, x)").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn diff_in_equation() {
        let expr = parse("diff(y, x) = x").unwrap();
        match &expr {
            Expression::Equation { left, .. } => {
                assert!(matches!(**left, Expression::Derivative { .. }));
            }
            other => panic!("Expected Relation, got {:?}", other),
        }
    }

    #[test]
    fn diff_in_larger_expression() {
        // diff(y, x, 2) + y = 0
        let expr = parse("diff(y, x, 2) + y = 0").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }
}

mod partial_function {
    use super::*;

    #[test]
    fn partial_first_order() {
        let expr = parse("partial(f, x)").unwrap();
        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected PartialDerivative, got {:?}", other),
        }
    }

    #[test]
    fn partial_second_order() {
        let expr = parse("partial(f, x, 2)").unwrap();
        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 2);
            }
            other => panic!("Expected PartialDerivative, got {:?}", other),
        }
    }

    #[test]
    fn partial_mixed_two_vars() {
        // partial(f, x, y) → ∂/∂x(∂/∂y(f))
        let expr = parse("partial(f, x, y)").unwrap();
        match expr {
            Expression::PartialDerivative {
                expr: inner,
                var,
                order,
            } => {
                assert_eq!(var, "x");
                assert_eq!(order, 1);
                match *inner {
                    Expression::PartialDerivative { expr, var, order } => {
                        assert_eq!(*expr, Expression::Variable("f".to_string()));
                        assert_eq!(var, "y");
                        assert_eq!(order, 1);
                    }
                    other => panic!("Expected inner PartialDerivative, got {:?}", other),
                }
            }
            other => panic!("Expected PartialDerivative, got {:?}", other),
        }
    }

    #[test]
    fn partial_with_complex_expression() {
        let expr = parse("partial(x^2*y, x)").unwrap();
        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected PartialDerivative, got {:?}", other),
        }
    }

    #[test]
    fn partial_in_equation() {
        let expr = parse("partial(f, x) = 2*x*y").unwrap();
        match &expr {
            Expression::Equation { left, .. } => {
                assert!(matches!(**left, Expression::PartialDerivative { .. }));
            }
            other => panic!("Expected Relation, got {:?}", other),
        }
    }
}

mod gradient_notation {
    use super::*;

    #[test]
    fn grad_with_parens() {
        let expr = parse("grad(f)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn nabla_with_parens() {
        let expr = parse("nabla(f)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn unicode_nabla_with_identifier() {
        let expr = parse("∇f").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn unicode_nabla_with_parens() {
        let expr = parse("∇(x^2 + y^2)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn nabla_with_complex_expression() {
        let expr = parse("nabla(x^2*y + z)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn grad_without_parens() {
        let expr = parse("grad f").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            other => panic!("Expected Gradient, got {:?}", other),
        }
    }

    #[test]
    fn nabla_in_equation() {
        let expr = parse("∇f = 0").unwrap();
        match &expr {
            Expression::Equation { left, .. } => {
                assert!(matches!(**left, Expression::Gradient { .. }));
            }
            other => panic!("Expected Equation, got {:?}", other),
        }
    }
}

mod integrate_function {
    use super::*;
    use crate::ast::IntegralBounds;

    #[test]
    fn indefinite_with_dx() {
        let expr = parse("integrate(x^2, dx)").unwrap();
        match expr {
            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                assert!(matches!(*integrand, Expression::Binary { .. }));
                assert_eq!(var, "x");
                assert!(bounds.is_none());
            }
            other => panic!("Expected Integral, got {:?}", other),
        }
    }

    #[test]
    fn indefinite_with_bare_var() {
        let expr = parse("integrate(sin(x), x)").unwrap();
        match expr {
            Expression::Integral { integrand, var, .. } => {
                assert!(matches!(*integrand, Expression::Function { .. }));
                assert_eq!(var, "x");
            }
            other => panic!("Expected Integral, got {:?}", other),
        }
    }

    #[test]
    fn definite_with_bounds() {
        let expr = parse("integrate(x, dx, 0, 1)").unwrap();
        match expr {
            Expression::Integral { var, bounds, .. } => {
                assert_eq!(var, "x");
                let b = bounds.unwrap();
                assert_eq!(*b.lower, Expression::Integer(0));
                assert_eq!(*b.upper, Expression::Integer(1));
            }
            other => panic!("Expected Integral, got {:?}", other),
        }
    }

    #[test]
    fn definite_with_symbolic_bounds() {
        let expr = parse("integrate(sin(x), dx, 0, pi)").unwrap();
        match expr {
            Expression::Integral { var, bounds, .. } => {
                assert_eq!(var, "x");
                let b = bounds.unwrap();
                assert_eq!(*b.upper, Expression::Constant(MathConstant::Pi));
            }
            other => panic!("Expected Integral, got {:?}", other),
        }
    }

    #[test]
    fn alias_integral() {
        let expr = parse("integral(x, dx)").unwrap();
        assert!(matches!(expr, Expression::Integral { .. }));
    }

    #[test]
    fn alias_int() {
        let expr = parse("int(x, dx)").unwrap();
        assert!(matches!(expr, Expression::Integral { .. }));
    }

    #[test]
    fn in_equation() {
        let expr = parse("integrate(f, dx, a, b) = F(b) - F(a)").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }
}

mod sum_function {
    use super::*;

    #[test]
    fn basic_sum() {
        let expr = parse("sum(i^2, i, 1, n)").unwrap();
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
                assert!(matches!(*body, Expression::Binary { .. }));
            }
            other => panic!("Expected Sum, got {:?}", other),
        }
    }

    #[test]
    fn sum_with_infinity() {
        let expr = parse("sum(1/n, n, 1, inf)").unwrap();
        match expr {
            Expression::Sum { upper, .. } => {
                assert_eq!(*upper, Expression::Constant(MathConstant::Infinity));
            }
            other => panic!("Expected Sum, got {:?}", other),
        }
    }

    #[test]
    fn alias_summation() {
        let expr = parse("summation(k, k, 0, 10)").unwrap();
        assert!(matches!(expr, Expression::Sum { .. }));
    }
}

mod product_function {
    use super::*;

    #[test]
    fn basic_product() {
        let expr = parse("product(k, k, 1, n)").unwrap();
        match expr {
            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                assert_eq!(index, "k");
                assert_eq!(*lower, Expression::Integer(1));
                assert_eq!(*upper, Expression::Variable("n".to_string()));
                assert_eq!(*body, Expression::Variable("k".to_string()));
            }
            other => panic!("Expected Product, got {:?}", other),
        }
    }

    #[test]
    fn alias_prod() {
        let expr = parse("prod(i, i, 1, 5)").unwrap();
        assert!(matches!(expr, Expression::Product { .. }));
    }
}

mod limit_function {
    use super::*;
    use crate::ast::Direction;

    #[test]
    fn two_sided_limit() {
        let expr = parse("limit(sin(x)/x, x, 0)").unwrap();
        match expr {
            Expression::Limit {
                expr,
                var,
                to,
                direction,
            } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
                assert_eq!(var, "x");
                assert_eq!(*to, Expression::Integer(0));
                assert_eq!(direction, Direction::Both);
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }

    #[test]
    fn right_hand_limit_plus() {
        let expr = parse("limit(1/x, x, 0, +)").unwrap();
        match expr {
            Expression::Limit { direction, .. } => {
                assert_eq!(direction, Direction::Right);
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }

    #[test]
    fn left_hand_limit_minus() {
        let expr = parse("limit(1/x, x, 0, -)").unwrap();
        match expr {
            Expression::Limit { direction, .. } => {
                assert_eq!(direction, Direction::Left);
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }

    #[test]
    fn limit_at_infinity() {
        let expr = parse("limit(1/x, x, inf)").unwrap();
        match expr {
            Expression::Limit { to, .. } => {
                assert_eq!(*to, Expression::Constant(MathConstant::Infinity));
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }

    #[test]
    fn alias_lim() {
        let expr = parse("lim(f, x, 0)").unwrap();
        assert!(matches!(expr, Expression::Limit { .. }));
    }

    #[test]
    fn direction_keyword_right() {
        let expr = parse("limit(f, x, 0, right)").unwrap();
        match expr {
            Expression::Limit { direction, .. } => {
                assert_eq!(direction, Direction::Right);
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }

    #[test]
    fn direction_keyword_left() {
        let expr = parse("limit(f, x, 0, left)").unwrap();
        match expr {
            Expression::Limit { direction, .. } => {
                assert_eq!(direction, Direction::Left);
            }
            other => panic!("Expected Limit, got {:?}", other),
        }
    }
}

mod operator_derivative {
    use super::*;

    #[test]
    fn d_expr_over_dx() {
        let expr = parse("d(x^2)/dx").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Binary { .. }));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn d_sin_over_dx() {
        let expr = parse("d(sin(x))/dx").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert!(matches!(*expr, Expression::Function { .. }));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn d_expr_over_d_paren_var() {
        // d(omega)/d(k) form
        let expr = parse("d(omega)/d(k)").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("omega".to_string()));
                assert_eq!(var, "k");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn d_u_over_dx() {
        let expr = parse("d(U)/dx").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(*expr, Expression::Variable("U".to_string()));
                assert_eq!(var, "x");
                assert_eq!(order, 1);
            }
            other => panic!("Expected Derivative, got {:?}", other),
        }
    }

    #[test]
    fn in_equation() {
        let expr = parse("d(x^2)/dx = 2*x").unwrap();
        assert!(matches!(expr, Expression::Equation { .. }));
    }
}
