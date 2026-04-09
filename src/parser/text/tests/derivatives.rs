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
