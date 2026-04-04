//! Stress tests and vector operation tests.

use super::*;

mod stress_tests {
    use super::*;

    #[test]
    fn test_deeply_nested_parentheses() {
        let expr = parse("((((((((((x))))))))))").unwrap();
        assert_eq!(expr, Expression::Variable("x".to_string()));
    }

    #[test]
    fn test_very_long_expression() {
        let expr = parse("1+2+3+4+5+6+7+8+9+10").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_nested_expression() {
        let expr = parse("2*sin(x^2 + 1)! - cos(y)/(z + 1)").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_many_nested_functions() {
        let expr = parse("f(g(x), h(y), i(z), j(a), k(b))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 5);
                for arg in args {
                    assert!(matches!(arg, Expression::Function { .. }));
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_alternating_operators() {
        let expr = parse("a + b - c + d - e").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_precedence_expression() {
        let expr = parse("((a + b) * c - d / e) ^ f").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected power"),
        }
    }

    #[test]
    fn test_parse_unicode_pi() {
        let expr = parse("2*π").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(*right, Expression::Constant(MathConstant::Pi)));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_parse_unicode_infinity() {
        let expr = parse("∞").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_unicode_sqrt() {
        let expr = parse("√4").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Integer(4)));
            }
            _ => panic!("Expected sqrt function call"),
        }
    }

    #[test]
    fn test_parse_unicode_sqrt_with_parens() {
        let expr = parse("√(x+1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected sqrt function call"),
        }
    }

    #[test]
    fn test_parse_subscript_with_single_digit() {
        let expr = parse("x_1").unwrap();
        assert_eq!(expr, Expression::Variable("x_1".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_identifier() {
        let expr = parse("alpha_i").unwrap();
        assert_eq!(expr, Expression::Variable("alpha_i".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multiple_digits() {
        let expr = parse("x_12").unwrap();
        assert_eq!(expr, Expression::Variable("x_12".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multi_char() {
        let expr = parse("x_ij").unwrap();
        assert_eq!(expr, Expression::Variable("x_ij".to_string()));
    }

    #[test]
    fn test_parse_subscript_in_expression() {
        let expr = parse("x_1 + y_2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x_1".to_string()));
                assert_eq!(*right, Expression::Variable("y_2".to_string()));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_parse_subscript_round_trip() {
        let input = "x_1";
        let expr = parse(input).unwrap();
        let output = format!("{}", expr);
        assert_eq!(output, input);
    }
}

mod vector_operations {
    use super::*;

    #[test]
    fn test_parse_dot_product() {
        let expr = parse("dot(u, v)").unwrap();
        match expr {
            Expression::DotProduct { left, right } => {
                assert_eq!(*left, Expression::Variable("u".to_string()));
                assert_eq!(*right, Expression::Variable("v".to_string()));
            }
            _ => panic!("Expected DotProduct, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_cross_product() {
        let expr = parse("cross(u, v)").unwrap();
        match expr {
            Expression::CrossProduct { left, right } => {
                assert_eq!(*left, Expression::Variable("u".to_string()));
                assert_eq!(*right, Expression::Variable("v".to_string()));
            }
            _ => panic!("Expected CrossProduct, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_dot_product_with_expressions() {
        let expr = parse("dot(a + b, c * d)").unwrap();
        match expr {
            Expression::DotProduct { left, right } => {
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
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("Expected DotProduct"),
        }
    }
}

mod vector_calculus {
    use super::*;

    #[test]
    fn test_parse_gradient() {
        let expr = parse("grad(f)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected Gradient, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_divergence() {
        let expr = parse("div(F)").unwrap();
        match expr {
            Expression::Divergence { field } => {
                assert_eq!(*field, Expression::Variable("F".to_string()));
            }
            _ => panic!("Expected Divergence, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_curl() {
        let expr = parse("curl(F)").unwrap();
        match expr {
            Expression::Curl { field } => {
                assert_eq!(*field, Expression::Variable("F".to_string()));
            }
            _ => panic!("Expected Curl, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_laplacian() {
        let expr = parse("laplacian(f)").unwrap();
        match expr {
            Expression::Laplacian { expr } => {
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected Laplacian, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_vector_calculus_with_expression() {
        let expr = parse("grad(x^2 + y^2)").unwrap();
        match expr {
            Expression::Gradient { expr } => {
                assert!(matches!(
                    *expr,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Gradient"),
        }
    }
}
