//! Stress tests and vector operation tests.

use super::*;

mod stress_tests {
    use super::*;

    #[test]
    fn test_deeply_nested_parentheses() {
        let expr = parse("((((((((((x))))))))))").unwrap();
        assert_eq!(expr, Expression::variable("x".to_string()));
    }

    #[test]
    fn test_very_long_expression() {
        let expr = parse("1+2+3+4+5+6+7+8+9+10").unwrap();
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_nested_expression() {
        let expr = parse("2*sin(x^2 + 1)! - cos(y)/(z + 1)").unwrap();
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_many_nested_functions() {
        let expr = parse("f(g(x), h(y), i(z), j(a), k(b))").unwrap();
        match &expr.kind {
            ExprKind::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 5);
                for arg in args {
                    assert!(matches!(arg.kind, ExprKind::Function { .. }));
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_alternating_operators() {
        let expr = parse("a + b - c + d - e").unwrap();
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_complex_precedence_expression() {
        let expr = parse("((a + b) * c - d / e) ^ f").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::variable("f".to_string()));
            }
            _ => panic!("Expected power"),
        }
    }

    #[test]
    fn test_parse_unicode_pi() {
        let expr = parse("2*π").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Integer(2)));
                assert!(matches!(right.kind, ExprKind::Constant(MathConstant::Pi)));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_parse_unicode_infinity() {
        let expr = parse("∞").unwrap();
        assert_eq!(expr, Expression::constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_unicode_sqrt() {
        let expr = parse("√4").unwrap();
        match &expr.kind {
            ExprKind::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0].kind, ExprKind::Integer(4)));
            }
            _ => panic!("Expected sqrt function call"),
        }
    }

    #[test]
    fn test_parse_unicode_sqrt_with_parens() {
        let expr = parse("√(x+1)").unwrap();
        match &expr.kind {
            ExprKind::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0].kind,
                    ExprKind::Binary {
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
        assert_eq!(expr, Expression::variable("x_1".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_identifier() {
        let expr = parse("alpha_i").unwrap();
        assert_eq!(expr, Expression::variable("alpha_i".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multiple_digits() {
        let expr = parse("x_12").unwrap();
        assert_eq!(expr, Expression::variable("x_12".to_string()));
    }

    #[test]
    fn test_parse_subscript_with_multi_char() {
        let expr = parse("x_ij").unwrap();
        assert_eq!(expr, Expression::variable("x_ij".to_string()));
    }

    #[test]
    fn test_parse_subscript_in_expression() {
        let expr = parse("x_1 + y_2").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("x_1".to_string()));
                assert_eq!(**right, Expression::variable("y_2".to_string()));
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
        match &expr.kind {
            ExprKind::DotProduct { left, right } => {
                assert_eq!(**left, Expression::variable("u".to_string()));
                assert_eq!(**right, Expression::variable("v".to_string()));
            }
            _ => panic!("Expected DotProduct, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_cross_product() {
        let expr = parse("cross(u, v)").unwrap();
        match &expr.kind {
            ExprKind::CrossProduct { left, right } => {
                assert_eq!(**left, Expression::variable("u".to_string()));
                assert_eq!(**right, Expression::variable("v".to_string()));
            }
            _ => panic!("Expected CrossProduct, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_dot_product_with_expressions() {
        let expr = parse("dot(a + b, c * d)").unwrap();
        match &expr.kind {
            ExprKind::DotProduct { left, right } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    (**right).kind,
                    ExprKind::Binary {
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
        match &expr.kind {
            ExprKind::Gradient { expr } => {
                assert_eq!(**expr, Expression::variable("f".to_string()));
            }
            _ => panic!("Expected Gradient, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_divergence() {
        let expr = parse("div(F)").unwrap();
        match &expr.kind {
            ExprKind::Divergence { field } => {
                assert_eq!(**field, Expression::variable("F".to_string()));
            }
            _ => panic!("Expected Divergence, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_curl() {
        let expr = parse("curl(F)").unwrap();
        match &expr.kind {
            ExprKind::Curl { field } => {
                assert_eq!(**field, Expression::variable("F".to_string()));
            }
            _ => panic!("Expected Curl, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_laplacian() {
        let expr = parse("laplacian(f)").unwrap();
        match &expr.kind {
            ExprKind::Laplacian { expr } => {
                assert_eq!(**expr, Expression::variable("f".to_string()));
            }
            _ => panic!("Expected Laplacian, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_vector_calculus_with_expression() {
        let expr = parse("grad(x^2 + y^2)").unwrap();
        match &expr.kind {
            ExprKind::Gradient { expr } => {
                assert!(matches!(
                    (**expr).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Gradient"),
        }
    }
}
