//! Implicit multiplication tests.

use super::*;

mod implicit_multiplication {
    use super::*;

    #[test]
    fn test_implicit_mult_number_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::integer(2));
                assert_eq!(**right, Expression::variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_float_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("3.14r", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Float(_)));
                assert_eq!(**right, Expression::variable("r".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_parens() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2(x+1)", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::integer(2));
                assert!(matches!(
                    (**right).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_variable_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("x y", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("x".to_string()));
                assert_eq!(**right, Expression::variable("y".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_variable_chain() {
        let config = ParserConfig::default();
        let expr = parse_with_config("x y z", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::variable("z".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_constant_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("pi x", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::constant(MathConstant::Pi));
                assert_eq!(**right, Expression::variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parens_parens() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)(b)", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("a".to_string()));
                assert_eq!(**right, Expression::variable("b".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parens_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)x", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("a".to_string()));
                assert_eq!(**right, Expression::variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_complex_expression() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 3y", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
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
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_implicit_mult_with_power() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x^2", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::integer(2));
                assert!(matches!(
                    (**right).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_no_implicit_mult_function_call() {
        let config = ParserConfig::default();
        let expr = parse_with_config("sin(x)", &config).unwrap();
        match &expr.kind {
            ExprKind::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function call, not implicit multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_disabled() {
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        let result = parse_with_config("2x", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_implicit_mult_mixed_with_explicit() {
        // "2x * 3y" — implicit and explicit mul share the same precedence
        // and left-associate, producing ((2*x)*3)*y.
        let config = ParserConfig::default();
        let expr = parse_with_config("2x * 3y", &config).unwrap();
        // Top level: Mul(((2*x)*3), y)
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                // right = y
                assert_eq!(**right, Expression::variable("y"));
                // left = (2*x)*3
                match &left.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        left: ll,
                        right: lr,
                    } => {
                        assert_eq!(**lr, Expression::integer(3));
                        // ll = 2*x
                        assert!(matches!(
                            ll.kind,
                            ExprKind::Binary {
                                op: BinaryOp::Mul,
                                ..
                            }
                        ));
                    }
                    _ => panic!("Expected nested multiplication, got {:?}", left),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parenthesized_sum() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a + b)(c + d)", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
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
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_function() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2sin(x)", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::integer(2));
                match &right.kind {
                    ExprKind::Function { name, .. } => assert_eq!(name, "sin"),
                    _ => panic!("Expected function on right"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_default_config() {
        let expr = parse("2x").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul, ..
            } => {}
            _ => panic!("Expected multiplication with default config"),
        }
    }

    #[test]
    fn test_implicit_mult_precedence() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 1", &config).unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(1));
            }
            _ => panic!("Expected addition at top level"),
        }
    }
}
