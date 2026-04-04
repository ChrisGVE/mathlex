//! Implicit multiplication tests.

use super::*;

mod implicit_multiplication {
    use super::*;

    #[test]
    fn test_implicit_mult_number_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_float_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("3.14r", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Float(_)));
                assert_eq!(*right, Expression::Variable("r".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_parens() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2(x+1)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
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
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Variable("y".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_variable_chain() {
        let config = ParserConfig::default();
        let expr = parse_with_config("x y z", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("z".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_constant_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("pi x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Constant(MathConstant::Pi));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    #[ignore] // TODO: Requires tracking parenthesized expressions through parser state
    fn test_implicit_mult_parens_parens() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)(b)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                assert_eq!(*right, Expression::Variable("b".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parens_variable() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a)x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_complex_expression() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 3y", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
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
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_implicit_mult_with_power() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x^2", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Binary {
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
        match expr {
            Expression::Function { name, args } => {
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
    #[ignore] // TODO: Requires resolving tokenizer multi-character identifier issue
    fn test_implicit_mult_mixed_with_explicit() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x * 3y", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
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
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_parenthesized_sum() {
        let config = ParserConfig::default();
        let expr = parse_with_config("(a + b)(c + d)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
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
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_number_function() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2sin(x)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Function { name, .. } => assert_eq!(name, "sin"),
                    _ => panic!("Expected function on right"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_default_config() {
        let expr = parse("2x").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul, ..
            } => {}
            _ => panic!("Expected multiplication with default config"),
        }
    }

    #[test]
    fn test_implicit_mult_precedence() {
        let config = ParserConfig::default();
        let expr = parse_with_config("2x + 1", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected addition at top level"),
        }
    }
}
