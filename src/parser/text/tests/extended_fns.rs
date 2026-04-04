//! Extended function, implicit multiplication, and error case tests.

use super::*;

mod extended_functions {
    use super::*;

    #[test]
    fn test_many_function_arguments() {
        // Test function with 10 arguments
        let expr = parse("f(a, b, c, d, e, f, g, h, i, j)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 10);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_deeply_nested_functions_three_levels() {
        // f(g(h(x)))
        let expr = parse("f(g(h(x)))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "g");
                        assert_eq!(args.len(), 1);
                        match &args[0] {
                            Expression::Function { name, args } => {
                                assert_eq!(name, "h");
                                assert_eq!(args.len(), 1);
                                assert_eq!(args[0], Expression::Variable("x".to_string()));
                            }
                            _ => panic!("Expected third level function"),
                        }
                    }
                    _ => panic!("Expected second level function"),
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_factorial_argument() {
        let expr = parse("sin(5!)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_equation_argument() {
        // solve(x = 5) - equation as argument should parse
        let expr = parse("solve(x = 5)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "solve");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Equation { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_inequality_argument() {
        let expr = parse("filter(x > 0)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "filter");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Inequality { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_all_operator_types() {
        // f(a+b, c*d, e^f, g!)
        let expr = parse("f(a+b, c*d, e^f, g!)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 4);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    args[1],
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    args[2],
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(
                    args[3],
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_with_nested_parentheses_argument() {
        let expr = parse("f(((x)))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "f");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function"),
        }
    }
}

mod extended_implicit_multiplication {
    use super::*;

    #[test]
    fn test_implicit_mult_factorial_then_variable() {
        // 5!x should parse as (5!)*x
        let config = ParserConfig::default();
        let expr = parse_with_config("5!x", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_factorial_then_parens() {
        // 5!(x+1) should parse as (5!)*(x+1)
        let config = ParserConfig::default();
        let expr = parse_with_config("5!(x+1)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Factorial,
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
    fn test_implicit_mult_chained_three_variables() {
        // a b c should parse as (a*b)*c
        let config = ParserConfig::default();
        let expr = parse_with_config("a b c", &config).unwrap();
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
                assert_eq!(*right, Expression::Variable("c".to_string()));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_constant_then_function() {
        // pi sin(x) should parse as pi*sin(x)
        let config = ParserConfig::default();
        let expr = parse_with_config("pi sin(x)", &config).unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Constant(MathConstant::Pi));
                assert!(matches!(*right, Expression::Function { .. }));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_implicit_mult_multiple_numbers_fails() {
        // 2 3 with implicit mult should fail
        let config = ParserConfig::default();
        let result = parse_with_config("2 3", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_implicit_mult_power_chain() {
        // 2x^3 should parse as 2*(x^3), not (2*x)^3
        let config = ParserConfig::default();
        let expr = parse_with_config("2x^3", &config).unwrap();
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
    fn test_implicit_mult_with_relation() {
        // 2x = 5 should parse as (2*x) = 5
        let config = ParserConfig::default();
        let expr = parse_with_config("2x = 5", &config).unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected equation"),
        }
    }
}

mod extended_error_cases {
    use super::*;

    #[test]
    fn test_error_leading_comma_in_function() {
        let result = parse("f(,x)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_trailing_comma_in_function() {
        let result = parse("f(x,)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_double_comma_in_function() {
        let result = parse("f(x,,y)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_lone_star_operator() {
        assert!(parse("*").is_err());
    }

    #[test]
    fn test_error_lone_slash_operator() {
        assert!(parse("/").is_err());
    }

    #[test]
    fn test_error_lone_caret_operator() {
        assert!(parse("^").is_err());
    }

    #[test]
    fn test_error_lone_percent_operator() {
        assert!(parse("%").is_err());
    }

    #[test]
    fn test_error_empty_parens_not_function() {
        let result = parse("()");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_just_whitespace() {
        let result = parse("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_mismatched_brackets() {
        let result = parse("[1]");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_double_operators() {
        let result = parse("2 */ 3");
        assert!(result.is_err(), "Mix of * and / should be an error");
        let result = parse("2 ^^ 3");
        assert!(result.is_err(), "Double caret should be an error");
    }

    #[test]
    fn test_error_triple_equals() {
        let result = parse("x === 5");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_incomplete_inequality() {
        let result = parse("x <");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parentheses_mismatch_extra_open() {
        let result = parse("((2 + 3)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_parentheses_mismatch_extra_close() {
        let result = parse("(2 + 3))");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_function_name() {
        // Without implicit mult, 2func(x) is an error
        let config = ParserConfig {
            implicit_multiplication: false,
            ..ParserConfig::default()
        };
        let result = parse_with_config("2func(x)", &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_function_call() {
        let result = parse("sin(x");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_comma_outside_function() {
        let result = parse("x, y");
        assert!(result.is_err());
    }
}
