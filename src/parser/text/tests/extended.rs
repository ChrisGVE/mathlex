//! Extended arithmetic and operator precedence tests.

use super::*;

mod extended_basic_arithmetic {
    use super::*;

    #[test]
    fn test_scientific_notation_positive_exponent() {
        let expr = parse("1.5e10").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 1.5e10);
            }
            _ => panic!("Expected float with scientific notation"),
        }
    }

    #[test]
    fn test_scientific_notation_negative_exponent() {
        let expr = parse("2.5e-5").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 2.5e-5);
            }
            _ => panic!("Expected float with negative exponent"),
        }
    }

    #[test]
    fn test_scientific_notation_uppercase_e() {
        let expr = parse("3.14E8").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 3.14e8);
            }
            _ => panic!("Expected float with uppercase E"),
        }
    }

    #[test]
    fn test_scientific_notation_with_positive_sign() {
        let expr = parse("1e+3").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 1000.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_very_large_integer() {
        let expr = parse("9223372036854775807").unwrap(); // i64::MAX
        assert!(matches!(expr, Expression::Integer(_)));
    }

    #[test]
    fn test_zero() {
        let expr = parse("0").unwrap();
        assert_eq!(expr, Expression::Integer(0));
    }

    #[test]
    fn test_zero_float() {
        let expr = parse("0.0").unwrap();
        match expr {
            Expression::Float(f) => {
                assert_eq!(f.value(), 0.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_negative_zero() {
        let expr = parse("-0").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert_eq!(*operand, Expression::Integer(0));
            }
            _ => panic!("Expected negation of zero"),
        }
    }

    #[test]
    fn test_mixed_int_float_operations() {
        let expr = parse("2 + 3.5").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expression::Integer(2)));
                assert!(matches!(*right, Expression::Float(_)));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_division_by_zero_parses() {
        let expr = parse("1 / 0").unwrap();
        assert!(matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));
    }
}

mod extended_operator_precedence {
    use super::*;

    #[test]
    fn test_unary_minus_with_power() {
        let expr = parse("-x^2").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match *operand {
                Expression::Binary {
                    op: BinaryOp::Pow,
                    left,
                    right,
                } => {
                    assert_eq!(*left, Expression::Variable("x".to_string()));
                    assert_eq!(*right, Expression::Integer(2));
                }
                _ => panic!("Expected power as operand"),
            },
            _ => panic!("Expected negation of power"),
        }
    }

    #[test]
    fn test_parenthesized_negation_with_power() {
        let expr = parse("(-x)^2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected power of negation"),
        }
    }

    #[test]
    fn test_factorial_then_addition() {
        let expr = parse("5! + 1").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
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
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_factorial_then_multiplication() {
        let expr = parse("5! * 2").unwrap();
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
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_factorial_then_division() {
        let expr = parse("5! / 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
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
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_complex_precedence_chain() {
        let expr = parse("a + b * c ^ d").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("a".to_string()));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        left: mul_left,
                        right: mul_right,
                    } => {
                        assert_eq!(*mul_left, Expression::Variable("b".to_string()));
                        assert!(matches!(
                            *mul_right,
                            Expression::Binary {
                                op: BinaryOp::Pow,
                                ..
                            }
                        ));
                    }
                    _ => panic!("Expected multiplication on right"),
                }
            }
            _ => panic!("Expected addition at top level"),
        }
    }

    #[test]
    fn test_left_associativity_of_subtraction() {
        let expr = parse("10 - 5 - 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Sub,
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
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected subtraction"),
        }
    }

    #[test]
    fn test_left_associativity_of_division() {
        let expr = parse("20 / 4 / 2").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_multiple_unary_negations() {
        let expr = parse("---5").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match *operand {
                Expression::Unary {
                    op: UnaryOp::Neg,
                    operand: inner,
                } => {
                    assert!(matches!(
                        *inner,
                        Expression::Unary {
                            op: UnaryOp::Neg,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected nested negation"),
            },
            _ => panic!("Expected negation"),
        }
    }

    #[test]
    fn test_mixed_unary_operators() {
        let expr = parse("-+5").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert!(matches!(
                    *operand,
                    Expression::Unary {
                        op: UnaryOp::Pos,
                        ..
                    }
                ));
            }
            _ => panic!("Expected negation of positive"),
        }
    }

    #[test]
    fn test_triple_factorial() {
        let expr = parse("5!!!").unwrap();
        let mut current = &expr;
        let mut factorial_count = 0;

        while let Expression::Unary {
            op: UnaryOp::Factorial,
            operand,
        } = current
        {
            factorial_count += 1;
            current = operand;
        }

        assert_eq!(factorial_count, 3);
        assert_eq!(*current, Expression::Integer(5));
    }

    #[test]
    fn test_negative_exponent() {
        let result = parse("2^-3");
        assert!(result.is_err(), "2^-3 should require parentheses");

        let expr = parse("2^(-3)").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                assert!(matches!(
                    *right,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("Expected power"),
        }
    }

    #[test]
    fn test_power_zero_and_one() {
        let expr = parse("x^0").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected power"),
        }

        let expr = parse("x^1").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(1));
            }
            _ => panic!("Expected power"),
        }
    }
}
