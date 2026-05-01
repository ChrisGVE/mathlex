//! Extended arithmetic and operator precedence tests.

use super::*;

mod extended_basic_arithmetic {
    use super::*;

    #[test]
    fn test_scientific_notation_positive_exponent() {
        let expr = parse("1.5e10").unwrap();
        match &expr.kind {
            ExprKind::Float(f) => {
                assert_eq!(f.value(), 1.5e10);
            }
            _ => panic!("Expected float with scientific notation"),
        }
    }

    #[test]
    fn test_scientific_notation_negative_exponent() {
        let expr = parse("2.5e-5").unwrap();
        match &expr.kind {
            ExprKind::Float(f) => {
                assert_eq!(f.value(), 2.5e-5);
            }
            _ => panic!("Expected float with negative exponent"),
        }
    }

    #[test]
    fn test_scientific_notation_uppercase_e() {
        let expr = parse("3.14E8").unwrap();
        match &expr.kind {
            ExprKind::Float(f) => {
                assert_eq!(f.value(), 3.14e8);
            }
            _ => panic!("Expected float with uppercase E"),
        }
    }

    #[test]
    fn test_scientific_notation_with_positive_sign() {
        let expr = parse("1e+3").unwrap();
        match &expr.kind {
            ExprKind::Float(f) => {
                assert_eq!(f.value(), 1000.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_very_large_integer() {
        let expr = parse("9223372036854775807").unwrap(); // i64::MAX
        assert!(matches!(expr.kind, ExprKind::Integer(_)));
    }

    #[test]
    fn test_zero() {
        let expr = parse("0").unwrap();
        assert_eq!(expr, Expression::integer(0));
    }

    #[test]
    fn test_zero_float() {
        let expr = parse("0.0").unwrap();
        match &expr.kind {
            ExprKind::Float(f) => {
                assert_eq!(f.value(), 0.0);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_negative_zero() {
        let expr = parse("-0").unwrap();
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert_eq!(**operand, Expression::integer(0));
            }
            _ => panic!("Expected negation of zero"),
        }
    }

    #[test]
    fn test_mixed_int_float_operations() {
        let expr = parse("2 + 3.5").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Integer(2)));
                assert!(matches!(right.kind, ExprKind::Float(_)));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_division_by_zero_parses() {
        let expr = parse("1 / 0").unwrap();
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
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
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match &operand.kind {
                ExprKind::Binary {
                    op: BinaryOp::Pow,
                    left,
                    right,
                } => {
                    assert_eq!(**left, Expression::variable("x".to_string()));
                    assert_eq!(**right, Expression::integer(2));
                }
                _ => panic!("Expected power as operand"),
            },
            _ => panic!("Expected negation of power"),
        }
    }

    #[test]
    fn test_parenthesized_negation_with_power() {
        let expr = parse("(-x)^2").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(2));
            }
            _ => panic!("Expected power of negation"),
        }
    }

    #[test]
    fn test_factorial_then_addition() {
        let expr = parse("5! + 1").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(1));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_factorial_then_multiplication() {
        let expr = parse("5! * 2").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(2));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_factorial_then_division() {
        let expr = parse("5! / 2").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Unary {
                        op: UnaryOp::Factorial,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_complex_precedence_chain() {
        let expr = parse("a + b * c ^ d").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("a".to_string()));
                match &right.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        left: mul_left,
                        right: mul_right,
                    } => {
                        assert_eq!(**mul_left, Expression::variable("b".to_string()));
                        assert!(matches!(
                            (**mul_right).kind,
                            ExprKind::Binary {
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
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Sub,
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
                assert_eq!(**right, Expression::integer(2));
            }
            _ => panic!("Expected subtraction"),
        }
    }

    #[test]
    fn test_left_associativity_of_division() {
        let expr = parse("20 / 4 / 2").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                assert!(matches!(
                    (**left).kind,
                    ExprKind::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
                assert_eq!(**right, Expression::integer(2));
            }
            _ => panic!("Expected division"),
        }
    }

    #[test]
    fn test_multiple_unary_negations() {
        let expr = parse("---5").unwrap();
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => match &operand.kind {
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    operand: inner,
                } => {
                    assert!(matches!(
                        (**inner).kind,
                        ExprKind::Unary {
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
        match &expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert!(matches!(
                    (**operand).kind,
                    ExprKind::Unary {
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

        while let ExprKind::Unary {
            op: UnaryOp::Factorial,
            operand,
        } = &current.kind
        {
            factorial_count += 1;
            current = operand;
        }

        assert_eq!(factorial_count, 3);
        assert_eq!(*current, Expression::integer(5));
    }

    #[test]
    fn test_negative_exponent() {
        let result = parse("2^-3");
        assert!(result.is_err(), "2^-3 should require parentheses");

        let expr = parse("2^(-3)").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::integer(2));
                assert!(matches!(
                    (**right).kind,
                    ExprKind::Unary {
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
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("x".to_string()));
                assert_eq!(**right, Expression::integer(0));
            }
            _ => panic!("Expected power"),
        }

        let expr = parse("x^1").unwrap();
        match &expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(**left, Expression::variable("x".to_string()));
                assert_eq!(**right, Expression::integer(1));
            }
            _ => panic!("Expected power"),
        }
    }
}
