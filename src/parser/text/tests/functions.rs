//! Comprehensive function tests.

use super::*;

mod comprehensive_functions {
    use super::*;

    #[test]
    fn test_trig_functions() {
        let expr = parse("sin(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("cos(2*pi)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "cos");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Binary { .. }));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("tan(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => {
                assert_eq!(name, "tan");
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_inverse_trig_functions() {
        // Aliases are normalized to canonical names.
        let expr = parse("asin(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arcsin"),
            _ => panic!("Expected function"),
        }

        let expr = parse("acos(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arccos"),
            _ => panic!("Expected function"),
        }

        let expr = parse("atan(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arctan"),
            _ => panic!("Expected function"),
        }

        // Canonical names still work unchanged.
        let expr = parse("arcsin(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arcsin"),
            _ => panic!("Expected function"),
        }

        let expr = parse("arccos(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arccos"),
            _ => panic!("Expected function"),
        }

        let expr = parse("arctan(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "arctan"),
            _ => panic!("Expected function"),
        }

        let expr = parse("atan2(y, x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "atan2");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_name_aliases() {
        // sign -> sgn
        let expr = parse("sign(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sgn");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }

        // log2 -> lg
        let expr = parse("log2(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "lg");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }

        // sgn canonical name still works
        let expr = parse("sgn(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sgn"),
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_additional_math_functions() {
        let expr = parse("cbrt(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "cbrt");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("round(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "round");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("pow(x, y)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
                assert!(matches!(args[1], Expression::Variable(ref v) if v == "y"));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("atan2(y, x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "atan2");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "y"));
                assert!(matches!(args[1], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_hyperbolic_functions() {
        let expr = parse("sinh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sinh"),
            _ => panic!("Expected function"),
        }

        let expr = parse("cosh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "cosh"),
            _ => panic!("Expected function"),
        }

        let expr = parse("tanh(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "tanh"),
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_logarithmic_functions() {
        let expr = parse("log(2, 8)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "log");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("ln(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "ln");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("exp(-x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "exp");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Unary { .. }));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_other_functions() {
        let expr = parse("sqrt(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sqrt"),
            _ => panic!("Expected function"),
        }

        let expr = parse("abs(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "abs"),
            _ => panic!("Expected function"),
        }

        let expr = parse("floor(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "floor"),
            _ => panic!("Expected function"),
        }

        let expr = parse("ceil(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "ceil"),
            _ => panic!("Expected function"),
        }

        let expr = parse("sgn(x)").unwrap();
        match expr {
            Expression::Function { name, .. } => assert_eq!(name, "sgn"),
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_multi_argument_functions() {
        let expr = parse("max(a, b)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("min(a, b, c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "min");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_deeply_nested_functions() {
        let expr = parse("sin(cos(x))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                match &args[0] {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "cos");
                        assert_eq!(args.len(), 1);
                    }
                    _ => panic!("Expected nested function"),
                }
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("max(min(a, b), c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "max");
                assert_eq!(args.len(), 2);
                match &args[0] {
                    Expression::Function { name, .. } => assert_eq!(name, "min"),
                    _ => panic!("Expected nested function"),
                }
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_functions_with_complex_expressions() {
        let expr = parse("sin(x + y)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("log(2, x^2 + 1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "log");
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expression::Integer(2)));
                assert!(matches!(
                    args[1],
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("sqrt(x^2 + y^2)").unwrap();
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
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_custom_function_names() {
        let expr = parse("myFunc(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "myFunc");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("customFunction(a, b, c)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "customFunction");
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_function_in_complex_expression() {
        let expr = parse("2 * sin(x) + cos(y)").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Mul, ..
                    } => {}
                    _ => panic!("Expected multiplication on left"),
                }
                match *right {
                    Expression::Function { name, .. } => assert_eq!(name, "cos"),
                    _ => panic!("Expected function on right"),
                }
            }
            _ => panic!("Expected binary addition"),
        }

        let expr = parse("pow(x, 2)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "pow");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_nested_parentheses() {
        let expr = parse("((2 + 3) * (4 + 5))").unwrap();
        assert!(matches!(expr, Expression::Binary { .. }));
    }

    #[test]
    fn test_multiple_unary_operators() {
        let expr = parse("--5").unwrap();
        match expr {
            Expression::Unary {
                op: UnaryOp::Neg,
                operand,
            } => {
                assert!(matches!(
                    *operand,
                    Expression::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("Expected negation"),
        }
    }

    #[test]
    fn test_log_two_args() {
        let expr = parse("log(x, 2)").unwrap();
        assert_eq!(
            expr,
            Expression::Function {
                name: "log".to_string(),
                args: vec![
                    Expression::Variable("x".to_string()),
                    Expression::Integer(2)
                ],
            }
        );
    }

    #[test]
    fn test_log_numeric_args() {
        let expr = parse("log(8, 2)").unwrap();
        assert_eq!(
            expr,
            Expression::Function {
                name: "log".to_string(),
                args: vec![Expression::Integer(8), Expression::Integer(2)],
            }
        );
    }

    // NumericSwift functions: trunc, clamp, lerp, rad, deg

    #[test]
    fn test_trunc() {
        let expr = parse("trunc(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "trunc");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_trunc_nested() {
        // trunc(sin(x))
        let expr = parse("trunc(sin(x))").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "trunc");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Function { ref name, .. } if name == "sin"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_clamp() {
        let expr = parse("clamp(x, 0, 1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "clamp");
                assert_eq!(args.len(), 3);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
                assert!(matches!(args[1], Expression::Integer(0)));
                assert!(matches!(args[2], Expression::Integer(1)));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_clamp_complex_first_arg() {
        // clamp(x^2, -1, 1)
        let expr = parse("clamp(x^2, -1, 1)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "clamp");
                assert_eq!(args.len(), 3);
                assert!(matches!(
                    args[0],
                    Expression::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_lerp() {
        let expr = parse("lerp(a, b, t)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "lerp");
                assert_eq!(args.len(), 3);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "a"));
                assert!(matches!(args[1], Expression::Variable(ref v) if v == "b"));
                assert!(matches!(args[2], Expression::Variable(ref v) if v == "t"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_lerp_numeric_bounds() {
        let expr = parse("lerp(0, 1, t)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "lerp");
                assert_eq!(args.len(), 3);
                assert!(matches!(args[0], Expression::Integer(0)));
                assert!(matches!(args[1], Expression::Integer(1)));
                assert!(matches!(args[2], Expression::Variable(ref v) if v == "t"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_rad() {
        let expr = parse("rad(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "rad");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_deg() {
        let expr = parse("deg(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "deg");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Variable(ref v) if v == "x"));
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_rad_deg_with_pi() {
        // rad(180) and deg(pi)
        let expr = parse("rad(180)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "rad");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Integer(180)));
            }
            _ => panic!("Expected function"),
        }

        let expr = parse("deg(pi)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "deg");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expression::Constant(MathConstant::Pi)));
            }
            _ => panic!("Expected function"),
        }
    }
}
