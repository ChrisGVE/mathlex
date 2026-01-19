// Function tests for LaTeX parser
use super::*;

// Trigonometric functions

#[test]
fn test_sin_braced() {
    let expr = parse_latex(r"\sin{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sin_parentheses() {
    let expr = parse_latex(r"\sin(x)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sin_unbraced() {
    let expr = parse_latex(r"\sin x").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_cos() {
    let expr = parse_latex(r"\cos{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "cos");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_tan() {
    let expr = parse_latex(r"\tan{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "tan");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sec() {
    let expr = parse_latex(r"\sec{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sec");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_csc() {
    let expr = parse_latex(r"\csc{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "csc");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_cot() {
    let expr = parse_latex(r"\cot{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "cot");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

// Inverse trigonometric functions

#[test]
fn test_arcsin() {
    let expr = parse_latex(r"\arcsin{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "arcsin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_arccos() {
    let expr = parse_latex(r"\arccos{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "arccos");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_arctan() {
    let expr = parse_latex(r"\arctan{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "arctan");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

// Hyperbolic functions

#[test]
fn test_sinh() {
    let expr = parse_latex(r"\sinh{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sinh");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_cosh() {
    let expr = parse_latex(r"\cosh{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "cosh");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_tanh() {
    let expr = parse_latex(r"\tanh{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "tanh");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

// Logarithmic and exponential functions

#[test]
fn test_ln() {
    let expr = parse_latex(r"\ln{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_log() {
    let expr = parse_latex(r"\log{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "log");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_exp() {
    let expr = parse_latex(r"\exp{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

// Other functions

#[test]
fn test_min() {
    let expr = parse_latex(r"\min{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "min");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_max() {
    let expr = parse_latex(r"\max{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "max");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_gcd() {
    let expr = parse_latex(r"\gcd{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "gcd");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_lcm() {
    let expr = parse_latex(r"\lcm{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "lcm");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

// Functions with complex arguments

#[test]
fn test_sin_of_addition() {
    let expr = parse_latex(r"\sin(x + y)").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expression::Binary { op: BinaryOp::Add, .. }));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_cos_of_multiplication() {
    let expr = parse_latex(r"\cos{2*x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "cos");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expression::Binary { op: BinaryOp::Mul, .. }));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_tan_of_fraction() {
    let expr = parse_latex(r"\tan{\frac{x}{2}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "tan");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expression::Binary { op: BinaryOp::Div, .. }));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_ln_of_power() {
    let expr = parse_latex(r"\ln{x^2}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expression::Binary { op: BinaryOp::Pow, .. }));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_exp_of_negative() {
    let expr = parse_latex(r"\exp{-x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expression::Unary { op: crate::ast::UnaryOp::Neg, .. }));
        }
        _ => panic!("Expected function call"),
    }
}

// Nested functions

#[test]
fn test_sin_of_cos() {
    let expr = parse_latex(r"\sin{\cos{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Function {
                    name: inner_name,
                    args: inner_args,
                } => {
                    assert_eq!(inner_name, "cos");
                    assert_eq!(inner_args.len(), 1);
                    assert_eq!(inner_args[0], Expression::Variable("x".to_string()));
                }
                _ => panic!("Expected nested function"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_ln_of_exp() {
    let expr = parse_latex(r"\ln{\exp{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Function {
                    name: inner_name, ..
                } => {
                    assert_eq!(inner_name, "exp");
                }
                _ => panic!("Expected nested function"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_exp_of_ln() {
    let expr = parse_latex(r"\exp{\ln{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Function {
                    name: inner_name, ..
                } => {
                    assert_eq!(inner_name, "ln");
                }
                _ => panic!("Expected nested function"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

// Functions in expressions

#[test]
fn test_sin_plus_cos() {
    let expr = parse_latex(r"\sin{x} + \cos{x}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            match (*left, *right) {
                (
                    Expression::Function { name: n1, .. },
                    Expression::Function { name: n2, .. },
                ) => {
                    assert_eq!(n1, "sin");
                    assert_eq!(n2, "cos");
                }
                _ => panic!("Expected two functions"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_sin_times_cos() {
    let expr = parse_latex(r"\sin{x} * \cos{x}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            match (*left, *right) {
                (
                    Expression::Function { name: n1, .. },
                    Expression::Function { name: n2, .. },
                ) => {
                    assert_eq!(n1, "sin");
                    assert_eq!(n2, "cos");
                }
                _ => panic!("Expected two functions"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_sin_squared() {
    let expr = parse_latex(r"\sin{x}^2").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            match *left {
                Expression::Function { name, .. } => assert_eq!(name, "sin"),
                _ => panic!("Expected function"),
            }
            assert_eq!(*right, Expression::Integer(2));
        }
        _ => panic!("Expected power"),
    }
}

#[test]
fn test_coefficient_times_sin() {
    let expr = parse_latex(r"2 * \sin{x}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            match *right {
                Expression::Function { name, .. } => assert_eq!(name, "sin"),
                _ => panic!("Expected function"),
            }
        }
        _ => panic!("Expected multiplication"),
    }
}

// Functions with Greek letters

#[test]
fn test_sin_of_alpha() {
    let expr = parse_latex(r"\sin{\alpha}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("alpha".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_ln_of_pi() {
    let expr = parse_latex(r"\ln{\pi}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Constant(MathConstant::Pi));
        }
        _ => panic!("Expected function call"),
    }
}

// Edge cases

#[test]
fn test_function_of_number() {
    let expr = parse_latex(r"\sin{0}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sin");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Integer(0));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_function_of_float() {
    let expr = parse_latex(r"\cos{3.14}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "cos");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Float(f) => assert!((f.value() - 3.14).abs() < 1e-10),
                _ => panic!("Expected float"),
            }
        }
        _ => panic!("Expected function call"),
    }
}
