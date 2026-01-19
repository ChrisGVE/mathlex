// Root tests for LaTeX parser
use super::*;

#[test]
fn test_sqrt_simple() {
    let expr = parse_latex(r"\sqrt{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_number() {
    let expr = parse_latex(r"\sqrt{2}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Integer(2));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_float() {
    let expr = parse_latex(r"\sqrt{3.14}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Float(f) => assert!((f.value() - 3.14).abs() < 1e-10),
                _ => panic!("Expected float"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_expression_addition() {
    let expr = parse_latex(r"\sqrt{x+1}").unwrap();
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
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_expression_power() {
    let expr = parse_latex(r"\sqrt{x^2+1}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            // Should be addition with power on left side
            match &args[0] {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left,
                    ..
                } => {
                    assert!(matches!(
                        **left,
                        Expression::Binary {
                            op: BinaryOp::Pow,
                            ..
                        }
                    ));
                }
                _ => panic!("Expected addition"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_expression_multiplication() {
    let expr = parse_latex(r"\sqrt{2*x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert!(matches!(
                args[0],
                Expression::Binary {
                    op: BinaryOp::Mul,
                    ..
                }
            ));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_nested() {
    let expr = parse_latex(r"\sqrt{\sqrt{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            // Inner should also be sqrt function
            match &args[0] {
                Expression::Function {
                    name: inner_name,
                    args: inner_args,
                } => {
                    assert_eq!(inner_name, "sqrt");
                    assert_eq!(inner_args.len(), 1);
                    assert_eq!(inner_args[0], Expression::Variable("x".to_string()));
                }
                _ => panic!("Expected nested sqrt function"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_triple_nested() {
    let expr = parse_latex(r"\sqrt{\sqrt{\sqrt{x}}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            // Verify nesting depth of 3
            match &args[0] {
                Expression::Function { name: n1, args: a1 } if n1 == "sqrt" => match &a1[0] {
                    Expression::Function { name: n2, args: a2 } if n2 == "sqrt" => {
                        assert_eq!(a2[0], Expression::Variable("x".to_string()));
                    }
                    _ => panic!("Expected third level sqrt"),
                },
                _ => panic!("Expected second level sqrt"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_nth_simple() {
    let expr = parse_latex(r"\sqrt[3]{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Integer(3));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_fourth() {
    let expr = parse_latex(r"\sqrt[4]{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Integer(4));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_variable_index() {
    let expr = parse_latex(r"\sqrt[n]{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert_eq!(args[1], Expression::Variable("n".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_expression_index() {
    let expr = parse_latex(r"\sqrt[n+1]{x}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
            assert!(matches!(
                args[1],
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_complex_radicand() {
    let expr = parse_latex(r"\sqrt[4]{x^3+2*x+1}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            // Radicand should be a complex expression
            assert!(matches!(
                args[0],
                Expression::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
            assert_eq!(args[1], Expression::Integer(4));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_root_nested_in_sqrt() {
    let expr = parse_latex(r"\sqrt{\sqrt[3]{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Function {
                    name: inner_name,
                    args: inner_args,
                } => {
                    assert_eq!(inner_name, "root");
                    assert_eq!(inner_args.len(), 2);
                }
                _ => panic!("Expected nested root function"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_in_root() {
    let expr = parse_latex(r"\sqrt[3]{\sqrt{x}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            match &args[0] {
                Expression::Function {
                    name: inner_name,
                    args: inner_args,
                } => {
                    assert_eq!(inner_name, "sqrt");
                    assert_eq!(inner_args.len(), 1);
                }
                _ => panic!("Expected nested sqrt function"),
            }
            assert_eq!(args[1], Expression::Integer(3));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_in_expression() {
    // Test: 1 + \sqrt{x}
    let expr = parse_latex(r"1 + \sqrt{x}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(1));
            match *right {
                Expression::Function { name, .. } => assert_eq!(name, "sqrt"),
                _ => panic!("Expected sqrt function"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_sqrt_with_fraction() {
    let expr = parse_latex(r"\sqrt{\frac{1}{2}}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert!(matches!(
                args[0],
                Expression::Binary {
                    op: BinaryOp::Div,
                    ..
                }
            ));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_sqrt_of_sqrt_plus_one() {
    let expr = parse_latex(r"\sqrt{\sqrt{x}+1}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left,
                    right,
                } => {
                    assert!(matches!(**left, Expression::Function { .. }));
                    assert_eq!(**right, Expression::Integer(1));
                }
                _ => panic!("Expected addition"),
            }
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_multiple_roots() {
    // Test: \sqrt{x} + \sqrt[3]{y}
    let expr = parse_latex(r"\sqrt{x} + \sqrt[3]{y}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => match (*left, *right) {
            (
                Expression::Function { name: n1, args: a1 },
                Expression::Function { name: n2, args: a2 },
            ) => {
                assert_eq!(n1, "sqrt");
                assert_eq!(a1.len(), 1);
                assert_eq!(n2, "root");
                assert_eq!(a2.len(), 2);
            }
            _ => panic!("Expected two function calls"),
        },
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_sqrt_with_greek_letter() {
    let expr = parse_latex(r"\sqrt{\alpha}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("alpha".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}
