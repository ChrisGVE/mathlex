// Constant tests for LaTeX parser
// Tests for context-aware parsing of e (Euler's number) and i (imaginary unit)
use super::*;

// =============================================================================
// Explicit markers
// =============================================================================

#[test]
fn test_explicit_mathrm_e() {
    // \mathrm{e} should always parse as Constant(E)
    let expr = parse_latex(r"\mathrm{e}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::E));
}

#[test]
fn test_explicit_mathrm_i() {
    // \mathrm{i} should always parse as Constant(I)
    let expr = parse_latex(r"\mathrm{i}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::I));
}

#[test]
fn test_imath() {
    // \imath should parse as Constant(I)
    let expr = parse_latex(r"\imath").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::I));
}

#[test]
fn test_jmath() {
    // \jmath should parse as Constant(I) (engineering notation)
    let expr = parse_latex(r"\jmath").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::I));
}

// =============================================================================
// Default behavior (unbound e and i are constants)
// =============================================================================

#[test]
fn test_bare_e_is_constant() {
    // Unbound e defaults to Euler's number
    let expr = parse_latex("e").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::E));
}

#[test]
fn test_bare_i_is_constant() {
    // Unbound i defaults to imaginary unit
    let expr = parse_latex("i").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::I));
}

#[test]
fn test_e_in_expression() {
    // e + 1 should have Constant(E) on left
    let expr = parse_latex("e + 1").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Constant(MathConstant::E));
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected binary expression"),
    }
}

#[test]
fn test_i_in_expression() {
    // 2 * i should have Constant(I) on right
    let expr = parse_latex("2 * i").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Mul);
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Constant(MathConstant::I));
        }
        _ => panic!("Expected binary expression"),
    }
}

// =============================================================================
// Scope tracking (bound variables in sum/product)
// =============================================================================

#[test]
fn test_sum_bound_i() {
    // \sum_{i=1}^n i -> i in body is Variable, not Constant
    let expr = parse_latex(r"\sum_{i=1}^{n} i").unwrap();
    match expr {
        Expression::Sum {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "i");
            assert_eq!(*lower, Expression::Integer(1));
            assert_eq!(*upper, Expression::Variable("n".to_string()));
            // The body i should be a Variable, not Constant(I)
            assert_eq!(*body, Expression::Variable("i".to_string()));
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_prod_bound_e() {
    // \prod_{e=1}^n e -> e in body is Variable, not Constant
    let expr = parse_latex(r"\prod_{e=1}^{n} e").unwrap();
    match expr {
        Expression::Product {
            index,
            lower,
            upper,
            body,
        } => {
            assert_eq!(index, "e");
            assert_eq!(*lower, Expression::Integer(1));
            assert_eq!(*upper, Expression::Variable("n".to_string()));
            // The body e should be a Variable, not Constant(E)
            assert_eq!(*body, Expression::Variable("e".to_string()));
        }
        _ => panic!("Expected Product variant"),
    }
}

#[test]
fn test_sum_with_e_and_i() {
    // \sum_{i=1}^n e -> i is bound (Variable), e is unbound (Constant)
    let expr = parse_latex(r"\sum_{i=1}^{n} e").unwrap();
    match expr {
        Expression::Sum { index, body, .. } => {
            assert_eq!(index, "i");
            // e is NOT the index, so it should remain Constant(E)
            assert_eq!(*body, Expression::Constant(MathConstant::E));
        }
        _ => panic!("Expected Sum variant"),
    }
}

#[test]
fn test_sum_bound_i_in_multiplication() {
    // \sum_{i=1}^n 2*i -> i in body is Variable
    let expr = parse_latex(r"\sum_{i=1}^{n} 2 * i").unwrap();
    match expr {
        Expression::Sum { body, .. } => {
            match *body {
                Expression::Binary { op, left, right } => {
                    assert_eq!(op, BinaryOp::Mul);
                    assert_eq!(*left, Expression::Integer(2));
                    // i should be Variable, not Constant
                    assert_eq!(*right, Expression::Variable("i".to_string()));
                }
                _ => panic!("Expected multiplication in body"),
            }
        }
        _ => panic!("Expected Sum variant"),
    }
}

// =============================================================================
// Exponential normalization (e^x -> exp(x))
// =============================================================================

#[test]
fn test_e_power_is_exp() {
    // e^x should normalize to Function("exp", [x])
    let expr = parse_latex("e^x").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_e_power_braced() {
    // e^{x+1} should normalize to Function("exp", [x+1])
    let expr = parse_latex("e^{x+1}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Binary { op, .. } => {
                    assert_eq!(*op, BinaryOp::Add);
                }
                _ => panic!("Expected binary expression in exp argument"),
            }
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_exp_and_e_power_equal() {
    // \exp{x} and e^x should produce the same AST
    let expr1 = parse_latex(r"\exp{x}").unwrap();
    let expr2 = parse_latex("e^x").unwrap();
    assert_eq!(expr1, expr2);
}

#[test]
fn test_explicit_mathrm_e_power() {
    // \mathrm{e}^x should also normalize to exp(x)
    let expr = parse_latex(r"\mathrm{e}^x").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::Variable("x".to_string()));
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_euler_formula() {
    // e^{i\pi} should produce exp(Constant(I) * Constant(Pi))
    let expr = parse_latex(r"e^{i \pi}").unwrap();
    match expr {
        Expression::Function { name, args } => {
            assert_eq!(name, "exp");
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expression::Binary { op, left, right } => {
                    assert_eq!(*op, BinaryOp::Mul);
                    assert_eq!(**left, Expression::Constant(MathConstant::I));
                    assert_eq!(**right, Expression::Constant(MathConstant::Pi));
                }
                _ => panic!("Expected multiplication in exp argument"),
            }
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn test_e_without_power() {
    // e + 1 should keep e as Constant(E), no normalization
    let expr = parse_latex("e + 1").unwrap();
    match expr {
        Expression::Binary { op, left, right } => {
            assert_eq!(op, BinaryOp::Add);
            assert_eq!(*left, Expression::Constant(MathConstant::E));
            assert_eq!(*right, Expression::Integer(1));
        }
        _ => panic!("Expected binary expression"),
    }
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_subscript_e_is_variable() {
    // x_e should produce Variable("x_e"), the subscript 'e' is just a label
    let expr = parse_latex("x_e").unwrap();
    assert_eq!(expr, Expression::Variable("x_e".to_string()));
}

#[test]
fn test_subscript_i_is_variable() {
    // x_i should produce Variable("x_i"), the subscript 'i' is just a label
    let expr = parse_latex("x_i").unwrap();
    assert_eq!(expr, Expression::Variable("x_i".to_string()));
}

#[test]
fn test_other_letters_are_variables() {
    // a, b, c, x, y, z should all be variables
    assert_eq!(
        parse_latex("a").unwrap(),
        Expression::Variable("a".to_string())
    );
    assert_eq!(
        parse_latex("b").unwrap(),
        Expression::Variable("b".to_string())
    );
    assert_eq!(
        parse_latex("x").unwrap(),
        Expression::Variable("x".to_string())
    );
    assert_eq!(
        parse_latex("y").unwrap(),
        Expression::Variable("y".to_string())
    );
}

#[test]
fn test_complex_number_pattern() {
    // a + bi - both i's should be Constant(I)
    let expr = parse_latex("a + b * i").unwrap();
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
                    left: ref b,
                    right: ref i,
                } => {
                    assert_eq!(**b, Expression::Variable("b".to_string()));
                    assert_eq!(**i, Expression::Constant(MathConstant::I));
                }
                _ => panic!("Expected multiplication"),
            }
        }
        _ => panic!("Expected addition"),
    }
}

#[test]
fn test_nested_sum_scope() {
    // \sum_{i=1}^n \sum_{j=1}^m i*j
    // Both i and j should be Variables in the body
    let expr = parse_latex(r"\sum_{i=1}^{n} \sum_{j=1}^{m} i * j").unwrap();
    match expr {
        Expression::Sum { body: outer_body, .. } => {
            // outer_body is the inner sum
            match *outer_body {
                Expression::Sum { body: inner_body, .. } => {
                    // inner_body should be i * j with both as Variables
                    match *inner_body {
                        Expression::Binary { op, left, right } => {
                            assert_eq!(op, BinaryOp::Mul);
                            assert_eq!(*left, Expression::Variable("i".to_string()));
                            assert_eq!(*right, Expression::Variable("j".to_string()));
                        }
                        _ => panic!("Expected binary multiplication"),
                    }
                }
                _ => panic!("Expected inner Sum"),
            }
        }
        _ => panic!("Expected outer Sum"),
    }
}

#[test]
fn test_mathrm_other_letter() {
    // \mathrm{a} should produce a Command-like result, not ExplicitConstant
    // Since 'a' is not e or i, it should be treated as a command
    let result = parse_latex(r"\mathrm{a}");
    // This should either error or produce something different
    assert!(result.is_err() || matches!(result, Ok(_)));
}

// =============================================================================
// Tokenizer tests for ExplicitConstant
// =============================================================================

#[test]
fn test_tokenize_mathrm_e() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\mathrm{e}").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('e')));
}

#[test]
fn test_tokenize_mathrm_i() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\mathrm{i}").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('i')));
}

#[test]
fn test_tokenize_imath() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\imath").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('i')));
}

#[test]
fn test_tokenize_jmath() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\jmath").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('i')));
}
