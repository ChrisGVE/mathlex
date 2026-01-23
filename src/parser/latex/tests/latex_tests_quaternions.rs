// Quaternion tests for LaTeX parser
// Tests for quaternion basis vector parsing (j and k)
use super::*;

// =============================================================================
// Explicit quaternion basis markers via \mathrm
// =============================================================================

#[test]
fn test_explicit_mathrm_j() {
    // \mathrm{j} should always parse as Constant(J)
    let expr = parse_latex(r"\mathrm{j}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::J));
}

#[test]
fn test_explicit_mathrm_k() {
    // \mathrm{k} should always parse as Constant(K)
    let expr = parse_latex(r"\mathrm{k}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::K));
}

// =============================================================================
// Explicit quaternion basis markers via \mathbf
// =============================================================================

#[test]
fn test_mathbf_j_is_quaternion_constant() {
    // \mathbf{j} should parse as Constant(J), not MarkedVector
    let expr = parse_latex(r"\mathbf{j}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::J));
}

#[test]
fn test_mathbf_k_is_quaternion_constant() {
    // \mathbf{k} should parse as Constant(K), not MarkedVector
    let expr = parse_latex(r"\mathbf{k}").unwrap();
    assert_eq!(expr, Expression::Constant(MathConstant::K));
}

#[test]
fn test_mathbf_i_is_still_marked_vector() {
    // \mathbf{i} should remain as MarkedVector (i is handled by \mathrm{i} for imaginary unit)
    // \mathbf{i} is typically used for unit vector, not imaginary unit
    let expr = parse_latex(r"\mathbf{i}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "i");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector for \\mathbf{{i}}, got {:?}", expr),
    }
}

// =============================================================================
// Default behavior (bare j and k are variables)
// =============================================================================

#[test]
fn test_bare_j_is_variable() {
    // Unbound j defaults to a variable (not quaternion constant)
    let expr = parse_latex("j").unwrap();
    assert_eq!(expr, Expression::Variable("j".to_string()));
}

#[test]
fn test_bare_k_is_variable() {
    // Unbound k defaults to a variable (not quaternion constant)
    let expr = parse_latex("k").unwrap();
    assert_eq!(expr, Expression::Variable("k".to_string()));
}

// =============================================================================
// Quaternion expressions
// =============================================================================

#[test]
fn test_quaternion_sum_explicit() {
    // 1 + \mathrm{i} + \mathrm{j} + \mathrm{k}
    let expr = parse_latex(r"1 + \mathrm{i} + \mathrm{j} + \mathrm{k}").unwrap();
    // This should parse as nested additions with quaternion constants
    match expr {
        Expression::Binary { op: BinaryOp::Add, .. } => {
            // Just verify it parses without error
        }
        _ => panic!("Expected binary addition, got {:?}", expr),
    }
}

#[test]
fn test_quaternion_with_coefficients() {
    // a*\mathrm{i} + b*\mathrm{j} + c*\mathrm{k}
    let expr = parse_latex(r"a * \mathrm{i} + b * \mathrm{j} + c * \mathrm{k}").unwrap();
    // Verify it parses - detailed structure validation would be extensive
    match expr {
        Expression::Binary { op: BinaryOp::Add, .. } => {}
        _ => panic!("Expected binary addition, got {:?}", expr),
    }
}

#[test]
fn test_mathbf_j_in_expression() {
    // x + \mathbf{j}
    let expr = parse_latex(r"x + \mathbf{j}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("x".to_string()));
            assert_eq!(*right, Expression::Constant(MathConstant::J));
        }
        _ => panic!("Expected binary addition, got {:?}", expr),
    }
}

#[test]
fn test_mathbf_k_in_expression() {
    // 2 * \mathbf{k}
    let expr = parse_latex(r"2 * \mathbf{k}").unwrap();
    match expr {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Integer(2));
            assert_eq!(*right, Expression::Constant(MathConstant::K));
        }
        _ => panic!("Expected binary multiplication, got {:?}", expr),
    }
}

// =============================================================================
// Other mathbf letters remain as MarkedVector
// =============================================================================

#[test]
fn test_mathbf_v_is_marked_vector() {
    // \mathbf{v} should be a MarkedVector (not affected by quaternion handling)
    let expr = parse_latex(r"\mathbf{v}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "v");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

#[test]
#[allow(non_snake_case)]
fn test_mathbf_F_is_marked_vector() {
    // \mathbf{F} should be a MarkedVector
    let expr = parse_latex(r"\mathbf{F}").unwrap();
    match expr {
        Expression::MarkedVector { name, notation } => {
            assert_eq!(name, "F");
            assert_eq!(notation, VectorNotation::Bold);
        }
        _ => panic!("Expected MarkedVector, got {:?}", expr),
    }
}

// =============================================================================
// Tokenizer tests for \mathrm{j} and \mathrm{k}
// =============================================================================

#[test]
fn test_tokenize_mathrm_j_parser() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\mathrm{j}").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('j')));
}

#[test]
fn test_tokenize_mathrm_k_parser() {
    use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
    let tokens = tokenize_latex(r"\mathrm{k}").unwrap();
    assert!(matches!(tokens[0].0, LatexToken::ExplicitConstant('k')));
}

// =============================================================================
// MathConstant::J and MathConstant::K properties
// =============================================================================

#[test]
fn test_j_and_k_are_distinct() {
    let j = Expression::Constant(MathConstant::J);
    let k = Expression::Constant(MathConstant::K);
    assert_ne!(j, k);
}

#[test]
fn test_j_k_different_from_i() {
    let i = Expression::Constant(MathConstant::I);
    let j = Expression::Constant(MathConstant::J);
    let k = Expression::Constant(MathConstant::K);
    assert_ne!(i, j);
    assert_ne!(i, k);
}
