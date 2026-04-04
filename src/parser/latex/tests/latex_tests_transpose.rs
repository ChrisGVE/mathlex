// Transpose shorthand tests for LaTeX parser
use super::*;

// A^T bare letter

#[test]
fn test_transpose_bare_t() {
    let expr = parse_latex("A^T").unwrap();
    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, crate::ast::UnaryOp::Transpose);
            assert_eq!(*operand, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Unary(Transpose, A), got {:?}", expr),
    }
}

// A^{T} braced

#[test]
fn test_transpose_braced_t() {
    let expr = parse_latex("A^{T}").unwrap();
    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, crate::ast::UnaryOp::Transpose);
            assert_eq!(*operand, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Unary(Transpose, A), got {:?}", expr),
    }
}

// A^\top command

#[test]
fn test_transpose_top_command() {
    let expr = parse_latex(r"A^\top").unwrap();
    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, crate::ast::UnaryOp::Transpose);
            assert_eq!(*operand, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Unary(Transpose, A), got {:?}", expr),
    }
}

// Non-capital T still forms a power

#[test]
fn test_power_lowercase_t_is_not_transpose() {
    let expr = parse_latex("x^t").unwrap();
    assert!(
        matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Pow,
                ..
            }
        ),
        "Expected Pow expression, got {:?}",
        expr
    );
}

// Numeric exponent is still a power

#[test]
fn test_power_number_is_not_transpose() {
    let expr = parse_latex("A^2").unwrap();
    assert!(
        matches!(
            expr,
            Expression::Binary {
                op: BinaryOp::Pow,
                ..
            }
        ),
        "Expected Pow expression, got {:?}",
        expr
    );
}

// Transpose of a more complex expression

#[test]
fn test_transpose_of_matrix_times_vector() {
    let expr = parse_latex(r"(Ax)^T").unwrap();
    match expr {
        Expression::Unary { op, operand } => {
            assert_eq!(op, crate::ast::UnaryOp::Transpose);
            // operand should be implicit product Ax
            assert!(
                matches!(
                    *operand,
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ),
                "Expected Mul expression inside transpose, got {:?}",
                operand
            );
        }
        _ => panic!("Expected Unary(Transpose, ...), got {:?}", expr),
    }
}

// Round-trip: to_latex emits {}^T, parser must re-parse it

#[test]
fn test_transpose_roundtrip_bare() {
    use crate::latex::ToLatex;
    let expr = parse_latex("A^T").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}

#[test]
fn test_transpose_roundtrip_top() {
    use crate::latex::ToLatex;
    let expr = parse_latex(r"A^\top").unwrap();
    let latex_out = expr.to_latex();
    let expr2 = parse_latex(&latex_out).unwrap();
    assert_eq!(
        expr, expr2,
        "Round-trip failed: {} -> {:?}",
        latex_out, expr2
    );
}
