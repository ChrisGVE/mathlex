//! Tests for quantifiers and logical operators in LaTeX parsing.

use crate::ast::{Expression, LogicalOp, SetRelation};
use crate::parser::parse_latex;

// ============================================================
// Universal Quantifier Tests
// ============================================================

#[test]
fn test_forall_basic() {
    let expr = parse_latex(r"\forall x P").unwrap();
    match expr {
        Expression::ForAll {
            variable,
            domain,
            body,
        } => {
            assert_eq!(variable, "x");
            assert!(domain.is_none());
            assert_eq!(*body, Expression::Variable("P".to_string()));
        }
        _ => panic!("Expected ForAll, got {:?}", expr),
    }
}

#[test]
fn test_forall_with_domain() {
    let expr = parse_latex(r"\forall x \in S P").unwrap();
    match expr {
        Expression::ForAll {
            variable,
            domain,
            body,
        } => {
            assert_eq!(variable, "x");
            assert!(domain.is_some());
            assert_eq!(
                *domain.unwrap(),
                Expression::Variable("S".to_string())
            );
            assert_eq!(*body, Expression::Variable("P".to_string()));
        }
        _ => panic!("Expected ForAll, got {:?}", expr),
    }
}

// ============================================================
// Existential Quantifier Tests
// ============================================================

#[test]
fn test_exists_basic() {
    let expr = parse_latex(r"\exists x P").unwrap();
    match expr {
        Expression::Exists {
            variable,
            domain,
            body,
            unique,
        } => {
            assert_eq!(variable, "x");
            assert!(domain.is_none());
            assert_eq!(*body, Expression::Variable("P".to_string()));
            assert!(!unique);
        }
        _ => panic!("Expected Exists, got {:?}", expr),
    }
}

#[test]
fn test_exists_with_domain() {
    let expr = parse_latex(r"\exists x \in S P").unwrap();
    match expr {
        Expression::Exists {
            variable,
            domain,
            unique,
            ..
        } => {
            assert_eq!(variable, "x");
            assert!(domain.is_some());
            assert!(!unique);
        }
        _ => panic!("Expected Exists, got {:?}", expr),
    }
}

// ============================================================
// Logical Operator Tests
// ============================================================

#[test]
fn test_logical_and() {
    let expr = parse_latex(r"P \land Q").unwrap();
    match expr {
        Expression::Logical { op, operands } => {
            assert_eq!(op, LogicalOp::And);
            assert_eq!(operands.len(), 2);
            assert_eq!(operands[0], Expression::Variable("P".to_string()));
            assert_eq!(operands[1], Expression::Variable("Q".to_string()));
        }
        _ => panic!("Expected Logical And, got {:?}", expr),
    }
}

#[test]
fn test_logical_or() {
    let expr = parse_latex(r"P \lor Q").unwrap();
    match expr {
        Expression::Logical { op, operands } => {
            assert_eq!(op, LogicalOp::Or);
            assert_eq!(operands.len(), 2);
        }
        _ => panic!("Expected Logical Or, got {:?}", expr),
    }
}

#[test]
fn test_logical_implies() {
    let expr = parse_latex(r"P \implies Q").unwrap();
    match expr {
        Expression::Logical { op, operands } => {
            assert_eq!(op, LogicalOp::Implies);
            assert_eq!(operands.len(), 2);
        }
        _ => panic!("Expected Logical Implies, got {:?}", expr),
    }
}

#[test]
fn test_logical_iff() {
    let expr = parse_latex(r"P \iff Q").unwrap();
    match expr {
        Expression::Logical { op, operands } => {
            assert_eq!(op, LogicalOp::Iff);
            assert_eq!(operands.len(), 2);
        }
        _ => panic!("Expected Logical Iff, got {:?}", expr),
    }
}

#[test]
fn test_logical_not() {
    let expr = parse_latex(r"\lnot P").unwrap();
    match expr {
        Expression::Logical { op, operands } => {
            assert_eq!(op, LogicalOp::Not);
            assert_eq!(operands.len(), 1);
            assert_eq!(operands[0], Expression::Variable("P".to_string()));
        }
        _ => panic!("Expected Logical Not, got {:?}", expr),
    }
}

#[test]
fn test_logical_neg_alias() {
    // \neg is an alias for \lnot
    let expr = parse_latex(r"\neg P").unwrap();
    match expr {
        Expression::Logical { op, .. } => {
            assert_eq!(op, LogicalOp::Not);
        }
        _ => panic!("Expected Logical Not, got {:?}", expr),
    }
}

// ============================================================
// Set Membership Tests
// ============================================================

#[test]
fn test_set_membership_in() {
    let expr = parse_latex(r"x \in S").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::In);
            assert_eq!(*element, Expression::Variable("x".to_string()));
            assert_eq!(*set, Expression::Variable("S".to_string()));
        }
        _ => panic!("Expected SetRelationExpr, got {:?}", expr),
    }
}

#[test]
fn test_set_membership_notin() {
    let expr = parse_latex(r"x \notin S").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::NotIn);
            assert_eq!(*element, Expression::Variable("x".to_string()));
            assert_eq!(*set, Expression::Variable("S".to_string()));
        }
        _ => panic!("Expected SetRelationExpr, got {:?}", expr),
    }
}

// ============================================================
// Precedence Tests
// ============================================================

#[test]
fn test_and_or_precedence() {
    // AND has higher precedence than OR: P \lor Q \land R = P \lor (Q \land R)
    let expr = parse_latex(r"P \lor Q \land R").unwrap();
    match expr {
        Expression::Logical {
            op: LogicalOp::Or,
            operands,
        } => {
            assert_eq!(operands[0], Expression::Variable("P".to_string()));
            // Second operand should be Q \land R
            match &operands[1] {
                Expression::Logical {
                    op: LogicalOp::And, ..
                } => {}
                _ => panic!("Expected And as second operand"),
            }
        }
        _ => panic!("Expected Logical Or at top level, got {:?}", expr),
    }
}

#[test]
fn test_implies_or_precedence() {
    // OR has higher precedence than IMPLIES: P \implies Q \lor R = P \implies (Q \lor R)
    let expr = parse_latex(r"P \implies Q \lor R").unwrap();
    match expr {
        Expression::Logical {
            op: LogicalOp::Implies,
            operands,
        } => {
            assert_eq!(operands[0], Expression::Variable("P".to_string()));
            match &operands[1] {
                Expression::Logical {
                    op: LogicalOp::Or, ..
                } => {}
                _ => panic!("Expected Or as second operand"),
            }
        }
        _ => panic!("Expected Logical Implies at top level, got {:?}", expr),
    }
}

// ============================================================
// Complex Expression Tests
// ============================================================

#[test]
fn test_quantifier_with_logical() {
    let expr = parse_latex(r"\forall x P \land Q").unwrap();
    match expr {
        Expression::ForAll { body, .. } => {
            // The body should be P \land Q
            match *body {
                Expression::Logical {
                    op: LogicalOp::And, ..
                } => {}
                _ => panic!("Expected body to be And expression, got {:?}", body),
            }
        }
        _ => panic!("Expected ForAll, got {:?}", expr),
    }
}
