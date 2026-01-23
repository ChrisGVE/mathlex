//! Tests for set theory parsing in LaTeX.

use crate::ast::{Expression, SetOp, SetRelation};
use crate::parser::parse_latex;

// ============================================================
// Set Operation Tests
// ============================================================

#[test]
fn test_set_union() {
    let expr = parse_latex(r"A \cup B").unwrap();
    match expr {
        Expression::SetOperation { op, left, right } => {
            assert_eq!(op, SetOp::Union);
            assert_eq!(*left, Expression::Variable("A".to_string()));
            assert_eq!(*right, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetOperation Union, got {:?}", expr),
    }
}

#[test]
fn test_set_intersection() {
    let expr = parse_latex(r"A \cap B").unwrap();
    match expr {
        Expression::SetOperation { op, left, right } => {
            assert_eq!(op, SetOp::Intersection);
            assert_eq!(*left, Expression::Variable("A".to_string()));
            assert_eq!(*right, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetOperation Intersection, got {:?}", expr),
    }
}

#[test]
fn test_set_difference() {
    let expr = parse_latex(r"A \setminus B").unwrap();
    match expr {
        Expression::SetOperation { op, left, right } => {
            assert_eq!(op, SetOp::Difference);
            assert_eq!(*left, Expression::Variable("A".to_string()));
            assert_eq!(*right, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetOperation Difference, got {:?}", expr),
    }
}

// ============================================================
// Set Relation Tests
// ============================================================

#[test]
fn test_subset() {
    let expr = parse_latex(r"A \subset B").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::Subset);
            assert_eq!(*element, Expression::Variable("A".to_string()));
            assert_eq!(*set, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetRelationExpr Subset, got {:?}", expr),
    }
}

#[test]
fn test_subseteq() {
    let expr = parse_latex(r"A \subseteq B").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::SubsetEq);
            assert_eq!(*element, Expression::Variable("A".to_string()));
            assert_eq!(*set, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetRelationExpr SubsetEq, got {:?}", expr),
    }
}

#[test]
fn test_superset() {
    let expr = parse_latex(r"A \supset B").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::Superset);
            assert_eq!(*element, Expression::Variable("A".to_string()));
            assert_eq!(*set, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetRelationExpr Superset, got {:?}", expr),
    }
}

#[test]
fn test_supseteq() {
    let expr = parse_latex(r"A \supseteq B").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation,
            element,
            set,
        } => {
            assert_eq!(relation, SetRelation::SupersetEq);
            assert_eq!(*element, Expression::Variable("A".to_string()));
            assert_eq!(*set, Expression::Variable("B".to_string()));
        }
        _ => panic!("Expected SetRelationExpr SupersetEq, got {:?}", expr),
    }
}

// ============================================================
// Empty Set Tests
// ============================================================

#[test]
fn test_emptyset() {
    let expr = parse_latex(r"\emptyset").unwrap();
    assert_eq!(expr, Expression::EmptySet);
}

#[test]
fn test_varnothing() {
    let expr = parse_latex(r"\varnothing").unwrap();
    assert_eq!(expr, Expression::EmptySet);
}

// ============================================================
// Power Set Tests
// ============================================================

#[test]
fn test_powerset_parens() {
    let expr = parse_latex(r"\mathcal{P}(A)").unwrap();
    match expr {
        Expression::PowerSet { set } => {
            assert_eq!(*set, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected PowerSet, got {:?}", expr),
    }
}

#[test]
fn test_powerset_braces() {
    let expr = parse_latex(r"\mathcal{P}{A}").unwrap();
    match expr {
        Expression::PowerSet { set } => {
            assert_eq!(*set, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected PowerSet, got {:?}", expr),
    }
}

// ============================================================
// Precedence Tests
// ============================================================

#[test]
fn test_intersection_higher_than_union() {
    // Intersection binds tighter: A \cup B \cap C = A \cup (B \cap C)
    let expr = parse_latex(r"A \cup B \cap C").unwrap();
    match expr {
        Expression::SetOperation {
            op: SetOp::Union,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("A".to_string()));
            // Right should be B \cap C
            match *right {
                Expression::SetOperation {
                    op: SetOp::Intersection,
                    ..
                } => {}
                _ => panic!("Expected Intersection as right operand"),
            }
        }
        _ => panic!("Expected Union at top level, got {:?}", expr),
    }
}

#[test]
fn test_union_left_associative() {
    // A \cup B \cup C = (A \cup B) \cup C
    let expr = parse_latex(r"A \cup B \cup C").unwrap();
    match expr {
        Expression::SetOperation {
            op: SetOp::Union,
            left,
            right,
        } => {
            assert_eq!(*right, Expression::Variable("C".to_string()));
            // Left should be A \cup B
            match *left {
                Expression::SetOperation {
                    op: SetOp::Union, ..
                } => {}
                _ => panic!("Expected Union as left operand"),
            }
        }
        _ => panic!("Expected Union at top level, got {:?}", expr),
    }
}

// ============================================================
// Complex Expression Tests
// ============================================================

#[test]
fn test_set_operations_with_membership() {
    // x \in A \cup B
    let expr = parse_latex(r"x \in A \cup B").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation: SetRelation::In,
            element,
            set,
        } => {
            assert_eq!(*element, Expression::Variable("x".to_string()));
            // Set should be A \cup B
            match *set {
                Expression::SetOperation {
                    op: SetOp::Union, ..
                } => {}
                _ => panic!("Expected Union as set operand"),
            }
        }
        _ => panic!("Expected SetRelationExpr, got {:?}", expr),
    }
}

#[test]
fn test_subset_of_union() {
    // A \subset B \cup C
    let expr = parse_latex(r"A \subset B \cup C").unwrap();
    match expr {
        Expression::SetRelationExpr {
            relation: SetRelation::Subset,
            element,
            set,
        } => {
            assert_eq!(*element, Expression::Variable("A".to_string()));
            match *set {
                Expression::SetOperation {
                    op: SetOp::Union, ..
                } => {}
                _ => panic!("Expected Union as set operand"),
            }
        }
        _ => panic!("Expected SetRelationExpr Subset, got {:?}", expr),
    }
}

#[test]
fn test_difference_with_intersection() {
    // A \setminus B \cap C = A \setminus (B \cap C)
    let expr = parse_latex(r"A \setminus B \cap C").unwrap();
    match expr {
        Expression::SetOperation {
            op: SetOp::Difference,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::Variable("A".to_string()));
            // Right should be B \cap C
            match *right {
                Expression::SetOperation {
                    op: SetOp::Intersection,
                    ..
                } => {}
                _ => panic!("Expected Intersection as right operand"),
            }
        }
        _ => panic!("Expected Difference at top level, got {:?}", expr),
    }
}

#[test]
fn test_emptyset_in_union() {
    // \emptyset \cup A
    let expr = parse_latex(r"\emptyset \cup A").unwrap();
    match expr {
        Expression::SetOperation {
            op: SetOp::Union,
            left,
            right,
        } => {
            assert_eq!(*left, Expression::EmptySet);
            assert_eq!(*right, Expression::Variable("A".to_string()));
        }
        _ => panic!("Expected Union with EmptySet, got {:?}", expr),
    }
}
