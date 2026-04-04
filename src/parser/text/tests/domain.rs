//! Domain-specific tests: quantifiers, sets, logical operators, integration.

use super::*;

mod quantifiers {
    use super::*;

    #[test]
    fn test_parse_forall_without_domain() {
        let expr = parse("forall x, x > 0").unwrap();
        match expr {
            Expression::ForAll {
                variable,
                domain,
                body,
            } => {
                assert_eq!(variable, "x");
                assert!(domain.is_none());
                assert!(matches!(*body, Expression::Inequality { .. }));
            }
            _ => panic!("Expected ForAll, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_forall_with_domain() {
        let expr = parse("forall x in S, x > 0").unwrap();
        match expr {
            Expression::ForAll {
                variable,
                domain,
                body,
            } => {
                assert_eq!(variable, "x");
                assert!(domain.is_some());
                assert_eq!(*domain.unwrap(), Expression::Variable("S".to_string()));
                assert!(matches!(*body, Expression::Inequality { .. }));
            }
            _ => panic!("Expected ForAll, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_exists_without_domain() {
        let expr = parse("exists x, x = 0").unwrap();
        match expr {
            Expression::Exists {
                variable,
                domain,
                body,
                unique,
            } => {
                assert_eq!(variable, "x");
                assert!(domain.is_none());
                assert!(!unique);
                assert!(matches!(*body, Expression::Equation { .. }));
            }
            _ => panic!("Expected Exists, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_exists_with_domain() {
        let expr = parse("exists y in R, y^2 = 2").unwrap();
        match expr {
            Expression::Exists {
                variable,
                domain,
                body,
                unique,
            } => {
                assert_eq!(variable, "y");
                assert!(domain.is_some());
                assert_eq!(*domain.unwrap(), Expression::Variable("R".to_string()));
                assert!(!unique);
                assert!(matches!(*body, Expression::Equation { .. }));
            }
            _ => panic!("Expected Exists, got {:?}", expr),
        }
    }
}

mod set_operations {
    use super::*;

    #[test]
    fn test_parse_union() {
        let expr = parse("A union B").unwrap();
        match expr {
            Expression::SetOperation { op, left, right } => {
                assert_eq!(op, SetOp::Union);
                assert_eq!(*left, Expression::Variable("A".to_string()));
                assert_eq!(*right, Expression::Variable("B".to_string()));
            }
            _ => panic!("Expected SetOperation, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_intersect() {
        let expr = parse("A intersect B").unwrap();
        match expr {
            Expression::SetOperation { op, left, right } => {
                assert_eq!(op, SetOp::Intersection);
                assert_eq!(*left, Expression::Variable("A".to_string()));
                assert_eq!(*right, Expression::Variable("B".to_string()));
            }
            _ => panic!("Expected SetOperation, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_set_membership() {
        let expr = parse("x in S").unwrap();
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
    fn test_parse_set_non_membership() {
        let expr = parse("x notin S").unwrap();
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

    #[test]
    fn test_parse_chained_set_operations() {
        let expr = parse("A union B intersect C").unwrap();
        match expr {
            Expression::SetOperation {
                op: SetOp::Intersection,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expression::SetOperation {
                        op: SetOp::Union,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Variable("C".to_string()));
            }
            _ => panic!("Expected chained SetOperation, got {:?}", expr),
        }
    }
}

mod logical_operators {
    use super::*;

    #[test]
    fn test_parse_and() {
        let expr = parse("P and Q").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::And);
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[0], Expression::Variable("P".to_string()));
                assert_eq!(operands[1], Expression::Variable("Q".to_string()));
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_or() {
        let expr = parse("P or Q").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::Or);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_not() {
        let expr = parse("not P").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::Not);
                assert_eq!(operands.len(), 1);
                assert_eq!(operands[0], Expression::Variable("P".to_string()));
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_implies() {
        let expr = parse("P implies Q").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::Implies);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_iff() {
        let expr = parse("P iff Q").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::Iff);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_complex_logical_expression() {
        let expr = parse("P and Q or R").unwrap();
        // Should parse as (P and Q) or R due to left-associativity
        match expr {
            Expression::Logical {
                op: LogicalOp::Or,
                operands,
            } => {
                assert_eq!(operands.len(), 2);
                assert!(matches!(
                    operands[0],
                    Expression::Logical {
                        op: LogicalOp::And,
                        ..
                    }
                ));
                assert_eq!(operands[1], Expression::Variable("R".to_string()));
            }
            _ => panic!("Expected Logical with Or, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_not_with_expression() {
        let expr = parse("not (x > 0)").unwrap();
        match expr {
            Expression::Logical { op, operands } => {
                assert_eq!(op, LogicalOp::Not);
                assert_eq!(operands.len(), 1);
                assert!(matches!(operands[0], Expression::Inequality { .. }));
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }
}

mod operator_precedence_extended {
    use super::*;

    #[test]
    fn test_arithmetic_before_logical() {
        let expr = parse("x + 1 > 0 and y < 2").unwrap();
        // Should parse as: (x + 1 > 0) and (y < 2)
        match expr {
            Expression::Logical {
                op: LogicalOp::And,
                operands,
            } => {
                assert_eq!(operands.len(), 2);
                assert!(matches!(operands[0], Expression::Inequality { .. }));
                assert!(matches!(operands[1], Expression::Inequality { .. }));
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }

    #[test]
    fn test_set_operations_before_logical() {
        let expr = parse("x in A and y in B").unwrap();
        // Should parse as: (x in A) and (y in B)
        match expr {
            Expression::Logical {
                op: LogicalOp::And,
                operands,
            } => {
                assert_eq!(operands.len(), 2);
                assert!(matches!(operands[0], Expression::SetRelationExpr { .. }));
                assert!(matches!(operands[1], Expression::SetRelationExpr { .. }));
            }
            _ => panic!("Expected Logical, got {:?}", expr),
        }
    }
}

mod integration_tests {
    use super::*;

    #[test]
    fn test_complex_expression_with_all_features() {
        let expr = parse("forall x in S, grad(f) > 0 and x notin T").unwrap();
        match expr {
            Expression::ForAll {
                variable,
                domain,
                body,
            } => {
                assert_eq!(variable, "x");
                assert!(domain.is_some());
                match *body {
                    Expression::Logical {
                        op: LogicalOp::And,
                        operands,
                    } => {
                        assert_eq!(operands.len(), 2);
                        assert!(matches!(operands[0], Expression::Inequality { .. }));
                        assert!(matches!(operands[1], Expression::SetRelationExpr { .. }));
                    }
                    _ => panic!("Expected Logical in body"),
                }
            }
            _ => panic!("Expected ForAll"),
        }
    }

    #[test]
    fn test_vector_operations_in_equations() {
        let expr = parse("dot(u, v) = 0").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(*left, Expression::DotProduct { .. }));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected Equation"),
        }
    }

    #[test]
    fn test_nested_quantifiers() {
        let expr = parse("forall x, exists y, x + y = 0").unwrap();
        match expr {
            Expression::ForAll { body, .. } => match *body {
                Expression::Exists { body, .. } => {
                    assert!(matches!(*body, Expression::Equation { .. }));
                }
                _ => panic!("Expected Exists in body"),
            },
            _ => panic!("Expected ForAll"),
        }
    }

    #[test]
    fn test_vector_calculus_with_sets() {
        let expr = parse("div(F) in R").unwrap();
        match expr {
            Expression::SetRelationExpr {
                relation,
                element,
                set,
            } => {
                assert_eq!(relation, SetRelation::In);
                assert!(matches!(*element, Expression::Divergence { .. }));
                assert_eq!(*set, Expression::Variable("R".to_string()));
            }
            _ => panic!("Expected SetRelationExpr"),
        }
    }

    #[test]
    fn test_logical_with_arithmetic() {
        let expr = parse("x^2 + y^2 < 1 or x > 2").unwrap();
        match expr {
            Expression::Logical {
                op: LogicalOp::Or,
                operands,
            } => {
                assert_eq!(operands.len(), 2);
                assert!(matches!(operands[0], Expression::Inequality { .. }));
                assert!(matches!(operands[1], Expression::Inequality { .. }));
            }
            _ => panic!("Expected Logical"),
        }
    }

    #[test]
    fn test_backward_compatibility_arithmetic() {
        let expr = parse("2 + 3 * 4^5").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add, ..
            } => {}
            _ => panic!("Expected binary addition"),
        }
    }

    #[test]
    fn test_backward_compatibility_functions() {
        let expr = parse("sin(x) + cos(y)").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add, ..
            } => {}
            _ => panic!("Expected binary addition"),
        }
    }
}
