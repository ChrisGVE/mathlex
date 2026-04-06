use crate::ast::{BinaryOp, Direction, Expression, IntegralBounds, MathConstant, UnaryOp};

// ── node count via fold ───────────────────────────────────────────────────────

#[test]
fn test_fold_count_equals_node_count_leaf() {
    let expr = Expression::Integer(42);
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_count_equals_node_count_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_count_equals_node_count_nested() {
    // ((x + y) * z): 5 nodes
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Variable("y".to_string())),
        }),
        right: Box::new(Expression::Variable("z".to_string())),
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 5);
}

#[test]
fn test_fold_count_equals_node_count_function() {
    let expr = Expression::Function {
        name: "f".to_string(),
        args: vec![
            Expression::Integer(1),
            Expression::Integer(2),
            Expression::Integer(3),
        ],
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 4);
}

// ── sum integers ──────────────────────────────────────────────────────────────

#[test]
fn test_fold_sum_integers_leaf() {
    let expr = Expression::Integer(7);
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 7);
}

#[test]
fn test_fold_sum_integers_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(3)),
        right: Box::new(Expression::Integer(4)),
    };
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 7);
}

#[test]
fn test_fold_sum_integers_nested() {
    // (1 + 2) + (3 + 4) = 10
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(3)),
            right: Box::new(Expression::Integer(4)),
        }),
    };
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 10);
}

#[test]
fn test_fold_sum_ignores_non_integers() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(5)),
        right: Box::new(Expression::Variable("x".to_string())),
    };
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 5);
}

// ── collect variable names ────────────────────────────────────────────────────

#[test]
fn test_fold_collect_variables_matches_find_variables() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("y".to_string())),
    };
    let folded: std::collections::HashSet<String> =
        expr.fold(std::collections::HashSet::new(), |mut acc, e| {
            if let Expression::Variable(name) = e {
                acc.insert(name.clone());
            }
            acc
        });
    let found = expr.find_variables();
    assert_eq!(folded, found);
}

#[test]
fn test_fold_collect_variables_nested() {
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Variable("a".to_string())),
            right: Box::new(Expression::Variable("b".to_string())),
        }),
        right: Box::new(Expression::Variable("c".to_string())),
    };
    let folded: std::collections::HashSet<String> =
        expr.fold(std::collections::HashSet::new(), |mut acc, e| {
            if let Expression::Variable(name) = e {
                acc.insert(name.clone());
            }
            acc
        });
    assert_eq!(folded.len(), 3);
    assert!(folded.contains("a"));
    assert!(folded.contains("b"));
    assert!(folded.contains("c"));
}

// ── additional variants ───────────────────────────────────────────────────────

#[test]
fn test_fold_unary() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Integer(9)),
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 2);
}

#[test]
fn test_fold_integral_counts_all_nodes() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(0)),
            upper: Box::new(Expression::Integer(1)),
        }),
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_limit() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::Constant(MathConstant::Infinity)),
        direction: Direction::Both,
    };
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_vector() {
    let expr = Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Integer(2),
        Expression::Integer(3),
    ]);
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 6);
}

#[test]
fn test_fold_matrix() {
    let expr = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(4)],
    ]);
    let sum = expr.fold(0i64, |acc, e| match e {
        Expression::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 10);
}

// ── traversal order ───────────────────────────────────────────────────────────

#[test]
fn test_fold_visits_leaves_before_parent() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };
    let order = std::cell::RefCell::new(Vec::new());
    expr.fold((), |_, e| {
        let label = match e {
            Expression::Integer(n) => format!("int({})", n),
            Expression::Binary { .. } => "binary".to_string(),
            _ => "other".to_string(),
        };
        order.borrow_mut().push(label);
    });
    let order = order.into_inner();
    assert_eq!(order[0], "int(1)");
    assert_eq!(order[1], "int(2)");
    assert_eq!(order[2], "binary");
}
