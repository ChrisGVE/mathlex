use crate::ast::{
    BinaryOp, Direction, ExprKind, Expression, IntegralBounds, MathConstant, UnaryOp,
};

// ── node count via fold ───────────────────────────────────────────────────────

#[test]
fn test_fold_count_equals_node_count_leaf() {
    let expr = Expression::integer(42);
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_count_equals_node_count_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_count_equals_node_count_nested() {
    // ((x + y) * z): 5 nodes
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("x".to_string())),
                right: Box::new(Expression::variable("y".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("z".to_string())),
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 5);
}

#[test]
fn test_fold_count_equals_node_count_function() {
    let expr: Expression = ExprKind::Function {
        name: "f".to_string(),
        args: vec![
            Expression::integer(1),
            Expression::integer(2),
            Expression::integer(3),
        ],
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 4);
}

// ── sum integers ──────────────────────────────────────────────────────────────

#[test]
fn test_fold_sum_integers_leaf() {
    let expr = Expression::integer(7);
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 7);
}

#[test]
fn test_fold_sum_integers_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(3)),
        right: Box::new(Expression::integer(4)),
    }
    .into();
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 7);
}

#[test]
fn test_fold_sum_integers_nested() {
    // (1 + 2) + (3 + 4) = 10
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(1)),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        right: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(3)),
                right: Box::new(Expression::integer(4)),
            }
            .into(),
        ),
    }
    .into();
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 10);
}

#[test]
fn test_fold_sum_ignores_non_integers() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(5)),
        right: Box::new(Expression::variable("x".to_string())),
    }
    .into();
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 5);
}

// ── collect variable names ────────────────────────────────────────────────────

#[test]
fn test_fold_collect_variables_matches_find_variables() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("y".to_string())),
    }
    .into();
    let folded: std::collections::HashSet<String> =
        expr.fold(std::collections::HashSet::new(), |mut acc, e| {
            if let ExprKind::Variable(ref name) = e.kind {
                acc.insert(name.clone());
            }
            acc
        });
    let found = expr.find_variables();
    assert_eq!(folded, found);
}

#[test]
fn test_fold_collect_variables_nested() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::variable("a".to_string())),
                right: Box::new(Expression::variable("b".to_string())),
            }
            .into(),
        ),
        right: Box::new(Expression::variable("c".to_string())),
    }
    .into();
    let folded: std::collections::HashSet<String> =
        expr.fold(std::collections::HashSet::new(), |mut acc, e| {
            if let ExprKind::Variable(ref name) = e.kind {
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
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::integer(9)),
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
    assert_eq!(count, 2);
}

#[test]
fn test_fold_integral_counts_all_nodes() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::variable("x".to_string())),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(0)),
            upper: Box::new(Expression::integer(1)),
        }),
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_limit() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::variable("f".to_string())),
        var: "x".to_string(),
        to: Box::new(Expression::constant(MathConstant::Infinity)),
        direction: Direction::Both,
    }
    .into();
    let count = expr.fold(0usize, |acc, _| acc + 1);
    assert_eq!(count, expr.node_count());
}

#[test]
fn test_fold_vector() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::integer(2),
        Expression::integer(3),
    ])
    .into();
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 6);
}

#[test]
fn test_fold_matrix() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ])
    .into();
    let sum = expr.fold(0i64, |acc, e| match &e.kind {
        ExprKind::Integer(n) => acc + n,
        _ => acc,
    });
    assert_eq!(sum, 10);
}

// ── traversal order ───────────────────────────────────────────────────────────

#[test]
fn test_fold_visits_leaves_before_parent() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let order = std::cell::RefCell::new(Vec::new());
    expr.fold((), |_, e| {
        let label = match &e.kind {
            ExprKind::Integer(n) => format!("int({})", n),
            ExprKind::Binary { .. } => "binary".to_string(),
            _ => "other".to_string(),
        };
        order.borrow_mut().push(label);
    });
    let order = order.into_inner();
    assert_eq!(order[0], "int(1)");
    assert_eq!(order[1], "int(2)");
    assert_eq!(order[2], "binary");
}
