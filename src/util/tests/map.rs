use crate::ast::{
    BinaryOp, Direction, ExprKind, Expression, IntegralBounds, MathConstant, UnaryOp,
};

fn double_integers(e: Expression) -> Expression {
    match &e.kind {
        ExprKind::Integer(n) => ExprKind::Integer(n * 2).into(),
        _ => e,
    }
}

fn rename_x_to_y(e: Expression) -> Expression {
    match &e.kind {
        ExprKind::Variable(ref name) if name == "x" => ExprKind::Variable("y".to_string()).into(),
        _ => e,
    }
}

// ── identity ─────────────────────────────────────────────────────────────────

#[test]
fn test_map_identity_leaf() {
    let expr = Expression::integer(42);
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_variable() {
    let expr = Expression::variable("x".to_string());
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_constant() {
    let expr = Expression::constant(MathConstant::Pi);
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    assert_eq!(expr.map(|e| e), expr);
}

// ── integer transform ─────────────────────────────────────────────────────────

#[test]
fn test_map_double_integers_leaf() {
    let expr = Expression::integer(3);
    assert_eq!(expr.map(double_integers), Expression::integer(6));
}

#[test]
fn test_map_double_integers_binary() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(2)),
        right: Box::new(Expression::integer(3)),
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Binary { left, right, .. } => {
            assert_eq!(**left, Expression::integer(4));
            assert_eq!(**right, Expression::integer(6));
        }
        _ => panic!("expected Binary"),
    }
}

#[test]
fn test_map_double_integers_nested() {
    // (1 + 2) * 3  →  (2 + 4) * 6
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Mul,
        left: Box::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expression::integer(1)),
                right: Box::new(Expression::integer(2)),
            }
            .into(),
        ),
        right: Box::new(Expression::integer(3)),
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(**right, Expression::integer(6));
            match &left.kind {
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    left: ll,
                    right: lr,
                } => {
                    assert_eq!(**ll, Expression::integer(2));
                    assert_eq!(**lr, Expression::integer(4));
                }
                _ => panic!("expected inner Binary Add"),
            }
        }
        _ => panic!("expected Binary Mul"),
    }
}

// ── variable rename ───────────────────────────────────────────────────────────

#[test]
fn test_map_rename_variable() {
    let expr = Expression::variable("x".to_string());
    assert_eq!(
        expr.map(rename_x_to_y),
        Expression::variable("y".to_string())
    );
}

#[test]
fn test_map_rename_leaves_other_variables() {
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::variable("x".to_string())),
        right: Box::new(Expression::variable("z".to_string())),
    }
    .into();
    let result = expr.map(rename_x_to_y);
    match &result.kind {
        ExprKind::Binary { left, right, .. } => {
            assert_eq!(**left, Expression::variable("y".to_string()));
            assert_eq!(**right, Expression::variable("z".to_string()));
        }
        _ => panic!("expected Binary"),
    }
}

// ── unary ─────────────────────────────────────────────────────────────────────

#[test]
fn test_map_unary() {
    let expr: Expression = ExprKind::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::integer(5)),
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Unary { operand, .. } => {
            assert_eq!(**operand, Expression::integer(10));
        }
        _ => panic!("expected Unary"),
    }
}

// ── function ──────────────────────────────────────────────────────────────────

#[test]
fn test_map_function_args() {
    let expr: Expression = ExprKind::Function {
        name: "f".to_string(),
        args: vec![Expression::integer(1), Expression::integer(2)],
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "f");
            assert_eq!(args[0], Expression::integer(2));
            assert_eq!(args[1], Expression::integer(4));
        }
        _ => panic!("expected Function"),
    }
}

// ── derivative ────────────────────────────────────────────────────────────────

#[test]
fn test_map_derivative_transforms_body() {
    let expr: Expression = ExprKind::Derivative {
        expr: Box::new(Expression::integer(7)),
        var: "x".to_string(),
        order: 1,
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Derivative {
            expr: inner,
            var,
            order,
        } => {
            assert_eq!(**inner, Expression::integer(14));
            assert_eq!(var, "x");
            assert_eq!(*order, 1);
        }
        _ => panic!("expected Derivative"),
    }
}

// ── integral ──────────────────────────────────────────────────────────────────

#[test]
fn test_map_integral_transforms_integrand_and_bounds() {
    let expr: Expression = ExprKind::Integral {
        integrand: Box::new(Expression::integer(3)),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::integer(1)),
            upper: Box::new(Expression::integer(2)),
        }),
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Integral {
            integrand, bounds, ..
        } => {
            assert_eq!(**integrand, Expression::integer(6));
            let b = bounds.as_ref().unwrap();
            assert_eq!(*b.lower, Expression::integer(2));
            assert_eq!(*b.upper, Expression::integer(4));
        }
        _ => panic!("expected Integral"),
    }
}

// ── sum / product ─────────────────────────────────────────────────────────────

#[test]
fn test_map_sum_transforms_bounds_and_body() {
    let expr: Expression = ExprKind::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::integer(1)),
        upper: Box::new(Expression::integer(5)),
        body: Box::new(Expression::integer(3)),
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Sum {
            lower, upper, body, ..
        } => {
            assert_eq!(**lower, Expression::integer(2));
            assert_eq!(**upper, Expression::integer(10));
            assert_eq!(**body, Expression::integer(6));
        }
        _ => panic!("expected Sum"),
    }
}

// ── vector / matrix ───────────────────────────────────────────────────────────

#[test]
fn test_map_vector() {
    let expr: Expression = ExprKind::Vector(vec![
        Expression::integer(1),
        Expression::integer(2),
        Expression::integer(3),
    ])
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Vector(elems) => {
            assert_eq!(elems[0], Expression::integer(2));
            assert_eq!(elems[1], Expression::integer(4));
            assert_eq!(elems[2], Expression::integer(6));
        }
        _ => panic!("expected Vector"),
    }
}

#[test]
fn test_map_matrix() {
    let expr: Expression = ExprKind::Matrix(vec![
        vec![Expression::integer(1), Expression::integer(2)],
        vec![Expression::integer(3), Expression::integer(4)],
    ])
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Matrix(rows) => {
            assert_eq!(rows[0][0], Expression::integer(2));
            assert_eq!(rows[0][1], Expression::integer(4));
            assert_eq!(rows[1][0], Expression::integer(6));
            assert_eq!(rows[1][1], Expression::integer(8));
        }
        _ => panic!("expected Matrix"),
    }
}

// ── limit ─────────────────────────────────────────────────────────────────────

#[test]
fn test_map_limit() {
    let expr: Expression = ExprKind::Limit {
        expr: Box::new(Expression::integer(1)),
        var: "x".to_string(),
        to: Box::new(Expression::integer(2)),
        direction: Direction::Both,
    }
    .into();
    let result = expr.map(double_integers);
    match &result.kind {
        ExprKind::Limit {
            expr: inner, to, ..
        } => {
            assert_eq!(**inner, Expression::integer(2));
            assert_eq!(**to, Expression::integer(4));
        }
        _ => panic!("expected Limit"),
    }
}

// ── bottom-up ordering ────────────────────────────────────────────────────────

#[test]
fn test_map_is_bottom_up() {
    // Replace Integer(1) with Integer(10), then double all integers.
    // Bottom-up means: first 1→10, then 10→20.
    let expr = Expression::integer(1);
    let result = expr.map(|e| match &e.kind {
        ExprKind::Integer(1) => ExprKind::Integer(10).into(),
        ExprKind::Integer(n) => ExprKind::Integer(n * 2).into(),
        _ => e,
    });
    // The single leaf visits once; 1 → 10 is the only transformation.
    // (There's no parent to double it after.)
    assert_eq!(result, Expression::integer(10));
}

#[test]
fn test_map_bottom_up_nested() {
    // Wrap each Integer in a Unary::Neg, then verify structure
    let expr: Expression = ExprKind::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::integer(1)),
        right: Box::new(Expression::integer(2)),
    }
    .into();
    let visit_order: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let _ = expr.map(|e| {
        let label = match &e.kind {
            ExprKind::Integer(n) => format!("int({})", n),
            ExprKind::Binary { .. } => "binary".to_string(),
            _ => "other".to_string(),
        };
        visit_order.lock().unwrap().push(label);
        e
    });
    let order = visit_order.into_inner().unwrap();
    // leaves come before the binary node
    assert_eq!(&order[0], "int(1)");
    assert_eq!(&order[1], "int(2)");
    assert_eq!(&order[2], "binary");
}
