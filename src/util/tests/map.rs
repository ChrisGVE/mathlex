use crate::ast::{BinaryOp, Direction, Expression, IntegralBounds, MathConstant, UnaryOp};

fn double_integers(e: Expression) -> Expression {
    match e {
        Expression::Integer(n) => Expression::Integer(n * 2),
        other => other,
    }
}

fn rename_x_to_y(e: Expression) -> Expression {
    match e {
        Expression::Variable(ref name) if name == "x" => Expression::Variable("y".to_string()),
        other => other,
    }
}

// ── identity ─────────────────────────────────────────────────────────────────

#[test]
fn test_map_identity_leaf() {
    let expr = Expression::Integer(42);
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_variable() {
    let expr = Expression::Variable("x".to_string());
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_constant() {
    let expr = Expression::Constant(MathConstant::Pi);
    assert_eq!(expr.map(|e| e), expr);
}

#[test]
fn test_map_identity_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };
    assert_eq!(expr.map(|e| e), expr);
}

// ── integer transform ─────────────────────────────────────────────────────────

#[test]
fn test_map_double_integers_leaf() {
    let expr = Expression::Integer(3);
    assert_eq!(expr.map(double_integers), Expression::Integer(6));
}

#[test]
fn test_map_double_integers_binary() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(2)),
        right: Box::new(Expression::Integer(3)),
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Binary { left, right, .. } => {
            assert_eq!(*left, Expression::Integer(4));
            assert_eq!(*right, Expression::Integer(6));
        }
        _ => panic!("expected Binary"),
    }
}

#[test]
fn test_map_double_integers_nested() {
    // (1 + 2) * 3  →  (2 + 4) * 6
    let expr = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expression::Integer(1)),
            right: Box::new(Expression::Integer(2)),
        }),
        right: Box::new(Expression::Integer(3)),
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } => {
            assert_eq!(*right, Expression::Integer(6));
            match *left {
                Expression::Binary {
                    op: BinaryOp::Add,
                    left: ll,
                    right: lr,
                } => {
                    assert_eq!(*ll, Expression::Integer(2));
                    assert_eq!(*lr, Expression::Integer(4));
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
    let expr = Expression::Variable("x".to_string());
    assert_eq!(
        expr.map(rename_x_to_y),
        Expression::Variable("y".to_string())
    );
}

#[test]
fn test_map_rename_leaves_other_variables() {
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Variable("x".to_string())),
        right: Box::new(Expression::Variable("z".to_string())),
    };
    let result = expr.map(rename_x_to_y);
    match result {
        Expression::Binary { left, right, .. } => {
            assert_eq!(*left, Expression::Variable("y".to_string()));
            assert_eq!(*right, Expression::Variable("z".to_string()));
        }
        _ => panic!("expected Binary"),
    }
}

// ── unary ─────────────────────────────────────────────────────────────────────

#[test]
fn test_map_unary() {
    let expr = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Integer(5)),
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Unary { operand, .. } => {
            assert_eq!(*operand, Expression::Integer(10));
        }
        _ => panic!("expected Unary"),
    }
}

// ── function ──────────────────────────────────────────────────────────────────

#[test]
fn test_map_function_args() {
    let expr = Expression::Function {
        name: "f".to_string(),
        args: vec![Expression::Integer(1), Expression::Integer(2)],
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Function { name, args } => {
            assert_eq!(name, "f");
            assert_eq!(args[0], Expression::Integer(2));
            assert_eq!(args[1], Expression::Integer(4));
        }
        _ => panic!("expected Function"),
    }
}

// ── derivative ────────────────────────────────────────────────────────────────

#[test]
fn test_map_derivative_transforms_body() {
    let expr = Expression::Derivative {
        expr: Box::new(Expression::Integer(7)),
        var: "x".to_string(),
        order: 1,
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Derivative {
            expr: inner,
            var,
            order,
        } => {
            assert_eq!(*inner, Expression::Integer(14));
            assert_eq!(var, "x");
            assert_eq!(order, 1);
        }
        _ => panic!("expected Derivative"),
    }
}

// ── integral ──────────────────────────────────────────────────────────────────

#[test]
fn test_map_integral_transforms_integrand_and_bounds() {
    let expr = Expression::Integral {
        integrand: Box::new(Expression::Integer(3)),
        var: "x".to_string(),
        bounds: Some(IntegralBounds {
            lower: Box::new(Expression::Integer(1)),
            upper: Box::new(Expression::Integer(2)),
        }),
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Integral {
            integrand, bounds, ..
        } => {
            assert_eq!(*integrand, Expression::Integer(6));
            let b = bounds.unwrap();
            assert_eq!(*b.lower, Expression::Integer(2));
            assert_eq!(*b.upper, Expression::Integer(4));
        }
        _ => panic!("expected Integral"),
    }
}

// ── sum / product ─────────────────────────────────────────────────────────────

#[test]
fn test_map_sum_transforms_bounds_and_body() {
    let expr = Expression::Sum {
        index: "i".to_string(),
        lower: Box::new(Expression::Integer(1)),
        upper: Box::new(Expression::Integer(5)),
        body: Box::new(Expression::Integer(3)),
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Sum {
            lower, upper, body, ..
        } => {
            assert_eq!(*lower, Expression::Integer(2));
            assert_eq!(*upper, Expression::Integer(10));
            assert_eq!(*body, Expression::Integer(6));
        }
        _ => panic!("expected Sum"),
    }
}

// ── vector / matrix ───────────────────────────────────────────────────────────

#[test]
fn test_map_vector() {
    let expr = Expression::Vector(vec![
        Expression::Integer(1),
        Expression::Integer(2),
        Expression::Integer(3),
    ]);
    let result = expr.map(double_integers);
    match result {
        Expression::Vector(elems) => {
            assert_eq!(elems[0], Expression::Integer(2));
            assert_eq!(elems[1], Expression::Integer(4));
            assert_eq!(elems[2], Expression::Integer(6));
        }
        _ => panic!("expected Vector"),
    }
}

#[test]
fn test_map_matrix() {
    let expr = Expression::Matrix(vec![
        vec![Expression::Integer(1), Expression::Integer(2)],
        vec![Expression::Integer(3), Expression::Integer(4)],
    ]);
    let result = expr.map(double_integers);
    match result {
        Expression::Matrix(rows) => {
            assert_eq!(rows[0][0], Expression::Integer(2));
            assert_eq!(rows[0][1], Expression::Integer(4));
            assert_eq!(rows[1][0], Expression::Integer(6));
            assert_eq!(rows[1][1], Expression::Integer(8));
        }
        _ => panic!("expected Matrix"),
    }
}

// ── limit ─────────────────────────────────────────────────────────────────────

#[test]
fn test_map_limit() {
    let expr = Expression::Limit {
        expr: Box::new(Expression::Integer(1)),
        var: "x".to_string(),
        to: Box::new(Expression::Integer(2)),
        direction: Direction::Both,
    };
    let result = expr.map(double_integers);
    match result {
        Expression::Limit {
            expr: inner, to, ..
        } => {
            assert_eq!(*inner, Expression::Integer(2));
            assert_eq!(*to, Expression::Integer(4));
        }
        _ => panic!("expected Limit"),
    }
}

// ── bottom-up ordering ────────────────────────────────────────────────────────

#[test]
fn test_map_is_bottom_up() {
    // Replace Integer(1) with Integer(10), then double all integers.
    // Bottom-up means: first 1→10, then 10→20.
    let expr = Expression::Integer(1);
    let result = expr.map(|e| match e {
        Expression::Integer(1) => Expression::Integer(10),
        Expression::Integer(n) => Expression::Integer(n * 2),
        other => other,
    });
    // The single leaf visits once; 1 → 10 is the only transformation.
    // (There's no parent to double it after.)
    assert_eq!(result, Expression::Integer(10));
}

#[test]
fn test_map_bottom_up_nested() {
    // Wrap each Integer in a Unary::Neg, then verify structure
    let expr = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Integer(1)),
        right: Box::new(Expression::Integer(2)),
    };
    let visit_order: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let _ = expr.map(|e| {
        let label = match &e {
            Expression::Integer(n) => format!("int({})", n),
            Expression::Binary { .. } => "binary".to_string(),
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
