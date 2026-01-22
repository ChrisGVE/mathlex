//! Tests for precedence correctness in serialization.
//!
//! Verifies that Display and ToLatex produce output that parses back
//! with the correct precedence and associativity.

use mathlex::{parse, parse_latex, Expression, BinaryOp, UnaryOp, ToLatex};

// Helper to create a variable expression
fn var(name: &str) -> Expression {
    Expression::Variable(name.to_string())
}

// Helper to create an integer expression
fn int(n: i64) -> Expression {
    Expression::Integer(n)
}

// =============================================================================
// Plain Text Precedence Tests
// =============================================================================

#[test]
fn test_unary_neg_of_sum() {
    // -(a + b) should serialize with parens and round-trip correctly
    let ast = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
    };
    let s = ast.to_string();
    assert!(s.contains("("), "Should have parens: {}", s);
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "-(a+b) should round-trip");
}

#[test]
fn test_unary_neg_of_product() {
    // -(a * b) should serialize with parens
    let ast = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
    };
    let s = ast.to_string();
    assert!(s.contains("("), "Should have parens: {}", s);
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "-(a*b) should round-trip");
}

#[test]
fn test_power_left_associativity_needs_parens() {
    // (a^b)^c should serialize with parens (power is right-associative)
    let ast = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s = ast.to_string();
    assert!(s.contains("("), "Should have parens for (a^b)^c: {}", s);
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "(a^b)^c should round-trip");
}

#[test]
fn test_power_right_associativity_no_parens() {
    // a^(b^c) is natural right-associativity, no parens needed
    let ast = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(var("a")),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(var("b")),
            right: Box::new(var("c")),
        }),
    };
    let s = ast.to_string();
    // May or may not have parens, but should round-trip
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "a^(b^c) should round-trip");
}

#[test]
fn test_sum_in_product_needs_parens() {
    // (a + b) * c needs parens around the sum
    let ast = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s = ast.to_string();
    assert!(s.contains("("), "Should have parens: {}", s);
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "(a+b)*c should round-trip");
}

#[test]
fn test_product_in_sum_no_parens() {
    // a * b + c doesn't need parens (mul has higher precedence)
    let ast = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s = ast.to_string();
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "a*b+c should round-trip");
}

#[test]
fn test_subtraction_associativity() {
    // (a - b) - c vs a - (b - c)
    // Left: (a - b) - c = a - b - c (left associative, no parens needed)
    let left_assoc = Expression::Binary {
        op: BinaryOp::Sub,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s1 = left_assoc.to_string();
    let parsed1 = parse(&s1).unwrap();
    assert_eq!(left_assoc, parsed1, "(a-b)-c should round-trip");

    // Right: a - (b - c) needs explicit parens
    let right_assoc = Expression::Binary {
        op: BinaryOp::Sub,
        left: Box::new(var("a")),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Sub,
            left: Box::new(var("b")),
            right: Box::new(var("c")),
        }),
    };
    let s2 = right_assoc.to_string();
    assert!(s2.contains("("), "a-(b-c) should have parens: {}", s2);
    let parsed2 = parse(&s2).unwrap();
    assert_eq!(right_assoc, parsed2, "a-(b-c) should round-trip");
}

#[test]
fn test_division_associativity() {
    // Similar to subtraction - division is left associative
    // a / (b / c) needs explicit parens
    let ast = Expression::Binary {
        op: BinaryOp::Div,
        left: Box::new(var("a")),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(var("b")),
            right: Box::new(var("c")),
        }),
    };
    let s = ast.to_string();
    assert!(s.contains("("), "a/(b/c) should have parens: {}", s);
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "a/(b/c) should round-trip");
}

#[test]
fn test_nested_unary() {
    // --a should round-trip
    let ast = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(var("a")),
        }),
    };
    let s = ast.to_string();
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "--a should round-trip");
}

#[test]
fn test_complex_precedence_chain() {
    // a + b * c^d should parse correctly without parens
    let ast = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(var("a")),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Mul,
            left: Box::new(var("b")),
            right: Box::new(Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(var("c")),
                right: Box::new(var("d")),
            }),
        }),
    };
    let s = ast.to_string();
    let parsed = parse(&s).unwrap();
    assert_eq!(ast, parsed, "a + b * c^d should round-trip");
}

// =============================================================================
// LaTeX Precedence Tests
// =============================================================================

#[test]
fn test_latex_unary_neg_of_sum() {
    let ast = Expression::Unary {
        op: UnaryOp::Neg,
        operand: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
    };
    let s = ast.to_latex();
    assert!(s.contains("(") || s.contains("\\left"), "Should have parens: {}", s);
    let parsed = parse_latex(&s).unwrap();
    assert_eq!(ast, parsed, "LaTeX -(a+b) should round-trip");
}

#[test]
fn test_latex_fraction_in_sum() {
    // a + (1/2) where 1/2 is a fraction
    let ast = Expression::Binary {
        op: BinaryOp::Add,
        left: Box::new(var("a")),
        right: Box::new(Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(int(1)),
            right: Box::new(int(2)),
        }),
    };
    let s = ast.to_latex();
    let parsed = parse_latex(&s).unwrap();
    assert_eq!(ast, parsed, "LaTeX a + frac should round-trip");
}

#[test]
fn test_latex_power_precedence() {
    // (a^b)^c needs parens
    let ast = Expression::Binary {
        op: BinaryOp::Pow,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Pow,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s = ast.to_latex();
    let parsed = parse_latex(&s).unwrap();
    assert_eq!(ast, parsed, "LaTeX (a^b)^c should round-trip");
}

#[test]
fn test_latex_mul_precedence() {
    // (a + b) \cdot c needs parens
    let ast = Expression::Binary {
        op: BinaryOp::Mul,
        left: Box::new(Expression::Binary {
            op: BinaryOp::Add,
            left: Box::new(var("a")),
            right: Box::new(var("b")),
        }),
        right: Box::new(var("c")),
    };
    let s = ast.to_latex();
    let parsed = parse_latex(&s).unwrap();
    assert_eq!(ast, parsed, "LaTeX (a+b)*c should round-trip");
}
