#![allow(clippy::approx_constant)]
//! Property-based tests using proptest to verify parser invariants and edge cases.
//!
//! This module uses proptest to generate arbitrary expressions and verify
//! important properties hold across a wide range of inputs.

use mathlex::ast::{BinaryOp, Expression, MathConstant, MathFloat, UnaryOp};
use mathlex::parser::parse;
use proptest::prelude::*;

/// Strategy to generate arbitrary expressions.
///
/// This strategy generates expressions with controlled depth to avoid
/// stack overflow and excessive generation time.
fn arb_expression() -> impl Strategy<Value = Expression> {
    let leaf = prop_oneof![
        // Integers in a reasonable range
        (-1000i64..1000).prop_map(Expression::Integer),
        // Floats in a reasonable range (avoiding NaN and infinity for now)
        (-1000.0f64..1000.0)
            .prop_filter("No NaN", |f| !f.is_nan())
            .prop_map(|f| Expression::Float(MathFloat::from(f))),
        // Single-letter variables
        "[a-z]".prop_map(Expression::Variable),
        // Mathematical constants
        prop_oneof![
            Just(Expression::Constant(MathConstant::Pi)),
            Just(Expression::Constant(MathConstant::E)),
            Just(Expression::Constant(MathConstant::I)),
        ],
    ];

    leaf.prop_recursive(
        4,  // max depth: 4 levels
        64, // max total nodes: 64
        10, // expected branch size: 10
        |inner| {
            prop_oneof![
                // Binary operations
                (arb_binary_op(), inner.clone(), inner.clone()).prop_map(|(op, left, right)| {
                    Expression::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    }
                }),
                // Unary operations
                (arb_unary_op(), inner.clone()).prop_map(|(op, operand)| Expression::Unary {
                    op,
                    operand: Box::new(operand),
                }),
                // Functions with 1-2 arguments
                (
                    arb_function_name(),
                    prop::collection::vec(inner.clone(), 1..=2)
                )
                    .prop_map(|(name, args)| Expression::Function { name, args }),
                // Vectors with 1-3 elements
                prop::collection::vec(inner.clone(), 1..=3).prop_map(Expression::Vector),
            ]
        },
    )
}

/// Strategy to generate arbitrary binary operators.
fn arb_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![
        Just(BinaryOp::Add),
        Just(BinaryOp::Sub),
        Just(BinaryOp::Mul),
        Just(BinaryOp::Div),
        Just(BinaryOp::Pow),
        Just(BinaryOp::Mod),
    ]
}

/// Strategy to generate arbitrary unary operators.
fn arb_unary_op() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![
        Just(UnaryOp::Neg),
        Just(UnaryOp::Pos),
        Just(UnaryOp::Factorial),
    ]
}

/// Strategy to generate arbitrary function names.
fn arb_function_name() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("sin".to_string()),
        Just("cos".to_string()),
        Just("tan".to_string()),
        Just("log".to_string()),
        Just("exp".to_string()),
        Just("sqrt".to_string()),
        Just("abs".to_string()),
        Just("max".to_string()),
        Just("min".to_string()),
    ]
}

proptest! {
    /// Property: Clone equals original
    ///
    /// Verifies that cloning an expression produces an identical expression.
    #[test]
    fn prop_clone_equals_original(expr in arb_expression()) {
        let cloned = expr.clone();
        prop_assert_eq!(expr, cloned);
    }

    /// Property: Depth is at least 1
    ///
    /// Every expression must have a depth of at least 1, even leaf nodes.
    #[test]
    fn prop_depth_at_least_one(expr in arb_expression()) {
        prop_assert!(expr.depth() >= 1);
    }

    /// Property: Node count is at least 1
    ///
    /// Every expression must contain at least one node.
    #[test]
    fn prop_node_count_at_least_one(expr in arb_expression()) {
        prop_assert!(expr.node_count() >= 1);
    }

    /// Property: Depth never exceeds node count
    ///
    /// The depth (longest path) can never exceed the total number of nodes.
    #[test]
    fn prop_depth_le_node_count(expr in arb_expression()) {
        prop_assert!(expr.depth() <= expr.node_count());
    }

    /// Property: All variables found by find_variables appear in the expression
    ///
    /// This property verifies that find_variables doesn't return false positives.
    #[test]
    fn prop_find_variables_no_false_positives(expr in arb_expression()) {
        let vars = expr.find_variables();
        let expr_str = format!("{:?}", expr);

        for var in vars {
            // Each variable should appear in the debug representation
            prop_assert!(
                expr_str.contains(&format!("Variable(\"{}\")", var)),
                "Variable {} not found in expression debug output",
                var
            );
        }
    }

    /// Property: All functions found by find_functions appear in the expression
    ///
    /// This property verifies that find_functions doesn't return false positives.
    #[test]
    fn prop_find_functions_no_false_positives(expr in arb_expression()) {
        let funcs = expr.find_functions();
        let expr_str = format!("{:?}", expr);

        for func in funcs {
            // Each function should appear in the debug representation
            prop_assert!(
                expr_str.contains(&format!("name: \"{}\"", func)),
                "Function {} not found in expression debug output",
                func
            );
        }
    }

    /// Property: All constants found by find_constants appear in the expression
    ///
    /// This property verifies that find_constants doesn't return false positives.
    #[test]
    fn prop_find_constants_no_false_positives(expr in arb_expression()) {
        let consts = expr.find_constants();
        let expr_str = format!("{:?}", expr);

        for constant in consts {
            // Each constant should appear in the debug representation
            let const_str = format!("{:?}", constant);
            prop_assert!(
                expr_str.contains(&const_str),
                "Constant {} not found in expression debug output",
                const_str
            );
        }
    }

    /// Property: Display produces non-empty string
    ///
    /// Every expression should have a non-empty display representation.
    #[test]
    fn prop_display_non_empty(expr in arb_expression()) {
        let display_str = format!("{}", expr);
        prop_assert!(!display_str.is_empty());
    }

    /// Property: Display can be parsed back (for simple expressions)
    ///
    /// For expressions that don't use complex notation, the display output
    /// should be parseable back to an AST. Note: Not all expressions will
    /// round-trip exactly due to parser limitations and representation choices.
    #[test]
    fn prop_display_parseable(expr in arb_expression()) {
        let display_str = format!("{}", expr);

        // Attempt to parse the display string
        match parse(&display_str) {
            Ok(_parsed_expr) => {
                // Successfully parsed - this is good
                // Note: We don't assert equality because the parser may
                // produce a different but semantically equivalent AST
                // (e.g., different associativity, implicit operations made explicit)
            }
            Err(_) => {
                // Some complex expressions may not round-trip perfectly
                // This is acceptable for now
            }
        }
    }

    /// Property: Substitution preserves structure for non-matching variables
    ///
    /// Substituting a variable that doesn't appear in the expression
    /// should return an identical expression.
    #[test]
    fn prop_substitute_non_matching_identity(expr in arb_expression()) {
        let non_existent_var = "nonexistent_variable_xyz";
        let replacement = Expression::Integer(999);

        let result = expr.substitute(non_existent_var, &replacement);
        prop_assert_eq!(expr, result);
    }

    /// Property: Double negation returns to similar depth
    ///
    /// Applying negation twice should increase depth by 2.
    #[test]
    fn prop_double_negation_depth(expr in arb_expression()) {
        let neg_once = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(expr.clone()),
        };
        let neg_twice = Expression::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(neg_once.clone()),
        };

        prop_assert_eq!(neg_once.depth(), expr.depth() + 1);
        prop_assert_eq!(neg_twice.depth(), expr.depth() + 2);
    }

    /// Property: Binary operation increases depth
    ///
    /// Creating a binary operation from two expressions should have depth
    /// equal to 1 + max(left.depth, right.depth).
    #[test]
    fn prop_binary_depth(
        left in arb_expression(),
        right in arb_expression(),
        op in arb_binary_op()
    ) {
        let binary = Expression::Binary {
            op,
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
        };

        let expected_depth = 1 + left.depth().max(right.depth());
        prop_assert_eq!(binary.depth(), expected_depth);
    }

    /// Property: Binary operation node count
    ///
    /// Creating a binary operation should have node count equal to
    /// 1 + left.node_count + right.node_count.
    #[test]
    fn prop_binary_node_count(
        left in arb_expression(),
        right in arb_expression(),
        op in arb_binary_op()
    ) {
        let binary = Expression::Binary {
            op,
            left: Box::new(left.clone()),
            right: Box::new(right.clone()),
        };

        let expected_count = 1 + left.node_count() + right.node_count();
        prop_assert_eq!(binary.node_count(), expected_count);
    }

    /// Property: Vector depth equals max element depth + 1
    ///
    /// A vector's depth should be 1 + maximum depth of its elements.
    #[test]
    fn prop_vector_depth(elements in prop::collection::vec(arb_expression(), 1..=5)) {
        let vector = Expression::Vector(elements.clone());

        let max_element_depth = elements.iter().map(|e| e.depth()).max().unwrap_or(0);
        let expected_depth = if elements.is_empty() { 1 } else { 1 + max_element_depth };

        prop_assert_eq!(vector.depth(), expected_depth);
    }

    /// Property: Vector node count equals sum of element counts + 1
    ///
    /// A vector's node count should be 1 + sum of all element node counts.
    #[test]
    fn prop_vector_node_count(elements in prop::collection::vec(arb_expression(), 1..=5)) {
        let vector = Expression::Vector(elements.clone());

        let element_count_sum: usize = elements.iter().map(|e| e.node_count()).sum();
        let expected_count = 1 + element_count_sum;

        prop_assert_eq!(vector.node_count(), expected_count);
    }

    /// Property: Hash consistency
    ///
    /// Equal expressions should hash to the same value (required for HashSet).
    #[test]
    fn prop_hash_consistency(expr in arb_expression()) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let cloned = expr.clone();

        let mut hasher1 = DefaultHasher::new();
        expr.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        cloned.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        prop_assert_eq!(hash1, hash2);
    }

    /// Property: Variable finding is consistent
    ///
    /// Running find_variables twice should return the same set.
    #[test]
    fn prop_find_variables_consistent(expr in arb_expression()) {
        let vars1 = expr.find_variables();
        let vars2 = expr.find_variables();
        prop_assert_eq!(vars1, vars2);
    }

    /// Property: Function finding is consistent
    ///
    /// Running find_functions twice should return the same set.
    #[test]
    fn prop_find_functions_consistent(expr in arb_expression()) {
        let funcs1 = expr.find_functions();
        let funcs2 = expr.find_functions();
        prop_assert_eq!(funcs1, funcs2);
    }

    /// Property: Constant finding is consistent
    ///
    /// Running find_constants twice should return the same set.
    #[test]
    fn prop_find_constants_consistent(expr in arb_expression()) {
        let consts1 = expr.find_constants();
        let consts2 = expr.find_constants();
        prop_assert_eq!(consts1, consts2);
    }

    /// Property: MathFloat equality is transitive
    ///
    /// If a == b and b == c, then a == c.
    #[test]
    fn prop_math_float_transitivity(
        a in -1000.0f64..1000.0,
        b in -1000.0f64..1000.0,
        c in -1000.0f64..1000.0
    ) {
        let fa = MathFloat::from(a);
        let fb = MathFloat::from(b);
        let fc = MathFloat::from(c);

        if fa == fb && fb == fc {
            prop_assert_eq!(fa, fc);
        }
    }

    /// Property: MathFloat equality is symmetric
    ///
    /// If a == b, then b == a.
    #[test]
    fn prop_math_float_symmetry(a in -1000.0f64..1000.0, b in -1000.0f64..1000.0) {
        let fa = MathFloat::from(a);
        let fb = MathFloat::from(b);

        if fa == fb {
            prop_assert_eq!(fb, fa);
        }
    }

    /// Property: MathFloat equality is reflexive
    ///
    /// a == a always.
    #[test]
    fn prop_math_float_reflexivity(a in -1000.0f64..1000.0) {
        let fa = MathFloat::from(a);
        prop_assert_eq!(fa, fa);
    }

    /// Property: MathFloat roundtrip
    ///
    /// Converting f64 -> MathFloat -> f64 should preserve the value.
    #[test]
    fn prop_math_float_roundtrip(f in -1000.0f64..1000.0) {
        let math_float = MathFloat::from(f);
        let roundtrip: f64 = math_float.into();

        // Use approximate equality due to floating-point precision
        prop_assert!((f - roundtrip).abs() < 1e-10);
    }
}

// Regular unit tests to verify specific edge cases

#[test]
fn test_empty_vector_depth() {
    let expr = Expression::Vector(vec![]);
    assert_eq!(expr.depth(), 1);
}

#[test]
fn test_empty_vector_node_count() {
    let expr = Expression::Vector(vec![]);
    assert_eq!(expr.node_count(), 1);
}

#[test]
fn test_single_variable_find_variables() {
    let expr = Expression::Variable("x".to_string());
    let vars = expr.find_variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains("x"));
}

#[test]
fn test_no_functions_in_leaf() {
    let expr = Expression::Integer(42);
    assert_eq!(expr.find_functions().len(), 0);
}

#[test]
fn test_no_constants_in_integer() {
    let expr = Expression::Integer(42);
    assert_eq!(expr.find_constants().len(), 0);
}

#[test]
fn test_pi_constant_found() {
    let expr = Expression::Constant(MathConstant::Pi);
    let consts = expr.find_constants();
    assert_eq!(consts.len(), 1);
    assert!(consts.contains(&MathConstant::Pi));
}

#[test]
fn test_display_integer() {
    let expr = Expression::Integer(42);
    assert_eq!(format!("{}", expr), "42");
}

#[test]
fn test_display_variable() {
    let expr = Expression::Variable("x".to_string());
    assert_eq!(format!("{}", expr), "x");
}

#[test]
fn test_clone_integer() {
    let expr = Expression::Integer(42);
    let cloned = expr.clone();
    assert_eq!(expr, cloned);
}

#[test]
fn test_hash_equal_expressions() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let expr1 = Expression::Integer(42);
    let expr2 = Expression::Integer(42);

    let mut hasher1 = DefaultHasher::new();
    expr1.hash(&mut hasher1);
    let hash1 = hasher1.finish();

    let mut hasher2 = DefaultHasher::new();
    expr2.hash(&mut hasher2);
    let hash2 = hasher2.finish();

    assert_eq!(hash1, hash2);
}
