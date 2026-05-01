//! Integration tests using real-world mathematical expressions.
//!
//! Tests cover physics equations, calculus formulas, and common mathematical notation.
//! Note: Some tests are adjusted to match current parser capabilities.

use mathlex::ast::{BinaryOp, ExprKind, Expression, MathConstant};
use mathlex::parser::parse_latex;

// ============================================================
// Basic Physics and Engineering
// ============================================================

#[test]
fn test_quadratic_formula_simplified() {
    // Simplified version: (-b + √(b² - 4ac)) / (2a)
    let expr = parse_latex(r"\frac{-b + \sqrt{b^2 - 4 * a * c}}{2 * a}").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Div, ..
        } => {
            // Structure is correct - division at top level
        }
        _ => panic!("Expected division at top level, got {:?}", expr),
    }
}

#[test]
fn test_pythagorean_theorem() {
    // c² = a² + b²
    let expr = parse_latex(r"c^2 = a^2 + b^2").unwrap();
    match &expr.kind {
        ExprKind::Equation { left, right, .. } => {
            // Both sides should be valid
            assert!(matches!(
                left.kind,
                ExprKind::Binary {
                    op: BinaryOp::Pow,
                    ..
                }
            ));
            assert!(matches!(
                right.kind,
                ExprKind::Binary {
                    op: BinaryOp::Add,
                    ..
                }
            ));
        }
        _ => panic!("Expected Equation, got {:?}", expr),
    }
}

#[test]
fn test_einsteins_mass_energy() {
    // E = mc²
    let expr = parse_latex(r"E = m * c^2").unwrap();
    match &expr.kind {
        ExprKind::Equation { left, .. } => {
            assert_eq!(**left, Expression::variable("E".to_string()));
        }
        _ => panic!("Expected Equation, got {:?}", expr),
    }
}

// ============================================================
// Calculus Expressions
// ============================================================

#[test]
fn test_definite_integral() {
    // ∫₀^π sin(x) dx
    let expr = parse_latex(r"\int_0^\pi \sin(x) dx").unwrap();
    match &expr.kind {
        ExprKind::Integral { bounds, var, .. } => {
            assert!(bounds.is_some());
            assert_eq!(var, "x");
        }
        _ => panic!("Expected Integral, got {:?}", expr),
    }
}

#[test]
fn test_limit() {
    // lim_{x→0} sin(x)/x
    let expr = parse_latex(r"\lim_{x \to 0} \frac{\sin(x)}{x}").unwrap();
    match &expr.kind {
        ExprKind::Limit { var, to, .. } => {
            assert_eq!(var, "x");
            assert_eq!(**to, Expression::integer(0));
        }
        _ => panic!("Expected Limit, got {:?}", expr),
    }
}

#[test]
fn test_summation() {
    // Σᵢ₌₁ⁿ i
    let expr = parse_latex(r"\sum_{i=1}^{n} i").unwrap();
    match &expr.kind {
        ExprKind::Sum {
            index,
            lower,
            upper,
            ..
        } => {
            assert_eq!(index, "i");
            assert_eq!(**lower, Expression::integer(1));
            assert_eq!(**upper, Expression::variable("n".to_string()));
        }
        _ => panic!("Expected Sum, got {:?}", expr),
    }
}

#[test]
fn test_product_notation() {
    // Πᵢ₌₁ⁿ aᵢ
    let expr = parse_latex(r"\prod_{i=1}^{n} a_i").unwrap();
    match &expr.kind {
        ExprKind::Product { index, .. } => {
            assert_eq!(index, "i");
        }
        _ => panic!("Expected Product, got {:?}", expr),
    }
}

// ============================================================
// Trigonometric and Exponential
// ============================================================

#[test]
fn test_euler_formula() {
    // e^{iπ} + 1
    let expr = parse_latex(r"e^{i * \pi} + 1").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            ..
        } => {
            // e^{iπ} should be exp function
            match &left.kind {
                ExprKind::Function { name, .. } if name == "exp" => {}
                _ => panic!("Expected exp function for e^{{ix}}, got {:?}", left),
            }
        }
        _ => panic!("Expected addition, got {:?}", expr),
    }
}

#[test]
fn test_trig_functions() {
    // sin(x) + cos(x)
    let expr = parse_latex(r"\sin(x) + \cos(x)").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert!(matches!(&left.kind, ExprKind::Function { name, .. } if name == "sin"));
            assert!(matches!(&right.kind, ExprKind::Function { name, .. } if name == "cos"));
        }
        _ => panic!("Expected addition of trig functions, got {:?}", expr),
    }
}

#[test]
fn test_natural_logarithm() {
    // ln(e) = 1
    let expr = parse_latex(r"\ln(e)").unwrap();
    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "ln");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::constant(MathConstant::E));
        }
        _ => panic!("Expected ln function, got {:?}", expr),
    }
}

// ============================================================
// Complex Numbers
// ============================================================

#[test]
fn test_complex_number() {
    // 3 + 4i (using explicit marker)
    let expr = parse_latex(r"3 + 4 * \mathrm{i}").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert_eq!(**left, Expression::integer(3));
            match &right.kind {
                ExprKind::Binary {
                    op: BinaryOp::Mul,
                    left: l,
                    right: r,
                } => {
                    assert_eq!(**l, Expression::integer(4));
                    assert_eq!(**r, Expression::constant(MathConstant::I));
                }
                _ => panic!("Expected multiplication with i, got {:?}", right),
            }
        }
        _ => panic!("Expected addition, got {:?}", expr),
    }
}

#[test]
fn test_imaginary_unit() {
    // i² = -1 (just i²)
    let expr = parse_latex(r"\mathrm{i}^2").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Pow,
            left,
            right,
        } => {
            assert_eq!(**left, Expression::constant(MathConstant::I));
            assert_eq!(**right, Expression::integer(2));
        }
        _ => panic!("Expected power of i, got {:?}", expr),
    }
}

// ============================================================
// Matrices and Linear Algebra
// ============================================================

#[test]
fn test_matrix_basic() {
    let expr = parse_latex(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}").unwrap();
    match &expr.kind {
        ExprKind::Matrix(rows) => {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].len(), 2);
        }
        _ => panic!("Expected Matrix, got {:?}", expr),
    }
}

#[test]
fn test_determinant_function() {
    // det is currently parsed as a function
    let expr = parse_latex(r"\det(A)").unwrap();
    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "det");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::variable("A".to_string()));
        }
        _ => panic!("Expected Function(det), got {:?}", expr),
    }
}

// ============================================================
// Vector Operations
// ============================================================

#[test]
fn test_vector_cross_product_in_context() {
    // τ = r × F (torque)
    let expr = parse_latex(r"\mathbf{r} \times \mathbf{F}").unwrap();
    match &expr.kind {
        ExprKind::CrossProduct { .. } => {
            // Correct structure
        }
        _ => panic!("Expected CrossProduct, got {:?}", expr),
    }
}

// ============================================================
// Multiple Integrals
// ============================================================

#[test]
fn test_double_integral_area() {
    // ∬_R f dA
    let expr = parse_latex(r"\iint_R f dA").unwrap();
    match &expr.kind {
        ExprKind::MultipleIntegral { dimension, .. } => {
            assert_eq!(*dimension, 2);
        }
        _ => panic!("Expected MultipleIntegral with dimension 2, got {:?}", expr),
    }
}

#[test]
fn test_triple_integral_volume() {
    // ∭_V f dV
    let expr = parse_latex(r"\iiint_V f dV").unwrap();
    match &expr.kind {
        ExprKind::MultipleIntegral { dimension, .. } => {
            assert_eq!(*dimension, 3);
        }
        _ => panic!("Expected MultipleIntegral with dimension 3, got {:?}", expr),
    }
}

#[test]
fn test_line_integral() {
    // ∮_C F dr
    let expr = parse_latex(r"\oint_C F dr").unwrap();
    match &expr.kind {
        ExprKind::ClosedIntegral {
            dimension, surface, ..
        } => {
            assert_eq!(*dimension, 1);
            assert_eq!(*surface, Some("C".to_string()));
        }
        _ => panic!("Expected ClosedIntegral, got {:?}", expr),
    }
}

// ============================================================
// Square Roots and Powers
// ============================================================

#[test]
fn test_square_root() {
    // sqrt is parsed as a Function
    let expr = parse_latex(r"\sqrt{2}").unwrap();
    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "sqrt");
            assert_eq!(args.len(), 1);
            assert_eq!(args[0], Expression::integer(2));
        }
        _ => panic!("Expected Function(sqrt), got {:?}", expr),
    }
}

#[test]
fn test_nth_root() {
    // nth roots are parsed as root function with index and radicand
    let expr = parse_latex(r"\sqrt[3]{8}").unwrap();
    match &expr.kind {
        ExprKind::Function { name, args } => {
            assert_eq!(name, "root");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Expression::integer(8)); // radicand
            assert_eq!(args[1], Expression::integer(3)); // index
        }
        _ => panic!("Expected Function(root), got {:?}", expr),
    }
}

// ============================================================
// Fractions
// ============================================================

#[test]
fn test_fraction() {
    let expr = parse_latex(r"\frac{a}{b}").unwrap();
    match &expr.kind {
        ExprKind::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            assert_eq!(**left, Expression::variable("a".to_string()));
            assert_eq!(**right, Expression::variable("b".to_string()));
        }
        _ => panic!("Expected division, got {:?}", expr),
    }
}
