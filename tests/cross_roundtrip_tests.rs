//! Cross-parser round-trip tests for mathlex.
//!
//! These tests verify that expressions can be converted between plain text and LaTeX
//! representations without loss of semantic information.
//!
//! # Test Paths
//!
//! 1. **text → AST → to_latex() → parse_latex() → AST₂**: Verify text expressions
//!    survive conversion to LaTeX and back.
//! 2. **LaTeX → AST → Display → parse() → AST₂**: Verify LaTeX expressions
//!    survive conversion to plain text and back.
//!
//! # Known Limitations
//!
//! Some expressions cannot cross round-trip due to representation differences:
//! - `i` in text is `Constant(I)`; in LaTeX `i` is also `Constant(I)` but `\mathrm{i}` is explicit
//! - Derivative `y'` has empty var in text; LaTeX requires explicit var
//! - Some LaTeX-only constructs (integrals, limits, sums) have no text equivalent yet

use mathlex::latex::ToLatex;
use mathlex::{parse, parse_latex};

/// Tests text → AST → to_latex() → parse_latex() → AST₂ equivalence.
fn assert_cross_text_to_latex(input: &str) {
    let expr1 = parse(input).unwrap_or_else(|e| {
        panic!("Failed to parse text: {}\nInput: {}", e, input);
    });
    let latex = expr1.to_latex();
    let expr2 = parse_latex(&latex).unwrap_or_else(|e| {
        panic!(
            "Failed to parse generated LaTeX: {}\nOriginal text: {}\nGenerated LaTeX: {}",
            e, input, latex
        );
    });
    assert_eq!(
        expr1, expr2,
        "ASTs differ after text→LaTeX cross round-trip\nText input: {}\nGenerated LaTeX: {}\nAST1: {:?}\nAST2: {:?}",
        input, latex, expr1, expr2
    );
}

/// Tests LaTeX → AST → Display → parse() → AST₂ equivalence.
fn assert_cross_latex_to_text(input: &str) {
    let expr1 = parse_latex(input).unwrap_or_else(|e| {
        panic!("Failed to parse LaTeX: {}\nInput: {}", e, input);
    });
    let text = expr1.to_string();
    let expr2 = parse(&text).unwrap_or_else(|e| {
        panic!(
            "Failed to parse generated text: {}\nOriginal LaTeX: {}\nGenerated text: {}",
            e, input, text
        );
    });
    assert_eq!(
        expr1, expr2,
        "ASTs differ after LaTeX→text cross round-trip\nLaTeX input: {}\nGenerated text: {}\nAST1: {:?}\nAST2: {:?}",
        input, text, expr1, expr2
    );
}

// ============================================================================
// Text → LaTeX Cross Round-trip Tests
// ============================================================================

mod text_to_latex {
    use super::*;

    // --- Literals ---

    #[test]
    fn integers() {
        assert_cross_text_to_latex("42");
        assert_cross_text_to_latex("0");
        assert_cross_text_to_latex("-17");
    }

    #[test]
    fn floats() {
        assert_cross_text_to_latex("3.14");
        assert_cross_text_to_latex("-2.5");
    }

    // --- Variables and Constants ---

    #[test]
    fn variables() {
        assert_cross_text_to_latex("x");
        assert_cross_text_to_latex("y");
        assert_cross_text_to_latex("z");
    }

    #[test]
    fn constants() {
        assert_cross_text_to_latex("pi");
        assert_cross_text_to_latex("inf");
    }

    // --- Arithmetic ---

    #[test]
    fn binary_arithmetic() {
        assert_cross_text_to_latex("1 + 2");
        assert_cross_text_to_latex("5 - 3");
        assert_cross_text_to_latex("2 * 3");
        assert_cross_text_to_latex("10 / 5");
        assert_cross_text_to_latex("2 ^ 3");
    }

    #[test]
    fn precedence_and_associativity() {
        assert_cross_text_to_latex("1 + 2 * 3");
        assert_cross_text_to_latex("(1 + 2) * 3");
        assert_cross_text_to_latex("2 ^ (3 ^ 4)");
        assert_cross_text_to_latex("5 - 3 - 1");
    }

    #[test]
    fn unary_negation_and_pos() {
        assert_cross_text_to_latex("-5");
        assert_cross_text_to_latex("+3");
    }

    // Note: n! (factorial) does not cross round-trip because LaTeX parser
    // does not support '!' as postfix factorial operator. This is a LaTeX
    // parser gap, not a serialization issue.

    // --- Functions ---

    #[test]
    fn trig_functions() {
        assert_cross_text_to_latex("sin(x)");
        assert_cross_text_to_latex("cos(x)");
        assert_cross_text_to_latex("tan(x)");
    }

    #[test]
    fn other_functions() {
        assert_cross_text_to_latex("exp(x)");
        assert_cross_text_to_latex("ln(x)");
        assert_cross_text_to_latex("sqrt(x)");
        assert_cross_text_to_latex("abs(x)");
    }

    #[test]
    fn nested_functions() {
        assert_cross_text_to_latex("sin(cos(x))");
        assert_cross_text_to_latex("exp(sin(x) + 1)");
    }

    // --- Equations and Inequalities ---

    #[test]
    fn equations() {
        assert_cross_text_to_latex("x = 5");
        assert_cross_text_to_latex("y = 2 * x + 1");
    }

    #[test]
    fn inequalities() {
        assert_cross_text_to_latex("x < 5");
        assert_cross_text_to_latex("y <= 10");
        assert_cross_text_to_latex("z > 0");
        assert_cross_text_to_latex("w >= -5");
        assert_cross_text_to_latex("a != b");
    }

    // --- Derivatives (new in v0.3.2+) ---

    #[test]
    fn diff_functional() {
        assert_cross_text_to_latex("diff(y, x)");
        assert_cross_text_to_latex("diff(y, x, 2)");
    }

    #[test]
    fn leibniz_derivatives() {
        assert_cross_text_to_latex("dy/dx");
        assert_cross_text_to_latex("df/dt");
    }

    // --- Vector Calculus ---

    #[test]
    fn gradient() {
        assert_cross_text_to_latex("grad(f)");
    }

    // --- Complex Expressions ---

    #[test]
    fn complex_compositions() {
        // Note: expressions with 'e' as base don't cross round-trip because
        // text parser treats 'e' as Constant(E), while LaTeX normalizes e^x
        // to exp(x). Use non-e variables for cross round-trip testing.
        assert_cross_text_to_latex("(a + b) * (c - d) / g ^ f");
        assert_cross_text_to_latex("sin(x) + cos(y) * tan(z)");
        assert_cross_text_to_latex("sqrt(x ^ 2 + y ^ 2)");
    }
}

// ============================================================================
// LaTeX → Text Cross Round-trip Tests
// ============================================================================

mod latex_to_text {
    use super::*;

    // --- Literals ---

    #[test]
    fn integers() {
        assert_cross_latex_to_text("42");
        assert_cross_latex_to_text("0");
        assert_cross_latex_to_text("-17");
    }

    #[test]
    fn floats() {
        assert_cross_latex_to_text("3.14");
        assert_cross_latex_to_text("-2.5");
    }

    // --- Variables and Constants ---

    #[test]
    fn variables() {
        assert_cross_latex_to_text("x");
        assert_cross_latex_to_text("y");
        assert_cross_latex_to_text("z");
    }

    #[test]
    fn constants() {
        assert_cross_latex_to_text(r"\pi");
        assert_cross_latex_to_text(r"\infty");
    }

    // --- Arithmetic ---

    #[test]
    fn binary_arithmetic() {
        assert_cross_latex_to_text("1 + 2");
        assert_cross_latex_to_text("5 - 3");
        assert_cross_latex_to_text("2 * 3");
        assert_cross_latex_to_text("2 ^ 3");
    }

    #[test]
    fn fractions() {
        assert_cross_latex_to_text(r"\frac{1}{2}");
        assert_cross_latex_to_text(r"\frac{a}{b}");
        assert_cross_latex_to_text(r"\frac{x + 1}{y - 2}");
    }

    #[test]
    fn precedence() {
        assert_cross_latex_to_text("1 + 2 * 3");
        assert_cross_latex_to_text("2 ^ 3 + 1");
    }

    #[test]
    fn unary_negation() {
        assert_cross_latex_to_text("-5");
    }

    // Note: n! does not cross round-trip — LaTeX parser doesn't support '!'

    // --- Functions ---

    #[test]
    fn trig_functions() {
        assert_cross_latex_to_text(r"\sin\left(x\right)");
        assert_cross_latex_to_text(r"\cos\left(x\right)");
        assert_cross_latex_to_text(r"\tan\left(x\right)");
    }

    #[test]
    fn other_functions() {
        assert_cross_latex_to_text(r"\exp\left(x\right)");
        assert_cross_latex_to_text(r"\ln\left(x\right)");
        assert_cross_latex_to_text(r"\sqrt{x}");
    }

    #[test]
    fn nested_functions() {
        assert_cross_latex_to_text(r"\sin\left(\cos\left(x\right)\right)");
    }

    // --- Equations and Inequalities ---

    #[test]
    fn equations() {
        assert_cross_latex_to_text("x = 5");
        assert_cross_latex_to_text("y = 2 * x + 1");
    }

    #[test]
    fn inequalities() {
        assert_cross_latex_to_text("x < 5");
        assert_cross_latex_to_text(r"y \leq 10");
        assert_cross_latex_to_text("z > 0");
        assert_cross_latex_to_text(r"w \geq -5");
        assert_cross_latex_to_text(r"a \neq b");
    }

    // --- Powers and Roots ---

    #[test]
    fn powers() {
        assert_cross_latex_to_text("x^{2}");
        assert_cross_latex_to_text(r"\sqrt{x}");
    }

    // --- Complex Expressions ---

    #[test]
    fn complex_compositions() {
        assert_cross_latex_to_text(r"\frac{a + b}{c - d}");
        assert_cross_latex_to_text(r"\frac{a^{2} + b^{2}}{c^{2} + d^{2}}");
    }
}

// ============================================================================
// Text → LaTeX: Calculus and Advanced
// ============================================================================

mod text_to_latex_advanced {
    use super::*;

    #[test]
    fn partial_derivatives() {
        assert_cross_text_to_latex("partial(f, x)");
        assert_cross_text_to_latex("partial(f, x, 2)");
    }

    #[test]
    fn higher_order_derivatives() {
        assert_cross_text_to_latex("diff(y, x, 2)");
        assert_cross_text_to_latex("diff(y, x, 3)");
        assert_cross_text_to_latex("d2y/dx2");
    }

    #[test]
    fn vector_calculus() {
        assert_cross_text_to_latex("grad(f)");
    }

    #[test]
    fn logical_operators() {
        assert_cross_text_to_latex("x and y");
        assert_cross_text_to_latex("x or y");
        assert_cross_text_to_latex("not x");
        assert_cross_text_to_latex("x implies y");
        assert_cross_text_to_latex("x iff y");
    }

    #[test]
    fn set_operations() {
        assert_cross_text_to_latex("x union y");
        assert_cross_text_to_latex("x intersect y");
    }

    #[test]
    fn derivative_in_equation() {
        assert_cross_text_to_latex("diff(y, x) = x * y");
        assert_cross_text_to_latex("dy/dx = x + 1");
    }

    #[test]
    fn nested_functions_in_arithmetic() {
        assert_cross_text_to_latex("sin(x) ^ 2 + cos(x) ^ 2");
        assert_cross_text_to_latex("ln(x + 1) / sqrt(x)");
    }

    #[test]
    fn integrals() {
        assert_cross_text_to_latex("integrate(x, x)");
        assert_cross_text_to_latex("integrate(sin(x), x, 0, pi)");
    }

    #[test]
    fn sums() {
        assert_cross_text_to_latex("sum(k, k, 1, n)");
    }

    #[test]
    fn products() {
        assert_cross_text_to_latex("product(k, k, 1, n)");
    }

    #[test]
    fn limits() {
        assert_cross_text_to_latex("limit(f, x, 0)");
    }

    #[test]
    fn operator_form_derivative() {
        assert_cross_text_to_latex("d(x^2)/dx");
    }
}

// ============================================================================
// LaTeX → Text: Calculus and Advanced
// ============================================================================

mod latex_to_text_advanced {
    use super::*;

    // Note: LaTeX derivative Display outputs d/dx(f), which text parser reads
    // as d / dx(f) (division). The d(f)/dx form IS supported but Display doesn't
    // produce it. This is a Display format issue, not a parser gap.

    #[test]
    fn gradient() {
        assert_cross_latex_to_text(r"\nabla f");
    }

    #[test]
    fn logical_operators() {
        assert_cross_latex_to_text(r"x \land y");
        assert_cross_latex_to_text(r"x \lor y");
        assert_cross_latex_to_text(r"\lnot x");
        assert_cross_latex_to_text(r"x \implies y");
        assert_cross_latex_to_text(r"x \iff y");
    }

    #[test]
    fn set_operations() {
        assert_cross_latex_to_text(r"x \cup y");
        assert_cross_latex_to_text(r"x \cap y");
    }

    #[test]
    fn nested_functions() {
        assert_cross_latex_to_text(r"\ln\left(x + 1\right)");
        assert_cross_latex_to_text(r"\sqrt{x^{2} + y^{2}}");
    }
}

// ============================================================================
// Known Cross-Parser Limitations (documented, not tested as assertions)
// ============================================================================
//
// The following expressions do NOT cross round-trip. Documented here for
// reference and to track future parity improvements.
//
// 1. Factorial: n! — LaTeX parser does not support '!' postfix operator
// 2. e^x normalization: text 'e' is Constant(E), LaTeX normalizes e^x to exp(x)
// 3. Prime notation: y' has empty var — no LaTeX equivalent for implicit var
// 4. Integrals: LaTeX \int produces Expression::Integral, no text equivalent yet
// 5. Sums/Products: LaTeX \sum/\prod have no text equivalent yet
// 6. Limits: LaTeX \lim has no text equivalent yet
// 7. Vectors/Matrices: LaTeX \begin{pmatrix} has no text equivalent
// 8. Subscripts: LaTeX x_{i+1} supports expression subscripts, text only simple
// 9. Greek letters: text 'alpha' is Variable("alpha"), LaTeX \alpha is Variable("alpha")
//    (this actually works, but rendering differs)
