// Test for standard derivative notation (without explicit multiplication operator)
use mathlex::ast::Expression;
use mathlex::parser::parse_latex;

#[test]
fn test_standard_first_order_derivative() {
    // Standard notation: \frac{d}{dx}f(x)
    let result = parse_latex(r"\frac{d}{dx}x");
    match result {
        Ok(expr) => match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 1);
                assert_eq!(*expr, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected Derivative variant, got: {:?}", expr),
        },
        Err(e) => panic!("Parsing failed: {}", e),
    }
}

#[test]
fn test_standard_second_order_derivative() {
    // Standard notation: \frac{d^2}{dx^2}f
    let result = parse_latex(r"\frac{d^2}{dx^2}f");
    match result {
        Ok(expr) => match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 2);
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected Derivative variant, got: {:?}", expr),
        },
        Err(e) => panic!("Parsing failed: {}", e),
    }
}

#[test]
fn test_standard_partial_derivative() {
    // Standard notation: \frac{\partial}{\partial x}f
    let result = parse_latex(r"\frac{\partial}{\partial x}f");
    match result {
        Ok(expr) => match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 1);
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected PartialDerivative variant, got: {:?}", expr),
        },
        Err(e) => panic!("Parsing failed: {}", e),
    }
}
