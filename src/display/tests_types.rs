/// Tests for MathConstant, BinaryOp, UnaryOp, Direction, InequalityOp, IntegralBounds Display.
use crate::ast::{
    BinaryOp, Direction, Expression, InequalityOp, IntegralBounds, MathConstant, UnaryOp,
};

// Tests for MathConstant Display

#[test]
fn test_math_constant_pi() {
    assert_eq!(format!("{}", MathConstant::Pi), "pi");
}

#[test]
fn test_math_constant_e() {
    assert_eq!(format!("{}", MathConstant::E), "e");
}

#[test]
fn test_math_constant_i() {
    assert_eq!(format!("{}", MathConstant::I), "i");
}

#[test]
fn test_math_constant_infinity() {
    assert_eq!(format!("{}", MathConstant::Infinity), "inf");
}

#[test]
fn test_math_constant_neg_infinity() {
    assert_eq!(format!("{}", MathConstant::NegInfinity), "-inf");
}

// Tests for BinaryOp Display

#[test]
fn test_binary_op_add() {
    assert_eq!(format!("{}", BinaryOp::Add), "+");
}

#[test]
fn test_binary_op_sub() {
    assert_eq!(format!("{}", BinaryOp::Sub), "-");
}

#[test]
fn test_binary_op_mul() {
    assert_eq!(format!("{}", BinaryOp::Mul), "*");
}

#[test]
fn test_binary_op_div() {
    assert_eq!(format!("{}", BinaryOp::Div), "/");
}

#[test]
fn test_binary_op_pow() {
    assert_eq!(format!("{}", BinaryOp::Pow), "^");
}

#[test]
fn test_binary_op_mod() {
    assert_eq!(format!("{}", BinaryOp::Mod), "%");
}

// Tests for UnaryOp Display

#[test]
fn test_unary_op_neg() {
    assert_eq!(format!("{}", UnaryOp::Neg), "-");
}

#[test]
fn test_unary_op_pos() {
    assert_eq!(format!("{}", UnaryOp::Pos), "+");
}

#[test]
fn test_unary_op_factorial() {
    assert_eq!(format!("{}", UnaryOp::Factorial), "!");
}

#[test]
fn test_unary_op_transpose() {
    assert_eq!(format!("{}", UnaryOp::Transpose), "'");
}

// Tests for Direction Display

#[test]
fn test_direction_left() {
    assert_eq!(format!("{}", Direction::Left), "-");
}

#[test]
fn test_direction_right() {
    assert_eq!(format!("{}", Direction::Right), "+");
}

#[test]
fn test_direction_both() {
    assert_eq!(format!("{}", Direction::Both), "");
}

// Tests for InequalityOp Display

#[test]
fn test_inequality_op_lt() {
    assert_eq!(format!("{}", InequalityOp::Lt), "<");
}

#[test]
fn test_inequality_op_le() {
    assert_eq!(format!("{}", InequalityOp::Le), "<=");
}

#[test]
fn test_inequality_op_gt() {
    assert_eq!(format!("{}", InequalityOp::Gt), ">");
}

#[test]
fn test_inequality_op_ge() {
    assert_eq!(format!("{}", InequalityOp::Ge), ">=");
}

#[test]
fn test_inequality_op_ne() {
    assert_eq!(format!("{}", InequalityOp::Ne), "!=");
}

// Tests for IntegralBounds Display

#[test]
fn test_integral_bounds_simple() {
    let bounds = IntegralBounds {
        lower: Box::new(Expression::Integer(0)),
        upper: Box::new(Expression::Integer(1)),
    };
    assert_eq!(format!("{}", bounds), "0, 1");
}

#[test]
fn test_integral_bounds_variables() {
    let bounds = IntegralBounds {
        lower: Box::new(Expression::Variable("a".to_string())),
        upper: Box::new(Expression::Variable("b".to_string())),
    };
    assert_eq!(format!("{}", bounds), "a, b");
}
