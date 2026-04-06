// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Converts an expression to a subscript string representation.
    /// For complex expressions, creates a flattened representation suitable for variable names.
    pub(super) fn expression_to_subscript_string(&self, expr: &Expression) -> ParseResult<String> {
        match expr {
            Expression::Integer(n) => Ok(n.to_string()),
            Expression::Variable(s) => Ok(s.clone()),
            // Constants in subscripts are converted to their letter representation
            Expression::Constant(c) => Ok(match c {
                MathConstant::E => "e".to_string(),
                MathConstant::I => "i".to_string(),
                MathConstant::J => "j".to_string(),
                MathConstant::K => "k".to_string(),
                MathConstant::Pi => "pi".to_string(),
                MathConstant::Infinity => "inf".to_string(),
                MathConstant::NegInfinity => "neginf".to_string(),
                MathConstant::NaN => "nan".to_string(),
            }),
            Expression::Binary { op, left, right } => {
                let left_str = self.expression_to_subscript_string(left)?;
                let right_str = self.expression_to_subscript_string(right)?;
                let op_str = match op {
                    BinaryOp::Add => "plus",
                    BinaryOp::Sub => "minus",
                    BinaryOp::Mul => "times",
                    BinaryOp::Div => "div",
                    BinaryOp::Pow => "pow",
                    BinaryOp::Mod => "mod",
                    BinaryOp::PlusMinus => "pm",
                    BinaryOp::MinusPlus => "mp",
                };
                Ok(format!("{}{}{}", left_str, op_str, right_str))
            }
            Expression::Unary { op, operand } => {
                let operand_str = self.expression_to_subscript_string(operand)?;
                let op_str = match op {
                    crate::ast::UnaryOp::Neg => "neg",
                    crate::ast::UnaryOp::Pos => "pos",
                    crate::ast::UnaryOp::Factorial => "fact",
                    crate::ast::UnaryOp::Transpose => "T",
                };
                Ok(format!("{}{}", op_str, operand_str))
            }
            _ => Err(ParseError::invalid_subscript(
                "subscript contains unsupported expression type",
                Some(self.current_span()),
            )),
        }
    }

    /// Extracts (is_partial, order) from a derivative numerator expression.
    /// Returns None if the expression is not a valid derivative numerator.
    pub(super) fn match_derivative_numerator(expr: &Expression) -> Option<(bool, u32)> {
        match expr {
            Expression::Variable(s) if s == "d" => Some((false, 1)),
            Expression::Variable(s) if s == "partial" => Some((true, 1)),
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                let is_partial = match &**left {
                    Expression::Variable(s) if s == "d" => false,
                    Expression::Variable(s) if s == "partial" => true,
                    _ => return None,
                };
                let order = match &**right {
                    Expression::Integer(n) if *n > 0 => *n as u32,
                    _ => return None,
                };
                Some((is_partial, order))
            }
            _ => None,
        }
    }

    /// Extracts (is_partial, var, order) from a derivative denominator expression.
    /// Returns None if the expression is not a valid derivative denominator.
    pub(super) fn match_derivative_denominator(expr: &Expression) -> Option<(bool, String, u32)> {
        let Expression::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } = expr
        else {
            return None;
        };

        let is_partial = match &**left {
            Expression::Variable(s) if s == "d" => false,
            Expression::Variable(s) if s == "partial" => true,
            _ => return None,
        };

        match &**right {
            Expression::Variable(v) => Some((is_partial, v.clone(), 1)),
            Expression::Binary {
                op: BinaryOp::Pow,
                left: var_expr,
                right: order_expr,
            } => {
                let var = match &**var_expr {
                    Expression::Variable(v) => v.clone(),
                    _ => return None,
                };
                let order = match &**order_expr {
                    Expression::Integer(n) if *n > 0 => *n as u32,
                    _ => return None,
                };
                Some((is_partial, var, order))
            }
            _ => None,
        }
    }

    /// Tries to parse a \frac as a derivative.
    /// Returns Some(derivative_expr) if it matches the pattern, None otherwise.
    pub(super) fn try_parse_derivative(
        &mut self,
        numerator: Expression,
        denominator: Expression,
    ) -> ParseResult<Option<Expression>> {
        let Some((is_partial, num_order)) = Self::match_derivative_numerator(&numerator) else {
            return Ok(None);
        };

        let Some((denom_is_partial, var, denom_order)) =
            Self::match_derivative_denominator(&denominator)
        else {
            return Ok(None);
        };

        // Numerator and denominator must use the same operator (d vs \partial)
        if is_partial != denom_is_partial || num_order != denom_order {
            return Ok(None);
        }

        let expr = self.parse_power()?;

        let derivative = if is_partial {
            Expression::PartialDerivative {
                expr: Box::new(expr),
                var,
                order: num_order,
            }
        } else {
            Expression::Derivative {
                expr: Box::new(expr),
                var,
                order: num_order,
            }
        };

        Ok(Some(derivative))
    }
}
