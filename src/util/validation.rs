//! Matrix shape validation helpers.

use crate::ast::{ExprKind, Expression};

impl Expression {
    /// Returns whether this expression is a well-formed rectangular matrix.
    ///
    /// A valid matrix has at least one row, and all rows have the same length.
    /// Returns `false` for non-Matrix expressions.
    ///
    /// Parsers always produce valid matrices. This method is useful for
    /// validating manually constructed matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, ExprKind};
    ///
    /// let valid: Expression = ExprKind::Matrix(vec![
    ///     vec![Expression::integer(1), Expression::integer(2)],
    ///     vec![Expression::integer(3), Expression::integer(4)],
    /// ]).into();
    /// assert!(valid.is_valid_matrix());
    ///
    /// let ragged: Expression = ExprKind::Matrix(vec![
    ///     vec![Expression::integer(1), Expression::integer(2)],
    ///     vec![Expression::integer(3)],
    /// ]).into();
    /// assert!(!ragged.is_valid_matrix());
    ///
    /// let not_matrix = Expression::integer(42);
    /// assert!(!not_matrix.is_valid_matrix());
    /// ```
    pub fn is_valid_matrix(&self) -> bool {
        match &self.kind {
            ExprKind::Matrix(rows) => {
                if rows.is_empty() {
                    return false;
                }
                let cols = rows[0].len();
                cols > 0 && rows.iter().all(|row| row.len() == cols)
            }
            _ => false,
        }
    }

    /// Returns the dimensions `(rows, cols)` of a valid rectangular matrix.
    ///
    /// Returns `None` if this is not a Matrix expression or if the matrix is
    /// empty or ragged (non-rectangular).
    ///
    /// # Examples
    ///
    /// ```
    /// use mathlex::ast::{Expression, ExprKind};
    ///
    /// let matrix: Expression = ExprKind::Matrix(vec![
    ///     vec![Expression::integer(1), Expression::integer(2), Expression::integer(3)],
    ///     vec![Expression::integer(4), Expression::integer(5), Expression::integer(6)],
    /// ]).into();
    /// assert_eq!(matrix.matrix_dimensions(), Some((2, 3)));
    ///
    /// let not_matrix = Expression::integer(42);
    /// assert_eq!(not_matrix.matrix_dimensions(), None);
    /// ```
    pub fn matrix_dimensions(&self) -> Option<(usize, usize)> {
        if self.is_valid_matrix() {
            if let ExprKind::Matrix(rows) = &self.kind {
                return Some((rows.len(), rows[0].len()));
            }
        }
        None
    }
}
