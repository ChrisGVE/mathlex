//! Matrix shape validation helpers.

use crate::ast::Expression;

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
    /// use mathlex::ast::Expression;
    ///
    /// let valid = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3), Expression::Integer(4)],
    /// ]);
    /// assert!(valid.is_valid_matrix());
    ///
    /// let ragged = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2)],
    ///     vec![Expression::Integer(3)],
    /// ]);
    /// assert!(!ragged.is_valid_matrix());
    ///
    /// let not_matrix = Expression::Integer(42);
    /// assert!(!not_matrix.is_valid_matrix());
    /// ```
    pub fn is_valid_matrix(&self) -> bool {
        match self {
            Expression::Matrix(rows) => {
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
    /// use mathlex::ast::Expression;
    ///
    /// let matrix = Expression::Matrix(vec![
    ///     vec![Expression::Integer(1), Expression::Integer(2), Expression::Integer(3)],
    ///     vec![Expression::Integer(4), Expression::Integer(5), Expression::Integer(6)],
    /// ]);
    /// assert_eq!(matrix.matrix_dimensions(), Some((2, 3)));
    ///
    /// let not_matrix = Expression::Integer(42);
    /// assert_eq!(not_matrix.matrix_dimensions(), None);
    /// ```
    pub fn matrix_dimensions(&self) -> Option<(usize, usize)> {
        if self.is_valid_matrix() {
            if let Expression::Matrix(rows) = self {
                return Some((rows.len(), rows[0].len()));
            }
        }
        None
    }
}
