// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

use super::*;

impl LatexParser {
    /// Parses a marked vector: \mathbf{v}, \vec{a}, \hat{n}, \underline{u}
    /// Returns MarkedVector with the given notation style.
    ///
    /// Special case: \mathbf{j} and \mathbf{k} return quaternion basis constants
    /// MathConstant::J and MathConstant::K respectively.
    pub(super) fn parse_marked_vector(
        &mut self,
        notation: VectorNotation,
    ) -> ParseResult<Expression> {
        // The argument is in braces or a single letter
        let name = if self.check(&LatexToken::LBrace) {
            // Parse the braced content as a vector name (consecutive letters)
            self.braced(|p| p.parse_vector_name())?
        } else {
            // Single letter without braces: \vec a
            match self.peek() {
                Some((LatexToken::Letter(ch), _)) => {
                    let ch = *ch;
                    self.next();
                    ch.to_string()
                }
                Some((LatexToken::Command(cmd), _)) => {
                    // Greek letter: \vec\alpha
                    let cmd = cmd.clone();
                    self.next();
                    cmd
                }
                _ => {
                    return Err(ParseError::custom(
                        "expected variable name after vector notation command".to_string(),
                        Some(self.current_span()),
                    ));
                }
            }
        };

        // Special case: \mathbf{j} and \mathbf{k} are quaternion basis vectors
        if notation == VectorNotation::Bold {
            match name.as_str() {
                "j" => return Ok(Expression::Constant(MathConstant::J)),
                "k" => return Ok(Expression::Constant(MathConstant::K)),
                _ => {}
            }
        }

        Ok(Expression::MarkedVector { name, notation })
    }

    /// Parses a vector name from consecutive letters or a single command.
    /// Used for \overrightarrow{AB}, \mathbf{v}, etc.
    pub(super) fn parse_vector_name(&mut self) -> ParseResult<String> {
        let mut name = String::new();

        // Collect consecutive letters
        while let Some((token, _)) = self.peek() {
            match token {
                LatexToken::Letter(ch) => {
                    name.push(*ch);
                    self.next();
                }
                LatexToken::Command(cmd) => {
                    // Greek letter: append command name
                    if name.is_empty() {
                        name = cmd.clone();
                        self.next();
                        break; // Only one command allowed
                    } else {
                        break; // Can't mix letters and commands
                    }
                }
                _ => break,
            }
        }

        if name.is_empty() {
            return Err(ParseError::custom(
                "expected variable name in vector notation".to_string(),
                Some(self.current_span()),
            ));
        }

        Ok(name)
    }

    /// Parses nabla-based expressions: \nabla f, \nabla \cdot F, \nabla \times F
    pub(super) fn parse_nabla(&mut self) -> ParseResult<Expression> {
        // Check what follows \nabla
        match self.peek() {
            Some((LatexToken::Cdot, _)) => {
                // \nabla \cdot F (divergence)
                self.next(); // consume \cdot
                let field = self.parse_power()?;
                Ok(Expression::Divergence {
                    field: Box::new(field),
                })
            }
            Some((LatexToken::Bullet, _)) => {
                // \nabla \bullet F (divergence with bullet)
                self.next(); // consume \bullet
                let field = self.parse_power()?;
                Ok(Expression::Divergence {
                    field: Box::new(field),
                })
            }
            Some((LatexToken::Cross, _)) => {
                // \nabla \times F (curl)
                self.next(); // consume \times
                let field = self.parse_power()?;
                Ok(Expression::Curl {
                    field: Box::new(field),
                })
            }
            Some((LatexToken::Caret, _)) => {
                // \nabla^2 f (Laplacian)
                self.next(); // consume ^
                let power = self.parse_braced_or_atom()?;
                if let Expression::Integer(2) = power {
                    let expr = self.parse_power()?;
                    Ok(Expression::Laplacian {
                        expr: Box::new(expr),
                    })
                } else {
                    Err(ParseError::custom(
                        "expected \\nabla^2 for Laplacian".to_string(),
                        Some(self.current_span()),
                    ))
                }
            }
            _ => {
                // \nabla f (gradient) - just nabla followed by expression
                let expr = self.parse_power()?;
                Ok(Expression::Gradient {
                    expr: Box::new(expr),
                })
            }
        }
    }

    /// Parses the row/column content of a matrix environment until `\end{env_name}`.
    pub(super) fn parse_matrix_rows(
        &mut self,
        env_name: &str,
    ) -> ParseResult<Vec<Vec<Expression>>> {
        let mut rows: Vec<Vec<Expression>> = Vec::new();
        let mut current_row: Vec<Expression> = Vec::new();

        loop {
            // Check for end of environment
            if let Some((LatexToken::EndEnv(end_name), _)) = self.peek() {
                let end_name = end_name.clone();
                self.next();

                if end_name != env_name {
                    return Err(ParseError::custom(
                        format!(
                            "mismatched environment: \\begin{{{}}} ended with \\end{{{}}}",
                            env_name, end_name
                        ),
                        Some(self.current_span()),
                    ));
                }

                if !current_row.is_empty() {
                    rows.push(current_row);
                }
                break;
            }

            current_row.push(self.parse_expression()?);

            match self.peek() {
                Some((LatexToken::Ampersand, _)) => {
                    self.next(); // consume & — continue current row
                }
                Some((LatexToken::DoubleBackslash, _)) => {
                    self.next(); // consume \\ — end row
                    rows.push(current_row);
                    current_row = Vec::new();
                }
                Some((LatexToken::EndEnv(_), _)) => {
                    // Handled at the top of the next iteration
                }
                Some((token, span)) => {
                    return Err(ParseError::unexpected_token(
                        vec!["&", "\\\\", "\\end"],
                        format!("{:?}", token),
                        Some(*span),
                    ));
                }
                None => {
                    return Err(ParseError::unexpected_eof(
                        vec!["&", "\\\\", "\\end"],
                        Some(self.current_span()),
                    ));
                }
            }
        }

        Ok(rows)
    }

    /// Validates that all rows of a matrix have the same number of columns.
    pub(super) fn validate_matrix_rows(
        rows: &[Vec<Expression>],
        span: crate::error::Span,
    ) -> ParseResult<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let first_col_count = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != first_col_count {
                return Err(ParseError::custom(
                    format!(
                        "inconsistent matrix row lengths: row 0 has {} columns, row {} has {} columns",
                        first_col_count, i, row.len()
                    ),
                    Some(span),
                ));
            }
        }
        Ok(())
    }

    /// Parses a matrix environment (\begin{matrix}...\end{matrix} and variants).
    pub(super) fn parse_matrix_environment(&mut self, env_name: &str) -> ParseResult<Expression> {
        match env_name {
            "matrix" | "bmatrix" | "pmatrix" | "vmatrix" | "Bmatrix" | "Vmatrix" => {}
            _ => {
                return Err(ParseError::invalid_latex_command(
                    format!("\\begin{{{}}}", env_name),
                    Some(self.current_span()),
                ));
            }
        }

        let rows = self.parse_matrix_rows(env_name)?;
        let span = self.current_span();
        Self::validate_matrix_rows(&rows, span)?;

        // Single-column matrix → column vector
        if !rows.is_empty() && rows[0].len() == 1 {
            let elements: Vec<Expression> = rows.into_iter().map(|mut row| row.remove(0)).collect();
            Ok(Expression::Vector(elements))
        } else {
            Ok(Expression::Matrix(rows))
        }
    }
}
