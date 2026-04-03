// Allow large error variants - boxing would be a breaking API change
#![allow(clippy::result_large_err)]

//! LaTeX expression parser for mathematical notation.
//!
//! This module provides parsing capabilities for LaTeX mathematical expressions,
//! converting tokenized LaTeX input into an Abstract Syntax Tree (AST).
//!
//! # Supported Constructs
//!
//! - **Fractions**: `\frac{num}{denom}` → Binary division
//! - **Roots**: `\sqrt{x}`, `\sqrt[n]{x}` → Function calls
//! - **Powers**: `x^2`, `x^{expr}` → Binary exponentiation
//! - **Subscripts**: `x_1`, `x_i`, `x_{i+1}` → Variables with subscripts (supports expressions)
//! - **Greek letters**: `\alpha`, `\beta`, etc. → Variables
//! - **Constants**: `\pi`, `\infty`, `e`, `i` → Mathematical constants
//! - **Trigonometric functions**: `\sin`, `\cos`, `\tan`, etc. → Functions
//! - **Basic operators**: `+`, `-`, `*`, `/`
//!
//! # Context-Aware Parsing of `e` and `i`
//!
//! The letters `e` and `i` receive special treatment to distinguish between their use as
//! mathematical constants (Euler's number and the imaginary unit) versus as variables.
//!
//! ## Explicit Markers (Always Constants)
//!
//! Use explicit markers to unambiguously specify constants:
//! - `\mathrm{e}` → Euler's number `e ≈ 2.71828`
//! - `\mathrm{i}` → Imaginary unit `i`
//! - `\imath` → Imaginary unit (mathematical notation)
//! - `\jmath` → Imaginary unit (engineering notation)
//!
//! ## Bound Variables in Iterators
//!
//! Index variables in `\sum` and `\prod` take precedence over constant interpretation:
//! - `\sum_{i=1}^{n} i` → `i` is a variable (the summation index)
//! - `\prod_{e=1}^{n} e` → `e` is a variable (the product index)
//!
//! ## Default Behavior
//!
//! When unbound and without explicit markers:
//! - `e` defaults to `Constant(E)` (Euler's number)
//! - `i` defaults to `Constant(I)` (imaginary unit)
//!
//! ## Exponential Normalization
//!
//! When `e` (Euler's number) is raised to a power, it's normalized to `exp()`:
//! - `e^x` → `Function("exp", [x])`
//! - `e^{i\pi}` → `Function("exp", [Constant(I) * Constant(Pi)])`
//!
//! This ensures equivalence with `\exp{x}`.
//!
//! ## Known Limitations
//!
//! 1. **Integral scope timing**: In `\int f(i) di`, the integrand is parsed before
//!    the differential variable is known. The `i` in `f(i)` will be `Constant(I)`.
//!    Workaround: use `\mathrm{i}` explicitly if you need `i` as a variable.
//!
//! 2. **Single-letter index only**: Index variables in `\sum` and `\prod` must be
//!    single ASCII letters.
//!
//! 3. **No complex pattern detection**: Patterns like `a + bi` are not specially
//!    detected; `i` defaults to constant regardless of context.
//!
//! # Example
//!
//! ```ignore
//! use mathlex::parser::parse_latex;
//!
//! let expr = parse_latex(r"\frac{1}{2}").unwrap();
//! // Returns: Binary { op: Div, left: Integer(1), right: Integer(2) }
//! ```

use std::collections::HashSet;

use crate::ast::{
    BinaryOp, Direction, Expression, IndexType, InequalityOp, IntegralBounds, LogicalOp,
    MathConstant, MathFloat, RelationOp, SetOp, SetRelation, TensorIndex, VectorNotation,
};
use crate::error::{ParseError, ParseOutput, ParseResult, Span};
use crate::parser::latex_tokenizer::{tokenize_latex, LatexToken};
use crate::parser::Spanned;

/// Parses a LaTeX mathematical expression.
///
/// # Arguments
///
/// * `input` - The LaTeX string to parse
///
/// # Returns
///
/// A parsed AST expression or a parse error.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_latex;
///
/// // Simple fraction
/// let expr = parse_latex(r"\frac{1}{2}").unwrap();
///
/// // Square root
/// let expr = parse_latex(r"\sqrt{x}").unwrap();
///
/// // Power
/// let expr = parse_latex(r"x^{2+3}").unwrap();
/// ```
pub fn parse_latex(input: &str) -> ParseResult<Expression> {
    let tokens = tokenize_latex(input)?;
    let parser = LatexParser::new(tokens, false);
    parser.parse_strict()
}

/// Parses a LaTeX mathematical expression in lenient (error-recovering) mode.
///
/// Instead of stopping at the first error, this function collects all errors
/// and returns a partial AST where possible.
///
/// # Arguments
///
/// * `input` - The LaTeX string to parse
///
/// # Returns
///
/// A [`ParseOutput`] containing the partial AST and all errors found.
///
/// # Examples
///
/// ```
/// use mathlex::parser::parse_latex_lenient;
///
/// let output = parse_latex_lenient(r"\frac{1}{} + x");
/// assert!(output.has_errors());
/// ```
pub fn parse_latex_lenient(input: &str) -> ParseOutput {
    let tokens = match tokenize_latex(input) {
        Ok(tokens) => tokens,
        Err(err) => {
            return ParseOutput {
                expression: None,
                errors: vec![err],
            }
        }
    };
    let parser = LatexParser::new(tokens, true);
    parser.parse_lenient()
}

/// Internal parser state for LaTeX expressions.
struct LatexParser {
    /// Token stream with positions
    tokens: Vec<Spanned<LatexToken>>,
    /// Current position in token stream
    pos: usize,
    /// Stack of bound variable scopes (for sum/product index variables)
    bound_scopes: Vec<HashSet<String>>,
    /// Flag indicating we're parsing inside an integral context where
    /// 'dx' should not be parsed as a Differential
    in_integral_context: bool,
    /// Flag indicating we're parsing inside a fraction where
    /// 'dx' should not be parsed as a Differential (to allow d/dx derivative notation)
    in_fraction_context: bool,
    /// Errors collected during lenient parsing
    collected_errors: Vec<ParseError>,
}

impl LatexParser {
    /// Creates a new parser from a token stream.
    fn new(tokens: Vec<Spanned<LatexToken>>, _lenient: bool) -> Self {
        Self {
            tokens,
            pos: 0,
            bound_scopes: Vec::new(),
            in_integral_context: false,
            in_fraction_context: false,
            collected_errors: Vec::new(),
        }
    }

    /// Pushes a new scope with the given bound variables.
    fn push_scope(&mut self, vars: impl IntoIterator<Item = String>) {
        self.bound_scopes.push(vars.into_iter().collect());
    }

    /// Pops the current scope.
    fn pop_scope(&mut self) {
        self.bound_scopes.pop();
    }

    /// Checks if a variable name is bound in any current scope.
    fn is_bound(&self, name: &str) -> bool {
        self.bound_scopes.iter().any(|scope| scope.contains(name))
    }

    /// Resolves a letter to either a constant (for `e` and `i`) or a variable.
    ///
    /// Resolution rules:
    /// 1. Bound variables (in sum/product scopes) are always variables
    /// 2. Explicit markers (`\mathrm{e}`, `\mathrm{i}`, `\imath`, `\jmath`) are always constants
    /// 3. By default, `e` is Euler's number and `i` is the imaginary unit
    fn resolve_letter(&self, ch: char, is_explicit: bool) -> Expression {
        let name = ch.to_string();

        // Rule 1: Bound variables are always variables
        if self.is_bound(&name) {
            return Expression::Variable(name);
        }

        // Rule 2: Explicit markers are always constants
        // This includes e, i (complex) and j, k (quaternion) from \mathrm{x}
        if is_explicit {
            return match ch {
                'e' => Expression::Constant(MathConstant::E),
                'i' => Expression::Constant(MathConstant::I),
                'j' => Expression::Constant(MathConstant::J),
                'k' => Expression::Constant(MathConstant::K),
                _ => Expression::Variable(name),
            };
        }

        // Rule 3: Default for e and i (unbound) - these are constants
        // Note: j and k default to variables (quaternion context requires explicit markers)
        if ch == 'e' || ch == 'i' {
            return match ch {
                'e' => Expression::Constant(MathConstant::E),
                'i' => Expression::Constant(MathConstant::I),
                _ => unreachable!(),
            };
        }

        Expression::Variable(name)
    }

    /// Returns the current token without consuming it.
    fn peek(&self) -> Option<&Spanned<LatexToken>> {
        self.tokens.get(self.pos)
    }

    /// Returns the token at offset positions ahead without consuming it.
    fn peek_ahead(&self, offset: usize) -> Option<&Spanned<LatexToken>> {
        self.tokens.get(self.pos + offset)
    }

    /// Returns the current token and advances position.
    fn next(&mut self) -> Option<Spanned<LatexToken>> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    /// Returns the current position/span for error reporting.
    fn current_span(&self) -> Span {
        self.peek().map(|(_, span)| *span).unwrap_or_else(|| {
            // Use the last token's end position if we're at EOF
            if let Some((_, last_span)) = self.tokens.last() {
                Span::at(last_span.end)
            } else {
                Span::start()
            }
        })
    }

    /// Checks if current token matches a pattern without consuming.
    fn check(&self, expected: &LatexToken) -> bool {
        self.peek().map(|(tok, _)| tok == expected).unwrap_or(false)
    }

    /// Consumes a token if it matches the expected token.
    fn consume(&mut self, expected: LatexToken) -> ParseResult<Span> {
        if let Some((token, span)) = self.next() {
            if token == expected {
                Ok(span)
            } else {
                Err(ParseError::unexpected_token(
                    vec![format!("{:?}", expected)],
                    format!("{:?}", token),
                    Some(span),
                ))
            }
        } else {
            Err(ParseError::unexpected_eof(
                vec![format!("{:?}", expected)],
                Some(self.current_span()),
            ))
        }
    }

    /// Returns true if the token is a synchronization point for error recovery.
    fn is_sync_token(token: &LatexToken) -> bool {
        matches!(
            token,
            LatexToken::RBrace
                | LatexToken::Plus
                | LatexToken::Minus
                | LatexToken::Equals
                | LatexToken::Eof
        ) || matches!(token, LatexToken::Command(cmd) if matches!(
            cmd.as_str(),
            "frac" | "sqrt" | "sum" | "prod" | "int" | "iint" | "iiint" | "oint" | "lim"
        ))
    }

    /// Advances past tokens until a synchronization point is found.
    /// Used in lenient mode to recover from errors.
    fn synchronize(&mut self) {
        while let Some((token, _)) = self.peek() {
            if Self::is_sync_token(token) {
                return;
            }
            self.next();
        }
    }

    /// Strict entry point: fails on first error.
    fn parse_strict(mut self) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;

        // Ensure we consumed all non-EOF tokens
        if let Some((token, span)) = self.peek() {
            if !matches!(token, LatexToken::Eof) {
                return Err(ParseError::unexpected_token(
                    vec!["end of input"],
                    format!("{:?}", token),
                    Some(*span),
                ));
            }
        }

        Ok(expr)
    }

    /// Lenient entry point: collects errors and returns partial AST.
    fn parse_lenient(mut self) -> ParseOutput {
        let mut parts: Vec<Expression> = Vec::new();

        while self.peek().is_some() && !matches!(self.peek(), Some((LatexToken::Eof, _))) {
            match self.parse_expression() {
                Ok(expr) => {
                    parts.push(expr);
                    // If there are remaining non-EOF tokens that aren't consumed,
                    // try to continue parsing
                    if let Some((token, _)) = self.peek() {
                        if !matches!(token, LatexToken::Eof) {
                            // Record the unconsumed token as an error but keep going
                            let span = self.current_span();
                            self.collected_errors.push(ParseError::unexpected_token(
                                vec!["end of input or operator"],
                                format!("{:?}", token),
                                Some(span),
                            ));
                            self.synchronize();
                        }
                    }
                }
                Err(err) => {
                    self.collected_errors.push(err);
                    self.synchronize();
                    // Skip the sync token itself if it's not EOF
                    if let Some((token, _)) = self.peek() {
                        if !matches!(token, LatexToken::Eof) && matches!(token, LatexToken::RBrace)
                        {
                            self.next();
                        }
                    }
                }
            }
        }

        let expression = match parts.len() {
            0 => None,
            1 => Some(parts.remove(0)),
            _ => {
                // Multiple successfully-parsed segments: return the first
                Some(parts.remove(0))
            }
        };

        ParseOutput {
            expression,
            errors: self.collected_errors,
        }
    }

    /// Parses an expression (entry point for recursive descent).
    fn parse_expression(&mut self) -> ParseResult<Expression> {
        self.parse_function_signature()
    }

    /// Parses function signature: f: A → B
    /// Lowest precedence (type annotations).
    fn parse_function_signature(&mut self) -> ParseResult<Expression> {
        let left = self.parse_logical_iff()?;

        // Check for function signature (colon followed by \to)
        if let Some((LatexToken::Colon, _)) = self.peek() {
            // We have a potential function signature
            // Extract function name (must be a simple variable)
            let name = match &left {
                Expression::Variable(n) => n.clone(),
                _ => {
                    // Not a valid function signature, just return the expression
                    return Ok(left);
                }
            };

            self.next(); // consume colon
            let domain = self.parse_logical_iff()?;

            // Expect \to token
            if let Some((LatexToken::To, _)) = self.peek() {
                self.next(); // consume \to
                let codomain = self.parse_logical_iff()?;

                return Ok(Expression::FunctionSignature {
                    name,
                    domain: Box::new(domain),
                    codomain: Box::new(codomain),
                });
            } else {
                // Missing \to, return error
                return Err(ParseError::custom(
                    "expected \\to after domain in function signature".to_string(),
                    Some(self.current_span()),
                ));
            }
        }

        Ok(left)
    }

    /// Parses biconditional (iff) expressions: P \iff Q
    /// Lowest logical precedence.
    fn parse_logical_iff(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_implies()?;

        while let Some((LatexToken::Iff, _)) = self.peek() {
            self.next(); // consume \iff
            let right = self.parse_logical_implies()?;
            left = Expression::Logical {
                op: LogicalOp::Iff,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses implication expressions: P \implies Q
    fn parse_logical_implies(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_or()?;

        while let Some((LatexToken::Implies, _)) = self.peek() {
            self.next(); // consume \implies
            let right = self.parse_logical_or()?;
            left = Expression::Logical {
                op: LogicalOp::Implies,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses logical OR expressions: P \lor Q
    fn parse_logical_or(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_logical_and()?;

        while let Some((LatexToken::Lor, _)) = self.peek() {
            self.next(); // consume \lor
            let right = self.parse_logical_and()?;
            left = Expression::Logical {
                op: LogicalOp::Or,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses logical AND expressions: P \land Q
    fn parse_logical_and(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_set_membership()?;

        while let Some((LatexToken::Land, _)) = self.peek() {
            self.next(); // consume \land
            let right = self.parse_set_membership()?;
            left = Expression::Logical {
                op: LogicalOp::And,
                operands: vec![left, right],
            };
        }

        Ok(left)
    }

    /// Parses set membership and subset relations.
    /// Handles: x \in S, x \notin S, A \subset B, A \subseteq B, A \supset B, A \supseteq B
    fn parse_set_membership(&mut self) -> ParseResult<Expression> {
        let left = self.parse_relation()?;

        // Check for set membership and subset operators
        if let Some((token, _)) = self.peek() {
            let relation = match token {
                LatexToken::In => Some(SetRelation::In),
                LatexToken::NotIn => Some(SetRelation::NotIn),
                LatexToken::Subset => Some(SetRelation::Subset),
                LatexToken::SubsetEq => Some(SetRelation::SubsetEq),
                LatexToken::Superset => Some(SetRelation::Superset),
                LatexToken::SupersetEq => Some(SetRelation::SupersetEq),
                _ => None,
            };

            if let Some(rel) = relation {
                self.next(); // consume the relation token
                let right = self.parse_relation()?;
                return Ok(Expression::SetRelationExpr {
                    relation: rel,
                    element: Box::new(left),
                    set: Box::new(right),
                });
            }
        }

        Ok(left)
    }

    /// Matches a similarity/equivalence relation token to its operator.
    fn match_math_relation(token: &LatexToken) -> Option<RelationOp> {
        match token {
            LatexToken::Sim => Some(RelationOp::Similar),
            LatexToken::Equiv => Some(RelationOp::Equivalent),
            LatexToken::Cong => Some(RelationOp::Congruent),
            LatexToken::Approx => Some(RelationOp::Approx),
            _ => None,
        }
    }

    /// Matches an equation/inequality token to an optional InequalityOp (None = Equation).
    fn match_ineq_relation(token: &LatexToken) -> Option<Option<InequalityOp>> {
        match token {
            LatexToken::Equals => Some(None),
            LatexToken::Less => Some(Some(InequalityOp::Lt)),
            LatexToken::Greater => Some(Some(InequalityOp::Gt)),
            LatexToken::Command(cmd) => match cmd.as_str() {
                "lt" => Some(Some(InequalityOp::Lt)),
                "gt" => Some(Some(InequalityOp::Gt)),
                "leq" | "le" => Some(Some(InequalityOp::Le)),
                "geq" | "ge" => Some(Some(InequalityOp::Ge)),
                "neq" | "ne" => Some(Some(InequalityOp::Ne)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns true if the token is any relation operator (used to detect chaining).
    fn is_relation_token(token: &LatexToken) -> bool {
        matches!(
            token,
            LatexToken::Equals
                | LatexToken::Less
                | LatexToken::Greater
                | LatexToken::Sim
                | LatexToken::Equiv
                | LatexToken::Cong
                | LatexToken::Approx
        ) || matches!(
            token,
            LatexToken::Command(cmd) if matches!(
                cmd.as_str(),
                "lt" | "gt" | "leq" | "le" | "geq" | "ge" | "neq" | "ne"
            )
        )
    }

    /// Builds an Equation or Inequality expression from two operands.
    fn build_equation_or_ineq(
        rel_op: Option<InequalityOp>,
        left: Expression,
        right: Expression,
    ) -> Expression {
        match rel_op {
            None => Expression::Equation {
                left: Box::new(left),
                right: Box::new(right),
            },
            Some(op) => Expression::Inequality {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        }
    }

    /// Parses relational expressions (=, <, >, \leq, \geq, \neq, etc.).
    fn parse_relation(&mut self) -> ParseResult<Expression> {
        let left = self.parse_additive()?;

        if let Some((token, span)) = self.peek() {
            let span = *span;

            // Check similarity/equivalence/congruence/approximation relations
            if let Some(rel_op) = Self::match_math_relation(token) {
                self.next();
                let right = self.parse_additive()?;
                return Ok(Expression::Relation {
                    op: rel_op,
                    left: Box::new(left),
                    right: Box::new(right),
                });
            }

            // Check equation/inequality relations
            if let Some(rel_op) = Self::match_ineq_relation(token) {
                self.next();
                let right = self.parse_additive()?;

                // Reject chained relations
                if let Some((next_token, next_span)) = self.peek() {
                    if Self::is_relation_token(next_token) {
                        return Err(ParseError::custom(
                            "chained relations are not supported; use explicit grouping"
                                .to_string(),
                            Some(*next_span),
                        ));
                    }
                }

                return Ok(Self::build_equation_or_ineq(rel_op, left, right));
            }

            let _ = span; // suppress unused warning
        }

        Ok(left)
    }

    /// Parses additive expressions (+, -, \pm, \mp) and set union/difference.
    fn parse_additive(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        while let Some((token, _)) = self.peek() {
            // Check for set operations first (union, difference)
            match token {
                LatexToken::Cup => {
                    self.next(); // consume \cup
                    let right = self.parse_multiplicative()?;
                    left = Expression::SetOperation {
                        op: SetOp::Union,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                    continue;
                }
                LatexToken::Setminus => {
                    self.next(); // consume \setminus
                    let right = self.parse_multiplicative()?;
                    left = Expression::SetOperation {
                        op: SetOp::Difference,
                        left: Box::new(left),
                        right: Box::new(right),
                    };
                    continue;
                }
                _ => {}
            }

            // Standard arithmetic operators
            let op = match token {
                LatexToken::Plus => BinaryOp::Add,
                LatexToken::Minus => BinaryOp::Sub,
                LatexToken::Command(cmd) if cmd == "pm" => BinaryOp::PlusMinus,
                LatexToken::Command(cmd) if cmd == "mp" => BinaryOp::MinusPlus,
                _ => break,
            };

            self.next(); // consume operator
            let right = self.parse_multiplicative()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Tries to parse a named product operator (\circ, \bullet, \otimes, \wedge, \times).
    /// Returns `Ok(new_left)` if a product operator was consumed, `Err(left)` otherwise.
    fn try_parse_named_product(
        &mut self,
        token: &LatexToken,
        left: Expression,
    ) -> ParseResult<Result<Expression, Expression>> {
        match token {
            LatexToken::Circ => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::Composition {
                    outer: Box::new(left),
                    inner: Box::new(right),
                }))
            }
            LatexToken::Bullet => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::DotProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Otimes => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::OuterProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Wedge => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::WedgeProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            LatexToken::Cross => {
                self.next();
                let right = self.parse_power()?;
                Ok(Ok(Expression::CrossProduct {
                    left: Box::new(left),
                    right: Box::new(right),
                }))
            }
            _ => Ok(Err(left)),
        }
    }

    /// Parses multiplicative expressions (*, /) and set intersection.
    fn parse_multiplicative(&mut self) -> ParseResult<Expression> {
        let mut left = self.parse_power()?;

        loop {
            // Clone the token so we drop the immutable borrow before any `&mut self` call.
            let token = match self.peek() {
                Some((t, _)) => t.clone(),
                None => break,
            };

            // Set intersection
            if matches!(token, LatexToken::Cap) {
                self.next();
                let right = self.parse_power()?;
                left = Expression::SetOperation {
                    op: SetOp::Intersection,
                    left: Box::new(left),
                    right: Box::new(right),
                };
                continue;
            }

            // Named product operators (\circ, \bullet, \otimes, \wedge, \times)
            match self.try_parse_named_product(&token, left)? {
                Ok(new_left) => {
                    left = new_left;
                    continue;
                }
                Err(returned) => {
                    left = returned;
                }
            }

            // Regular scalar multiplication (token was already cloned above)
            let op = match &token {
                LatexToken::Star | LatexToken::Cdot => BinaryOp::Mul,
                LatexToken::Slash => BinaryOp::Div,
                _ => {
                    if self.should_insert_implicit_mult(&left) {
                        BinaryOp::Mul
                    } else {
                        break;
                    }
                }
            };

            // Consume the explicit operator (not for implicit multiplication)
            if matches!(
                token,
                LatexToken::Star | LatexToken::Cdot | LatexToken::Slash
            ) {
                self.next();
            }

            let right = self.parse_power()?;
            left = Expression::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Determines if implicit multiplication should be inserted in LaTeX.
    /// This is used for patterns like 2x, xy, 2\pi, i\pi, etc.
    fn should_insert_implicit_mult(&self, left: &Expression) -> bool {
        // Only insert implicit mult when left is a simple variable, number, constant, or differential
        let is_valid_left = matches!(
            left,
            Expression::Variable(_)
                | Expression::Integer(_)
                | Expression::Float(_)
                | Expression::Constant(_)
                | Expression::Differential { .. }
        );
        if !is_valid_left {
            return false;
        }

        // Check if next token is something that could start a multiplicand
        match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                // In integral context, don't trigger implicit mult for 'd' followed by a letter
                // This allows the integral parser to handle 'dx' properly
                if self.in_integral_context && *ch == 'd' {
                    if let Some((LatexToken::Letter(_), _)) = self.peek_ahead(1) {
                        return false;
                    }
                }
                true
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Exclude relation commands and right delimiters - they should not trigger implicit mult
                !matches!(
                    cmd.as_str(),
                    "lt" | "gt"
                        | "leq"
                        | "le"
                        | "geq"
                        | "ge"
                        | "neq"
                        | "ne"
                        | "pm"
                        | "mp"
                        | "cdot"
                        | "times"
                        | "div"
                        | "rfloor"
                        | "rceil"
                )
            }
            Some((LatexToken::LParen, _)) => true,
            Some((LatexToken::LBrace, _)) => true,
            _ => false,
        }
    }

    /// Parses power expressions (^) and subscripts (_).
    ///
    /// Note: When the base is Euler's number `e` (Constant(E)), the expression
    /// `e^x` is normalized to `exp(x)` for consistency with `\exp{x}`.
    fn parse_power(&mut self) -> ParseResult<Expression> {
        let mut base = self.parse_postfix()?;

        // Handle superscript (power)
        if self.check(&LatexToken::Caret) {
            self.next(); // consume ^
            let exponent = self.parse_braced_or_atom()?;

            // Normalize e^{...} to exp(...)
            if matches!(base, Expression::Constant(MathConstant::E)) {
                return Ok(Expression::Function {
                    name: "exp".to_string(),
                    args: vec![exponent],
                });
            }

            base = Expression::Binary {
                op: BinaryOp::Pow,
                left: Box::new(base),
                right: Box::new(exponent),
            };
        }

        // Handle subscript (append to variable name if base is a variable)
        if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let subscript = self.parse_braced_or_atom()?;

            // Convert base to variable with subscript
            base = match base {
                Expression::Variable(name) => {
                    // Format: var_subscript
                    let subscript_str = self.expression_to_subscript_string(&subscript)?;
                    Expression::Variable(format!("{}_{}", name, subscript_str))
                }
                _ => {
                    return Err(ParseError::invalid_subscript(
                        "subscript can only be applied to variables",
                        Some(self.current_span()),
                    ));
                }
            };
        }

        Ok(base)
    }

    /// Parses postfix expressions (currently just primary, extensible for factorial, etc.).
    fn parse_postfix(&mut self) -> ParseResult<Expression> {
        self.parse_primary()
    }

    /// Parses a letter token, handling differential detection and e/i constant resolution.
    fn parse_letter_token(&mut self, ch: char, _span: Span) -> ParseResult<Expression> {
        // Detect `d<var>` as a differential outside integral/fraction context
        if ch == 'd' && !self.in_integral_context && !self.in_fraction_context {
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek_ahead(1) {
                let var_ch = *var_ch;
                self.next(); // consume 'd'
                self.next(); // consume variable letter
                return Ok(Expression::Differential {
                    var: var_ch.to_string(),
                });
            }
        }
        self.next();
        if ch == 'e' || ch == 'i' {
            Ok(self.resolve_letter(ch, false))
        } else {
            Ok(Expression::Variable(ch.to_string()))
        }
    }

    /// Parses a number-set token (\mathbb{N/Z/Q/R/C/H}).
    fn parse_number_set_token(&mut self, token: &LatexToken) -> Option<Expression> {
        use crate::ast::NumberSet;
        let set = match token {
            LatexToken::Naturals => NumberSet::Natural,
            LatexToken::Integers => NumberSet::Integer,
            LatexToken::Rationals => NumberSet::Rational,
            LatexToken::Reals => NumberSet::Real,
            LatexToken::Complexes => NumberSet::Complex,
            LatexToken::Quaternions => NumberSet::Quaternion,
            _ => return None,
        };
        self.next();
        Some(Expression::NumberSetExpr(set))
    }

    /// Parses vector notation, nabla, multiple/closed integrals, and matrix environments.
    /// Returns `None` when the token does not match any of those forms.
    fn parse_vector_integral_token(
        &mut self,
        token: &LatexToken,
        span: Span,
    ) -> ParseResult<Option<Expression>> {
        let expr = match token {
            LatexToken::Nabla => {
                self.next();
                self.parse_nabla()?
            }
            LatexToken::Mathbf | LatexToken::Boldsymbol => {
                self.next();
                self.parse_marked_vector(VectorNotation::Bold)?
            }
            LatexToken::Vec | LatexToken::Overrightarrow => {
                self.next();
                self.parse_marked_vector(VectorNotation::Arrow)?
            }
            LatexToken::Hat => {
                self.next();
                self.parse_marked_vector(VectorNotation::Hat)?
            }
            LatexToken::Underline => {
                self.next();
                self.parse_marked_vector(VectorNotation::Underline)?
            }
            LatexToken::DoubleIntegral => {
                self.next();
                self.parse_multiple_integral(2)?
            }
            LatexToken::TripleIntegral => {
                self.next();
                self.parse_multiple_integral(3)?
            }
            LatexToken::QuadIntegral => {
                self.next();
                self.parse_multiple_integral(4)?
            }
            LatexToken::ClosedIntegral => {
                self.next();
                self.parse_closed_integral(1)?
            }
            LatexToken::ClosedSurface => {
                self.next();
                self.parse_closed_integral(2)?
            }
            LatexToken::ClosedVolume => {
                self.next();
                self.parse_closed_integral(3)?
            }
            LatexToken::BeginEnv(env_name) => {
                let env_name = env_name.clone();
                self.next();
                self.parse_matrix_environment(&env_name)?
            }
            _ => return Ok(None),
        };
        let _ = span; // span available for future error reporting
        Ok(Some(expr))
    }

    /// Parses set/logic atoms, quantifiers, power-set, number sets, vectors, and integrals.
    fn parse_primary_extended(
        &mut self,
        token: &LatexToken,
        span: Span,
    ) -> ParseResult<Expression> {
        if let Some(expr) = self.parse_number_set_token(token) {
            return Ok(expr);
        }

        if let Some(expr) = self.parse_vector_integral_token(token, span)? {
            return Ok(expr);
        }

        match token {
            LatexToken::PowerSet => {
                self.next();
                let set = if self.check(&LatexToken::LParen) {
                    self.next();
                    let expr = self.parse_expression()?;
                    self.consume(LatexToken::RParen)?;
                    expr
                } else if self.check(&LatexToken::LBrace) {
                    self.braced(|p| p.parse_expression())?
                } else {
                    self.parse_power()?
                };
                Ok(Expression::PowerSet { set: Box::new(set) })
            }
            LatexToken::EmptySet => {
                self.next();
                Ok(Expression::EmptySet)
            }
            LatexToken::Lnot => {
                self.next();
                let operand = self.parse_power()?;
                Ok(Expression::Logical {
                    op: crate::ast::LogicalOp::Not,
                    operands: vec![operand],
                })
            }
            LatexToken::ForAll => {
                self.next();
                self.parse_forall()
            }
            LatexToken::Exists => {
                self.next();
                self.parse_exists()
            }
            _ => Err(ParseError::unexpected_token(
                vec!["expression"],
                format!("{:?}", token),
                Some(span),
            )),
        }
    }

    /// Parses primary expressions (atoms, commands, parenthesized expressions).
    fn parse_primary(&mut self) -> ParseResult<Expression> {
        // Clone both token and span upfront so the immutable borrow on `self` ends
        // before any `&mut self` call below.
        let (token, span) = match self.peek() {
            Some((t, s)) => (t.clone(), *s),
            None => {
                return Err(ParseError::unexpected_eof(
                    vec!["expression"],
                    Some(self.current_span()),
                ))
            }
        };

        match token {
            LatexToken::Number(ref num_str) => {
                let num_str = num_str.clone();
                self.next();
                self.parse_number(&num_str, span)
            }
            LatexToken::Letter(ch) => self.parse_letter_token(ch, span),
            LatexToken::ExplicitConstant(ch) => {
                self.next();
                Ok(self.resolve_letter(ch, true))
            }
            LatexToken::Command(ref cmd) => {
                let cmd = cmd.clone();
                self.next();
                self.parse_command(&cmd, span)
            }
            LatexToken::LParen => {
                self.next();
                let expr = self.parse_expression()?;
                self.consume(LatexToken::RParen)?;
                Ok(expr)
            }
            LatexToken::LBrace => self.braced(|parser| parser.parse_expression()),
            LatexToken::Pipe => {
                self.next();
                let expr = self.parse_expression()?;
                self.consume(LatexToken::Pipe)?;
                Ok(Expression::Function {
                    name: "abs".to_string(),
                    args: vec![expr],
                })
            }
            LatexToken::Minus => {
                self.next();
                let operand = self.parse_power()?;
                if matches!(operand, Expression::Constant(MathConstant::Infinity)) {
                    Ok(Expression::Constant(MathConstant::NegInfinity))
                } else {
                    Ok(Expression::Unary {
                        op: crate::ast::UnaryOp::Neg,
                        operand: Box::new(operand),
                    })
                }
            }
            LatexToken::Plus => {
                self.next();
                let operand = self.parse_power()?;
                Ok(Expression::Unary {
                    op: crate::ast::UnaryOp::Pos,
                    operand: Box::new(operand),
                })
            }
            LatexToken::Infty => {
                self.next();
                Ok(Expression::Constant(MathConstant::Infinity))
            }
            // All remaining tokens are handled by the extended helper.
            // Pass the already-cloned token to avoid re-borrowing self.
            ref t => {
                let t = t.clone();
                self.parse_primary_extended(&t, span)
            }
        }
    }

    /// Parses a number (integer or float).
    fn parse_number(&self, num_str: &str, span: Span) -> ParseResult<Expression> {
        if num_str.contains('.') {
            // Float
            num_str
                .parse::<f64>()
                .map(|f| Expression::Float(MathFloat::from(f)))
                .map_err(|_| ParseError::invalid_number(num_str, "invalid float", Some(span)))
        } else {
            // Integer
            num_str
                .parse::<i64>()
                .map(Expression::Integer)
                .map_err(|_| ParseError::invalid_number(num_str, "invalid integer", Some(span)))
        }
    }

    /// Parses \frac{num}{denom}, promoting to a derivative when the pattern matches.
    fn parse_frac_command(&mut self) -> ParseResult<Expression> {
        self.in_fraction_context = true;
        let numerator = self.braced(|p| p.parse_expression())?;
        let denominator = self.braced(|p| p.parse_expression())?;
        self.in_fraction_context = false;

        if let Some(deriv) = self.try_parse_derivative(numerator.clone(), denominator.clone())? {
            return Ok(deriv);
        }

        Ok(Expression::Binary {
            op: BinaryOp::Div,
            left: Box::new(numerator),
            right: Box::new(denominator),
        })
    }

    /// Parses \log (with optional base subscript) or \ln.
    fn parse_log_command(&mut self, is_log: bool) -> ParseResult<Expression> {
        if is_log && self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let base = self.parse_braced_or_atom()?;
            let arg = self.parse_function_arg()?;
            return Ok(Expression::Function {
                name: "log".to_string(),
                args: vec![arg, base],
            });
        }
        let arg = self.parse_function_arg()?;
        Ok(Expression::Function {
            name: if is_log { "log" } else { "ln" }.to_string(),
            args: vec![arg],
        })
    }

    /// Parses \lfloor expr \rfloor or \lceil expr \rceil.
    fn parse_floor_ceil_command(&mut self, is_floor: bool) -> ParseResult<Expression> {
        let expr = self.parse_expression()?;
        let (close_cmd, fn_name, err_msg) = if is_floor {
            ("rfloor", "floor", "expected \\rfloor after \\lfloor")
        } else {
            ("rceil", "ceil", "expected \\rceil after \\lceil")
        };
        if let Some((LatexToken::Command(c), _)) = self.peek() {
            if c == close_cmd {
                self.next();
                return Ok(Expression::Function {
                    name: fn_name.to_string(),
                    args: vec![expr],
                });
            }
        }
        Err(ParseError::custom(
            err_msg.to_string(),
            Some(self.current_span()),
        ))
    }

    /// Parses \delta (Kronecker) or \varepsilon / \epsilon (Levi-Civita).
    /// Falls back to a plain variable when no tensor indices follow.
    fn parse_tensor_symbol_command(&mut self, cmd: &str) -> ParseResult<Expression> {
        if self.looks_like_tensor_index() {
            let indices = self.parse_tensor_indices()?;
            if !indices.is_empty() {
                return Ok(if cmd == "delta" {
                    Expression::KroneckerDelta { indices }
                } else {
                    Expression::LeviCivita { indices }
                });
            }
        }
        Ok(Expression::Variable(cmd.to_string()))
    }

    /// Dispatches a LaTeX command to the appropriate sub-parser.
    fn parse_command(&mut self, cmd: &str, span: Span) -> ParseResult<Expression> {
        match cmd {
            "frac" => self.parse_frac_command(),

            "sqrt" => {
                if self.check(&LatexToken::LBracket) {
                    let n = self.bracketed(|p| p.parse_expression())?;
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "root".to_string(),
                        args: vec![x, n],
                    })
                } else {
                    let x = self.braced(|p| p.parse_expression())?;
                    Ok(Expression::Function {
                        name: "sqrt".to_string(),
                        args: vec![x],
                    })
                }
            }

            "delta" | "varepsilon" | "epsilon" => self.parse_tensor_symbol_command(cmd),

            // Greek letters → variables (\pi is a constant)
            "alpha" | "beta" | "gamma" | "zeta" | "eta" | "theta" | "iota" | "kappa" | "lambda"
            | "mu" | "nu" | "xi" | "omicron" | "pi" | "rho" | "sigma" | "tau" | "upsilon"
            | "phi" | "chi" | "psi" | "omega" | "Gamma" | "Delta" | "Theta" | "Lambda" | "Xi"
            | "Pi" | "Sigma" | "Upsilon" | "Phi" | "Psi" | "Omega" => {
                if cmd == "pi" {
                    Ok(Expression::Constant(MathConstant::Pi))
                } else {
                    Ok(Expression::Variable(cmd.to_string()))
                }
            }

            "partial" => Ok(Expression::Variable("partial".to_string())),

            // Single-argument functions
            "sin" | "cos" | "tan" | "sec" | "csc" | "cot" | "arcsin" | "arccos" | "arctan"
            | "sinh" | "cosh" | "tanh" | "exp" | "det" | "min" | "max" | "gcd" | "lcm" | "abs"
            | "floor" | "ceil" | "sgn" => {
                let arg = self.parse_function_arg()?;
                Ok(Expression::Function {
                    name: cmd.to_string(),
                    args: vec![arg],
                })
            }

            "ln" => self.parse_log_command(false),
            "log" => self.parse_log_command(true),
            "lfloor" => self.parse_floor_ceil_command(true),
            "lceil" => self.parse_floor_ceil_command(false),

            "int" => self.parse_integral(),
            "lim" => self.parse_limit(),
            "sum" => self.parse_sum(),
            "prod" => self.parse_product(),

            _ => Err(ParseError::invalid_latex_command(cmd, Some(span))),
        }
    }

    /// Parses a function argument (either braced or a primary expression).
    fn parse_function_arg(&mut self) -> ParseResult<Expression> {
        if self.check(&LatexToken::LBrace) {
            self.braced(|p| p.parse_expression())
        } else if self.check(&LatexToken::LParen) {
            self.next(); // consume (
            let expr = self.parse_expression()?;
            self.consume(LatexToken::RParen)?;
            Ok(expr)
        } else {
            // For LaTeX, functions can take unbraced simple arguments: \sin x
            self.parse_power()
        }
    }

    /// Parses an expression in braces {...} or a single atom.
    fn parse_braced_or_atom(&mut self) -> ParseResult<Expression> {
        if self.check(&LatexToken::LBrace) {
            self.braced(|p| p.parse_expression())
        } else {
            // Parse a single atom (number, letter, etc.)
            self.parse_primary()
        }
    }

    /// Helper: parses content within braces {...}.
    fn braced<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBrace)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBrace)?;
        Ok(result)
    }

    /// Helper: parses content within brackets [...].
    fn bracketed<F, T>(&mut self, parser_fn: F) -> ParseResult<T>
    where
        F: FnOnce(&mut Self) -> ParseResult<T>,
    {
        self.consume(LatexToken::LBracket)?;
        let result = parser_fn(self)?;
        self.consume(LatexToken::RBracket)?;
        Ok(result)
    }

    /// Checks if the next subscript/superscript looks like a tensor index (letters)
    /// rather than a power or regular subscript (numbers/expressions).
    /// This helps distinguish between \delta^i_j (tensor) and \delta^2 (power).
    fn looks_like_tensor_index(&self) -> bool {
        // Must have ^ or _ next
        if !self.check(&LatexToken::Caret) && !self.check(&LatexToken::Underscore) {
            return false;
        }

        // Look at what follows ^ or _
        // We need to peek 2 tokens ahead
        let next_pos = self.pos + 1;
        if let Some((token, _)) = self.tokens.get(next_pos) {
            match token {
                // Single letter index: ^i or _j
                LatexToken::Letter(_) => true,
                // Greek letter index: ^\mu or _\nu
                LatexToken::Command(_) => true,
                // Braced group: ^{ij} or _{kl} - check first char inside braces
                LatexToken::LBrace => {
                    // Look inside the braces
                    if let Some((inner, _)) = self.tokens.get(next_pos + 1) {
                        matches!(inner, LatexToken::Letter(_) | LatexToken::Command(_))
                    } else {
                        false
                    }
                }
                // Number means this is a power, not tensor index
                LatexToken::Number(_) => false,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Parses tensor indices from the current position.
    /// Handles patterns like ^{ij}_{kl}, ^i_j, _{ij}, etc.
    /// Returns a vector of TensorIndex with the appropriate index types.
    fn parse_tensor_indices(&mut self) -> ParseResult<Vec<TensorIndex>> {
        let mut indices = Vec::new();

        // Parse upper indices (^{...} or ^x)
        if self.check(&LatexToken::Caret) {
            self.next(); // consume ^
            let upper_indices = self.parse_index_group(IndexType::Upper)?;
            indices.extend(upper_indices);
        }

        // Parse lower indices (_{...} or _x)
        if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let lower_indices = self.parse_index_group(IndexType::Lower)?;
            indices.extend(lower_indices);
        }

        // Handle mixed notation: T^i_j^k (rare but valid)
        // Check for additional upper indices after lower
        if self.check(&LatexToken::Caret) {
            self.next();
            let more_upper = self.parse_index_group(IndexType::Upper)?;
            indices.extend(more_upper);
        }

        Ok(indices)
    }

    /// Parses a group of indices (either braced or single character).
    /// Returns a vector of TensorIndex all with the specified index type.
    fn parse_index_group(&mut self, index_type: IndexType) -> ParseResult<Vec<TensorIndex>> {
        let mut indices = Vec::new();

        if self.check(&LatexToken::LBrace) {
            // Braced group: ^{ij} or _{kl}
            self.next(); // consume {

            // Parse letters inside braces until we hit }
            while !self.check(&LatexToken::RBrace) && !self.check(&LatexToken::Eof) {
                match self.peek() {
                    Some((LatexToken::Letter(ch), _)) => {
                        let ch = *ch;
                        self.next();
                        indices.push(TensorIndex {
                            name: ch.to_string(),
                            index_type,
                        });
                    }
                    Some((LatexToken::Command(cmd), _)) => {
                        // Handle Greek letter indices like μ, ν
                        let cmd = cmd.clone();
                        self.next();
                        indices.push(TensorIndex {
                            name: cmd,
                            index_type,
                        });
                    }
                    Some((_, span)) => {
                        return Err(ParseError::custom(
                            "expected letter in tensor index".to_string(),
                            Some(*span),
                        ));
                    }
                    None => {
                        return Err(ParseError::unexpected_eof(
                            vec!["tensor index"],
                            Some(self.current_span()),
                        ));
                    }
                }
            }

            self.consume(LatexToken::RBrace)?;
        } else {
            // Single character: ^i or _j
            match self.peek() {
                Some((LatexToken::Letter(ch), _)) => {
                    let ch = *ch;
                    self.next();
                    indices.push(TensorIndex {
                        name: ch.to_string(),
                        index_type,
                    });
                }
                Some((LatexToken::Command(cmd), _)) => {
                    // Greek letter index
                    let cmd = cmd.clone();
                    self.next();
                    indices.push(TensorIndex {
                        name: cmd,
                        index_type,
                    });
                }
                Some((_, span)) => {
                    return Err(ParseError::custom(
                        "expected letter in tensor index".to_string(),
                        Some(*span),
                    ));
                }
                None => {
                    return Err(ParseError::unexpected_eof(
                        vec!["tensor index"],
                        Some(self.current_span()),
                    ));
                }
            }
        }

        Ok(indices)
    }

    /// Converts an expression to a subscript string representation.
    /// For complex expressions, creates a flattened representation suitable for variable names.
    fn expression_to_subscript_string(&self, expr: &Expression) -> ParseResult<String> {
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
    fn match_derivative_numerator(expr: &Expression) -> Option<(bool, u32)> {
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
    fn match_derivative_denominator(expr: &Expression) -> Option<(bool, String, u32)> {
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
    fn try_parse_derivative(
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

    /// Parses an integral: \int f(x) dx or \int_a^b f(x) dx
    fn parse_integral(&mut self) -> ParseResult<Expression> {
        // Check for subscript (lower bound)
        let bounds = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let lower = self.parse_braced_or_atom()?;

            // Must have superscript (upper bound) if we have subscript
            if !self.check(&LatexToken::Caret) {
                return Err(ParseError::custom(
                    "integral with lower bound must also have upper bound".to_string(),
                    Some(self.current_span()),
                ));
            }
            self.next(); // consume ^
            let upper = self.parse_braced_or_atom()?;

            Some(IntegralBounds {
                lower: Box::new(lower),
                upper: Box::new(upper),
            })
        } else if self.check(&LatexToken::Caret) {
            // Upper bound without lower bound is an error
            return Err(ParseError::custom(
                "integral with upper bound must also have lower bound".to_string(),
                Some(self.current_span()),
            ));
        } else {
            None
        };

        // Parse integrand - use multiplicative level so x + 1 parses as (int x) + 1
        // For greedy parsing like \int x + 1 dx, parentheses are required: \int (x + 1) dx
        // Set integral context to prevent 'dx' from being parsed as a Differential
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Expect 'd' followed by variable name
        if let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'

            // Next should be the variable
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                let var = var_ch.to_string();
                self.next(); // consume variable

                Ok(Expression::Integral {
                    integrand: Box::new(integrand),
                    var,
                    bounds,
                })
            } else {
                Err(ParseError::custom(
                    "expected variable name after 'd' in integral".to_string(),
                    Some(self.current_span()),
                ))
            }
        } else {
            Err(ParseError::custom(
                "expected 'd' followed by variable in integral".to_string(),
                Some(self.current_span()),
            ))
        }
    }

    /// Parses a multiple integral: \iint, \iiint, \iiiint
    /// Format: \iint f(x,y) dy dx or \iint_{bounds} f(x,y) dy dx
    fn parse_multiple_integral(&mut self, dimension: u8) -> ParseResult<Expression> {
        // Check for optional bounds subscript
        let bounds = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let _region = self.parse_braced_or_atom()?;
            // For simplicity, we don't track the region expression
            // In full implementation, this could be enhanced to store bounds
            None
        } else {
            None
        };

        // Parse integrand with integral context set
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Parse the differential variables (dy dx, dz dy dx, etc.)
        let mut vars = Vec::new();
        while let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                vars.push(var_ch.to_string());
                self.next(); // consume variable
            } else {
                return Err(ParseError::custom(
                    "expected variable name after 'd' in multiple integral".to_string(),
                    Some(self.current_span()),
                ));
            }
        }

        if vars.is_empty() {
            return Err(ParseError::custom(
                format!(
                    "expected {} differential variables for {}-dimensional integral",
                    dimension,
                    match dimension {
                        2 => "double",
                        3 => "triple",
                        4 => "quadruple",
                        _ => "multiple",
                    }
                ),
                Some(self.current_span()),
            ));
        }

        Ok(Expression::MultipleIntegral {
            dimension,
            integrand: Box::new(integrand),
            bounds,
            vars,
        })
    }

    /// Parses a closed/contour integral: \oint, \oiint, \oiiint
    /// Format: \oint_C f(x) dx or \oiint_S f(x,y) dA
    fn parse_closed_integral(&mut self, dimension: u8) -> ParseResult<Expression> {
        // Check for optional surface/curve subscript
        let surface = if self.check(&LatexToken::Underscore) {
            self.next(); // consume _
            let surface_expr = self.parse_braced_or_atom()?;
            // Extract name if it's a variable
            match surface_expr {
                Expression::Variable(name) => Some(name),
                _ => None,
            }
        } else {
            None
        };

        // Parse integrand with integral context set
        self.in_integral_context = true;
        let integrand = self.parse_multiplicative()?;
        self.in_integral_context = false;

        // Parse the differential variable
        let var = if let Some((LatexToken::Letter('d'), _)) = self.peek() {
            self.next(); // consume 'd'
            if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                let v = var_ch.to_string();
                self.next(); // consume variable
                v
            } else {
                // Could be dA, dS, etc. - check for uppercase
                if let Some((LatexToken::Letter(var_ch), _)) = self.peek() {
                    let v = var_ch.to_string();
                    self.next();
                    v
                } else {
                    "".to_string() // Will error below
                }
            }
        } else {
            "".to_string()
        };

        if var.is_empty() {
            return Err(ParseError::custom(
                "expected differential variable in closed integral".to_string(),
                Some(self.current_span()),
            ));
        }

        Ok(Expression::ClosedIntegral {
            dimension,
            integrand: Box::new(integrand),
            surface,
            var,
        })
    }

    // ============================================================
    // Quantifier Parsing
    // ============================================================

    /// Parses a universal quantifier: \forall x P(x) or \forall x \in S P(x)
    fn parse_forall(&mut self) -> ParseResult<Expression> {
        // Expect variable
        let variable = match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                let v = ch.to_string();
                self.next();
                v
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Greek letter variable
                let v = cmd.clone();
                self.next();
                v
            }
            _ => {
                return Err(ParseError::custom(
                    "expected variable after \\forall".to_string(),
                    Some(self.current_span()),
                ));
            }
        };

        // Check for optional domain: \in S
        let domain = if let Some((LatexToken::In, _)) = self.peek() {
            self.next(); // consume \in
            let set = self.parse_power()?;
            Some(Box::new(set))
        } else {
            None
        };

        // Parse the body expression
        let body = self.parse_expression()?;

        Ok(Expression::ForAll {
            variable,
            domain,
            body: Box::new(body),
        })
    }

    /// Parses an existential quantifier: \exists x P(x) or \exists! x P(x)
    fn parse_exists(&mut self) -> ParseResult<Expression> {
        // Check for unique existence: \exists!
        let unique = if let Some((LatexToken::Command(cmd), _)) = self.peek() {
            if cmd == "!" {
                self.next(); // consume !
                true
            } else {
                false
            }
        } else {
            false
        };

        // Expect variable
        let variable = match self.peek() {
            Some((LatexToken::Letter(ch), _)) => {
                let v = ch.to_string();
                self.next();
                v
            }
            Some((LatexToken::Command(cmd), _)) => {
                // Greek letter variable
                let v = cmd.clone();
                self.next();
                v
            }
            _ => {
                return Err(ParseError::custom(
                    "expected variable after \\exists".to_string(),
                    Some(self.current_span()),
                ));
            }
        };

        // Check for optional domain: \in S
        let domain = if let Some((LatexToken::In, _)) = self.peek() {
            self.next(); // consume \in
            let set = self.parse_power()?;
            Some(Box::new(set))
        } else {
            None
        };

        // Parse the body expression
        let body = self.parse_expression()?;

        Ok(Expression::Exists {
            variable,
            domain,
            body: Box::new(body),
            unique,
        })
    }

    /// Parses a limit: \lim_{x \to a} or \lim_{x \to a^+}
    fn parse_limit(&mut self) -> ParseResult<Expression> {
        // Expect subscript with pattern: var \to value
        if !self.check(&LatexToken::Underscore) {
            return Err(ParseError::custom(
                "limit must have subscript with approach pattern".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume _

        // Parse the subscript content
        self.consume(LatexToken::LBrace)?;

        // Expect variable
        let var = if let Some((LatexToken::Letter(ch), _)) = self.peek() {
            let v = ch.to_string();
            self.next(); // consume variable
            v
        } else {
            return Err(ParseError::custom(
                "expected variable in limit subscript".to_string(),
                Some(self.current_span()),
            ));
        };

        // Expect \to
        if let Some((LatexToken::To, _)) = self.peek() {
            self.next(); // consume \to
        } else {
            return Err(ParseError::custom(
                "expected \\to in limit subscript".to_string(),
                Some(self.current_span()),
            ));
        }

        // Parse approach value (just primary for now, will be a number, variable, or constant)
        let to = self.parse_primary()?;

        // Check for direction (^+ or ^-) before the closing brace
        let direction = if self.check(&LatexToken::Caret) {
            self.next(); // consume ^

            match self.peek() {
                Some((LatexToken::Plus, _)) => {
                    self.next();
                    Direction::Right
                }
                Some((LatexToken::Minus, _)) => {
                    self.next();
                    Direction::Left
                }
                _ => {
                    return Err(ParseError::custom(
                        "expected + or - after ^ in limit direction".to_string(),
                        Some(self.current_span()),
                    ));
                }
            }
        } else {
            Direction::Both
        };

        self.consume(LatexToken::RBrace)?;

        // Parse the expression - use parse_multiplicative to capture full expressions
        let expr = self.parse_multiplicative()?;

        Ok(Expression::Limit {
            expr: Box::new(expr),
            var,
            to: Box::new(to),
            direction,
        })
    }

    /// Parses a sum: \sum_{i=1}^{n} expr
    fn parse_sum(&mut self) -> ParseResult<Expression> {
        let (index, lower, upper) = self.parse_iterator_bounds()?;
        // Bind the index variable in scope while parsing the body
        self.push_scope(std::iter::once(index.clone()));
        let body = self.parse_multiplicative()?;
        self.pop_scope();

        Ok(Expression::Sum {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Parses a product: \prod_{i=1}^{n} expr
    fn parse_product(&mut self) -> ParseResult<Expression> {
        let (index, lower, upper) = self.parse_iterator_bounds()?;
        // Bind the index variable in scope while parsing the body
        self.push_scope(std::iter::once(index.clone()));
        let body = self.parse_multiplicative()?;
        self.pop_scope();

        Ok(Expression::Product {
            index,
            lower: Box::new(lower),
            upper: Box::new(upper),
            body: Box::new(body),
        })
    }

    /// Helper to parse iterator bounds: _{var=lower}^{upper}
    fn parse_iterator_bounds(&mut self) -> ParseResult<(String, Expression, Expression)> {
        // Expect subscript with pattern: var = value
        if !self.check(&LatexToken::Underscore) {
            return Err(ParseError::custom(
                "iterator must have subscript with index=lower pattern".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume _

        // Parse the subscript content
        self.consume(LatexToken::LBrace)?;

        // Expect variable
        let index = if let Some((LatexToken::Letter(ch), _)) = self.peek() {
            let v = ch.to_string();
            self.next(); // consume variable
            v
        } else {
            return Err(ParseError::custom(
                "expected index variable in iterator subscript".to_string(),
                Some(self.current_span()),
            ));
        };

        // Expect =
        if let Some((LatexToken::Equals, _)) = self.peek() {
            self.next(); // consume =
        } else {
            return Err(ParseError::custom(
                "expected = in iterator subscript".to_string(),
                Some(self.current_span()),
            ));
        }

        // Parse lower bound
        let lower = self.parse_additive()?;

        self.consume(LatexToken::RBrace)?;

        // Expect superscript with upper bound
        if !self.check(&LatexToken::Caret) {
            return Err(ParseError::custom(
                "iterator must have superscript with upper bound".to_string(),
                Some(self.current_span()),
            ));
        }
        self.next(); // consume ^

        let upper = self.parse_braced_or_atom()?;

        Ok((index, lower, upper))
    }

    // ============================================================
    // Vector Notation Parsing
    // ============================================================

    /// Parses a marked vector: \mathbf{v}, \vec{a}, \hat{n}, \underline{u}
    /// Returns MarkedVector with the given notation style.
    ///
    /// Special case: \mathbf{j} and \mathbf{k} return quaternion basis constants
    /// MathConstant::J and MathConstant::K respectively.
    fn parse_marked_vector(&mut self, notation: VectorNotation) -> ParseResult<Expression> {
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
    fn parse_vector_name(&mut self) -> ParseResult<String> {
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
    fn parse_nabla(&mut self) -> ParseResult<Expression> {
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
    fn parse_matrix_rows(&mut self, env_name: &str) -> ParseResult<Vec<Vec<Expression>>> {
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
    fn validate_matrix_rows(rows: &[Vec<Expression>], span: crate::error::Span) -> ParseResult<()> {
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
    fn parse_matrix_environment(&mut self, env_name: &str) -> ParseResult<Expression> {
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

#[cfg(test)]
#[path = "latex/tests/latex_tests_greek.rs"]
mod greek_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_fractions.rs"]
mod fractions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_roots.rs"]
#[allow(clippy::approx_constant)]
mod roots_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_powers_subscripts.rs"]
mod powers_subscripts_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_functions.rs"]
#[allow(clippy::approx_constant)]
mod functions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_calculus.rs"]
mod calculus_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_constants.rs"]
mod constants_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_errors.rs"]
mod errors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_tensors.rs"]
mod tensors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_vectors.rs"]
mod vectors_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_multiple_integrals.rs"]
mod multiple_integrals_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_logic.rs"]
mod logic_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_sets.rs"]
mod sets_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_quaternions.rs"]
mod quaternions_tests;

#[cfg(test)]
#[path = "latex/tests/latex_tests_differential_forms.rs"]
mod differential_forms_tests;

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_number() {
        let expr = parse_latex("42").unwrap();
        assert_eq!(expr, Expression::Integer(42));
    }

    #[test]
    fn test_parse_float() {
        let expr = parse_latex("3.14").unwrap();
        match expr {
            Expression::Float(f) => {
                assert!((f.value() - 3.14).abs() < 1e-10);
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_latex("x").unwrap();
        assert_eq!(expr, Expression::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_latex("1 + 2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse_latex("5 - 3").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Sub);
                assert_eq!(*left, Expression::Integer(5));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse_latex("2 * 3").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Mul);
                assert_eq!(*left, Expression::Integer(2));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_division() {
        let expr = parse_latex("6 / 2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Div);
                assert_eq!(*left, Expression::Integer(6));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_power() {
        let expr = parse_latex("x^2").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Pow);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_power_braced() {
        let expr = parse_latex("x^{2+3}").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Add, ..
                    } => {}
                    _ => panic!("Expected addition in exponent"),
                }
            }
            _ => panic!("Expected power expression"),
        }
    }

    #[test]
    fn test_parse_frac() {
        let expr = parse_latex(r"\frac{1}{2}").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Div);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary division"),
        }
    }

    #[test]
    fn test_parse_sqrt() {
        let expr = parse_latex(r"\sqrt{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sqrt_nth() {
        let expr = parse_latex(r"\sqrt[3]{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "root");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
                assert_eq!(args[1], Expression::Integer(3));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_subscript() {
        let expr = parse_latex("x_1").unwrap();
        assert_eq!(expr, Expression::Variable("x_1".to_string()));
    }

    #[test]
    fn test_parse_subscript_braced() {
        let expr = parse_latex("x_{12}").unwrap();
        assert_eq!(expr, Expression::Variable("x_12".to_string()));
    }

    #[test]
    fn test_parse_greek_letter() {
        let expr = parse_latex(r"\alpha").unwrap();
        assert_eq!(expr, Expression::Variable("alpha".to_string()));
    }

    #[test]
    fn test_parse_pi_constant() {
        let expr = parse_latex(r"\pi").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Pi));
    }

    #[test]
    fn test_parse_infinity() {
        let expr = parse_latex(r"\infty").unwrap();
        assert_eq!(expr, Expression::Constant(MathConstant::Infinity));
    }

    #[test]
    fn test_parse_sin() {
        let expr = parse_latex(r"\sin{x}").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sin_unbraced() {
        let expr = parse_latex(r"\sin x").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_sin_parentheses() {
        let expr = parse_latex(r"\sin(x)").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "sin");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse_latex("(1 + 2)").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*left, Expression::Integer(1));
                assert_eq!(*right, Expression::Integer(2));
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_parse_absolute_value() {
        let expr = parse_latex("|x|").unwrap();
        match expr {
            Expression::Function { name, args } => {
                assert_eq!(name, "abs");
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_parse_unary_minus() {
        let expr = parse_latex("-x").unwrap();
        match expr {
            Expression::Unary { op, operand } => {
                assert_eq!(op, crate::ast::UnaryOp::Neg);
                assert_eq!(*operand, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected unary expression"),
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        // (2 + 3) * 4
        let expr = parse_latex("(2 + 3) * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Add, ..
                    } => {}
                    _ => panic!("Expected addition in left"),
                }
                assert_eq!(*right, Expression::Integer(4));
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_operator_precedence() {
        // 2 + 3 * 4 should be 2 + (3 * 4)
        let expr = parse_latex("2 + 3 * 4").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Mul, ..
                    } => {}
                    _ => panic!("Expected multiplication in right"),
                }
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_power_precedence() {
        // 2 * x^3 should be 2 * (x^3)
        let expr = parse_latex("2 * x^3").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert_eq!(*left, Expression::Integer(2));
                match *right {
                    Expression::Binary {
                        op: BinaryOp::Pow, ..
                    } => {}
                    _ => panic!("Expected power in right"),
                }
            }
            _ => panic!("Expected multiplication"),
        }
    }

    #[test]
    fn test_nested_frac() {
        // \frac{\frac{1}{2}}{3}
        let expr = parse_latex(r"\frac{\frac{1}{2}}{3}").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Div,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Div, ..
                    } => {}
                    _ => panic!("Expected nested division"),
                }
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected division"),
        }
    }

    // Relation tests

    #[test]
    fn test_latex_simple_equation() {
        let expr = parse_latex("x = 5").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_latex_inequality_less() {
        let expr = parse_latex("x < 5").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Lt);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_less_command() {
        let expr = parse_latex(r"x \lt 5").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Lt);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_greater() {
        let expr = parse_latex("x > 0").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Gt);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_greater_command() {
        let expr = parse_latex(r"x \gt 0").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Gt);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_leq() {
        let expr = parse_latex(r"x \leq 3").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Le);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_le() {
        let expr = parse_latex(r"x \le 3").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Le);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_geq() {
        let expr = parse_latex(r"x \geq -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_ge() {
        let expr = parse_latex(r"x \ge -1").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ge);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_neq() {
        let expr = parse_latex(r"x \neq 0").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Ne);
                assert_eq!(*left, Expression::Variable("x".to_string()));
                assert_eq!(*right, Expression::Integer(0));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_inequality_ne() {
        let expr = parse_latex(r"a \ne b").unwrap();
        match expr {
            Expression::Inequality { op, .. } => {
                assert_eq!(op, InequalityOp::Ne);
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_complex_equation() {
        // \frac{x}{2} = 3
        let expr = parse_latex(r"\frac{x}{2} = 3").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    #[test]
    fn test_latex_complex_inequality() {
        // a + b < c + d
        let expr = parse_latex("a + b < c + d").unwrap();
        match expr {
            Expression::Inequality { op, left, right } => {
                assert_eq!(op, InequalityOp::Lt);
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("Expected Inequality variant"),
        }
    }

    #[test]
    fn test_latex_chained_relation_error() {
        // a < b < c should error
        let result = parse_latex("a < b < c");
        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("chained relations"));
        }
    }

    #[test]
    fn test_latex_relation_precedence() {
        // 2 + 3 = 5 should parse as (2 + 3) = 5
        let expr = parse_latex("2 + 3 = 5").unwrap();
        match expr {
            Expression::Equation { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert_eq!(*right, Expression::Integer(5));
            }
            _ => panic!("Expected Equation variant"),
        }
    }

    // Matrix and vector tests
    #[test]
    fn test_parse_empty_matrix() {
        let expr = parse_latex(r"\begin{matrix}\end{matrix}").unwrap();
        assert_eq!(expr, Expression::Matrix(vec![]));
    }

    #[test]
    fn test_parse_1x1_matrix() {
        let expr = parse_latex(r"\begin{matrix}1\end{matrix}").unwrap();
        assert_eq!(expr, Expression::Vector(vec![Expression::Integer(1)]));
    }

    #[test]
    fn test_parse_column_vector() {
        let expr = parse_latex(r"\begin{matrix}1 \\ 2 \\ 3\end{matrix}").unwrap();
        assert_eq!(
            expr,
            Expression::Vector(vec![
                Expression::Integer(1),
                Expression::Integer(2),
                Expression::Integer(3)
            ])
        );
    }

    #[test]
    fn test_parse_row_matrix() {
        let expr = parse_latex(r"\begin{matrix}1 & 2 & 3\end{matrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 3);
                assert_eq!(rows[0][0], Expression::Integer(1));
                assert_eq!(rows[0][1], Expression::Integer(2));
                assert_eq!(rows[0][2], Expression::Integer(3));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_2x2_matrix() {
        let expr = parse_latex(r"\begin{matrix}1 & 2 \\ 3 & 4\end{matrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
                assert_eq!(rows[1].len(), 2);
                assert_eq!(rows[0][0], Expression::Integer(1));
                assert_eq!(rows[0][1], Expression::Integer(2));
                assert_eq!(rows[1][0], Expression::Integer(3));
                assert_eq!(rows[1][1], Expression::Integer(4));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_bmatrix() {
        let expr = parse_latex(r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_pmatrix() {
        let expr = parse_latex(r"\begin{pmatrix}x & y \\ z & w\end{pmatrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0], Expression::Variable("x".to_string()));
                assert_eq!(rows[0][1], Expression::Variable("y".to_string()));
                assert_eq!(rows[1][0], Expression::Variable("z".to_string()));
                assert_eq!(rows[1][1], Expression::Variable("w".to_string()));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_vmatrix() {
        let expr = parse_latex(r"\begin{vmatrix}a & b \\ c & d\end{vmatrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_big_bmatrix() {
        let expr = parse_latex(r"\begin{Bmatrix}1 & 2\end{Bmatrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 2);
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_big_vmatrix() {
        let expr = parse_latex(r"\begin{Vmatrix}1\end{Vmatrix}").unwrap();
        assert_eq!(expr, Expression::Vector(vec![Expression::Integer(1)]));
    }

    #[test]
    fn test_parse_matrix_with_expressions() {
        let expr = parse_latex(r"\begin{matrix}x+1 & 2 \\ 3 & y^2\end{matrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
                // First element should be x+1
                match &rows[0][0] {
                    Expression::Binary { op, .. } => assert_eq!(*op, BinaryOp::Add),
                    _ => panic!("Expected binary expression"),
                }
                // Last element should be y^2
                match &rows[1][1] {
                    Expression::Binary { op, .. } => assert_eq!(*op, BinaryOp::Pow),
                    _ => panic!("Expected binary expression"),
                }
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_3x3_matrix() {
        let expr = parse_latex(r"\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}")
            .unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 3);
                assert_eq!(rows[0].len(), 3);
                assert_eq!(rows[1].len(), 3);
                assert_eq!(rows[2].len(), 3);
                assert_eq!(rows[2][2], Expression::Integer(9));
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_matrix_trailing_backslash() {
        // Matrix with trailing \\ should not add empty row
        let expr = parse_latex(r"\begin{matrix}1 \\ 2 \\\end{matrix}").unwrap();
        match expr {
            Expression::Vector(elements) => {
                assert_eq!(elements.len(), 2);
                assert_eq!(elements[0], Expression::Integer(1));
                assert_eq!(elements[1], Expression::Integer(2));
            }
            _ => panic!("Expected Vector variant"),
        }
    }

    #[test]
    fn test_parse_ragged_matrix_error() {
        let result = parse_latex(r"\begin{matrix}1 & 2 \\ 3\end{matrix}");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("inconsistent matrix row lengths"));
        }
    }

    #[test]
    fn test_parse_mismatched_environment_error() {
        let result = parse_latex(r"\begin{matrix}1\end{bmatrix}");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("mismatched environment"));
        }
    }

    #[test]
    fn test_parse_invalid_matrix_environment() {
        let result = parse_latex(r"\begin{invalid}1\end{invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_matrix_with_floats() {
        let expr = parse_latex(r"\begin{matrix}1.5 & 2.7 \\ 3.2 & 4.9\end{matrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                match &rows[0][0] {
                    Expression::Float(f) => assert!((f.value() - 1.5).abs() < 1e-10),
                    _ => panic!("Expected float"),
                }
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    #[test]
    fn test_parse_matrix_mixed_types() {
        let expr = parse_latex(r"\begin{pmatrix}1 & x & 2.5 \\ y & 3 & z\end{pmatrix}").unwrap();
        match expr {
            Expression::Matrix(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 3);
                assert_eq!(rows[0][0], Expression::Integer(1));
                assert_eq!(rows[0][1], Expression::Variable("x".to_string()));
                match &rows[0][2] {
                    Expression::Float(f) => assert!((f.value() - 2.5).abs() < 1e-10),
                    _ => panic!("Expected float"),
                }
            }
            _ => panic!("Expected Matrix variant"),
        }
    }

    // Calculus tests

    // Derivative tests
    #[test]
    fn test_parse_derivative_first_order() {
        let expr = parse_latex(r"\frac{d}{d*x}x").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 1);
                assert_eq!(*expr, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected Derivative variant"),
        }
    }

    #[test]
    fn test_parse_derivative_second_order() {
        let expr = parse_latex(r"\frac{d^2}{d*x^2}f").unwrap();
        match expr {
            Expression::Derivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 2);
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected Derivative variant"),
        }
    }

    // Partial derivative tests
    #[test]
    fn test_parse_partial_derivative_first_order() {
        let expr = parse_latex(r"\frac{\partial}{\partial * x}f").unwrap();
        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 1);
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected PartialDerivative variant"),
        }
    }

    #[test]
    fn test_parse_partial_derivative_second_order() {
        let expr = parse_latex(r"\frac{\partial^2}{\partial * x^2}f").unwrap();
        match expr {
            Expression::PartialDerivative { expr, var, order } => {
                assert_eq!(var, "x");
                assert_eq!(order, 2);
                assert_eq!(*expr, Expression::Variable("f".to_string()));
            }
            _ => panic!("Expected PartialDerivative variant"),
        }
    }

    // Test that regular fractions still work
    #[test]
    fn test_parse_frac_not_derivative() {
        let expr = parse_latex(r"\frac{x+1}{y-2}").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Div);
                assert!(matches!(*left, Expression::Binary { .. }));
                assert!(matches!(*right, Expression::Binary { .. }));
            }
            _ => panic!("Expected Binary division"),
        }
    }

    // Integral tests
    #[test]
    fn test_parse_integral_indefinite() {
        let expr = parse_latex(r"\int x dx").unwrap();
        match expr {
            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                assert_eq!(var, "x");
                assert_eq!(*integrand, Expression::Variable("x".to_string()));
                assert!(bounds.is_none());
            }
            _ => panic!("Expected Integral variant"),
        }
    }

    #[test]
    fn test_parse_integral_definite() {
        let expr = parse_latex(r"\int_0^1 x dx").unwrap();
        match expr {
            Expression::Integral {
                integrand,
                var,
                bounds,
            } => {
                assert_eq!(var, "x");
                assert_eq!(*integrand, Expression::Variable("x".to_string()));
                assert!(bounds.is_some());
                let bounds = bounds.unwrap();
                assert_eq!(*bounds.lower, Expression::Integer(0));
                assert_eq!(*bounds.upper, Expression::Integer(1));
            }
            _ => panic!("Expected Integral variant"),
        }
    }

    // Limit tests
    #[test]
    fn test_parse_limit_both_sides() {
        let expr = parse_latex(r"\lim_{x \to 0} x").unwrap();
        match expr {
            Expression::Limit {
                expr,
                var,
                to,
                direction,
            } => {
                assert_eq!(var, "x");
                assert_eq!(*to, Expression::Integer(0));
                assert_eq!(direction, Direction::Both);
                assert_eq!(*expr, Expression::Variable("x".to_string()));
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    #[test]
    fn test_parse_limit_from_right() {
        let expr = parse_latex(r"\lim_{x \to 0^+} x").unwrap();
        match expr {
            Expression::Limit {
                expr: _,
                var,
                to,
                direction,
            } => {
                assert_eq!(var, "x");
                assert_eq!(*to, Expression::Integer(0));
                assert_eq!(direction, Direction::Right);
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    #[test]
    fn test_parse_limit_from_left() {
        let expr = parse_latex(r"\lim_{x \to 0^-} x").unwrap();
        match expr {
            Expression::Limit {
                expr: _,
                var,
                to,
                direction,
            } => {
                assert_eq!(var, "x");
                assert_eq!(*to, Expression::Integer(0));
                assert_eq!(direction, Direction::Left);
            }
            _ => panic!("Expected Limit variant"),
        }
    }

    // Sum tests
    #[test]
    fn test_parse_sum_simple() {
        let expr = parse_latex(r"\sum_{i=1}^{n} i").unwrap();
        match expr {
            Expression::Sum {
                index,
                lower,
                upper,
                body,
            } => {
                assert_eq!(index, "i");
                assert_eq!(*lower, Expression::Integer(1));
                assert_eq!(*upper, Expression::Variable("n".to_string()));
                assert_eq!(*body, Expression::Variable("i".to_string()));
            }
            _ => panic!("Expected Sum variant"),
        }
    }

    // Product tests
    #[test]
    fn test_parse_product_simple() {
        let expr = parse_latex(r"\prod_{i=1}^{n} i").unwrap();
        match expr {
            Expression::Product {
                index,
                lower,
                upper,
                body,
            } => {
                assert_eq!(index, "i");
                assert_eq!(*lower, Expression::Integer(1));
                assert_eq!(*upper, Expression::Variable("n".to_string()));
                assert_eq!(*body, Expression::Variable("i".to_string()));
            }
            _ => panic!("Expected Product variant"),
        }
    }

    // Multiplication command tests
    #[test]
    fn test_parse_cdot_multiplication() {
        let expr = parse_latex(r"a \cdot b").unwrap();
        match expr {
            Expression::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Mul);
                assert_eq!(*left, Expression::Variable("a".to_string()));
                assert_eq!(*right, Expression::Variable("b".to_string()));
            }
            _ => panic!("Expected binary multiplication"),
        }
    }

    #[test]
    fn test_parse_times_cross_product() {
        // \times is parsed as CrossProduct in mathematical notation
        // Consuming libraries can interpret this as scalar multiplication when operands are scalars
        let expr = parse_latex(r"2 \times 3").unwrap();
        match expr {
            Expression::CrossProduct { left, right } => {
                assert_eq!(*left, Expression::Integer(2));
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected cross product, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_cdot_complex_expression() {
        let expr = parse_latex(r"2 \cdot x + 3").unwrap();
        match expr {
            Expression::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                match *left {
                    Expression::Binary {
                        op: BinaryOp::Mul,
                        left: ref l,
                        right: ref r,
                    } => {
                        assert_eq!(**l, Expression::Integer(2));
                        assert_eq!(**r, Expression::Variable("x".to_string()));
                    }
                    _ => panic!("Expected multiplication in left operand"),
                }
                assert_eq!(*right, Expression::Integer(3));
            }
            _ => panic!("Expected addition"),
        }
    }

    #[test]
    fn test_parse_times_with_parentheses() {
        // \times is parsed as CrossProduct
        let expr = parse_latex(r"(a + b) \times (c - d)").unwrap();
        match expr {
            Expression::CrossProduct { left, right } => {
                assert!(matches!(
                    *left,
                    Expression::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(
                    *right,
                    Expression::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
            }
            _ => panic!("Expected cross product, got {:?}", expr),
        }
    }

    #[test]
    fn test_parse_mixed_multiplication_operators() {
        // Test that * and \cdot both work
        let expr1 = parse_latex(r"a * b").unwrap();
        let expr2 = parse_latex(r"a \cdot b").unwrap();

        assert_eq!(expr1, expr2);
    }
}
