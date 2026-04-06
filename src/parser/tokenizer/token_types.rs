//! Token enum, `Spanned<T>`, and SpannedToken type alias.

use crate::error::Span;

/// A token in a mathematical expression.
///
/// Represents the atomic elements of a mathematical expression including
/// literals, operators, delimiters, and special symbols.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    /// Integer literal (e.g., 42, -17)
    Integer(i64),
    /// Floating point literal (e.g., 3.14, 1.5e-3)
    Float(f64),
    /// Identifier/variable name (e.g., x, theta, var_1)
    Identifier(String),

    // Operators
    /// Addition operator (+)
    Plus,
    /// Subtraction operator (-)
    Minus,
    /// Multiplication operator (*)
    Star,
    /// Exponentiation operator (**) - plain text only
    DoubleStar,
    /// Division operator (/)
    Slash,
    /// Exponentiation operator (^)
    Caret,
    /// Modulo operator (%)
    Percent,
    /// Factorial operator (!)
    Bang,

    // Delimiters
    /// Left parenthesis (()
    LParen,
    /// Right parenthesis ())
    RParen,
    /// Left square bracket ([)
    LBracket,
    /// Right square bracket (])
    RBracket,
    /// Left curly brace ({)
    LBrace,
    /// Right curly brace (})
    RBrace,
    /// Comma separator (,)
    Comma,
    /// Semicolon separator (;)
    Semicolon,

    // Relations
    /// Equality (=)
    Equals,
    /// Inequality (!=)
    NotEquals,
    /// Less than (<)
    Less,
    /// Less than or equal (<=)
    LessEq,
    /// Greater than (>)
    Greater,
    /// Greater than or equal (>=)
    GreaterEq,

    // Special
    /// Underscore for subscripts (_)
    Underscore,
    /// End of input
    Eof,

    // Unicode mathematical symbols
    /// Pi constant (π)
    Pi,
    /// Infinity constant (∞)
    Infinity,
    /// Square root symbol (√)
    Sqrt,

    // Keywords for new operations
    /// Dot product keyword
    Dot,
    /// Cross product keyword
    Cross,
    /// Gradient keyword
    Grad,
    /// Divergence keyword
    Div,
    /// Curl keyword
    Curl,
    /// Laplacian keyword
    Laplacian,
    /// Universal quantifier keyword
    ForAll,
    /// Existential quantifier keyword
    Exists,
    /// Set union keyword
    Union,
    /// Set intersection keyword
    Intersect,
    /// Set membership keyword (element in set)
    In,
    /// Set non-membership keyword
    NotIn,
    /// Logical AND keyword
    And,
    /// Logical OR keyword
    Or,
    /// Logical NOT keyword
    Not,
    /// Logical implication keyword
    Implies,
    /// Logical biconditional keyword
    Iff,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Integer(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::DoubleStar => write!(f, "**"),
            Token::Slash => write!(f, "/"),
            Token::Caret => write!(f, "^"),
            Token::Percent => write!(f, "%"),
            Token::Bang => write!(f, "!"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Equals => write!(f, "="),
            Token::NotEquals => write!(f, "!="),
            Token::Less => write!(f, "<"),
            Token::LessEq => write!(f, "<="),
            Token::Greater => write!(f, ">"),
            Token::GreaterEq => write!(f, ">="),
            Token::Underscore => write!(f, "_"),
            Token::Eof => write!(f, "<EOF>"),
            Token::Pi => write!(f, "π"),
            Token::Infinity => write!(f, "∞"),
            Token::Sqrt => write!(f, "√"),
            Token::Dot => write!(f, "dot"),
            Token::Cross => write!(f, "cross"),
            Token::Grad => write!(f, "grad"),
            Token::Div => write!(f, "div"),
            Token::Curl => write!(f, "curl"),
            Token::Laplacian => write!(f, "laplacian"),
            Token::ForAll => write!(f, "forall"),
            Token::Exists => write!(f, "exists"),
            Token::Union => write!(f, "union"),
            Token::Intersect => write!(f, "intersect"),
            Token::In => write!(f, "in"),
            Token::NotIn => write!(f, "notin"),
            Token::And => write!(f, "and"),
            Token::Or => write!(f, "or"),
            Token::Not => write!(f, "not"),
            Token::Implies => write!(f, "implies"),
            Token::Iff => write!(f, "iff"),
        }
    }
}

/// A value with an associated span in the source text.
///
/// This wrapper type attaches position information to tokens for
/// precise error reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    /// The wrapped value
    pub value: T,
    /// The span in the source text
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Creates a new spanned value.
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }
}

/// A token with span information.
pub type SpannedToken = Spanned<Token>;
