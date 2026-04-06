//! LatexToken enum definition.

/// A LaTeX token representing a lexical element in LaTeX math mode.
#[derive(Debug, Clone, PartialEq)]
pub enum LatexToken {
    // Commands
    /// LaTeX command without backslash (e.g., "frac", "sin", "alpha")
    Command(String),

    // Literals
    /// Number literal (will be parsed as int or float later)
    Number(String),
    /// Single letter variable
    Letter(char),
    /// Explicit constant from \mathrm{e}, \mathrm{i}, \imath, \jmath
    ExplicitConstant(char),
    /// NaN constant from \text{NaN}, \text{nan}, or \mathrm{NaN}
    NaNConstant,

    // Operators
    /// Plus operator (+)
    Plus,
    /// Minus operator (-)
    Minus,
    /// Multiplication operator (*)
    Star,
    /// Division operator (/)
    Slash,
    /// Exponentiation operator (^)
    Caret,
    /// Subscript operator (_)
    Underscore,
    /// Equals operator (=)
    Equals,
    /// Less than operator (<)
    Less,
    /// Greater than operator (>)
    Greater,

    // Delimiters
    /// Left brace ({)
    LBrace,
    /// Right brace (})
    RBrace,
    /// Left parenthesis (()
    LParen,
    /// Right parenthesis ())
    RParen,
    /// Left bracket ([)
    LBracket,
    /// Right bracket (])
    RBracket,
    /// Pipe (|) for absolute value
    Pipe,

    // Environment
    /// Begin environment (\begin{name})
    BeginEnv(String),
    /// End environment (\end{name})
    EndEnv(String),
    /// Ampersand (&) for column separator
    Ampersand,
    /// Double backslash (\\) for row separator
    DoubleBackslash,

    // Special
    /// Comma (,)
    Comma,
    /// Colon (:)
    Colon,
    /// Arrow (\to)
    To,
    /// Infinity (\infty)
    Infty,

    // Multiple integrals
    /// \iint - double integral
    DoubleIntegral,
    /// \iiint - triple integral
    TripleIntegral,
    /// \iiiint - quadruple integral (rare)
    QuadIntegral,

    // Closed integrals
    /// \oint - closed line integral
    ClosedIntegral,
    /// \oiint - closed surface integral
    ClosedSurface,
    /// \oiiint - closed volume integral
    ClosedVolume,

    // Quantifiers
    /// \forall - universal quantifier
    ForAll,
    /// \exists - existential quantifier
    Exists,

    // Logical connectives
    /// \land, \wedge - logical AND
    Land,
    /// \lor, \vee - logical OR
    Lor,
    /// \lnot, \neg - logical NOT
    Lnot,
    /// \implies, \Rightarrow - implication
    Implies,
    /// \iff, \Leftrightarrow - if and only if
    Iff,

    // Membership
    /// \in - element of
    In,
    /// \notin - not element of
    NotIn,

    // Number sets (via \mathbb{X})
    /// \mathbb{N} - natural numbers
    Naturals,
    /// \mathbb{Z} - integers
    Integers,
    /// \mathbb{Q} - rationals
    Rationals,
    /// \mathbb{R} - reals
    Reals,
    /// \mathbb{C} - complex numbers
    Complexes,
    /// \mathbb{H} - quaternions
    Quaternions,

    // Set operations
    /// \cup - union
    Cup,
    /// \cap - intersection
    Cap,
    /// \setminus - set difference
    Setminus,
    /// \triangle, \bigtriangleup - symmetric difference
    Triangle,

    // Set relations
    /// \subset - proper subset
    Subset,
    /// \subseteq - subset or equal
    SubsetEq,
    /// \supset - proper superset
    Superset,
    /// \supseteq - superset or equal
    SupersetEq,

    // Set notation
    /// \emptyset, \varnothing - empty set
    EmptySet,
    /// \mid - set builder separator
    SetMid,
    /// \mathcal{P} - power set
    PowerSet,

    // Vector notation
    /// \mathbf{...} - bold vector notation
    Mathbf,
    /// \boldsymbol{...} - bold symbol
    Boldsymbol,
    /// \vec{...} - arrow vector notation
    Vec,
    /// \overrightarrow{...} - long arrow
    Overrightarrow,
    /// \hat{...} - unit vector (hat)
    Hat,
    /// \underline{...} - underline notation
    Underline,

    // Vector/tensor operations
    /// \cdot used as dot product operator
    Cdot,
    /// \bullet - alternative dot
    Bullet,
    /// \otimes - tensor/outer product
    Otimes,
    /// \wedge - wedge product
    Wedge,
    /// \times - cross product (also for curl: nabla x F)
    Cross,

    // Nabla
    /// \nabla - del/nabla operator
    Nabla,

    // Relations
    /// \sim - similarity relation
    Sim,
    /// \equiv - equivalence relation
    Equiv,
    /// \cong - congruence relation
    Cong,
    /// \approx - approximation relation
    Approx,

    // Composition
    /// \circ - function composition
    Circ,

    /// End of file marker
    Eof,
}
