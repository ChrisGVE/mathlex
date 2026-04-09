//! Function name suggestion helpers using Levenshtein distance.

/// Known mathematical functions that the parser recognizes.
///
/// Used for generating suggestions when an unknown function is encountered.
const KNOWN_FUNCTIONS: &[&str] = &[
    // Trigonometric
    "sin",
    "cos",
    "tan",
    "csc",
    "sec",
    "cot",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    // Exponential / logarithmic
    "log",
    "ln",
    "exp",
    "sqrt",
    "abs",
    "floor",
    "ceil",
    "round",
    "sign",
    "min",
    "max",
    // Number theory
    "gcd",
    "lcm",
    // Aliases
    "atan2",
    "cbrt",
    "pow",
    "sgn",
    "lg",
    "asin",
    "acos",
    "atan",
    "log2",
    // Calculus (special forms handled by parser)
    "diff",
    "partial",
    "integrate",
    "integral",
    "sum",
    "summation",
    "product",
    "prod",
    "limit",
    "lim",
    // Vector calculus
    "grad",
    "nabla",
    "div",
    "curl",
    "laplacian",
    "dot",
    "cross",
    // Linear algebra
    "det",
    "tr",
    "rank",
    // Special functions
    "gamma",
    "beta",
    "erf",
    "erfc",
    "zeta",
    "bessel_j",
    "bessel_y",
    "bessel_i",
    "bessel_k",
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    // Transforms
    "laplace",
    "fourier",
    "ilaplace",
    "ifourier",
    "laplace_transform",
    "fourier_transform",
];

/// Computes the Levenshtein distance between two strings.
///
/// The Levenshtein distance is the minimum number of single-character edits
/// (insertions, deletions, or substitutions) required to change one string
/// into another.
///
/// # Arguments
///
/// * `a` - First string
/// * `b` - Second string
///
/// # Returns
///
/// The Levenshtein distance between the two strings.
///
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row: Vec<usize> = vec![0; b_len + 1];

    for (i, a_char) in a_chars.iter().enumerate() {
        curr_row[0] = i + 1;

        for (j, b_char) in b_chars.iter().enumerate() {
            let cost = if a_char == b_char { 0 } else { 1 };
            curr_row[j + 1] = std::cmp::min(
                std::cmp::min(curr_row[j] + 1, prev_row[j + 1] + 1),
                prev_row[j] + cost,
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// Suggests a known function name similar to the unknown name.
///
/// Uses Levenshtein distance to find known functions that are close to the
/// given unknown function name. Only suggests functions within an edit distance
/// of 2.
///
/// # Arguments
///
/// * `unknown` - The unknown function name
///
/// # Returns
///
/// An optional suggestion string if a similar function is found.
///
/// # Example
///
/// ```
/// use mathlex::error::suggest_function;
///
/// assert_eq!(suggest_function("sen"), Some("Did you mean 'sin'?".to_string()));
/// assert_eq!(suggest_function("coz"), Some("Did you mean 'cos'?".to_string()));
/// assert_eq!(suggest_function("xyz"), None);
/// ```
pub fn suggest_function(unknown: &str) -> Option<String> {
    KNOWN_FUNCTIONS
        .iter()
        .filter(|&&f| levenshtein(unknown, f) <= 2)
        .min_by_key(|&&f| levenshtein(unknown, f))
        .map(|&f| format!("Did you mean '{}'?", f))
}
