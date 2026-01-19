/// MathLex - Swift wrapper for the mathlex mathematical expression parser
///
/// This module provides a Swift-friendly API for parsing mathematical expressions
/// in LaTeX and plain text formats. It wraps the Rust-based mathlex library.
///
/// ## Overview
///
/// MathLex is a pure parsing library that converts mathematical notation into
/// an Abstract Syntax Tree (AST). It does NOT evaluate expressions or perform
/// any mathematical computations.
///
/// ## Quick Start
///
/// ```swift
/// import MathLex
///
/// // Parse plain text expression
/// let expr = try MathExpression.parse("2*x + sin(y)")
///
/// // Parse LaTeX expression
/// let latexExpr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
///
/// // Get variables used in expression
/// let variables = expr.variables
///
/// // Convert to LaTeX
/// let latex = expr.latex
/// ```

import Foundation

// Import the generated swift-bridge bindings
// Note: This will be available once the XCFramework is built
// For now, we create placeholder types that match the expected interface
#if canImport(MathLexRust)
import MathLexRust
#endif

/// Errors that can occur during mathematical expression parsing
public enum MathLexError: Error {
    /// A parsing error occurred with the given message
    case parseError(String)

    /// An internal error occurred
    case internalError(String)
}

extension MathLexError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .parseError(let message):
            return "Parse error: \(message)"
        case .internalError(let message):
            return "Internal error: \(message)"
        }
    }
}

/// A mathematical expression represented as an Abstract Syntax Tree (AST)
///
/// This type provides a Swift-friendly wrapper around the Rust-based mathlex parser.
/// It supports parsing both plain text and LaTeX mathematical notation.
///
/// ## Parsing
///
/// ```swift
/// // Parse plain text
/// let expr1 = try MathExpression.parse("x^2 + 2*x + 1")
///
/// // Parse LaTeX
/// let expr2 = try MathExpression.parseLatex(#"x^2 + 2x + 1"#)
/// ```
///
/// ## Querying
///
/// ```swift
/// let expr = try MathExpression.parse("sin(x) + cos(y)")
///
/// // Get all variables
/// let vars = expr.variables  // {"x", "y"}
///
/// // Get all functions
/// let funcs = expr.functions  // {"sin", "cos"}
///
/// // Get tree depth
/// let depth = expr.depth  // Tree depth
///
/// // Get node count
/// let count = expr.nodeCount  // Total nodes
/// ```
///
/// ## Conversion
///
/// ```swift
/// let expr = try MathExpression.parse("1/2")
///
/// // Convert to string (plain text)
/// let text = expr.description  // "1 / 2"
///
/// // Convert to LaTeX
/// let latex = expr.latex  // "\\frac{1}{2}"
/// ```
public struct MathExpression {
    #if canImport(MathLexRust)
    // The underlying Rust expression
    // This will be populated when swift-bridge bindings are available
    private let inner: Any  // TODO: Replace with actual Expression type from MathLexRust
    #else
    // Placeholder for development
    private let placeholder: String
    #endif

    #if canImport(MathLexRust)
    /// Internal initializer from Rust expression
    private init(inner: Any) {
        self.inner = inner
    }
    #else
    /// Placeholder initializer
    private init(placeholder: String) {
        self.placeholder = placeholder
    }
    #endif

    // MARK: - Parsing

    /// Parses a plain text mathematical expression
    ///
    /// This method parses standard mathematical notation using common operators
    /// and function syntax.
    ///
    /// ## Supported Notation
    ///
    /// - Binary operators: `+`, `-`, `*`, `/`, `^`, `%`
    /// - Functions: `sin(x)`, `cos(x)`, `log(x)`, etc.
    /// - Variables: `x`, `y`, `theta`, etc.
    /// - Constants: `pi`, `e`, `i`
    /// - Parentheses for grouping
    ///
    /// ## Examples
    ///
    /// ```swift
    /// let simple = try MathExpression.parse("2 + 3")
    /// let complex = try MathExpression.parse("sin(x)^2 + cos(x)^2")
    /// let calculus = try MathExpression.parse("d/dx(x^2)")
    /// ```
    ///
    /// - Parameter input: The mathematical expression string to parse
    /// - Returns: A parsed `MathExpression`
    /// - Throws: `MathLexError.parseError` if the input is invalid
    public static func parse(_ input: String) throws -> MathExpression {
        #if canImport(MathLexRust)
        // TODO: Call actual Rust parser when bindings are available
        // let result = parse_expression(input)
        // return MathExpression(inner: result)
        throw MathLexError.internalError("MathLexRust bindings not yet available")
        #else
        // Placeholder implementation
        return MathExpression(placeholder: input)
        #endif
    }

    /// Parses a LaTeX mathematical expression
    ///
    /// This method parses LaTeX mathematical notation, supporting common
    /// LaTeX commands for mathematical expressions.
    ///
    /// ## Supported LaTeX Commands
    ///
    /// - Fractions: `\frac{a}{b}`
    /// - Powers: `x^2` or `x^{2n}`
    /// - Roots: `\sqrt{x}`, `\sqrt[n]{x}`
    /// - Functions: `\sin{x}`, `\cos{x}`, etc.
    /// - Greek letters: `\pi`, `\theta`, `\alpha`, etc.
    /// - Derivatives: `\frac{d}{dx}`, `\frac{\partial}{\partial x}`
    /// - Integrals: `\int`, `\int_a^b`
    /// - Summations: `\sum_{i=1}^{n}`
    /// - Matrices: `\begin{pmatrix}...\end{pmatrix}`
    ///
    /// ## Examples
    ///
    /// ```swift
    /// let frac = try MathExpression.parseLatex(#"\frac{1}{2}"#)
    /// let integral = try MathExpression.parseLatex(#"\int_0^1 x^2 dx"#)
    /// let sum = try MathExpression.parseLatex(#"\sum_{i=1}^{n} i"#)
    /// ```
    ///
    /// - Parameter input: The LaTeX expression string to parse
    /// - Returns: A parsed `MathExpression`
    /// - Throws: `MathLexError.parseError` if the input is invalid
    public static func parseLatex(_ input: String) throws -> MathExpression {
        #if canImport(MathLexRust)
        // TODO: Call actual Rust LaTeX parser when bindings are available
        // let result = parse_latex(input)
        // return MathExpression(inner: result)
        throw MathLexError.internalError("MathLexRust bindings not yet available")
        #else
        // Placeholder implementation
        return MathExpression(placeholder: "latex:\(input)")
        #endif
    }

    // MARK: - Conversion

    /// A string representation of the expression in plain text format
    ///
    /// This converts the AST back to a human-readable mathematical expression
    /// using standard notation.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
    /// print(expr.description)  // "1 / 2"
    /// ```
    public var description: String {
        #if canImport(MathLexRust)
        // TODO: Call actual conversion when bindings are available
        return "Expression"
        #else
        return placeholder
        #endif
    }

    /// A LaTeX representation of the expression
    ///
    /// This converts the AST to LaTeX notation, suitable for rendering
    /// with LaTeX engines or display in documentation.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parse("1/2")
    /// print(expr.latex)  // "\\frac{1}{2}"
    /// ```
    public var latex: String {
        #if canImport(MathLexRust)
        // TODO: Call actual LaTeX conversion when bindings are available
        return ""
        #else
        return placeholder.hasPrefix("latex:")
            ? String(placeholder.dropFirst(6))
            : placeholder
        #endif
    }

    // MARK: - Querying

    /// All unique variable names used in the expression
    ///
    /// Returns a set of all variable identifiers found in the expression,
    /// including index variables from summations and integration variables.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parse("x + y + x")
    /// print(expr.variables)  // {"x", "y"}
    /// ```
    public var variables: Set<String> {
        #if canImport(MathLexRust)
        // TODO: Call actual variable extraction when bindings are available
        return []
        #else
        // Placeholder: extract simple variable names
        let pattern = #"[a-zA-Z_][a-zA-Z0-9_]*"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        let text = placeholder.hasPrefix("latex:")
            ? String(placeholder.dropFirst(6))
            : placeholder
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range)

        return Set(matches.compactMap { match -> String? in
            guard let range = Range(match.range, in: text) else { return nil }
            return String(text[range])
        })
        #endif
    }

    /// All unique function names used in the expression
    ///
    /// Returns a set of all function identifiers found in the expression.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parse("sin(x) + cos(y)")
    /// print(expr.functions)  // {"sin", "cos"}
    /// ```
    public var functions: Set<String> {
        #if canImport(MathLexRust)
        // TODO: Call actual function extraction when bindings are available
        return []
        #else
        // Placeholder: extract function names (word followed by parenthesis)
        let pattern = #"([a-zA-Z_][a-zA-Z0-9_]*)\("#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        let text = placeholder.hasPrefix("latex:")
            ? String(placeholder.dropFirst(6))
            : placeholder
        let range = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: range)

        return Set(matches.compactMap { match -> String? in
            guard match.numberOfRanges > 1,
                  let range = Range(match.range(at: 1), in: text) else { return nil }
            return String(text[range])
        })
        #endif
    }

    /// All mathematical constants used in the expression
    ///
    /// Returns a set of constant identifiers (π, e, i, ∞) found in the expression.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parse("2*pi + e")
    /// print(expr.constants)  // {"pi", "e"}
    /// ```
    public var constants: Set<String> {
        #if canImport(MathLexRust)
        // TODO: Call actual constant extraction when bindings are available
        return []
        #else
        // Placeholder
        let knownConstants = Set(["pi", "e", "i", "inf"])
        return variables.intersection(knownConstants)
        #endif
    }

    /// The maximum depth of the expression tree
    ///
    /// This measures the longest path from the root to any leaf node.
    /// Leaf nodes (literals, variables, constants) have depth 1.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let simple = try MathExpression.parse("x")
    /// print(simple.depth)  // 1
    ///
    /// let nested = try MathExpression.parse("(x + y) * z")
    /// print(nested.depth)  // 3
    /// ```
    public var depth: Int {
        #if canImport(MathLexRust)
        // TODO: Call actual depth calculation when bindings are available
        return 0
        #else
        return 1  // Placeholder
        #endif
    }

    /// The total number of nodes in the expression tree
    ///
    /// This counts all AST nodes, providing a measure of expression complexity.
    ///
    /// ## Example
    ///
    /// ```swift
    /// let expr = try MathExpression.parse("x + y")
    /// print(expr.nodeCount)  // 3 (Add + x + y)
    /// ```
    public var nodeCount: Int {
        #if canImport(MathLexRust)
        // TODO: Call actual node counting when bindings are available
        return 0
        #else
        return 1  // Placeholder
        #endif
    }
}

// MARK: - CustomStringConvertible

extension MathExpression: CustomStringConvertible {
    // Uses the description property defined above
}

// MARK: - Equatable

extension MathExpression: Equatable {
    public static func == (lhs: MathExpression, rhs: MathExpression) -> Bool {
        #if canImport(MathLexRust)
        // TODO: Implement proper equality when bindings are available
        return false
        #else
        return lhs.placeholder == rhs.placeholder
        #endif
    }
}

// MARK: - Hashable

extension MathExpression: Hashable {
    public func hash(into hasher: inout Hasher) {
        #if canImport(MathLexRust)
        // TODO: Implement proper hashing when bindings are available
        hasher.combine(0)
        #else
        hasher.combine(placeholder)
        #endif
    }
}
