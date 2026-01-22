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
import MathLexRust

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
    // The underlying Rust expression
    private let inner: Expression

    /// Internal initializer from Rust expression
    private init(inner: Expression) {
        self.inner = inner
    }

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
        do {
            let expr = try MathLexRust.parseText(input)
            return MathExpression(inner: expr)
        } catch let error as RustString {
            throw MathLexError.parseError(error.toString())
        }
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
        do {
            let expr = try MathLexRust.parseLatex(input)
            return MathExpression(inner: expr)
        } catch let error as RustString {
            throw MathLexError.parseError(error.toString())
        }
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
        MathLexRust.toString(inner).toString()
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
        MathLexRust.toLatex(inner).toString()
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
        let rustVec = MathLexRust.findVariables(inner)
        var result = Set<String>()
        for i in 0..<rustVec.len() {
            if let rustStr = rustVec.get(i) {
                result.insert(rustStr.toString())
            }
        }
        return result
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
        let rustVec = MathLexRust.findFunctions(inner)
        var result = Set<String>()
        for i in 0..<rustVec.len() {
            if let rustStr = rustVec.get(i) {
                result.insert(rustStr.toString())
            }
        }
        return result
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
        // Constants are a subset of variables in the AST
        // Filter for known mathematical constants
        let knownConstants = Set(["pi", "e", "i", "inf"])
        return variables.intersection(knownConstants)
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
        Int(MathLexRust.depth(inner))
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
        Int(MathLexRust.nodeCount(inner))
    }
}

// MARK: - CustomStringConvertible

extension MathExpression: CustomStringConvertible {
    // Uses the description property defined above
}

// MARK: - Equatable

extension MathExpression: Equatable {
    public static func == (lhs: MathExpression, rhs: MathExpression) -> Bool {
        // Compare based on string representation
        // This ensures structural equality of the AST
        lhs.description == rhs.description
    }
}

// MARK: - Hashable

extension MathExpression: Hashable {
    public func hash(into hasher: inout Hasher) {
        // Hash based on string representation for consistency with Equatable
        hasher.combine(description)
    }
}
