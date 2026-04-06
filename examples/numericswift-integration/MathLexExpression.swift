/// MathLexExpression.swift
/// Decodable Swift types that mirror the mathlex Rust `Expression` AST.
///
/// Paste this file (and `Evaluator.swift`) into your NumericSwift target.
/// Decode JSON produced by `MathExpression.toJSON()`:
///
///     let ast = try JSONDecoder().decode(MathLexExpression.self, from: jsonData)
///
/// The JSON uses serde's externally-tagged format — every node is a single-key
/// object whose key is the variant name. Unit variants (`Nabla`, `EmptySet`)
/// are bare strings. See `docs/json-ast-schema.md` for the full reference.

import Foundation

// MARK: - Operator enums

/// Binary arithmetic and algebraic operators.
public enum BinaryOp: String, Decodable {
  case add = "Add"
  case sub = "Sub"
  case mul = "Mul"
  case div = "Div"
  case pow = "Pow"
  case mod = "Mod"
  case plusMinus = "PlusMinus"
  case minusPlus = "MinusPlus"
}

/// Unary prefix/postfix operators.
public enum UnaryOp: String, Decodable {
  case neg = "Neg"
  case pos = "Pos"
  case factorial = "Factorial"
  case transpose = "Transpose"
}

/// Relational comparison operators used in `Inequality` nodes.
public enum InequalityOp: String, Decodable {
  case lt = "Lt"
  case le = "Le"
  case gt = "Gt"
  case ge = "Ge"
  case ne = "Ne"
}

/// Logical connectives used in `Logical` nodes.
public enum LogicalOp: String, Decodable {
  case and = "And"
  case or = "Or"
  case not = "Not"
  case implies = "Implies"
  case iff = "Iff"
}

/// Mathematical relation operators (similarity, equivalence, etc.).
public enum RelationOp: String, Decodable {
  case similar = "Similar"
  case equivalent = "Equivalent"
  case congruent = "Congruent"
  case approx = "Approx"
}

/// Named mathematical constants.
public enum MathConstant: String, Decodable {
  case pi = "Pi"
  case e = "E"
  case i = "I"
  case j = "J"
  case k = "K"
  case infinity = "Infinity"
  case negInfinity = "NegInfinity"
  case nan = "NaN"
}

/// Direction of approach for limit nodes.
public enum Direction: String, Decodable {
  case left = "Left"
  case right = "Right"
  case both = "Both"
}

/// Binary set operations.
public enum SetOp: String, Decodable {
  case union = "Union"
  case intersection = "Intersection"
  case difference = "Difference"
  case symmetricDiff = "SymmetricDiff"
  case cartesianProd = "CartesianProd"
}

/// Set membership and subset relations.
public enum SetRelation: String, Decodable {
  case `in` = "In"
  case notIn = "NotIn"
  case subset = "Subset"
  case subsetEq = "SubsetEq"
  case superset = "Superset"
  case supersetEq = "SupersetEq"
}

/// Standard number sets (ℕ, ℤ, ℚ, ℝ, ℂ, ℍ).
public enum NumberSet: String, Decodable {
  case natural = "Natural"
  case integer = "Integer"
  case rational = "Rational"
  case real = "Real"
  case complex = "Complex"
  case quaternion = "Quaternion"
}

/// Visual notation styles for named vectors.
public enum VectorNotation: String, Decodable {
  case bold = "Bold"
  case arrow = "Arrow"
  case hat = "Hat"
  case underline = "Underline"
  case plain = "Plain"
}

/// Index position in tensor notation (contravariant vs. covariant).
public enum IndexType: String, Decodable {
  case upper = "Upper"
  case lower = "Lower"
}

// MARK: - Supporting types

/// A single tensor index with its name and position.
public struct TensorIndex: Decodable {
  public let name: String
  public let index_type: IndexType
}

/// Lower and upper bounds for a definite integral.
public struct IntegralBounds: Decodable {
  public let lower: MathLexExpression
  public let upper: MathLexExpression
}

/// Bounds collection for a multiple integral (one per dimension).
public struct MultipleBounds: Decodable {
  public let bounds: [IntegralBounds]
}

// MARK: - Expression

/// A decoded mathlex AST node.
///
/// This indirect enum mirrors the Rust `Expression` type. Decode it from the
/// compact JSON produced by `MathExpression.toJSON()`:
///
///     let ast = try JSONDecoder().decode(MathLexExpression.self, from: data)
public indirect enum MathLexExpression: Decodable {

  // Literals
  case integer(Int64)
  /// `nil` means a non-finite float (NaN/±Inf) — treat as a decode error.
  case float(Double?)
  case variable(String)
  case constant(MathConstant)

  // Numeric structures
  case rational(numerator: MathLexExpression, denominator: MathLexExpression)
  case complex(real: MathLexExpression, imaginary: MathLexExpression)
  case quaternion(
    real: MathLexExpression, i: MathLexExpression,
    j: MathLexExpression, k: MathLexExpression)

  // Core arithmetic
  case binary(op: BinaryOp, left: MathLexExpression, right: MathLexExpression)
  case unary(op: UnaryOp, operand: MathLexExpression)
  case function(name: String, args: [MathLexExpression])

  // Calculus
  case derivative(expr: MathLexExpression, var: String, order: UInt32)
  case partialDerivative(expr: MathLexExpression, var: String, order: UInt32)
  case integral(integrand: MathLexExpression, var: String, bounds: IntegralBounds?)
  case multipleIntegral(
    dimension: UInt8, integrand: MathLexExpression,
    bounds: MultipleBounds?, vars: [String])
  case closedIntegral(
    dimension: UInt8, integrand: MathLexExpression,
    surface: String?, var: String)
  case limit(
    expr: MathLexExpression, var: String,
    to: MathLexExpression, direction: Direction)
  case sum(
    index: String, lower: MathLexExpression,
    upper: MathLexExpression, body: MathLexExpression)
  case product(
    index: String, lower: MathLexExpression,
    upper: MathLexExpression, body: MathLexExpression)

  // Collections
  case vector([MathLexExpression])
  case matrix([[MathLexExpression]])

  // Relations and logic
  case equation(left: MathLexExpression, right: MathLexExpression)
  case inequality(op: InequalityOp, left: MathLexExpression, right: MathLexExpression)
  case forAll(variable: String, domain: MathLexExpression?, body: MathLexExpression)
  case exists(
    variable: String, domain: MathLexExpression?,
    body: MathLexExpression, unique: Bool)
  case logical(op: LogicalOp, operands: [MathLexExpression])

  // Linear algebra
  case markedVector(name: String, notation: VectorNotation)
  case dotProduct(left: MathLexExpression, right: MathLexExpression)
  case crossProduct(left: MathLexExpression, right: MathLexExpression)
  case outerProduct(left: MathLexExpression, right: MathLexExpression)
  case gradient(expr: MathLexExpression)
  case divergence(field: MathLexExpression)
  case curl(field: MathLexExpression)
  case laplacian(expr: MathLexExpression)
  case nabla
  case determinant(matrix: MathLexExpression)
  case trace(matrix: MathLexExpression)
  case rank(matrix: MathLexExpression)
  case conjugateTranspose(matrix: MathLexExpression)
  case matrixInverse(matrix: MathLexExpression)

  // Set theory
  case numberSetExpr(NumberSet)
  case setOperation(op: SetOp, left: MathLexExpression, right: MathLexExpression)
  case setRelationExpr(
    relation: SetRelation, element: MathLexExpression,
    set: MathLexExpression)
  case setBuilder(
    variable: String, domain: MathLexExpression?,
    predicate: MathLexExpression)
  case emptySet
  case powerSet(set: MathLexExpression)

  // Tensor calculus
  case tensor(name: String, indices: [TensorIndex])
  case kroneckerDelta(indices: [TensorIndex])
  case leviCivita(indices: [TensorIndex])

  // Functions and forms
  case functionSignature(name: String, domain: MathLexExpression, codomain: MathLexExpression)
  case composition(outer: MathLexExpression, inner: MathLexExpression)
  case differential(var: String)
  case wedgeProduct(left: MathLexExpression, right: MathLexExpression)
  case relation(op: RelationOp, left: MathLexExpression, right: MathLexExpression)
}

