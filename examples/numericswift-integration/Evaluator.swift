/// Evaluator.swift
/// Walks a decoded `MathLexExpression` AST and computes a `Double` result.
///
/// ## Usage
///
///     let evaluator = Evaluator()
///     let result = try evaluator.evaluate(ast, env: ["x": 1.5, "y": .pi])
///
/// ## Limitations
///
/// Only nodes that reduce to a single finite `Double` are handled. Symbolic
/// calculus nodes (Derivative, Integral, Limit, …), set-theory nodes, tensor
/// nodes, and logical nodes return `nil` from the underlying walk and cause an
/// `EvaluatorError.unsupportedNode` to be thrown.

import Foundation

// MARK: - Errors

/// Errors thrown by `Evaluator`.
public enum EvaluatorError: Error, CustomStringConvertible {
  /// A variable name was not found in the environment.
  case undefinedVariable(String)
  /// A function name is not in the built-in dispatch table.
  case unknownFunction(String, argCount: Int)
  /// An AST node type that the evaluator does not support (e.g. integrals).
  case unsupportedNode(String)
  /// A non-finite float was decoded (NaN or ±Inf stored as `null` in JSON).
  case nonFiniteFloat
  /// Division by zero or other numeric domain error.
  case domainError(String)

  public var description: String {
    switch self {
    case .undefinedVariable(let n): return "Undefined variable: \(n)"
    case .unknownFunction(let n, let c): return "Unknown function: \(n)/\(c)"
    case .unsupportedNode(let n): return "Unsupported AST node: \(n)"
    case .nonFiniteFloat: return "Non-finite float in AST"
    case .domainError(let msg): return "Domain error: \(msg)"
    }
  }
}

// MARK: - Evaluator

/// Evaluates a `MathLexExpression` tree to a `Double`.
///
/// Variable bindings are supplied via the `env` dictionary. Constants (π, e, ∞)
/// are resolved automatically.
public struct Evaluator {

  /// Variable binding environment: maps variable names to `Double` values.
  public typealias Env = [String: Double]

  public init() {}

  /// Evaluates `expr` using `env` for variable bindings.
  ///
  /// - Parameters:
  ///   - expr: The decoded AST to evaluate.
  ///   - env: Variable bindings (default: empty).
  /// - Returns: The computed `Double` value.
  /// - Throws: `EvaluatorError` on undefined variables, unknown functions, or
  ///   unsupported node types.
  public func evaluate(_ expr: MathLexExpression, env: Env = [:]) throws -> Double {
    switch expr {
    case .integer(let n):
      return Double(n)
    case .float(let v):
      guard let v else { throw EvaluatorError.nonFiniteFloat }
      return v
    case .variable(let name):
      guard let v = env[name] else { throw EvaluatorError.undefinedVariable(name) }
      return v
    case .constant(let c):
      return try resolveConstant(c)
    case .rational(let num, let den):
      return try evaluate(num, env: env) / evaluate(den, env: env)
    case .binary(let op, let lhs, let rhs):
      return try applyBinary(op, lhs, rhs, env: env)
    case .unary(let op, let operand):
      return try applyUnary(op, operand, env: env)
    case .function(let name, let args):
      return try applyFunction(name, args, env: env)
    default:
      throw EvaluatorError.unsupportedNode(nodeLabel(expr))
    }
  }

  // MARK: Constants

  private func resolveConstant(_ c: MathConstant) throws -> Double {
    switch c {
    case .pi: return Double.pi
    case .e: return M_E
    case .infinity: return Double.infinity
    case .negInfinity: return -Double.infinity
    case .nan: return Double.nan
    case .i, .j, .k:
      throw EvaluatorError.unsupportedNode("Constant(\(c)) requires complex arithmetic")
    }
  }

  // MARK: Binary operators

  private func applyBinary(
    _ op: BinaryOp,
    _ lhs: MathLexExpression,
    _ rhs: MathLexExpression,
    env: Env
  ) throws -> Double {
    let l = try evaluate(lhs, env: env)
    let r = try evaluate(rhs, env: env)
    switch op {
    case .add: return l + r
    case .sub: return l - r
    case .mul: return l * r
    case .div: return l / r  // IEEE 754: x/0 = ±Inf
    case .pow: return pow(l, r)
    case .mod: return l.truncatingRemainder(dividingBy: r)
    case .plusMinus, .minusPlus:
      throw EvaluatorError.unsupportedNode("BinaryOp(\(op))")
    }
  }

  // MARK: Unary operators

  private func applyUnary(
    _ op: UnaryOp,
    _ operand: MathLexExpression,
    env: Env
  ) throws -> Double {
    let v = try evaluate(operand, env: env)
    switch op {
    case .neg: return -v
    case .pos: return v
    case .factorial:
      guard v >= 0 else { throw EvaluatorError.domainError("factorial of negative") }
      return tgamma(v + 1)
    case .transpose:
      throw EvaluatorError.unsupportedNode("UnaryOp.transpose (matrix operation)")
    }
  }

  // MARK: Functions

  private func applyFunction(
    _ name: String,
    _ args: [MathLexExpression],
    env: Env
  ) throws -> Double {
    let vals = try args.map { try evaluate($0, env: env) }
    if let result = applyBuiltin(name, vals) { return result }
    throw EvaluatorError.unknownFunction(name, argCount: vals.count)
  }

  /// Dispatch table for built-in functions. Returns `nil` for unknown names
  /// or wrong argument counts so the caller can produce a typed error.
  private func applyBuiltin(_ name: String, _ args: [Double]) -> Double? {
    switch (name, args.count) {
    // Trigonometric
    case ("sin", 1): return sin(args[0])
    case ("cos", 1): return cos(args[0])
    case ("tan", 1): return tan(args[0])
    case ("asin", 1): return asin(args[0])
    case ("acos", 1): return acos(args[0])
    case ("atan", 1): return atan(args[0])
    case ("atan2", 2): return atan2(args[0], args[1])
    case ("sinh", 1): return sinh(args[0])
    case ("cosh", 1): return cosh(args[0])
    case ("tanh", 1): return tanh(args[0])
    // Roots and powers
    case ("sqrt", 1): return sqrt(args[0])
    case ("cbrt", 1): return cbrt(args[0])
    case ("pow", 2): return pow(args[0], args[1])
    // Exponential and logarithmic
    case ("exp", 1): return exp(args[0])
    case ("ln", 1): return log(args[0])
    case ("log", 1): return log10(args[0])
    case ("log", 2): return log(args[0]) / log(args[1])
    case ("log2", 1): return log2(args[0])
    // Rounding and clamping
    case ("abs", 1): return abs(args[0])
    case ("ceil", 1): return ceil(args[0])
    case ("floor", 1): return floor(args[0])
    case ("round", 1): return Foundation.round(args[0])
    case ("trunc", 1): return trunc(args[0])
    case ("sign", 1): return args[0] < 0 ? -1 : (args[0] > 0 ? 1 : 0)
    // clamp(x, lo, hi)
    case ("clamp", 3): return min(max(args[0], args[1]), args[2])
    // lerp(a, b, t) — linear interpolation
    case ("lerp", 3): return args[0] + (args[1] - args[0]) * args[2]
    // Angle conversion
    case ("rad", 1): return args[0] * (.pi / 180.0)
    case ("deg", 1): return args[0] * (180.0 / .pi)
    // Minimum / maximum
    case ("min", 2): return Swift.min(args[0], args[1])
    case ("max", 2): return Swift.max(args[0], args[1])
    // Hypot
    case ("hypot", 2): return hypot(args[0], args[1])
    default: return nil
    }
  }

  // MARK: Node label helper

  private func nodeLabel(_ expr: MathLexExpression) -> String {
    switch expr {
    case .derivative: return "Derivative"
    case .partialDerivative: return "PartialDerivative"
    case .integral: return "Integral"
    case .multipleIntegral: return "MultipleIntegral"
    case .closedIntegral: return "ClosedIntegral"
    case .limit: return "Limit"
    case .sum: return "Sum"
    case .product: return "Product"
    case .complex: return "Complex"
    case .quaternion: return "Quaternion"
    case .matrix: return "Matrix"
    case .vector: return "Vector"
    case .equation: return "Equation"
    case .inequality: return "Inequality"
    case .logical: return "Logical"
    case .forAll: return "ForAll"
    case .exists: return "Exists"
    case .markedVector: return "MarkedVector"
    case .dotProduct: return "DotProduct"
    case .crossProduct: return "CrossProduct"
    case .outerProduct: return "OuterProduct"
    case .gradient: return "Gradient"
    case .divergence: return "Divergence"
    case .curl: return "Curl"
    case .laplacian: return "Laplacian"
    case .nabla: return "Nabla"
    case .determinant: return "Determinant"
    case .trace: return "Trace"
    case .rank: return "Rank"
    case .conjugateTranspose: return "ConjugateTranspose"
    case .matrixInverse: return "MatrixInverse"
    case .numberSetExpr: return "NumberSetExpr"
    case .setOperation: return "SetOperation"
    case .setRelationExpr: return "SetRelationExpr"
    case .setBuilder: return "SetBuilder"
    case .emptySet: return "EmptySet"
    case .powerSet: return "PowerSet"
    case .tensor: return "Tensor"
    case .kroneckerDelta: return "KroneckerDelta"
    case .leviCivita: return "LeviCivita"
    case .functionSignature: return "FunctionSignature"
    case .composition: return "Composition"
    case .differential: return "Differential"
    case .wedgeProduct: return "WedgeProduct"
    case .relation: return "Relation"
    default: return "Unknown"
    }
  }
}
