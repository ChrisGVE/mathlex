/// ComplexEvaluator.swift
/// Skeleton showing how to extend the evaluator pattern to complex numbers.
///
/// This file is intentionally illustrative. It demonstrates the structure
/// NumericSwift would use to evaluate `MathLexExpression` trees over ℂ.
/// The actual complex-number type would come from NumericSwift's own `Complex`
/// implementation (or the `Numerics` package's `Complex<Double>`).
///
/// Key insight: the recursive tree-walk structure from `Evaluator.swift` is
/// identical — only the leaf conversions and the math dispatch change.

import Foundation

// MARK: - Placeholder complex type

/// Minimal complex number for illustration. Replace with NumericSwift.Complex.
public struct ComplexDouble: CustomStringConvertible {
  public var real: Double
  public var imaginary: Double

  public init(_ real: Double, _ imaginary: Double = 0) {
    self.real = real
    self.imaginary = imaginary
  }

  public static let i = ComplexDouble(0, 1)

  public static func + (l: Self, r: Self) -> Self {
    Self(l.real + r.real, l.imaginary + r.imaginary)
  }
  public static func - (l: Self, r: Self) -> Self {
    Self(l.real - r.real, l.imaginary - r.imaginary)
  }
  public static func * (l: Self, r: Self) -> Self {
    Self(
      l.real * r.real - l.imaginary * r.imaginary,
      l.real * r.imaginary + l.imaginary * r.real)
  }
  public static func / (l: Self, r: Self) -> Self {
    let denom = r.real * r.real + r.imaginary * r.imaginary
    return Self(
      (l.real * r.real + l.imaginary * r.imaginary) / denom,
      (l.imaginary * r.real - l.real * r.imaginary) / denom)
  }
  public static prefix func - (v: Self) -> Self { Self(-v.real, -v.imaginary) }

  public var magnitude: Double { sqrt(real * real + imaginary * imaginary) }
  public var conjugate: Self { Self(real, -imaginary) }

  public var description: String {
    imaginary >= 0 ? "\(real)+\(imaginary)i" : "\(real)\(imaginary)i"
  }
}

// MARK: - Complex evaluator

/// Evaluates a `MathLexExpression` AST over ℂ (complex numbers).
///
/// Variable bindings and the return type are `ComplexDouble` rather than
/// `Double`. Everything else follows the same pattern as `Evaluator`.
public struct ComplexEvaluator {

  public typealias Env = [String: ComplexDouble]

  public init() {}

  /// Evaluates `expr` using `env` for variable bindings.
  public func evaluate(_ expr: MathLexExpression, env: Env = [:]) throws -> ComplexDouble {
    switch expr {
    case .integer(let n):
      return ComplexDouble(Double(n))
    case .float(let v):
      guard let v else { throw EvaluatorError.nonFiniteFloat }
      return ComplexDouble(v)
    case .variable(let name):
      guard let v = env[name] else { throw EvaluatorError.undefinedVariable(name) }
      return v
    case .constant(let c):
      return try resolveConstant(c)
    case .rational(let num, let den):
      return try evaluate(num, env: env) / evaluate(den, env: env)
    case .complex(let re, let im):
      // A Complex node from the AST carries real and imaginary sub-trees.
      let r = try evaluate(re, env: env)
      let i = try evaluate(im, env: env)
      // Both sub-trees must be purely real for this simple path.
      return ComplexDouble(r.real, i.real)
    case .binary(let op, let lhs, let rhs):
      return try applyBinary(op, lhs, rhs, env: env)
    case .unary(let op, let operand):
      return try applyUnary(op, operand, env: env)
    case .function(let name, let args):
      return try applyFunction(name, args, env: env)
    default:
      throw EvaluatorError.unsupportedNode(
        String(describing: expr).components(separatedBy: "(").first ?? "?")
    }
  }

  // MARK: Constants

  private func resolveConstant(_ c: MathConstant) throws -> ComplexDouble {
    switch c {
    case .pi: return ComplexDouble(Double.pi)
    case .e: return ComplexDouble(M_E)
    case .i: return ComplexDouble.i
    case .infinity: return ComplexDouble(Double.infinity)
    case .negInfinity: return ComplexDouble(-Double.infinity)
    case .nan: return ComplexDouble(Double.nan)
    case .j, .k:
      throw EvaluatorError.unsupportedNode("Quaternion basis \(c) requires quaternion arithmetic")
    }
  }

  // MARK: Binary operators

  private func applyBinary(
    _ op: BinaryOp,
    _ lhs: MathLexExpression,
    _ rhs: MathLexExpression,
    env: Env
  ) throws -> ComplexDouble {
    let l = try evaluate(lhs, env: env)
    let r = try evaluate(rhs, env: env)
    switch op {
    case .add: return l + r
    case .sub: return l - r
    case .mul: return l * r
    case .div: return l / r
    case .pow:
      // Complex power: z^w = exp(w * ln(z)) — simplified real-exponent path
      if r.imaginary == 0 {
        return complexPow(l, r.real)
      }
      throw EvaluatorError.unsupportedNode("complex^complex power")
    case .mod:
      throw EvaluatorError.unsupportedNode("modulo over complex numbers")
    case .plusMinus, .minusPlus:
      throw EvaluatorError.unsupportedNode("BinaryOp(\(op))")
    }
  }

  // MARK: Unary operators

  private func applyUnary(
    _ op: UnaryOp,
    _ operand: MathLexExpression,
    env: Env
  ) throws -> ComplexDouble {
    let v = try evaluate(operand, env: env)
    switch op {
    case .neg: return -v
    case .pos: return v
    case .factorial:
      guard v.imaginary == 0, v.real >= 0 else {
        throw EvaluatorError.domainError("factorial requires non-negative real")
      }
      return ComplexDouble(tgamma(v.real + 1))
    case .transpose:
      throw EvaluatorError.unsupportedNode("transpose (matrix operation)")
    }
  }

  // MARK: Functions

  private func applyFunction(
    _ name: String,
    _ args: [MathLexExpression],
    env: Env
  ) throws -> ComplexDouble {
    let vals = try args.map { try evaluate($0, env: env) }
    if let result = applyBuiltin(name, vals) { return result }
    throw EvaluatorError.unknownFunction(name, argCount: vals.count)
  }

  /// Complex function dispatch. Handles real-valued and complex inputs.
  private func applyBuiltin(_ name: String, _ args: [ComplexDouble]) -> ComplexDouble? {
    // For purely real arguments delegate to the real evaluator path for clarity.
    let allReal = args.allSatisfy { $0.imaginary == 0 }
    let reals = args.map(\.real)

    switch (name, args.count) {
    case ("abs", 1):
      return ComplexDouble(args[0].magnitude)
    case ("conj", 1):
      return args[0].conjugate
    case ("re", 1):
      return ComplexDouble(args[0].real)
    case ("im", 1):
      return ComplexDouble(args[0].imaginary)
    // For trig/exp/log, use real fast paths when possible.
    case ("exp", 1) where allReal: return ComplexDouble(exp(reals[0]))
    case ("exp", 1):
      // exp(a+bi) = exp(a)(cos(b)+i·sin(b))
      let ea = exp(args[0].real)
      return ComplexDouble(ea * cos(args[0].imaginary), ea * sin(args[0].imaginary))
    case ("ln", 1) where allReal && reals[0] > 0: return ComplexDouble(log(reals[0]))
    case ("ln", 1):
      // ln(z) = ln|z| + i·arg(z)
      return ComplexDouble(
        log(args[0].magnitude),
        atan2(args[0].imaginary, args[0].real))
    case ("sin", 1) where allReal: return ComplexDouble(sin(reals[0]))
    case ("cos", 1) where allReal: return ComplexDouble(cos(reals[0]))
    case ("sqrt", 1) where allReal && reals[0] >= 0: return ComplexDouble(sqrt(reals[0]))
    case ("sqrt", 1):
      let r = args[0].magnitude
      let theta = atan2(args[0].imaginary, args[0].real)
      return ComplexDouble(sqrt(r) * cos(theta / 2), sqrt(r) * sin(theta / 2))
    case ("clamp", 3) where allReal:
      return ComplexDouble(min(max(reals[0], reals[1]), reals[2]))
    case ("lerp", 3):
      return args[0] + (args[1] - args[0]) * args[2]
    case ("rad", 1) where allReal: return ComplexDouble(reals[0] * (.pi / 180))
    case ("deg", 1) where allReal: return ComplexDouble(reals[0] * (180 / .pi))
    default: return nil
    }
  }

  // MARK: Helpers

  /// Real-exponent complex power: z^n.
  private func complexPow(_ z: ComplexDouble, _ n: Double) -> ComplexDouble {
    let r = pow(z.magnitude, n)
    let theta = atan2(z.imaginary, z.real) * n
    return ComplexDouble(r * cos(theta), r * sin(theta))
  }
}
