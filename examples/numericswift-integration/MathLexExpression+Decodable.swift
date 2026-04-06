/// MathLexExpression+Decodable.swift
/// JSONDecoder conformance for MathLexExpression.
///
/// Split from MathLexExpression.swift to keep file sizes within the 400-line
/// limit. This file contains only the Decodable implementation; all type
/// declarations live in MathLexExpression.swift.

import Foundation

// MARK: - CodingKey for MathLexExpression

/// Coding keys for all MathLexExpression variants.
/// Declared at file scope so decoder helpers in multiple extension files can use it.
enum MathLexTopKey: String, CodingKey {
  case Integer, Float, Variable, Constant
  case Rational, Complex, Quaternion
  case Binary, Unary, Function
  case Derivative, PartialDerivative, Integral, MultipleIntegral
  case ClosedIntegral, Limit, Sum, Product
  case Vector, Matrix
  case Equation, Inequality
  case ForAll, Exists, Logical
  case MarkedVector, DotProduct, CrossProduct, OuterProduct
  case Gradient, Divergence, Curl, Laplacian, Nabla
  case Determinant, Trace, Rank, ConjugateTranspose, MatrixInverse
  case NumberSetExpr, SetOperation, SetRelationExpr, SetBuilder
  case EmptySet, PowerSet
  case Tensor, KroneckerDelta, LeviCivita
  case FunctionSignature, Composition, Differential, WedgeProduct
  case Relation
}

// MARK: - Decodable

extension MathLexExpression {

  public init(from decoder: Decoder) throws {
    // Unit variants arrive as bare JSON strings.
    if let str = try? decoder.singleValueContainer().decode(String.self) {
      switch str {
      case "Nabla":
        self = .nabla
        return
      case "EmptySet":
        self = .emptySet
        return
      default: break
      }
    }
    let c = try decoder.container(keyedBy: MathLexTopKey.self)
    if let result = try Self.decodeLiterals(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeArithmetic(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeCalculus(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeCollectionsAndRelations(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeLinearAlgebra(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeSetTheory(from: c) {
      self = result
      return
    }
    if let result = try Self.decodeTensorsAndForms(from: c) {
      self = result
      return
    }
    throw DecodingError.dataCorrupted(
      DecodingError.Context(
        codingPath: decoder.codingPath,
        debugDescription: "Unknown MathLexExpression variant"
      ))
  }

  // MARK: Literals and simple primitives

  private static func decodeLiterals(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if let v = try c.decodeIfPresent(Int64.self, forKey: .Integer) {
      return .integer(v)
    }
    if c.contains(.Float) {
      let v = try c.decodeIfPresent(Double.self, forKey: .Float)
      return .float(v)
    }
    if let v = try c.decodeIfPresent(String.self, forKey: .Variable) {
      return .variable(v)
    }
    if let v = try c.decodeIfPresent(MathConstant.self, forKey: .Constant) {
      return .constant(v)
    }
    if let v = try c.decodeIfPresent(NumberSet.self, forKey: .NumberSetExpr) {
      return .numberSetExpr(v)
    }
    return nil
  }

  // MARK: Arithmetic nodes

  private static func decodeArithmetic(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Rational) {
      struct P: Decodable { let numerator, denominator: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Rational)
      return .rational(numerator: p.numerator, denominator: p.denominator)
    }
    if c.contains(.Complex) {
      struct P: Decodable { let real, imaginary: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Complex)
      return .complex(real: p.real, imaginary: p.imaginary)
    }
    if c.contains(.Quaternion) {
      struct P: Decodable { let real, i, j, k: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Quaternion)
      return .quaternion(real: p.real, i: p.i, j: p.j, k: p.k)
    }
    if c.contains(.Binary) {
      struct P: Decodable {
        let op: BinaryOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Binary)
      return .binary(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.Unary) {
      struct P: Decodable {
        let op: UnaryOp
        let operand: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Unary)
      return .unary(op: p.op, operand: p.operand)
    }
    if c.contains(.Function) {
      struct P: Decodable {
        let name: String
        let args: [MathLexExpression]
      }
      let p = try c.decode(P.self, forKey: .Function)
      return .function(name: p.name, args: p.args)
    }
    return nil
  }

  // MARK: Calculus nodes

  private static func decodeCalculus(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Derivative) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let order: UInt32
      }
      let p = try c.decode(P.self, forKey: .Derivative)
      return .derivative(expr: p.expr, var: p.var, order: p.order)
    }
    if c.contains(.PartialDerivative) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let order: UInt32
      }
      let p = try c.decode(P.self, forKey: .PartialDerivative)
      return .partialDerivative(expr: p.expr, var: p.var, order: p.order)
    }
    if c.contains(.Integral) {
      struct P: Decodable {
        let integrand: MathLexExpression
        let `var`: String
        let bounds: IntegralBounds?
      }
      let p = try c.decode(P.self, forKey: .Integral)
      return .integral(integrand: p.integrand, var: p.var, bounds: p.bounds)
    }
    if c.contains(.MultipleIntegral) {
      struct P: Decodable {
        let dimension: UInt8
        let integrand: MathLexExpression
        let bounds: MultipleBounds?
        let vars: [String]
      }
      let p = try c.decode(P.self, forKey: .MultipleIntegral)
      return .multipleIntegral(
        dimension: p.dimension, integrand: p.integrand, bounds: p.bounds, vars: p.vars)
    }
    if c.contains(.ClosedIntegral) {
      struct P: Decodable {
        let dimension: UInt8
        let integrand: MathLexExpression
        let surface: String?
        let `var`: String
      }
      let p = try c.decode(P.self, forKey: .ClosedIntegral)
      return .closedIntegral(
        dimension: p.dimension, integrand: p.integrand, surface: p.surface, var: p.var)
    }
    if c.contains(.Limit) {
      struct P: Decodable {
        let expr: MathLexExpression
        let `var`: String
        let to: MathLexExpression
        let direction: Direction
      }
      let p = try c.decode(P.self, forKey: .Limit)
      return .limit(expr: p.expr, var: p.var, to: p.to, direction: p.direction)
    }
    if c.contains(.Sum) {
      struct P: Decodable {
        let index: String
        let lower, upper, body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Sum)
      return .sum(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
    }
    if c.contains(.Product) {
      struct P: Decodable {
        let index: String
        let lower, upper, body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Product)
      return .product(index: p.index, lower: p.lower, upper: p.upper, body: p.body)
    }
    return nil
  }

  // MARK: Collections, equations, and logic

  private static func decodeCollectionsAndRelations(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Vector) {
      return .vector(try c.decode([MathLexExpression].self, forKey: .Vector))
    }
    if c.contains(.Matrix) {
      return .matrix(try c.decode([[MathLexExpression]].self, forKey: .Matrix))
    }
    if c.contains(.Equation) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Equation)
      return .equation(left: p.left, right: p.right)
    }
    if c.contains(.Inequality) {
      struct P: Decodable {
        let op: InequalityOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Inequality)
      return .inequality(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.ForAll) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let body: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .ForAll)
      return .forAll(variable: p.variable, domain: p.domain, body: p.body)
    }
    if c.contains(.Exists) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let body: MathLexExpression
        let unique: Bool
      }
      let p = try c.decode(P.self, forKey: .Exists)
      return .exists(variable: p.variable, domain: p.domain, body: p.body, unique: p.unique)
    }
    if c.contains(.Logical) {
      struct P: Decodable {
        let op: LogicalOp
        let operands: [MathLexExpression]
      }
      let p = try c.decode(P.self, forKey: .Logical)
      return .logical(op: p.op, operands: p.operands)
    }
    return nil
  }

  // MARK: Linear algebra

  private static func decodeLinearAlgebra(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.MarkedVector) {
      struct P: Decodable {
        let name: String
        let notation: VectorNotation
      }
      let p = try c.decode(P.self, forKey: .MarkedVector)
      return .markedVector(name: p.name, notation: p.notation)
    }
    if c.contains(.DotProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .DotProduct)
      return .dotProduct(left: p.left, right: p.right)
    }
    if c.contains(.CrossProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .CrossProduct)
      return .crossProduct(left: p.left, right: p.right)
    }
    if c.contains(.OuterProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .OuterProduct)
      return .outerProduct(left: p.left, right: p.right)
    }
    if c.contains(.Gradient) {
      struct P: Decodable { let expr: MathLexExpression }
      return .gradient(expr: try c.decode(P.self, forKey: .Gradient).expr)
    }
    if c.contains(.Divergence) {
      struct P: Decodable { let field: MathLexExpression }
      return .divergence(field: try c.decode(P.self, forKey: .Divergence).field)
    }
    if c.contains(.Curl) {
      struct P: Decodable { let field: MathLexExpression }
      return .curl(field: try c.decode(P.self, forKey: .Curl).field)
    }
    if c.contains(.Laplacian) {
      struct P: Decodable { let expr: MathLexExpression }
      return .laplacian(expr: try c.decode(P.self, forKey: .Laplacian).expr)
    }
    if c.contains(.Determinant) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .determinant(matrix: try c.decode(P.self, forKey: .Determinant).matrix)
    }
    if c.contains(.Trace) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .trace(matrix: try c.decode(P.self, forKey: .Trace).matrix)
    }
    if c.contains(.Rank) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .rank(matrix: try c.decode(P.self, forKey: .Rank).matrix)
    }
    if c.contains(.ConjugateTranspose) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .conjugateTranspose(matrix: try c.decode(P.self, forKey: .ConjugateTranspose).matrix)
    }
    if c.contains(.MatrixInverse) {
      struct P: Decodable { let matrix: MathLexExpression }
      return .matrixInverse(matrix: try c.decode(P.self, forKey: .MatrixInverse).matrix)
    }
    return nil
  }

}
