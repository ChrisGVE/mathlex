/// MathLexExpression+DecodableExtra.swift
/// Set theory, tensor, and differential-form decode helpers for MathLexExpression.
///
/// Split from MathLexExpression+Decodable.swift to keep file sizes within limits.

import Foundation

extension MathLexExpression {


  // MARK: Set theory

  private static func decodeSetTheory(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.SetOperation) {
      struct P: Decodable {
        let op: SetOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetOperation)
      return .setOperation(op: p.op, left: p.left, right: p.right)
    }
    if c.contains(.SetRelationExpr) {
      struct P: Decodable {
        let relation: SetRelation
        let element, set: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetRelationExpr)
      return .setRelationExpr(relation: p.relation, element: p.element, set: p.set)
    }
    if c.contains(.SetBuilder) {
      struct P: Decodable {
        let variable: String
        let domain: MathLexExpression?
        let predicate: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .SetBuilder)
      return .setBuilder(variable: p.variable, domain: p.domain, predicate: p.predicate)
    }
    if c.contains(.PowerSet) {
      struct P: Decodable { let set: MathLexExpression }
      return .powerSet(set: try c.decode(P.self, forKey: .PowerSet).set)
    }
    return nil
  }

  // MARK: Tensors, differential forms, and function algebra

  private static func decodeTensorsAndForms(
    from c: KeyedDecodingContainer<MathLexTopKey>
  ) throws -> MathLexExpression? {
    if c.contains(.Tensor) {
      struct P: Decodable {
        let name: String
        let indices: [TensorIndex]
      }
      let p = try c.decode(P.self, forKey: .Tensor)
      return .tensor(name: p.name, indices: p.indices)
    }
    if c.contains(.KroneckerDelta) {
      struct P: Decodable { let indices: [TensorIndex] }
      return .kroneckerDelta(indices: try c.decode(P.self, forKey: .KroneckerDelta).indices)
    }
    if c.contains(.LeviCivita) {
      struct P: Decodable { let indices: [TensorIndex] }
      return .leviCivita(indices: try c.decode(P.self, forKey: .LeviCivita).indices)
    }
    if c.contains(.FunctionSignature) {
      struct P: Decodable {
        let name: String
        let domain, codomain: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .FunctionSignature)
      return .functionSignature(name: p.name, domain: p.domain, codomain: p.codomain)
    }
    if c.contains(.Composition) {
      struct P: Decodable { let outer, inner: MathLexExpression }
      let p = try c.decode(P.self, forKey: .Composition)
      return .composition(outer: p.outer, inner: p.inner)
    }
    if c.contains(.Differential) {
      struct P: Decodable { let `var`: String }
      return .differential(var: try c.decode(P.self, forKey: .Differential).var)
    }
    if c.contains(.WedgeProduct) {
      struct P: Decodable { let left, right: MathLexExpression }
      let p = try c.decode(P.self, forKey: .WedgeProduct)
      return .wedgeProduct(left: p.left, right: p.right)
    }
    if c.contains(.Relation) {
      struct P: Decodable {
        let op: RelationOp
        let left, right: MathLexExpression
      }
      let p = try c.decode(P.self, forKey: .Relation)
      return .relation(op: p.op, left: p.left, right: p.right)
    }
    return nil
}
