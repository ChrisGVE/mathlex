# NumericSwift Integration Example

This directory contains reference Swift code showing how NumericSwift can consume
the JSON AST produced by mathlex.

## Architecture

```
[LaTeX/Text Input] → [mathlex Parser] → [JSON AST] → [Swift Decoder] → [Evaluator] → [Result]
```

1. **mathlex Parser** (Rust) parses a LaTeX or plain-text expression and produces
   an `Expression` AST in memory.
2. **JSON AST** is obtained by calling `expr.toJSON()` on the Swift `MathExpression`
   wrapper, which calls through to the Rust `serde_json` serializer.
3. **Swift Decoder** (`MathLexExpression.swift`) decodes the JSON string into a
   native Swift `MathLexExpression` indirect enum using `JSONDecoder`.
4. **Evaluator** (`Evaluator.swift`) walks the decoded AST and computes a `Double`
   result given a variable binding environment.

## File overview

| File | Purpose |
|------|---------|
| `MathLexExpression.swift` | Complete `Decodable` types mirroring the Rust AST |
| `Evaluator.swift` | Working `Double` evaluator for numeric expressions |
| `ComplexEvaluator.swift` | Skeleton showing how Complex evaluation would extend the pattern |

## Usage

```swift
import MathLex
import Foundation

// 1. Parse with mathlex
let expr = try MathExpression.parse("sin(x)^2 + cos(x)^2")

// 2. Serialize to JSON
let json = try expr.toJSON()

// 3. Decode to Swift AST
let data = Data(json.utf8)
let ast = try JSONDecoder().decode(MathLexExpression.self, from: data)

// 4. Evaluate with variable bindings
let evaluator = Evaluator()
let result = try evaluator.evaluate(ast, env: ["x": Double.pi / 4])
// result ≈ 1.0
```

## JSON format

mathlex uses serde's externally-tagged format. Every expression is a single-key
JSON object where the key is the variant name:

```json
{ "Binary": { "op": "Add", "left": { "Variable": "x" }, "right": { "Integer": 1 } } }
```

Unit variants (`Nabla`, `EmptySet`) serialize as bare strings. See
`docs/json-ast-schema.md` in the mathlex repository for the full reference.

## Extending to other numeric types

To evaluate to `Float`, `Decimal`, or a custom number type, copy `Evaluator.swift`
and replace `Double` with your type. Only the leaf conversions and the math
function dispatch need to change — the recursive tree walk is identical.

For complex-number evaluation, see `ComplexEvaluator.swift` for the pattern.
