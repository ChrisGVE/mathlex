import Foundation
import XCTest

@testable import MathLex

/// Tests for the `toJSON()` and `toJSONPretty()` methods on `MathExpression`.
///
/// Each test verifies one or more of the following properties:
/// - The call succeeds without throwing
/// - The returned string is valid JSON (parseable by `JSONSerialization`)
/// - Key structural fields are present in the JSON object graph
/// - Pretty-printed output contains newlines
///
/// JSON shape mirrors the Rust serde representation, e.g.:
/// - `{"Integer":42}`
/// - `{"Variable":"x"}`
/// - `{"Constant":"Pi"}`
/// - `{"Binary":{"op":"Add","left":...,"right":...}}`
final class JSONSerializationTests: XCTestCase {

  // MARK: - Helpers

  /// Decode a JSON string and return the top-level object, or fail the test.
  private func decode(_ json: String, file: StaticString = #file, line: UInt = #line) -> Any? {
    guard let data = json.data(using: .utf8) else {
      XCTFail("JSON string could not be encoded as UTF-8", file: file, line: line)
      return nil
    }
    do {
      return try JSONSerialization.jsonObject(with: data, options: [])
    } catch {
      XCTFail("Invalid JSON: \(error)  —  input: \(json)", file: file, line: line)
      return nil
    }
  }

  // MARK: - Basic Types

  func testIntegerJSON() throws {
    let expr = try MathExpression.parse("42")
    let json = try expr.toJSON()

    XCTAssertFalse(json.isEmpty, "JSON must not be empty")
    let obj = decode(json) as? [String: Any]
    XCTAssertNotNil(obj, "JSON top level must be a dictionary")
    XCTAssertNotNil(obj?["Integer"], "Integer node must have 'Integer' key")
    XCTAssertEqual(obj?["Integer"] as? Int, 42)
  }

  func testFloatJSON() throws {
    let expr = try MathExpression.parse("3.14")
    let json = try expr.toJSON()

    XCTAssertFalse(json.isEmpty)
    XCTAssertNotNil(decode(json), "Float must produce valid JSON")
    XCTAssertTrue(json.contains("Float"), "JSON must contain 'Float' key")
  }

  func testVariableJSON() throws {
    let expr = try MathExpression.parse("x")
    let json = try expr.toJSON()

    XCTAssertFalse(json.isEmpty)
    let obj = decode(json) as? [String: Any]
    XCTAssertNotNil(obj, "JSON top level must be a dictionary")
    XCTAssertEqual(obj?["Variable"] as? String, "x", "Variable name must be 'x'")
  }

  // MARK: - Operations

  func testBinaryAdditionJSON() throws {
    let expr = try MathExpression.parse("2 + 3")
    let json = try expr.toJSON()

    let obj = decode(json) as? [String: Any]
    XCTAssertNotNil(obj, "JSON top level must be a dictionary")
    XCTAssertTrue(json.contains("Binary"), "Binary operation must use 'Binary' key")
    XCTAssertTrue(json.contains("Add"), "Addition must encode op as 'Add'")
  }

  func testFunctionCallJSON() throws {
    let expr = try MathExpression.parse("sin(x)")
    let json = try expr.toJSON()

    XCTAssertNotNil(decode(json), "Function call must produce valid JSON")
    XCTAssertTrue(json.contains("Function"), "Function call must use 'Function' key")
    XCTAssertTrue(json.contains("sin"), "Function name 'sin' must appear in JSON")
  }

  // MARK: - Complex Expressions

  func testPythagoreanIdentityJSON() throws {
    let expr = try MathExpression.parse("sin(x)^2 + cos(x)^2")
    let json = try expr.toJSON()

    XCTAssertNotNil(decode(json), "Pythagorean identity must produce valid JSON")
    XCTAssertTrue(json.contains("sin"), "JSON must reference 'sin'")
    XCTAssertTrue(json.contains("cos"), "JSON must reference 'cos'")
    XCTAssertTrue(json.contains("Binary"), "JSON must contain Binary node")
  }

  func testLatexFractionJSON() throws {
    let expr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
    let json = try expr.toJSON()

    XCTAssertNotNil(decode(json), "LaTeX fraction must produce valid JSON")
    // \frac{1}{2} is a division binary operation
    XCTAssertTrue(json.contains("Binary"), "Fraction must be encoded as Binary")
  }

  // MARK: - Pretty Print

  func testPrettyPrintHasNewlines() throws {
    let expr = try MathExpression.parse("x + 1")
    let pretty = try expr.toJSONPretty()

    XCTAssertTrue(
      pretty.contains("\n"),
      "Pretty-printed JSON must contain at least one newline"
    )
  }

  func testPrettyPrintIsValidJSON() throws {
    let expr = try MathExpression.parse("2 * x + 1")
    let pretty = try expr.toJSONPretty()

    XCTAssertNotNil(decode(pretty), "Pretty-printed JSON must be valid JSON")
  }

  func testCompactAndPrettyDecodeSameTopLevelKeys() throws {
    let expr = try MathExpression.parse("x + y")
    let compact = try expr.toJSON()
    let pretty = try expr.toJSONPretty()

    let compactObj = decode(compact) as? [String: Any]
    let prettyObj = decode(pretty) as? [String: Any]

    XCTAssertNotNil(compactObj, "Compact JSON top level must be a dictionary")
    XCTAssertNotNil(prettyObj, "Pretty JSON top level must be a dictionary")

    // Both must share the same top-level keys
    let compactKeys = Set(compactObj.map { Array($0.keys) } ?? [])
    let prettyKeys = Set(prettyObj.map { Array($0.keys) } ?? [])
    XCTAssertEqual(
      compactKeys, prettyKeys, "Compact and pretty JSON must have identical top-level keys")
  }

  // MARK: - Constants

  func testConstantPiJSON() throws {
    let expr = try MathExpression.parse("pi")
    let json = try expr.toJSON()

    let obj = decode(json) as? [String: Any]
    XCTAssertNotNil(obj, "Constant 'pi' must produce a JSON object")
    XCTAssertTrue(
      json.contains("Constant"),
      "pi must be encoded under 'Constant' key"
    )
    XCTAssertTrue(json.contains("Pi"), "pi must be encoded as 'Pi'")
  }

  func testConstantEJSON() throws {
    let expr = try MathExpression.parse("e")
    let json = try expr.toJSON()

    XCTAssertNotNil(decode(json), "Constant 'e' must produce valid JSON")
    XCTAssertTrue(json.contains("Constant"), "e must be encoded under 'Constant' key")
  }

  func testConstantNaNJSON() throws {
    let expr = try MathExpression.parse("nan")
    let json = try expr.toJSON()

    // NaN must be encoded as a JSON-safe string value, not a bare NaN literal
    XCTAssertNotNil(decode(json), "Constant 'nan' must produce valid JSON")
    XCTAssertTrue(json.contains("Constant"), "nan must be encoded under 'Constant' key")
    XCTAssertTrue(json.contains("NaN"), "nan must be encoded as 'NaN'")
  }

  // MARK: - Round-trip Batch

  func testBatchRoundTrip() throws {
    let inputs = [
      "x + y",
      "sin(x)",
      "cos(pi)",
      "2^10",
      "log(e)",
    ]

    for input in inputs {
      let expr = try MathExpression.parse(input)
      let json = try expr.toJSON()

      XCTAssertFalse(json.isEmpty, "Batch: '\(input)' produced empty JSON")
      XCTAssertNotNil(decode(json), "Batch: '\(input)' produced invalid JSON")
    }
  }
}
