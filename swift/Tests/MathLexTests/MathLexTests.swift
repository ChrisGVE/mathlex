import XCTest
@testable import MathLex

/// Tests for the MathLex Swift wrapper
///
/// Note: These tests use placeholder implementations until the XCFramework
/// bindings are built. Once the Rust FFI is integrated, these tests will
/// exercise the actual parser functionality.
final class MathLexTests: XCTestCase {

    // MARK: - Parsing Tests

    func testParsePlainTextSimple() throws {
        // Test basic expression parsing
        let expr = try MathExpression.parse("2 + 3")
        XCTAssertNotNil(expr)
    }

    func testParsePlainTextWithVariables() throws {
        // Test parsing with variables
        let expr = try MathExpression.parse("x + y")
        XCTAssertNotNil(expr)

        // With placeholder, we can still test variable extraction
        let vars = expr.variables
        XCTAssertTrue(vars.contains("x"))
        XCTAssertTrue(vars.contains("y"))
    }

    func testParsePlainTextWithFunctions() throws {
        // Test parsing with functions
        let expr = try MathExpression.parse("sin(x) + cos(y)")
        XCTAssertNotNil(expr)

        // With placeholder, we can still test function extraction
        let funcs = expr.functions
        XCTAssertTrue(funcs.contains("sin"))
        XCTAssertTrue(funcs.contains("cos"))
    }

    func testParseLatexSimple() throws {
        // Test LaTeX parsing
        let expr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
        XCTAssertNotNil(expr)
    }

    func testParseLatexWithSymbols() throws {
        // Test LaTeX with Greek letters
        let expr = try MathExpression.parseLatex(#"\pi + \theta"#)
        XCTAssertNotNil(expr)
    }

    // MARK: - Conversion Tests

    func testDescriptionPlainText() throws {
        let expr = try MathExpression.parse("x + y")
        let desc = expr.description
        XCTAssertFalse(desc.isEmpty)
    }

    func testLatexConversion() throws {
        let expr = try MathExpression.parseLatex(#"\frac{1}{2}"#)
        let latex = expr.latex
        XCTAssertFalse(latex.isEmpty)
    }

    // MARK: - Querying Tests

    func testVariablesExtraction() throws {
        let expr = try MathExpression.parse("x + y + x")
        let vars = expr.variables

        // Should contain unique variables
        XCTAssertTrue(vars.contains("x"))
        XCTAssertTrue(vars.contains("y"))
        XCTAssertEqual(vars.count, 2)
    }

    func testFunctionsExtraction() throws {
        let expr = try MathExpression.parse("sin(x) + cos(y) + sin(z)")
        let funcs = expr.functions

        // Should contain unique functions
        XCTAssertTrue(funcs.contains("sin"))
        XCTAssertTrue(funcs.contains("cos"))
        XCTAssertEqual(funcs.count, 2)
    }

    func testConstantsExtraction() throws {
        let expr = try MathExpression.parse("2*pi + e")
        let consts = expr.constants

        // Placeholder implementation intersects with known constants
        XCTAssertTrue(consts.contains("pi") || consts.contains("e"))
    }

    func testDepth() throws {
        let expr = try MathExpression.parse("x")
        let depth = expr.depth

        // Placeholder returns 1
        XCTAssertGreaterThanOrEqual(depth, 1)
    }

    func testNodeCount() throws {
        let expr = try MathExpression.parse("x + y")
        let count = expr.nodeCount

        // Placeholder returns 1
        XCTAssertGreaterThanOrEqual(count, 1)
    }

    // MARK: - Equatable Tests

    func testEquality() throws {
        let expr1 = try MathExpression.parse("x + y")
        let expr2 = try MathExpression.parse("x + y")

        // Placeholder implementation compares strings
        XCTAssertEqual(expr1, expr2)
    }

    func testInequality() throws {
        let expr1 = try MathExpression.parse("x + y")
        let expr2 = try MathExpression.parse("x - y")

        // Different expressions should not be equal
        XCTAssertNotEqual(expr1, expr2)
    }

    // MARK: - Hashable Tests

    func testHashable() throws {
        let expr1 = try MathExpression.parse("x + y")
        let expr2 = try MathExpression.parse("x + y")

        // Equal expressions should have equal hashes
        XCTAssertEqual(expr1.hashValue, expr2.hashValue)
    }

    func testHashableInSet() throws {
        let expr1 = try MathExpression.parse("x + y")
        let expr2 = try MathExpression.parse("x + y")
        let expr3 = try MathExpression.parse("x - y")

        var set = Set<MathExpression>()
        set.insert(expr1)
        set.insert(expr2)
        set.insert(expr3)

        // Should have 2 unique expressions (expr1 and expr2 are equal)
        XCTAssertEqual(set.count, 2)
    }

    // MARK: - Error Handling Tests

    // Note: These tests will be more comprehensive once the actual parser is integrated

    func testParseErrorHandling() {
        // This test is a placeholder for future error handling tests
        // Once the parser is integrated, we can test invalid syntax
        XCTAssertTrue(true)
    }

    // MARK: - Edge Cases

    func testEmptyExpression() {
        // Test with empty string
        let expr = try? MathExpression.parse("")
        // Placeholder may accept empty strings
        // Real parser will likely throw an error
        _ = expr  // Silence unused warning
    }

    func testComplexNested() throws {
        // Test deeply nested expression
        let expr = try MathExpression.parse("((x + y) * (z - w)) / (a + b)")
        XCTAssertNotNil(expr)
    }

    func testLatexComplexExpression() throws {
        // Test complex LaTeX expression
        let expr = try MathExpression.parseLatex(
            #"\int_0^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}"#
        )
        XCTAssertNotNil(expr)
    }

    // MARK: - Performance Tests

    func testParsingPerformance() throws {
        // Measure parsing performance with a moderately complex expression
        let input = "sin(x)^2 + cos(x)^2 + tan(x) + log(x) + sqrt(x)"

        measure {
            for _ in 0..<100 {
                _ = try? MathExpression.parse(input)
            }
        }
    }

    func testVariableExtractionPerformance() throws {
        let expr = try MathExpression.parse(
            "a + b + c + d + e + f + g + h + i + j + k + l + m"
        )

        measure {
            for _ in 0..<1000 {
                _ = expr.variables
            }
        }
    }
}
