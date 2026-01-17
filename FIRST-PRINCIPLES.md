# First Principles - mathlex

## Principle 1: Test Driven Development

**Philosophy**: Systematic TDD - write unit test immediately after each logical unit of code.

**Implementation implications**:

- Each logical unit (function, object, method) needs at least one unit test
- Cover edge cases and validation errors (multiple tests per unit)
- Run tests after atomic changes; amend tests only after first run
- Use LSP to identify calling/called code relationships

## Principle 2: Leverage Existing Solutions

**Philosophy**: Reuse mature, well-maintained libraries rather than reinventing functionality.

**Implementation Implications**:

- Prefer established, actively maintained libraries with strong community support
- Choose mature solutions with proven track record (but not stale/unmaintained)
- Follow standard protocols and interfaces when available
- Ensure compatibility with existing toolchains and ecosystems
- Evaluate library health: recent updates, active issues/PRs, documentation quality
- Align with industry best practices and conventions

## Principle 3: Pure Parsing - No Evaluation

**Philosophy**: mathlex exclusively parses mathematical notation into AST representation; it never evaluates, simplifies, or performs mathematical operations.

**Implementation implications**:

- AST nodes represent syntax, not computed values
- Calculus constructs (derivatives, integrals, limits) are structural representations only
- Client libraries (thales, NumericSwift) are responsible for interpretation
- Never include functions that compute numerical results
- Reject any feature request that requires mathematical computation

## Principle 4: AST as Universal Contract

**Philosophy**: The AST definition is the contract between mathlex and all consumers; it must be stable, complete, and unambiguous.

**Implementation implications**:

- AST changes require careful consideration of downstream impact
- Every AST node must have clear semantics documented
- Both plain text and LaTeX parsers must produce identical AST for equivalent expressions
- Round-trip consistency: parse -> to_string -> parse yields equivalent AST
- All AST variants must be serializable (for debugging and persistence)

## Principle 5: Input Format Agnosticism

**Philosophy**: The same mathematical concept should parse to the same AST regardless of whether input is LaTeX or plain text.

**Implementation implications**:

- `parse("sin(x)")` and `parse_latex(r"\sin{x}")` yield identical Expression::Function nodes
- Both parsers share common AST construction logic where possible
- Test coverage includes cross-format equivalence tests
- Error messages should be format-aware but AST output must be format-agnostic

## Principle 6: Graceful Error Handling

**Philosophy**: Parse errors should be informative, localized, and recoverable where possible.

**Implementation implications**:

- Every error includes position information (line, column, offset)
- Error messages describe what was expected vs. what was found
- Common mistakes get specific, helpful error messages
- Parser attempts recovery to report multiple errors when feasible
- Errors never panic; always return Result types

## Principle 7: Zero Consumer Dependencies

**Philosophy**: mathlex must never depend on libraries that consume it (thales, NumericSwift, or future consumers).

**Implementation implications**:

- No imports from thales or NumericSwift crates
- No feature flags that pull in consumer-specific code
- AST design decisions made independently of specific consumer needs
- Changes to consumers never require changes to mathlex
- mathlex can be used by any Rust/Swift project without transitive dependencies
