//! Tests for Position, Span, ParseErrorKind, and ParseError types.

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_position_new() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.line, 5);
        assert_eq!(pos.column, 10);
        assert_eq!(pos.offset, 42);
    }

    #[test]
    fn test_position_start() {
        let pos = Position::start();
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);
    }

    #[test]
    fn test_position_display() {
        let pos = Position::new(5, 10, 42);
        assert_eq!(pos.to_string(), "5:10");
    }

    #[test]
    fn test_position_equality() {
        let pos1 = Position::new(1, 1, 0);
        let pos2 = Position::new(1, 1, 0);
        let pos3 = Position::new(1, 2, 1);

        assert_eq!(pos1, pos2);
        assert_ne!(pos1, pos3);
    }

    #[test]
    fn test_span_new() {
        let start = Position::new(1, 1, 0);
        let end = Position::new(1, 5, 4);
        let span = Span::new(start, end);

        assert_eq!(span.start, start);
        assert_eq!(span.end, end);
    }

    #[test]
    fn test_span_at() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);

        assert_eq!(span.start, pos);
        assert_eq!(span.end, pos);
    }

    #[test]
    fn test_span_start() {
        let span = Span::start();
        assert_eq!(span.start, Position::start());
        assert_eq!(span.end, Position::start());
    }

    #[test]
    fn test_span_display_single_position() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        assert_eq!(span.to_string(), "1:5");
    }

    #[test]
    fn test_span_display_range() {
        let start = Position::new(1, 1, 0);
        let end = Position::new(1, 5, 4);
        let span = Span::new(start, end);
        assert_eq!(span.to_string(), "1:1-1:5");
    }

    #[test]
    fn test_parse_error_kind_unexpected_token() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec!["number".to_string()],
            found: "+".to_string(),
        };
        assert_eq!(kind.to_string(), "unexpected token '+', expected number");
    }

    #[test]
    fn test_parse_error_kind_unexpected_token_multiple_expected() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec!["number".to_string(), "variable".to_string()],
            found: "+".to_string(),
        };
        assert_eq!(
            kind.to_string(),
            "unexpected token '+', expected one of: number, variable"
        );
    }

    #[test]
    fn test_parse_error_kind_unexpected_token_no_expected() {
        let kind = ParseErrorKind::UnexpectedToken {
            expected: vec![],
            found: "+".to_string(),
        };
        assert_eq!(kind.to_string(), "unexpected token '+'");
    }

    #[test]
    fn test_parse_error_kind_unexpected_eof() {
        let kind = ParseErrorKind::UnexpectedEof {
            expected: vec!["number".to_string()],
        };
        assert_eq!(kind.to_string(), "unexpected end of input, expected number");
    }

    #[test]
    fn test_parse_error_kind_unexpected_eof_multiple_expected() {
        let kind = ParseErrorKind::UnexpectedEof {
            expected: vec!["number".to_string(), "variable".to_string()],
        };
        assert_eq!(
            kind.to_string(),
            "unexpected end of input, expected one of: number, variable"
        );
    }

    #[test]
    fn test_parse_error_kind_unmatched_delimiter() {
        let pos = Position::new(1, 5, 4);
        let kind = ParseErrorKind::UnmatchedDelimiter {
            opening: '(',
            position: pos,
        };
        assert_eq!(kind.to_string(), "unmatched opening delimiter '(' at 1:5");
    }

    #[test]
    fn test_parse_error_kind_invalid_number() {
        let kind = ParseErrorKind::InvalidNumber {
            value: "123.45.67".to_string(),
            reason: "multiple decimal points".to_string(),
        };
        assert_eq!(
            kind.to_string(),
            "invalid number '123.45.67': multiple decimal points"
        );
    }

    #[test]
    fn test_parse_error_kind_invalid_latex_command() {
        let kind = ParseErrorKind::InvalidLatexCommand {
            command: r"\unknowncommand".to_string(),
        };
        assert_eq!(kind.to_string(), r"invalid LaTeX command '\unknowncommand'");
    }

    #[test]
    fn test_parse_error_kind_unknown_function() {
        let kind = ParseErrorKind::UnknownFunction {
            name: "unknownfunc".to_string(),
        };
        assert_eq!(kind.to_string(), "unknown function 'unknownfunc'");
    }

    #[test]
    fn test_parse_error_kind_invalid_subscript() {
        let kind = ParseErrorKind::InvalidSubscript {
            reason: "missing expression".to_string(),
        };
        assert_eq!(kind.to_string(), "invalid subscript: missing expression");
    }

    #[test]
    fn test_parse_error_kind_invalid_superscript() {
        let kind = ParseErrorKind::InvalidSuperscript {
            reason: "missing expression".to_string(),
        };
        assert_eq!(kind.to_string(), "invalid superscript: missing expression");
    }

    #[test]
    fn test_parse_error_kind_malformed_matrix() {
        let kind = ParseErrorKind::MalformedMatrix {
            reason: "inconsistent row lengths".to_string(),
        };
        assert_eq!(
            kind.to_string(),
            "malformed matrix: inconsistent row lengths"
        );
    }

    #[test]
    fn test_parse_error_kind_empty_expression() {
        let kind = ParseErrorKind::EmptyExpression;
        assert_eq!(kind.to_string(), "empty expression");
    }

    #[test]
    fn test_parse_error_kind_custom() {
        let kind = ParseErrorKind::Custom("custom error message".to_string());
        assert_eq!(kind.to_string(), "custom error message");
    }

    #[test]
    fn test_parse_error_new() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None);
        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, None);
        assert_eq!(error.context, None);
    }

    #[test]
    fn test_parse_error_with_context() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_context("while parsing function arguments");

        assert_eq!(
            error.context,
            Some("while parsing function arguments".to_string())
        );
    }

    #[test]
    fn test_parse_error_display_no_span() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None);
        assert_eq!(error.to_string(), "empty expression");
    }

    #[test]
    fn test_parse_error_display_with_span() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        let error = ParseError::new(ParseErrorKind::EmptyExpression, Some(span));
        assert_eq!(error.to_string(), "empty expression at 1:5");
    }

    #[test]
    fn test_parse_error_display_with_context() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_context("while parsing function arguments");
        assert_eq!(
            error.to_string(),
            "empty expression (while parsing function arguments)"
        );
    }

    #[test]
    fn test_parse_error_display_with_span_and_context() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        let error = ParseError::new(ParseErrorKind::EmptyExpression, Some(span))
            .with_context("while parsing function arguments");
        assert_eq!(
            error.to_string(),
            "empty expression at 1:5 (while parsing function arguments)"
        );
    }

    #[test]
    fn test_parse_error_unexpected_token() {
        let error = ParseError::unexpected_token(vec!["number"], "+", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnexpectedToken {
                expected: vec!["number".to_string()],
                found: "+".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let error = ParseError::unexpected_eof(vec!["closing parenthesis"], None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnexpectedEof {
                expected: vec!["closing parenthesis".to_string()],
            }
        );
    }

    #[test]
    fn test_parse_error_unmatched_delimiter() {
        let pos = Position::new(1, 1, 0);
        let error = ParseError::unmatched_delimiter('(', pos, None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnmatchedDelimiter {
                opening: '(',
                position: pos,
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_number() {
        let error = ParseError::invalid_number("123.45.67", "multiple decimal points", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidNumber {
                value: "123.45.67".to_string(),
                reason: "multiple decimal points".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_latex_command() {
        let error = ParseError::invalid_latex_command(r"\unknowncommand", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidLatexCommand {
                command: r"\unknowncommand".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_unknown_function() {
        let error = ParseError::unknown_function("unknownfunc", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnknownFunction {
                name: "unknownfunc".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_subscript() {
        let error = ParseError::invalid_subscript("missing expression", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidSubscript {
                reason: "missing expression".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_invalid_superscript() {
        let error = ParseError::invalid_superscript("missing expression", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::InvalidSuperscript {
                reason: "missing expression".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_malformed_matrix() {
        let error = ParseError::malformed_matrix("inconsistent row lengths", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::MalformedMatrix {
                reason: "inconsistent row lengths".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_error_empty_expression() {
        let error = ParseError::empty_expression(None);
        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
    }

    #[test]
    fn test_parse_error_custom() {
        let error = ParseError::custom("custom error message", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::Custom("custom error message".to_string())
        );
    }

    #[test]
    fn test_parse_result_ok() {
        let result: ParseResult<i32> = Ok(42);
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_parse_result_err() {
        let result: ParseResult<i32> = Err(ParseError::empty_expression(None));
        assert!(result.is_err());
    }
}
