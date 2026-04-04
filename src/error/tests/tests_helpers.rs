//! Tests for ErrorBuilder, levenshtein, and suggest_function.

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_error_builder_basic() {
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression).build();

        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, None);
        assert_eq!(error.context, None);
    }

    #[test]
    fn test_error_builder_with_span() {
        let span = Span::at(Position::new(1, 5, 4));
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_span(span)
            .build();

        assert_eq!(error.span, Some(span));
    }

    #[test]
    fn test_error_builder_with_position() {
        let pos = Position::new(1, 5, 4);
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_position(pos)
            .build();

        assert_eq!(error.span, Some(Span::at(pos)));
    }

    #[test]
    fn test_error_builder_with_context() {
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .with_context("in function body")
            .build();

        assert_eq!(error.context, Some("in function body".to_string()));
    }

    #[test]
    fn test_error_builder_complete() {
        let pos = Position::new(1, 5, 4);
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .at_position(pos)
            .with_context("in function body")
            .build();

        assert_eq!(error.kind, ParseErrorKind::EmptyExpression);
        assert_eq!(error.span, Some(Span::at(pos)));
        assert_eq!(error.context, Some("in function body".to_string()));
    }

    #[test]
    fn test_error_builder_with_suggestion() {
        let error = ErrorBuilder::new(ParseErrorKind::EmptyExpression)
            .with_suggestion("Did you mean 'sin'?")
            .build();

        assert_eq!(error.suggestion, Some("Did you mean 'sin'?".to_string()));
    }

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein("sin", "sin"), 0);
    }

    #[test]
    fn test_levenshtein_substitution() {
        assert_eq!(levenshtein("sin", "sen"), 1);
    }

    #[test]
    fn test_levenshtein_insertion() {
        assert_eq!(levenshtein("sin", "sign"), 1);
    }

    #[test]
    fn test_levenshtein_deletion() {
        assert_eq!(levenshtein("sign", "sin"), 1);
    }

    #[test]
    fn test_levenshtein_multiple_edits() {
        assert_eq!(levenshtein("cos", "xyz"), 3);
    }

    #[test]
    fn test_levenshtein_empty_strings() {
        assert_eq!(levenshtein("", "sin"), 3);
        assert_eq!(levenshtein("cos", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
    }

    #[test]
    fn test_suggest_function_close_match() {
        assert_eq!(
            suggest_function("sen"),
            Some("Did you mean 'sin'?".to_string())
        );
        assert_eq!(
            suggest_function("coz"),
            Some("Did you mean 'cos'?".to_string())
        );
        assert_eq!(
            suggest_function("sqr"),
            Some("Did you mean 'sqrt'?".to_string())
        );
    }

    #[test]
    fn test_suggest_function_exact_match() {
        assert_eq!(
            suggest_function("sin"),
            Some("Did you mean 'sin'?".to_string())
        );
    }

    #[test]
    fn test_suggest_function_no_match() {
        assert_eq!(suggest_function("xyz"), None);
        assert_eq!(suggest_function("foobar"), None);
    }

    #[test]
    fn test_suggest_function_distance_2() {
        assert_eq!(
            suggest_function("sinn"),
            Some("Did you mean 'sin'?".to_string())
        );
    }

    #[test]
    fn test_suggest_function_distance_3() {
        assert_eq!(suggest_function("zzz"), None);
    }

    #[test]
    fn test_unknown_function_with_suggestion() {
        let error = ParseError::unknown_function("sen", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnknownFunction {
                name: "sen".to_string(),
            }
        );
        assert_eq!(error.suggestion, Some("Did you mean 'sin'?".to_string()));
    }

    #[test]
    fn test_unknown_function_no_suggestion() {
        let error = ParseError::unknown_function("xyz", None);
        assert_eq!(
            error.kind,
            ParseErrorKind::UnknownFunction {
                name: "xyz".to_string(),
            }
        );
        assert_eq!(error.suggestion, None);
    }

    #[test]
    fn test_error_display_with_suggestion() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_suggestion("Did you mean 'sin'?");
        assert_eq!(error.to_string(), "empty expression Did you mean 'sin'?");
    }

    #[test]
    fn test_error_display_with_span_and_suggestion() {
        let pos = Position::new(1, 5, 4);
        let span = Span::at(pos);
        let error = ParseError::new(ParseErrorKind::EmptyExpression, Some(span))
            .with_suggestion("Did you mean 'sin'?");
        assert_eq!(
            error.to_string(),
            "empty expression at 1:5 Did you mean 'sin'?"
        );
    }

    #[test]
    fn test_error_display_with_context_and_suggestion() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_context("while parsing function arguments")
            .with_suggestion("Did you mean 'sin'?");
        assert_eq!(
            error.to_string(),
            "empty expression (while parsing function arguments) Did you mean 'sin'?"
        );
    }

    #[test]
    fn test_with_suggestion_method() {
        let error = ParseError::new(ParseErrorKind::EmptyExpression, None)
            .with_suggestion("Try using 'sin' instead");

        assert_eq!(
            error.suggestion,
            Some("Try using 'sin' instead".to_string())
        );
    }
}
