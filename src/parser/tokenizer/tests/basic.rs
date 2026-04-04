//! Basic tokenizer tests: literals, operators, delimiters, relations, span tracking.

#[cfg(test)]
#[allow(clippy::approx_constant)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_tokenize_integer() {
        let tokens = tokenize("42").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Integer(42));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize("3.14").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].value, Token::Float(f) if (f - 3.14).abs() < 0.001));
    }

    #[test]
    fn test_tokenize_scientific_notation() {
        let tokens = tokenize("1.5e-3").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].value, Token::Float(f) if (f - 0.0015).abs() < 0.0001));
    }

    #[test]
    fn test_tokenize_identifier() {
        let tokens = tokenize("x").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
    }

    #[test]
    fn test_tokenize_multi_char_identifier() {
        let tokens = tokenize("theta").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Identifier("theta".to_string()));
    }

    #[test]
    fn test_tokenize_operators() {
        let tokens = tokenize("+ - * / ^ %").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::Plus);
        assert_eq!(tokens[1].value, Token::Minus);
        assert_eq!(tokens[2].value, Token::Star);
        assert_eq!(tokens[3].value, Token::Slash);
        assert_eq!(tokens[4].value, Token::Caret);
        assert_eq!(tokens[5].value, Token::Percent);
    }

    #[test]
    fn test_tokenize_delimiters() {
        let tokens = tokenize("( ) [ ] { }").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::LParen);
        assert_eq!(tokens[1].value, Token::RParen);
        assert_eq!(tokens[2].value, Token::LBracket);
        assert_eq!(tokens[3].value, Token::RBracket);
        assert_eq!(tokens[4].value, Token::LBrace);
        assert_eq!(tokens[5].value, Token::RBrace);
    }

    #[test]
    fn test_tokenize_relations() {
        let tokens = tokenize("= != < <= > >=").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::Equals);
        assert_eq!(tokens[1].value, Token::NotEquals);
        assert_eq!(tokens[2].value, Token::Less);
        assert_eq!(tokens[3].value, Token::LessEq);
        assert_eq!(tokens[4].value, Token::Greater);
        assert_eq!(tokens[5].value, Token::GreaterEq);
    }

    #[test]
    fn test_tokenize_unicode_relations() {
        let tokens = tokenize("≤ ≥ ≠").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::LessEq);
        assert_eq!(tokens[1].value, Token::GreaterEq);
        assert_eq!(tokens[2].value, Token::NotEquals);
    }

    #[test]
    fn test_tokenize_expression() {
        let tokens = tokenize("2 + x * 3.14").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Plus);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::Star);
        assert!(matches!(tokens[4].value, Token::Float(_)));
    }

    #[test]
    fn test_tokenize_function_call() {
        let tokens = tokenize("sin(x)").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].value, Token::Identifier("sin".to_string()));
        assert_eq!(tokens[1].value, Token::LParen);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::RParen);
    }

    #[test]
    fn test_tokenize_factorial() {
        let tokens = tokenize("5!").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Integer(5));
        assert_eq!(tokens[1].value, Token::Bang);
    }

    #[test]
    fn test_tokenize_underscore() {
        let tokens = tokenize("x_1").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[1].value, Token::Underscore);
        assert_eq!(tokens[2].value, Token::Integer(1));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let tokens = tokenize("   ").unwrap();
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_invalid_character() {
        let result = tokenize("@");
        assert!(result.is_err());
    }

    #[test]
    fn test_span_tracking() {
        let tokens = tokenize("x + y").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].span.start.column, 1);
        assert_eq!(tokens[0].span.end.column, 2);
        assert_eq!(tokens[2].span.start.column, 5);
        assert_eq!(tokens[2].span.end.column, 6);
    }

    #[test]
    fn test_tokenize_double_star() {
        let tokens = tokenize("2**3").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::DoubleStar);
        assert_eq!(tokens[2].value, Token::Integer(3));
    }

    #[test]
    fn test_tokenize_star_vs_double_star() {
        let tokens = tokenize("2*3**4").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Star);
        assert_eq!(tokens[2].value, Token::Integer(3));
        assert_eq!(tokens[3].value, Token::DoubleStar);
        assert_eq!(tokens[4].value, Token::Integer(4));
    }
}
