//! Basic tokenizer tests: literals, operators, delimiters, position tracking.

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize_latex("").unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(tokens[0].0, LatexToken::Eof));
    }

    #[test]
    fn test_tokenize_simple_number() {
        let tokens = tokenize_latex("42").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Number("42".to_string()));
    }

    #[test]
    fn test_tokenize_float() {
        let tokens = tokenize_latex("3.14").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Number("3.14".to_string()));
    }

    #[test]
    fn test_tokenize_letter() {
        let tokens = tokenize_latex("x").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
    }

    #[test]
    fn test_tokenize_operators() {
        let input = "+ - * / ^ _ = < >";
        let tokens = tokenize_latex(input).unwrap();
        assert!(matches!(tokens[0].0, LatexToken::Plus));
        assert!(matches!(tokens[1].0, LatexToken::Minus));
        assert!(matches!(tokens[2].0, LatexToken::Star));
        assert!(matches!(tokens[3].0, LatexToken::Slash));
        assert!(matches!(tokens[4].0, LatexToken::Caret));
        assert!(matches!(tokens[5].0, LatexToken::Underscore));
        assert!(matches!(tokens[6].0, LatexToken::Equals));
        assert!(matches!(tokens[7].0, LatexToken::Less));
        assert!(matches!(tokens[8].0, LatexToken::Greater));
    }

    #[test]
    fn test_tokenize_delimiters() {
        let input = "{ } ( ) [ ] |";
        let tokens = tokenize_latex(input).unwrap();
        assert!(matches!(tokens[0].0, LatexToken::LBrace));
        assert!(matches!(tokens[1].0, LatexToken::RBrace));
        assert!(matches!(tokens[2].0, LatexToken::LParen));
        assert!(matches!(tokens[3].0, LatexToken::RParen));
        assert!(matches!(tokens[4].0, LatexToken::LBracket));
        assert!(matches!(tokens[5].0, LatexToken::RBracket));
        assert!(matches!(tokens[6].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_ampersand() {
        let tokens = tokenize_latex("&").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Ampersand));
    }

    #[test]
    fn test_tokenize_comma() {
        let tokens = tokenize_latex(",").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Comma));
    }

    #[test]
    fn test_tokenize_multiple_digits() {
        let tokens = tokenize_latex("123").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Number("123".to_string()));
    }

    #[test]
    fn test_tokenize_float_at_end() {
        let result = tokenize_latex("3.");
        assert!(
            result.is_err() || {
                let tokens = result.unwrap();
                tokens.len() >= 2 && tokens[0].0 == LatexToken::Number("3".to_string())
            }
        );
    }

    #[test]
    fn test_tokenize_whitespace_ignored() {
        let tokens1 = tokenize_latex("x+y").unwrap();
        let tokens2 = tokenize_latex("x + y").unwrap();
        let tokens3 = tokenize_latex("  x  +  y  ").unwrap();
        assert_eq!(tokens1.len(), tokens2.len());
        assert_eq!(tokens1.len(), tokens3.len());
        for i in 0..tokens1.len() {
            assert_eq!(tokens1[i].0, tokens2[i].0);
            assert_eq!(tokens1[i].0, tokens3[i].0);
        }
    }

    #[test]
    fn test_tokenize_position_tracking() {
        let tokens = tokenize_latex("x+y").unwrap();
        assert_eq!(tokens[0].1.start.line, 1);
        assert_eq!(tokens[0].1.start.column, 1);
        assert_eq!(tokens[0].1.start.offset, 0);
        assert_eq!(tokens[1].1.start.line, 1);
        assert_eq!(tokens[1].1.start.column, 2);
        assert_eq!(tokens[1].1.start.offset, 1);
        assert_eq!(tokens[2].1.start.line, 1);
        assert_eq!(tokens[2].1.start.column, 3);
        assert_eq!(tokens[2].1.start.offset, 2);
    }

    #[test]
    fn test_tokenize_multiline_position_tracking() {
        let tokens = tokenize_latex("x\n+\ny").unwrap();
        assert_eq!(tokens[0].1.start.line, 1);
        assert_eq!(tokens[1].1.start.line, 2);
        assert_eq!(tokens[1].1.start.column, 1);
        assert_eq!(tokens[2].1.start.line, 3);
        assert_eq!(tokens[2].1.start.column, 1);
    }

    #[test]
    fn test_error_invalid_command() {
        let result = tokenize_latex(r"\");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_begin_without_brace() {
        let result = tokenize_latex(r"\begin");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_begin_unclosed() {
        let result = tokenize_latex(r"\begin{matrix");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unexpected_character() {
        let result = tokenize_latex("@");
        assert!(result.is_err());
    }
}
