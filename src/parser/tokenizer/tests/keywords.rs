//! Tests for Unicode symbols and keyword token recognition.

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_tokenize_unicode_pi() {
        let tokens = tokenize("2*π").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].value, Token::Integer(2));
        assert_eq!(tokens[1].value, Token::Star);
        assert_eq!(tokens[2].value, Token::Pi);
    }

    #[test]
    fn test_tokenize_unicode_infinity() {
        let tokens = tokenize("∞").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, Token::Infinity);
    }

    #[test]
    fn test_tokenize_unicode_sqrt() {
        let tokens = tokenize("√4").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Sqrt);
        assert_eq!(tokens[1].value, Token::Integer(4));
    }

    #[test]
    fn test_tokenize_unicode_sqrt_with_parens() {
        let tokens = tokenize("√(x+1)").unwrap();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0].value, Token::Sqrt);
        assert_eq!(tokens[1].value, Token::LParen);
        assert_eq!(tokens[2].value, Token::Identifier("x".to_string()));
        assert_eq!(tokens[3].value, Token::Plus);
        assert_eq!(tokens[4].value, Token::Integer(1));
        assert_eq!(tokens[5].value, Token::RParen);
    }

    #[test]
    fn test_tokenize_vector_keywords() {
        let tokens = tokenize("dot cross").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::Dot);
        assert_eq!(tokens[1].value, Token::Cross);
    }

    #[test]
    fn test_tokenize_vector_calculus_keywords() {
        let tokens = tokenize("grad div curl laplacian").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].value, Token::Grad);
        assert_eq!(tokens[1].value, Token::Div);
        assert_eq!(tokens[2].value, Token::Curl);
        assert_eq!(tokens[3].value, Token::Laplacian);
    }

    #[test]
    fn test_tokenize_quantifier_keywords() {
        let tokens = tokenize("forall exists").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].value, Token::ForAll);
        assert_eq!(tokens[1].value, Token::Exists);
    }

    #[test]
    fn test_tokenize_set_keywords() {
        let tokens = tokenize("union intersect in notin").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].value, Token::Union);
        assert_eq!(tokens[1].value, Token::Intersect);
        assert_eq!(tokens[2].value, Token::In);
        assert_eq!(tokens[3].value, Token::NotIn);
    }

    #[test]
    fn test_tokenize_logical_keywords() {
        let tokens = tokenize("and or not implies iff").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].value, Token::And);
        assert_eq!(tokens[1].value, Token::Or);
        assert_eq!(tokens[2].value, Token::Not);
        assert_eq!(tokens[3].value, Token::Implies);
        assert_eq!(tokens[4].value, Token::Iff);
    }
}
