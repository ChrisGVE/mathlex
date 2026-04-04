//! Tests for LaTeX command tokenization.

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::*;

    #[test]
    fn test_tokenize_command() {
        let tokens = tokenize_latex(r"\frac").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("frac".to_string()));
    }

    #[test]
    fn test_tokenize_greek_letter() {
        let tokens = tokenize_latex(r"\alpha").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("alpha".to_string()));
    }

    #[test]
    fn test_tokenize_special_to() {
        let tokens = tokenize_latex(r"\to").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::To));
    }

    #[test]
    fn test_tokenize_special_infty() {
        let tokens = tokenize_latex(r"\infty").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Infty));
    }

    #[test]
    fn test_tokenize_double_backslash() {
        let tokens = tokenize_latex(r"\\").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::DoubleBackslash));
    }

    #[test]
    fn test_tokenize_begin_env() {
        let tokens = tokenize_latex(r"\begin{matrix}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::BeginEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_end_env() {
        let tokens = tokenize_latex(r"\end{matrix}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::EndEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_left_paren() {
        let tokens = tokenize_latex(r"\left(").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::LParen));
    }

    #[test]
    fn test_tokenize_right_paren() {
        let tokens = tokenize_latex(r"\right)").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::RParen));
    }

    #[test]
    fn test_tokenize_left_bracket() {
        let tokens = tokenize_latex(r"\left[").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::LBracket));
    }

    #[test]
    fn test_tokenize_right_bracket() {
        let tokens = tokenize_latex(r"\right]").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::RBracket));
    }

    #[test]
    fn test_tokenize_left_pipe() {
        let tokens = tokenize_latex(r"\left|").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_right_pipe() {
        let tokens = tokenize_latex(r"\right|").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Pipe));
    }

    #[test]
    fn test_tokenize_cdot() {
        let tokens = tokenize_latex(r"\cdot").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Cdot));
    }

    #[test]
    fn test_tokenize_times() {
        let tokens = tokenize_latex(r"\times").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Cross));
    }

    #[test]
    fn test_tokenize_mathrm_j() {
        let tokens = tokenize_latex(r"\mathrm{j}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('j'));
    }

    #[test]
    fn test_tokenize_mathrm_k() {
        let tokens = tokenize_latex(r"\mathrm{k}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('k'));
    }

    #[test]
    fn test_bare_j_is_letter() {
        let tokens = tokenize_latex("j").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('j'));
    }

    #[test]
    fn test_bare_k_is_letter() {
        let tokens = tokenize_latex("k").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Letter('k'));
    }

    #[test]
    fn test_mathrm_i_still_works() {
        let tokens = tokenize_latex(r"\mathrm{i}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('i'));
    }

    #[test]
    fn test_mathrm_e_still_works() {
        let tokens = tokenize_latex(r"\mathrm{e}").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::ExplicitConstant('e'));
    }

    #[test]
    fn test_tokenize_forall() {
        let tokens = tokenize_latex(r"\forall").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ForAll));
    }

    #[test]
    fn test_tokenize_exists() {
        let tokens = tokenize_latex(r"\exists").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Exists));
    }

    #[test]
    fn test_tokenize_land() {
        let tokens = tokenize_latex(r"\land").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Land));
    }

    #[test]
    fn test_tokenize_lor() {
        let tokens = tokenize_latex(r"\lor").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lor));
    }

    #[test]
    fn test_tokenize_lnot() {
        let tokens = tokenize_latex(r"\lnot").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lnot));
    }

    #[test]
    fn test_tokenize_neg_alias() {
        let tokens = tokenize_latex(r"\neg").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Lnot));
    }

    #[test]
    fn test_tokenize_implies() {
        let tokens = tokenize_latex(r"\implies").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Implies));
    }

    #[test]
    fn test_tokenize_iff() {
        let tokens = tokenize_latex(r"\iff").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::Iff));
    }

    #[test]
    fn test_tokenize_in() {
        let tokens = tokenize_latex(r"\in").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::In));
    }

    #[test]
    fn test_tokenize_notin() {
        let tokens = tokenize_latex(r"\notin").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::NotIn));
    }

    #[test]
    fn test_tokenize_double_integral() {
        let tokens = tokenize_latex(r"\iint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::DoubleIntegral));
    }

    #[test]
    fn test_tokenize_triple_integral() {
        let tokens = tokenize_latex(r"\iiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::TripleIntegral));
    }

    #[test]
    fn test_tokenize_quad_integral() {
        let tokens = tokenize_latex(r"\iiiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::QuadIntegral));
    }

    #[test]
    fn test_tokenize_closed_integral() {
        let tokens = tokenize_latex(r"\oint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedIntegral));
    }

    #[test]
    fn test_tokenize_closed_surface() {
        let tokens = tokenize_latex(r"\oiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedSurface));
    }

    #[test]
    fn test_tokenize_closed_volume() {
        let tokens = tokenize_latex(r"\oiiint").unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(tokens[0].0, LatexToken::ClosedVolume));
    }

    #[test]
    fn test_tokenize_single_int_still_works() {
        let tokens = tokenize_latex(r"\int").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].0, LatexToken::Command("int".to_string()));
    }

    #[test]
    fn test_tokenize_integral() {
        let tokens = tokenize_latex(r"\int_0^\infty").unwrap();
        assert_eq!(tokens[0].0, LatexToken::Command("int".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::Underscore));
        assert_eq!(tokens[2].0, LatexToken::Number("0".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::Caret));
        assert!(matches!(tokens[4].0, LatexToken::Infty));
    }

    #[test]
    fn test_tokenize_sqrt() {
        let tokens = tokenize_latex(r"\sqrt{2}").unwrap();
        assert_eq!(tokens[0].0, LatexToken::Command("sqrt".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LBrace));
        assert_eq!(tokens[2].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::RBrace));
    }

    #[test]
    fn test_tokenize_frac_expression() {
        let tokens = tokenize_latex(r"\frac{1}{2}").unwrap();
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[0].0, LatexToken::Command("frac".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LBrace));
        assert_eq!(tokens[2].0, LatexToken::Number("1".to_string()));
        assert!(matches!(tokens[3].0, LatexToken::RBrace));
        assert!(matches!(tokens[4].0, LatexToken::LBrace));
        assert_eq!(tokens[5].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::RBrace));
        assert!(matches!(tokens[7].0, LatexToken::Eof));
    }

    #[test]
    fn test_tokenize_superscript() {
        let tokens = tokenize_latex("x^2").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[1].0, LatexToken::Caret));
        assert_eq!(tokens[2].0, LatexToken::Number("2".to_string()));
    }

    #[test]
    fn test_tokenize_subscript() {
        let tokens = tokenize_latex("x_i").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[1].0, LatexToken::Underscore));
        assert_eq!(tokens[2].0, LatexToken::Letter('i'));
    }

    #[test]
    fn test_tokenize_complex_expression() {
        let tokens = tokenize_latex(r"\sin(x) + \cos(y)").unwrap();
        assert_eq!(tokens[0].0, LatexToken::Command("sin".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::LParen));
        assert_eq!(tokens[2].0, LatexToken::Letter('x'));
        assert!(matches!(tokens[3].0, LatexToken::RParen));
        assert!(matches!(tokens[4].0, LatexToken::Plus));
        assert_eq!(tokens[5].0, LatexToken::Command("cos".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::LParen));
        assert_eq!(tokens[7].0, LatexToken::Letter('y'));
        assert!(matches!(tokens[8].0, LatexToken::RParen));
    }

    #[test]
    fn test_tokenize_matrix() {
        let input = r"\begin{matrix}
            1 & 2 \\
            3 & 4
        \end{matrix}";
        let tokens = tokenize_latex(input).unwrap();
        assert_eq!(tokens[0].0, LatexToken::BeginEnv("matrix".to_string()));
        assert_eq!(tokens[1].0, LatexToken::Number("1".to_string()));
        assert!(matches!(tokens[2].0, LatexToken::Ampersand));
        assert_eq!(tokens[3].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[4].0, LatexToken::DoubleBackslash));
        assert_eq!(tokens[5].0, LatexToken::Number("3".to_string()));
        assert!(matches!(tokens[6].0, LatexToken::Ampersand));
        assert_eq!(tokens[7].0, LatexToken::Number("4".to_string()));
        assert_eq!(tokens[8].0, LatexToken::EndEnv("matrix".to_string()));
    }

    #[test]
    fn test_tokenize_cdot_multiplication() {
        let tokens = tokenize_latex(r"a \cdot b").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Letter('a'));
        assert!(matches!(tokens[1].0, LatexToken::Cdot));
        assert_eq!(tokens[2].0, LatexToken::Letter('b'));
    }

    #[test]
    fn test_tokenize_times_multiplication() {
        let tokens = tokenize_latex(r"2 \times 3").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].0, LatexToken::Number("2".to_string()));
        assert!(matches!(tokens[1].0, LatexToken::Cross));
        assert_eq!(tokens[2].0, LatexToken::Number("3".to_string()));
    }
}
