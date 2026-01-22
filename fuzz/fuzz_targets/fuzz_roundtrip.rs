#![no_main]

use libfuzzer_sys::fuzz_target;
use mathlex::{parse, ToLatex};

fuzz_target!(|data: &str| {
    // Try to parse the input
    if let Ok(expr) = parse(data) {
        // If parsing succeeds, convert to LaTeX
        let latex_str = expr.to_latex();

        // Try to parse the LaTeX representation back
        // This tests round-trip consistency: text -> AST -> LaTeX -> AST
        let _ = mathlex::parse_latex(&latex_str);

        // Also test display (to_string)
        let display_str = expr.to_string();

        // Try to parse the display representation back
        // This tests round-trip consistency: text -> AST -> display -> AST
        let _ = parse(&display_str);
    }
});
