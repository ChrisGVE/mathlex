#![no_main]

use libfuzzer_sys::fuzz_target;
use mathlex::parse_latex;

fuzz_target!(|data: &str| {
    // Attempt to parse the input string as LaTeX
    // We don't care if parsing succeeds or fails, just that it doesn't panic
    let _ = parse_latex(data);
});
