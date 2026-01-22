#![no_main]

use libfuzzer_sys::fuzz_target;
use mathlex::parse;

fuzz_target!(|data: &str| {
    // Attempt to parse the input string
    // We don't care if parsing succeeds or fails, just that it doesn't panic
    let _ = parse(data);
});
