//! Test suite for the display module, split into focused sub-files.

#[allow(clippy::approx_constant)]
mod tests_types {
    include!("tests_types.rs");
}

#[allow(clippy::approx_constant)]
mod tests_expr_basic {
    include!("tests_expr_basic.rs");
}

mod tests_expr_calculus {
    include!("tests_expr_calculus.rs");
}

mod tests_precedence {
    include!("tests_precedence.rs");
}
