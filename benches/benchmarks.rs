//! Benchmark suite for mathlex parser and utilities
//!
//! This benchmark suite measures performance of:
//! - Text parser: simple, medium, complex expressions, and new features (grad, div, curl, logical)
//! - LaTeX parser: fractions, integrals, matrices, vectors, vector calculus, quaternions, sets, logic
//! - Realistic expressions: Maxwell's equations, Green's and Stokes' theorems, nested vector ops
//! - Utilities: find_variables, find_functions, metrics, to_string, to_latex, substitute

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mathlex::{parse, parse_latex, ToLatex};

/// Benchmark plain text parser with simple expressions
fn benchmark_text_parser_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_parser_simple");

    group.bench_function("literal_integer", |b| b.iter(|| parse(black_box("42"))));

    group.bench_function("literal_float", |b| b.iter(|| parse(black_box("3.14159"))));

    group.bench_function("variable", |b| b.iter(|| parse(black_box("x"))));

    group.bench_function("addition", |b| b.iter(|| parse(black_box("2 + 3"))));

    group.bench_function("multiplication", |b| b.iter(|| parse(black_box("2 * 3"))));

    group.bench_function("power", |b| b.iter(|| parse(black_box("x^2"))));

    group.finish();
}

/// Benchmark plain text parser with medium complexity expressions
fn benchmark_text_parser_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_parser_medium");

    group.bench_function("polynomial", |b| {
        b.iter(|| parse(black_box("x^2 + 2*x + 1")))
    });

    group.bench_function("nested_operations", |b| {
        b.iter(|| parse(black_box("(a + b) * (c - d)")))
    });

    group.bench_function("function_call", |b| b.iter(|| parse(black_box("sin(x)"))));

    group.bench_function("multiple_functions", |b| {
        b.iter(|| parse(black_box("sin(x) + cos(y)")))
    });

    group.bench_function("division_fraction", |b| {
        b.iter(|| parse(black_box("(x + 1) / (x - 1)")))
    });

    group.bench_function("exponentiation_chain", |b| {
        b.iter(|| parse(black_box("2^3^4")))
    });

    group.finish();
}

/// Benchmark plain text parser with complex expressions
fn benchmark_text_parser_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_parser_complex");

    group.bench_function("nested_functions", |b| {
        b.iter(|| parse(black_box("sin(cos(tan(x)))")))
    });

    group.bench_function("long_polynomial", |b| {
        b.iter(|| parse(black_box("x^5 + 3*x^4 - 2*x^3 + 7*x^2 - 5*x + 11")))
    });

    group.bench_function("complex_mixed", |b| {
        b.iter(|| parse(black_box("sin(x^2) + cos(y) * exp(-z) / (1 + tan(w))")))
    });

    group.bench_function("deeply_nested", |b| {
        b.iter(|| parse(black_box("((((a + b) * c) - d) / e) ^ f")))
    });

    group.bench_function("function_with_multiple_args", |b| {
        b.iter(|| parse(black_box("max(x, y, z, w)")))
    });

    group.bench_function("equation", |b| {
        b.iter(|| parse(black_box("x^2 + 2*x + 1 = 0")))
    });

    group.finish();
}

/// Benchmark LaTeX parser with fractions
fn benchmark_latex_parser_fractions(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_fractions");

    group.bench_function("simple_fraction", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{1}{2}")))
    });

    group.bench_function("fraction_with_variables", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{x + 1}{x - 1}")))
    });

    group.bench_function("nested_fractions", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{\frac{1}{2}}{\frac{3}{4}}")))
    });

    group.bench_function("complex_fraction", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{x^2 + 2x + 1}{x^2 - 1}")))
    });

    group.finish();
}

/// Benchmark LaTeX parser with integrals
fn benchmark_latex_parser_integrals(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_integrals");

    group.bench_function("indefinite_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\int x \, dx")))
    });

    group.bench_function("definite_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\int_{0}^{1} x \, dx")))
    });

    group.bench_function("integral_with_function", |b| {
        b.iter(|| parse_latex(black_box(r"\int \sin(x) \, dx")))
    });

    group.bench_function("complex_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\int_{0}^{\infty} e^{-x^2} \, dx")))
    });

    group.finish();
}

/// Benchmark LaTeX parser with matrices
fn benchmark_latex_parser_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_matrices");

    group.bench_function("vector_2d", |b| {
        b.iter(|| parse_latex(black_box(r"\begin{pmatrix} 1 \\ 2 \end{pmatrix}")))
    });

    group.bench_function("vector_3d", |b| {
        b.iter(|| parse_latex(black_box(r"\begin{pmatrix} x \\ y \\ z \end{pmatrix}")))
    });

    group.bench_function("matrix_2x2", |b| {
        b.iter(|| parse_latex(black_box(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}")))
    });

    group.bench_function("matrix_3x3", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}",
            ))
        })
    });

    group.bench_function("matrix_with_expressions", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\begin{pmatrix} x + 1 & y \\ z & w - 1 \end{pmatrix}",
            ))
        })
    });

    group.finish();
}

/// Benchmark LaTeX parser with mixed complex expressions
fn benchmark_latex_parser_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_complex");

    group.bench_function("quadratic_formula", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")))
    });

    group.bench_function("summation", |b| {
        b.iter(|| parse_latex(black_box(r"\sum_{i=1}^{n} i^2")))
    });

    group.bench_function("product", |b| {
        b.iter(|| parse_latex(black_box(r"\prod_{i=1}^{n} i")))
    });

    group.bench_function("limit", |b| {
        b.iter(|| parse_latex(black_box(r"\lim_{x \to 0} \frac{\sin(x)}{x}")))
    });

    group.bench_function("derivative", |b| {
        b.iter(|| parse_latex(black_box(r"\frac{d}{dx} x^2")))
    });

    group.finish();
}

/// Benchmark utility functions: find_variables
fn benchmark_utilities_find_variables(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities_find_variables");

    // Pre-parse expressions for fair comparison
    let simple = parse("x + y").unwrap();
    let medium = parse("sin(x) + cos(y) * exp(z)").unwrap();
    let complex = parse("x^2 + y^2 + z^2 + w^2 + u^2 + v^2").unwrap();

    group.bench_function("simple", |b| b.iter(|| black_box(&simple).find_variables()));

    group.bench_function("medium", |b| b.iter(|| black_box(&medium).find_variables()));

    group.bench_function("complex", |b| {
        b.iter(|| black_box(&complex).find_variables())
    });

    group.finish();
}

/// Benchmark utility functions: find_functions
fn benchmark_utilities_find_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities_find_functions");

    // Pre-parse expressions
    let simple = parse("sin(x)").unwrap();
    let medium = parse("sin(x) + cos(y) + tan(z)").unwrap();
    let complex = parse("sin(cos(tan(x)))").unwrap();

    group.bench_function("simple", |b| b.iter(|| black_box(&simple).find_functions()));

    group.bench_function("medium", |b| b.iter(|| black_box(&medium).find_functions()));

    group.bench_function("complex", |b| {
        b.iter(|| black_box(&complex).find_functions())
    });

    group.finish();
}

/// Benchmark utility functions: depth and node_count
fn benchmark_utilities_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities_metrics");

    // Pre-parse expressions with varying depth
    let shallow = parse("x + y").unwrap();
    let medium = parse("(x + y) * (z - w)").unwrap();
    let deep = parse("((((a + b) * c) - d) / e) ^ f").unwrap();

    group.bench_function("depth_shallow", |b| b.iter(|| black_box(&shallow).depth()));

    group.bench_function("depth_medium", |b| b.iter(|| black_box(&medium).depth()));

    group.bench_function("depth_deep", |b| b.iter(|| black_box(&deep).depth()));

    group.bench_function("node_count_shallow", |b| {
        b.iter(|| black_box(&shallow).node_count())
    });

    group.bench_function("node_count_medium", |b| {
        b.iter(|| black_box(&medium).node_count())
    });

    group.bench_function("node_count_deep", |b| {
        b.iter(|| black_box(&deep).node_count())
    });

    group.finish();
}

/// Benchmark Display trait (to_string)
fn benchmark_display_to_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("display_to_string");

    // Pre-parse expressions
    let simple = parse("x + y").unwrap();
    let medium = parse("sin(x^2) + cos(y)").unwrap();
    let complex = parse("(a + b) * (c - d) / (e + f)").unwrap();

    group.bench_function("simple", |b| b.iter(|| format!("{}", black_box(&simple))));

    group.bench_function("medium", |b| b.iter(|| format!("{}", black_box(&medium))));

    group.bench_function("complex", |b| b.iter(|| format!("{}", black_box(&complex))));

    group.finish();
}

/// Benchmark ToLatex trait
fn benchmark_to_latex(c: &mut Criterion) {
    let mut group = c.benchmark_group("to_latex");

    // Pre-parse expressions
    let simple = parse("x + y").unwrap();
    let fraction = parse("(x + 1) / (x - 1)").unwrap();
    let complex = parse("sin(x^2) + cos(y) * exp(z)").unwrap();

    group.bench_function("simple", |b| b.iter(|| black_box(&simple).to_latex()));

    group.bench_function("fraction", |b| b.iter(|| black_box(&fraction).to_latex()));

    group.bench_function("complex", |b| b.iter(|| black_box(&complex).to_latex()));

    group.finish();
}

/// Benchmark substitute utility
fn benchmark_utilities_substitute(c: &mut Criterion) {
    let mut group = c.benchmark_group("utilities_substitute");

    // Pre-parse expressions
    let simple = parse("x + 2").unwrap();
    let medium = parse("x^2 + 2*x + 1").unwrap();
    let complex = parse("sin(x) + cos(x) * exp(x)").unwrap();
    let replacement = parse("y + 1").unwrap();

    group.bench_function("simple", |b| {
        b.iter(|| black_box(&simple).substitute(black_box("x"), black_box(&replacement)))
    });

    group.bench_function("medium", |b| {
        b.iter(|| black_box(&medium).substitute(black_box("x"), black_box(&replacement)))
    });

    group.bench_function("complex", |b| {
        b.iter(|| black_box(&complex).substitute(black_box("x"), black_box(&replacement)))
    });

    group.finish();
}

/// Benchmark vector notation parsing
fn benchmark_latex_parser_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_vectors");

    group.bench_function("bold_vector", |b| {
        b.iter(|| parse_latex(black_box(r"\mathbf{F}")))
    });

    group.bench_function("arrow_vector", |b| {
        b.iter(|| parse_latex(black_box(r"\vec{v}")))
    });

    group.bench_function("hat_vector", |b| {
        b.iter(|| parse_latex(black_box(r"\hat{n}")))
    });

    group.bench_function("underline_vector", |b| {
        b.iter(|| parse_latex(black_box(r"\underline{a}")))
    });

    group.finish();
}

/// Benchmark vector calculus operations
fn benchmark_latex_parser_vector_calculus(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_vector_calculus");

    group.bench_function("gradient", |b| {
        b.iter(|| parse_latex(black_box(r"\nabla f")))
    });

    group.bench_function("divergence", |b| {
        b.iter(|| parse_latex(black_box(r"\nabla \cdot \mathbf{F}")))
    });

    group.bench_function("curl", |b| {
        b.iter(|| parse_latex(black_box(r"\nabla \times \mathbf{F}")))
    });

    group.bench_function("laplacian", |b| {
        b.iter(|| parse_latex(black_box(r"\nabla^2 f")))
    });

    group.bench_function("dot_product", |b| {
        b.iter(|| parse_latex(black_box(r"\mathbf{a} \cdot \mathbf{b}")))
    });

    group.bench_function("cross_product", |b| {
        b.iter(|| parse_latex(black_box(r"\mathbf{a} \times \mathbf{b}")))
    });

    group.finish();
}

/// Benchmark multiple integrals
fn benchmark_latex_parser_multiple_integrals(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_multiple_integrals");

    group.bench_function("double_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\iint f \, dA")))
    });

    group.bench_function("double_integral_bounds", |b| {
        b.iter(|| parse_latex(black_box(r"\iint_{R} f(x,y) \, dx\,dy")))
    });

    group.bench_function("triple_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\iiint f \, dV")))
    });

    group.bench_function("closed_integral", |b| {
        b.iter(|| parse_latex(black_box(r"\oint \mathbf{F} \cdot d\mathbf{r}")))
    });

    group.finish();
}

/// Benchmark quaternion expressions
fn benchmark_latex_parser_quaternions(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_quaternions");

    group.bench_function("simple_quaternion", |b| {
        b.iter(|| parse_latex(black_box(r"1 + 2i + 3j + 4k")))
    });

    group.bench_function("quaternion_with_constants", |b| {
        b.iter(|| parse_latex(black_box(r"a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}")))
    });

    group.bench_function("quaternion_multiplication", |b| {
        b.iter(|| parse_latex(black_box(r"(i + j) \cdot (j + k)")))
    });

    group.finish();
}

/// Benchmark set operations
fn benchmark_latex_parser_sets(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_sets");

    group.bench_function("union", |b| {
        b.iter(|| parse_latex(black_box(r"A \cup B")))
    });

    group.bench_function("intersection", |b| {
        b.iter(|| parse_latex(black_box(r"A \cap B")))
    });

    group.bench_function("set_difference", |b| {
        b.iter(|| parse_latex(black_box(r"A \setminus B")))
    });

    group.bench_function("membership", |b| {
        b.iter(|| parse_latex(black_box(r"x \in S")))
    });

    group.bench_function("subset", |b| {
        b.iter(|| parse_latex(black_box(r"A \subseteq B")))
    });

    group.bench_function("number_set", |b| {
        b.iter(|| parse_latex(black_box(r"x \in \mathbb{R}")))
    });

    group.finish();
}

/// Benchmark quantifiers and logical operations
fn benchmark_latex_parser_logic(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_logic");

    group.bench_function("forall", |b| {
        b.iter(|| parse_latex(black_box(r"\forall x, P(x)")))
    });

    group.bench_function("exists", |b| {
        b.iter(|| parse_latex(black_box(r"\exists x, P(x)")))
    });

    group.bench_function("logical_and", |b| {
        b.iter(|| parse_latex(black_box(r"P \land Q")))
    });

    group.bench_function("logical_or", |b| {
        b.iter(|| parse_latex(black_box(r"P \lor Q")))
    });

    group.bench_function("implication", |b| {
        b.iter(|| parse_latex(black_box(r"P \implies Q")))
    });

    group.bench_function("biconditional", |b| {
        b.iter(|| parse_latex(black_box(r"P \iff Q")))
    });

    group.bench_function("negation", |b| {
        b.iter(|| parse_latex(black_box(r"\neg P")))
    });

    group.finish();
}

/// Benchmark complex real-world expressions
fn benchmark_latex_parser_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_realistic");

    // Maxwell's equation (divergence)
    group.bench_function("maxwell_divergence", |b| {
        b.iter(|| parse_latex(black_box(r"\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}")))
    });

    // Maxwell's equation (curl)
    group.bench_function("maxwell_curl", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
            ))
        })
    });

    // Green's theorem
    group.bench_function("greens_theorem", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_R (\nabla \times \mathbf{F}) \cdot \hat{n} \, dA"
            ))
        })
    });

    // Stokes' theorem
    group.bench_function("stokes_theorem", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S} = \oint_{\partial S} \mathbf{F} \cdot d\mathbf{r}"
            ))
        })
    });

    // Nested vector calculus
    group.bench_function("nested_vector_ops", |b| {
        b.iter(|| {
            parse_latex(black_box(
                r"\nabla \cdot (\nabla \times \mathbf{F}) = 0",
            ))
        })
    });

    group.finish();
}

/// Benchmark plain text parser with new features
fn benchmark_text_parser_new_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_parser_new_features");

    group.bench_function("gradient_keyword", |b| {
        b.iter(|| parse(black_box("grad(f)")))
    });

    group.bench_function("divergence_keyword", |b| {
        b.iter(|| parse(black_box("div(F)")))
    });

    group.bench_function("curl_keyword", |b| {
        b.iter(|| parse(black_box("curl(F)")))
    });

    group.bench_function("logical_and", |b| {
        b.iter(|| parse(black_box("P and Q")))
    });

    group.bench_function("logical_or", |b| {
        b.iter(|| parse(black_box("P or Q")))
    });

    group.bench_function("implication", |b| {
        b.iter(|| parse(black_box("P implies Q")))
    });

    group.bench_function("forall", |b| {
        b.iter(|| parse(black_box("forall x, P(x)")))
    });

    group.bench_function("exists", |b| {
        b.iter(|| parse(black_box("exists x, P(x)")))
    });

    group.finish();
}

/// Benchmark matrix operations
fn benchmark_latex_parser_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("latex_parser_matrix_ops");

    group.bench_function("determinant", |b| {
        b.iter(|| parse_latex(black_box(r"\det(A)")))
    });

    group.bench_function("trace", |b| {
        b.iter(|| parse_latex(black_box(r"\tr(A)")))
    });

    group.bench_function("rank", |b| {
        b.iter(|| parse_latex(black_box(r"\rank(A)")))
    });

    group.bench_function("inverse", |b| {
        b.iter(|| parse_latex(black_box(r"A^{-1}")))
    });

    group.bench_function("transpose", |b| {
        b.iter(|| parse_latex(black_box(r"A^T")))
    });

    group.bench_function("conjugate_transpose", |b| {
        b.iter(|| parse_latex(black_box(r"A^\dagger")))
    });

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    benchmark_text_parser_simple,
    benchmark_text_parser_medium,
    benchmark_text_parser_complex,
    benchmark_text_parser_new_features,
    benchmark_latex_parser_fractions,
    benchmark_latex_parser_integrals,
    benchmark_latex_parser_multiple_integrals,
    benchmark_latex_parser_matrices,
    benchmark_latex_parser_matrix_ops,
    benchmark_latex_parser_vectors,
    benchmark_latex_parser_vector_calculus,
    benchmark_latex_parser_quaternions,
    benchmark_latex_parser_sets,
    benchmark_latex_parser_logic,
    benchmark_latex_parser_complex,
    benchmark_latex_parser_realistic,
    benchmark_utilities_find_variables,
    benchmark_utilities_find_functions,
    benchmark_utilities_metrics,
    benchmark_display_to_string,
    benchmark_to_latex,
    benchmark_utilities_substitute,
);

criterion_main!(benches);
