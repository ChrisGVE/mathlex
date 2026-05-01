use mathlex::Expression;
use std::fs;
use std::path::Path;

pub fn assert_golden(expr: &Expression, fixture_name: &str) {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/serde")
        .join(format!("{}.json", fixture_name));
    let actual = serde_json::to_string_pretty(expr).unwrap();

    if std::env::var("MATHLEX_UPDATE_GOLDEN").is_ok() {
        fs::create_dir_all(fixture_path.parent().unwrap()).unwrap();
        fs::write(&fixture_path, format!("{}\n", actual)).unwrap();
        return;
    }

    let expected = fs::read_to_string(&fixture_path).unwrap_or_else(|_| {
        panic!(
            "Missing fixture: {}\nRun with MATHLEX_UPDATE_GOLDEN=1 to generate",
            fixture_path.display()
        )
    });

    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Golden file mismatch for {}\nRun with MATHLEX_UPDATE_GOLDEN=1 to update",
        fixture_name
    );
}
