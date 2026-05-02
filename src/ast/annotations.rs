use std::collections::BTreeMap;

/// Annotation set carried by each Expression node.
///
/// v0.4.0 ships the substrate only — no semantic interpretation.
/// thales v0.9.0 passes this through unchanged; v0.10.0 consumes it.
///
/// When empty, the field is omitted from JSON serialization.
/// Deserializing a missing field yields the empty set (backward compat).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnnotationSet {
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "BTreeMap::is_empty")
    )]
    pub entries: BTreeMap<String, String>,
}

impl AnnotationSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.entries.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }
}
