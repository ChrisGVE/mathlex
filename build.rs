fn main() {
    #[cfg(feature = "ffi")]
    {
        swift_bridge_build::parse_bridges(vec!["src/ffi.rs"])
            .write_all_concatenated("./generated", env!("CARGO_PKG_NAME"));
    }
}
