# MathLexRust - Swift Bridge Bindings

This directory contains the generated Swift bindings from the Rust mathlex library via swift-bridge.

## Build Process

The contents of this directory will be automatically generated during the XCFramework build process by the `build.sh` script. The script will:

1. Build the Rust library with the `ffi` feature enabled
2. Generate Swift bindings using swift-bridge
3. Create an XCFramework containing:
   - iOS arm64 (device)
   - iOS x86_64 and arm64 (simulator)
   - macOS arm64 and x86_64
4. Copy the generated Swift files into this directory
5. Update the Package.swift to link against the XCFramework

## Do Not Edit

Files in this directory should not be manually edited. They are automatically generated from the Rust source code and will be overwritten during each build.

## Swift Bridge

The bindings are generated using [swift-bridge](https://github.com/chinedufn/swift-bridge), which provides:

- Automatic Swift/Rust FFI code generation
- Type-safe cross-language communication
- Memory-safe resource management
- Idiomatic APIs for both languages

For more information about the FFI implementation, see `src/ffi.rs` in the Rust source code.
