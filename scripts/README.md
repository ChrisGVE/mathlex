# Build Scripts

## XCFramework Build

The `build-xcframework.sh` script creates a universal XCFramework for iOS, iOS Simulator, and macOS distribution.

### Required Rust Targets

Before running the script, install all required Rust targets:

```bash
# iOS Device (ARM64)
rustup target add aarch64-apple-ios

# iOS Simulator (ARM64 - Apple Silicon Macs)
rustup target add aarch64-apple-ios-sim

# iOS Simulator (x86_64 - Intel Macs)
rustup target add x86_64-apple-ios

# macOS (ARM64 - Apple Silicon)
rustup target add aarch64-apple-darwin

# macOS (x86_64 - Intel)
rustup target add x86_64-apple-darwin
```

Or install all at once:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios aarch64-apple-darwin x86_64-apple-darwin
```

### Usage

From the project root:

```bash
./scripts/build-xcframework.sh
```

The XCFramework will be created at `target/xcframework/MathLex.xcframework`.

### What the Script Does

1. Cleans the build directory
2. Builds the Rust library for all required targets with the `ffi` feature
3. Creates fat libraries for:
   - iOS Simulator (combines arm64 and x86_64)
   - macOS (combines arm64 and x86_64)
4. Creates the XCFramework with:
   - iOS device library (arm64)
   - iOS simulator library (universal)
   - macOS library (universal)
   - C headers from `generated/mathlex/`

### Requirements

- Xcode Command Line Tools
- Rust with all targets installed (see above)
- `cargo` in PATH
- `swift-bridge-cli` installed (if regenerating bindings)

### Troubleshooting

If the build fails:
- Ensure all Rust targets are installed: `rustup target list --installed`
- Verify Xcode tools are installed: `xcode-select --install`
- Check that the `ffi` feature is properly configured in `Cargo.toml`
- Ensure C headers exist in `generated/mathlex/` (run `cargo build --features ffi` first)
