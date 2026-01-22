#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(pwd)
BUILD_DIR="$PROJECT_DIR/target/xcframework"
FRAMEWORK_NAME="MathLex"
SWIFT_TARGET_DIR="$PROJECT_DIR/Sources/MathLexRust"

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$SWIFT_TARGET_DIR"

# Build for all targets
# Note: build.rs will generate Swift bindings in ./generated/ during the first build
cargo build --release --features ffi --target aarch64-apple-ios
cargo build --release --features ffi --target aarch64-apple-ios-sim
cargo build --release --features ffi --target x86_64-apple-ios
cargo build --release --features ffi --target aarch64-apple-darwin
cargo build --release --features ffi --target x86_64-apple-darwin

# Copy generated Swift bindings to Swift package target
# swift-bridge generates:
#   - generated/mathlex/mathlex.swift (project-specific bindings)
#   - generated/SwiftBridgeCore.swift (core swift-bridge support)
echo "Copying generated Swift bindings to $SWIFT_TARGET_DIR"
if [ -f "generated/mathlex/mathlex.swift" ]; then
    cp "generated/mathlex/mathlex.swift" "$SWIFT_TARGET_DIR/"
    echo "✓ Copied mathlex.swift"
else
    echo "ERROR: generated/mathlex/mathlex.swift not found"
    exit 1
fi

if [ -f "generated/SwiftBridgeCore.swift" ]; then
    cp "generated/SwiftBridgeCore.swift" "$SWIFT_TARGET_DIR/"
    echo "✓ Copied SwiftBridgeCore.swift"
else
    echo "ERROR: generated/SwiftBridgeCore.swift not found"
    exit 1
fi

# Create fat library for iOS Simulator (arm64 + x86_64)
lipo -create \
    target/aarch64-apple-ios-sim/release/libmathlex.a \
    target/x86_64-apple-ios/release/libmathlex.a \
    -output "$BUILD_DIR/libmathlex-ios-sim.a"

# Create fat library for macOS (arm64 + x86_64)
lipo -create \
    target/aarch64-apple-darwin/release/libmathlex.a \
    target/x86_64-apple-darwin/release/libmathlex.a \
    -output "$BUILD_DIR/libmathlex-macos.a"

# Create XCFramework
xcodebuild -create-xcframework \
    -library target/aarch64-apple-ios/release/libmathlex.a \
    -headers generated/mathlex \
    -library "$BUILD_DIR/libmathlex-ios-sim.a" \
    -headers generated/mathlex \
    -library "$BUILD_DIR/libmathlex-macos.a" \
    -headers generated/mathlex \
    -output "$BUILD_DIR/$FRAMEWORK_NAME.xcframework"

echo "XCFramework created at $BUILD_DIR/$FRAMEWORK_NAME.xcframework"
