#!/bin/bash
set -e

PROJECT_DIR=$(pwd)
BUILD_DIR="$PROJECT_DIR/target/xcframework"
FRAMEWORK_NAME="MathLex"

# Clean
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Build for all targets
cargo build --release --features ffi --target aarch64-apple-ios
cargo build --release --features ffi --target aarch64-apple-ios-sim
cargo build --release --features ffi --target x86_64-apple-ios
cargo build --release --features ffi --target aarch64-apple-darwin
cargo build --release --features ffi --target x86_64-apple-darwin

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
