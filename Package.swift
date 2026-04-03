// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "MathLex",
  platforms: [
    .iOS(.v15),
    .macOS(.v12),
  ],
  products: [
    .library(
      name: "MathLex",
      targets: ["MathLex"]
    )
  ],
  targets: [
    // C headers for the swift-bridge FFI symbols
    .target(
      name: "MathLexBridge",
      dependencies: [],
      path: "Sources/MathLexBridge",
      publicHeadersPath: "include"
    ),

    // Generated swift-bridge Swift bindings
    .target(
      name: "MathLexRust",
      dependencies: ["MathLexBridge"],
      path: "Sources/MathLexRust",
      exclude: ["README.md"],
      linkerSettings: [
        .unsafeFlags([
          "-L", "target/release",
          "-lmathlex",
        ])
      ]
    ),

    // Swift wrapper providing idiomatic Swift API
    .target(
      name: "MathLex",
      dependencies: ["MathLexRust"],
      path: "Sources/MathLex",
      exclude: ["MathLex.docc"]
    ),

    // Tests
    .testTarget(
      name: "MathLexTests",
      dependencies: ["MathLex"],
      path: "swift/Tests/MathLexTests"
    ),
  ]
)
