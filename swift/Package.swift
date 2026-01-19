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
        ),
    ],
    targets: [
        // MathLexRust contains the generated swift-bridge bindings
        // This will be populated by the build script from the XCFramework
        .target(
            name: "MathLexRust",
            dependencies: [],
            path: "Sources/MathLexRust"
        ),

        // MathLex is the Swift wrapper providing idiomatic Swift API
        .target(
            name: "MathLex",
            dependencies: ["MathLexRust"],
            path: "Sources/MathLex"
        ),

        // Tests
        .testTarget(
            name: "MathLexTests",
            dependencies: ["MathLex"]
        ),
    ]
)
