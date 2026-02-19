// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "ShadowProxyWidget",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "ShadowProxyWidget",
            path: "Sources/ShadowProxyWidget"
        )
    ]
)
