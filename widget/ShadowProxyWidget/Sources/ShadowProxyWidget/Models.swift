import Foundation

struct StatsResponse: Codable {
    let totalRequests: UInt64
    let inputTokens: UInt64
    let outputTokens: UInt64
    let toolCalls: UInt64

    enum CodingKeys: String, CodingKey {
        case totalRequests = "total_requests"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
        case toolCalls = "tool_calls"
    }
}

enum ProxyMode: String, Codable, CaseIterable, Identifiable {
    case anthropicOnly = "anthropic-only"
    case shadowOnly = "shadow-only"
    case both = "both"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .anthropicOnly: return "Claude"
        case .shadowOnly: return "GLM"
        case .both: return "Shadow"
        }
    }
}

struct ModeResponse: Codable {
    let mode: ProxyMode
}

struct SetModeRequest: Codable {
    let mode: ProxyMode
}

struct TracingResponse: Codable {
    let enabled: Bool
}

struct SetTracingRequest: Codable {
    let enabled: Bool
}
