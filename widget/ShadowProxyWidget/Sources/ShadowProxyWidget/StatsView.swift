import SwiftUI

struct StatsView: View {
    let stats: StatsResponse

    var body: some View {
        VStack(spacing: 6) {
            StatRow(label: "Requests", value: formatNumber(stats.totalRequests))
            StatRow(
                label: "Tokens",
                value: "\(formatNumber(stats.inputTokens))in / \(formatNumber(stats.outputTokens))out"
            )
            StatRow(label: "Tool Calls", value: formatNumber(stats.toolCalls))
        }
    }

    private func formatNumber(_ n: UInt64) -> String {
        if n >= 1_000_000 {
            return String(format: "%.1fM", Double(n) / 1_000_000)
        } else if n >= 1_000 {
            return String(format: "%.1fK", Double(n) / 1_000)
        } else {
            return "\(n)"
        }
    }
}

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
                .monospacedDigit()
        }
    }
}
