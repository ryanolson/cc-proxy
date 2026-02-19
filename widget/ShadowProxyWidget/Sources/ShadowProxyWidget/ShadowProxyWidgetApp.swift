import SwiftUI

@main
struct ShadowProxyWidgetApp: App {
    @State private var client = ProxyClient()

    var body: some Scene {
        MenuBarExtra {
            VStack(spacing: 0) {
                // Header
                HStack(spacing: 6) {
                    Circle()
                        .fill(client.isConnected ? .green : .red)
                        .frame(width: 8, height: 8)
                    Text(client.isConnected ? "Connected" : "Disconnected")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("Dynamo Coding Widget")
                        .font(.caption)
                        .fontWeight(.medium)
                }
                .padding(.horizontal)
                .padding(.vertical, 8)

                Divider()

                // Stats
                StatsView(stats: client.stats)
                    .padding(.horizontal)
                    .padding(.vertical, 8)

                Divider()

                // Mode Picker
                ModePickerView(selection: $client.mode) { newMode in
                    Task { await client.setMode(newMode) }
                }
                .padding(.vertical, 8)

                // Trace logging toggle
                Toggle("Allow trace logging", isOn: Binding(
                    get: { client.tracingEnabled },
                    set: { newValue in
                        Task { await client.setTracing(newValue) }
                    }
                ))
                .toggleStyle(.switch)
                .padding(.horizontal)
                .padding(.vertical, 6)

                Divider()

                // Quit
                Button {
                    NSApplication.shared.terminate(nil)
                } label: {
                    Text("Quit")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .keyboardShortcut("q", modifiers: .command)
                .buttonStyle(.plain)
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            .frame(width: 260)
            .onAppear {
                client.startPolling()
            }
        } label: {
            Label("Dynamo Coding Widget", systemImage: "arrow.triangle.branch")
        }
        .menuBarExtraStyle(.window)
    }
}
