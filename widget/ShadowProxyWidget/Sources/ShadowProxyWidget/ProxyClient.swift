import Foundation

@Observable
class ProxyClient {
    var stats = StatsResponse(totalRequests: 0, inputTokens: 0, outputTokens: 0, toolCalls: 0)
    var mode: ProxyMode = .both
    var tracingEnabled = false
    var isConnected = false

    private let baseURL: URL
    private let session: URLSession
    private var timer: Timer?

    init(baseURL: String = "http://localhost:3080") {
        self.baseURL = URL(string: baseURL)!
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 2
        self.session = URLSession(configuration: config)
    }

    func startPolling() {
        fetchAll()
        timer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            self?.fetchAll()
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    private func fetchAll() {
        Task { await fetchStats() }
        Task { await fetchMode() }
        Task { await fetchTracing() }
    }

    func fetchStats() async {
        let url = baseURL.appendingPathComponent("api/stats")
        do {
            let (data, _) = try await session.data(from: url)
            let decoded = try JSONDecoder().decode(StatsResponse.self, from: data)
            await MainActor.run {
                self.stats = decoded
                self.isConnected = true
            }
        } catch {
            await MainActor.run {
                self.isConnected = false
            }
        }
    }

    func fetchMode() async {
        let url = baseURL.appendingPathComponent("api/mode")
        do {
            let (data, _) = try await session.data(from: url)
            let decoded = try JSONDecoder().decode(ModeResponse.self, from: data)
            await MainActor.run {
                self.mode = decoded.mode
            }
        } catch {
            // mode fetch failure handled silently
        }
    }

    func fetchTracing() async {
        let url = baseURL.appendingPathComponent("api/tracing")
        do {
            let (data, _) = try await session.data(from: url)
            let decoded = try JSONDecoder().decode(TracingResponse.self, from: data)
            await MainActor.run {
                self.tracingEnabled = decoded.enabled
            }
        } catch {
            // tracing fetch failure handled silently
        }
    }

    func setTracing(_ enabled: Bool) async {
        let url = baseURL.appendingPathComponent("api/tracing")
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = SetTracingRequest(enabled: enabled)
        request.httpBody = try? JSONEncoder().encode(body)
        do {
            let (_, _) = try await session.data(for: request)
            await MainActor.run {
                self.tracingEnabled = enabled
            }
        } catch {
            // set tracing failure handled silently
        }
    }

    func setMode(_ newMode: ProxyMode) async {
        let url = baseURL.appendingPathComponent("api/mode")
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = SetModeRequest(mode: newMode)
        request.httpBody = try? JSONEncoder().encode(body)
        do {
            let (_, _) = try await session.data(for: request)
            await MainActor.run {
                self.mode = newMode
            }
        } catch {
            // set mode failure handled silently
        }
    }
}
