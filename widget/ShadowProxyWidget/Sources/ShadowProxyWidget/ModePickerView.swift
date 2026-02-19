import SwiftUI

struct ModePickerView: View {
    @Binding var selection: ProxyMode
    var onChange: (ProxyMode) -> Void

    var body: some View {
        Picker("Mode", selection: $selection) {
            ForEach(ProxyMode.allCases) { mode in
                Text(mode.displayName).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: selection) { _, newValue in
            onChange(newValue)
        }
        .padding(.horizontal)
    }
}
