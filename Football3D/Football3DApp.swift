import SwiftUI

@main
struct Football3DApp: App {
    var body: some Scene {
        WindowGroup("Football3D") {
            ContentView()
                .frame(minWidth: 1_100, minHeight: 650)
        }
        .windowResizability(.contentSize)
    }
}
