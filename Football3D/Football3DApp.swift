import SwiftUI

@main
struct Football3DApp: App {
    var body: some Scene {
        WindowGroup("Football3D") {
            Viewer()
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowResizability(.contentSize)
    }
}
