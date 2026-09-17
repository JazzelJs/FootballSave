import SwiftUI
import WebKit

struct Viewer: NSViewRepresentable {
    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.preferences.isElementFullscreenEnabled = true
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.setValue(false, forKey: "drawsBackground")

        guard let webRoot = Bundle.main.url(forResource: "Web", withExtension: nil),
              let index = URL(string: "index.html", relativeTo: webRoot) else {
            webView.loadHTMLString("<h1>Football3D resources are missing.</h1>", baseURL: nil)
            return webView
        }

        webView.loadFileURL(index, allowingReadAccessTo: webRoot)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {}
}
