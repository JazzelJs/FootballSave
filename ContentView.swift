import AVFoundation
import AVKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var video = VideoAnalysisModel()
    @State private var choosingVideo = false

    var body: some View {
        HSplitView {
            VStack(spacing: 12) {
                VideoPlayer(player: video.player)
                    .frame(minWidth: 420, minHeight: 320)
                HStack {
                    Button("Open video…") { choosingVideo = true }
                    Text(video.status)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .padding()
            Viewer()
                .frame(minWidth: 600, minHeight: 500)
        }
        .fileImporter(isPresented: $choosingVideo, allowedContentTypes: [.movie]) { result in
            guard case let .success(url) = result else { return }
            video.load(url: url)
        }
    }
}

@MainActor
final class VideoAnalysisModel: ObservableObject {
    @Published private(set) var player = AVPlayer()
    @Published private(set) var status = "Open an MP4 to run the Core ML detector."

    private var output: AVPlayerItemVideoOutput?
    private var timeObserver: Any?
    private let queue = DispatchQueue(label: "football3d.coreml")
    private var detector: FootballDetector?
    private var isAnalysing = false

    init() {
        do {
            detector = try FootballDetector()
            status = "Core ML detector ready. Open an MP4."
        } catch {
            status = "Could not load the Core ML detector: \(error.localizedDescription)"
        }
    }

    func load(url: URL) {
        if let timeObserver { player.removeTimeObserver(timeObserver) }
        let item = AVPlayerItem(url: url)
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        item.add(output)
        self.output = output
        player = AVPlayer(playerItem: item)
        status = "Playing \(url.lastPathComponent) — detecting twice per second."
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(seconds: 0.5, preferredTimescale: 600), queue: .main) { [weak self] time in
            Task { @MainActor in self?.analyse(time: time) }
        }
        player.play()
    }

    private func analyse(time: CMTime) {
        guard !isAnalysing, let output, let detector,
              output.hasNewPixelBuffer(forItemTime: time),
              let frame = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else { return }
        isAnalysing = true
        queue.async { [weak self] in
            let result = Result { try detector.detect(in: frame) }
            DispatchQueue.main.async {
                guard let self else { return }
                self.isAnalysing = false
                switch result {
                case let .success(detections):
                    let players = detections.filter { $0.label == "player" || $0.label == "goalkeeper" }.count
                    self.status = "Core ML: \(players) people, \(detections.count) total detections"
                case let .failure(error):
                    self.status = "Core ML error: \(error.localizedDescription)"
                }
            }
        }
    }
}
