import CoreImage
import CoreML
import CoreVideo
import Foundation

struct FootballDetection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let rect: CGRect // Normalized x, y, width, height in the 1280 × 1280 model input.
}

final class FootballDetector {
    private static let labels = ["ball", "goalkeeper", "player", "referee"]
    private static let inputSize = 1280
    private let model: MLModel
    private let context = CIContext()

    init() throws {
        guard let url = Bundle.main.url(forResource: "football-player-detection-v9", withExtension: "mlmodelc") else {
            throw CocoaError(.fileNoSuchFile)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    func detect(in frame: CVPixelBuffer, confidenceThreshold: Float = 0.25) throws -> [FootballDetection] {
        let input = try resized(frame)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": input,
            "iouThreshold": 0.7,
            "confidenceThreshold": Double(confidenceThreshold),
        ])
        let output = try model.prediction(from: provider)
        guard let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue,
              let confidence = output.featureValue(for: "confidence")?.multiArrayValue else {
            throw CocoaError(.coderReadCorrupt)
        }
        return decode(coordinates: coordinates, confidence: confidence, threshold: confidenceThreshold)
    }

    private func resized(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        var resized: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, Self.inputSize, Self.inputSize,
                                  kCVPixelFormatType_32BGRA, attributes, &resized) == kCVReturnSuccess,
              let resized else {
            throw CocoaError(.fileWriteOutOfSpace)
        }
        let scaleX = CGFloat(Self.inputSize) / CGFloat(CVPixelBufferGetWidth(frame))
        let scaleY = CGFloat(Self.inputSize) / CGFloat(CVPixelBufferGetHeight(frame))
        let image = CIImage(cvPixelBuffer: frame).transformed(by: .init(scaleX: scaleX, y: scaleY))
        context.render(image, to: resized,
                       bounds: CGRect(x: 0, y: 0, width: Self.inputSize, height: Self.inputSize),
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        return resized
    }

    private func decode(coordinates: MLMultiArray, confidence: MLMultiArray, threshold: Float) -> [FootballDetection] {
        let boxCount = coordinates.shape[0].intValue
        let coordinateStride = coordinates.strides[0].intValue
        let confidenceStride = confidence.strides[0].intValue
        let classCount = confidence.shape[1].intValue
        let coordinateValues = coordinates.dataPointer.bindMemory(to: Float32.self, capacity: coordinates.count)
        let confidenceValues = confidence.dataPointer.bindMemory(to: Float32.self, capacity: confidence.count)

        return (0 ..< boxCount).compactMap { index in
            let scores = (0 ..< min(classCount, Self.labels.count)).map {
                confidenceValues[index * confidenceStride + $0]
            }
            guard let best = scores.enumerated().max(by: { $0.element < $1.element }), best.element >= threshold else {
                return nil
            }
            let start = index * coordinateStride
            let x = CGFloat(coordinateValues[start])
            let y = CGFloat(coordinateValues[start + 1])
            let width = CGFloat(coordinateValues[start + 2])
            let height = CGFloat(coordinateValues[start + 3])
            return FootballDetection(label: Self.labels[best.offset], confidence: best.element,
                                     rect: CGRect(x: x - width / 2, y: y - height / 2,
                                                  width: width, height: height))
        }
    }
}
