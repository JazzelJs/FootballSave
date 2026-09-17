# Football3D macOS app

Open `Football3D.xcodeproj` in Xcode and run the `Football3D` scheme.

The build phase copies the current viewer (including local Three.js files), track JSON, and
the corrected `data/pose_1131.smpl` mesh into the app bundle. Generated source data remains outside
the Xcode project and is picked up from the repository at build time.

The target also compiles `data/models/football-player-detection-v9.mlpackage`.
After launching the app, click **Open video…**, choose an MP4 such as
`data/clips/clip04.mp4`, and the left pane reports Core ML detections twice per
second while the 3D viewer remains on the right.

Command-line build:

```sh
xcodebuild -project Football3D.xcodeproj -scheme Football3D -configuration Release build
```
