# Football3D macOS app

Open `Football3D.xcodeproj` in Xcode and run the `Football3D` scheme.

The build phase copies the current viewer (including local Three.js files), track JSON, and
`data/pose_166.smpl` into the app bundle. Generated source data remains outside
the Xcode project and is picked up from the repository at build time.

Command-line build:

```sh
xcodebuild -project Football3D.xcodeproj -scheme Football3D -configuration Release build
```
