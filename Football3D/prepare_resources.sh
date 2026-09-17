#!/bin/zsh
set -euo pipefail

# Xcode passes the built app's Resources directory as this variable.
resource_root="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}"
web_root="${resource_root}/Web"

rm -rf "${web_root}"
mkdir -p "${web_root}"
cp -R "${SRCROOT}/src/viewer/." "${web_root}/"

# Generated data is intentionally not committed; copy whatever exists locally.
if [[ -d "${SRCROOT}/data/tracks" || -d "${SRCROOT}/data/camera" || -f "${SRCROOT}/data/pose_1131.smpl" ]]; then
  mkdir -p "${web_root}/data/tracks"
  if [[ -d "${SRCROOT}/data/tracks" ]]; then
    cp -R "${SRCROOT}/data/tracks/." "${web_root}/data/tracks/"
  fi
  if [[ -f "${SRCROOT}/data/pose_1131.smpl" ]]; then
    cp "${SRCROOT}/data/pose_1131.smpl" "${web_root}/data/"
  fi
  if [[ -d "${SRCROOT}/data/camera" ]]; then
    mkdir -p "${web_root}/data/camera"
    cp -R "${SRCROOT}/data/camera/." "${web_root}/data/camera/"
  fi
fi

echo "Football3D resources staged in ${resource_root}"
