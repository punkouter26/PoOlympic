#!/usr/bin/env bash
# Build libmujoco.so (arm64-v8a) for the Android player, from the SAME MuJoCo release as the Unity plugin
# (Packages/org.mujoco = 3.11.0; its C# bindings mirror that release's struct layouts, so the library version must match).
#
# Why not the prebuilt joanllobera/mujoco-bin (AGENTS.md)? That package ships MuJoCo 3.5.0 for the 3.5.0 plugin; mixing it
# with the 3.11 plugin corrupts memory. We follow its approach instead: the upstream sources + the Android NDK (the one
# bundled with Unity), no source patches.
#   - API level 28: bionic declares aligned_alloc from API 28 (player min SDK = 28, AndroidBuild.cs)
#   - -D_POSIX_C_SOURCE: MuJoCo only picks localtime_r when it is defined (bionic has it)
# Output: Assets/Plugins/MuJoCo/Android/arm64-v8a/libmujoco.so (stripped, ~5 MB; needs only libc/libm/libdl)
# Usage (Git Bash on Windows): tools/build_mujoco_android.sh [work dir, default ../mujoco-android]
set -euo pipefail
VERSION=3.11.0
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${1:-$REPO/../mujoco-android}"
UNITY="C:/Program Files/Unity/Hub/Editor/6000.6.0f1/Editor/Data/PlaybackEngines/AndroidPlayer"
NDK="$UNITY/NDK"
BIN="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin"
mkdir -p "$WORK" && cd "$WORK"
[ -d mujoco ] || git clone --depth 1 --branch "$VERSION" https://github.com/google-deepmind/mujoco.git mujoco
cmake -S mujoco -B build-arm64 -G Ninja -DCMAKE_MAKE_PROGRAM="$(cygpath -m "$(which ninja)")" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "-DCMAKE_C_FLAGS=-D_POSIX_C_SOURCE=200809L" \
  -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_TEST_PYTHON_UTIL=OFF \
  -DMUJOCO_ENABLE_AVX=OFF -DMUJOCO_ENABLE_AVX_INTRINSICS=OFF -DMUJOCO_ENABLE_LTO=OFF
cmake --build build-arm64 --target mujoco -j "$(nproc)"
OUT="$REPO/Assets/Plugins/MuJoCo/Android/arm64-v8a"
mkdir -p "$OUT"
"$BIN/llvm-strip.exe" --strip-unneeded -o "$OUT/libmujoco.so" build-arm64/lib/libmujoco.so
"$BIN/llvm-readelf.exe" -h "$OUT/libmujoco.so" | grep -E "Class|Machine"
echo "installed $OUT/libmujoco.so ($(stat -c %s "$OUT/libmujoco.so") bytes), MuJoCo $VERSION"
