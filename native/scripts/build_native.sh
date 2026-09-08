#!/usr/bin/env bash
# Builds libcbstools_native.{so,dylib,dll} from the cbstools-public Java sources plus the
# thin de.mpg.cbs.crashs wrapper layer, using GraalVM Native Image (no JVM needed at runtime).
#
# Usage: build_native.sh <os-tag>
#   <os-tag> is one of the cibuildwheel matrix values (ubuntu-latest, macos-14, macos-14-large,
#   windows-2022) and only affects which GraalVM/OS-specific branch runs; the actual javac +
#   native-image invocation is identical across platforms.
#
# Requires GRAALVM_HOME (or JAVA_HOME pointing at a GraalVM JDK with native-image) to be set
# before this script runs. On CI this is arranged by graalvm/setup-graalvm@v1 (macOS/Windows,
# host-level install) or by this script itself downloading a GraalVM tarball (Linux, where
# cibuildwheel runs inside a manylinux container that a host-level action can't reach).

set -euo pipefail
OS_TAG="${1:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NATIVE_DIR="$ROOT/native"
CBS="$NATIVE_DIR/cbstools-public"
BUILD="$NATIVE_DIR/build"
GRAALVM_VERSION="21"

mkdir -p "$BUILD/classes" "$BUILD/out"

# --- 1. Ensure a GraalVM JDK with native-image is on PATH ------------------------------------

# On Windows, GraalVM's native-image is shipped as "native-image.cmd". Git-bash's PATH lookup
# (unlike cmd.exe/PowerShell, which consult %PATHEXT%) only resolves bare/`.exe` names, so
# `command -v native-image` reports "not found" even when the GraalVM bin dir (containing
# native-image.cmd) is correctly on PATH. Resolve the actual command name up front and use it
# everywhere below instead of assuming the extension-less "native-image".
find_native_image() {
    for cand in native-image native-image.cmd native-image.exe; do
        if command -v "$cand" >/dev/null 2>&1; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

NATIVE_IMAGE="$(find_native_image || true)"

if [ -z "$NATIVE_IMAGE" ]; then
    case "$OS_TAG" in
        ubuntu-*)
            # Running inside a manylinux container during cibuildwheel's Linux leg -
            # graalvm/setup-graalvm@v1 (host-level) is not visible here, so fetch our own.
            GRAAL_TARBALL="graalvm-jdk-${GRAALVM_VERSION}_linux-x64_bin.tar.gz"
            curl -fsSL "https://download.oracle.com/graalvm/${GRAALVM_VERSION}/latest/${GRAAL_TARBALL}" -o /tmp/graalvm.tar.gz
            mkdir -p /tmp/graalvm && tar xzf /tmp/graalvm.tar.gz -C /tmp/graalvm --strip-components=1
            export GRAALVM_HOME=/tmp/graalvm
            export PATH="$GRAALVM_HOME/bin:$PATH"
            NATIVE_IMAGE="$(find_native_image || true)"
            ;;
        macos-*)
            # macOS cibuildwheel runs natively (not containerized); expect the workflow's
            # graalvm/setup-graalvm@v1 step to have already put native-image on PATH.
            echo "native-image not found on PATH; ensure graalvm/setup-graalvm ran first" >&2
            exit 1
            ;;
        windows-*)
            echo "native-image not found on PATH; ensure graalvm/setup-graalvm ran first" >&2
            exit 1
            ;;
        *)
            echo "Unrecognized OS tag '$OS_TAG' and native-image not on PATH" >&2
            exit 1
            ;;
    esac
fi

if [ -z "$NATIVE_IMAGE" ]; then
    echo "native-image could not be located even after setup" >&2
    exit 1
fi

echo "Using: $("$NATIVE_IMAGE" --version | head -1)"

# --- 2. Compile: the 5 needed cbstools-public core classes (javac -sourcepath pulls in only ---
# --- their actual transitive dependencies - NOT a directory-wide compile, which would also ---
# --- drag in the MIPAV-dependent NiftiInterface.java dead code sitting in utilities/) --------

CLASSPATH="$NATIVE_DIR/java/lib/commons-math3-3.5.jar:$NATIVE_DIR/java/lib/Jama-mipav.jar"

javac -d "$BUILD/classes" -cp "$CLASSPATH" -sourcepath "$CBS:$NATIVE_DIR/java/src" \
  "$CBS/de/mpg/cbs/core/shape/ShapeTopologyCorrection2.java" \
  "$CBS/de/mpg/cbs/core/cortex/CortexOptimCRUISE.java" \
  "$CBS/de/mpg/cbs/core/surface/SurfaceLevelsetToMesh.java" \
  "$CBS/de/mpg/cbs/core/surface/SurfaceInflation.java" \
  "$CBS/de/mpg/cbs/core/laminar/LaminarVolumetricLayering.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/NativeArrayUtil.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/TopologyCorrectionEntry.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/CruiseCortexEntry.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/LevelsetToMeshEntry.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/SurfaceInflationEntry.java" \
  "$NATIVE_DIR/java/src/de/mpg/cbs/crashs/VolumetricLayeringEntry.java"

# --- 3. native-image --shared: AOT-compile to a shared library, no reflection/JNI config -----
# --- needed (all 5 classes and their call chains are confirmed reflection/JNI/thread-free) ---

# On macOS, native-image's final link step (via the system `cc`) embeds the *build host's*
# OS version as the dylib's minimum target (e.g. minos 14.0 on a macos-14 runner) - setting
# the MACOSX_DEPLOYMENT_TARGET env var does NOT get forwarded by native-image, so it has no
# effect. delocate-wheel then refuses to repair the wheel because the bundled dylib's minos
# doesn't match the wheel's declared minimum - which differs by architecture: cibuildwheel's
# floor is macosx_11_0 for arm64 (Apple Silicon never shipped below macOS 11) but macosx_10_9
# for x86_64. Fix: pass -mmacosx-version-min directly to the linker, matching the actual
# build host's architecture (macos-14 = arm64, macos-14-large = x86_64).
EXTRA_NATIVE_IMAGE_ARGS=()
case "$OS_TAG" in
    macos-*)
        if [ "$(uname -m)" = "arm64" ]; then
            MACOS_MIN=11.0
        else
            MACOS_MIN=10.9
        fi
        EXTRA_NATIVE_IMAGE_ARGS+=("-H:NativeLinkerOption=-mmacosx-version-min=$MACOS_MIN")
        ;;
esac

# ${EXTRA_NATIVE_IMAGE_ARGS[@]+"${EXTRA_NATIVE_IMAGE_ARGS[@]}"} (not the plain
# "${EXTRA_NATIVE_IMAGE_ARGS[@]}"): older bash (e.g. manylinux's bash 4.2) treats expanding
# an empty array as an unbound variable under `set -u`, even though it was declared above.
"$NATIVE_IMAGE" --shared \
  -H:Name=libcbstools_native \
  -H:Path="$BUILD/out" \
  -cp "$BUILD/classes:$CLASSPATH" \
  --gc=serial \
  --no-fallback \
  ${EXTRA_NATIVE_IMAGE_ARGS[@]+"${EXTRA_NATIVE_IMAGE_ARGS[@]}"}

echo "Native library built at: $BUILD/out/"
ls -la "$BUILD/out/"
