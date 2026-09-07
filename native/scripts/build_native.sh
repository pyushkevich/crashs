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

if ! command -v native-image >/dev/null 2>&1; then
    case "$OS_TAG" in
        ubuntu-*)
            # Running inside a manylinux container during cibuildwheel's Linux leg -
            # graalvm/setup-graalvm@v1 (host-level) is not visible here, so fetch our own.
            GRAAL_TARBALL="graalvm-jdk-${GRAALVM_VERSION}_linux-x64_bin.tar.gz"
            curl -fsSL "https://download.oracle.com/graalvm/${GRAALVM_VERSION}/latest/${GRAAL_TARBALL}" -o /tmp/graalvm.tar.gz
            mkdir -p /tmp/graalvm && tar xzf /tmp/graalvm.tar.gz -C /tmp/graalvm --strip-components=1
            export GRAALVM_HOME=/tmp/graalvm
            export PATH="$GRAALVM_HOME/bin:$PATH"
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

echo "Using: $(native-image --version | head -1)"

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

native-image --shared \
  -H:Name=libcbstools_native \
  -H:Path="$BUILD/out" \
  -cp "$BUILD/classes:$CLASSPATH" \
  --gc=serial \
  --no-fallback

echo "Native library built at: $BUILD/out/"
ls -la "$BUILD/out/"
