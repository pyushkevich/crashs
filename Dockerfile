# Start from the official Debian image
FROM debian:bullseye

# Install necessary tools and dependencies. Note: no JDK is installed here - the
# GraalVM JDK used to compile the native cbstools bindings is downloaded automatically
# by native/scripts/build_native.sh (it needs one only at build time, not at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ================================================
# CRASHS install
# ================================================

# Install the bigger dependencies for faster builds
RUN python3 -m pip install numpy torch pykeops monai nnunetv2

# Copy the contents (including the native/cbstools-public git submodule - make sure
# it's checked out locally with `git submodule update --init` before building this image)
COPY . /tk/crashs
WORKDIR /tk/crashs

# Build the native cbstools library (downloads a GraalVM JDK itself for this platform)
RUN bash native/scripts/build_native.sh ubuntu-latest

RUN python3 -m pip install .
