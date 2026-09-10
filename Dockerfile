# Start from the official Debian image
FROM debian:bullseye

# Install necessary tools and dependencies. No Java/JDK needed: the cbstools binding
# crashs depends on (crashs-cbstools-bindings) ships as a prebuilt binary wheel, so
# there's no compiled-from-source step here at all.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ================================================
# CRASHS install
# ================================================

# Install the bigger dependencies for faster builds
RUN python3 -m pip install numpy torch pykeops monai nnunetv2

# Copy the contents
COPY . /tk/crashs
WORKDIR /tk/crashs
RUN python3 -m pip install .
