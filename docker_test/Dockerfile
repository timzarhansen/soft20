# SO(3) cuFFT Backend Test Container
# Uses NVIDIA CUDA 13.1 base image for compatibility with borrowed GPU hardware

FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libfftw3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Copy standalone CMakeLists.txt
COPY docker_test/CMakeLists.txt /workspace/

# Copy soft20 source and include directories
COPY src /workspace/soft20/src
COPY include /workspace/soft20/include

# Build with standalone CMakeLists.txt (builds both FFTW and cuFFT backends)
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70 80 86" \
    && make -j$(nproc) test_cufft_compare \
    && strip build/test_cufft_compare

# Set entrypoint to run the comparison test
ENTRYPOINT ["/workspace/build/test_cufft_compare"]

# Default command (can be overridden)
CMD []
