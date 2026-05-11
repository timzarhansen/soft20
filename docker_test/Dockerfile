FROM nvidia/cuda:13.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libfftw3-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

COPY docker_test/CMakeLists.txt /workspace/
COPY src /workspace/soft20/src
COPY include /workspace/soft20/include
COPY examples /workspace/soft20/examples
COPY examples1 /workspace/soft20/examples1

RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="80" \
    && make -j$(nproc) test_fftw test_cufft test_soft_fftw_orig \
    && strip /workspace/build/test_fftw

ENTRYPOINT ["/workspace/build/test_cufft"]

CMD []