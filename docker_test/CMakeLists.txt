cmake_minimum_required(VERSION 3.20)
project(soft20_standalone LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
find_package(CUDAToolkit 11.0 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/soft20/include/soft20)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/soft20/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/soft20/src/lib1)
include_directories(${FFTW3_INCLUDE_DIRS})

set(LIB1_C_SOURCES
    soft20/src/lib1/rotate_so3_fftw.c
    soft20/src/lib1/rotate_so3_utils.c
    soft20/src/lib1/s2_cospmls.c
    soft20/src/lib1/s2_legendreTransforms.c
    soft20/src/lib1/s2_semi_fly.c
    soft20/src/lib1/s2_semi_memo.c
    soft20/src/lib1/so3_correlate_fftw.c
    soft20/src/lib1/soft_fftw.c
    soft20/src/lib1/soft_fftw_nt.c
    soft20/src/lib1/soft_fftw_pc.c
    soft20/src/lib1/utils_vec_cx.c
    soft20/src/lib1/wignerTransforms_fftw.c
    soft20/src/lib1/wrap_s2_rotate_fftw.c
    soft20/src/lib1/wrap_soft_fftw.c
    soft20/src/lib1/wrap_soft_fftw_cor2.c
)

set(LIB1_CUDA_SOURCES
    soft20/src/lib1/s2_semi_memo_cufft.cu
    soft20/src/lib1/soft_cufft.cu
    soft20/src/lib1/wrap_soft_cufft.cu
)

set(COMMON_C_SOURCES
    soft20/src/common/csecond.c
    soft20/src/common/makeWigner.c
    soft20/src/common/makeweights.c
    soft20/src/common/s2_primitive.c
    soft20/src/common/utils_so3.c
    soft20/src/common/wignerTransforms.c
)

# FFTW CPU backend library (full library with all sources)
add_library(soft20_fftw STATIC
    ${LIB1_C_SOURCES}
    ${COMMON_C_SOURCES}
)

target_link_libraries(soft20_fftw PRIVATE ${FFTW3_LIBRARIES} m)

# cuFFT GPU backend library (CUDA sources + common C sources for shared utilities)
add_library(soft20_cufft STATIC
    ${LIB1_CUDA_SOURCES}
    ${COMMON_C_SOURCES}
)

target_link_libraries(soft20_cufft PRIVATE ${FFTW3_LIBRARIES} cuda cufft m)

# Test binary for FFTW backend
add_executable(test_fftw
    soft20/examples/test_cufft_compare.cu
)

target_link_libraries(test_fftw
    soft20_fftw
    ${FFTW3_LIBRARIES}
    m
)

set_target_properties(test_fftw PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Test binary for cuFFT backend
add_executable(test_cufft
    soft20/examples/test_cufft_compare.cu
)

target_link_libraries(test_cufft
    soft20_cufft
    cuda
    cufft
    ${FFTW3_LIBRARIES}
    m
)

set_target_properties(test_cufft PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
