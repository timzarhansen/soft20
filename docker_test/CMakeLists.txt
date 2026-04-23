cmake_minimum_required(VERSION 3.20)
project(soft20_standalone LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "70 80 86")

find_package(FFTW3 REQUIRED)
find_package(CUDAToolkit 11.0 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/soft20)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/lib1)

set(LIB1_C_SOURCES
    ../src/lib1/rotate_so3_fftw.c
    ../src/lib1/rotate_so3_utils.c
    ../src/lib1/s2_cospmls.c
    ../src/lib1/s2_legendreTransforms.c
    ../src/lib1/s2_semi_fly.c
    ../src/lib1/s2_semi_memo.c
    ../src/lib1/so3_correlate_fftw.c
    ../src/lib1/soft_fftw.c
    ../src/lib1/soft_fftw_nt.c
    ../src/lib1/soft_fftw_pc.c
    ../src/lib1/utils_vec_cx.c
    ../src/lib1/wignerTransforms_fftw.c
    ../src/lib1/wrap_s2_rotate_fftw.c
    ../src/lib1/wrap_soft_fftw.c
    ../src/lib1/wrap_soft_fftw_cor2.c
)

set(LIB1_CUDA_SOURCES
    ../src/lib1/s2_semi_memo_cufft.cu
    ../src/lib1/soft_cufft.cu
    ../src/lib1/wrap_soft_cufft.cu
)

set(COMMON_C_SOURCES
    ../src/common/csecond.c
    ../src/common/makeWigner.c
    ../src/common/makeweights.c
    ../src/common/s2_primitive.c
    ../src/common/utils_so3.c
    ../src/common/wignerTransforms.c
)

add_library(soft20_fftw STATIC
    ${LIB1_C_SOURCES}
    ${COMMON_C_SOURCES}
)

target_link_libraries(soft20_fftw PRIVATE fftw3 m)

add_library(soft20_cufft STATIC
    ${LIB1_CUDA_SOURCES}
)

target_link_libraries(soft20_cufft PRIVATE cuda cufft)

add_executable(test_cufft_compare
    ../examples/test_cufft_compare.cu
)

target_link_libraries(test_cufft_compare
    soft20_fftw
    soft20_cufft
    cuda
    cufft
    fftw3
    m
)
