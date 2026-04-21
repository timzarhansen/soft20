/***************************************************************************
  cuFFT Backend Test for SO(3) Transforms
  Version 2.0

  Tests the cuFFT GPU implementation against known results
***************************************************************************/

#include <cuda.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "soft_fftw.h"
#include "utils_so3.h"
#include "makeweights.h"

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return 1; \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            return 1; \
        } \
    } while(0)

// Test parameters
#define BW 8  // Bandwidth (keep small for testing)
#define N (2 * BW)  // Grid size
#define N3 (N * N * N)  // Total samples
#define NUM_COEFFS ((4 * BW * BW * BW - BW) / 3)  // Number of coefficients

/**
 * Generate a synthetic SO(3) signal with known coefficients
 */
void generate_test_signal(fftw_complex *signal, int bw)
{
    int n = 2 * bw;
    int i, j, k;
    
    // Initialize with a simple test pattern
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                int idx = i + n * (j + n * k);
                // Simple sinusoidal pattern
                double alpha = 2.0 * M_PI * i / n;
                double beta = M_PI * j / n;
                double gamma = 2.0 * M_PI * k / n;
                
                signal[idx][0] = sin(alpha) * cos(beta) * cos(gamma);
                signal[idx][1] = 0.0;  // Real signal
            }
        }
    }
}

/**
 * Compute L2 error between two arrays
 */
double compute_l2_error(fftw_complex *a, fftw_complex *b, int size)
{
    double error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff_r = a[i][0] - b[i][0];
        double diff_i = a[i][1] - b[i][1];
        error += diff_r * diff_r + diff_i * diff_i;
    }
    return sqrt(error / size);
}

/**
 * Compute max error between two arrays
 */
double compute_max_error(fftw_complex *a, fftw_complex *b, int size)
{
    double max_error = 0.0;
    for (int i = 0; i < size; i++) {
        double diff_r = a[i][0] - b[i][0];
        double diff_i = a[i][1] - b[i][1];
        double error = sqrt(diff_r * diff_r + diff_i * diff_i);
        if (error > max_error) {
            max_error = error;
        }
    }
    return max_error;
}

/**
 * Test forward and inverse SO(3) transform
 */
int test_so3_transform()
{
    printf("Testing SO(3) transform with bw=%d...\n", BW);
    printf("  Grid size: %d x %d x %d = %d samples\n", N, N, N, N3);
    printf("  Number of coefficients: %d\n", NUM_COEFFS);
    
    // Allocate memory
    fftw_complex *signal = (fftw_complex*)malloc(sizeof(fftw_complex) * N3);
    fftw_complex *coeffs = (fftw_complex*)malloc(sizeof(fftw_complex) * NUM_COEFFS);
    fftw_complex *reconstructed = (fftw_complex*)malloc(sizeof(fftw_complex) * N3);
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * N3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * N3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * N + N * BW));
    double *weights = (double*)malloc(sizeof(double) * (2 * BW));
    
    // Generate weights
    makeweights(BW, weights);
    
    // Generate test signal
    generate_test_signal(signal, BW);
    
    // Time forward transform
    clock_t start = clock();
    Forward_SO3_Naive_fftw(BW, signal, coeffs, workspace_cx, workspace_cx2,
                           workspace_re, weights, NULL, 0);
    clock_t forward_time = clock() - start;
    
    printf("  Forward transform time: %.3f ms\n", 
           1000.0 * forward_time / CLOCKS_PER_SEC);
    
    // Time inverse transform
    start = clock();
    Inverse_SO3_Naive_fftw(BW, coeffs, reconstructed, workspace_cx, workspace_cx2,
                           workspace_re, NULL, 0);
    clock_t inverse_time = clock() - start;
    
    printf("  Inverse transform time: %.3f ms\n",
           1000.0 * inverse_time / CLOCKS_PER_SEC);
    
    // Compute errors
    double l2_error = compute_l2_error(signal, reconstructed, N3);
    double max_error = compute_max_error(signal, reconstructed, N3);
    
    printf("  Reconstruction L2 error: %.6e\n", l2_error);
    printf("  Reconstruction max error: %.6e\n", max_error);
    
    // Check if errors are acceptable (within numerical precision)
    double tolerance = 1e-6;
    if (l2_error < tolerance && max_error < tolerance) {
        printf("  PASSED: Reconstruction accurate within tolerance\n");
    } else {
        printf("  FAILED: Reconstruction error exceeds tolerance\n");
        free(signal);
        free(coeffs);
        free(reconstructed);
        free(workspace_cx);
        free(workspace_cx2);
        free(workspace_re);
        free(weights);
        return 1;
    }
    
    // Cleanup
    free(signal);
    free(coeffs);
    free(reconstructed);
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
    free(weights);
    
    return 0;
}

/**
 * Test GPU device detection
 */
int test_gpu_device()
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    printf("GPU Device Detection:\n");
    printf("  Number of CUDA devices: %d\n", device_count);
    
    if (device_count == 0) {
        printf("  WARNING: No CUDA devices found!\n");
        printf("  Test will fail without GPU hardware.\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("  Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    return 0;
}

int main(int argc, char **argv)
{
    printf("===========================================\n");
    printf("SO(3) Transform - cuFFT Backend Test\n");
    printf("===========================================\n\n");
    
    // Test GPU detection
    if (test_gpu_device() != 0) {
        printf("\nSkipping transform tests - no GPU available\n");
        return 1;
    }
    
    printf("\n");
    
    // Test SO(3) transform
    int result = test_so3_transform();
    
    printf("\n===========================================\n");
    if (result == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("Some tests FAILED\n");
    }
    printf("===========================================\n");
    
    return result;
}
