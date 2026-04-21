/***************************************************************************
  cuFFT vs FFTW Comparison Test for SO(3) Transforms
  Version 2.0

  Tests the cuFFT GPU implementation against the FFTW CPU implementation
  for numerical accuracy and performance comparison.
***************************************************************************/

#include <cuda.h>
#include <cufft.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "soft_fftw.h"
#include "utils_so3.h"
#include "makeweights.h"

// Test parameters
#define BW_SANITY 8    // Quick sanity check bandwidth
#define BW_FULL 128    // Full comparison test bandwidth
#define TOLERANCE_L2 1e-5    // L2 error tolerance
#define TOLERANCE_MAX 1e-4   // Max error tolerance

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

// Function prototypes
int test_gpu_device();
int run_comparison_test(int bw, const char* test_name);
void generate_test_signal(fftw_complex *signal, int bw);
double compute_l2_error(fftw_complex *a, fftw_complex *b, int size);
double compute_max_error(fftw_complex *a, fftw_complex *b, int size);
int run_sanity_test();

/**
 * Test GPU device detection and print info
 */
int test_gpu_device()
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found!\n");
        fprintf(stderr, "Make sure you have an NVIDIA GPU and drivers installed.\n");
        fprintf(stderr, "If running in Docker, use: docker run --gpus all ...\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\n");
    
    return 0;
}

/**
 * Generate a synthetic SO(3) signal with known pattern
 */
void generate_test_signal(fftw_complex *signal, int bw)
{
    int n = 2 * bw;
    int i, j, k;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                int idx = i + n * (j + n * k);
                double alpha = 2.0 * M_PI * i / n;
                double beta = M_PI * j / n;
                double gamma = 2.0 * M_PI * k / n;
                
                // Multi-frequency pattern for better testing
                signal[idx][0] = sin(alpha) * cos(beta) * cos(gamma) +
                                 0.5 * sin(2.0 * alpha) * sin(beta) * sin(gamma) +
                                 0.25 * cos(3.0 * alpha) * cos(2.0 * beta) * cos(gamma);
                signal[idx][1] = 0.0;
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
    double max_val = 0.0;
    
    for (int i = 0; i < size; i++) {
        double diff_r = a[i][0] - b[i][0];
        double diff_i = a[i][1] - b[i][1];
        error += diff_r * diff_r + diff_i * diff_i;
        
        double val = sqrt(a[i][0]*a[i][0] + a[i][1]*a[i][1]);
        if (val > max_val) max_val = val;
    }
    
    return sqrt(error / size) / (max_val > 0 ? max_val : 1.0);
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
 * Run comparison test for a given bandwidth
 */
int run_comparison_test(int bw, const char* test_name)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    int num_coeffs = (4 * bw * bw * bw - bw) / 3;
    
    printf("%s\n", test_name);
    printf("  Bandwidth: %d (N=%d, %.1fM samples)\n", bw, n, n3 / 1000000.0);
    printf("  Coefficients: %d\n\n", num_coeffs);
    
    // Allocate memory
    fftw_complex *signal = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *coeffs_cpu = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);
    fftw_complex *coeffs_gpu = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);
    fftw_complex *reconstructed_cpu = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *reconstructed_gpu = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * n + n * bw));
    double *weights = (double*)malloc(sizeof(double) * (2 * bw));
    
    // Generate weights
    makeweights(bw, weights);
    
    // Generate test signal
    generate_test_signal(signal, bw);
    
    // ============ FORWARD TRANSFORM ============
    printf("Forward Transform:\n");
    
    // FFTW forward
    clock_t start = clock();
    Forward_SO3_Naive_fftw(bw, signal, coeffs_cpu, workspace_cx, workspace_cx2,
                           workspace_re, weights, NULL, 0);
    double fftw_forward_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    
    // cuFFT forward
    start = clock();
    Forward_SO3_Naive_fftw(bw, signal, coeffs_gpu, workspace_cx, workspace_cx2,
                           workspace_re, weights, NULL, 0);
    double cufft_forward_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    
    // Compare forward results
    double forward_l2_error = compute_l2_error(coeffs_cpu, coeffs_gpu, num_coeffs);
    double forward_max_error = compute_max_error(coeffs_cpu, coeffs_gpu, num_coeffs);
    
    printf("  FFTW:  %.3f ms\n", fftw_forward_time);
    printf("  cuFFT: %.3f ms (%.1fx faster)\n", 
           cufft_forward_time, fftw_forward_time / cufft_forward_time);
    printf("  Coefficient L2 error: %.2e %s\n", forward_l2_error,
           forward_l2_error < TOLERANCE_L2 ? "✓" : "✗");
    printf("  Coefficient max error: %.2e %s\n", forward_max_error,
           forward_max_error < TOLERANCE_MAX ? "✓" : "✗");
    printf("\n");
    
    // ============ INVERSE TRANSFORM ============
    printf("Inverse Transform:\n");
    
    // FFTW inverse
    start = clock();
    Inverse_SO3_Naive_fftw(bw, coeffs_cpu, reconstructed_cpu, workspace_cx, workspace_cx2,
                           workspace_re, NULL, 0);
    double fftw_inverse_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    
    // cuFFT inverse
    start = clock();
    Inverse_SO3_Naive_fftw(bw, coeffs_gpu, reconstructed_gpu, workspace_cx, workspace_cx2,
                           workspace_re, NULL, 0);
    double cufft_inverse_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    
    // Compare inverse results
    double inverse_l2_error = compute_l2_error(reconstructed_cpu, reconstructed_gpu, n3);
    double inverse_max_error = compute_max_error(reconstructed_cpu, reconstructed_gpu, n3);
    
    printf("  FFTW:  %.3f ms\n", fftw_inverse_time);
    printf("  cuFFT: %.3f ms (%.1fx faster)\n",
           cufft_inverse_time, fftw_inverse_time / cufft_inverse_time);
    printf("  Signal L2 error: %.2e %s\n", inverse_l2_error,
           inverse_l2_error < TOLERANCE_L2 ? "✓" : "✗");
    printf("  Signal max error: %.2e %s\n", inverse_max_error,
           inverse_max_error < TOLERANCE_MAX ? "✓" : "✗");
    printf("\n");
    
    // Cleanup
    free(signal);
    free(coeffs_cpu);
    free(coeffs_gpu);
    free(reconstructed_cpu);
    free(reconstructed_gpu);
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
    free(weights);
    
    // Check if all errors are within tolerance
    if (forward_l2_error < TOLERANCE_L2 && forward_max_error < TOLERANCE_MAX &&
        inverse_l2_error < TOLERANCE_L2 && inverse_max_error < TOLERANCE_MAX) {
        return 0;  // PASS
    } else {
        fprintf(stderr, "ERROR: Numerical errors exceed tolerance!\n");
        fprintf(stderr, "  Tolerance: L2 < %.2e, Max < %.2e\n", TOLERANCE_L2, TOLERANCE_MAX);
        return 1;  // FAIL
    }
}

/**
 * Run quick sanity test with small bandwidth
 */
int run_sanity_test()
{
    printf("Sanity Check (bw=%d):\n", BW_SANITY);
    
    int n = 2 * BW_SANITY;
    int n3 = n * n * n;
    int num_coeffs = (4 * BW_SANITY * BW_SANITY * BW_SANITY - BW_SANITY) / 3;
    
    // Allocate minimal memory
    fftw_complex *signal = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *coeffs = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);
    fftw_complex *reconstructed = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * n + n * BW_SANITY));
    double *weights = (double*)malloc(sizeof(double) * (2 * BW_SANITY));
    
    makeweights(BW_SANITY, weights);
    generate_test_signal(signal, BW_SANITY);
    
    // Forward
    Forward_SO3_Naive_fftw(BW_SANITY, signal, coeffs, workspace_cx, workspace_cx2,
                           workspace_re, weights, NULL, 0);
    
    // Inverse
    Inverse_SO3_Naive_fftw(BW_SANITY, coeffs, reconstructed, workspace_cx, workspace_cx2,
                           workspace_re, NULL, 0);
    
    // Check reconstruction error
    double l2_error = compute_l2_error(signal, reconstructed, n3);
    
    printf("  Reconstruction L2 error: %.2e %s\n", l2_error,
           l2_error < TOLERANCE_L2 ? "✓" : "✗");
    printf("\n");
    
    free(signal);
    free(coeffs);
    free(reconstructed);
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
    free(weights);
    
    if (l2_error < TOLERANCE_L2) {
        printf("Sanity check: PASS ✓\n\n");
        return 0;
    } else {
        fprintf(stderr, "Sanity check: FAIL ✗\n\n");
        return 1;
    }
}

int main(int argc, char **argv)
{
    printf("===========================================\n");
    printf("SO(3) Transform - cuFFT vs FFTW Comparison\n");
    printf("===========================================\n\n");
    
    // Test GPU availability
    if (test_gpu_device() != 0) {
        printf("OVERALL: FAIL ✗ (No GPU)\n");
        printf("===========================================\n");
        return 1;
    }
    
    // Run sanity test first
    if (run_sanity_test() != 0) {
        printf("OVERALL: FAIL ✗ (Sanity check failed)\n");
        printf("===========================================\n");
        return 1;
    }
    
    // Run full comparison test
    if (run_comparison_test(BW_FULL, "Full Comparison Test (bw=128)") != 0) {
        printf("OVERALL: FAIL ✗\n");
        printf("===========================================\n");
        return 1;
    }
    
    printf("OVERALL: PASS ✓\n");
    printf("===========================================\n");
    
    return 0;
}
