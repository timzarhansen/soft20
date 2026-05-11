#include <cuda.h>
#include <cufft.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "soft_fftw.h"
#include "wrap_fftw.h"
#include "utils_so3.h"
#include "makeweights.h"

#define BW_SANITY 8
#define BW_FULL 128
#define TOLERANCE_L2 1e-5
#define TOLERANCE_MAX 1e-4

#ifdef TEST_BACKEND_CUFFT
    #define BACKEND_NAME "cufft"
#else
    #define BACKEND_NAME "fftw"
#endif

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

int test_gpu_device();
int run_forward_inverse_test(int bw, const char* test_name);
int run_inverse_forward_test(int bw, const char* test_name);
void generate_test_signal(fftw_complex *signal, int bw);
void generate_test_coeffs(fftw_complex *coeffs, int bw);
double compute_l2_error(fftw_complex *a, fftw_complex *b, int size);
double compute_max_error(fftw_complex *a, fftw_complex *b, int size);

int test_gpu_device()
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n\n", prop.multiProcessorCount);

    return 0;
}

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

                signal[idx][0] = sin(alpha) * cos(beta) * cos(gamma) +
                                 0.5 * sin(2.0 * alpha) * sin(beta) * sin(gamma) +
                                 0.25 * cos(3.0 * alpha) * cos(2.0 * beta) * cos(gamma);
                signal[idx][1] = 0.0;
            }
        }
    }
}

void generate_test_coeffs(fftw_complex *coeffs, int bw)
{
    time_t seed;
    time(&seed);
    srand48(seed);

    int num_coeffs = totalCoeffs_so3(bw);
    for (int i = 0; i < num_coeffs; i++) {
        coeffs[i][0] = 2.0 * (drand48() - 0.5);
        coeffs[i][1] = 2.0 * (drand48() - 0.5);
    }
}

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

int run_forward_inverse_test(int bw, const char* test_name)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    int num_coeffs = totalCoeffs_so3(bw);

    printf("%s [Forward->Inverse, %s backend]\n", test_name, BACKEND_NAME);
    printf("  Bandwidth: %d (N=%d, %.1fM samples)\n", bw, n, n3 / 1000000.0);
    printf("  Coefficients: %d\n\n", num_coeffs);

    fftw_complex *signal = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *coeffs = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);
    fftw_complex *reconstructed = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);

    if (!signal || !coeffs || !reconstructed) {
        fprintf(stderr, "ERROR: Failed to allocate memory\n");
        return 1;
    }

    generate_test_signal(signal, bw);

    printf("Forward Transform:\n");
    clock_t start = clock();
    Forward_SO3_Naive_fftw_W(bw, signal, coeffs, 0);
    double forward_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    printf("  %s: %.3f ms\n\n", BACKEND_NAME, forward_time);

    printf("Inverse Transform:\n");
    start = clock();
    Inverse_SO3_Naive_fftw_W(bw, coeffs, reconstructed, 0);
    double inverse_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    printf("  %s: %.3f ms\n", BACKEND_NAME, inverse_time);

    double recon_l2_error = compute_l2_error(signal, reconstructed, n3);
    double recon_max_error = compute_max_error(signal, reconstructed, n3);

    printf("  Reconstruction L2 error: %.2e %s\n", recon_l2_error,
           recon_l2_error < TOLERANCE_L2 ? "PASS" : "FAIL");
    printf("  Reconstruction max error: %.2e %s\n", recon_max_error,
           recon_max_error < TOLERANCE_MAX ? "PASS" : "FAIL");
    printf("\n");

    free(signal);
    free(coeffs);
    free(reconstructed);

    if (recon_l2_error < TOLERANCE_L2 && recon_max_error < TOLERANCE_MAX) {
        return 0;
    } else {
        fprintf(stderr, "ERROR: Reconstruction errors exceed tolerance!\n");
        fprintf(stderr, "  Tolerance: L2 < %.2e, Max < %.2e\n", TOLERANCE_L2, TOLERANCE_MAX);
        return 1;
    }
}

int run_inverse_forward_test(int bw, const char* test_name)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    int num_coeffs = totalCoeffs_so3(bw);

    printf("%s [Inverse->Forward, %s backend]\n", test_name, BACKEND_NAME);
    printf("  Bandwidth: %d (N=%d, %.1fM samples)\n", bw, n, n3 / 1000000.0);
    printf("  Coefficients: %d\n\n", num_coeffs);

    fftw_complex *coeffsIn = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);
    fftw_complex *signal = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *coeffsOut = (fftw_complex*)malloc(sizeof(fftw_complex) * num_coeffs);

    if (!coeffsIn || !signal || !coeffsOut) {
        fprintf(stderr, "ERROR: Failed to allocate memory\n");
        return 1;
    }

    generate_test_coeffs(coeffsIn, bw);

    printf("Inverse Transform:\n");
    clock_t start = clock();
    Inverse_SO3_Naive_fftw_W(bw, coeffsIn, signal, 0);
    double inverse_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    printf("  %s: %.3f ms\n\n", BACKEND_NAME, inverse_time);

    printf("Forward Transform:\n");
    start = clock();
    Forward_SO3_Naive_fftw_W(bw, signal, coeffsOut, 0);
    double forward_time = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;
    printf("  %s: %.3f ms\n", BACKEND_NAME, forward_time);

    double recon_l2_error = compute_l2_error(coeffsIn, coeffsOut, num_coeffs);
    double recon_max_error = compute_max_error(coeffsIn, coeffsOut, num_coeffs);

    printf("  Reconstruction L2 error: %.2e %s\n", recon_l2_error,
           recon_l2_error < TOLERANCE_L2 ? "PASS" : "FAIL");
    printf("  Reconstruction max error: %.2e %s\n", recon_max_error,
           recon_max_error < TOLERANCE_MAX ? "PASS" : "FAIL");
    printf("\n");

    free(coeffsIn);
    free(signal);
    free(coeffsOut);

    if (recon_l2_error < TOLERANCE_L2 && recon_max_error < TOLERANCE_MAX) {
        return 0;
    } else {
        fprintf(stderr, "ERROR: Reconstruction errors exceed tolerance!\n");
        fprintf(stderr, "  Tolerance: L2 < %.2e, Max < %.2e\n", TOLERANCE_L2, TOLERANCE_MAX);
        return 1;
    }
}

int main(int argc, char **argv)
{
    printf("===========================================\n");
    printf("SO(3) Transform - Backend Roundtrip Test\n");
    printf("===========================================\n\n");

    if (test_gpu_device() != 0) {
        printf("OVERALL: FAIL (No GPU)\n");
        printf("===========================================\n");
        return 1;
    }

    printf("--- Sanity Check (bw=%d) ---\n\n", BW_SANITY);

    if (run_forward_inverse_test(BW_SANITY, "Sanity Check") != 0) {
        printf("OVERALL: FAIL (Sanity check F->I failed)\n");
        printf("===========================================\n");
        return 1;
    }

    if (run_inverse_forward_test(BW_SANITY, "Sanity Check") != 0) {
        printf("OVERALL: FAIL (Sanity check I->F failed)\n");
        printf("===========================================\n");
        return 1;
    }

    printf("--- Full Test (bw=%d) ---\n\n", BW_FULL);

    if (run_forward_inverse_test(BW_FULL, "Full Test") != 0) {
        printf("OVERALL: FAIL (Full test F->I failed)\n");
        printf("===========================================\n");
        return 1;
    }

    if (run_inverse_forward_test(BW_FULL, "Full Test") != 0) {
        printf("OVERALL: FAIL (Full test I->F failed)\n");
        printf("===========================================\n");
        return 1;
    }

    printf("OVERALL: PASS\n");
    printf("===========================================\n");

    return 0;
}
