/***************************************************************************
  SOFT: SO(3) Fourier Transforms - cuFFT Wrapper Functions
  Version 2.0

  High-level wrapper functions with automatic memory management
***************************************************************************/

#include <cuda.h>
#include <cufft.h>
#include <fftw3.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "soft_fftw.h"
#include "makeweights.h"

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Forward SO(3) transform with automatic workspace management
 * 
 * bw: bandwidth
 * signal: input signal of size (2*bw)^3
 * coeffs: output coefficients of size (4*bw^3-bw)/3
 * flag: 0 = complex, 1 = real
 */
void Forward_SO3_Naive_fftw_W(int bw,
                              fftw_complex *signal,
                              fftw_complex *coeffs,
                              int flag)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    
    // Allocate workspaces
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * n + n * bw));
    double *weights = (double*)malloc(sizeof(double) * (2 * bw));
    
    // Generate quadrature weights
    makeweights(bw, weights);
    
    // Call the transform
    Forward_SO3_Naive_fftw(bw, signal, coeffs, workspace_cx, workspace_cx2,
                           workspace_re, weights, NULL, flag);
    
    // Cleanup
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
    free(weights);
}

/**
 * Inverse SO(3) transform with automatic workspace management
 * 
 * bw: bandwidth
 * coeffs: input coefficients of size (4*bw^3-bw)/3
 * signal: output signal of size (2*bw)^3
 * flag: 0 = complex, 1 = real
 */
void Inverse_SO3_Naive_fftw_W(int bw,
                              fftw_complex *coeffs,
                              fftw_complex *signal,
                              int flag)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    
    // Allocate workspaces
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * n + n * bw));
    
    // Call the transform
    Inverse_SO3_Naive_fftw(bw, coeffs, signal, workspace_cx, workspace_cx2,
                           workspace_re, NULL, flag);
    
    // Cleanup
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
}
