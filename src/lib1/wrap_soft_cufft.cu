/***************************************************************************
  Wrapper Functions for cuFFT SO(3) Transforms
  
  High-level API matching FFTW version for seamless integration
 **************************************************************************/

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <stdio.h>

#include "soft_fftw.h"
#include "makeweights.h"

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * Forward SO(3) transform wrapper
 * 
 * Simplified interface that handles workspace allocation internally
 */
void Forward_SO3_Naive_fftw_W(int bw,
                               fftw_complex *signal,
                               fftw_complex *coeffs,
                               int flag)
{
    int n = 2 * bw;
    int n3 = n * n * n;
    int totalCoeffs = (4 * bw * bw * bw - bw) / 3;
    
    // Allocate workspaces
    fftw_complex *workspace_cx = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    fftw_complex *workspace_cx2 = (fftw_complex*)malloc(sizeof(fftw_complex) * n3);
    double *workspace_re = (double*)malloc(sizeof(double) * (12 * n + n * bw));
    double *weights = (double*)malloc(sizeof(double) * (4 * bw));
    
    // Create FFTW plan (used for metadata even in cuFFT version)
    fftw_plan p1;
    int rank = 2;
    int na[2] = {1, n};
    int inembed[2] = {n, n * n};
    int onembed[2] = {n, n * n};
    int howmany = n * n;
    int istride = 1, idist = n;
    int ostride = 1, odist = n;
    
    p1 = fftw_plan_many_dft(rank, na, howmany,
                            workspace_cx2, inembed, istride, idist,
                            workspace_cx, onembed, ostride, odist,
                            FFTW_BACKWARD, FFTW_ESTIMATE);
    
    // Generate quadrature weights
    makeweights(bw, weights);
    
    // Call the main transform
    Forward_SO3_Naive_fftw(bw, signal, coeffs, workspace_cx, workspace_cx2,
                           workspace_re, weights, &p1, flag);
    
    // Cleanup
    fftw_destroy_plan(p1);
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
    free(weights);
}

/**
 * Inverse SO(3) transform wrapper
 * 
 * Simplified interface that handles workspace allocation internally
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
    
    // Create FFTW plan (used for metadata even in cuFFT version)
    fftw_plan p1;
    int rank = 2;
    int na[2] = {1, n};
    int inembed[2] = {n, n * n};
    int onembed[2] = {n, n * n};
    int howmany = n * n;
    int istride = 1, idist = n;
    int ostride = 1, odist = n;
    
    p1 = fftw_plan_many_dft(rank, na, howmany,
                            workspace_cx2, inembed, istride, idist,
                            workspace_cx, onembed, ostride, odist,
                            FFTW_FORWARD, FFTW_ESTIMATE);
    
    // Call the main transform
    Inverse_SO3_Naive_fftw(bw, coeffs, signal, workspace_cx, workspace_cx2,
                           workspace_re, &p1, flag);
    
    // Cleanup
    fftw_destroy_plan(p1);
    free(workspace_cx);
    free(workspace_cx2);
    free(workspace_re);
}
