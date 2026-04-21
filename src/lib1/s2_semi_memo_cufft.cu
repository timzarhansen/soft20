/***************************************************************************
  cuFFT Implementation of S² (Spherical) Harmonic Transforms
  
  GPU-accelerated FST_semi_memo using NVIDIA cuFFT
 **************************************************************************/

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "makeweights.h"
#include "s2_cospmls.h"
#include "s2_primitive.h"
#include "s2_legendreTransforms.h"

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult result = call; \
        if (result != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: error code %d\n", __FILE__, __LINE__, result); \
            exit(1); \
        } \
    } while(0)

/**
 * Forward Spherical Harmonic Transform using cuFFT
 * 
 * GPU-accelerated version of FST_semi_memo
 * Uses cuFFT for the phi-direction FFTs, keeps DCT and semi-naive on CPU
 */
void FST_semi_memo(double *rdata, double *idata,
                   double *rcoeffs, double *icoeffs,
                   int bw,
                   double **seminaive_naive_table,
                   double *workspace,
                   int dataformat,
                   int cutoff,
                   fftw_plan *dctPlan,
                   fftw_plan *fftPlan,
                   double *weights)
{
    int size, m, i, j;
    double *rres, *ires;
    double *fltres, *scratchpad;
    double *eval_pts;
    double tmpSize, tmpA;
    
    size = 2 * bw;
    tmpSize = 1.0 / ((double)size);
    tmpA = sqrt(2.0 * M_PI);
    
    // Workspace allocation
    rres = workspace;              // (size * size)
    ires = rres + (size * size);   // (size * size)
    fltres = ires + (size * size); // bw
    eval_pts = fltres + bw;        // (2*bw)
    scratchpad = eval_pts + (2*bw); // (4*bw)
    
    // Device pointers for FFT
    float *d_rdata, *d_idata, *d_rres, *d_ires;
    cufftHandle fftPlanGPU;
    size_t data_size = sizeof(float) * size * size;
    
    // Allocate device memory (cuFFT uses float for better performance)
    CUDA_CHECK(cudaMalloc(&d_rdata, data_size));
    CUDA_CHECK(cudaMalloc(&d_idata, data_size));
    CUDA_CHECK(cudaMalloc(&d_rres, data_size));
    CUDA_CHECK(cudaMalloc(&d_ires, data_size));
    
    // Copy input data to device (convert double to float)
    float *h_rdata_f = (float*)malloc(data_size);
    float *h_idata_f = (float*)malloc(data_size);
    for (int k = 0; k < size * size; k++) {
        h_rdata_f[k] = (float)rdata[k];
        h_idata_f[k] = (float)idata[k];
    }
    
    CUDA_CHECK(cudaMemcpy(d_rdata, h_rdata_f, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata_f, data_size, cudaMemcpyHostToDevice));
    
    free(h_rdata_f);
    free(h_idata_f);
    
    // Create cuFFT plan for 2D FFT (size x size)
    // We need to FFT along phi (rows), so we do size FFTs of length size
    int rank = 1;
    int n = size;
    int howmany = size;
    
    CUFFT_CHECK(cufftPlanMany(&fftPlanGPU, rank, &n,
                              d_rdata, n, n, 1,
                              d_idata, n, n, 1,
                              d_rres, n, n, 1,
                              d_ires, n, n, 1,
                              CUFFT_C2C, howmany));
    
    // Do the FFTs along phi
    CUFFT_CHECK(cufftExecC2C(fftPlanGPU, d_rdata, d_idata, d_rres, d_ires, CUFFT_INVERSE));
    
    // Copy results back to host
    float *h_rres_f = (float*)malloc(data_size);
    float *h_ires_f = (float*)malloc(data_size);
    
    CUDA_CHECK(cudaMemcpy(h_rres_f, d_rres, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ires_f, d_ires, data_size, cudaMemcpyDeviceToHost));
    
    // Convert to double and normalize
    tmpSize *= tmpA;
    for (j = 0; j < size * size; j++) {
        rres[j] = h_rres_f[j] * tmpSize;
        ires[j] = h_ires_f[j] * tmpSize;
    }
    
    free(h_rres_f);
    free(h_ires_f);
    
    // Cleanup FFT resources
    cufftDestroy(fftPlanGPU);
    CUDA_CHECK(cudaFree(d_rdata));
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_rres));
    CUDA_CHECK(cudaFree(d_ires));
    
    // Point to start of output data buffers
    double *rdataptr = rcoeffs;
    double *idataptr = icoeffs;
    
    // Process each order m using semi-naive or naive algorithm
    for (m = 0; m < bw; m++) {
        if (m < cutoff) {
            // Semi-naive algorithm (uses precomputed tables)
            // Real part
            SemiNaiveReduced(rres + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            dctPlan);
            
            // Copy real coefficients
            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);
            
            // Imaginary part
            SemiNaiveReduced(ires + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            dctPlan);
            
            // Copy imaginary coefficients
            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        } else {
            // Naive algorithm (direct computation)
            // Real part
            Naive(rres + (m * size),
                 bw,
                 m,
                 fltres,
                 scratchpad,
                 eval_pts);
            
            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);
            
            // Imaginary part
            Naive(ires + (m * size),
                 bw,
                 m,
                 fltres,
                 scratchpad,
                 eval_pts);
            
            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
