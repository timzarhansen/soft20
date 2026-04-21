/***************************************************************************
  S² Spherical Harmonic Transform - cuFFT GPU Backend
  Version 2.0

  GPU Implementation using cuFFT for FFT operations
***************************************************************************/

#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "makeweights.h"
#include "s2_cospmls.h"
#include "s2_primitive.h"
#include "s2_legendreTransforms.h"

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
 * Forward spherical harmonic transform using cuFFT
 * 
 * rdata, idata: input signal samples (size x size arrays, host memory)
 * rcoeffs, icoeffs: output coefficients (bw x bw arrays, host memory)
 * bw: bandwidth
 * seminaive_naive_table: precomputed transform tables
 * workspace: scratch space
 * dataformat: 0 = complex, 1 = real
 * cutoff: order to switch from semi-naive to naive algorithm
 * fftPlan: cuFFT handle for 1D FFTs
 * weights: quadrature weights
 */
void FST_semi_memo(double *rdata, double *idata,
                   double *rcoeffs, double *icoeffs,
                   int bw,
                   double **seminaive_naive_table,
                   double *workspace,
                   int dataformat,
                   int cutoff,
                   void *fftPlan,  // cufftHandle* for GPU, fftw_plan* for CPU
                   double *weights)
{
    int size = 2 * bw;
    int m, j;
    double tmpSize = 1.0 / ((double)size);
    double tmpA = sqrt(2.0 * M_PI);
    
    // Workspace pointers
    double *rres = workspace;           // size*size
    double *ires = rres + (size * size); // size*size
    double *fltres = ires + (size * size); // bw
    double *eval_pts = fltres + bw;      // 2*bw
    double *scratchpad = eval_pts + (2 * bw); // 4*bw
    
    // Device pointers for GPU FFT
    double *d_rdata, *d_idata, *d_rres, *d_ires;
    cufftHandle plan;
    
    size_t data_size = sizeof(double) * size * size;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_rdata, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_idata, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_rres, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ires, data_size));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_rdata, rdata, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idata, idata, data_size, cudaMemcpyHostToDevice));
    
    // Create cuFFT plan for many 1D FFTs
    // We need to FFT along phi (rows), so: many = size, n = size
    int rank = 1;
    int n = size;
    int howmany = size;
    
    CUFFT_CHECK(cufftPlanMany(&plan, rank, &n,
                              d_rdata, 1, size, d_idata, 1, size,
                              d_rres, 1, size, d_ires, 1, size,
                              CUFFT_C2C, howmany));
    
    // Stage 1: FFT along phi (rows)
    CUFFT_CHECK(cufftExecC2C(plan, d_rdata, d_rres, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_idata, d_ires, CUFFT_FORWARD));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(rres, d_rres, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ires, d_ires, data_size, cudaMemcpyDeviceToHost));
    
    // Normalize
    tmpSize *= tmpA;
    for (j = 0; j < size * size; j++) {
        rres[j] *= tmpSize;
        ires[j] *= tmpSize;
    }
    
    // Point to start of output data buffers
    double *rdataptr = rcoeffs;
    double *idataptr = icoeffs;
    
    // Stage 2: For each order m, do the transform
    for (m = 0; m < bw; m++) {
        if (m < cutoff) {
            // Semi-naive algorithm for low orders
            // Do real part
            SemiNaiveReduced(rres + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            NULL);  // dctPlan not used in GPU version yet
            
            // Load real part of coefficients
            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);
            
            // Do imaginary part
            SemiNaiveReduced(ires + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            NULL);
            
            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        } else {
            // Naive algorithm for high orders
            NaiveReduced(rres + (m * size),
                        bw,
                        m,
                        fltres,
                        scratchpad,
                        seminaive_naive_table[m],
                        weights,
                        NULL);
            
            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);
            
            NaiveReduced(ires + (m * size),
                        bw,
                        m,
                        fltres,
                        scratchpad,
                        seminaive_naive_table[m],
                        weights,
                        NULL);
            
            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_rdata));
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_rres));
    CUDA_CHECK(cudaFree(d_ires));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Inverse spherical harmonic transform using cuFFT
 */
void InvFST_semi_memo(double *rcoeffs, double *icoeffs,
                      double *rdata, double *idata,
                      int bw,
                      double **seminaive_naive_table,
                      double *workspace,
                      int dataformat,
                      int cutoff,
                      void *fftPlan,
                      double *weights)
{
    int size = 2 * bw;
    int m, j;
    double tmpSize = 1.0 / ((double)size);
    double tmpA = sqrt(2.0 * M_PI);
    
    // Workspace pointers
    double *rres = workspace;
    double *ires = rres + (size * size);
    double *fltres = ires + (size * size);
    double *eval_pts = fltres + bw;
    double *scratchpad = eval_pts + (2 * bw);
    
    // Device pointers
    double *d_rres, *d_ires, *d_rdata, *d_idata;
    cufftHandle plan;
    
    size_t data_size = sizeof(double) * size * size;
    
    CUDA_CHECK(cudaMalloc((void**)&d_rres, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ires, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_rdata, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_idata, data_size));
    
    // Initialize device output to zero
    CUDA_CHECK(cudaMemset(d_rdata, 0, data_size));
    CUDA_CHECK(cudaMemset(d_idata, 0, data_size));
    
    // Create cuFFT plan
    int rank = 1;
    int n = size;
    int howmany = size;
    
    CUFFT_CHECK(cufftPlanMany(&plan, rank, &n,
                              d_rres, 1, size, d_ires, 1, size,
                              d_rdata, 1, size, d_idata, 1, size,
                              CUFFT_C2C, howmany));
    
    // Point to start of input coefficient buffers
    double *rcoeffptr = rcoeffs;
    double *icoeffptr = icoeffs;
    
    // For each order m, do inverse transform
    for (m = 0; m < bw; m++) {
        if (m < cutoff) {
            // Semi-naive inverse
            InvSemiNaiveReduced(rcoeffptr,
                               bw,
                               m,
                               fltres,
                               scratchpad,
                               seminaive_naive_table[m],
                               weights,
                               NULL);
            
            // Copy to device workspace
            CUDA_CHECK(cudaMemcpy(d_rres + (m * size), fltres, 
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            
            InvSemiNaiveReduced(icoeffptr,
                               bw,
                               m,
                               fltres,
                               scratchpad,
                               seminaive_naive_table[m],
                               weights,
                               NULL);
            
            CUDA_CHECK(cudaMemcpy(d_ires + (m * size), fltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            
            rcoeffptr += (bw - m);
            icoeffptr += (bw - m);
        } else {
            // Naive inverse
            InvNaiveReduced(rcoeffptr,
                           bw,
                           m,
                           fltres,
                           scratchpad,
                           seminaive_naive_table[m],
                           weights,
                           NULL);
            
            CUDA_CHECK(cudaMemcpy(d_rres + (m * size), fltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            
            InvNaiveReduced(icoeffptr,
                           bw,
                           m,
                           fltres,
                           scratchpad,
                           seminaive_naive_table[m],
                           weights,
                           NULL);
            
            CUDA_CHECK(cudaMemcpy(d_ires + (m * size), fltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            
            rcoeffptr += (bw - m);
            icoeffptr += (bw - m);
        }
    }
    
    // Inverse FFT along phi
    CUFFT_CHECK(cufftExecC2C(plan, d_rres, d_rdata, CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecC2C(plan, d_ires, d_idata, CUFFT_INVERSE));
    
    // Copy back and normalize
    CUDA_CHECK(cudaMemcpy(rdata, d_rdata, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(idata, d_idata, data_size, cudaMemcpyDeviceToHost));
    
    tmpSize /= tmpA;
    for (j = 0; j < size * size; j++) {
        rdata[j] *= tmpSize;
        idata[j] *= tmpSize;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_rres));
    CUDA_CHECK(cudaFree(d_ires));
    CUDA_CHECK(cudaFree(d_rdata));
    CUDA_CHECK(cudaFree(d_idata));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaDeviceSynchronize());
}
