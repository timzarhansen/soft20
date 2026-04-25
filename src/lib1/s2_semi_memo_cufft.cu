/***************************************************************************
  S² Spherical Harmonic Transform - cuFFT GPU Backend
  Version 2.0

  GPU Implementation using cuFFT for FFT operations
  Uses interleaved cuDoubleComplex (CUFFT_Z2Z) for double-precision FFTs
***************************************************************************/

#include <cuda.h>
#include <cufft.h>
#include <fftw3.h>
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
 * CUDA kernel: interleave split real/imag arrays into cuDoubleComplex
 */
__global__ void interleave_split_to_z(const double *r, const double *i,
                                       cuDoubleComplex *z, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = make_cuDoubleComplex(r[idx], i[idx]);
    }
}

/**
 * CUDA kernel: deinterleave cuDoubleComplex into split real/imag arrays
 */
__global__ void deinterleave_z_to_split(const cuDoubleComplex *z,
                                         double *r, double *i, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        r[idx] = cuCreal(z[idx]);
        i[idx] = cuCimag(z[idx]);
    }
}

/**
 * Forward spherical harmonic transform using cuFFT
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
    int size = 2 * bw;
    int m, j;
    double tmpSize = 1.0 / ((double)size);
    double tmpA = sqrt(2.0 * M_PI);

    // Workspace pointers (matches s2_semi_memo.c layout)
    double *rres = workspace;           // size*size
    double *ires = rres + (size * size); // size*size
    double *fltres = ires + (size * size); // bw
    double *eval_pts = fltres + bw;      // 2*bw
    double *scratchpad = eval_pts + (2 * bw); // 4*bw

    // Device pointers for GPU FFT (interleaved cuDoubleComplex)
    cuDoubleComplex *d_zdata, *d_zres;
    cufftHandle plan;

    size_t zdata_size = sizeof(cuDoubleComplex) * size * size;
    size_t data_size = sizeof(double) * size * size;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_zdata, zdata_size));
    CUDA_CHECK(cudaMalloc((void**)&d_zres, zdata_size));

    // Copy split host data → device split → interleave on device
    double *d_rdata, *d_idata;
    CUDA_CHECK(cudaMalloc((void**)&d_rdata, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_idata, data_size));
    CUDA_CHECK(cudaMemcpy(d_rdata, rdata, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idata, idata, data_size, cudaMemcpyHostToDevice));

    int n = size * size;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    interleave_split_to_z<<<blocks, threads>>>(d_rdata, d_idata, d_zdata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create cuFFT plan for many 1D Z2Z transforms
    int rank = 1;
    int dims[1] = {size};
    int inembed[1] = {size};
    int onembed[1] = {size};
    int howmany = size;

    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              inembed, 1, size,
                              onembed, 1, size,
                              CUFFT_Z2Z, howmany));

    // Stage 1: FFT along phi (rows) - forward direction
    CUFFT_CHECK(cufftExecZ2Z(plan, d_zdata, d_zres, CUFFT_FORWARD));

    // Deinterleave device → split host
    deinterleave_z_to_split<<<blocks, threads>>>(d_zres, rres, ires, n);
    CUDA_CHECK(cudaDeviceSynchronize());

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
            SemiNaiveReduced(rres + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            dctPlan);

            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);

            SemiNaiveReduced(ires + (m * size),
                            bw,
                            m,
                            fltres,
                            scratchpad,
                            seminaive_naive_table[m],
                            weights,
                            dctPlan);

            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        } else {
            // Naive algorithm for high orders
            Naive_AnalysisX(rres + (m * size),
                            bw,
                            m,
                            weights,
                            fltres,
                            seminaive_naive_table[m],
                            scratchpad);

            memcpy(rdataptr, fltres, sizeof(double) * (bw - m));
            rdataptr += (bw - m);

            Naive_AnalysisX(ires + (m * size),
                            bw,
                            m,
                            weights,
                            fltres,
                            seminaive_naive_table[m],
                            scratchpad);

            memcpy(idataptr, fltres, sizeof(double) * (bw - m));
            idataptr += (bw - m);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_zdata));
    CUDA_CHECK(cudaFree(d_zres));
    CUDA_CHECK(cudaFree(d_rdata));
    CUDA_CHECK(cudaFree(d_idata));
    CUFFT_CHECK(cufftDestroy(plan));
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
                      fftw_plan *idctPlan,
                      fftw_plan *ifftPlan)
{
    int size = 2 * bw;
    int m, j, i;
    double tmpSize = 1.0 / ((double)size);
    double tmpA = sqrt(2.0 * M_PI);

    // Workspace pointers (matches s2_semi_memo.c layout)
    double *rfourdata = workspace;                  // size*size
    double *ifourdata = rfourdata + (size * size);  // size*size
    double *rinvfltres = ifourdata + (size * size); // 2*bw
    double *iminvfltres = rinvfltres + (2 * bw);    // 2*bw
    double *sin_values = iminvfltres + (2 * bw);    // 2*bw
    double *eval_pts = sin_values + (2 * bw);       // 2*bw
    double *scratchpad = eval_pts + (2 * bw);       // 2*bw

    // Device pointers for GPU FFT (interleaved cuDoubleComplex)
    cuDoubleComplex *d_zres, *d_zdata;
    cufftHandle plan;

    size_t zdata_size = sizeof(cuDoubleComplex) * size * size;

    CUDA_CHECK(cudaMalloc((void**)&d_zres, zdata_size));
    CUDA_CHECK(cudaMalloc((void**)&d_zdata, zdata_size));

    // Initialize device output to zero
    CUDA_CHECK(cudaMemset(d_zdata, 0, zdata_size));

    // Precompute eval points and sin values
    ArcCosEvalPts(size, eval_pts);
    for (i = 0; i < size; i++)
        sin_values[i] = sin(eval_pts[i]);

    // Create cuFFT plan for many 1D Z2Z transforms (inverse)
    int rank = 1;
    int dims[1] = {size};
    int inembed[1] = {size};
    int onembed[1] = {size};
    int howmany = size;

    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              inembed, 1, size,
                              onembed, 1, size,
                              CUFFT_Z2Z, howmany));

    // Point to start of input coefficient buffers
    double *rcoeffptr = rcoeffs;
    double *icoeffptr = icoeffs;

    // For each order m, do inverse Legendre transform
    for (m = 0; m < bw; m++) {
        if (m < cutoff) {
            // Semi-naive inverse
            InvSemiNaiveReduced(rcoeffptr,
                               bw,
                               m,
                               rinvfltres,
                               seminaive_naive_table[m],
                               sin_values,
                               scratchpad,
                               idctPlan);

            InvSemiNaiveReduced(icoeffptr,
                               bw,
                               m,
                               iminvfltres,
                               seminaive_naive_table[m],
                               sin_values,
                               scratchpad,
                               idctPlan);

            // Copy to device workspace (interleaved)
            CUDA_CHECK(cudaMemcpy(d_zres + (m * size), rinvfltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_zres + bw + (m * size), iminvfltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));

            rcoeffptr += (bw - m);
            icoeffptr += (bw - m);
        } else {
            // Naive inverse
            Naive_SynthesizeX(rcoeffptr,
                             bw,
                             m,
                             rinvfltres,
                             seminaive_naive_table[m]);

            Naive_SynthesizeX(icoeffptr,
                             bw,
                             m,
                             iminvfltres,
                             seminaive_naive_table[m]);

            // Copy to device workspace (interleaved)
            CUDA_CHECK(cudaMemcpy(d_zres + (m * size), rinvfltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_zres + bw + (m * size), iminvfltres,
                                 sizeof(double) * size, cudaMemcpyHostToDevice));

            rcoeffptr += (bw - m);
            icoeffptr += (bw - m);
        }
    }

    // Fill in zero values where m = bw (from problem definition)
    CUDA_CHECK(cudaMemset(d_zres + (bw * size), 0, sizeof(cuDoubleComplex) * size));

    // Inverse FFT along phi
    CUFFT_CHECK(cufftExecZ2Z(plan, d_zres, d_zdata, CUFFT_INVERSE));

    // Deinterleave device → split host
    int n = size * size;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    deinterleave_z_to_split<<<blocks, threads>>>(d_zdata, rdata, idata, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Normalize
    tmpSize /= tmpA;
    for (j = 0; j < size * size; j++) {
        rdata[j] *= tmpSize;
        idata[j] *= tmpSize;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_zres));
    CUDA_CHECK(cudaFree(d_zdata));
    CUFFT_CHECK(cufftDestroy(plan));
}
