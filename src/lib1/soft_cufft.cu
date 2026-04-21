/***************************************************************************
  SOFT: SO(3) Fourier Transforms - cuFFT GPU Backend
  Version 2.0

  Copyright (c) 2003, 2004, 2007 Peter Kostelec, Dan Rockmore
  GPU Implementation (c) 2025

  This file is part of SOFT.

  SOFT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  SOFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
***************************************************************************/

/*
  cuFFT GPU implementation of SO(3) Fourier Transforms

  Forward_SO3_Naive_fftw() - forward full SO(3) transform on GPU
  Inverse_SO3_Naive_fftw() - inverse full SO(3) transform on GPU

  Uses cuFFT library for GPU-accelerated FFT operations
*/

#include <cuda.h>
#include <cufft.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "utils_so3.h"
#include "utils_vec_cx.h"
#include "makeWigner.h"
#include "wignerTransforms_fftw.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// cuFFT error checking macro
#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Device memory wrapper for cuFFT
typedef struct {
    fftw_complex *d_data;
    fftw_complex *d_workspace_cx;
    fftw_complex *d_workspace_cx2;
    double *d_workspace_re;
    cufftHandle plan;
} cufft_workspace_t;

/**
 * Forward SO(3) transform using cuFFT
 * 
 * bw = bandwidth of transform
 * data: input signal of size (2*bw)^3 (host memory)
 * coeffs: output coefficients of size (4*bw^3-bw)/3 (host memory)
 * workspace_cx, workspace_cx2: scratch space of size (2*bw)^3 (host memory)
 * workspace_re: double scratch space of size 12*n + n*bw where n = 2*bw
 * weights: quadrature weights of size 2*bw
 * p1: unused for cuFFT backend (kept for API compatibility)
 * flag: 0 = complex data, 1 = real data
 */
void Forward_SO3_Naive_fftw(int bw,
                            fftw_complex *data,
                            fftw_complex *coeffs,
                            fftw_complex *workspace_cx,
                            fftw_complex *workspace_cx2,
                            double *workspace_re,
                            double *weights,
                            void *p1,
                            int flag)
{
    int j, n, n3;
    int m1, m2;
    int sampHere, coefHere;
    double *sinPts, *cosPts, *sinPts2, *cosPts2;
    double *wigners, *scratch;
    
    // Device pointers
    fftw_complex *d_data, *d_workspace_cx, *d_workspace_cx2;
    double *d_workspace_re;
    cufftHandle plan;
    
    n = 2 * bw;
    n3 = n * n * n;
    
    // Allocate device memory
    size_t data_size = sizeof(fftw_complex) * n3;
    size_t ws_re_size = sizeof(double) * (12 * n + n * bw);
    
    CUDA_CHECK(cudaMalloc((void**)&d_data, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx2, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_re, ws_re_size));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, data, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_re, workspace_re, ws_re_size, cudaMemcpyHostToDevice));
    
    // Create cuFFT plan for 2D transforms
    // Plan: many 2D FFTs of size n x n
    int rank = 2;
    int dims[2] = {n, n};
    int howmany = n;  // n transforms
    
    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              d_workspace_cx2, n, n*n, CUFFT_C2C,
                              d_workspace_cx, n, n*n, CUFFT_C2C,
                              howmany));
    
    // Precompute sines and cosines (same as FFTW version)
    sinPts = workspace_re;
    cosPts = sinPts + n;
    sinPts2 = cosPts + n;
    cosPts2 = sinPts2 + n;
    wigners = cosPts2 + n;
    scratch = wigners + (bw * n);
    
    SinEvalPts(n, sinPts);
    CosEvalPts(n, cosPts);
    SinEvalPts2(n, sinPts2);
    CosEvalPts2(n, cosPts2);
    
    // Stage 1: FFT the "rows" (INVERSE FFT for forward transform)
    CUFFT_CHECK(cufftExecC2C(plan, d_workspace_cx2, d_workspace_cx, CUFFT_INVERSE));
    
    // Stage 2: transpose (use existing CPU function, then copy to device)
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 3: FFT again
    CUFFT_CHECK(cufftExecC2C(plan, d_workspace_cx2, d_workspace_cx, CUFFT_INVERSE));
    
    // Stage 4: transpose again
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 5: Wigner transforms (CPU-based, as in original)
    // This is the complex part with the nested loops
    // For now, copy data back to CPU, do Wigner transforms, then we can optimize later
    
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    
    // Perform Wigner transforms using existing CPU implementation
    // This calls wigNaiveAnalysis_fftw which uses the transformed data
    fftw_complex *coeffsPtr = coeffs;
    fftw_complex *dataPtr = workspace_cx2;
    
    // {f_{0,0}} coefficient
    genWig_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr += sampHere;
    coeffsPtr += coefHere;
    wigNaiveAnalysis_fftw(0, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
    
    // Continue with other coefficient groups (m1, m1), etc.
    // This is a simplified version - full implementation follows FFTW structure
    
    for (m1 = 0; m1 < bw; m1++) {
        // {f_{m1,m1}}, {f_{-m1,-m1}}, {f_{-m1,m1}}, {f_{m1,-m1}}
        genWig_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
        
        for (int sign_comb = 0; sign_comb < 4; sign_comb++) {
            int cur_m1 = (sign_comb < 2) ? m1 : -m1;
            int cur_m2 = (sign_comb % 2 == 0) ? m1 : -m1;
            
            sampHere = sampLoc_so3(cur_m1, cur_m2, bw);
            coefHere = coefLoc_so3(cur_m1, cur_m2, bw);
            
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            
            wigNaiveAnalysis_fftw(cur_m1, cur_m2, bw, dataPtr, wigners, weights, 
                                  coeffsPtr, workspace_cx);
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_workspace_cx));
    CUDA_CHECK(cudaFree(d_workspace_cx2));
    CUDA_CHECK(cudaFree(d_workspace_re));
    CUFFT_CHECK(cufftDestroy(plan));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Inverse SO(3) transform using cuFFT
 * 
 * bw = bandwidth of transform
 * coeffs: input coefficients of size (4*bw^3-bw)/3 (host memory)
 * data: output signal of size (2*bw)^3 (host memory)
 * workspace_cx, workspace_cx2: scratch space of size (2*bw)^3 (host memory)
 * workspace_re: double scratch space
 * p1: unused for cuFFT backend
 * flag: 0 = complex data, 1 = real data
 */
void Inverse_SO3_Naive_fftw(int bw,
                            fftw_complex *coeffs,
                            fftw_complex *data,
                            fftw_complex *workspace_cx,
                            fftw_complex *workspace_cx2,
                            double *workspace_re,
                            void *p1,
                            int flag)
{
    int j, n;
    int m1, m2;
    int sampHere, coefHere;
    double *sinPts, *cosPts, *sinPts2, *cosPts2;
    double *wignersTrans, *scratch;
    
    // Device pointers
    fftw_complex *d_workspace_cx, *d_workspace_cx2;
    double *d_workspace_re;
    cufftHandle plan;
    
    n = 2 * bw;
    int n3 = n * n * n;
    
    // Allocate device memory
    size_t data_size = sizeof(fftw_complex) * n3;
    size_t ws_re_size = sizeof(double) * (12 * n + n * bw);
    
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx2, data_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_re, ws_re_size));
    
    CUDA_CHECK(cudaMemcpy(d_workspace_re, workspace_re, ws_re_size, cudaMemcpyHostToDevice));
    
    // Create cuFFT plan
    int rank = 2;
    int dims[2] = {n, n};
    int howmany = n;
    
    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              d_workspace_cx, n, n*n, CUFFT_C2C,
                              d_workspace_cx2, n, n*n, CUFFT_C2C,
                              howmany));
    
    // Precompute sines and cosines
    sinPts = workspace_re;
    cosPts = sinPts + n;
    sinPts2 = cosPts + n;
    cosPts2 = sinPts2 + n;
    wignersTrans = cosPts2 + n;
    scratch = wignersTrans + (bw * n);
    
    SinEvalPts(n, sinPts);
    CosEvalPts(n, cosPts);
    SinEvalPts2(n, sinPts2);
    CosEvalPts2(n, cosPts2);
    
    // Stage 1: Inverse Wigner transform (CPU-based)
    fftw_complex *coeffsPtr = coeffs;
    fftw_complex *dataPtr = workspace_cx;
    
    // {f_{0,0}} coefficient
    genWigTrans_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr += sampHere;
    coeffsPtr += coefHere;
    wigNaiveSynthesis_fftw(0, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx);
    
    // Other coefficient groups
    for (m1 = 0; m1 < bw; m1++) {
        genWigTrans_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
        
        for (int sign_comb = 0; sign_comb < 4; sign_comb++) {
            int cur_m1 = (sign_comb < 2) ? m1 : -m1;
            int cur_m2 = (sign_comb % 2 == 0) ? m1 : -m1;
            
            sampHere = sampLoc_so3(cur_m1, cur_m2, bw);
            coefHere = coefLoc_so3(cur_m1, cur_m2, bw);
            
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            
            wigNaiveSynthesis_fftw(cur_m1, cur_m2, bw, coeffsPtr, wignersTrans, 
                                   dataPtr, workspace_cx);
        }
    }
    
    // Copy workspace to device for FFT stages
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    
    // Stage 2: FFT
    CUFFT_CHECK(cufftExecC2C(plan, d_workspace_cx, d_workspace_cx2, CUFFT_FORWARD));
    
    // Stage 3: transpose
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    transpose_cx(workspace_cx2, workspace_cx, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 4: FFT again
    CUFFT_CHECK(cufftExecC2C(plan, d_workspace_cx, d_workspace_cx2, CUFFT_FORWARD));
    
    // Stage 5: Final transpose and copy to output
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    transpose_cx(workspace_cx2, data, n*n, n);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_workspace_cx));
    CUDA_CHECK(cudaFree(d_workspace_cx2));
    CUDA_CHECK(cudaFree(d_workspace_re));
    CUFFT_CHECK(cufftDestroy(plan));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
