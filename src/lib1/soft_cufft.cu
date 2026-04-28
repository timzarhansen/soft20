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
#include <fftw3.h>
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
#ifdef __cplusplus
extern "C" {
#endif

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
    int inembed[2] = {n, n};
    int onembed[2] = {n, n};
    int howmany = n;  // n transforms

    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              inembed, 1, n * n,
                              onembed, 1, n * n,
                              CUFFT_Z2Z, howmany));
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
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx2, (cufftDoubleComplex*)d_workspace_cx, CUFFT_INVERSE));
    
    // Stage 2: transpose (use existing CPU function, then copy to device)
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 3: FFT again
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx2, (cufftDoubleComplex*)d_workspace_cx, CUFFT_INVERSE));
    
    // Stage 4: transpose again
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 5: Wigner transforms (CPU-based, as in original)
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    
    fftw_complex *coeffsPtr = coeffs;
    fftw_complex *dataPtr = workspace_cx2;
    double fudge;
    int coefHere2;
    
    // {f_{0,0}} coefficient
    genWig_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr = workspace_cx2 + sampHere;
    coeffsPtr = coeffs + coefHere;
    wigNaiveAnalysis_fftw(0, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
    
    // m1 from 1 to bw-1: {f_{m1,m1}}, {f_{-m1,-m1}}, {f_{-m1,m1}}, {f_{m1,-m1}}
    for (m1 = 1; m1 < bw; m1++) {
        genWig_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
        
        // {f_{m1,m1}}
        sampHere = sampLoc_so3(m1, m1, bw);
        coefHere = coefLoc_so3(m1, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftw(m1, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        
        // {f_{-m1,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(-m1, -m1, bw);
            coefHere = coefLoc_so3(-m1, -m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftw(-m1, -m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        } else {
            coefHere = coefLoc_so3(m1, m1, bw);
            coefHere2 = coefLoc_so3(-m1, -m1, bw);
            for (j = 0; j < bw - m1; j++) {
                coeffs[coefHere2+j][0] = coeffs[coefHere+j][0];
                coeffs[coefHere2+j][1] = -coeffs[coefHere+j][1];
            }
        }
        
        // {f_{-m1,m1}}
        sampHere = sampLoc_so3(-m1, m1, bw);
        coefHere = coefLoc_so3(-m1, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftwY(-m1, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        
        // {f_{m1,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(m1, -m1, bw);
            coefHere = coefLoc_so3(m1, -m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwY(m1, -m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        } else {
            coefHere = coefLoc_so3(-m1, m1, bw);
            coefHere2 = coefLoc_so3(m1, -m1, bw);
            for (j = 0; j < bw - m1; j++) {
                coeffs[coefHere2+j][0] = coeffs[coefHere+j][0];
                coeffs[coefHere2+j][1] = -coeffs[coefHere+j][1];
            }
        }
    }
    
    // m1 from 1 to bw-1: {f_{m1,0}}, {f_{-m1,0}}, {f_{0,m1}}, {f_{0,-m1}}
    for (m1 = 1; m1 < bw; m1++) {
        genWig_L2(m1, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
        
        // {f_{m1,0}}
        sampHere = sampLoc_so3(m1, 0, bw);
        coefHere = coefLoc_so3(m1, 0, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftw(m1, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        
        // {f_{-m1,0}}
        if (flag == 0) {
            sampHere = sampLoc_so3(-m1, 0, bw);
            coefHere = coefLoc_so3(-m1, 0, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwX(-m1, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        } else {
            coefHere = coefLoc_so3(m1, 0, bw);
            coefHere2 = coefLoc_so3(-m1, 0, bw);
            fudge = ((m1 % 2) == 0) ? 1.0 : -1.0;
            for (j = 0; j < bw - m1; j++) {
                coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
            }
        }
        
        // {f_{0,m1}}
        sampHere = sampLoc_so3(0, m1, bw);
        coefHere = coefLoc_so3(0, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftwX(0, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        
        // {f_{0,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(0, -m1, bw);
            coefHere = coefLoc_so3(0, -m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftw(0, -m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
        } else {
            coefHere = coefLoc_so3(0, m1, bw);
            coefHere2 = coefLoc_so3(0, -m1, bw);
            fudge = ((m1 % 2) == 0) ? 1.0 : -1.0;
            for (j = 0; j < bw - m1; j++) {
                coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
            }
        }
    }
    
    // m1 from 1 to bw-1, m2 from m1+1 to bw-1: 8 combinations
    for (m1 = 1; m1 < bw; m1++) {
        for (m2 = m1 + 1; m2 < bw; m2++) {
            genWig_L2(m1, m2, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
            
            // {f_{m1,m2}}
            sampHere = sampLoc_so3(m1, m2, bw);
            coefHere = coefLoc_so3(m1, m2, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftw(m1, m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            
            // {f_{-m1,-m2}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m1, -m2, bw);
                coefHere = coefLoc_so3(-m1, -m2, bw);
                dataPtr = workspace_cx2 + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveAnalysis_fftwX(-m1, -m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            } else {
                coefHere = coefLoc_so3(m1, m2, bw);
                coefHere2 = coefLoc_so3(-m1, -m2, bw);
                fudge = (((m2-m1) % 2) == 0) ? 1.0 : -1.0;
                for (j = 0; j < bw - m2; j++) {
                    coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                    coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
                }
            }
            
            // {f_{m1,-m2}}
            sampHere = sampLoc_so3(m1, -m2, bw);
            coefHere = coefLoc_so3(m1, -m2, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwY(m1, -m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            
            // {f_{-m1,m2}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m1, m2, bw);
                coefHere = coefLoc_so3(-m1, m2, bw);
                dataPtr = workspace_cx2 + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveAnalysis_fftwY(-m1, m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            } else {
                coefHere = coefLoc_so3(m1, -m2, bw);
                coefHere2 = coefLoc_so3(-m1, m2, bw);
                fudge = (((m2-m1) % 2) == 0) ? 1.0 : -1.0;
                for (j = 0; j < bw - m2; j++) {
                    coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                    coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
                }
            }
            
            // {f_{m2,m1}}
            sampHere = sampLoc_so3(m2, m1, bw);
            coefHere = coefLoc_so3(m2, m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwX(m2, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            
            // {f_{-m2,-m1}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m2, -m1, bw);
                coefHere = coefLoc_so3(-m2, -m1, bw);
                dataPtr = workspace_cx2 + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveAnalysis_fftw(-m2, -m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            } else {
                coefHere = coefLoc_so3(m2, m1, bw);
                coefHere2 = coefLoc_so3(-m2, -m1, bw);
                fudge = (((m2-m1) % 2) == 0) ? 1.0 : -1.0;
                for (j = 0; j < bw - m2; j++) {
                    coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                    coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
                }
            }
            
            // {f_{m2,-m1}}
            sampHere = sampLoc_so3(m2, -m1, bw);
            coefHere = coefLoc_so3(m2, -m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwY(m1, -m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            
            // {f_{-m2,m1}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m2, m1, bw);
                coefHere = coefLoc_so3(-m2, m1, bw);
                dataPtr = workspace_cx2 + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveAnalysis_fftwY(-m1, m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);
            } else {
                coefHere = coefLoc_so3(m2, -m1, bw);
                coefHere2 = coefLoc_so3(-m2, m1, bw);
                fudge = (((m2-m1) % 2) == 0) ? 1.0 : -1.0;
                for (j = 0; j < bw - m2; j++) {
                    coeffs[coefHere2+j][0] = fudge * coeffs[coefHere+j][0];
                    coeffs[coefHere2+j][1] = -fudge * coeffs[coefHere+j][1];
                }
            }
        }
    }
    
    // Normalize coefficients
    double dn = (M_PI / ((double)(bw * n)));
    int tmpInt = totalCoeffs_so3(bw);
    coeffsPtr = coeffs;
    for (j = 0; j < tmpInt; j++) {
        coeffsPtr[j][0] *= dn;
        coeffsPtr[j][1] *= dn;
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
    int inembed[2] = {n, n};
    int onembed[2] = {n, n};
    int howmany = n;

    CUFFT_CHECK(cufftPlanMany(&plan, rank, dims,
                              inembed, 1, n * n,
                              onembed, 1, n * n,
                              CUFFT_Z2Z, howmany));
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
    double dn, fudge;
    int sampHere2;
    
    // {f_{0,0}} coefficient
    genWigTrans_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr = workspace_cx + sampHere;
    coeffsPtr = coeffs + coefHere;
    wigNaiveSynthesis_fftw(0, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
    
    // m1 from 1 to bw-1: {f_{m1,m1}}, {f_{-m1,-m1}}, {f_{-m1,m1}}, {f_{m1,-m1}}
    for (m1 = 1; m1 < bw; m1++) {
        genWigTrans_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
        
        // {f_{m1,m1}}
        sampHere = sampLoc_so3(m1, m1, bw);
        coefHere = coefLoc_so3(m1, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftw(m1, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        
        // {f_{-m1,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(-m1, -m1, bw);
            coefHere = coefLoc_so3(-m1, -m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftw(-m1, -m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        } else {
            sampHere = sampLoc_so3(m1, m1, bw);
            sampHere2 = sampLoc_so3(-m1, -m1, bw);
            for (j = 0; j < 2*bw; j++) {
                workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
            }
        }
        
        // {f_{-m1,m1}}
        sampHere = sampLoc_so3(-m1, m1, bw);
        coefHere = coefLoc_so3(-m1, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftwY(-m1, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        
        // {f_{m1,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(m1, -m1, bw);
            coefHere = coefLoc_so3(m1, -m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwY(m1, -m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        } else {
            sampHere = sampLoc_so3(-m1, m1, bw);
            sampHere2 = sampLoc_so3(m1, -m1, bw);
            for (j = 0; j < 2*bw; j++) {
                workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
            }
        }
    }
    
    // m1 from 1 to bw-1: {f_{m1,0}}, {f_{-m1,0}}, {f_{0,m1}}, {f_{0,-m1}}
    for (m1 = 1; m1 < bw; m1++) {
        genWigTrans_L2(m1, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
        
        // {f_{m1,0}}
        sampHere = sampLoc_so3(m1, 0, bw);
        coefHere = coefLoc_so3(m1, 0, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftw(m1, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        
        // {f_{-m1,0}}
        if (flag == 0) {
            sampHere = sampLoc_so3(-m1, 0, bw);
            coefHere = coefLoc_so3(-m1, 0, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwX(-m1, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        } else {
            sampHere = sampLoc_so3(m1, 0, bw);
            sampHere2 = sampLoc_so3(-m1, 0, bw);
            for (j = 0; j < 2*bw; j++) {
                workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
            }
        }
        
        // {f_{0,m1}}
        sampHere = sampLoc_so3(0, m1, bw);
        coefHere = coefLoc_so3(0, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftwX(0, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        
        // {f_{0,-m1}}
        if (flag == 0) {
            sampHere = sampLoc_so3(0, -m1, bw);
            coefHere = coefLoc_so3(0, -m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftw(0, -m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
        } else {
            sampHere = sampLoc_so3(0, m1, bw);
            sampHere2 = sampLoc_so3(0, -m1, bw);
            for (j = 0; j < 2*bw; j++) {
                workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
            }
        }
    }
    
    // m1 from 1 to bw-1, m2 from m1+1 to bw-1: 8 combinations
    for (m1 = 1; m1 < bw; m1++) {
        for (m2 = m1 + 1; m2 < bw; m2++) {
            genWigTrans_L2(m1, m2, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
            
            // {f_{m1,m2}}
            sampHere = sampLoc_so3(m1, m2, bw);
            coefHere = coefLoc_so3(m1, m2, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftw(m1, m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            
            // {f_{-m1,-m2}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m1, -m2, bw);
                coefHere = coefLoc_so3(-m1, -m2, bw);
                dataPtr = workspace_cx + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveSynthesis_fftwX(-m1, -m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            } else {
                sampHere = sampLoc_so3(m1, m2, bw);
                sampHere2 = sampLoc_so3(-m1, -m2, bw);
                for (j = 0; j < 2*bw; j++) {
                    workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                    workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
                }
            }
            
            // {f_{m1,-m2}}
            sampHere = sampLoc_so3(m1, -m2, bw);
            coefHere = coefLoc_so3(m1, -m2, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwY(m1, -m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            
            // {f_{-m1,m2}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m1, m2, bw);
                coefHere = coefLoc_so3(-m1, m2, bw);
                dataPtr = workspace_cx + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveSynthesis_fftwY(-m1, m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            } else {
                sampHere = sampLoc_so3(m1, -m2, bw);
                sampHere2 = sampLoc_so3(-m1, m2, bw);
                for (j = 0; j < 2*bw; j++) {
                    workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                    workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
                }
            }
            
            // {f_{m2,m1}}
            sampHere = sampLoc_so3(m2, m1, bw);
            coefHere = coefLoc_so3(m2, m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwX(m2, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            
            // {f_{-m2,-m1}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m2, -m1, bw);
                coefHere = coefLoc_so3(-m2, -m1, bw);
                dataPtr = workspace_cx + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveSynthesis_fftw(-m2, -m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            } else {
                sampHere = sampLoc_so3(m2, m1, bw);
                sampHere2 = sampLoc_so3(-m2, -m1, bw);
                for (j = 0; j < 2*bw; j++) {
                    workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                    workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
                }
            }
            
            // {f_{m2,-m1}}
            sampHere = sampLoc_so3(m2, -m1, bw);
            coefHere = coefLoc_so3(m2, -m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwY(m1, -m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            
            // {f_{-m2,m1}}
            if (flag == 0) {
                sampHere = sampLoc_so3(-m2, m1, bw);
                coefHere = coefLoc_so3(-m2, m1, bw);
                dataPtr = workspace_cx + sampHere;
                coeffsPtr = coeffs + coefHere;
                wigNaiveSynthesis_fftwY(-m1, m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);
            } else {
                sampHere = sampLoc_so3(m2, -m1, bw);
                sampHere2 = sampLoc_so3(-m2, m1, bw);
                for (j = 0; j < 2*bw; j++) {
                    workspace_cx[sampHere2+j][0] = workspace_cx[sampHere+j][0];
                    workspace_cx[sampHere2+j][1] = -workspace_cx[sampHere+j][1];
                }
            }
        }
    }
    
    // Zero out unused coefficient regions
    dataPtr = workspace_cx + (n)*(bw);
    for (m1 = 0; m1 < bw; m1++) {
        memset(dataPtr, 0, sizeof(fftw_complex) * n);
        dataPtr += (2*n)*(bw);
    }
    
    dataPtr = workspace_cx + bw*n*(n);
    memset(dataPtr, 0, sizeof(fftw_complex) * n * n);
    dataPtr += n * n + n*bw;
    
    for (m1 = 1; m1 < bw; m1++) {
        memset(dataPtr, 0, sizeof(fftw_complex) * n);
        dataPtr += (2*n)*(bw);
    }
    
    // Copy workspace to device for FFT stages
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    
    // Stage 2: FFT
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx, (cufftDoubleComplex*)d_workspace_cx2, CUFFT_FORWARD));
    
    // Stage 3: transpose
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    transpose_cx(workspace_cx2, workspace_cx, n*n, n);
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));
    
    // Stage 4: FFT again
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx, (cufftDoubleComplex*)d_workspace_cx2, CUFFT_FORWARD));
    
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

#ifdef __cplusplus
}
#endif
