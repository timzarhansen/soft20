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
    int coefHere2;
    double *sinPts, *cosPts, *sinPts2, *cosPts2;
    double *wigners, *scratch;
    double fudge;

    fftw_complex *d_workspace_cx, *d_workspace_cx2;
    cufftHandle plan;

    n = 2 * bw;
    n3 = n * n * n;

   /* cuFFT plan: rank-1 FFT of size n, n*n batches, contiguous layout.
        Contiguous layout: batch k accesses elements [k*n, k*n+n-1].
        D2H/H2D copies transfer data_size = n³ elements. */
    size_t padded_size = sizeof(fftw_complex) * (2 * n3);
    size_t data_size = sizeof(fftw_complex) * n3;

    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx, padded_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx2, padded_size));
    CUDA_CHECK(cudaMemset(d_workspace_cx, 0, padded_size));
    CUDA_CHECK(cudaMemset(d_workspace_cx2, 0, padded_size));

    int rank = 1;
    int nfft = n;
    int howmany = n * n;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, &nfft,
                                NULL, 1, n,
                                NULL, 1, n,
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

    /* Copy input data to device (same as FFTW: memcpy workspace_cx2 from data) */
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, data, data_size, cudaMemcpyHostToDevice));
    fprintf(stderr, "### CUFFT_FWD_COPY: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n", data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]);

    /* Stage 1: FFT the "rows" (INVERSE FFT for forward transform)
       FFTW: fftw_execute(p1) — workspace_cx2 -> workspace_cx (FFTW_BACKWARD, /N)
       cuFFT INVERSE does NOT normalize, so we compensate later */
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx2,
                              (cufftDoubleComplex*)d_workspace_cx, CUFFT_INVERSE));

    /* D2H copy before transpose (FFTW works entirely in host memory) */
    CUDA_CHECK(cudaMemcpy(workspace_cx, d_workspace_cx, data_size, cudaMemcpyDeviceToHost));
    fprintf(stderr, "### CUFFT_FWD_FFT1: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n", workspace_cx[0][0], workspace_cx[0][1], workspace_cx[1][0], workspace_cx[1][1], workspace_cx[2][0], workspace_cx[2][1]);

    /* Stage 2: transpose (same as FFTW) */
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    fprintf(stderr, "### CUFFT_FWD_TR1:  [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n", workspace_cx2[0][0], workspace_cx2[0][1], workspace_cx2[1][0], workspace_cx2[1][1], workspace_cx2[2][0], workspace_cx2[2][1]);

    /* H2D copy for next FFT stage */
    CUDA_CHECK(cudaMemcpy(d_workspace_cx2, workspace_cx2, data_size, cudaMemcpyHostToDevice));

    /* Stage 3: FFT again */
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx2,
                              (cufftDoubleComplex*)d_workspace_cx, CUFFT_INVERSE));

    /* D2H copy before transpose */
    CUDA_CHECK(cudaMemcpy(workspace_cx, d_workspace_cx, data_size, cudaMemcpyDeviceToHost));
    fprintf(stderr, "### CUFFT_FWD_FFT2: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n", workspace_cx[0][0], workspace_cx[0][1], workspace_cx[1][0], workspace_cx[1][1], workspace_cx[2][0], workspace_cx[2][1]);

    /* Stage 4: transpose again */
    transpose_cx(workspace_cx, workspace_cx2, n*n, n);
    fprintf(stderr, "### CUFFT_FWD_TR2:  [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n", workspace_cx2[0][0], workspace_cx2[0][1], workspace_cx2[1][0], workspace_cx2[1][1], workspace_cx2[2][0], workspace_cx2[2][1]);

    /* Stage 5: Wigner transforms (CPU-based, reads from workspace_cx2)
       Same logic as FFTW version */
    fftw_complex *coeffsPtr = coeffs;
    fftw_complex *dataPtr = workspace_cx2;

    /* {f_{0,0}} coefficient */
    genWig_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr = workspace_cx2 + sampHere;
    coeffsPtr = coeffs + coefHere;
    wigNaiveAnalysis_fftw(0, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

    /* m1 from 1 to bw-1: {f_{m1,m1}}, {f_{-m1,-m1}}, {f_{-m1,m1}}, {f_{m1,-m1}} */
    for (m1 = 1; m1 < bw; m1++) {
        genWig_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);

        sampHere = sampLoc_so3(m1, m1, bw);
        coefHere = coefLoc_so3(m1, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftw(m1, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

        sampHere = sampLoc_so3(-m1, m1, bw);
        coefHere = coefLoc_so3(-m1, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftwY(-m1, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

    /* m1 from 1 to bw-1: {f_{m1,0}}, {f_{-m1,0}}, {f_{0,m1}}, {f_{0,-m1}} */
    for (m1 = 1; m1 < bw; m1++) {
        genWig_L2(m1, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);

        sampHere = sampLoc_so3(m1, 0, bw);
        coefHere = coefLoc_so3(m1, 0, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftw(m1, 0, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

        sampHere = sampLoc_so3(0, m1, bw);
        coefHere = coefLoc_so3(0, m1, bw);
        dataPtr = workspace_cx2 + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveAnalysis_fftwX(0, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

    /* m1 from 1 to bw-1, m2 from m1+1 to bw-1: 8 combinations */
    for (m1 = 1; m1 < bw; m1++) {
        for (m2 = m1 + 1; m2 < bw; m2++) {
            genWig_L2(m1, m2, bw, sinPts, cosPts, sinPts2, cosPts2, wigners, scratch);

            sampHere = sampLoc_so3(m1, m2, bw);
            coefHere = coefLoc_so3(m1, m2, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftw(m1, m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

            sampHere = sampLoc_so3(m1, -m2, bw);
            coefHere = coefLoc_so3(m1, -m2, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwY(m1, -m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

            sampHere = sampLoc_so3(m2, m1, bw);
            coefHere = coefLoc_so3(m2, m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwX(m2, m1, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

            sampHere = sampLoc_so3(m2, -m1, bw);
            coefHere = coefLoc_so3(m2, -m1, bw);
            dataPtr = workspace_cx2 + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveAnalysis_fftwY(m1, -m2, bw, dataPtr, wigners, weights, coeffsPtr, workspace_cx);

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

   /* Normalize coefficients
        FFTW: two BACKWARD transforms divide by n^2, then coeffs *= M_PI/(bw*n)
        cuFFT: two INVERSE transforms do NOT divide
        Both produce same total: M_PI/(bw*n^3) */
    double dn = M_PI / ((double)(bw * n * n * n));
    int tmpInt = totalCoeffs_so3(bw);
    for (j = 0; j < tmpInt; j++) {
        coeffs[j][0] *= dn;
        coeffs[j][1] *= dn;
    }

    CUDA_CHECK(cudaFree(d_workspace_cx));
    CUDA_CHECK(cudaFree(d_workspace_cx2));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaDeviceSynchronize());
}

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
    int n3;
    int m1, m2;
    int sampHere, coefHere;
    int sampHere2;
    double *sinPts, *cosPts, *sinPts2, *cosPts2;
    double *wignersTrans, *scratch;
    double fudge;

    fftw_complex *d_workspace_cx, *d_workspace_cx2;
    cufftHandle plan;

    n = 2 * bw;
    n3 = n * n * n;

   /* cuFFT plan: rank-1 FFT of size n, n*n batches, contiguous layout.
        Contiguous layout: batch k accesses elements [k*n, k*n+n-1].
        D2H/H2D copies transfer data_size = n³ elements. */
    size_t padded_size = sizeof(fftw_complex) * (2 * n3);
    size_t data_size = sizeof(fftw_complex) * n3;

    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx, padded_size));
    CUDA_CHECK(cudaMalloc((void**)&d_workspace_cx2, padded_size));
    CUDA_CHECK(cudaMemset(d_workspace_cx, 0, padded_size));
    CUDA_CHECK(cudaMemset(d_workspace_cx2, 0, padded_size));

    int rank = 1;
    int nfft = n;
    int howmany = n * n;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, &nfft,
                                NULL, 1, n,
                                NULL, 1, n,
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

    /* Stage 1: Inverse Wigner transform (CPU-based)
       Wigner synthesis writes to workspace_cx */
    fftw_complex *coeffsPtr = coeffs;
    fftw_complex *dataPtr = workspace_cx;

    /* {f_{0,0}} coefficient */
    genWigTrans_L2(0, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);
    sampHere = sampLoc_so3(0, 0, bw);
    coefHere = coefLoc_so3(0, 0, bw);
    dataPtr = workspace_cx + sampHere;
    coeffsPtr = coeffs + coefHere;
    wigNaiveSynthesis_fftw(0, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

    /* m1 from 1 to bw-1: {f_{m1,m1}}, {f_{-m1,-m1}}, {f_{-m1,m1}}, {f_{m1,-m1}} */
    for (m1 = 1; m1 < bw; m1++) {
        genWigTrans_L2(m1, m1, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);

        sampHere = sampLoc_so3(m1, m1, bw);
        coefHere = coefLoc_so3(m1, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftw(m1, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

        sampHere = sampLoc_so3(-m1, m1, bw);
        coefHere = coefLoc_so3(-m1, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftwY(-m1, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

    /* m1 from 1 to bw-1: {f_{m1,0}}, {f_{-m1,0}}, {f_{0,m1}}, {f_{0,-m1}} */
    for (m1 = 1; m1 < bw; m1++) {
        genWigTrans_L2(m1, 0, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);

        sampHere = sampLoc_so3(m1, 0, bw);
        coefHere = coefLoc_so3(m1, 0, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftw(m1, 0, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

        sampHere = sampLoc_so3(0, m1, bw);
        coefHere = coefLoc_so3(0, m1, bw);
        dataPtr = workspace_cx + sampHere;
        coeffsPtr = coeffs + coefHere;
        wigNaiveSynthesis_fftwX(0, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

    /* m1 from 1 to bw-1, m2 from m1+1 to bw-1: 8 combinations */
    for (m1 = 1; m1 < bw; m1++) {
        for (m2 = m1 + 1; m2 < bw; m2++) {
            genWigTrans_L2(m1, m2, bw, sinPts, cosPts, sinPts2, cosPts2, wignersTrans, scratch);

            sampHere = sampLoc_so3(m1, m2, bw);
            coefHere = coefLoc_so3(m1, m2, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftw(m1, m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

            sampHere = sampLoc_so3(m1, -m2, bw);
            coefHere = coefLoc_so3(m1, -m2, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwY(m1, -m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

            sampHere = sampLoc_so3(m2, m1, bw);
            coefHere = coefLoc_so3(m2, m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwX(m2, m1, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

            sampHere = sampLoc_so3(m2, -m1, bw);
            coefHere = coefLoc_so3(m2, -m1, bw);
            dataPtr = workspace_cx + sampHere;
            coeffsPtr = coeffs + coefHere;
            wigNaiveSynthesis_fftwY(m1, -m2, bw, coeffsPtr, wignersTrans, dataPtr, workspace_cx2);

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

    /* Zero out unused regions in workspace_cx */
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

    fprintf(stderr, "### INV_WIG:   [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            workspace_cx[0][0], workspace_cx[0][1],
            workspace_cx[1][0], workspace_cx[1][1],
            workspace_cx[2][0], workspace_cx[2][1]);

    /* Stage 2: transpose workspace_cx -> workspace_cx2 */
    transpose_cx(workspace_cx, workspace_cx2, n, n*n);

    fprintf(stderr, "### INV_TR1:   [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            workspace_cx2[0][0], workspace_cx2[0][1],
            workspace_cx2[1][0], workspace_cx2[1][1],
            workspace_cx2[2][0], workspace_cx2[2][1]);

    /* H2D for FFT stage */
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx2, data_size, cudaMemcpyHostToDevice));

    fprintf(stderr, "### INV_H2D1:  [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            workspace_cx2[0][0], workspace_cx2[0][1],
            workspace_cx2[1][0], workspace_cx2[1][1],
            workspace_cx2[2][0], workspace_cx2[2][1]);

    /* Stage 3: FFT (FORWARD) */
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx,
                              (cufftDoubleComplex*)d_workspace_cx2, CUFFT_FORWARD));

    /* D2H before transpose - contiguous output, so data_size is sufficient */
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    fprintf(stderr, "### INV_FFT1:  [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            workspace_cx2[0][0], workspace_cx2[0][1],
            workspace_cx2[1][0], workspace_cx2[1][1],
            workspace_cx2[2][0], workspace_cx2[2][1]);

    /* Stage 4: transpose workspace_cx2 -> workspace_cx (n, n*n) to match FFTW */
    transpose_cx(workspace_cx2, workspace_cx, n, n*n);

    fprintf(stderr, "### INV_TR2:   [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            workspace_cx[0][0], workspace_cx[0][1],
            workspace_cx[1][0], workspace_cx[1][1],
            workspace_cx[2][0], workspace_cx[2][1]);

    /* H2D for next FFT stage */
    CUDA_CHECK(cudaMemcpy(d_workspace_cx, workspace_cx, data_size, cudaMemcpyHostToDevice));

    /* Stage 5: FFT again (FORWARD) */
    CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)d_workspace_cx,
                              (cufftDoubleComplex*)d_workspace_cx2, CUFFT_FORWARD));

    /* D2H and copy to output - contiguous output, so data_size is sufficient */
    CUDA_CHECK(cudaMemcpy(workspace_cx2, d_workspace_cx2, data_size, cudaMemcpyDeviceToHost));
    memcpy(data, workspace_cx2, data_size);

    fprintf(stderr, "### INV_OUTPUT: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f]\n",
            data[0][0], data[0][1],
            data[1][0], data[1][1],
            data[2][0], data[2][1]);

  /* Normalize output
        FFTW: two FORWARD transforms (no normalization), then data *= bw/(M_PI*n)
        cuFFT: two FORWARD transforms each divide by n, total 1/n^2
        Need to multiply by n^2 to compensate, so: bw*n^2/(M_PI*n) = bw*n/M_PI
        Wait, FFTW does: bw/(M_PI*n). cuFFT divides by n^2 extra. So multiply by n^2:
        dn = bw/(M_PI*n) * n^2 = bw*n/M_PI */
    double dn = ((double)bw * (double)n) / M_PI;
    for (j = 0; j < n3; j++) {
        data[j][0] *= dn;
        data[j][1] *= dn;
    }
    CUDA_CHECK(cudaFree(d_workspace_cx));
    CUDA_CHECK(cudaFree(d_workspace_cx2));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaDeviceSynchronize());
}

#ifdef __cplusplus
}
#endif
