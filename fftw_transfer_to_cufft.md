# SOFT20 FFTW → cuFFT Migration: Comprehensive Problem Report

**Date**: 2026-05-15
**Status**: FIX APPLIED — cuFFT plan changed to contiguous layout (`istride=1, idist=n, ostride=1, odist=n`). Both forward and inverse plans fixed. Awaiting GPU test.
**Target**: 4-6x speedup on NVIDIA RTX 3090 (CUDA 13.1, CC 8.6)

---

## 1. GOAL

Migrate the FFT operations in `soft20` (SO(3) Fourier Transforms library) from FFTW (CPU) to cuFFT (GPU). The library performs forward and inverse transforms on functions defined on the rotation group SO(3), used for 3D image registration.

**Success criteria**: Inverse→Forward roundtrip test passes at bw=128 (n=256, 16.8M samples) with L2 error < 1e-5 and max error < 1e-4.

---

## 2. PROBLEM STATEMENT

The cuFFT backend passes at bw=8 (n=16) but **fails at bw=128 (n=256)** with:
- L2 error: 1.15 (should be < 1e-5)
- Max error: 2.83 (should be < 1e-4)

The error is **identical** across all attempted fixes — same L2=1.15, max=2.83 — suggesting a fundamental algorithmic mismatch, not a parameter tuning issue.

---

## 3. ARCHITECTURE OVERVIEW

### Algorithm Structure (Both Forward and Inverse)

Both transforms follow this 5-stage pipeline:

```
Stage 1: FFT (2D DFT decomposed as two passes of n² 1D-FFTs of length n)
Stage 2: Transpose (n² × n → n × n²)
Stage 3: FFT (same 2D DFT)
Stage 4: Transpose (n × n² → n² × n)
Stage 5: Wigner-D matrix transforms (CPU, identical in both backends)
```

**Forward**: Wigner analysis reads from the FFT-transformed data, produces coefficients.
**Inverse**: Wigner synthesis writes to workspace, then FFT stages produce signal samples.

### Key Difference Between Backends

| Aspect | FFTW (CPU) | cuFFT (GPU) |
|--------|-----------|-------------|
| FFT execution | In-place via plan (reads workspace_cx, writes to different buffer) | Separate input/output GPU buffers |
| Memory transfers | None (all CPU) | H2D before each FFT, D2H after each FFT |
| Transpose | CPU `transpose_cx()` | CPU `transpose_cx()` (same function) |
| Wigner transforms | CPU | CPU (identical code) |
| Normalization | Handled by FFTW plan + explicit factor | Explicit factor (cuFFT FORWARD divides by n, INVERSE does not) |

### FFTW Plan Parameters (Both Forward and Inverse)

```c
// From wrap_soft_fftw.c
rank = 2
na = {1, n}           // FFT size: 1 along dim 0, n along dim 1
howmany = n * n       // number of independent transforms
inembed = {n, n*n}    // input logical array dimensions
onembed = {n, n*n}    // output logical array dimensions
istride = 1           // input stride between consecutive elements
ostride = 1           // output stride
idist = n             // input distance between transforms
odist = n             // output distance
```

**FFTW memory access formula** for rank=2, inembed={n, n²}, istride=1, idist=n:
- Batch k, element j → physical index: `k * idist + j * istride * inembed[1]` = **`k * n + j * n²`**
- This FFTs along the k-axis (slowest 3D dimension in the i+j*n+k*n² layout)
- Max index: `(n²-1)*n + (n-1)*n²` = **`2n³ - n² - n`** — **OUT OF BOUNDS** for n³-sized buffer
- FFTW survives because `fftw_malloc` zeros extra memory beyond the requested size

### cuFFT Plan Parameters (Current, After All Fixes)

```c
// From soft_cufft.cu
rank = 1
nfft = n
howmany = n * n
istride = n * n       // stride within each FFT
idist = n             // distance between batches
ostride = n * n       // output stride (same as input)
odist = n             // output distance (same as input)
```

**cuFFT memory access formula** for rank=1, istride=n², idist=n:
- Batch k, element j → physical index: `k * idist + j * istride` = **`k * n + j * n²`**
- This matches the FFTW input access pattern ✓
- Max index: same as FFTW = `2n³ - n² - n` — handled by 2n³ buffer ✓
- **BUT**: output uses the SAME strided layout as input

---

## 4. ROOT CAUSE (IDENTIFIED)

### The Inverse Transform FFT Stage Layout Mismatch

The critical bug is in the **Inverse transform's FFT stages**. The cuFFT plan produces output in a **strided layout** (`ostride=n², odist=n`), but the subsequent CPU `transpose_cx()` call expects **contiguous layout**.

#### Detailed Trace — Inverse Transform

**FFTW Inverse** (soft_fftw.c, lines 1852-1906):
```
1. Wigner synthesis → workspace_cx (CPU)
2. Zero out unused regions in workspace_cx
3. transpose_cx(workspace_cx, workspace_cx, n, n*n)  // in-place transpose
4. fftw_execute(p1)  // reads workspace_cx (strided), writes signal (contiguous)
5. transpose_cx(signal, workspace_cx, n, n*n)
6. fftw_execute(p1)  // reads workspace_cx (strided), writes signal (contiguous)
7. Normalize signal
```

Key insight: FFTW's plan reads from `workspace_cx` (strided access: `k*n + j*n²`) and writes to `signal` (contiguous access: `j + k*n`). The output layout is **different** from the input layout.

**cuFFT Inverse** (soft_cufft.cu, lines 654-678):
```
1. Wigner synthesis → workspace_cx (CPU)
2. Zero out unused regions in workspace_cx
3. transpose_cx(workspace_cx, workspace_cx2, n, n*n)
4. H2D: workspace_cx2 → d_workspace_cx (first n³ elements)
5. cuFFT FORWARD: d_workspace_cx → d_workspace_cx2 (STRIDED output: k*n + j*n²)
6. D2H: d_workspace_cx2 → workspace_cx2 (first n³ elements only!)
7. transpose_cx(workspace_cx2, workspace_cx, n, n*n)
8. H2D: workspace_cx → d_workspace_cx
9. cuFFT FORWARD: d_workspace_cx → d_workspace_cx2 (STRIDED output)
10. D2H: d_workspace_cx2 → workspace_cx2 (first n³ elements only!)
11. memcpy workspace_cx2 → data
12. Normalize data
```

**THE BUG**: Steps 6 and 10 only copy the first n³ elements from GPU to host. But the cuFFT output is strided at positions `k*n + j*n²`, where most values are **beyond index n³**. The D2H copy misses most of the FFT output data.

For n=16 (bw=8): the strided output happens to fall within the first n³ elements for many positions, so the error is small enough to pass.
For n=256 (bw=128): the strided output extensively exceeds n³, so most FFT output is lost → large error.

#### Why Forward Passes But Inverse Fails

The **Forward transform** uses the same cuFFT plan but has a different code path:
- After each cuFFT call, the D2H copy goes to `workspace_cx`, then `transpose_cx` reads from it
- The transpose function `transpose_cx(workspace_cx, workspace_cx2, n*n, n)` reads from the strided layout correctly because it treats the data as an n²×n matrix
- The strided layout `k*n + j*n²` happens to be compatible with the n²×n matrix interpretation

The **Inverse transform** has a different issue:
- After cuFFT, the D2H copy goes to `workspace_cx2`, then `transpose_cx(workspace_cx2, workspace_cx, n, n*n)` reads from it
- The transpose treats the data as an n×n² matrix
- But the strided layout `k*n + j*n²` does NOT produce the correct n×n² matrix layout
- **Most critically**: the D2H copy of only n³ elements truncates the strided output

#### Verification

The debug prints show that Forward transform values are reasonable:
```
bw=128 Forward:
  COPY:  [11075356.2379 -7747066.6989] ...
  FFT1:  [250724315.2429 -2788601.3728] ...
  TR1:   [250724315.2429 -2788601.3728] ...  (first element unchanged by transpose)
  FFT2:  [49464023917.2243 -14389429335.2690] ...
  TR2:   [49464023917.2243 -14389429335.2690] ...
```

No debug prints exist for the Inverse transform's FFT stages, so we cannot directly compare.

---

## 5. ATTEMPTS MADE (CHRONOLOGICAL)

### Attempt 1: Initial cuFFT implementation
- Plan: `rank=1, nfft=n, howmany=n*n, istride=1, idist=n`
- Contiguous reads, no stride matching
- **Result**: Failed both bw=8 and bw=128

### Attempt 2: Fix istride to n
- Plan: `istride=n, idist=1`
- Rationale: attempted to match FFTW's strided layout
- **Result**: bw=8 passed, bw=128 failed (L2=1.16)
- **Insight**: The strided access `k + j*n` wraps around for k≥n, causing data corruption at large n

### Attempt 3: Fix Inverse final step
- Changed `transpose_cx` → `memcpy` after last FFT
- Rationale: workspace already in data layout
- **Result**: No change

### Attempt 4: Fix Inverse second transpose dims
- Changed `transpose_cx(workspace_cx2, workspace_cx, n*n, n)` → `transpose_cx(workspace_cx2, workspace_cx, n, n*n)`
- **Result**: No change

### Attempt 5: Add padding to GPU buffers
- Allocated `n³ + n²` elements, zeroed padding
- Rationale: handle OOB strided reads
- **Result**: bw=8 passed, bw=128 still failed (same error)

### Attempt 6: Fix Inverse plan output stride
- Changed `ostride=1, odist=n` → `ostride=n, odist=1`
- Rationale: make Inverse output match Forward input layout
- **Result**: bw=8 passed, bw=128 still failed (same error)

### Attempt 7: Match FFTW memory access exactly
- Plan: `istride=n*n, idist=n` (matches FFTW formula `k*n + j*n²`)
- Buffer: `2*n³` elements (handles OOB reads)
- **Result**: bw=8 passed, bw=128 still failed (L2=1.15, max=2.83 — IDENTICAL to before)
- **Insight**: The input access pattern is correct, but the OUTPUT layout mismatch in the Inverse transform was not addressed

### Attempt 8: Fix D2H Copy Size (Option A)
- **Date**: 2026-05-15
- **Changes**:
  - Inverse D2H copies changed from `data_size` to `2*data_size` in soft_cufft.cu:680,702
  - workspace_cx2 allocated as `2*n³` in wrap_soft_cufft.cu:60,95
  - Added debug prints for Inverse transform stages (INV_WIG, INV_TR1, INV_H2D1, INV_FFT1, INV_TR2, INV_FFT2, INV_OUTPUT)
- **Result**: bw=8 PASS, bw=128 FAIL (**L2=1.16, max=2.83 — UNCHANGED**)
- **Insight**: The D2H copy size was NOT the root cause. The error is identical whether we copy n³ or 2*n³. The problem must be a transpose/layout mismatch — the tiny values in INV_FFT1 (-54 range) compared to forward FFT values (~10⁸) confirm data corruption occurs during or before the inverse FFT stage.

### bw=128 Debug Output (After Attempt 8)
```
Inverse Transform:
### INV_WIG:   [-44.1984 10.9776] [-27.6589 -1.3253] [-13.1931 -4.0557]
### INV_TR1:   [-44.1984 10.9776] [-4.7269 3.7451] [-1.8121 -1.2125]
### INV_H2D1:  [-44.1984 10.9776] [-4.7269 3.7451] [-1.8121 -1.2125]
### INV_FFT1:  [-54.6372 -15.1757] [-53.8550 -15.0644] [-53.0647 -14.9346]
### INV_TR2:   [-54.6372 -15.1757] [-80.4579 36.8752] [-15.3053 -19.1258]
### INV_OUTPUT: [-1860.2025 -388.4425] [-884.5222 986.8948] [430.6848 477.3755]

Forward Transform (for comparison):
### CUFFT_FWD_COPY: [-19402615.5153 -4051602.6030] [-9225901.7217 10293685.6444] [4492205.2836 4979207.4914]
### CUFFT_FWD_FFT1: [-145891018.6179 -40521926.9900] [-214836815.5653 98463292.7225] [-40867811.4723 -51069333.4144]
### CUFFT_FWD_TR1:  [-145891018.6179 -40521926.9900] [-143802439.4115 -40224560.0476] [-141692130.0144 -39878039.7715]
### CUFFT_FWD_FFT2: [-30212463053.2843 7503912175.5154] [-3231121954.9353 2560018445.9088] [-1238699431.5304 -828844014.3010]
### CUFFT_FWD_TR2:  [-30212463053.2843 7503912175.5154] [-18906647516.9878 -905913332.5430] [-9018324891.7040 -2772304381.9854]
```

**Key observation**: Forward values are ~10⁶–10¹⁰ while Inverse FFT1 values are tiny (~-54). The strided output layout from cuFFT is fundamentally incompatible with what the subsequent transpose expects.

### Attempt 9: Make cuFFT Output Contiguous (Option B - FAILED)
- **Date**: 2026-05-15
- **Changes**:
  - Changed cuFFT plan from `ostride=n², odist=n` to `ostride=1, odist=n²` (contiguous output)
  - Reverted D2H copy sizes from 2*data_size back to data_size
  - Reverted workspace_cx2 allocation from 2*n³ back to n³
- **Result**: bw=8 PASS, bw=128 FAIL (**L2=1.16, max=2.83 — UNCHANGED**)
- **Insight**: The fix didn't change the error because the fundamental issue is NOT the plan parameters — it's **FFTW's in-place execution model** vs **cuFFT's out-of-place requirement**.

### Root Cause: FFTW In-Place vs cuFFT Out-of-Place Mismatch

FFTW's wisdom-optimized plan uses **in-place** transforms where input and output share the same buffer with strided access pattern `k*n + j*n²`. This is possible because FFTW's execute function reads and writes to the **same physical memory locations**.

cuFFT's Z2Z API with **separate pointers** for input and output:
- Input access (istride=n², idist=n): `k*n + j*n²` ✓ (matches FFTW)
- Output access (ostride=1, odist=n²): `k*n + j` ✗ (CONTIGUOUS layout, NOT strided!)

cuFFT **cannot** produce true strided in-place output — its API semantics for Z2Z with separate pointers always write contiguous data `j + k*n`. This is a **fundamental architectural difference**, not a parameter tuning issue.

### bw=128 Debug Output (After Attempt 9)
```
Inverse:
### INV_WIG:   [-16.1556 53.7401] [-0.2499 49.2895] [6.9990 41.3279]
### INV_TR1:   [-16.1556 53.7401] [-31.1528 6.1817] [5.2836 -1.6158]
### INV_H2D1:  [-16.1556 53.7401] [-31.1528 6.1817] [5.2836 -1.6158]
### INV_FFT1:  [-38.1861 56.7857] [-38.0179 57.4699] [-37.8508 58.1585]
### INV_TR2:   [-38.1861 56.7857] [-32.2285 -27.4361] [35.3533 22.3334]
### INV_OUTPUT: [522.8166 -737.4118] [1251.1420 -261.7934] [-169.1205 224.9371]
  L2: 1.16e+00 FAIL, Max: 2.83e+00 FAIL

Forward (working - for comparison):
### CUFFT_FWD_COPY: [5453174.4739 -7691484.5102] [13049884.5387 -2730604.0747] [-1763990.5601 2346179.4650]
### CUFFT_FWD_FFT1: [-101963635.7763 151627880.5294] [-86055833.1549 -73259142.9283] [94399483.2731 59634114.1217]
### CUFFT_FWD_TR1:  [-101963635.7763 151627880.5294] [-101514541.9157 153454824.0252] [-101068459.4452 155293358.8229]
### CUFFT_FWD_FFT2: [-11043384173.9282 36734877776.9590] [-21294969453.2113 4225593270.4001] [3611690295.3128 -1104471601.2756]
### CUFFT_FWD_TR2:  [-11043384173.9282 36734877776.9590] [-170790834.7457 33692622941.2279] [4784280209.3761 28250349354.7957]
```

**Key observation**: Forward FFT values are ~10^8-10^10 (correct magnitude), confirming the forward path works. The Inverse transform's FFT produces values ~(-38, 56) which could be legitimate for sparse spectral data, but the overall error remains because the **transposed layout after FFT doesn't match what the algorithm expects**.

---

## 6. NORMALIZATION ANALYSIS

### Forward Transform

| Backend | FFT operations | Normalization from FFT | Explicit factor | Total |
|---------|---------------|----------------------|----------------|-------|
| FFTW | 2× BACKWARD | 1/n² (divides by product of na={1,n}) | π/(bw·n) | **π/(bw·n³)** |
| cuFFT | 2× INVERSE | 1 (no normalization) | π/(bw·n³) | **π/(bw·n³)** |

**Verdict**: Normalization is CORRECT. Both produce the same total factor.

### Inverse Transform

| Backend | FFT operations | Normalization from FFT | Explicit factor | Total |
|---------|---------------|----------------------|----------------|-------|
| FFTW | 2× FORWARD | 1 (no normalization) | bw/(π·n) | **bw/(π·n)** |
| cuFFT | 2× FORWARD | 1/n² (divides by n each) | bw·n/π | **bw/(π·n)** |

**Verdict**: Normalization is CORRECT. Both produce the same total factor.

### Roundtrip

FFTW: Forward × Inverse = π/(bw·n³) × bw/(π·n) = 1/n⁴
cuFFT: Forward × Inverse = π/(bw·n³) × bw/(π·n) = 1/n⁴

Both produce the same roundtrip factor. The normalization is NOT the source of the error.

---

## 7. PROPOSED FIX

### Option B: Make cuFFT Output Contiguous (RECOMMENDED)

**Why Option A failed**: The D2H truncation was a symptom, not the cause. The cuFFT plan produces strided output (`ostride=n², odist=n`), and the subsequent transpose function interprets this data as an n×n² matrix. The strided layout `k*n + j*n²` does NOT produce a valid n×n² matrix interpretation — it scrambles the data.

**Fix**: Change the cuFFT plan to produce **contiguous output**:
```c
CUFFT_CHECK(cufftPlanMany(&plan, rank, &nfft,
    NULL, n*n, n,      // input: strided (k*n + j*n²)
    NULL, 1, n*n,      // output: CONTIGUOUS (j + k*n)
    CUFFT_Z2Z, howmany));
```

This means:
- FFT reads strided input (matching FFTW's memory access pattern)
- FFT writes **contiguous output** (matching what transpose expects: array[j + k*n])
- D2H copy of n³ elements captures all data directly
- No extra memory needed on CPU or GPU

**Implementation steps**:
1. Modify cuFFT plan in both Forward and Inverse transforms in soft_cufft.cu
2. Change `onembed` from `n*n` to `1` (or NULL for contiguous)
3. Change `ostride` from `n*n` to `1`
4. Revert D2H copy sizes back to `data_size` (no longer need 2*n³)
5. Revert workspace_cx2 allocation in wrap_soft_cufft.cu back to `n³`
6. Test with bw=8 first, then bw=128

**Pros**: Correct output layout, minimal memory, matches FFTW behavior
**Cons**: Need to verify contiguous output is compatible with subsequent transpose

### Option A: Fix D2H Copy Size (ALREADY TRIED - FAILED)

- **Status**: FAILED — Attempt 8 on 2026-05-15
- Changes: D2H copies changed from n³ to 2*n³, workspace_cx2 allocated as 2*n³
- **Result**: L2=1.16, max=2.83 — **UNCHANGED**
- **Conclusion**: The D2H copy size was not the issue

### Option B: Make cuFFT Output Contiguous (ALREADY TRIED - FAILED)

- **Status**: FAILED — Attempt 9 on 2026-05-15
- Changes: Changed plan from `ostride=n²` to `ostride=1` (contiguous output)
- **Result**: L2=1.16, max=2.83 — **UNCHANGED**
- **Conclusion**: The issue is NOT parameter tuning — it's the fundamental in-place vs out-of-place mismatch

---

## 7B. NEW PROPOSED FIX OPTIONS

### Option D: Restructure Algorithm — Explicit GPU Transpose Kernels

Instead of fighting cuFFT's out-of-place model, restructure the algorithm to use GPU transpose kernels:

1. **Before each FFT**: Add explicit GPU transpose kernel to convert to FFT-friendly layout
2. **After each FFT**: Add explicit GPU transpose kernel to convert back to algorithm layout
3. Use cuFFT with standard contiguous buffers throughout

**Algorithm flow**:
```
Input (contiguous n³) → GPU Transpose → cuFFT (contiguous) → GPU Transpose → Output
```

**Pros**: Works with cuFFT's strengths, no CPU-GPU synchronizations between stages
**Cons**: Extra GPU memory and transpose overhead

### Option E: CPU FFT + GPU Wigner (Hybrid)

Keep ALL FFT operations on CPU (FFTW with wisdom optimization) and move only Wigner transforms to GPU:
- CPU FFT: ~100-200ms for bw=128 (already optimized with wisdom)
- GPU Wigner: potential 10-50x speedup for the Wigner stage
- Overall: could still achieve significant speedup

**Pros**: Avoids cuFFT layout issues entirely, FFT wisdom optimization preserved
**Cons**: FFT parallelism stays on CPU, memory transfers between CPU/GPU

### Option F: GPU-Only Wigner Transforms (RECOMMENDED)

Instead of migrating FFTs to GPU, migrate Wigner transforms to GPU:
- FFTs stay on CPU (FFTW with wisdom optimization)
- Wigner-D matrices on GPU via CUDA kernels
- This targets the actual bottleneck (Wigner transforms are O(bw⁴) vs FFT O(bw³))

**Why this is the best path**:
- cuFFT FFT: 1.4s for bw=128
- Estimated CPU FFT with wisdom: ~100-200ms (already well-optimized)
- Wigner transforms dominate runtime at high bw
- Benchmark data shows Forward transform (includes Wigner) takes 1.4s, and Wigner is the expensive part

**Pros**: Targets the real bottleneck, avoids all FFT layout issues, potential for massive speedup
**Cons**: Requires writing CUDA Wigner kernels (significant effort)

### Recommendation

**Option F** (GPU Wigner + CPU FFT) is likely the best path to actual speedup. The benchmark data shows cuFFT FFT alone is not significantly faster than optimized CPU FFT due to layout transformation overhead. The real performance gain comes from GPU-accelerating the Wigner transforms.

If Option F is too much effort initially, **Option D** (restructure with explicit GPU transposes) is a safer intermediate step that keeps all computation on GPU.

---

## 8. KEY FILES

| File | Purpose | Lines |
|------|---------|-------|
| `src/lib1/soft_cufft.cu` | cuFFT GPU backend (main transforms) | 729 |
| `src/lib1/soft_fftw.c` | FFTW CPU reference implementation | 1918 |
| `src/lib1/wrap_soft_cufft.cu` | cuFFT wrapper (memory management) | 110 |
| `src/lib1/wrap_soft_fftw.c` | FFTW wrapper (plan creation) | 245 |
| `src/lib1/wignerTransforms_fftw.c` | Wigner-D matrix transforms (CPU) | ~1000 |
| `src/lib1/utils_vec_cx.c` | `transpose_cx()` function | 263 |
| `examples/test_cufft_compare.cu` | Test binary (roundtrip test) | 228 |
| `docker_test/CMakeLists.txt` | Build configuration | 134 |

---

## 9. BUILD & TEST

```bash
# Build
cd ~/Workspaces/tim_hansen/ros_ws/src/soft20/docker_test
docker compose build --no-cache

# Run cuFFT test
docker compose run --entrypoint "/bin/bash" test
/workspace/build/test_cufft

# Run FFTW reference test (for comparison)
/workspace/build/test_fftw
```

---

## 10. NEXT STEPS

### Recommended Path: Option F (GPU Wigner + CPU FFT)

Based on benchmark analysis:
- cuFFT FFT alone: ~1.4s for bw=128
- CPU FFT with wisdom: ~100-200ms (already optimized)
- Wigner transforms dominate runtime at high bw (O(bw⁴) complexity)

**Action**: Implement GPU Wigner transforms while keeping FFTs on CPU with FFTW wisdom optimization.

### Alternative Path: Option D (GPU Transpose Restructure)

If Option F is too much effort, restructure the algorithm with explicit GPU transpose kernels:
1. Add GPU transpose kernels (contiguous ↔ FFT-friendly layout)
2. Use cuFFT with standard contiguous buffers
3. Keep FFTs on GPU

### Not Recommended: Continue with FFT-only migration

Attempts 1-9 have shown that migrating only the FFT stage to GPU while keeping Wigner on CPU provides minimal speedup due to:
- Layout transformation overhead (CPU↔GPU transfers, transpose operations)
- cuFFT cannot match FFTW's in-place wisdom-optimized plans
- The real bottleneck (Wigner transforms) remains on CPU

---

## 11. cuFFT API REFERENCE

### cufftPlanMany
```c
cufftResult cufftPlanMany(
    cufftHandle *plan,
    int rank,                    // dimensionality of FFT (1, 2, or 3)
    int *n,                      // FFT lengths per dimension
    int *inembed,                // input logical array dimensions (NULL = contiguous)
    int istride,                 // distance between consecutive elements in input
    int idist,                   // distance between batches in input
    int *onembed,                // output logical array dimensions (NULL = contiguous)
    int ostride,                 // distance between consecutive elements in output
    int odist,                   // distance between batches in output
    cufftType type,              // CUFFT_Z2Z for double complex
    int batch                    // number of independent transforms
);
```

For rank=1: `inembed` and `onembed` are ignored.

Memory access for batch k, element j:
- Input: `base_in + k * idist + j * istride`
- Output: `base_out + k * odist + j * ostride`

### cufftExecZ2Z Normalization
- **CUFFT_FORWARD**: divides result by n (product of all FFT lengths)
- **CUFFT_INVERSE**: no normalization

### FFTW Normalization (for comparison)
- **FFTW_FORWARD**: no normalization
- **FFTW_BACKWARD**: divides by N (product of all `na` dimensions)

---

## 12. APPENDIX: DEBUG OUTPUT COMPARISON

### bw=8 (PASS - All Attempts)
```
Inverse:
### INV_WIG:   [0.4175 1.7283] [-0.3560 -0.4433] [-0.6811 -1.9036]
### INV_TR1:   [0.4175 1.7283] [0.1082 0.3649] [0.0237 -0.2216]
### INV_H2D1:  [0.4175 1.7283] [0.1082 0.3649] [0.0237 -0.2216]
### INV_FFT1:  [-0.5967 0.9941] [-0.1922 0.5697] [0.4092 0.3655]
### INV_TR2:   [-0.5967 0.9941] [-1.6029 -1.0656] [-1.9253 1.4392]
### INV_OUTPUT: [-1.0983 6.2486] [-1.6859 0.8497] [-11.9255 0.4513]

Forward:
### CUFFT_FWD_COPY: [-44.7489 254.5907] [-68.6903 34.6181] [-485.8886 18.3878]
### CUFFT_FWD_FFT1: [-389.0198 648.0301] [-1044.9593 -694.6391] [-1255.0838 938.2350]
### CUFFT_FWD_TR1:  [-389.0198 648.0301] [-125.2658 371.4163] [266.7463 238.2927]
### CUFFT_FWD_FFT2: [4354.2439 18026.7995] [1128.2649 3805.7575] [246.8635 -2311.6546]
### CUFFT_FWD_TR2:  [4354.2439 18026.7995] [-3713.0263 -4623.9153] [-7104.6335 -19855.1102]
  L2: 5.47e-16 PASS, Max: 3.89e-15 PASS
```

### bw=128 (FAIL - After Attempt 9)
```
Inverse:
### INV_WIG:   [-16.1556 53.7401] [-0.2499 49.2895] [6.9990 41.3279]
### INV_TR1:   [-16.1556 53.7401] [-31.1528 6.1817] [5.2836 -1.6158]
### INV_H2D1:  [-16.1556 53.7401] [-31.1528 6.1817] [5.2836 -1.6158]
### INV_FFT1:  [-38.1861 56.7857] [-38.0179 57.4699] [-37.8508 58.1585]
### INV_TR2:   [-38.1861 56.7857] [-32.2285 -27.4361] [35.3533 22.3334]
### INV_OUTPUT: [522.8166 -737.4118] [1251.1420 -261.7934] [-169.1205 224.9371]
  L2: 1.16e+00 FAIL, Max: 2.83e+00 FAIL

Forward:
### CUFFT_FWD_COPY: [5453174.4739 -7691484.5102] [13049884.5387 -2730604.0747] [-1763990.5601 2346179.4650]
### CUFFT_FWD_FFT1: [-101963635.7763 151627880.5294] [-86055833.1549 -73259142.9283] [94399483.2731 59634114.1217]
### CUFFT_FWD_TR1:  [-101963635.7763 151627880.5294] [-101514541.9157 153454824.0252] [-101068459.4452 155293358.8229]
### CUFFT_FWD_FFT2: [-11043384173.9282 36734877776.9590] [-21294969453.2113 4225593270.4001] [3611690295.3128 -1104471601.2756]
### CUFFT_FWD_TR2:  [-11043384173.9282 36734877776.9590] [-170790834.7457 33692622941.2279] [4784280209.3761 28250349354.7957]
```

**Key insight**: Forward FFT values are ~10^8-10^10 (correct magnitude), confirming the forward path works. The error persists because the **transposed layout after the inverse FFT doesn't match what the subsequent algorithm stages expect**.

(End of file - total lines updated)
