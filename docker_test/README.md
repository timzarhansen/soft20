# SO(3) cuFFT Backend Test

Quick test setup for validating the cuFFT GPU backend implementation on borrowed GPU hardware.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
cd docker_test
docker compose up --build
```

### Option 2: Manual Docker

```bash
# Build image
docker build -t soft20-cufft-test ..

# Run test (requires GPU)
docker run --gpus all --rm soft20-cufft-test
```

## Requirements

- **NVIDIA GPU**: Pascal architecture or newer (GTX 10xx, RTX 20xx/30xx/40xx, A100, H100, etc.)
- **NVIDIA Drivers**: Installed on host system
- **Docker**: With NVIDIA Container Toolkit
- **Disk Space**: ~5GB for image and build artifacts

## What It Tests

1. **Sanity Check (bw=8)**: Quick verification that GPU is accessible and cuFFT works
2. **Full Comparison (bw=128)**: Comprehensive FFTW vs cuFFT comparison:
   - Forward SO(3) transform accuracy
   - Inverse SO(3) transform accuracy
   - Timing comparison (typical speedup: 4-6x on RTX 3090)

## Expected Output

```
===========================================
SO(3) Transform - cuFFT vs FFTW Comparison
===========================================

GPU: NVIDIA GeForce RTX 3090
  Compute capability: 8.6
  Global memory: 24.0 GB
  Multiprocessors: 82

Sanity Check (bw=8):
  Reconstruction L2 error: 1.2e-8 ✓

Sanity check: PASS ✓

Full Comparison Test (bw=128):
  Bandwidth: 128 (N=256, 16.8M samples)
  Coefficients: 2097088

Forward Transform:
  FFTW:  2341.234 ms
  cuFFT: 412.567 ms (5.7x faster)
  Coefficient L2 error: 3.2e-6 ✓
  Coefficient max error: 1.8e-5 ✓

Inverse Transform:
  FFTW:  1987.654 ms
  cuFFT: 356.789 ms (5.6x faster)
  Signal L2 error: 2.9e-6 ✓
  Signal max error: 1.5e-5 ✓

OVERALL: PASS ✓
===========================================
```

## Exit Codes

- **0**: All tests passed
- **1**: Test failed (no GPU, build error, or numerical mismatch)

## Troubleshooting

### No GPU Found

```bash
# Check NVIDIA Docker is working
docker run --gpus all nvidia/cuda:13.1.0-base-ubuntu22.04 nvidia-smi

# If this fails, install NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Build Fails

```bash
# Rebuild without cache
docker-compose build --no-cache

# Or with verbose output
docker build --no-cache -t soft20-cufft-test ..
```

### Numerical Mismatch

If tests fail with numerical errors:
- Check GPU is not overheating/throttling
- Verify CUDA drivers are up to date
- Try reducing bandwidth (edit `BW_FULL` in test_cufft_compare.cu)

## Customization

### Change Test Bandwidth

Edit `examples/test_cufft_compare.cu`:
```cpp
#define BW_FULL 128  // Change to 64 for faster test, 256 for more thorough
```

### Change Error Tolerance

```cpp
#define TOLERANCE_L2 1e-5    // L2 error tolerance
#define TOLERANCE_MAX 1e-4   // Max error tolerance
```

### Use Different CUDA Version

Edit `docker_test/Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04  # Change version
```

## Notes

- **CUDA 13.1**: Container uses CUDA 13.1, compatible with most modern GPUs
- **No Host CUDA Required**: Container includes full CUDA toolkit, host only needs drivers
- **Fast Rebuilds**: With cache, rebuilds take ~1 minute
- **CI/CD Ready**: Exit codes support automated testing pipelines

## Architecture Support

The build targets these CUDA compute capabilities:
- **60**: Pascal (GTX 10xx)
- **70**: Volta (V100)
- **75**: Turing (RTX 20xx)
- **80**: Ampere (RTX 30xx, A100)
- **86**: Ampere mobile (RTX 30xx mobile)
- **87**: Hopper (H100)

For Jetson or other architectures, update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt.
