#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

static inline void checkKernelLaunch() {
    CUDA_CHECK(cudaGetLastError());
}

// 3D index calculation (host-side utility)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int BX = 8;
constexpr int BY = 8;
constexpr int BZ = 8;
constexpr int SX = BX + 2;
constexpr int SY = BY + 2;
constexpr int SXY = SX * SY;
constexpr int TILE_SIZE = SXY * (BZ + 2);

__device__ __forceinline__ int clamp_i32(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

__global__ void initConcentrationKernel(double* __restrict__ c, size_t n, unsigned long long vol) {
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) return;

    const unsigned long long linear_id = (unsigned long long)i;
    const unsigned long long tmp = ((linear_id + 1ULL) * 1299709ULL) % vol;
    const double pseudo = (double)tmp / (double)vol;
    c[i] = -1.0 + 2.0 * pseudo;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                       size_t nx, size_t ny, size_t nz,
                                       double invdx2, double invdy2, double invdz2,
                                       double gamma, double e_AA, double e_BB, double e_AB) {
    __shared__ double tile[TILE_SIZE];

    const int tid = (int)threadIdx.x + BX * ((int)threadIdx.y + BY * (int)threadIdx.z);
    const int nthreads = BX * BY * BZ;

    const int ox = (int)blockIdx.x * BX;
    const int oy = (int)blockIdx.y * BY;
    const int oz = (int)blockIdx.z * BZ;

    const int nx_i = (int)nx;
    const int ny_i = (int)ny;
    const int nz_i = (int)nz;

    const size_t plane = nx * ny;

    for (int i = tid; i < TILE_SIZE; i += nthreads) {
        const int sx = i % SX;
        const int sy = (i / SX) % SY;
        const int sz = i / SXY;

        int gx = ox + (sx - 1);
        int gy = oy + (sy - 1);
        int gz = oz + (sz - 1);

        gx = clamp_i32(gx, 0, nx_i - 1);
        gy = clamp_i32(gy, 0, ny_i - 1);
        gz = clamp_i32(gz, 0, nz_i - 1);

        const size_t g = (size_t)gz * plane + (size_t)gy * nx + (size_t)gx;
        tile[i] = c[g];
    }
    __syncthreads();

    const size_t x = (size_t)ox + (size_t)threadIdx.x;
    const size_t y = (size_t)oy + (size_t)threadIdx.y;
    const size_t z = (size_t)oz + (size_t)threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t g = z * plane + y * nx + x;

    const int s = ((int)threadIdx.z + 1) * SXY + ((int)threadIdx.y + 1) * SX + ((int)threadIdx.x + 1);
    const double cv = tile[s];

    const double cxx = (tile[s + 1] + tile[s - 1] - 2.0 * cv) * invdx2;
    const double cyy = (tile[s + SX] + tile[s - SX] - 2.0 * cv) * invdy2;
    const double czz = (tile[s + SXY] + tile[s - SXY] - 2.0 * cv) * invdz2;
    const double lap = cxx + cyy + czz;

    const double term = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv
                      - gamma * lap;

    mu[g] = term;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                            const double* __restrict__ mu,
                            size_t nx, size_t ny, size_t nz,
                            double invdx2, double invdy2, double invdz2,
                            double dtD) {
    __shared__ double tile[TILE_SIZE];

    const int tid = (int)threadIdx.x + BX * ((int)threadIdx.y + BY * (int)threadIdx.z);
    const int nthreads = BX * BY * BZ;

    const int ox = (int)blockIdx.x * BX;
    const int oy = (int)blockIdx.y * BY;
    const int oz = (int)blockIdx.z * BZ;

    const int nx_i = (int)nx;
    const int ny_i = (int)ny;
    const int nz_i = (int)nz;

    const size_t plane = nx * ny;

    for (int i = tid; i < TILE_SIZE; i += nthreads) {
        const int sx = i % SX;
        const int sy = (i / SX) % SY;
        const int sz = i / SXY;

        int gx = ox + (sx - 1);
        int gy = oy + (sy - 1);
        int gz = oz + (sz - 1);

        gx = clamp_i32(gx, 0, nx_i - 1);
        gy = clamp_i32(gy, 0, ny_i - 1);
        gz = clamp_i32(gz, 0, nz_i - 1);

        const size_t g = (size_t)gz * plane + (size_t)gy * nx + (size_t)gx;
        tile[i] = mu[g];
    }
    __syncthreads();

    const size_t x = (size_t)ox + (size_t)threadIdx.x;
    const size_t y = (size_t)oy + (size_t)threadIdx.y;
    const size_t z = (size_t)oz + (size_t)threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t g = z * plane + y * nx + x;

    const int s = ((int)threadIdx.z + 1) * SXY + ((int)threadIdx.y + 1) * SX + ((int)threadIdx.x + 1);
    const double mv = tile[s];

    const double mxx = (tile[s + 1] + tile[s - 1] - 2.0 * mv) * invdx2;
    const double myy = (tile[s + SX] + tile[s - SX] - 2.0 * mv) * invdy2;
    const double mzz = (tile[s + SXY] + tile[s - SXY] - 2.0 * mv) * invdz2;
    const double lap = mxx + myy + mzz;

    cnew[g] = cold[g] + dtD * lap;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        printf("CUDA device not found; this benchmark requires a GPU.\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);

    double* cold_d = nullptr;
    double* cnew_d = nullptr;
    double* mu_d = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&cold_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&cnew_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&mu_d, bytes));

    printf("Initializing concentration field...\n");
    {
        const int threads = 256;
        const int blocks = (int)((gridSize + (size_t)threads - 1) / (size_t)threads);
        initConcentrationKernel<<<blocks, threads>>>(cold_d, gridSize, (unsigned long long)gridSize);
        checkKernelLaunch();
    }

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    const dim3 block(BX, BY, BZ);
    const dim3 grid((unsigned int)((nx + BX - 1) / BX),
                    (unsigned int)((ny + BY - 1) / BY),
                    (unsigned int)((nz + BZ - 1) / BZ));

    printf("Running Cahn-Hilliard simulation...\n");

    cudaEvent_t startEv, endEv;
    CUDA_CHECK(cudaEventCreate(&startEv));
    CUDA_CHECK(cudaEventCreate(&endEv));

    CUDA_CHECK(cudaEventRecord(startEv));
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, block>>>(cold_d, mu_d, nx, ny, nz, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        updateKernel<<<grid, block>>>(cnew_d, cold_d, mu_d, nx, ny, nz, invdx2, invdy2, invdz2, dtD);
        checkKernelLaunch();
        std::swap(cold_d, cnew_d);
    }
    CUDA_CHECK(cudaEventRecord(endEv));
    CUDA_CHECK(cudaEventSynchronize(endEv));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEv, endEv));

    const long elapsedMsLong = (long)llround((double)elapsedMs);
    printf("Computation time: %ld ms\n", elapsedMsLong);

    const double cellUpdates = (double)gridSize * (double)iterations;
    const double mcups = cellUpdates / ((double)elapsedMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold(gridSize);
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(cold.data(), cold_d, bytes, cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        print_results(cold, "Concentration");
    }

    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(cold, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        CUDA_CHECK(cudaFree(mu_d));
        CUDA_CHECK(cudaFree(cnew_d));
        CUDA_CHECK(cudaFree(cold_d));
        CUDA_CHECK(cudaEventDestroy(startEv));
        CUDA_CHECK(cudaEventDestroy(endEv));
        return valid ? 0 : 1;
    }

    CUDA_CHECK(cudaFree(mu_d));
    CUDA_CHECK(cudaFree(cnew_d));
    CUDA_CHECK(cudaFree(cold_d));
    CUDA_CHECK(cudaEventDestroy(startEv));
    CUDA_CHECK(cudaEventDestroy(endEv));

    return 0;
}
