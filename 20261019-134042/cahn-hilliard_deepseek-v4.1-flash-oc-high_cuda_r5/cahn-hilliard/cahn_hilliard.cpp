#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

namespace {

constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 8;
constexpr int BLOCK_THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;

constexpr int C_TILE_X = BLOCK_X + 4;
constexpr int C_TILE_Y = BLOCK_Y + 4;
constexpr int C_TILE_Z = BLOCK_Z + 4;
constexpr int C_TILE_SIZE = C_TILE_X * C_TILE_Y * C_TILE_Z;

constexpr int MU_TILE_X = BLOCK_X + 2;
constexpr int MU_TILE_Y = BLOCK_Y + 2;
constexpr int MU_TILE_Z = BLOCK_Z + 2;
constexpr int MU_TILE_SIZE = MU_TILE_X * MU_TILE_Y * MU_TILE_Z;

__device__ __forceinline__ int cSharedIndex(const int x, const int y, const int z) noexcept {
    return (z * C_TILE_Y + y) * C_TILE_X + x;
}

__device__ __forceinline__ int muSharedIndex(const int x, const int y, const int z) noexcept {
    return (z * MU_TILE_Y + y) * MU_TILE_X + x;
}

void checkCuda(cudaError_t err, const char* message) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; id < vol; id += stride) {
        const size_t linear_id = id;
        const double pseudo = ((((linear_id + 1) * static_cast<size_t>(1299709)) % vol) / static_cast<double>(vol));
        c[id] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS)
cahnHilliardStepKernel(const double* __restrict__ cold,
                       double* __restrict__ cnew,
                       const size_t nx, const size_t ny, const size_t nz,
                       const double invDx2, const double invDy2, const double invDz2,
                       const double gamma, const double e_AA, const double e_BB, const double e_AB,
                       const double dtD) {
    __shared__ double sc[C_TILE_SIZE];
    __shared__ double smu[MU_TILE_SIZE];

    const int tid = static_cast<int>(threadIdx.x) +
                    static_cast<int>(threadIdx.y) * BLOCK_X +
                    static_cast<int>(threadIdx.z) * BLOCK_X * BLOCK_Y;

    const long long baseX = static_cast<long long>(blockIdx.x) * BLOCK_X;
    const long long baseY = static_cast<long long>(blockIdx.y) * BLOCK_Y;
    const long long baseZ = static_cast<long long>(blockIdx.z) * BLOCK_Z;

    for (int t = tid; t < C_TILE_SIZE; t += BLOCK_THREADS) {
        const int sx = t % C_TILE_X;
        const int rem = t / C_TILE_X;
        const int sy = rem % C_TILE_Y;
        const int sz = rem / C_TILE_Y;

        long long gx = baseX + sx - 2;
        long long gy = baseY + sy - 2;
        long long gz = baseZ + sz - 2;
        if (gx < 0) gx = 0;
        else if (gx >= static_cast<long long>(nx)) gx = static_cast<long long>(nx) - 1;
        if (gy < 0) gy = 0;
        else if (gy >= static_cast<long long>(ny)) gy = static_cast<long long>(ny) - 1;
        if (gz < 0) gz = 0;
        else if (gz >= static_cast<long long>(nz)) gz = static_cast<long long>(nz) - 1;

        sc[t] = cold[idx3(static_cast<size_t>(gx), static_cast<size_t>(gy), static_cast<size_t>(gz), nx, ny)];
    }

    __syncthreads();

    for (int t = tid; t < MU_TILE_SIZE; t += BLOCK_THREADS) {
        const int sx = t % MU_TILE_X;
        const int rem = t / MU_TILE_X;
        const int sy = rem % MU_TILE_Y;
        const int sz = rem / MU_TILE_Y;

        const long long gx = baseX + sx - 1;
        const long long gy = baseY + sy - 1;
        const long long gz = baseZ + sz - 1;

        if (gx >= 0 && gx < static_cast<long long>(nx) &&
            gy >= 0 && gy < static_cast<long long>(ny) &&
            gz >= 0 && gz < static_cast<long long>(nz)) {
            const int cx = sx + 1;
            const int cy = sy + 1;
            const int cz = sz + 1;

            const double cv = sc[cSharedIndex(cx, cy, cz)];
            const double cxx = (sc[cSharedIndex(cx + 1, cy, cz)] +
                                sc[cSharedIndex(cx - 1, cy, cz)] -
                                2.0 * cv) * invDx2;
            const double cyy = (sc[cSharedIndex(cx, cy + 1, cz)] +
                                sc[cSharedIndex(cx, cy - 1, cz)] -
                                2.0 * cv) * invDy2;
            const double czz = (sc[cSharedIndex(cx, cy, cz + 1)] +
                                sc[cSharedIndex(cx, cy, cz - 1)] -
                                2.0 * cv) * invDz2;

            const double t1 = fma(cv + 1.0, e_AA, (cv - 1.0) * e_BB);
            const double t2 = fma(-(cv + cv), e_AB, t1);
            const double t3 = fma(4.5, t2, 3.0 * cv);
            const double t4 = fma(cv * cv, cv, t3);
            smu[t] = fma(-gamma, cxx + cyy + czz, t4);
        } else {
            smu[t] = 0.0;
        }
    }

    __syncthreads();

    const int lx = static_cast<int>(threadIdx.x);
    const int ly = static_cast<int>(threadIdx.y);
    const int lz = static_cast<int>(threadIdx.z);

    const long long gx = baseX + lx;
    const long long gy = baseY + ly;
    const long long gz = baseZ + lz;

    if (gx < static_cast<long long>(nx) &&
        gy < static_cast<long long>(ny) &&
        gz < static_cast<long long>(nz)) {
        const long long xm = (gx > 0) ? gx - 1 : 0;
        const long long xp = (gx + 1 < static_cast<long long>(nx)) ? gx + 1 : static_cast<long long>(nx) - 1;
        const long long ym = (gy > 0) ? gy - 1 : 0;
        const long long yp = (gy + 1 < static_cast<long long>(ny)) ? gy + 1 : static_cast<long long>(ny) - 1;
        const long long zm = (gz > 0) ? gz - 1 : 0;
        const long long zp = (gz + 1 < static_cast<long long>(nz)) ? gz + 1 : static_cast<long long>(nz) - 1;

        const int ixm = static_cast<int>(xm - baseX) + 1;
        const int ixp = static_cast<int>(xp - baseX) + 1;
        const int iym = static_cast<int>(ym - baseY) + 1;
        const int iyp = static_cast<int>(yp - baseY) + 1;
        const int izm = static_cast<int>(zm - baseZ) + 1;
        const int izp = static_cast<int>(zp - baseZ) + 1;

        const int cx = lx + 1;
        const int cy = ly + 1;
        const int cz = lz + 1;

        const double muv = smu[muSharedIndex(cx, cy, cz)];
        const double lap = (smu[muSharedIndex(ixp, cy, cz)] +
                            smu[muSharedIndex(ixm, cy, cz)] -
                            2.0 * muv) * invDx2
                         + (smu[muSharedIndex(cx, iyp, cz)] +
                            smu[muSharedIndex(cx, iym, cz)] -
                            2.0 * muv) * invDy2
                         + (smu[muSharedIndex(cx, cy, izp)] +
                            smu[muSharedIndex(cx, cy, izm)] -
                            2.0 * muv) * invDz2;

        const size_t idx = idx3(static_cast<size_t>(gx), static_cast<size_t>(gy), static_cast<size_t>(gz), nx, ny);
        const double cv = sc[cSharedIndex(lx + 2, ly + 2, lz + 2)];
        cnew[idx] = fma(dtD, lap, cv);
    }
}

}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
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

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    checkCuda(cudaMalloc(&d_cold, bytes), "cudaMalloc cold failed");
    checkCuda(cudaMalloc(&d_cnew, bytes), "cudaMalloc cnew failed");

    printf("Initializing concentration field...\n");
    const int initThreads = 256;
    const int initBlocks = static_cast<int>(std::min<size_t>((gridSize + initThreads - 1) / initThreads, 65535));
    initializeConcentrationKernel<<<initBlocks, initThreads>>>(d_cold, nx, ny, nz);
    checkCuda(cudaGetLastError(), "initializeConcentrationKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "initializeConcentrationKernel execution failed");

    printf("Running Cahn-Hilliard simulation...\n");

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));

    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        cahnHilliardStepKernel<<<grid, block>>>(d_cold, d_cnew, nx, ny, nz,
                                                invDx2, invDy2, invDz2,
                                                gamma, e_AA, e_BB, e_AB, dtD);
        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "cahnHilliardStepKernel execution failed");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> cold(gridSize);
        checkCuda(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy result failed");

        if (printResults) {
            print_results(cold, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
                checkCuda(cudaFree(d_cold), "cudaFree cold failed");
                checkCuda(cudaFree(d_cnew), "cudaFree cnew failed");
                return 0;
            } else {
                printf("Validation: FAILED\n");
                checkCuda(cudaFree(d_cold), "cudaFree cold failed");
                checkCuda(cudaFree(d_cnew), "cudaFree cnew failed");
                return 1;
            }
        }
    }

    checkCuda(cudaFree(d_cold), "cudaFree cold failed");
    checkCuda(cudaFree(d_cnew), "cudaFree cnew failed");

    return 0;
}
