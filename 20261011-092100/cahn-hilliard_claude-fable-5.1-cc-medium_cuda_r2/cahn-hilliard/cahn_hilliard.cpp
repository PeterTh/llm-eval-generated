#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                                      \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

// Tile geometry of the fused kernel. Each thread block owns a TX x TY column
// of the grid and marches through a chunk of planes along z. A rolling buffer of
// three chemical-potential planes (with a one-cell halo) lives in shared memory.
constexpr int TX = 32;
constexpr int TY = 16;
constexpr int SX = TX + 2;  // shared-memory plane width (halo of 1)
constexpr int SY = TY + 2;  // shared-memory plane height (halo of 1)

// Physical parameters that are constant for the whole simulation.
struct Params {
    int nx, ny, nz;
    double inv_dx2, inv_dy2, inv_dz2;  // 1/(dx*dx), ... (dx = dy = dz = 1, so exact)
    double gamma;
    double e_AA, e_BB, e_AB;
    double dtD;  // dt * D
    int zchunk;  // number of z planes handled per thread block
};

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Chemical potential at (x, y, z), using the same clamped-boundary Laplacian
// and the exact same floating-point expression order as the reference code.
__device__ __forceinline__ double chemicalPotentialAt(const double* __restrict__ c, const Params& p,
                                                      const int x, const int y, const int z) {
    const int xp = (x < p.nx - 1) ? x + 1 : x;
    const int yp = (y < p.ny - 1) ? y + 1 : y;
    const int zp = (z < p.nz - 1) ? z + 1 : z;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (z > 0) ? z - 1 : 0;

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t base = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * p.nx;

    const double cv = __ldg(c + base + x);
    const double cxx = (__ldg(c + base + xp) + __ldg(c + base + xn) - 2.0 * cv) * p.inv_dx2;
    const double cyy = (__ldg(c + static_cast<size_t>(z) * plane + static_cast<size_t>(yp) * p.nx + x) +
                        __ldg(c + static_cast<size_t>(z) * plane + static_cast<size_t>(yn) * p.nx + x) - 2.0 * cv) *
                       p.inv_dy2;
    const double czz = (__ldg(c + static_cast<size_t>(zp) * plane + static_cast<size_t>(y) * p.nx + x) +
                        __ldg(c + static_cast<size_t>(zn) * plane + static_cast<size_t>(y) * p.nx + x) - 2.0 * cv) *
                       p.inv_dz2;
    const double lap = cxx + cyy + czz;

    return 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB) + 3.0 * cv + cv * cv * cv -
           p.gamma * lap;
}

// Fill one shared-memory plane with mu(x, y, z) for the block tile plus a
// one-cell halo. Coordinates outside the domain are clamped, which is exactly
// the "clamped index" rule of the reference Laplacian (mu[-1] == mu[0]).
__device__ __forceinline__ void loadMuPlane(double* __restrict__ s_mu, const double* __restrict__ c,
                                            const Params& p, const int x0, const int y0, const int z) {
    const int tid = threadIdx.y * TX + threadIdx.x;
    for (int i = tid; i < SX * SY; i += TX * TY) {
        const int lx = i % SX;
        const int ly = i / SX;
        int gx = x0 - 1 + lx;
        int gy = y0 - 1 + ly;
        gx = (gx < 0) ? 0 : ((gx > p.nx - 1) ? p.nx - 1 : gx);
        gy = (gy < 0) ? 0 : ((gy > p.ny - 1) ? p.ny - 1 : gy);
        s_mu[i] = chemicalPotentialAt(c, p, gx, gy, z);
    }
}

// Fused Cahn-Hilliard step: mu = f(c) - gamma * lap(c); cnew = c + dt*D*lap(mu).
// mu is never written to global memory; it is produced on the fly into a
// rolling 3-plane shared-memory buffer and consumed immediately.
__global__ void __launch_bounds__(TX * TY) cahnHilliardFusedKernel(double* __restrict__ cnew,
                                                                   const double* __restrict__ cold,
                                                                   const Params p) {
    __shared__ double s_mu[3][SX * SY];

    const int x0 = blockIdx.x * TX;
    const int y0 = blockIdx.y * TY;
    const int z0 = blockIdx.z * p.zchunk;
    const int z1 = min(z0 + p.zchunk, p.nz);  // exclusive

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int gx = x0 + tx;
    const int gy = y0 + ty;
    const bool active = (gx < p.nx) && (gy < p.ny);

    // Prime the rolling buffer with planes z0-1 (clamped) and z0.
    int prev = 0, cur = 1, next = 2;
    loadMuPlane(s_mu[prev], cold, p, x0, y0, (z0 > 0) ? z0 - 1 : 0);
    loadMuPlane(s_mu[cur], cold, p, x0, y0, z0);
    __syncthreads();

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const int sc = (ty + 1) * SX + (tx + 1);  // this thread's cell in the shared plane

    for (int z = z0; z < z1; ++z) {
        // Produce mu at plane z+1 (clamped at the top boundary).
        loadMuPlane(s_mu[next], cold, p, x0, y0, (z < p.nz - 1) ? z + 1 : z);
        __syncthreads();

        if (active) {
            const double* mc = s_mu[cur];
            const double mv = mc[sc];
            const double mxx = (mc[sc + 1] + mc[sc - 1] - 2.0 * mv) * p.inv_dx2;
            const double myy = (mc[sc + SX] + mc[sc - SX] - 2.0 * mv) * p.inv_dy2;
            const double mzz = (s_mu[next][sc] + s_mu[prev][sc] - 2.0 * mv) * p.inv_dz2;
            const double lap = mxx + myy + mzz;

            const size_t idx = static_cast<size_t>(z) * plane + static_cast<size_t>(gy) * p.nx + gx;
            cnew[idx] = __ldg(cold + idx) + p.dtD * lap;
        }
        __syncthreads();  // everyone is done with 'prev' before it is reused as 'next'

        const int t = prev;
        prev = cur;
        cur = next;
        next = t;
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
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

    // Parse command line arguments
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

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    size_t gridSize = nx * ny * nz;

    // Allocate arrays
    std::vector<double> cold(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Device setup
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    Params p{};
    p.nx = static_cast<int>(nx);
    p.ny = static_cast<int>(ny);
    p.nz = static_cast<int>(nz);
    p.inv_dx2 = 1.0 / (dx * dx);
    p.inv_dy2 = 1.0 / (dy * dy);
    p.inv_dz2 = 1.0 / (dz * dz);
    p.gamma = gamma;
    p.e_AA = e_AA;
    p.e_BB = e_BB;
    p.e_AB = e_AB;
    p.dtD = dt * D;
    // Choose the z extent per block so that the grid exposes enough blocks to
    // fill the GPU; longer chunks amortize the two extra halo planes per chunk.
    {
        const long long blocksXY = static_cast<long long>((nx + TX - 1) / TX) * static_cast<long long>((ny + TY - 1) / TY);
        const long long targetBlocks = 2LL * prop.multiProcessorCount;
        const long long zc = (static_cast<long long>(nz) * blocksXY) / targetBlocks;
        p.zchunk = static_cast<int>(std::clamp<long long>(zc, 8, 32));
    }

    const size_t bytes = gridSize * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TX, TY, 1);
    const dim3 grid(static_cast<unsigned>((nx + TX - 1) / TX), static_cast<unsigned>((ny + TY - 1) / TY),
                    static_cast<unsigned>((nz + p.zchunk - 1) / p.zchunk));

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential and update concentration in one fused pass
        cahnHilliardFusedKernel<<<grid, block>>>(d_cnew, d_cold, p);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the final field back to the host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
