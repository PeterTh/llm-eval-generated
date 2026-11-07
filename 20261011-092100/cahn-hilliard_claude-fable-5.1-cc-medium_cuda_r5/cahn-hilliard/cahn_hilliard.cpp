// Cahn-Hilliard phase separation benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt). The two stencil passes
// of every time step (chemical potential, then concentration update) run as
// GPU kernels over the full 3D grid. Each thread owns one (x, y) column and
// marches along z, keeping the three z-neighbours of the stencil in registers
// so that every grid value is loaded from global memory only once per pass
// (the x/y neighbours are served from L1 thanks to the coalesced block layout).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                             \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    } while (0)

// Thread block layout: BX threads along x (coalesced), BY along y; every
// thread processes a run of consecutive z planes whose length (zchunk) is
// chosen at runtime so that even small grids produce enough blocks to fill
// the GPU.
constexpr int BX = 32;
constexpr int BY = 4;
constexpr int MAX_ZCHUNK = 16;

// Chemical potential kernel: mu = f'(c) - gamma * lap(c)
__global__ void __launch_bounds__(BX * BY)
chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                        const int nx, const int ny, const int nz,
                        const double inv_dx2, const double inv_dy2, const double inv_dz2,
                        const double gamma, const double e_AA, const double e_BB, const double e_AB,
                        const int zchunk) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int z0 = blockIdx.z * zchunk;
    const int z1 = min(z0 + zchunk, nz);

    const size_t plane = static_cast<size_t>(nx) * ny;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const size_t base = static_cast<size_t>(y) * nx + x;        // offset within a plane
    const ptrdiff_t oxp = xp - x, oxn = xn - x;
    const ptrdiff_t oyp = static_cast<ptrdiff_t>(yp - y) * nx, oyn = static_cast<ptrdiff_t>(yn - y) * nx;

    // Register pipeline along z: znv = c[z-1], cv = c[z], zpv = c[z+1]
    const int zn0 = (z0 > 0) ? z0 - 1 : 0;
    double znv = c[zn0 * plane + base];
    double cv  = c[static_cast<size_t>(z0) * plane + base];

    for (int z = z0; z < z1; ++z) {
        const int zp = (z < nz - 1) ? z + 1 : z;
        const double zpv = c[static_cast<size_t>(zp) * plane + base];
        const size_t idx = static_cast<size_t>(z) * plane + base;

        const double cxx = (c[idx + oxp] + c[idx + oxn] - 2.0 * cv) * inv_dx2;
        const double cyy = (c[idx + oyp] + c[idx + oyn] - 2.0 * cv) * inv_dy2;
        const double czz = (zpv + znv - 2.0 * cv) * inv_dz2;
        const double lap = cxx + cyy + czz;

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma * lap;

        znv = cv;
        cv = zpv;
    }
}

// Concentration update kernel: cnew = cold + dt * D * lap(mu)
__global__ void __launch_bounds__(BX * BY)
cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                         const double* __restrict__ mu,
                         const int nx, const int ny, const int nz,
                         const double D, const double dt,
                         const double inv_dx2, const double inv_dy2, const double inv_dz2,
                         const int zchunk) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int z0 = blockIdx.z * zchunk;
    const int z1 = min(z0 + zchunk, nz);

    const size_t plane = static_cast<size_t>(nx) * ny;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const size_t base = static_cast<size_t>(y) * nx + x;
    const ptrdiff_t oxp = xp - x, oxn = xn - x;
    const ptrdiff_t oyp = static_cast<ptrdiff_t>(yp - y) * nx, oyn = static_cast<ptrdiff_t>(yn - y) * nx;

    const double dtD = dt * D;

    const int zn0 = (z0 > 0) ? z0 - 1 : 0;
    double znv = mu[zn0 * plane + base];
    double mv  = mu[static_cast<size_t>(z0) * plane + base];

    for (int z = z0; z < z1; ++z) {
        const int zp = (z < nz - 1) ? z + 1 : z;
        const double zpv = mu[static_cast<size_t>(zp) * plane + base];
        const size_t idx = static_cast<size_t>(z) * plane + base;

        const double cxx = (mu[idx + oxp] + mu[idx + oxn] - 2.0 * mv) * inv_dx2;
        const double cyy = (mu[idx + oyp] + mu[idx + oyn] - 2.0 * mv) * inv_dy2;
        const double czz = (zpv + znv - 2.0 * mv) * inv_dz2;
        const double lap = cxx + cyy + czz;

        cnew[idx] = cold[idx] + dtD * lap;

        znv = mv;
        mv = zpv;
    }
}

// Initialize concentration field with the reference pseudo-random pattern
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vol) return;
    // Generate pseudo-random value in [-1, 1] (64-bit unsigned arithmetic as in the reference)
    const double pseudo = static_cast<double>(((i + 1) * 1299709ull) % vol) / static_cast<double>(vol);
    c[i] = -1.0 + 2.0 * pseudo;
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
    
    // Allocate device arrays
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    {
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((gridSize + threads - 1) / threads);
        initializeConcentrationKernel<<<blocks, threads>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Pick the z-run length per thread: long runs amortise the register
    // pipeline on big grids, while small grids need shorter runs to produce
    // enough blocks (aim for several resident blocks per SM).
    int numSMs = 1;
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    }
    const size_t blocksXY = ((nx + BX - 1) / BX) * ((ny + BY - 1) / BY);
    const size_t targetBlocks = static_cast<size_t>(numSMs) * 32;
    const int zchunk = static_cast<int>(std::max<size_t>(1, std::min<size_t>(MAX_ZCHUNK, (blocksXY * nz) / targetBlocks)));
    
    const dim3 block(BX, BY, 1);
    const dim3 grid(static_cast<unsigned>((nx + BX - 1) / BX),
                    static_cast<unsigned>((ny + BY - 1) / BY),
                    static_cast<unsigned>((nz + zchunk - 1) / zchunk));
    const int inx = static_cast<int>(nx), iny = static_cast<int>(ny), inz = static_cast<int>(nz);
    // FP64 division is very expensive on consumer GPUs; the stencil divides by the
    // squared grid spacings, so multiply by their reciprocals instead. For the unit
    // (power-of-two) spacings used here the reciprocal is exact and the result is
    // bitwise identical to the division.
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // One time step = chemical potential pass followed by the concentration
    // update, after which the old/new buffers are swapped. Two consecutive
    // steps return the buffers to their original roles, so a pair of steps is
    // captured once into a CUDA graph and replayed; this removes most of the
    // per-kernel launch latency that dominates small grids.
    auto launchStep = [&](cudaStream_t s, double* cur, double* next) {
        // Compute chemical potential
        chemicalPotentialKernel<<<grid, block, 0, s>>>(cur, d_mu, inx, iny, inz, inv_dx2, inv_dy2, inv_dz2,
                                                       gamma, e_AA, e_BB, e_AB, zchunk);
        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block, 0, s>>>(next, cur, d_mu, inx, iny, inz,
                                                        D, dt, inv_dx2, inv_dy2, inv_dz2, zchunk);
    };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    const int pairs = iterations / 2;
    if (pairs > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launchStep(stream, d_cold, d_cnew);
        launchStep(stream, d_cnew, d_cold);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));
    }

    for (int t = 0; t < pairs; ++t) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    if (iterations % 2 != 0) {
        launchStep(stream, d_cold, d_cnew);
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy the final concentration field back to the host
    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    if (graphExec) CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
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
