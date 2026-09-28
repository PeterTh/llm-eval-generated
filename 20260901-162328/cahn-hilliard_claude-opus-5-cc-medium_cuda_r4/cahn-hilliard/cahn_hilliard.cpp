#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                     \
        if (err_ != cudaSuccess) {                                                                           \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,               \
                   cudaGetErrorString(err_));                                                                \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

// Tile dimensions of the in-plane (x,y) thread block. The z dimension is
// traversed sequentially by each thread, keeping a sliding window of three
// z-values in registers so that every grid point is loaded from global memory
// only once per kernel (plus the tile halo in x/y).
static constexpr int TILE_X = 32;
static constexpr int TILE_Y = 4;

// Target number of resident thread blocks per SM, used to pick the number of
// z-slabs the grid is split into.
static constexpr int BLOCKS_PER_SM = 64;

// Simulation parameters passed to the kernels by value.
struct Params {
    int nx, ny, nz;
    double dx, dy, dz;
    double gamma, e_AA, e_BB, e_AB;
    double D, dt;
    double rdx2, rdy2, rdz2; // 1/(dx*dx), ... (only used when exactly representable)
};

enum StencilMode { MODE_CHEMICAL_POTENTIAL = 0, MODE_UPDATE = 1 };

// Fused 7-point stencil kernel.
//   MODE_CHEMICAL_POTENTIAL: out = mu(c)        (in = c)
//   MODE_UPDATE:             out = cold + dt*D*laplacian(mu)  (in = mu)
// Boundary conditions are clamped, matching the scalar reference, and the
// arithmetic is performed in the same order as in the reference code.
// RECIP replaces the (expensive) FP64 divisions by the grid spacing with
// multiplications by the reciprocal; it is only enabled when that transformation
// is exact (see main()).
template <int MODE, bool RECIP>
__global__ void stencilKernel(const double* __restrict__ in, const double* __restrict__ cold,
                              double* __restrict__ out, const Params p, const int zchunk) {
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    const int x = blockIdx.x * TILE_X + threadIdx.x;
    const int y = blockIdx.y * TILE_Y + threadIdx.y;
    const bool active = (x < p.nx) && (y < p.ny);

    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;

    // Shared-memory offsets of the in-plane neighbours, already clamped at the
    // domain boundary (a clamped neighbour resolves to the centre element).
    const int sxp = (x < p.nx - 1) ? tx + 1 : tx;
    const int sxn = (x > 0) ? tx - 1 : tx;
    const int syp = (y < p.ny - 1) ? ty + 1 : ty;
    const int syn = (y > 0) ? ty - 1 : ty;

    const size_t plane = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const int z0 = blockIdx.z * zchunk;
    const int z1 = min(z0 + zchunk, p.nz);

    size_t base = static_cast<size_t>(z0) * plane + static_cast<size_t>(y) * p.nx + x;

    // Sliding window: vm = value at z-1, vc = value at z, vp = value at z+1.
    double vc = active ? in[base] : 0.0;
    double vm = (z0 > 0 && active) ? in[base - plane] : vc;

    for (int z = z0; z < z1; ++z) {
        __syncthreads(); // previous iteration finished reading the tile

        if (active) {
            tile[ty][tx] = vc;
            if (threadIdx.x == 0 && x > 0) tile[ty][0] = in[base - 1];
            if (threadIdx.x == TILE_X - 1 && x < p.nx - 1) tile[ty][TILE_X + 1] = in[base + 1];
            if (threadIdx.y == 0 && y > 0) tile[0][tx] = in[base - p.nx];
            if (threadIdx.y == TILE_Y - 1 && y < p.ny - 1) tile[TILE_Y + 1][tx] = in[base + p.nx];
        }

        // Load the leading z-plane value while the tile stores are in flight.
        const double vp = (active && z < p.nz - 1) ? in[base + plane] : vc;

        __syncthreads();

        if (active) {
            const double sxx = tile[ty][sxp] + tile[ty][sxn] - 2.0 * vc;
            const double syy = tile[syp][tx] + tile[syn][tx] - 2.0 * vc;
            const double szz = vp + vm - 2.0 * vc;

            const double cxx = RECIP ? sxx * p.rdx2 : sxx / (p.dx * p.dx);
            const double cyy = RECIP ? syy * p.rdy2 : syy / (p.dy * p.dy);
            const double czz = RECIP ? szz * p.rdz2 : szz / (p.dz * p.dz);
            const double lap = cxx + cyy + czz;

            if (MODE == MODE_CHEMICAL_POTENTIAL) {
                out[base] = 4.5 * ((vc + 1.0) * p.e_AA + (vc - 1.0) * p.e_BB - 2.0 * vc * p.e_AB) //
                            + 3.0 * vc + vc * vc * vc                                            //
                            - p.gamma * lap;
            } else {
                out[base] = cold[base] + p.dt * p.D * lap;
            }
        }

        vm = vc;
        vc = vp;
        base += plane;
    }
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < vol; i += stride) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[i] = -1.0 + 2.0 * pseudo;
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

    const Params params{static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                        dx,                   dy,                   dz,
                        gamma,                e_AA,                 e_BB,
                        e_AB,                 D,                    dt,
                        1.0 / (dx * dx),      1.0 / (dy * dy),      1.0 / (dz * dz)};

    // Dividing by h*h and multiplying by 1/(h*h) give identical results for
    // every operand if h*h is a power of two (the reciprocal is then exact and
    // the scaling introduces no rounding). Only then is the reciprocal used.
    const auto isPowerOfTwo = [](const double v) {
        int exp = 0;
        return v > 0.0 && std::isfinite(v) && std::frexp(v, &exp) == 0.5;
    };
    const bool useReciprocal = isPowerOfTwo(dx * dx) && isPowerOfTwo(dy * dy) && isPowerOfTwo(dz * dz);

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
        const int block = 256;
        const int grid = static_cast<int>(std::min<size_t>((gridSize + block - 1) / block, 65535));
        initializeConcentrationKernel<<<grid, block>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Launch configuration: enough blocks in z to fill the device, while each
    // block still sweeps several z-planes to amortize the halo loads.
    const dim3 blockDim(TILE_X, TILE_Y, 1);
    const unsigned int gridX = static_cast<unsigned int>((nx + TILE_X - 1) / TILE_X);
    const unsigned int gridY = static_cast<unsigned int>((ny + TILE_Y - 1) / TILE_Y);

    int device = 0;
    int numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    const size_t targetBlocks = static_cast<size_t>(numSMs) * BLOCKS_PER_SM;
    size_t gridZ = (targetBlocks + static_cast<size_t>(gridX) * gridY - 1) / (static_cast<size_t>(gridX) * gridY);
    gridZ = std::max<size_t>(1, std::min<size_t>(gridZ, nz));
    const int zchunk = static_cast<int>((nz + gridZ - 1) / gridZ);
    gridZ = (nz + zchunk - 1) / zchunk;
    const dim3 gridDim(gridX, gridY, static_cast<unsigned int>(gridZ));

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (useReciprocal) {
            // Compute chemical potential
            stencilKernel<MODE_CHEMICAL_POTENTIAL, true><<<gridDim, blockDim>>>(d_cold, nullptr, d_mu, params, zchunk);
            // Update concentration
            stencilKernel<MODE_UPDATE, true><<<gridDim, blockDim>>>(d_mu, d_cold, d_cnew, params, zchunk);
        } else {
            stencilKernel<MODE_CHEMICAL_POTENTIAL, false><<<gridDim, blockDim>>>(d_cold, nullptr, d_mu, params, zchunk);
            stencilKernel<MODE_UPDATE, false><<<gridDim, blockDim>>>(d_mu, d_cold, d_cnew, params, zchunk);
        }

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_us);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance (from the microsecond-resolution timing, as the GPU
    // run time can be well below one millisecond for small grids)
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (elapsed_us.count() / 1e6) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

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
