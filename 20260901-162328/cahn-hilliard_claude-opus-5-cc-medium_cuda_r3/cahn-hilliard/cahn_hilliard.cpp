#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// Thread block covers BX cells in x (one full warp per row) and BY rows in y; each
// thread then sweeps a column of zTile cells in z, keeping the z-neighbours in registers.
static constexpr int BX = 32;
static constexpr int BY = 4;

// Grid spacing enters the Laplacian as a division by d*d. Dividing is exactly
// equivalent to multiplying by the reciprocal iff d*d is a power of two, which is
// checked on the host; otherwise the (slower) division is used to keep results identical.
struct Spacing {
    double kx, ky, kz;  // either d*d or its reciprocal
    int reciprocal;
};

__device__ __forceinline__ double scaleTerm(const double v, const double k, const int reciprocal) {
    return reciprocal ? v * k : v / k;
}

// Number of z-levels a thread evaluates per loop iteration. The z-column values are
// held in registers, so the independent levels give the (long-latency) FP64 pipeline
// several instruction streams to interleave.
static constexpr int ZU = 2;

// 7-point Laplacian with clamped boundary conditions, for the cell at (xc, yc) whose
// z-column values f(z-1), f(z), f(z+1) are passed in as cm, c0, cp.
// UNIT_SPACING drops the (then exactly neutral) division by d*d == 1.
template <bool UNIT_SPACING, typename I>
__device__ __forceinline__ double laplacianAt(const double* __restrict__ f, const I idx,
                                              const I nx, const I ny, const I xc, const I yc,
                                              const double cm, const double c0, const double cp,
                                              const Spacing sp) {
    // x-neighbours come from the neighbouring lanes; only the warp edges load.
    const int lane = threadIdx.x;
    double xm = __shfl_up_sync(0xffffffffu, c0, 1);
    double xp = __shfl_down_sync(0xffffffffu, c0, 1);
    if (lane == 0) xm = (xc > 0) ? f[idx - 1] : c0;
    if (xc == nx - 1) xp = c0;
    else if (lane == BX - 1) xp = f[idx + 1];

    const double ym = (yc > 0) ? f[idx - nx] : c0;
    const double yp = (yc < ny - 1) ? f[idx + nx] : c0;

    const double twoc = 2.0 * c0;
    if constexpr (UNIT_SPACING) {
        return (xp + xm - twoc) + (yp + ym - twoc) + (cp + cm - twoc);
    } else {
        return scaleTerm(xp + xm - twoc, sp.kx, sp.reciprocal) +
               scaleTerm(yp + ym - twoc, sp.ky, sp.reciprocal) +
               scaleTerm(cp + cm - twoc, sp.kz, sp.reciprocal);
    }
}

// The field is laid out as idx = z * (nx * ny) + y * nx + x.
//
// Sweeps the z-range assigned to this block, computing the Laplacian of f and handing
// (index, centre value, Laplacian) to op for every cell.
template <bool UNIT_SPACING, typename I, typename Op>
__device__ __forceinline__ void laplacianSweep(const double* __restrict__ f,
                                               const I nx, const I ny, const I nz,
                                               const int zTile, const Spacing sp, Op op) {
    const I z0 = static_cast<I>(blockIdx.z) * zTile;
    if (z0 >= nz) return;  // uniform for the whole block: warp-wide shuffles stay safe
    const I z1 = min(z0 + zTile, nz);

    const I x = static_cast<I>(blockIdx.x) * BX + threadIdx.x;
    const I y = static_cast<I>(blockIdx.y) * BY + threadIdx.y;
    const bool active = (x < nx) && (y < ny);
    // Out-of-range threads keep running (they take part in the shuffles) but read
    // clamped locations and never store.
    const I xc = active ? x : nx - 1;
    const I yc = active ? y : ny - 1;

    const I slice = nx * ny;
    const I base = yc * nx + xc;

    for (I z = z0; z < z1; z += ZU) {
        const int levels = static_cast<int>(min(static_cast<I>(ZU), z1 - z));

        // z-column window: w[k] holds f(z - 1 + k), clamped at the domain boundary.
        double w[ZU + 2];
#pragma unroll
        for (int k = 0; k < ZU + 2; ++k) {
            const I zz = min(max(z - 1 + k, static_cast<I>(0)), nz - 1);
            w[k] = f[base + zz * slice];
        }

        if (levels == ZU) {  // common case: fully unrolled, ZU independent streams
#pragma unroll
            for (int k = 0; k < ZU; ++k) {
                const I idx = base + (z + k) * slice;
                const double lap = laplacianAt<UNIT_SPACING>(f, idx, nx, ny, xc, yc, w[k], w[k + 1], w[k + 2], sp);
                if (active) op(idx, w[k + 1], lap);
            }
        } else {  // partial tail tile
#pragma unroll
            for (int k = 0; k < ZU; ++k) {
                if (k < levels) {
                    const I idx = base + (z + k) * slice;
                    const double lap = laplacianAt<UNIT_SPACING>(f, idx, nx, ny, xc, yc, w[k], w[k + 1], w[k + 2], sp);
                    if (active) op(idx, w[k + 1], lap);
                }
            }
        }
    }
}

// Compute chemical potential
template <bool UNIT_SPACING, typename I>
__global__ __launch_bounds__(BX* BY) void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const I nx, const I ny, const I nz, const int zTile, const Spacing sp,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    laplacianSweep<UNIT_SPACING>(c, nx, ny, nz, zTile, sp, [&](const I idx, const double cv, const double lap) {
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                  gamma * lap;
    });
}

// Cahn-Hilliard update step
template <bool UNIT_SPACING, typename I>
__global__ __launch_bounds__(BX* BY) void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    const I nx, const I ny, const I nz, const int zTile, const Spacing sp,
    const double D, const double dt) {
    laplacianSweep<UNIT_SPACING>(mu, nx, ny, nz, zTile, sp, [&](const I idx, double, const double lap) {
        cnew[idx] = cold[idx] + dt * D * lap;
    });
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t linear_id = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (linear_id >= vol) return;
    // Generate pseudo-random value in [-1, 1]
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[linear_id] = -1.0 + 2.0 * pseudo;
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

// A division by k is bit-identical to a multiplication by 1/k exactly when k is a
// power of two (the reciprocal is then exact).
static bool isPowerOfTwo(const double k) {
    int exp = 0;
    return k > 0.0 && std::isfinite(k) && std::frexp(k, &exp) == 0.5;
}

// All parameters the time loop needs to hand to the kernels.
struct SimParams {
    size_t nx, ny, nz;
    int zTile;
    Spacing sp;
    bool unitSpacing;
    double gamma, e_AA, e_BB, e_AB, D, dt;
    int iterations;
};

// Runs the time loop with I as the kernel index type (32-bit whenever the grid allows,
// which keeps address arithmetic off the critical path).
template <bool UNIT_SPACING, typename I>
static void runTimeSteps(double*& d_cold, double*& d_cnew, double* const d_mu,
                         const dim3 grid, const dim3 block, const SimParams& p) {
    const I nx = static_cast<I>(p.nx), ny = static_cast<I>(p.ny), nz = static_cast<I>(p.nz);
    for (int t = 0; t < p.iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<UNIT_SPACING, I><<<grid, block>>>(
            d_cold, d_mu, nx, ny, nz, p.zTile, p.sp, p.gamma, p.e_AA, p.e_BB, p.e_AB);

        // Update concentration
        cahnHilliardUpdateKernel<UNIT_SPACING, I><<<grid, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, p.zTile, p.sp, p.D, p.dt);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
}

static void runTimeSteps(double*& d_cold, double*& d_cnew, double* const d_mu,
                         const dim3 grid, const dim3 block, const SimParams& p) {
    // 32-bit indices suffice as long as every address (including the +-nx halo offsets)
    // stays inside the signed 32-bit range.
    const bool narrow = (p.nx * p.ny * p.nz + 2 * p.nx * p.ny) < 0x7fffffffull;
    if (p.unitSpacing) {
        if (narrow) runTimeSteps<true, int>(d_cold, d_cnew, d_mu, grid, block, p);
        else runTimeSteps<true, long long>(d_cold, d_cnew, d_mu, grid, block, p);
    } else {
        if (narrow) runTimeSteps<false, int>(d_cold, d_cnew, d_mu, grid, block, p);
        else runTimeSteps<false, long long>(d_cold, d_cnew, d_mu, grid, block, p);
    }
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

    SimParams params;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.sp.reciprocal = isPowerOfTwo(dx * dx) && isPowerOfTwo(dy * dy) && isPowerOfTwo(dz * dz);
    params.sp.kx = params.sp.reciprocal ? 1.0 / (dx * dx) : dx * dx;
    params.sp.ky = params.sp.reciprocal ? 1.0 / (dy * dy) : dy * dy;
    params.sp.kz = params.sp.reciprocal ? 1.0 / (dz * dz) : dz * dz;
    params.unitSpacing = (dx == 1.0) && (dy == 1.0) && (dz == 1.0);
    params.gamma = gamma;
    params.e_AA = e_AA;
    params.e_BB = e_BB;
    params.e_AB = e_AB;
    params.D = D;
    params.dt = dt;
    params.iterations = iterations;

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    // Allocate arrays (device resident; only the final field is copied back)
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        const int threads = 256;
        const size_t blocks = (gridSize + threads - 1) / threads;
        initializeConcentrationKernel<<<blocks, threads>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pick the z-tile (a multiple of ZU) so that there is enough work for every SM.
    const size_t xyBlocks = ((nx + BX - 1) / BX) * ((ny + BY - 1) / BY);
    const size_t targetBlocks = static_cast<size_t>(prop.multiProcessorCount) * 8;
    int zTile = 4 * ZU;
    while (zTile > ZU && xyBlocks * ((nz + zTile - 1) / zTile) < targetBlocks) zTile /= 2;
    params.zTile = zTile;

    const dim3 block(BX, BY, 1);
    const dim3 grid(static_cast<unsigned>((nx + BX - 1) / BX), static_cast<unsigned>((ny + BY - 1) / BY),
                    static_cast<unsigned>((nz + zTile - 1) / zTile));

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0) {
        runTimeSteps(d_cold, d_cnew, d_mu, grid, block, params);
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

    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
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
