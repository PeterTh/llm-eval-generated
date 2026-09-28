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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA implementation
//
// The two sweeps of the original code (chemical potential, then concentration
// update) are fused into a single kernel.  Both sweeps are 7-point Laplacian
// stencils with clamped boundaries, so the fused operator has a radius of two.
//
// Each block owns a BX x BY column of the grid and marches along z, keeping a
// sliding window of three z-planes of the concentration field and three
// z-planes of the chemical potential in shared memory.  This way every mu
// value is evaluated only once (apart from the one-cell x/y halo) while the
// intermediate mu field never leaves the chip: global memory traffic per cell
// update is one read of c plus one write of cnew.
//
// All floating point expressions keep the operand order of the scalar code.
// The grid spacings enter only as the exact reciprocals 1/(dx*dx) etc.
// ---------------------------------------------------------------------------

// 32 x 16 threads per block: 32 along x keeps the global loads coalesced, and
// the resulting 34x18 mu tile evaluates only 1.20 mu values per cell update.
static constexpr int BX = 32;
static constexpr int BY = 16;
static constexpr int CW = BX + 4;   // shared c tile width  (two-cell halo)
static constexpr int CH = BY + 4;   // shared c tile height
static constexpr int MW = BX + 2;   // shared mu tile width (one-cell halo)
static constexpr int MH = BY + 2;   // shared mu tile height

struct Params {
    int nx, ny, nz;
    double ix2, iy2, iz2;   // 1/(dx*dx), 1/(dy*dy), 1/(dz*dz)
    double gamma;
    double e_AA, e_BB, e_AB;
    double dtD;             // dt * D
};

__device__ __forceinline__ int clampi(const int v, const int lo, const int hi) {
    return min(max(v, lo), hi);
}

// Chemical potential from the 7-point neighbourhood of one cell
__device__ __forceinline__ double chemPotential(const double cv, const double cxp, const double cxn,
                                                const double cyp, const double cyn, const double czp,
                                                const double czn, const Params p) {
    const double cxx = (cxp + cxn - 2.0 * cv) * p.ix2;
    const double cyy = (cyp + cyn - 2.0 * cv) * p.iy2;
    const double czz = (czp + czn - 2.0 * cv) * p.iz2;

    return 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
           + 3.0 * cv + cv * cv * cv
           - p.gamma * (cxx + cyy + czz);
}

__global__ __launch_bounds__(BX * BY) void cahnHilliardStepKernel(const double* __restrict__ cold,
                                                                  double* __restrict__ cnew,
                                                                  const Params p, const int zchunk) {
    constexpr int CP = CW * CH;         // cells per shared c plane
    constexpr int MP = MW * MH;         // cells per shared mu plane
    __shared__ double sc[3 * CP];
    __shared__ double smu[3 * MP];

    const int nx = p.nx, ny = p.ny, nz = p.nz;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tid = ty * BX + tx;
    constexpr int nthreads = BX * BY;
    constexpr int LD_ITEMS = (CP + nthreads - 1) / nthreads;
    constexpr int MU_ITEMS = (MP + nthreads - 1) / nthreads;

    const int x0 = static_cast<int>(blockIdx.x) * BX;
    const int y0 = static_cast<int>(blockIdx.y) * BY;
    const int zs = static_cast<int>(blockIdx.z) * zchunk;
    if (zs >= nz) return;
    const int ze = min(zs + zchunk, nz);

    const int gx = x0 + tx;
    const int gy = y0 + ty;
    const bool active = (gx < nx) && (gy < ny);
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);

    // All in-plane addressing is independent of z, so it is resolved once here
    // and kept in registers; the z-loop below only does loads, flops and stores.
    int ldDst[LD_ITEMS];    // destination offset inside a shared c plane
    int ldSrc[LD_ITEMS];    // source offset inside a global z-plane
#pragma unroll
    for (int k = 0; k < LD_ITEMS; ++k) {
        const int i = tid + k * nthreads;
        ldDst[k] = (i < CP) ? i : -1;
        const int lx = (i < CP) ? (i % CW) : 0;
        const int ly = (i < CP) ? (i / CW) : 0;
        ldSrc[k] = clampi(y0 - 2 + ly, 0, ny - 1) * nx + clampi(x0 - 2 + lx, 0, nx - 1);
    }

    int muDst[MU_ITEMS];    // destination offset inside a shared mu plane
    int muC[MU_ITEMS];      // offset of the evaluation point inside a shared c plane
#pragma unroll
    for (int k = 0; k < MU_ITEMS; ++k) {
        const int i = tid + k * nthreads;
        muDst[k] = (i < MP) ? i : -1;
        const int lx = (i < MP) ? (i % MW) : 0;
        const int ly = (i < MP) ? (i / MW) : 0;
        // Halo entries evaluate mu of the clamped in-domain cell, which is what
        // the clamped boundary treatment of the second Laplacian requires.
        const int qx = clampi(x0 - 1 + lx, 0, nx - 1);
        const int qy = clampi(y0 - 1 + ly, 0, ny - 1);
        // Because the c tile is filled with clamped global coordinates, the
        // neighbours of an evaluation point are simply the adjacent tile cells:
        // tile cell (qx-x0+2)-1 holds c[clamp(qx-1)], and likewise for +1/y.
        muC[k] = (qy - y0 + 2) * CW + (qx - x0 + 2);
    }

    // Load one clamped z-plane of c into the ring slot belonging to it.
    // Slot invariant: sc slot s holds the most recently loaded plane q with q % 3 == s.
    auto loadC = [&](const int zq) {
        const int q = clampi(zq, 0, nz - 1);
        const size_t base = static_cast<size_t>(q) * plane;
        double* dst = sc + (q % 3) * CP;
#pragma unroll
        for (int k = 0; k < LD_ITEMS; ++k) {
            if (ldDst[k] >= 0) dst[ldDst[k]] = cold[base + ldSrc[k]];
        }
    };

    // Evaluate mu on plane q (plus a one-cell x/y halo) from the resident c planes.
    auto computeMu = [&](const int q) {
        const double* c0 = sc + (q % 3) * CP;
        const double* cn = sc + (clampi(q - 1, 0, nz - 1) % 3) * CP;
        const double* cp = sc + (clampi(q + 1, 0, nz - 1) % 3) * CP;
        double* dst = smu + (q % 3) * MP;
#pragma unroll
        for (int k = 0; k < MU_ITEMS; ++k) {
            if (muDst[k] >= 0) {
                const int o = muC[k];
                dst[muDst[k]] = chemPotential(c0[o], c0[o + 1], c0[o - 1], c0[o + CW], c0[o - CW],
                                              cp[o], cn[o], p);
            }
        }
    };

    // Offsets of this thread's own cell in the shared tiles and in the output.
    const int muCtr = (ty + 1) * MW + (tx + 1);
    const int cCtr = (ty + 2) * CW + (tx + 2);
    const size_t outOff = static_cast<size_t>(gy) * nx + gx;

    // Prologue: make mu planes max(zs-1,0) and zs resident.
    loadC(zs - 2);
    loadC(zs - 1);
    loadC(zs);
    __syncthreads();
    if (zs > 0) {
        computeMu(zs - 1);
        __syncthreads();
        loadC(zs + 1);          // reuses the slot of plane zs-2
        __syncthreads();
    } else {
        loadC(zs + 1);
        __syncthreads();
    }
    computeMu(zs);

    for (int z = zs; z < ze; ++z) {
        __syncthreads();
        loadC(z + 2);           // reuses the slot of plane z-1
        __syncthreads();
        if (z + 1 < nz) computeMu(z + 1);
        __syncthreads();

        if (active) {
            const double* m0p = smu + (z % 3) * MP;
            const double* mnp = smu + (clampi(z - 1, 0, nz - 1) % 3) * MP;
            const double* mpp = smu + (clampi(z + 1, 0, nz - 1) % 3) * MP;

            const double m0 = m0p[muCtr];
            const double mxx = (m0p[muCtr + 1] + m0p[muCtr - 1] - 2.0 * m0) * p.ix2;
            const double myy = (m0p[muCtr + MW] + m0p[muCtr - MW] - 2.0 * m0) * p.iy2;
            const double mzz = (mpp[muCtr] + mnp[muCtr] - 2.0 * m0) * p.iz2;

            cnew[static_cast<size_t>(z) * plane + outOff] =
                sc[(z % 3) * CP + cCtr] + p.dtD * (mxx + myy + mzz);
        }
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

    // Set up the GPU
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));

    Params p;
    p.nx = static_cast<int>(nx);
    p.ny = static_cast<int>(ny);
    p.nz = static_cast<int>(nz);
    p.ix2 = 1.0 / (dx * dx);
    p.iy2 = 1.0 / (dy * dy);
    p.iz2 = 1.0 / (dz * dz);
    p.gamma = gamma;
    p.e_AA = e_AA;
    p.e_BB = e_BB;
    p.e_AB = e_AB;
    p.dtD = dt * D;

    // Blocks tile x/y and march along z.  z is cut into slabs so that even
    // small grids produce enough blocks to fill the device, but slabs are kept
    // long enough that the sliding-window prologue (one extra mu plane and a
    // few extra c planes per slab) stays a small fraction of the work.
    const int nbx = static_cast<int>((nx + BX - 1) / BX);
    const int nby = static_cast<int>((ny + BY - 1) / BY);
    const long long blocksXY = static_cast<long long>(nbx) * nby;
    const long long targetBlocks = static_cast<long long>(prop.multiProcessorCount) * 25;
    long long slabs = (targetBlocks + blocksXY - 1) / blocksXY;
    const long long slabsCap = static_cast<long long>(nz) / 8;
    if (slabs > slabsCap) slabs = slabsCap;
    if (slabs < 1) slabs = 1;
    const int zchunk = static_cast<int>((static_cast<long long>(nz) + slabs - 1) / slabs);
    const int nbz = static_cast<int>((nz + zchunk - 1) / zchunk);

    const dim3 blockDim(BX, BY, 1);
    const dim3 gridDim(nbx, nby, nbz);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    // Warm-up launch: forces module load / code JIT out of the timed region.
    // It only writes the scratch buffer, which the first real step overwrites.
    cahnHilliardStepKernel<<<gridDim, blockDim>>>(d_cold, d_cnew, p, zchunk);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        cahnHilliardStepKernel<<<gridDim, blockDim>>>(d_cold, d_cnew, p, zchunk);
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

    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
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
