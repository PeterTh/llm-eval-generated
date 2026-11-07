// Cahn-Hilliard phase separation benchmark — CUDA implementation.
// This file is compiled as CUDA (see CMakeLists.txt) and runs the whole
// simulation on the GPU; only the final field is copied back to the host.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err_));                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Thread block tile in the XY plane; each thread walks a slab of Z planes.
// Splitting Z into slabs multiplies the number of resident threads so that
// FP64 latency on consumer GPUs is hidden by occupancy.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 4;
constexpr int Z_SLAB = 8;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Stencil parameters shared by both kernels.
struct GridParams {
    int nx, ny, nz;
    double dx2, dy2, dz2;      // dx*dx etc.
    double rdx2, rdy2, rdz2;   // 1/(dx*dx) etc., used only when multiplication is exact
};

// True if dividing by v is bit-identical to multiplying by 1/v for every
// finite double, i.e. v is a power of two (so 1/v is exactly representable
// and the product is a pure exponent shift). Holds for the default dx=1.
static bool reciprocalIsExact(const double v) {
    int e;
    const double m = std::frexp(v, &e);
    return std::isfinite(v) && v != 0.0 && m == 0.5 && std::isfinite(1.0 / v) && (1.0 / v) != 0.0;
}

// Laplacian with clamped boundary conditions, evaluated from values already
// loaded into registers (centre, +-x, +-y, +-z). Mirrors the CPU formulation:
// (c[+] + c[-] - 2c) / (d*d) per axis, summed.
// EXACT_RECIP selects multiplication by the precomputed reciprocal, which the
// host only enables when it is bit-identical to the division (see
// reciprocalIsExact); otherwise the division is performed as in the original.
template <bool EXACT_RECIP>
__device__ __forceinline__ double laplacian(const double c, const double cxp, const double cxn,
                                            const double cyp, const double cyn, const double czp,
                                            const double czn, const GridParams& g) {
    const double sx = cxp + cxn - 2.0 * c;
    const double sy = cyp + cyn - 2.0 * c;
    const double sz = czp + czn - 2.0 * c;
    double cxx, cyy, czz;
    if constexpr (EXACT_RECIP) {
        cxx = sx * g.rdx2;
        cyy = sy * g.rdy2;
        czz = sz * g.rdz2;
    } else {
        cxx = sx / g.dx2;
        cyy = sy / g.dy2;
        czz = sz / g.dz2;
    }
    return cxx + cyy + czz;
}

// Generic stencil kernel: every thread owns one (x,y) column within a Z slab
// and sweeps it, keeping the z-1 / z / z+1 planes in registers so each cell's
// centre value is loaded from global memory only once per slab. Neighbour
// loads in x/y hit L1/L2 because adjacent threads touch adjacent addresses.
// op(centre_value, laplacian, idx) produces the output for the cell.
template <bool EXACT_RECIP, typename Op>
__global__ void __launch_bounds__(TILE_X * TILE_Y)
stencil_kernel(const double* __restrict__ in, double* __restrict__ out, const GridParams g, const Op op) {
    const int x = blockIdx.x * TILE_X + threadIdx.x;
    const int y = blockIdx.y * TILE_Y + threadIdx.y;
    if (x >= g.nx || y >= g.ny) return;
    const int z0 = blockIdx.z * Z_SLAB;
    const int z1 = min(z0 + Z_SLAB, g.nz);

    const size_t nx = g.nx, ny = g.ny;
    const size_t plane = nx * ny;
    const int xp = (x < g.nx - 1) ? x + 1 : x;
    const int yp = (y < g.ny - 1) ? y + 1 : y;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;

    // Column base offsets (z = 0)
    const size_t o_c = idx3(x, y, 0, nx, ny);
    const size_t o_xp = idx3(xp, y, 0, nx, ny);
    const size_t o_xn = idx3(xn, y, 0, nx, ny);
    const size_t o_yp = idx3(x, yp, 0, nx, ny);
    const size_t o_yn = idx3(x, yn, 0, nx, ny);

    // Sliding window in z. Clamped boundary: at z=0, z-1 maps to z itself.
    const int zn0 = (z0 > 0) ? z0 - 1 : 0;
    double c_prev = __ldg(in + o_c + static_cast<size_t>(zn0) * plane);  // value at z-1 (clamped)
    double c_cur = __ldg(in + o_c + static_cast<size_t>(z0) * plane);    // value at z
    double c_next;                                                       // value at z+1 (clamped)

    for (int z = z0; z < z1; ++z) {
        const size_t zoff = static_cast<size_t>(z) * plane;
        c_next = (z < g.nz - 1) ? __ldg(in + o_c + zoff + plane) : c_cur;

        const double cxp = __ldg(in + o_xp + zoff);
        const double cxn = __ldg(in + o_xn + zoff);
        const double cyp = __ldg(in + o_yp + zoff);
        const double cyn = __ldg(in + o_yn + zoff);

        const double lap = laplacian<EXACT_RECIP>(c_cur, cxp, cxn, cyp, cyn, c_next, c_prev, g);
        const size_t idx = o_c + zoff;
        out[idx] = op(c_cur, lap, idx);

        c_prev = c_cur;
        c_cur = c_next;
    }
}

// mu = 4.5*((c+1)eAA + (c-1)eBB - 2c eAB) + 3c + c^3 - gamma * lap(c)
struct ChemicalPotentialOp {
    double gamma, e_AA, e_BB, e_AB;
    __device__ __forceinline__ double operator()(const double cv, const double lap, size_t) const {
        return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * lap;
    }
};

// cnew = cold + dt*D*lap(mu)
struct UpdateOp {
    const double* __restrict__ cold;
    double dtD;
    __device__ __forceinline__ double operator()(const double, const double lap, const size_t idx) const {
        return __ldg(cold + idx) + dtD * lap;
    }
};

template <typename Op>
static void launchStencil(const double* in, double* out, const GridParams& g, const Op& op,
                          const bool exactRecip, const dim3 grid, const dim3 block) {
    if (exactRecip)
        stencil_kernel<true><<<grid, block>>>(in, out, g, op);
    else
        stencil_kernel<false><<<grid, block>>>(in, out, g, op);
}

// Compute chemical potential (GPU)
void computeChemicalPotential(const double* d_c, double* d_mu, const GridParams& g,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const bool exactRecip, const dim3 grid, const dim3 block) {
    launchStencil(d_c, d_mu, g, ChemicalPotentialOp{gamma, e_AA, e_BB, e_AB}, exactRecip, grid, block);
}

// Cahn-Hilliard update step (GPU)
void cahnHilliardUpdate(double* d_cnew, const double* d_cold, const double* d_mu, const GridParams& g,
                        const double D, const double dt, const bool exactRecip, const dim3 grid, const dim3 block) {
    launchStencil(d_mu, d_cnew, g, UpdateOp{d_cold, dt * D}, exactRecip, grid, block);
}

// Initialize concentration field (GPU). Same pseudo-random formula as the
// original CPU code, evaluated in 64-bit integer arithmetic.
__global__ void initializeKernel(double* __restrict__ c, const size_t vol) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= vol) return;
    const size_t linear_id = idx;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx] = -1.0 + 2.0 * pseudo;
}

void initializeConcentration(double* d_c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((vol + threads - 1) / threads);
    initializeKernel<<<blocks, threads>>>(d_c, vol);
    CUDA_CHECK(cudaGetLastError());
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
    
    GridParams g;
    g.nx = static_cast<int>(nx);
    g.ny = static_cast<int>(ny);
    g.nz = static_cast<int>(nz);
    g.dx2 = dx * dx; g.dy2 = dy * dy; g.dz2 = dz * dz;
    g.rdx2 = 1.0 / g.dx2; g.rdy2 = 1.0 / g.dy2; g.rdz2 = 1.0 / g.dz2;
    // Replace the per-axis divisions by multiplications only when the result
    // is guaranteed bit-identical (grid spacings whose squares are powers of two).
    const bool exactRecip = reciprocalIsExact(g.dx2) && reciprocalIsExact(g.dy2) && reciprocalIsExact(g.dz2);

    const dim3 block(TILE_X, TILE_Y, 1);
    const dim3 grid(static_cast<unsigned>((nx + TILE_X - 1) / TILE_X),
                    static_cast<unsigned>((ny + TILE_Y - 1) / TILE_Y),
                    static_cast<unsigned>((nz + Z_SLAB - 1) / Z_SLAB));

    // Allocate arrays (device) and host result buffer
    CUDA_CHECK(cudaSetDevice(0));
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    std::vector<double> cold(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(d_cold, d_mu, g, gamma, e_AA, e_BB, e_AB, exactRecip, grid, block);
        
        // Update concentration
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, g, D, dt, exactRecip, grid, block);
        
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
    
    // Copy final field back to host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
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
