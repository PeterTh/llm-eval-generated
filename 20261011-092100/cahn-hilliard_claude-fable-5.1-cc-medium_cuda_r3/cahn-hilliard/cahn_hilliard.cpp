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
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Stencil tile configuration: each block owns a BX x BY tile of the XY plane and
// sweeps along Z through a chunk of slices, keeping the z-1 / z / z+1 column values
// in registers and the current XY plane (with a 1-cell halo) in shared memory.
constexpr int BX = 32;
constexpr int BY = 8;
constexpr int SX = BX + 2;
constexpr int SY = BY + 2;

struct StencilParams {
    size_t nx, ny, nz;
    int zchunk;
    double inv_dx2, inv_dy2, inv_dz2;
    // Chemical potential coefficients: the bulk term
    //   4.5 * ((c + 1) e_AA + (c - 1) e_BB - 2 c e_AB) + 3 c + c^3
    // is linear in c apart from c^3, so it is folded into  lin_c * c + lin_0 + c^3
    // (coefficients computed once on the host).
    double gamma, lin_c, lin_0;
    // Update coefficient (dt * D)
    double dtD;
};

// Functor: chemical potential from concentration value and its Laplacian
struct ChemicalPotentialOp {
    __device__ __forceinline__ double operator()(const double* /*aux*/, const size_t /*idx*/,
                                                 const double cv, const double lap,
                                                 const StencilParams& p) const {
        return (p.lin_c * cv + p.lin_0) + cv * cv * cv - p.gamma * lap;
    }
};

// Functor: Cahn-Hilliard explicit update from old concentration and Laplacian of mu
struct UpdateOp {
    __device__ __forceinline__ double operator()(const double* cold, const size_t idx,
                                                 const double /*muv*/, const double lap,
                                                 const StencilParams& p) const {
        return cold[idx] + p.dtD * lap;
    }
};

// Generic 7-point Laplacian stencil with clamped (zero-flux) boundaries.
// out[idx] = op(aux, idx, in[idx], laplacian(in)[idx])
template <typename Op>
__global__ void __launch_bounds__(BX * BY)
laplacianStencilKernel(const double* __restrict__ in, const double* __restrict__ aux,
                       double* __restrict__ out, const StencilParams p, const Op op) {
    // Three rotating XY planes (with 1-cell halo): plane z is read while plane z+1
    // is being filled; a third buffer keeps slow threads from being overwritten.
    __shared__ double tile[3][SY][SX];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    const int nx = static_cast<int>(p.nx);
    const int ny = static_cast<int>(p.ny);
    const int nz = static_cast<int>(p.nz);

    const int x0 = blockIdx.x * BX;
    const int y0 = blockIdx.y * BY;
    const int x = x0 + tx;
    const int y = y0 + ty;
    const bool active = (x < nx) && (y < ny);

    const int z_begin = blockIdx.z * p.zchunk;
    const int z_end = min(z_begin + p.zchunk, nz);
    if (z_begin >= nz) return;

    const size_t plane = p.nx * p.ny;
    const size_t xy = active ? idx3(static_cast<size_t>(x), static_cast<size_t>(y), 0, p.nx, p.ny) : 0;

    // Cooperative load of the (BX+2) x (BY+2) XY tile of plane gz into buffer b,
    // with index clamping so that halo cells beyond the domain replicate the edge.
    auto loadPlane = [&](const int b, const int gz) {
        const size_t zoff = static_cast<size_t>(gz) * plane;
        for (int i = tid; i < SX * SY; i += BX * BY) {
            const int sy = i / SX;
            const int sx = i - sy * SX;
            int gx = x0 + sx - 1;
            int gy = y0 + sy - 1;
            gx = gx < 0 ? 0 : (gx > nx - 1 ? nx - 1 : gx);
            gy = gy < 0 ? 0 : (gy > ny - 1 ? ny - 1 : gy);
            tile[b][sy][sx] = in[zoff + static_cast<size_t>(gy) * p.nx + static_cast<size_t>(gx)];
        }
    };

    // Prime the window: plane z_begin lives in buffer 0, z_begin-1 (clamped) in registers.
    loadPlane(0, z_begin);
    double below = 0.0;
    if (active) {
        const int zn = (z_begin > 0) ? z_begin - 1 : 0;
        below = in[xy + static_cast<size_t>(zn) * plane];
    }

    int b = 0;
    for (int z = z_begin; z < z_end; ++z) {
        const int bn = (b + 1 == 3) ? 0 : b + 1;
        const int zp = (z < nz - 1) ? z + 1 : z;
        loadPlane(bn, zp);
        __syncthreads();

        const double c = tile[b][ty + 1][tx + 1];
        const double above = tile[bn][ty + 1][tx + 1];
        if (active) {
            const double cxx = (tile[b][ty + 1][tx + 2] + tile[b][ty + 1][tx] - 2.0 * c) * p.inv_dx2;
            const double cyy = (tile[b][ty + 2][tx + 1] + tile[b][ty][tx + 1] - 2.0 * c) * p.inv_dy2;
            const double czz = (above + below - 2.0 * c) * p.inv_dz2;
            const double lap = cxx + cyy + czz;
            const size_t idx = xy + static_cast<size_t>(z) * plane;
            out[idx] = op(aux, idx, c, lap, p);
        }

        below = c;
        b = bn;
    }
}

// Initialize concentration field (pseudo-random values in [-1, 1])
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linear_id = i; linear_id < vol; linear_id += stride) {
        const double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
    }
}

void initializeConcentration(double* d_c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const int threads = 256;
    const size_t blocksNeeded = (vol + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<size_t>(blocksNeeded, 65535));
    initializeConcentrationKernel<<<blocks, threads>>>(d_c, vol);
    CUDA_CHECK(cudaGetLastError());
}

// Choose the Z chunk per block so that enough blocks exist to fill the GPU.
int chooseZChunk(const size_t nx, const size_t ny, const size_t nz, const int numSMs) {
    const size_t xyBlocks = ((nx + BX - 1) / BX) * ((ny + BY - 1) / BY);
    const size_t targetBlocks = static_cast<size_t>(numSMs) * 8;
    int zchunk = static_cast<int>(nz);
    while (zchunk > 4 && xyBlocks * ((nz + zchunk - 1) / zchunk) < targetBlocks) {
        zchunk /= 2;
    }
    return std::max(zchunk, 1);
}

dim3 stencilGrid(const StencilParams& p) {
    return dim3(static_cast<unsigned>((p.nx + BX - 1) / BX),
                static_cast<unsigned>((p.ny + BY - 1) / BY),
                static_cast<unsigned>((p.nz + p.zchunk - 1) / p.zchunk));
}

// Compute chemical potential
void computeChemicalPotential(const double* d_c, double* d_mu, const StencilParams& p) {
    laplacianStencilKernel<<<stencilGrid(p), dim3(BX, BY)>>>(d_c, nullptr, d_mu, p, ChemicalPotentialOp{});
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(double* d_cnew, const double* d_cold, const double* d_mu, const StencilParams& p) {
    laplacianStencilKernel<<<stencilGrid(p), dim3(BX, BY)>>>(d_mu, d_cold, d_cnew, p, UpdateOp{});
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

    // Select GPU and query SM count for launch sizing
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    StencilParams p{};
    p.nx = nx;
    p.ny = ny;
    p.nz = nz;
    p.zchunk = chooseZChunk(nx, ny, nz, numSMs);
    p.inv_dx2 = 1.0 / (dx * dx);
    p.inv_dy2 = 1.0 / (dy * dy);
    p.inv_dz2 = 1.0 / (dz * dz);
    p.gamma = gamma;
    p.lin_c = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    p.lin_0 = 4.5 * (e_AA - e_BB);
    p.dtD = dt * D;
    
    // Allocate arrays (device)
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, nx, ny, nz);

    // Force the stencil kernels to be loaded now so that module loading does not
    // land inside the timed region.
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, laplacianStencilKernel<ChemicalPotentialOp>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, laplacianStencilKernel<UpdateOp>));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(d_cold, d_mu, p);
        
        // Update concentration
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, p);
        
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

    // Copy final concentration field back to the host
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
