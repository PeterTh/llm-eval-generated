// Cahn-Hilliard phase separation benchmark - CUDA implementation.
//
// The whole simulation (initialisation, chemical potential, concentration
// update) runs on the GPU.  Both stencil kernels use a 2D x/y tile with a
// shared-memory halo and march along z keeping the z-1 / z / z+1 planes in
// registers, so that every cell is read from global memory once per kernel.
// Arithmetic order matches the original CPU code and FMA contraction is
// disabled in the build so results are bit-for-bit equivalent to the
// reference implementation (up to the host compiler's own contraction choice).

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
        const cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// Tile configuration: a block covers TILE_X x TILE_Y cells in x/y and marches
// over TILE_Z planes in z.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;
constexpr int TILE_Z = 8;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Physical parameters needed by the kernels
struct Params {
    double dx, dy, dz;
    double dt;
    double e_AA, e_BB, e_AB;
    double gamma;
    double D;
    // Reciprocals of dx*dx, dy*dy, dz*dz.  They are only used when the
    // squared spacing is an exact power of two, in which case multiplying by
    // the reciprocal is bit-identical to the division in the reference code.
    double inv_dx2, inv_dy2, inv_dz2;
};

// True if v is a finite, non-zero power of two (so 1/v is exact).
static bool isPowerOfTwo(const double v) {
    if (!(v > 0.0) || std::isinf(v)) return false;
    int e;
    return std::frexp(v, &e) == 0.5;
}

// Chemical potential as in the reference code (without the Laplacian term)
__device__ __forceinline__ double chemicalPotentialLocal(const double cv, const Params& p) {
    return 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
         + 3.0 * cv + cv * cv * cv;
}

// 7-point Laplacian with the same evaluation order as the CPU reference.
// EXACT_RECIP selects multiplication by the (exact) reciprocal of the squared
// spacing instead of the division; the host only enables it when both give
// identical results.
template <bool EXACT_RECIP>
__device__ __forceinline__ double laplacian7(const double xp, const double xn,
                                             const double yp, const double yn,
                                             const double zp, const double zn,
                                             const double c0, const Params& p) {
    const double sx = xp + xn - 2.0 * c0;
    const double sy = yp + yn - 2.0 * c0;
    const double sz = zp + zn - 2.0 * c0;
    double cxx, cyy, czz;
    if constexpr (EXACT_RECIP) {
        cxx = sx * p.inv_dx2;
        cyy = sy * p.inv_dy2;
        czz = sz * p.inv_dz2;
    } else {
        cxx = sx / (p.dx * p.dx);
        cyy = sy / (p.dy * p.dy);
        czz = sz / (p.dz * p.dz);
    }
    return cxx + cyy + czz;
}

// Generic z-marching stencil kernel.
//   MODE == 0 : out = chemicalPotential(in) - gamma * lap(in)         (mu kernel)
//   MODE == 1 : out = aux + dt * D * lap(in)                           (update kernel)
template <int MODE, bool EXACT_RECIP>
__global__ void __launch_bounds__(TILE_X * TILE_Y)
stencilKernel(const double* __restrict__ in, const double* __restrict__ aux,
              double* __restrict__ out, const int nx, const int ny, const int nz,
              const Params p) {
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * TILE_X + tx;
    const int y = blockIdx.y * TILE_Y + ty;
    const int z0 = blockIdx.z * TILE_Z;
    const int z1 = min(z0 + TILE_Z, nz);

    const bool active = (x < nx) && (y < ny);
    // Clamped in-plane coordinates (used for both the cell and its halo loads)
    const int xc = min(x, nx - 1);
    const int yc = min(y, ny - 1);
    const int xn = (xc > 0) ? xc - 1 : 0;
    const int xp = (xc < nx - 1) ? xc + 1 : xc;
    const int yn = (yc > 0) ? yc - 1 : 0;
    const int yp = (yc < ny - 1) ? yc + 1 : yc;

    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t base = static_cast<size_t>(yc) * nx + xc;

    // Register window along z: previous, current, next plane
    const int zm0 = (z0 > 0) ? z0 - 1 : 0;
    const int zp0 = (z0 < nz - 1) ? z0 + 1 : z0;
    double cm = in[base + static_cast<size_t>(zm0) * plane];
    double cc = in[base + static_cast<size_t>(z0) * plane];
    double cp = in[base + static_cast<size_t>(zp0) * plane];

    for (int z = z0; z < z1; ++z) {
        const size_t zoff = static_cast<size_t>(z) * plane;

        // Fill shared tile (interior + halo) for the current plane
        tile[ty + 1][tx + 1] = cc;
        if (tx == 0)          tile[ty + 1][0]          = in[zoff + static_cast<size_t>(yc) * nx + xn];
        if (tx == TILE_X - 1) tile[ty + 1][TILE_X + 1] = in[zoff + static_cast<size_t>(yc) * nx + xp];
        if (ty == 0)          tile[0][tx + 1]          = in[zoff + static_cast<size_t>(yn) * nx + xc];
        if (ty == TILE_Y - 1) tile[TILE_Y + 1][tx + 1] = in[zoff + static_cast<size_t>(yp) * nx + xc];
        __syncthreads();

        if (active) {
            // For cells on the tile edge that are also on the domain edge the
            // halo already holds the clamped (self) value, so this is exact.
            const double vxn = (x > 0)      ? tile[ty + 1][tx]     : cc;
            const double vxp = (x < nx - 1) ? tile[ty + 1][tx + 2] : cc;
            const double vyn = (y > 0)      ? tile[ty][tx + 1]     : cc;
            const double vyp = (y < ny - 1) ? tile[ty + 2][tx + 1] : cc;

            const double lap = laplacian7<EXACT_RECIP>(vxp, vxn, vyp, vyn, cp, cm, cc, p);
            const size_t idx = zoff + base;
            if constexpr (MODE == 0) {
                out[idx] = chemicalPotentialLocal(cc, p) - p.gamma * lap;
            } else {
                out[idx] = aux[idx] + p.dt * p.D * lap;
            }
        }
        __syncthreads();

        // Slide the register window one plane up
        cm = cc;
        cc = cp;
        const int znext = (z + 2 < nz) ? z + 2 : nz - 1;
        cp = in[base + static_cast<size_t>(znext) * plane];
    }
}

// Initialize concentration field (same pseudo-random sequence as the reference)
__global__ void initializeKernel(double* __restrict__ c, const size_t vol) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < vol) {
        const double pseudo = ((((i + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

static dim3 stencilGrid(const size_t nx, const size_t ny, const size_t nz) {
    return dim3(static_cast<unsigned>((nx + TILE_X - 1) / TILE_X),
                static_cast<unsigned>((ny + TILE_Y - 1) / TILE_Y),
                static_cast<unsigned>((nz + TILE_Z - 1) / TILE_Z));
}

// Whether multiplying by the reciprocal spacing is exact for all three axes
static bool exactReciprocalSpacing(const Params& p) {
    return isPowerOfTwo(p.dx * p.dx) && isPowerOfTwo(p.dy * p.dy) && isPowerOfTwo(p.dz * p.dz);
}

// Compute chemical potential (GPU)
void computeChemicalPotential(const double* d_c, double* d_mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const Params& p) {
    const dim3 grid = stencilGrid(nx, ny, nz);
    const dim3 block(TILE_X, TILE_Y);
    const int inx = static_cast<int>(nx), iny = static_cast<int>(ny), inz = static_cast<int>(nz);
    if (exactReciprocalSpacing(p)) {
        stencilKernel<0, true><<<grid, block>>>(d_c, nullptr, d_mu, inx, iny, inz, p);
    } else {
        stencilKernel<0, false><<<grid, block>>>(d_c, nullptr, d_mu, inx, iny, inz, p);
    }
}

// Cahn-Hilliard update step (GPU)
void cahnHilliardUpdate(double* d_cnew, const double* d_cold, const double* d_mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const Params& p) {
    const dim3 grid = stencilGrid(nx, ny, nz);
    const dim3 block(TILE_X, TILE_Y);
    const int inx = static_cast<int>(nx), iny = static_cast<int>(ny), inz = static_cast<int>(nz);
    if (exactReciprocalSpacing(p)) {
        stencilKernel<1, true><<<grid, block>>>(d_mu, d_cold, d_cnew, inx, iny, inz, p);
    } else {
        stencilKernel<1, false><<<grid, block>>>(d_mu, d_cold, d_cnew, inx, iny, inz, p);
    }
}

// Initialize concentration field (GPU)
void initializeConcentration(double* d_c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((vol + block - 1) / block);
    initializeKernel<<<grid, block>>>(d_c, vol);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    Params p;
    p.dx = 1.0;
    p.dy = 1.0;
    p.dz = 1.0;
    p.dt = 0.01;
    p.e_AA = -(2.0 / 9.0);
    p.e_BB = -(2.0 / 9.0);
    p.e_AB = (2.0 / 9.0);
    p.gamma = 0.5;
    p.D = 1.0;
    p.inv_dx2 = 1.0 / (p.dx * p.dx);
    p.inv_dy2 = 1.0 / (p.dy * p.dy);
    p.inv_dz2 = 1.0 / (p.dz * p.dz);
    
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
    initializeConcentration(d_cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz, p);
        
        // Update concentration
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz, p);
        
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
