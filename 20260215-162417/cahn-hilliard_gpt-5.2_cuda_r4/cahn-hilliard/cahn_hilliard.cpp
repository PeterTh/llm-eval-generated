#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---- CUDA implementation (unconditional) ----
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t _err = (call);                                                  \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

__device__ __forceinline__ double laplacian_clamp_3d(const double* __restrict__ a,
                                                     const int x, const int y, const int z,
                                                     const int nx, const int ny, const int nz,
                                                     const size_t plane,
                                                     const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t idx = (static_cast<size_t>(z) * plane) + (static_cast<size_t>(y) * static_cast<size_t>(nx)) + static_cast<size_t>(x);

    const size_t idx_xp = idx + ((x + 1 < nx) ? 1u : 0u);
    const size_t idx_xn = idx - ((x > 0) ? 1u : 0u);
    const size_t idx_yp = idx + ((y + 1 < ny) ? static_cast<size_t>(nx) : 0u);
    const size_t idx_yn = idx - ((y > 0) ? static_cast<size_t>(nx) : 0u);
    const size_t idx_zp = idx + ((z + 1 < nz) ? plane : 0u);
    const size_t idx_zn = idx - ((z > 0) ? plane : 0u);

    const double c0 = a[idx];
    const double cxx = (a[idx_xp] + a[idx_xn] - 2.0 * c0) * inv_dx2;
    const double cyy = (a[idx_yp] + a[idx_yn] - 2.0 * c0) * inv_dy2;
    const double czz = (a[idx_zp] + a[idx_zn] - 2.0 * c0) * inv_dz2;
    return cxx + cyy + czz;
}

__global__ void chemical_potential_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                         const int nx, const int ny, const int nz,
                                         const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                         const double gamma,
                                         const double e_AA, const double e_BB, const double e_AB) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = (static_cast<size_t>(z) * plane) + (static_cast<size_t>(y) * static_cast<size_t>(nx)) + static_cast<size_t>(x);

    const double cv = c[idx];
    const double lap = laplacian_clamp_3d(c, x, y, z, nx, ny, nz, plane, inv_dx2, inv_dy2, inv_dz2);

    // Preserve the original arithmetic structure as closely as possible.
    const double t1 = (cv + 1.0) * e_AA;
    const double t2 = (cv - 1.0) * e_BB;
    const double t3 = 2.0 * cv * e_AB;
    const double mix = (t1 + t2) - t3;
    const double bulk = (4.5 * mix) + (3.0 * cv) + (cv * cv * cv);

    mu[idx] = bulk - gamma * lap;
}

__global__ void cahn_hilliard_update_kernel(double* __restrict__ cnew,
                                           const double* __restrict__ cold,
                                           const double* __restrict__ mu,
                                           const int nx, const int ny, const int nz,
                                           const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                           const double D, const double dt) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = (static_cast<size_t>(z) * plane) + (static_cast<size_t>(y) * static_cast<size_t>(nx)) + static_cast<size_t>(x);

    const double lap_mu = laplacian_clamp_3d(mu, x, y, z, nx, ny, nz, plane, inv_dx2, inv_dy2, inv_dz2);
    cnew[idx] = cold[idx] + (dt * D) * lap_mu;
}

static inline dim3 defaultBlockDim() {
    // 256 threads/block with good X coalescing for row-major 3D grids.
    return dim3(32, 4, 2);
}

static inline dim3 gridDim3(const size_t nx, const size_t ny, const size_t nz, const dim3 block) {
    return dim3(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                static_cast<unsigned int>((ny + block.y - 1) / block.y),
                static_cast<unsigned int>((nz + block.z - 1) / block.z));
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
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

    // Host storage for initialization / optional output.
    std::vector<double> cold(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Device allocations
    const size_t bytes = gridSize * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const dim3 block = defaultBlockDim();
    const dim3 grid = gridDim3(nx, ny, nz, block);

    // Run simulation on GPU
    printf("Running Cahn-Hilliard simulation...\n");

    // Ensure CUDA context creation doesn't pollute the timed region.
    CUDA_CHECK(cudaFree(0));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iterations; ++t) {
        chemical_potential_kernel<<<grid, block>>>(d_cold, d_mu, inx, iny, inz, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        cahn_hilliard_update_kernel<<<grid, block>>>(d_cnew, d_cold, d_mu, inx, iny, inz, inv_dx2, inv_dy2, inv_dz2, D, dt);
        CUDA_CHECK(cudaGetLastError());

        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_f = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_f, start, stop));

    const double ms_d = static_cast<double>(ms_f);
    const long ms = static_cast<long>(std::llround(ms_d));
    printf("Computation time: %ld ms\n", ms);

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    const double seconds = std::max(1e-9, ms_d / 1000.0);
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy back results only if needed.
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));
    
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
