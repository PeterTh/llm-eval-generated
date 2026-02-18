#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    const cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        std::exit(1); \
    } \
} while (0)

__device__ __forceinline__ size_t idx3_dev(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void init_concentration_kernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t vol = nx * ny * nz;
    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = static_cast<double>((((linear_id + 1ull) * 1299709ull) % vol)) / static_cast<double>(vol);
    c[idx3_dev(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
}

__device__ __forceinline__ double laplacian_clamped(const double* __restrict__ a,
                                                   const size_t nx, const size_t ny, const size_t nz,
                                                   const double dx2, const double dy2, const double dz2,
                                                   const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? (x + 1) : x;
    const size_t yp = (y < ny - 1) ? (y + 1) : y;
    const size_t zp = (z < nz - 1) ? (z + 1) : z;
    const size_t xn = (x > 0) ? (x - 1) : 0;
    const size_t yn = (y > 0) ? (y - 1) : 0;
    const size_t zn = (z > 0) ? (z - 1) : 0;

    const size_t idx = idx3_dev(x, y, z, nx, ny);
    const double av = a[idx];

    const double axx = (a[idx3_dev(xp, y, z, nx, ny)] + a[idx3_dev(xn, y, z, nx, ny)] - 2.0 * av) / dx2;
    const double ayy = (a[idx3_dev(x, yp, z, nx, ny)] + a[idx3_dev(x, yn, z, nx, ny)] - 2.0 * av) / dy2;
    const double azz = (a[idx3_dev(x, y, zp, nx, ny)] + a[idx3_dev(x, y, zn, nx, ny)] - 2.0 * av) / dz2;

    return axx + ayy + azz;
}

__global__ void chemical_potential_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double dx2, const double dy2, const double dz2,
                                         const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3_dev(x, y, z, nx, ny);
    const double cv = c[idx];

    const double lap = laplacian_clamped(c, nx, ny, nz, dx2, dy2, dz2, x, y, z);
    const double term = ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB);

    mu[idx] = 4.5 * term + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void update_kernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double dx2, const double dy2, const double dz2,
                             const double D, const double dt) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = idx3_dev(x, y, z, nx, ny);
    const double lap_mu = laplacian_clamped(mu, nx, ny, nz, dx2, dy2, dz2, x, y, z);

    cnew[idx] = cold[idx] + dt * D * lap_mu;
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
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
    
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    // Allocate device arrays
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));

    const dim3 block(32, 4, 2);
    const dim3 grid((unsigned int)((nx + block.x - 1) / block.x),
                    (unsigned int)((ny + block.y - 1) / block.y),
                    (unsigned int)((nz + block.z - 1) / block.z));

    // Initialize concentration field on GPU (matches original deterministic formula)
    printf("Initializing concentration field...\n");
    init_concentration_kernel<<<grid, block>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());

    // Run simulation on GPU
    printf("Running Cahn-Hilliard simulation...\n");

    cudaEvent_t startEv, stopEv;
    CUDA_CHECK(cudaEventCreate(&startEv));
    CUDA_CHECK(cudaEventCreate(&stopEv));

    CUDA_CHECK(cudaEventRecord(startEv));
    for (int t = 0; t < iterations; ++t) {
        chemical_potential_kernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        update_kernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz, dx2, dy2, dz2, D, dt);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaEventRecord(stopEv));
    CUDA_CHECK(cudaEventSynchronize(stopEv));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEv, stopEv));
    const double elapsedMsD = std::max(1e-6, (double)elapsedMs);
    const double displayMsD = std::max(1.0, elapsedMsD);

    printf("Computation time: %ld ms\n", (long)llround(displayMsD));

    // Calculate performance
    const double cellUpdates = (double)gridSize * (double)iterations;
    const double mcups = cellUpdates / (elapsedMsD / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int rc = 0;

    // Copy results back only if needed
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(cold, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
            rc = 0;
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(startEv));
    CUDA_CHECK(cudaEventDestroy(stopEv));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));

    return rc;
}
