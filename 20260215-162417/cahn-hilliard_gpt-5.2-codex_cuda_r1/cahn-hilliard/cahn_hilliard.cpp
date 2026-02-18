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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                              \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

constexpr int kBlockSize = 256;

__device__ __forceinline__ double laplacianDevice(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                                  const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                                  const size_t x, const size_t y, const size_t z) {
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;

    const size_t slice = nx * ny;
    const size_t base = z * slice + y * nx;
    const double center = c[base + x];

    const double cxx = (c[base + xp] + c[base + xn] - 2.0 * center) * inv_dx2;
    const size_t base_yp = z * slice + yp * nx;
    const size_t base_yn = z * slice + yn * nx;
    const double cyy = (c[base_yp + x] + c[base_yn + x] - 2.0 * center) * inv_dy2;
    const size_t base_zp = zp * slice + y * nx;
    const size_t base_zn = zn * slice + y * nx;
    const double czz = (c[base_zp + x] + c[base_zn + x] - 2.0 * center) * inv_dz2;

    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }

    const unsigned long long linear_id = static_cast<unsigned long long>(idx);
    const double pseudo = (static_cast<double>(((linear_id + 1ULL) * 1299709ULL) % vol)) / static_cast<double>(vol);
    c[idx] = -1.0 + 2.0 * pseudo;
}

__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }

    const size_t x = idx % nx;
    const size_t y = (idx / nx) % ny;
    const size_t z = idx / (nx * ny);
    const double cv = c[idx];
    const double lap = laplacianDevice(c, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                        const double coeff) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t vol = nx * ny * nz;
    if (idx >= vol) {
        return;
    }

    const size_t x = idx % nx;
    const size_t y = (idx / nx) % ny;
    const size_t z = idx / (nx * ny);
    const double lap = laplacianDevice(mu, nx, ny, nz, inv_dx2, inv_dy2, inv_dz2, x, y, z);
    cnew[idx] = cold[idx] + coeff * lap;
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
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Allocate arrays
    std::vector<double> cold(gridSize);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    const unsigned int blocks = static_cast<unsigned int>((gridSize + kBlockSize - 1) / kBlockSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<blocks, kBlockSize>>>(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double coeff = dt * D;

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<blocks, kBlockSize>>>(d_cold, d_mu, nx, ny, nz,
                                                               inv_dx2, inv_dy2, inv_dz2,
                                                               gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        cahnHilliardUpdateKernel<<<blocks, kBlockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nz,
                                                         inv_dx2, inv_dy2, inv_dz2, coeff);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));

    // Calculate performance
    double cellUpdates = static_cast<double>(gridSize) * iterations;
    double mcups = cellUpdates / (static_cast<double>(elapsedMs) / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

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
