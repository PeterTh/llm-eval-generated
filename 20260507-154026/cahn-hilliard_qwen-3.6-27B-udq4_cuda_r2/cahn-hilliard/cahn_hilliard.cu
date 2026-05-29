#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation (host and device)
__device__ __host__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device helper: clamped neighbor index
__device__ inline size_t clamp_idx(size_t v, size_t lo, size_t hi) {
    return (v < lo) ? lo : (v >= hi) ? hi - 1 : v;
}

// CUDA kernel: initialize concentration field
__global__ void initKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t linear_id = z * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
}

// CUDA kernel: compute chemical potential (grid-stride for flexibility)
__global__ void computeMuKernel(const double* __restrict__ c, double* __restrict__ mu,
                                const size_t nx, const size_t ny, const size_t nz,
                                const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t base = z * ny * nx + y * nx;
    const size_t idx = base + x;
    const double cv = c[idx];

    const size_t xp = clamp_idx(x + 1, 0, nx);
    const size_t yp = clamp_idx(y + 1, 0, ny);
    const size_t zp = clamp_idx(z + 1, 0, nz);
    const size_t xn = clamp_idx(x > 0 ? x - 1 : 0, 0, nx);
    const size_t yn = clamp_idx(y > 0 ? y - 1 : 0, 0, ny);
    const size_t zn = clamp_idx(z > 0 ? z - 1 : 0, 0, nz);

    const double lap = (c[base + xp] + c[base + xn] - 2.0 * cv) * inv_dx2
                     + (c[z * ny * nx + yp * nx + x] + c[z * ny * nx + yn * nx + x] - 2.0 * cv) * inv_dy2
                     + (c[zp * ny * nx + y * nx + x] + c[zn * ny * nx + y * nx + x] - 2.0 * cv) * inv_dz2;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * lap;
}

// CUDA kernel: Cahn-Hilliard concentration update (grid-stride)
__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                             double* __restrict__ cnew,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double dtD, const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t base = z * ny * nx + y * nx;
    const size_t idx = base + x;
    const double muv = mu[idx];

    const size_t xp = clamp_idx(x + 1, 0, nx);
    const size_t yp = clamp_idx(y + 1, 0, ny);
    const size_t zp = clamp_idx(z + 1, 0, nz);
    const size_t xn = clamp_idx(x > 0 ? x - 1 : 0, 0, nx);
    const size_t yn = clamp_idx(y > 0 ? y - 1 : 0, 0, ny);
    const size_t zn = clamp_idx(z > 0 ? z - 1 : 0, 0, nz);

    const double lap = (mu[base + xp] + mu[base + xn] - 2.0 * muv) * inv_dx2
                     + (mu[z * ny * nx + yp * nx + x] + mu[z * ny * nx + yn * nx + x] - 2.0 * muv) * inv_dy2
                     + (mu[zp * ny * nx + y * nx + x] + mu[zn * ny * nx + y * nx + x] - 2.0 * muv) * inv_dz2;

    cnew[idx] = cold[idx] + dtD * lap;
}

// Helper to launch 3D kernels with proper grid/block dimensions
static dim3 makeBlock(const size_t nx, const size_t ny, const size_t nz) {
    const int bx = min((int)nx, 32);
    const int by = min((int)ny, 32);
    const int bz = min((int)nz, 8);
    // Ensure total threads per block <= 1024
    int total = bx * by * bz;
    if (total > 1024) {
        int s = 8;
        while (bx * by * s > 1024 && s > 1) --s;
        return dim3(bx, by, s);
    }
    return dim3(bx, by, bz);
}

static dim3 makeGrid(const size_t nx, const size_t ny, const size_t nz, const dim3 block) {
    return dim3(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z
    );
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
    
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    
    // Precompute inverse squared spacings
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    
    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    cudaMalloc(&d_cold, bytes);
    cudaMalloc(&d_cnew, bytes);
    cudaMalloc(&d_mu, bytes);
    
    // Host buffers for final result
    std::vector<double> h_cold(gridSize);
    
    // Initialize concentration field on GPU
    printf("Initializing concentration field...\n");
    {
        dim3 block = makeBlock(nx, ny, nz);
        dim3 grid = makeGrid(nx, ny, nz, block);
        initKernel<<<grid, block>>>(d_cold, nx, ny, nz);
        cudaDeviceSynchronize();
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        dim3 block = makeBlock(nx, ny, nz);
        dim3 grid = makeGrid(nx, ny, nz, block);
        
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential on GPU
            computeMuKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz,
                                             inv_dx2, inv_dy2, inv_dz2,
                                             gamma, e_AA, e_BB, e_AB);
            
            // Update concentration on GPU
            updateKernel<<<grid, block>>>(d_cold, d_mu, d_cnew, nx, ny, nz,
                                          dtD, inv_dx2, inv_dy2, inv_dz2);
            
            // Swap device pointers
            double* tmp = d_cold;
            d_cold = d_cnew;
            d_cnew = tmp;
        }
        cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy final result back to host
    cudaMemcpy(h_cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(h_cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_cold, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    
    return 0;
}
