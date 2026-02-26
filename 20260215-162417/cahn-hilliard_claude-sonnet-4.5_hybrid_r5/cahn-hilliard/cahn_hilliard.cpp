#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device version of idx3
__device__ inline size_t idx3_dev(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for computing Laplacian
__device__ double computeLaplacian_dev(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                       const double dx, const double dy, const double dz, 
                                       const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3_dev(xp, y, z, nx, ny)] + c[idx3_dev(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3_dev(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3_dev(x, yp, z, nx, ny)] + c[idx3_dev(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3_dev(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3_dev(x, y, zp, nx, ny)] + c[idx3_dev(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3_dev(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// CUDA kernel for computing chemical potential
__global__ void computeChemicalPotential_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, 
                                                const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3_dev(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacian_dev(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// CUDA kernel for Cahn-Hilliard update
__global__ void cahnHilliardUpdate_kernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, 
                                         const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3_dev(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacian_dev(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Compute chemical potential (GPU version)
void computeChemicalPotential(const double* d_c, double* d_mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    computeChemicalPotential_kernel<<<gridSize, blockSize>>>(
        d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

// Cahn-Hilliard update step (GPU version)
void cahnHilliardUpdate(double* d_cnew, const double* d_cold, const double* d_mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    cahnHilliardUpdate_kernel<<<gridSize, blockSize>>>(
        d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

// Initialize concentration field with MPI-aware domain decomposition
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                            const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1] using global z coordinate
                const size_t global_z = z + z_offset;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Set GPU device based on local rank
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = world_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition along Z
    const size_t nz_global = nz;
    const size_t nz_per_rank = (nz_global + world_size - 1) / world_size;
    const size_t z_start = world_rank * nz_per_rank;
    const size_t z_end = std::min(z_start + nz_per_rank, nz_global);
    const size_t nz_local = z_end - z_start;
    
    // Add ghost layers (1 on each side for neighbors)
    const size_t nz_with_ghost = nz_local + 2;
    const bool has_lower = (world_rank > 0);
    const bool has_upper = (world_rank < world_size - 1);
    
    if (world_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz_global);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
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
    
    size_t gridSize_local = nx * ny * nz_with_ghost;
    
    // Allocate host arrays
    std::vector<double> cold(gridSize_local);
    std::vector<double> cnew(gridSize_local);
    
    // Initialize concentration field
    if (world_rank == 0) printf("Initializing concentration field...\n");
    
    // Initialize without ghost layers
    std::vector<double> c_init(nx * ny * nz_local);
    initializeConcentration(c_init, nx, ny, nz_local, z_start, nz_global);
    
    // Copy to cold with ghost layer offset (start at z=1)
    #pragma omp parallel for
    for (size_t i = 0; i < c_init.size(); ++i) {
        size_t x = i % nx;
        size_t y = (i / nx) % ny;
        size_t z = i / (nx * ny);
        cold[idx3(x, y, z + 1, nx, ny)] = c_init[i];
    }
    
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize_local * sizeof(double)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize_local * sizeof(double), cudaMemcpyHostToDevice));
    
    // Halo exchange buffers
    std::vector<double> send_lower(nx * ny), recv_lower(nx * ny);
    std::vector<double> send_upper(nx * ny), recv_upper(nx * ny);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (world_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Halo exchange
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize_local * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Pack and exchange halos
        if (has_lower) {
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                send_lower[i] = cold[idx3(i % nx, (i / nx) % ny, 1, nx, ny)];
            }
            MPI_Sendrecv(send_lower.data(), nx * ny, MPI_DOUBLE, world_rank - 1, 0,
                        recv_lower.data(), nx * ny, MPI_DOUBLE, world_rank - 1, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                cold[idx3(i % nx, (i / nx) % ny, 0, nx, ny)] = recv_lower[i];
            }
        } else {
            // Copy boundary for clamped BC
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                cold[idx3(i % nx, (i / nx) % ny, 0, nx, ny)] = cold[idx3(i % nx, (i / nx) % ny, 1, nx, ny)];
            }
        }
        
        if (has_upper) {
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                send_upper[i] = cold[idx3(i % nx, (i / nx) % ny, nz_local, nx, ny)];
            }
            MPI_Sendrecv(send_upper.data(), nx * ny, MPI_DOUBLE, world_rank + 1, 1,
                        recv_upper.data(), nx * ny, MPI_DOUBLE, world_rank + 1, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                cold[idx3(i % nx, (i / nx) % ny, nz_local + 1, nx, ny)] = recv_upper[i];
            }
        } else {
            // Copy boundary for clamped BC
            #pragma omp parallel for
            for (size_t i = 0; i < nx * ny; ++i) {
                cold[idx3(i % nx, (i / nx) % ny, nz_local + 1, nx, ny)] = cold[idx3(i % nx, (i / nx) % ny, nz_local, nx, ny)];
            }
        }
        
        CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize_local * sizeof(double), cudaMemcpyHostToDevice));
        
        // Compute chemical potential
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz_with_ghost, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz_with_ghost, D, dt, dx, dy, dz);
        
        // Swap device pointers
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz_global) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize_local * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Extract non-ghost data
    std::vector<double> c_final(nx * ny * nz_local);
    #pragma omp parallel for
    for (size_t i = 0; i < c_final.size(); ++i) {
        size_t x = i % nx;
        size_t y = (i / nx) % ny;
        size_t z = i / (nx * ny);
        c_final[i] = cold[idx3(x, y, z + 1, nx, ny)];
    }
    
    // Gather results to rank 0 for validation and output
    std::vector<double> c_global;
    if (world_rank == 0) {
        c_global.resize(nx * ny * nz_global);
    }
    
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        size_t r_z_start = r * nz_per_rank;
        size_t r_z_end = std::min(r_z_start + nz_per_rank, nz_global);
        recvcounts[r] = nx * ny * (r_z_end - r_z_start);
        displs[r] = r_z_start * nx * ny;
    }
    
    MPI_Gatherv(c_final.data(), c_final.size(), MPI_DOUBLE,
                c_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && world_rank == 0) {
        print_results(c_global, "Concentration");
    }
    
    // Validation
    if (validate && world_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(c_global, nx, ny, nz_global);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    MPI_Finalize();
    
    if (validate && world_rank == 0) {
        return validateResult(c_global, nx, ny, nz_global) ? 0 : 1;
    }
    
    return 0;
}
