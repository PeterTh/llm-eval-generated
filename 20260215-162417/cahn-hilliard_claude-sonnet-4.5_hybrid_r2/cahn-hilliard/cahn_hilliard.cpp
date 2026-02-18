#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOSTDEV
#define CUDA_DEVICE
#define CUDA_GLOBAL
#endif

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)
#else
#define CUDA_CHECK(call) call
#endif

// 3D index calculation
CUDA_HOSTDEV inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

#ifdef __CUDACC__
// CUDA kernel for Laplacian computation
CUDA_DEVICE double computeLaplacianDevice(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                         const double dx, const double dy, const double dz, 
                                         const size_t x, const size_t y, const size_t z) {
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

// CUDA kernel for computing chemical potential
CUDA_GLOBAL void computeChemicalPotentialKernel(const double* c, double* mu,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, 
                                                const double e_BB, const double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDevice(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// CUDA kernel for Cahn-Hilliard update
CUDA_GLOBAL void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                          const size_t nx, const size_t ny, const size_t nz,
                                          const double D, const double dt, 
                                          const double dx, const double dy, const double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacianDevice(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}
#endif

// CPU version of Laplacian computation
double computeLaplacianCPU(const double* c, const size_t nx, const size_t ny, const size_t nz,
                           const double dx, const double dy, const double dz, 
                           const size_t x, const size_t y, const size_t z) {
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

// Compute Laplacian with clamped boundary conditions (for validation)
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    return computeLaplacianCPU(c.data(), nx, ny, nz, dx, dy, dz, x, y, z);
}

// Compute chemical potential (GPU-accelerated when available, OpenMP otherwise)
void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              [[maybe_unused]] bool use_cuda, [[maybe_unused]] const double* d_c, [[maybe_unused]] double* d_mu) {
#ifdef __CUDACC__
    if (use_cuda) {
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
        
        computeChemicalPotentialKernel<<<numBlocks, threadsPerBlock>>>(
            d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    
    // OpenMP CPU fallback
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianCPU(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step (GPU-accelerated when available, OpenMP otherwise)
void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        [[maybe_unused]] bool use_cuda, [[maybe_unused]] double* d_cnew, 
                        [[maybe_unused]] const double* d_cold, [[maybe_unused]] const double* d_mu) {
#ifdef __CUDACC__
    if (use_cuda) {
        dim3 threadsPerBlock(8, 8, 8);
        dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
        
        cahnHilliardUpdateKernel<<<numBlocks, threadsPerBlock>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    
    // OpenMP CPU fallback
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacianCPU(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(3)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Check for CUDA availability
    [[maybe_unused]] int deviceCount = 0;
    bool use_cuda = false;
#ifdef __CUDACC__
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    use_cuda = (err == cudaSuccess && deviceCount > 0);
    if (use_cuda) {
        int device = rank % deviceCount;
        CUDA_CHECK(cudaSetDevice(device));
    }
#endif
    
    // Domain decomposition in Z direction
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        local_nz++;
    }
    size_t z_end = z_start + local_nz;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA: %s\n", use_cuda ? "enabled" : "disabled (CPU fallback)");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank reports its local domain
    printf("Rank %d: local z range [%zu, %zu), size %zu\n", rank, z_start, z_end, local_nz);
    
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
    
    size_t local_gridSize = nx * ny * local_nz;
    
    // Allocate host arrays for local domain
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    // Initialize concentration field for local domain
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    
    // Initialize local portion
    const size_t global_vol = nx * ny * nz;
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_start + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
                cold[local_idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    
    // Allocate device arrays if CUDA is available
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    if (use_cuda) {
#ifdef __CUDACC__
        CUDA_CHECK(cudaMalloc(&d_cold, local_gridSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, local_gridSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, local_gridSize * sizeof(double)));
        
        // Copy initial data to device
        CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), local_gridSize * sizeof(double), cudaMemcpyHostToDevice));
#endif
    }
    
    // Allocate buffers for halo exchange
    std::vector<double> send_top(nx * ny), recv_top(nx * ny);
    std::vector<double> send_bottom(nx * ny), recv_bottom(nx * ny);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Halo exchange
        if (size > 1) {
            // Copy boundary data from device to host if using CUDA
            if (use_cuda) {
#ifdef __CUDACC__
                if (rank > 0) {
                    CUDA_CHECK(cudaMemcpy(send_bottom.data(), d_cold, nx * ny * sizeof(double), cudaMemcpyDeviceToHost));
                }
                if (rank < size - 1) {
                    CUDA_CHECK(cudaMemcpy(send_top.data(), d_cold + (local_nz - 1) * nx * ny, 
                              nx * ny * sizeof(double), cudaMemcpyDeviceToHost));
                }
#endif
            } else {
                if (rank > 0) {
                    std::copy(cold.begin(), cold.begin() + nx * ny, send_bottom.begin());
                }
                if (rank < size - 1) {
                    std::copy(cold.begin() + (local_nz - 1) * nx * ny, cold.end(), send_top.begin());
                }
            }
            
            // Exchange halos with neighbors
            MPI_Request reqs[4];
            int req_count = 0;
            
            if (rank > 0) {
                MPI_Isend(send_bottom.data(), nx * ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
                MPI_Irecv(recv_bottom.data(), nx * ny, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            if (rank < size - 1) {
                MPI_Isend(send_top.data(), nx * ny, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
                MPI_Irecv(recv_top.data(), nx * ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }
        
        // Compute chemical potential
        if (use_cuda) {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, local_nz, dx, dy, dz, 
                                    gamma, e_AA, e_BB, e_AB, true, d_cold, d_mu);
        } else {
            computeChemicalPotential(cold.data(), mu.data(), nx, ny, local_nz, dx, dy, dz, 
                                    gamma, e_AA, e_BB, e_AB, false, nullptr, nullptr);
        }
        
        // Update concentration
        if (use_cuda) {
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, local_nz, D, dt, dx, dy, dz,
                             true, d_cnew, d_cold, d_mu);
        } else {
            cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), nx, ny, local_nz, D, dt, dx, dy, dz,
                             false, nullptr, nullptr, nullptr);
        }
        
        // Swap buffers
        if (use_cuda) {
#ifdef __CUDACC__
            std::swap(d_cold, d_cnew);
#endif
        } else {
            std::swap(cold, cnew);
        }
    }
    
    if (use_cuda) {
#ifdef __CUDACC__
        CUDA_CHECK(cudaDeviceSynchronize());
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy final result back to host if using CUDA
    if (use_cuda) {
#ifdef __CUDACC__
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, local_gridSize * sizeof(double), cudaMemcpyDeviceToHost));
#endif
    }
    
    // Gather results at rank 0 for validation/printing
    std::vector<double> global_result;
    if (rank == 0) {
        global_result.resize(nx * ny * nz);
    }
    
    // Gather all local results to rank 0
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int local_count = local_gridSize;
    
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(cold.data(), local_gridSize, MPI_DOUBLE,
                global_result.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        // Print results for external validation
        if (printResults) {
            print_results(global_result, "Concentration");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_result, nx, ny, nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    // Cleanup
    if (use_cuda) {
#ifdef __CUDACC__
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
#endif
    }
    
    MPI_Finalize();
    return 0;
}
