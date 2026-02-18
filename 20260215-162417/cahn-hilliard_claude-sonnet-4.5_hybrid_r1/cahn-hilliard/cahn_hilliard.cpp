#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

#ifdef USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)
#endif

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#ifdef USE_CUDA
__device__ inline size_t idx3_device(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}
#endif

#ifdef USE_CUDA
// CUDA kernel for computing Laplacian
__device__ double computeLaplacian_device(const double* c, const size_t nx, const size_t ny, const size_t nz,
                                          const double dx, const double dy, const double dz, 
                                          const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3_device(xp, y, z, nx, ny)] + c[idx3_device(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3_device(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3_device(x, yp, z, nx, ny)] + c[idx3_device(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3_device(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3_device(x, y, zp, nx, ny)] + c[idx3_device(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3_device(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}
#endif

// Compute Laplacian with clamped boundary conditions (CPU version)
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

#ifdef USE_CUDA
// CUDA kernel for computing chemical potential
__global__ void computeChemicalPotential_kernel(const double* c, double* mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3_device(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacian_device(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}
#endif

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    #pragma omp parallel for collapse(3)
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

#ifdef USE_CUDA
// CUDA kernel for Cahn-Hilliard update
__global__ void cahnHilliardUpdate_kernel(double* cnew, const double* cold, const double* mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3_device(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacian_device(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}
#endif

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    #pragma omp parallel for collapse(3)
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
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz, 
                            const size_t global_z_offset, const size_t global_nz) {
    const size_t global_vol = nx * ny * global_nz;
    
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t global_z = global_z_offset + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
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
    
#ifdef USE_CUDA
    // Set GPU device based on rank (for multi-GPU nodes)
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
#endif
    
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
    
    // Decompose domain in Z direction across MPI ranks
    size_t global_nz = nz;
    size_t local_nz = global_nz / size;
    size_t remainder = global_nz % size;
    
    // Distribute remainder to first few ranks
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        local_nz++;
    }
    
    if (rank == 0) {
#ifdef USE_CUDA
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", deviceCount);
#else
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
#endif
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, global_nz);
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
    
    size_t local_gridSize = nx * ny * local_nz;
    
    // Allocate host arrays
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, z_start, global_nz);
    
#ifdef USE_CUDA
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_gridSize * sizeof(double)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), local_gridSize * sizeof(double), cudaMemcpyHostToDevice));
    
    // Setup CUDA grid and block dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (local_nz + blockDim.z - 1) / blockDim.z);
#endif
    
    // Halo exchange buffers for MPI (Z boundaries)
    std::vector<double> send_buf_up(nx * ny);
    std::vector<double> send_buf_down(nx * ny);
    std::vector<double> recv_buf_up(nx * ny);
    std::vector<double> recv_buf_down(nx * ny);
    
    int rank_up = (rank + 1) % size;
    int rank_down = (rank - 1 + size) % size;
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
#ifdef USE_CUDA
        // Compute chemical potential on GPU
        computeChemicalPotential_kernel<<<gridDim, blockDim>>>(d_cold, d_mu, nx, ny, local_nz, 
                                                               dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy back to host for MPI exchange (only boundary planes)
        if (size > 1) {
            CUDA_CHECK(cudaMemcpy(mu.data(), d_mu, local_gridSize * sizeof(double), cudaMemcpyDeviceToHost));
            
            // Pack boundary planes
            if (rank < size - 1) {
                #pragma omp parallel for collapse(2)
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        send_buf_up[y * nx + x] = mu[idx3(x, y, local_nz - 1, nx, ny)];
                    }
                }
            }
            
            if (rank > 0) {
                #pragma omp parallel for collapse(2)
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        send_buf_down[y * nx + x] = mu[idx3(x, y, 0, nx, ny)];
                    }
                }
            }
            
            // Exchange halos
            MPI_Request req[4];
            int req_count = 0;
            
            if (rank < size - 1) {
                MPI_Isend(send_buf_up.data(), nx * ny, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &req[req_count++]);
                MPI_Irecv(recv_buf_up.data(), nx * ny, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, &req[req_count++]);
            }
            
            if (rank > 0) {
                MPI_Isend(send_buf_down.data(), nx * ny, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, &req[req_count++]);
                MPI_Irecv(recv_buf_down.data(), nx * ny, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &req[req_count++]);
            }
            
            MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
            
            // Copy back to device after MPI exchange
            CUDA_CHECK(cudaMemcpy(d_mu, mu.data(), local_gridSize * sizeof(double), cudaMemcpyHostToDevice));
        }
        
        // Update concentration on GPU
        cahnHilliardUpdate_kernel<<<gridDim, blockDim>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, 
                                                         D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap device pointers
        double* temp = d_cold;
        d_cold = d_cnew;
        d_cnew = temp;
#else
        // CPU+OpenMP path
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // MPI halo exchange for boundaries
        if (size > 1) {
            // Pack boundary planes
            if (rank < size - 1) {
                #pragma omp parallel for collapse(2)
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        send_buf_up[y * nx + x] = mu[idx3(x, y, local_nz - 1, nx, ny)];
                    }
                }
            }
            
            if (rank > 0) {
                #pragma omp parallel for collapse(2)
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        send_buf_down[y * nx + x] = mu[idx3(x, y, 0, nx, ny)];
                    }
                }
            }
            
            // Exchange halos
            MPI_Request req[4];
            int req_count = 0;
            
            if (rank < size - 1) {
                MPI_Isend(send_buf_up.data(), nx * ny, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &req[req_count++]);
                MPI_Irecv(recv_buf_up.data(), nx * ny, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, &req[req_count++]);
            }
            
            if (rank > 0) {
                MPI_Isend(send_buf_down.data(), nx * ny, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, &req[req_count++]);
                MPI_Irecv(recv_buf_down.data(), nx * ny, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &req[req_count++]);
            }
            
            MPI_Waitall(req_count, req, MPI_STATUSES_IGNORE);
        }
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
#endif
    }
    
#ifdef USE_CUDA
    CUDA_CHECK(cudaDeviceSynchronize());
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
#ifdef USE_CUDA
    // Copy final result back to host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, local_gridSize * sizeof(double), cudaMemcpyDeviceToHost));
#endif
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        size_t global_gridSize = nx * ny * global_nz;
        double cellUpdates = (double)global_gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results at rank 0 for validation/output
    std::vector<double> global_result;
    if (rank == 0) {
        global_result.resize(nx * ny * global_nz);
    }
    
    // Prepare send counts and displacements
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
        size_t r_local_nz = global_nz / size;
        if (r < (int)(global_nz % size)) {
            r_local_nz++;
        }
        sendcounts[r] = nx * ny * r_local_nz;
        displs[r] = (r == 0) ? 0 : displs[r - 1] + sendcounts[r - 1];
    }
    
    MPI_Gatherv(cold.data(), local_gridSize, MPI_DOUBLE,
                global_result.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(global_result, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(global_result, nx, ny, global_nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
#ifdef USE_CUDA
    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
#endif
    
    MPI_Finalize();
    
    if (rank == 0 && validate) {
        bool valid = validateResult(global_result, nx, ny, global_nz);
        return valid ? 0 : 1;
    }
    
    return 0;
}
