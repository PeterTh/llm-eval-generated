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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for computing Laplacian
__device__ double computeLaplacianDevice(const double* c, const size_t nx, const size_t ny, const size_t nz,
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

// CUDA kernel for chemical potential computation
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                                              const size_t nx, const size_t ny, const size_t nz,
                                              const double dx, const double dy, const double dz,
                                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDevice(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// CUDA kernel for Cahn-Hilliard update
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacianDevice(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

// Compute Laplacian with clamped boundary conditions (CPU version for validation)
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

// Compute chemical potential using CUDA
void computeChemicalPotential(const double* d_c, double* d_mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    computeChemicalPotentialKernel<<<gridSize, blockSize>>>(d_c, d_mu, nx, ny, nz, dx, dy, dz, 
                                                            gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

// Cahn-Hilliard update step using CUDA
void cahnHilliardUpdate(double* d_cnew, const double* d_cold, const double* d_mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y,
                  (nz + blockSize.z - 1) / blockSize.z);
    
    cahnHilliardUpdateKernel<<<gridSize, blockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

// Initialize concentration field with OpenMP parallelization
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Set GPU for this MPI rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
    if (rank == 0) {
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
        
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    // Domain decomposition in Z dimension
    size_t local_nz = nz / nprocs;
    size_t remainder = nz % nprocs;
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) local_nz++;
    
    // Add halo zones (1 layer on each side)
    size_t nz_with_halo = local_nz + 2;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI processes: %d\n", nprocs);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Local grid per rank: %zu x %zu x %zu (with halos)\n", nx, ny, nz_with_halo);
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
    
    size_t local_gridSize = nx * ny * nz_with_halo;
    
    // Allocate host arrays
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    
    // Initialize concentration field for local domain
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    
    // Initialize full field on rank 0, then distribute
    if (rank == 0) {
        size_t full_gridSize = nx * ny * nz;
        std::vector<double> full_c(full_gridSize);
        initializeConcentration(full_c, nx, ny, nz);
        
        // Copy local portion to rank 0
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = idx3(x, y, z + 1, nx, ny);  // +1 for halo
                    size_t global_idx = idx3(x, y, z, nx, ny);
                    cold[local_idx] = full_c[global_idx];
                }
            }
        }
        
        // Send to other ranks
        for (int r = 1; r < nprocs; ++r) {
            size_t r_local_nz = nz / nprocs;
            size_t r_remainder = nz % nprocs;
            size_t r_z_start = r * r_local_nz + std::min((size_t)r, r_remainder);
            if (r < (int)r_remainder) r_local_nz++;
            
            std::vector<double> send_buf(nx * ny * r_local_nz);
            #pragma omp parallel for collapse(3)
            for (size_t z = 0; z < r_local_nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        size_t buf_idx = z * (nx * ny) + y * nx + x;
                        size_t global_idx = idx3(x, y, r_z_start + z, nx, ny);
                        send_buf[buf_idx] = full_c[global_idx];
                    }
                }
            }
            MPI_Send(send_buf.data(), nx * ny * r_local_nz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
        }
    } else {
        std::vector<double> recv_buf(nx * ny * local_nz);
        MPI_Recv(recv_buf.data(), nx * ny * local_nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = idx3(x, y, z + 1, nx, ny);  // +1 for halo
                    size_t buf_idx = z * (nx * ny) + y * nx + x;
                    cold[local_idx] = recv_buf[buf_idx];
                }
            }
        }
    }
    
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_gridSize * sizeof(double)));
    
    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), local_gridSize * sizeof(double), cudaMemcpyHostToDevice));
    
    // Buffers for halo exchange
    std::vector<double> send_top(nx * ny), send_bottom(nx * ny);
    std::vector<double> recv_top(nx * ny), recv_bottom(nx * ny);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange halo zones with neighbors
        int top_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int bottom_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
        
        // Copy data from device for halo exchange
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, local_gridSize * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Prepare send buffers
        #pragma omp parallel for collapse(2)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                send_bottom[y * nx + x] = cold[idx3(x, y, 1, nx, ny)];  // First real layer
                send_top[y * nx + x] = cold[idx3(x, y, local_nz, nx, ny)];  // Last real layer
            }
        }
        
        // Exchange halos
        MPI_Sendrecv(send_bottom.data(), nx * ny, MPI_DOUBLE, top_rank, 0,
                     recv_top.data(), nx * ny, MPI_DOUBLE, bottom_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(send_top.data(), nx * ny, MPI_DOUBLE, bottom_rank, 1,
                     recv_bottom.data(), nx * ny, MPI_DOUBLE, top_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Update halo zones
        if (bottom_rank != MPI_PROC_NULL) {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    cold[idx3(x, y, local_nz + 1, nx, ny)] = recv_top[y * nx + x];
                }
            }
        }
        
        if (top_rank != MPI_PROC_NULL) {
            #pragma omp parallel for collapse(2)
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    cold[idx3(x, y, 0, nx, ny)] = recv_bottom[y * nx + x];
                }
            }
        }
        
        // Copy updated halos back to device
        CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), local_gridSize * sizeof(double), cudaMemcpyHostToDevice));
        
        // Compute chemical potential on GPU
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz_with_halo, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration on GPU
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz_with_halo, D, dt, dx, dy, dz);
        
        // Swap device pointers
        double* temp = d_cold;
        d_cold = d_cnew;
        d_cnew = temp;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy final result back to host
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, local_gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Gather results at rank 0
    std::vector<double> full_result;
    if (rank == 0) {
        full_result.resize(nx * ny * nz);
        
        // Copy local portion from rank 0
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = idx3(x, y, z + 1, nx, ny);  // +1 for halo
                    size_t global_idx = idx3(x, y, z, nx, ny);
                    full_result[global_idx] = cold[local_idx];
                }
            }
        }
        
        // Receive from other ranks
        for (int r = 1; r < nprocs; ++r) {
            size_t r_local_nz = nz / nprocs;
            size_t r_remainder = nz % nprocs;
            size_t r_z_start = r * r_local_nz + std::min((size_t)r, r_remainder);
            if (r < (int)r_remainder) r_local_nz++;
            
            std::vector<double> recv_buf(nx * ny * r_local_nz);
            MPI_Recv(recv_buf.data(), nx * ny * r_local_nz, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            #pragma omp parallel for collapse(3)
            for (size_t z = 0; z < r_local_nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        size_t buf_idx = z * (nx * ny) + y * nx + x;
                        size_t global_idx = idx3(x, y, r_z_start + z, nx, ny);
                        full_result[global_idx] = recv_buf[buf_idx];
                    }
                }
            }
        }
    } else {
        std::vector<double> send_buf(nx * ny * local_nz);
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < local_nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t local_idx = idx3(x, y, z + 1, nx, ny);  // +1 for halo
                    size_t buf_idx = z * (nx * ny) + y * nx + x;
                    send_buf[buf_idx] = cold[local_idx];
                }
            }
        }
        MPI_Send(send_buf.data(), nx * ny * local_nz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        // Print results for external validation
        if (printResults) {
            print_results(full_result, "Concentration");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_result, nx, ny, nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    MPI_Finalize();
    
    return 0;
}
