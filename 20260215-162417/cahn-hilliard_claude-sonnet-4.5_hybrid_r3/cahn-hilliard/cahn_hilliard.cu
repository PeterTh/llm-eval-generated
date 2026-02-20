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
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, 
                                        const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for computing Laplacian
__device__ double computeLaplacianDevice(const double* c, const size_t nx, const size_t ny, 
                                         const size_t nz_local, const double dx, const double dy, 
                                         const double dz, const size_t x, const size_t y, const size_t z,
                                         const bool is_first_rank, const bool is_last_rank) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz_local - 1) ? z + 1 : z;
    if (z == nz_local - 1 && is_last_rank) zp = z;  // Clamp at global boundary
    
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : z;
    if (z == 0 && is_first_rank) zn = 0;  // Clamp at global boundary
    
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
                                               const size_t nx, const size_t ny, const size_t nz_local,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, 
                                               const double e_BB, const double e_AB,
                                               const bool is_first_rank, const bool is_last_rank) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz_local) {
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDevice(c, nx, ny, nz_local, dx, dy, dz, x, y, z, is_first_rank, is_last_rank);
    }
}

// CUDA kernel for Cahn-Hilliard update
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                        const size_t nx, const size_t ny, const size_t nz_local,
                                        const double D, const double dt, 
                                        const double dx, const double dy, const double dz,
                                        const bool is_first_rank, const bool is_last_rank) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz_local) {
        const size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacianDevice(mu, nx, ny, nz_local, dx, dy, dz, x, y, z, is_first_rank, is_last_rank);
    }
}

// Initialize concentration field
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz_local,
                             const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t z_global = z + z_offset;
                const size_t linear_id = z_global * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const double* c, const size_t gridSize, int rank) {
    // Check for NaN or Inf
    bool valid = true;
    #pragma omp parallel for reduction(&:valid)
    for (size_t i = 0; i < gridSize; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            valid = false;
        }
    }
    
    if (!valid) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Find min/max
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    
    // Global reduction
    double globalMin, globalMax;
    MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        
        if (globalMax > 10.0 || globalMin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <nprocs> %s [options]\n", progName);
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
    
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = rank % num_devices;
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition in Z direction
    size_t nz_global = nz;
    size_t nz_local = nz_global / nprocs;
    size_t remainder = nz_global % nprocs;
    
    // Distribute remainder among first ranks
    size_t z_offset = rank * nz_local + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        nz_local++;
    }
    
    bool is_first_rank = (rank == 0);
    bool is_last_rank = (rank == nprocs - 1);
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", nprocs);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz_global);
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
    
    size_t gridSize_local = nx * ny * nz_local;
    
    // Allocate host arrays
    double *h_cold, *h_cnew, *h_mu;
    CUDA_CHECK(cudaMallocHost(&h_cold, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_cnew, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_mu, gridSize_local * sizeof(double)));
    
    // Allocate device arrays
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize_local * sizeof(double)));
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(h_cold, nx, ny, nz_local, z_offset, nz_global);
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_cold, h_cold, gridSize_local * sizeof(double), cudaMemcpyHostToDevice));
    
    // Setup CUDA grid dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz_local + blockDim.z - 1) / blockDim.z);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential on GPU
        computeChemicalPotentialKernel<<<gridDim, blockDim>>>(
            d_cold, d_mu, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB, is_first_rank, is_last_rank);
        CUDA_CHECK(cudaGetLastError());
        
        // Update concentration on GPU
        cahnHilliardUpdateKernel<<<gridDim, blockDim>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local, D, dt, dx, dy, dz, is_first_rank, is_last_rank);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap device pointers
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy final result back to host
    CUDA_CHECK(cudaMemcpy(h_cold, d_cold, gridSize_local * sizeof(double), cudaMemcpyDeviceToHost));
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        size_t total_cells = nx * ny * nz_global;
        double cellUpdates = (double)total_cells * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for printing (only if requested)
    if (printResults) {
        std::vector<double> global_result;
        if (rank == 0) {
            global_result.resize(nx * ny * nz_global);
        }
        
        // Gather all data to rank 0
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        
        int local_count = gridSize_local;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(h_cold, gridSize_local, MPI_DOUBLE,
                   rank == 0 ? global_result.data() : nullptr,
                   recvcounts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            print_results(global_result, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(h_cold, gridSize_local, rank);
        
        // Global AND reduction
        bool global_valid;
        MPI_Reduce(&valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
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
    CUDA_CHECK(cudaFreeHost(h_cold));
    CUDA_CHECK(cudaFreeHost(h_cnew));
    CUDA_CHECK(cudaFreeHost(h_mu));
    
    MPI_Finalize();
    return 0;
}
