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

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#ifdef USE_CUDA
// CUDA kernel for stencil computation
__global__ void stencilKernel(const Real* __restrict__ input, 
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < nx - 1 && y < ny - 1 && z < nz - 1) {
        const size_t idx = z * (nx * ny) + y * nx + x;
        
        const Real center = input[idx];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        const Real bottom = input[idx - nx * ny];
        const Real top = input[idx + nx * ny];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// CUDA kernel for copying boundaries
__global__ void copyBoundariesKernel(const Real* __restrict__ input,
                                     Real* __restrict__ output,
                                     const size_t nx, const size_t ny, const size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    
    if (idx < total) {
        const size_t x = idx % nx;
        const size_t y = (idx / nx) % ny;
        const size_t z = idx / (nx * ny);
        
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        }
    }
}
#endif

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz_local, 
                   const size_t z_start, const size_t nz_global) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_start + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[local_idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// CPU fallback for stencil computation
void stencilIterationCPU(const std::vector<Real>& input, 
                        std::vector<Real>& output,
                        const size_t nx, const size_t ny, const size_t nz_local) {
    // Process interior points (not on boundaries)
    #pragma omp parallel for collapse(3)
    for (size_t z = 1; z < nz_local - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - nx * ny];
                const Real top = input[idx + nx * ny];
                
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz_local-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

#ifdef USE_CUDA
// Hybrid MPI+CUDA stencil iteration with halo exchange
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      std::vector<Real>& h_send_top, std::vector<Real>& h_send_bottom,
                      std::vector<Real>& h_recv_top, std::vector<Real>& h_recv_bottom,
                      const int rank, const int size, const size_t nz_global) {
    
    // Determine halo exchange needs
    const bool has_bottom = (rank > 0);
    const bool has_top = (rank < size - 1);
    
    const size_t plane_size = nx * ny;
    
    MPI_Request send_req[2], recv_req[2];
    int req_count = 0;
    
    // Post receives first
    if (has_bottom) {
        MPI_Irecv(h_recv_bottom.data(), plane_size, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &recv_req[req_count++]);
    }
    if (has_top) {
        MPI_Irecv(h_recv_top.data(), plane_size, MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &recv_req[req_count++]);
    }
    
    // Copy boundary planes from device to host
    if (has_bottom) {
        CUDA_CHECK(cudaMemcpy(h_send_bottom.data(), d_input + idx3(0, 0, 1, nx, ny),
                              plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    if (has_top) {
        CUDA_CHECK(cudaMemcpy(h_send_top.data(), d_input + idx3(0, 0, nz_local - 2, nx, ny),
                              plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    
    // Send boundary planes
    int send_count = 0;
    if (has_bottom) {
        MPI_Isend(h_send_bottom.data(), plane_size, MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &send_req[send_count++]);
    }
    if (has_top) {
        MPI_Isend(h_send_top.data(), plane_size, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &send_req[send_count++]);
    }
    
    // Wait for receives to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, recv_req, MPI_STATUSES_IGNORE);
    }
    
    // Copy received halos to device
    if (has_bottom) {
        CUDA_CHECK(cudaMemcpy((Real*)d_input + idx3(0, 0, 0, nx, ny),
                              h_recv_bottom.data(), plane_size * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
    if (has_top) {
        CUDA_CHECK(cudaMemcpy((Real*)d_input + idx3(0, 0, nz_local - 1, nx, ny),
                              h_recv_top.data(), plane_size * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }
    
    // Wait for sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, send_req, MPI_STATUSES_IGNORE);
    }
    
    // Launch CUDA kernel
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((nx - 2 + blockDim.x - 1) / blockDim.x,
                 (ny - 2 + blockDim.y - 1) / blockDim.y,
                 (nz_local - 2 + blockDim.z - 1) / blockDim.z);
    
    stencilKernel<<<gridDim, blockDim>>>(d_input, d_output, nx, ny, nz_local);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy boundaries
    const size_t total = nx * ny * nz_local;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    copyBoundariesKernel<<<blocks, threads>>>(d_input, d_output, nx, ny, nz_local);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
#else
// MPI+OpenMP stencil iteration with halo exchange (CPU only)
void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz_local,
                      std::vector<Real>& h_send_top, std::vector<Real>& h_send_bottom,
                      std::vector<Real>& h_recv_top, std::vector<Real>& h_recv_bottom,
                      const int rank, const int size, const size_t nz_global) {
    
    // Determine halo exchange needs
    const bool has_bottom = (rank > 0);
    const bool has_top = (rank < size - 1);
    
    const size_t plane_size = nx * ny;
    
    MPI_Request send_req[2], recv_req[2];
    int req_count = 0;
    
    // Post receives first
    if (has_bottom) {
        MPI_Irecv(h_recv_bottom.data(), plane_size, MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &recv_req[req_count++]);
    }
    if (has_top) {
        MPI_Irecv(h_recv_top.data(), plane_size, MPI_DOUBLE, rank + 1, 1,
                  MPI_COMM_WORLD, &recv_req[req_count++]);
    }
    
    // Prepare boundary planes for sending
    if (has_bottom) {
        #pragma omp parallel for
        for (size_t i = 0; i < plane_size; ++i) {
            h_send_bottom[i] = input[idx3(0, 0, 1, nx, ny) + i];
        }
    }
    if (has_top) {
        #pragma omp parallel for
        for (size_t i = 0; i < plane_size; ++i) {
            h_send_top[i] = input[idx3(0, 0, nz_local - 2, nx, ny) + i];
        }
    }
    
    // Send boundary planes
    int send_count = 0;
    if (has_bottom) {
        MPI_Isend(h_send_bottom.data(), plane_size, MPI_DOUBLE, rank - 1, 1,
                  MPI_COMM_WORLD, &send_req[send_count++]);
    }
    if (has_top) {
        MPI_Isend(h_send_top.data(), plane_size, MPI_DOUBLE, rank + 1, 0,
                  MPI_COMM_WORLD, &send_req[send_count++]);
    }
    
    // Wait for receives to complete
    if (req_count > 0) {
        MPI_Waitall(req_count, recv_req, MPI_STATUSES_IGNORE);
    }
    
    // Copy received halos
    if (has_bottom) {
        #pragma omp parallel for
        for (size_t i = 0; i < plane_size; ++i) {
            input[idx3(0, 0, 0, nx, ny) + i] = h_recv_bottom[i];
        }
    }
    if (has_top) {
        #pragma omp parallel for
        for (size_t i = 0; i < plane_size; ++i) {
            input[idx3(0, 0, nz_local - 1, nx, ny) + i] = h_recv_top[i];
        }
    }
    
    // Wait for sends to complete
    if (send_count > 0) {
        MPI_Waitall(send_count, send_req, MPI_STATUSES_IGNORE);
    }
    
    // Compute stencil on CPU
    stencilIterationCPU(input, output, nx, ny, nz_local);
}
#endif

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    bool valid = true;
    #pragma omp parallel for reduction(&& : valid)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            valid = false;
        }
    }
    
    if (!valid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
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
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));
#endif
    
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition in Z direction
    const size_t nz_per_rank = nz / size;
    const size_t nz_remainder = nz % size;
    
    size_t z_start = rank * nz_per_rank + std::min((size_t)rank, nz_remainder);
    size_t nz_local = nz_per_rank + (rank < (int)nz_remainder ? 1 : 0);
    
    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP");
#ifdef USE_CUDA
        printf("+CUDA)\n");
#else
        printf(")\n");
#endif
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_gridSize = nx * ny * nz_local;
    
    // Allocate halo buffers
    const size_t plane_size = nx * ny;
    std::vector<Real> h_send_top(plane_size);
    std::vector<Real> h_send_bottom(plane_size);
    std::vector<Real> h_recv_top(plane_size);
    std::vector<Real> h_recv_bottom(plane_size);
    
#ifdef USE_CUDA
    // Allocate host memory for local grids
    std::vector<Real> h_grid1(local_gridSize);
    std::vector<Real> h_grid2(local_gridSize);
    
    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_gridSize * sizeof(Real)));
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(h_grid1, nx, ny, nz_local, z_start, nz);
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid1.data(), local_gridSize * sizeof(Real),
                          cudaMemcpyHostToDevice));
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz_local,
                           h_send_top, h_send_bottom, h_recv_top, h_recv_bottom,
                           rank, size, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz_local,
                           h_send_top, h_send_bottom, h_recv_top, h_recv_bottom,
                           rank, size, nz);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy final result back to host
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real>& h_final = (iterations % 2 == 0) ? h_grid1 : h_grid2;
    CUDA_CHECK(cudaMemcpy(h_final.data(), d_final, local_gridSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    
    // Cleanup CUDA
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
#else
    // CPU-only path
    std::vector<Real> grid1(local_gridSize);
    std::vector<Real> grid2(local_gridSize);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz_local, z_start, nz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz_local,
                           h_send_top, h_send_bottom, h_recv_top, h_recv_bottom,
                           rank, size, nz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz_local,
                           h_send_top, h_send_bottom, h_recv_top, h_recv_bottom,
                           rank, size, nz);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::vector<Real>& h_final = (iterations % 2 == 0) ? grid1 : grid2;
#endif
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for validation and printing
    std::vector<Real> global_grid;
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        global_grid.resize(nx * ny * nz);
        for (int r = 0; r < size; ++r) {
            size_t nz_r = nz_per_rank + (r < (int)nz_remainder ? 1 : 0);
            size_t z_start_r = r * nz_per_rank + std::min((size_t)r, nz_remainder);
            recvcounts[r] = nx * ny * nz_r;
            displs[r] = nx * ny * z_start_r;
        }
    }
    
    MPI_Gatherv(h_final.data(), local_gridSize, MPI_DOUBLE,
                global_grid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
        print_results(global_grid, "Grid");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = false;
        if (rank == 0) {
            valid = validateResult(global_grid, nx, ny, nz);
        }
        MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
