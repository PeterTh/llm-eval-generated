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

// CUDA Error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Device inline functions
__device__ inline size_t idx3_d(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with ghost cells handling boundary conditions
// For local domain, z goes from 0 to local_nz-1.
// Ghost cells are at z=-1 (index in buffer: 0) and z=local_nz (index in buffer: local_nz+1).
// Real data starts at index 1 in Z dimension of buffer.
// So buffer index for local (x,y,z) is (z+1)*nx*ny + y*nx + x.
__device__ double computeLaplacian_d(const double* __restrict__ c, const size_t nx, const size_t ny, 
                                   const double dx, const double dy, const double dz, 
                                   const size_t x, const size_t y, const size_t z_local) {
    
    // Boundary conditions for X and Y are clamped (global domain boundaries)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    // For Z, we use the buffer which includes ghosts.
    // z_local is 0..nz_local-1.
    // The buffer has size (nz_local + 2) * nx * ny.
    // Plane z_local corresponds to buffer plane z_local + 1.
    // Neighbors are simply +1 and -1 in Z index of buffer.
    
    const size_t z_buf = z_local + 1;
    
    // Indices in the buffer
    const size_t idx_c = idx3_d(x, y, z_buf, nx, ny);
    const size_t idx_xp = idx3_d(xp, y, z_buf, nx, ny);
    const size_t idx_xn = idx3_d(xn, y, z_buf, nx, ny);
    const size_t idx_yp = idx3_d(x, yp, z_buf, nx, ny);
    const size_t idx_yn = idx3_d(x, yn, z_buf, nx, ny);
    const size_t idx_zp = idx3_d(x, y, z_buf + 1, nx, ny);
    const size_t idx_zn = idx3_d(x, y, z_buf - 1, nx, ny);
    
    const double val = c[idx_c];
    
    const double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * val) / (dx * dx);
    const double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * val) / (dy * dy);
    const double czz = (c[idx_zp] + c[idx_zn] - 2.0 * val) / (dz * dz);
    
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotential_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                              const size_t nx, const size_t ny, const size_t nz_local,
                                              const double dx, const double dy, const double dz,
                                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz_local) return;
    
    const size_t idx_buf = idx3_d(x, y, z + 1, nx, ny); // Write to real part of mu buffer
    const double cv = c[idx_buf]; // Read from real part of c buffer
    
    mu[idx_buf] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacian_d(c, nx, ny, dx, dy, dz, x, y, z);
}

__global__ void cahnHilliardUpdate_kernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                        const double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nz_local,
                                        const double D, const double dt, 
                                        const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz_local) return;
    
    const size_t idx_buf = idx3_d(x, y, z + 1, nx, ny);
    
    cnew[idx_buf] = cold[idx_buf] + dt * D * 
                   computeLaplacian_d(mu, nx, ny, dx, dy, dz, x, y, z);
}

__global__ void initialize_kernel(double* c, const size_t nx, const size_t ny, const size_t nz_global, 
                                const size_t z_offset, const size_t nz_local) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz_local) return;
    
    const size_t idx_buf = idx3_d(x, y, z + 1, nx, ny);
    const size_t z_global = z_offset + z;
    const size_t vol = nx * ny * nz_global;
    
    const size_t linear_id = z_global * (nx * ny) + y * nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx_buf] = -1.0 + 2.0 * pseudo;
}

// Helper to fill boundary ghosts (clamped)
__global__ void fill_boundary_ghosts(double* c, size_t nx, size_t ny, size_t nz_local, bool is_first, bool is_last) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) return;
    
    // Bottom ghost (z_buf = 0) copies from z_buf = 1
    if (is_first) {
        c[idx3_d(x, y, 0, nx, ny)] = c[idx3_d(x, y, 1, nx, ny)];
    }
    
    // Top ghost (z_buf = nz_local + 1) copies from z_buf = nz_local
    if (is_last) {
        c[idx3_d(x, y, nz_local + 1, nx, ny)] = c[idx3_d(x, y, nz_local, nx, ny)];
    }
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
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Select GPU based on rank (round-robin if fewer GPUs than ranks on node)
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    
    // Use OpenMP to set number of threads (hybrid approach requirement)
    // Also helps with memory pinning performance if multiple threads used
    #pragma omp parallel
    {
        #pragma omp single
        {
           // Just to initialize OpenMP runtime if needed
        }
    }
    
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int node_rank;
    MPI_Comm_rank(nodeComm, &node_rank);
    MPI_Comm_free(&nodeComm);
    
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(node_rank % num_devices));
    } else {
        if(rank==0) fprintf(stderr, "No CUDA devices found!\n");
        MPI_Finalize();
        return 1;
    }

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
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
                // Need clean exit in MPI
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }
    
    // Broadcast parameters
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain Decomposition (1D Slab in Z)
    size_t nz_per_rank = nz / size;
    size_t remainder = nz % size;
    size_t z_offset = rank * nz_per_rank + std::min((size_t)rank, remainder);
    size_t nz_local = nz_per_rank + (rank < (int)remainder ? 1 : 0);
    
    // Allocate device memory (includes 2 ghost layers in Z)
    // Buffer size: nx * ny * (nz_local + 2)
    size_t plane_size = nx * ny;
    size_t buffer_elements = plane_size * (nz_local + 2);
    size_t buffer_bytes = buffer_elements * sizeof(double);
    
    double *d_c, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_c, buffer_bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, buffer_bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, buffer_bytes));
    
    // Host buffers for Halo Exchange
    double *h_send_top, *h_recv_top, *h_send_bot, *h_recv_bot;
    // Use pinned memory for faster transfer
    CUDA_CHECK(cudaMallocHost(&h_send_top, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_bot, plane_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bot, plane_size * sizeof(double)));
    
    // Initialize
    if (rank == 0) printf("Initializing concentration field...\n");
    
    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz_local + block.z - 1) / block.z);
    
    initialize_kernel<<<grid, block>>>(d_c, nx, ny, nz, z_offset, nz_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Constants
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Boundary config
    bool is_first = (rank == 0);
    bool is_last = (rank == size - 1);
    
    dim3 grid_boundary((nx + 15) / 16, (ny + 15) / 16);
    dim3 block_boundary(16, 16);
    
    // Pre-create MPI requests array
    MPI_Request reqs[4];

    for (int t = 0; t < iterations; ++t) {
        
        // --- Step 1: Compute Chemical Potential ---
        
        // 1a. Exchange Halos for 'c'
        // Fill physical boundary ghosts (top/bottom of global domain)
        fill_boundary_ghosts<<<grid_boundary, block_boundary>>>(d_c, nx, ny, nz_local, is_first, is_last);
        
        // MPI Exchange
        int top_nbr = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
        int bot_nbr = (rank == 0) ? MPI_PROC_NULL : rank - 1;
        
        // Pack buffers (Device to Host)
        // Bottom real plane is at index 1 in buffer
        CUDA_CHECK(cudaMemcpyAsync(h_send_bot, d_c + plane_size, plane_size * sizeof(double), cudaMemcpyDeviceToHost)); 
        // Top real plane is at index nz_local in buffer
        CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_c + plane_size * nz_local, plane_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for copy
        
        MPI_Isend(h_send_bot, plane_size, MPI_DOUBLE, bot_nbr, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_recv_bot, plane_size, MPI_DOUBLE, bot_nbr, 0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(h_send_top, plane_size, MPI_DOUBLE, top_nbr, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(h_recv_top, plane_size, MPI_DOUBLE, top_nbr, 0, MPI_COMM_WORLD, &reqs[3]);
        
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        
        // Unpack buffers (Host to Device)
        if (bot_nbr != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_c, h_recv_bot, plane_size * sizeof(double), cudaMemcpyHostToDevice)); // To index 0
        }
        if (top_nbr != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_c + plane_size * (nz_local + 1), h_recv_top, plane_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 1b. Compute MU
        computeChemicalPotential_kernel<<<grid, block>>>(d_c, d_mu, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        
        // --- Step 2: Update Concentration ---
        
        // 2a. Exchange Halos for 'mu'
        fill_boundary_ghosts<<<grid_boundary, block_boundary>>>(d_mu, nx, ny, nz_local, is_first, is_last);
        
        CUDA_CHECK(cudaMemcpyAsync(h_send_bot, d_mu + plane_size, plane_size * sizeof(double), cudaMemcpyDeviceToHost)); 
        CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_mu + plane_size * nz_local, plane_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        MPI_Isend(h_send_bot, plane_size, MPI_DOUBLE, bot_nbr, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_recv_bot, plane_size, MPI_DOUBLE, bot_nbr, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(h_send_top, plane_size, MPI_DOUBLE, top_nbr, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(h_recv_top, plane_size, MPI_DOUBLE, top_nbr, 1, MPI_COMM_WORLD, &reqs[3]);
        
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        
        if (bot_nbr != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_mu, h_recv_bot, plane_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (top_nbr != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_mu + plane_size * (nz_local + 1), h_recv_top, plane_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 2b. Compute C_NEW
        cahnHilliardUpdate_kernel<<<grid, block>>>(d_cnew, d_c, d_mu, nx, ny, nz_local, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap buffers
        std::swap(d_c, d_cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        size_t gridSize = nx * ny * nz;
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    std::vector<double> full_c;
    if (rank == 0) {
        full_c.resize(nx * ny * nz);
    }
    
    // Copy real data from device (skip bottom ghost)
    std::vector<double> h_c_local(plane_size * nz_local);
    CUDA_CHECK(cudaMemcpy(h_c_local.data(), d_c + plane_size, plane_size * nz_local * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Gather sizes
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    int local_count = (int)(plane_size * nz_local);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(h_c_local.data(), local_count, MPI_DOUBLE, 
                full_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (printResults) {
            print_results(full_c, "Concentration");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            // Validate locally on host
             // Check for NaN or Inf
            for (const auto& val : full_c) {
                if (std::isnan(val) || std::isinf(val)) {
                    printf("Validation failed: found NaN or Inf value\n");
                    valid = false;
                    break;
                }
            }
            if (valid) {
                double minVal = full_c[0];
                double maxVal = full_c[0];
                for (const auto& val : full_c) {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFreeHost(h_send_bot));
    CUDA_CHECK(cudaFreeHost(h_recv_bot));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    MPI_Finalize();
    return 0;
}
