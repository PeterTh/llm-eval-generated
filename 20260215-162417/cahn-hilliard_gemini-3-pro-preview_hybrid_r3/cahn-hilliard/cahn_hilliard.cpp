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
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
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

__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                               const size_t nx, const size_t ny, const size_t nz,
                               const double dx, const double dy, const double dz,
                               const double gamma, const double e_AA, const double e_BB, const double e_AB,
                               const size_t z_offset) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Adjust z index for ghost cells in c buffer if needed, 
    // but here we assume c includes ghost layers so z maps directly if we offset properly?
    // Actually, let's keep it simple: the kernel operates on the LOCAL domain (1 to nz-2 if using ghosts, or 0 to nz-1).
    // Let's assume the kernel is launched for the range [0, nz_local).
    // The input arrays `c` and `mu` will have ghost layers.
    // So valid range in `c` is [1, nz_local].
    // We compute `mu` at [1, nz_local].
    
    // Let's adopt convention: pointers point to START of allocation (including ghosts).
    // Valid data starts at z=1.
    // Kernel index z is from 0 to local_nz-1.
    // Actual index in array is z + 1.
    
    size_t idx = idx3(x, y, z + 1, nx, ny);
    double cv = c[idx];
    
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, nz + 2, dx, dy, dz, x, y, z + 1);
}

__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold,
                        const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    // Similar indexing as above
    size_t idx = idx3(x, y, z + 1, nx, ny);
    
    // Note: computeLaplacian expects full dimensions including ghosts
    cnew[idx] = cold[idx] + dt * D * 
               computeLaplacian(mu, nx, ny, nz + 2, dx, dy, dz, x, y, z + 1);
}

// Helper to copy ghost layers
// z_src: z index to copy FROM
// z_dst: z index to copy TO
// count: number of elements (nx * ny)
void exchangeGhosts(double* d_data, double* h_buf_send_top, double* h_buf_send_bot,
                    double* h_buf_recv_top, double* h_buf_recv_bot,
                    int rank, int size, int prev, int next,
                    size_t nx, size_t ny, size_t local_nz) {
    
    size_t slice_size = nx * ny;
    size_t slice_bytes = slice_size * sizeof(double);

    // Copy from Device to Host
    // Top inner layer (to send to prev): index 1
    CUDA_CHECK(cudaMemcpy(h_buf_send_top, d_data + 1 * slice_size, slice_bytes, cudaMemcpyDeviceToHost));
    // Bottom inner layer (to send to next): index local_nz
    CUDA_CHECK(cudaMemcpy(h_buf_send_bot, d_data + local_nz * slice_size, slice_bytes, cudaMemcpyDeviceToHost));

    // MPI Exchange
    MPI_Request reqs[4];
    MPI_Irecv(h_buf_recv_top, slice_size, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(h_buf_recv_bot, slice_size, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(h_buf_send_top, slice_size, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(h_buf_send_bot, slice_size, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Copy from Host to Device
    // Top ghost layer (received from prev): index 0
    if (rank != 0) // Only if we have a prev
        CUDA_CHECK(cudaMemcpy(d_data, h_buf_recv_top, slice_bytes, cudaMemcpyHostToDevice));
    else {
        // Boundary condition at global top (z=0): replicate z=0 to z=-1 (clamped/Neumann)
        // Since we are at rank 0, our "inner" z=0 is at index 1. Ghost is at index 0.
        // We replicate index 1 to index 0.
        CUDA_CHECK(cudaMemcpy(d_data, d_data + 1 * slice_size, slice_bytes, cudaMemcpyDeviceToDevice));
    }

    // Bottom ghost layer (received from next): index local_nz + 1
    if (rank != size - 1) // Only if we have a next
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * slice_size, h_buf_recv_bot, slice_bytes, cudaMemcpyHostToDevice));
    else {
        // Boundary condition at global bottom: replicate last inner to ghost
        CUDA_CHECK(cudaMemcpy(d_data + (local_nz + 1) * slice_size, d_data + local_nz * slice_size, slice_bytes, cudaMemcpyDeviceToDevice));
    }
}


// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    // Parallel initialization with OpenMP
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
        int devCount;
        cudaGetDeviceCount(&devCount);
        printf("CUDA Devices: %d\n", devCount);
    }

    // Set CUDA device based on rank (round-robin if fewer GPUs than ranks)
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);

    // Domain Decomposition
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * local_nz + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) local_nz++;
    
    // Allocate local arrays with ghost layers (top and bottom)
    size_t slice_size = nx * ny;
    size_t local_vol_with_ghosts = (local_nz + 2) * slice_size; // +2 for ghosts
    size_t local_vol_inner = local_nz * slice_size;

    std::vector<double> h_cold(local_vol_inner); // Only inner for initialization
    std::vector<double> h_full_cold(local_vol_with_ghosts); // For transfer

    // Initialize full global concentration on rank 0 or parallel init?
    // Let's do parallel init. We need global coordinates.
    // Initialize h_cold (inner part)
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = z_start + z;
                // Generate pseudo-random value
                const size_t vol = nx * ny * nz;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                h_cold[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Copy to full buffer (offset by 1 slice)
    std::copy(h_cold.begin(), h_cold.end(), h_full_cold.begin() + slice_size);

    // Allocate Device Memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_vol_with_ghosts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_vol_with_ghosts * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_vol_with_ghosts * sizeof(double)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, h_full_cold.data(), local_vol_with_ghosts * sizeof(double), cudaMemcpyHostToDevice));

    // Halo exchange buffers (Host)
    double *h_buf_send_top, *h_buf_send_bot, *h_buf_recv_top, *h_buf_recv_bot;
    CUDA_CHECK(cudaMallocHost(&h_buf_send_top, slice_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_buf_send_bot, slice_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_buf_recv_top, slice_size * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_buf_recv_bot, slice_size * sizeof(double)));

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
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (local_nz + block.z - 1) / block.z);

    int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    for (int t = 0; t < iterations; ++t) {
        // 1. Exchange ghosts for c (d_cold)
        exchangeGhosts(d_cold, h_buf_send_top, h_buf_send_bot, h_buf_recv_top, h_buf_recv_bot,
                       rank, size, prev, next, nx, ny, local_nz);

        // 2. Compute Chemical Potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz, dx, dy, dz, 
                                                        gamma, e_AA, e_BB, e_AB, z_start);
        CUDA_CHECK(cudaGetLastError());

        // 3. Exchange ghosts for mu
        exchangeGhosts(d_mu, h_buf_send_top, h_buf_send_bot, h_buf_recv_top, h_buf_recv_bot,
                       rank, size, prev, next, nx, ny, local_nz);

        // 4. Update Concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        
        // 5. Swap
        std::swap(d_cold, d_cnew);
    }
    
    cudaDeviceSynchronize();
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

    // Gather results if needed
    if (validate || printResults) {
        // Copy inner part back to host
        CUDA_CHECK(cudaMemcpy(h_full_cold.data(), d_cold, local_vol_with_ghosts * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Copy from h_full_cold (offset 1) to h_cold
        std::copy(h_full_cold.begin() + slice_size, h_full_cold.begin() + slice_size + local_vol_inner, h_cold.begin());
        
        // Gather all parts to rank 0
        std::vector<double> global_c;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        // Calculate counts and displacements
        int local_count = (int)local_vol_inner;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
            global_c.resize(nx * ny * nz);
        }

        MPI_Gatherv(h_cold.data(), local_count, MPI_DOUBLE, 
                    global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_c, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    cudaFreeHost(h_buf_send_top);
    cudaFreeHost(h_buf_send_bot);
    cudaFreeHost(h_buf_recv_top);
    cudaFreeHost(h_buf_recv_bot);

    MPI_Finalize();
    return 0;
}
