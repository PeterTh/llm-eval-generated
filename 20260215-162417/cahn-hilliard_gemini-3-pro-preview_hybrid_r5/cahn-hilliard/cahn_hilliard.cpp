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
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Device: Compute Laplacian
__device__ double computeLaplacianDevice(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz_local,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    // z is local index in [1, nz_local] (including ghost layers 0 and nz_local+1)
    
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1; // Always valid due to ghost layers
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1; // Always valid due to ghost layers

    const double c_center = c[idx3(x, y, z, nx, ny)];

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * c_center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * c_center) / (dy * dy);
    // Note: for z, we rely on ghost layers being populated correctly
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * c_center) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Kernel: Compute Chemical Potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    // 3D grid: x, y map to threads, z loops? or 3D block?
    // Let's do simple 3D grid
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z_comput = blockIdx.z * blockDim.z + threadIdx.z; // 0 to nz_local-1

    if (x < nx && y < ny && z_comput < nz_local) {
        // Shift z by 1 to account for ghost layer
        const size_t z = z_comput + 1;
        const size_t idx = idx3(x, y, z, nx, ny);
        const double cv = c[idx];
        
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDevice(c, nx, ny, nz_local, dx, dy, dz, x, y, z);
    }
}

// Kernel: Cahn-Hilliard Update
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z_comput = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z_comput < nz_local) {
        const size_t z = z_comput + 1;
        const size_t idx = idx3(x, y, z, nx, ny);
        
        // Laplacian of mu
        cnew[idx] = cold[idx] + dt * D * 
                   computeLaplacianDevice(mu, nx, ny, nz_local, dx, dy, dz, x, y, z);
    }
}

// Initialize concentration field (Host)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_total, 
                           const size_t z_start_global, const size_t nz_local) {
    const size_t vol = nx * ny * nz_total;
    
    #pragma omp parallel for collapse(3)
    for (size_t z_local = 0; z_local < nz_local; ++z_local) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Local index in buffer (offset by 1 for ghost layer)
                const size_t idx = idx3(x, y, z_local + 1, nx, ny);
                
                // Global Z coordinate
                const size_t z_global = z_start_global + z_local;
                
                // Global linear ID for consistent random seed
                const size_t linear_id = z_global * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Host validation
bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local) {
    bool valid = true;
    double minVal = 1e30;
    double maxVal = -1e30;
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:valid)
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c[idx3(x, y, z, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    valid = false;
                }
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
        }
    }
    
    if (!valid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        // Only print on rank 0 usually, but validation is local here. 
        // We'll let caller handle aggregation or just print local failure.
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
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain Decomposition
    size_t nz_local = nz / size;
    size_t remainder = nz % size;
    size_t z_start_global = rank * nz_local + std::min((size_t)rank, remainder);
    if (rank < (int)remainder) {
        nz_local++;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+CUDA+OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI Ranks: %d\n", size);
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
    
    // Allocate Host arrays with ghost layers (top and bottom)
    // Local grid size: nx * ny * (nz_local + 2)
    // Indices: 0 (ghost), 1..nz_local (real), nz_local+1 (ghost)
    size_t planeSize = nx * ny;
    size_t localGridSize = planeSize * (nz_local + 2);
    
    std::vector<double> h_cold(localGridSize, 0.0);
    std::vector<double> h_cnew(localGridSize, 0.0);
    std::vector<double> h_mu(localGridSize, 0.0);
    
    // Initialize concentration field on Host
    if (rank == 0) {
        printf("Initializing concentration field...\n");
        #ifdef _OPENMP
        printf("Using OpenMP with max %d threads\n", omp_get_max_threads());
        #endif
    }
    initializeConcentration(h_cold, nx, ny, nz, z_start_global, nz_local);

    // CUDA Initialization
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, localGridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localGridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localGridSize * sizeof(double)));

    // Copy to Device
    CUDA_CHECK(cudaMemcpy(d_cold, h_cold.data(), localGridSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localGridSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localGridSize * sizeof(double)));

    // Prepare for exchange
    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    
    // Buffers for ghost exchange (Host side)
    // We need to exchange 1 plane
    std::vector<double> send_top(planeSize);
    std::vector<double> send_bottom(planeSize);
    std::vector<double> recv_top(planeSize);
    std::vector<double> recv_bottom(planeSize);

    // Boundary conditions for domain boundaries (non-periodic Z)
    // If rank 0, top ghost (idx 0) should duplicate idx 1 (Neumann)
    // If rank size-1, bottom ghost (idx nz_local+1) should duplicate idx nz_local (Neumann)
    // We will handle this by copying on device or host before/after exchange

    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz_local + block.z - 1) / block.z);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // --- Exchange Ghost Layers for 'cold' ---
        
        // 1. Copy boundary planes from Device to Host
        // Send to Top: Plane 1 (real data start)
        CUDA_CHECK(cudaMemcpy(send_top.data(), d_cold + planeSize, planeSize * sizeof(double), cudaMemcpyDeviceToHost));
        // Send to Bottom: Plane nz_local (real data end)
        CUDA_CHECK(cudaMemcpy(send_bottom.data(), d_cold + nz_local * planeSize, planeSize * sizeof(double), cudaMemcpyDeviceToHost));
        
        // 2. MPI Exchange
        MPI_Request reqs[4];
        MPI_Isend(send_top.data(), planeSize, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(send_bottom.data(), planeSize, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(recv_top.data(), planeSize, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_bottom.data(), planeSize, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // 3. Apply Boundary Conditions (Neumann at domain edges)
        if (rank == 0) {
            // No top neighbor: ghost 0 = plane 1
            std::memcpy(recv_top.data(), send_top.data(), planeSize * sizeof(double));
        }
        if (rank == size - 1) {
            // No bottom neighbor: ghost nz_local+1 = plane nz_local
            std::memcpy(recv_bottom.data(), send_bottom.data(), planeSize * sizeof(double));
        }

        // 4. Copy ghosts from Host to Device
        // Recv from Top goes to Plane 0 (ghost)
        CUDA_CHECK(cudaMemcpy(d_cold, recv_top.data(), planeSize * sizeof(double), cudaMemcpyHostToDevice));
        // Recv from Bottom goes to Plane nz_local+1 (ghost)
        CUDA_CHECK(cudaMemcpy(d_cold + (nz_local + 1) * planeSize, recv_bottom.data(), planeSize * sizeof(double), cudaMemcpyHostToDevice));

        // --- Compute Chemical Potential ---
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // --- Exchange Ghost Layers for 'mu' ---
        // Same process for mu
        CUDA_CHECK(cudaMemcpy(send_top.data(), d_mu + planeSize, planeSize * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(send_bottom.data(), d_mu + nz_local * planeSize, planeSize * sizeof(double), cudaMemcpyDeviceToHost));
        
        MPI_Isend(send_top.data(), planeSize, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(send_bottom.data(), planeSize, MPI_DOUBLE, bottom_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(recv_top.data(), planeSize, MPI_DOUBLE, top_neighbor, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(recv_bottom.data(), planeSize, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (rank == 0) std::memcpy(recv_top.data(), send_top.data(), planeSize * sizeof(double));
        if (rank == size - 1) std::memcpy(recv_bottom.data(), send_bottom.data(), planeSize * sizeof(double));

        CUDA_CHECK(cudaMemcpy(d_mu, recv_top.data(), planeSize * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mu + (nz_local + 1) * planeSize, recv_bottom.data(), planeSize * sizeof(double), cudaMemcpyHostToDevice));

        // --- Update Concentration ---
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz_local, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        
        // Swap pointers
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_cold.data(), d_cold, localGridSize * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather max duration across all ranks for performance reporting
    long local_duration = duration.count();
    long max_duration;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration);
        
        // Calculate performance
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / (max_duration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation (Global gather required if printing full field, 
    // but usually only rank 0 prints or we output per-rank. 
    // The original code printed all. Let's gather to rank 0 for simplicity if small enough, 
    // but typically benchmarks shouldn't print large results. 
    // The original supported -r. Let's support it via gather.)
    
    if (printResults) {
        // Collect all data to rank 0 (simple approach for verification)
        std::vector<double> global_c;
        if (rank == 0) global_c.resize(nx * ny * nz);
        
        // Remove ghost layers before gathering
        std::vector<double> local_real(nx * ny * nz_local);
        #pragma omp parallel for
        for (size_t z = 0; z < nz_local; ++z) {
             std::memcpy(local_real.data() + z * planeSize, 
                         h_cold.data() + (z + 1) * planeSize, 
                         planeSize * sizeof(double));
        }

        // Use Gatherv because nz_local might vary
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = nx * ny * nz_local;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(local_real.data(), local_count, MPI_DOUBLE, 
                    global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(global_c, "Concentration");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        // Validate locally
        bool local_valid = validateResult(h_cold, nx, ny, nz_local);
        
        // Global reduction
        int global_valid_int;
        int local_valid_int = local_valid ? 1 : 0;
        MPI_Allreduce(&local_valid_int, &global_valid_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Range check also needs global min/max.
            // But let's stick to the boolean result for now as per original code structure
            if (global_valid_int) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        if (!global_valid_int) {
            MPI_Finalize();
            return 1;
        }
    }
    
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    
    MPI_Finalize();
    return 0;
}
