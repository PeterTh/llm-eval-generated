#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(error)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) call
#endif

// CUDA kernels
#ifdef __CUDACC__
__global__ void computeLaplacianKernel(const double* c, double* lap, 
                                       size_t nx, size_t ny, size_t nz,
                                       double dx, double dy, double dz,
                                       size_t z_start, size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (x < nx && y < ny && z < z_end) {
        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t zp = (z < nz - 1) ? z + 1 : z;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yn = (y > 0) ? y - 1 : 0;
        size_t zn = (z > 0) ? z - 1 : 0;
        
        auto idx3_cuda = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) {
            return z * (nx * ny) + y * nx + x;
        };
        
        size_t idx = idx3_cuda(x, y, z, nx, ny);
        double cxx = (c[idx3_cuda(xp, y, z, nx, ny)] + c[idx3_cuda(xn, y, z, nx, ny)] - 
                     2.0 * c[idx]) / (dx * dx);
        double cyy = (c[idx3_cuda(x, yp, z, nx, ny)] + c[idx3_cuda(x, yn, z, nx, ny)] - 
                     2.0 * c[idx]) / (dy * dy);
        double czz = (c[idx3_cuda(x, y, zp, nx, ny)] + c[idx3_cuda(x, y, zn, nx, ny)] - 
                     2.0 * c[idx]) / (dz * dz);
        
        lap[idx] = cxx + cyy + czz;
    }
}

__global__ void computeChemicalPotentialKernel(const double* c, double* mu, const double* lap_c,
                                               size_t nx, size_t ny, size_t nz,
                                               double gamma, double e_AA, double e_BB, double e_AB,
                                               size_t z_start, size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (x < nx && y < ny && z < z_end) {
        auto idx3_cuda = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) {
            return z * (nx * ny) + y * nx + x;
        };
        
        size_t idx = idx3_cuda(x, y, z, nx, ny);
        double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                 + 3.0 * cv + cv * cv * cv - gamma * lap_c[idx];
    }
}

__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* lap_mu,
                                        size_t nx, size_t ny, size_t nz,
                                        double D, double dt,
                                        size_t z_start, size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;
    
    if (x < nx && y < ny && z < z_end) {
        auto idx3_cuda = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) {
            return z * (nx * ny) + y * nx + x;
        };
        
        size_t idx = idx3_cuda(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * lap_mu[idx];
    }
}
#endif

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Helper to get MPI rank and size
struct MPIInfo {
    int rank;
    int size;
    size_t z_start;
    size_t z_end;
};

MPIInfo getMPIInfo(size_t nz) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t z_per_rank = nz / size;
    size_t z_remainder = nz % size;
    
    size_t z_start = rank * z_per_rank + std::min((size_t)rank, z_remainder);
    size_t z_end = z_start + z_per_rank + (rank < (int)z_remainder ? 1 : 0);
    
    return {rank, size, z_start, z_end};
}

// MPI halo exchange for ghost cells (1 layer in z direction)
void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny, const size_t nz,
                  const MPIInfo& mpi) {
    if (mpi.size == 1) return;  // No need for exchange in serial mode
    
    int rank_prev = (mpi.rank - 1 + mpi.size) % mpi.size;
    int rank_next = (mpi.rank + 1) % mpi.size;
    
    // Create temporary buffers for halo data
    std::vector<double> send_bottom(nx * ny);
    std::vector<double> send_top(nx * ny);
    std::vector<double> recv_bottom(nx * ny);
    std::vector<double> recv_top(nx * ny);
    
    // Prepare bottom (z_start) and top (z_end-1) layers for sending
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            send_bottom[y * nx + x] = field[idx3(x, y, mpi.z_start, nx, ny)];
            send_top[y * nx + x] = field[idx3(x, y, mpi.z_end - 1, nx, ny)];
        }
    }
    
    // Send/receive with previous and next neighbors
    MPI_Request reqs[4];
    MPI_Isend(send_bottom.data(), nx * ny, MPI_DOUBLE, rank_prev, 1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(send_top.data(), nx * ny, MPI_DOUBLE, rank_next, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(recv_top.data(), nx * ny, MPI_DOUBLE, rank_prev, 0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(recv_bottom.data(), nx * ny, MPI_DOUBLE, rank_next, 1, MPI_COMM_WORLD, &reqs[3]);
    
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    
    // Update ghost cells if not at boundaries
    if (mpi.rank > 0) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                field[idx3(x, y, mpi.z_start - 1, nx, ny)] = recv_top[y * nx + x];
            }
        }
    }
    if (mpi.rank < mpi.size - 1) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                field[idx3(x, y, mpi.z_end, nx, ny)] = recv_bottom[y * nx + x];
            }
        }
    }
}

// Compute Laplacian with clamped boundary conditions
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

// Compute Laplacian field (OpenMP parallelized)
void computeLaplacianField(const std::vector<double>& c, std::vector<double>& lap,
                          const size_t nx, const size_t ny, const size_t nz,
                          const double dx, const double dy, const double dz,
                          const MPIInfo& mpi) {
#ifdef __CUDACC__
    double *d_c, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_c, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lap, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 threads(8, 8, 4);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (mpi.z_end - mpi.z_start + 3) / 4);
    computeLaplacianKernel<<<blocks, threads>>>(d_c, d_lap, nx, ny, nz, dx, dy, dz, mpi.z_start, mpi.z_end);
    
    CUDA_CHECK(cudaMemcpy(lap.data(), d_lap, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_lap));
#else
    #pragma omp parallel for collapse(3)
    for (size_t z = mpi.z_start; z < mpi.z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                lap[idx3(x, y, z, nx, ny)] = computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
#endif
}

// Compute chemical potential (OpenMP + CUDA parallelized)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const MPIInfo& mpi) {
    std::vector<double> lap_c(nx * ny * nz);
    computeLaplacianField(c, lap_c, nx, ny, nz, dx, dy, dz, mpi);
    
#ifdef __CUDACC__
    double *d_c, *d_mu, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_c, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lap, nx * ny * nz * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lap, lap_c.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 threads(8, 8, 4);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (mpi.z_end - mpi.z_start + 3) / 4);
    computeChemicalPotentialKernel<<<blocks, threads>>>(d_c, d_mu, d_lap, nx, ny, nz, gamma, e_AA, e_BB, e_AB, mpi.z_start, mpi.z_end);
    
    CUDA_CHECK(cudaMemcpy(mu.data(), d_mu, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_lap));
#else
    #pragma omp parallel for collapse(3)
    for (size_t z = mpi.z_start; z < mpi.z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv - gamma * lap_c[idx];
            }
        }
    }
#endif
}

// Cahn-Hilliard update step (OpenMP + CUDA parallelized)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const MPIInfo& mpi) {
    std::vector<double> lap_mu(nx * ny * nz);
    computeLaplacianField(mu, lap_mu, nx, ny, nz, dx, dy, dz, mpi);
    
#ifdef __CUDACC__
    double *d_cnew, *d_cold, *d_lap;
    CUDA_CHECK(cudaMalloc(&d_cnew, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cold, nx * ny * nz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_lap, nx * ny * nz * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lap, lap_mu.data(), nx * ny * nz * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 threads(8, 8, 4);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (mpi.z_end - mpi.z_start + 3) / 4);
    cahnHilliardUpdateKernel<<<blocks, threads>>>(d_cnew, d_cold, d_lap, nx, ny, nz, D, dt, mpi.z_start, mpi.z_end);
    
    CUDA_CHECK(cudaMemcpy(cnew.data(), d_cnew, nx * ny * nz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_lap));
#else
    #pragma omp parallel for collapse(3)
    for (size_t z = mpi.z_start; z < mpi.z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * lap_mu[idx];
            }
        }
    }
#endif
}

// Initialize concentration field (OpenMP parallelized)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const MPIInfo& mpi) {
    const size_t vol = nx * ny * nz;
    
    // Initialize full grid on all processes (needed for semantic equivalence)
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz,
                    const MPIInfo& mpi) {
    double minVal = c[0];
    double maxVal = c[0];
    bool hasNanInf = false;
    
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(||:hasNanInf)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            hasNanInf = true;
        }
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    
    if (hasNanInf) {
        if (mpi.rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }
    
    if (mpi.rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
        
        if (maxVal > 10.0 || minVal < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
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
            MPI_Finalize();
            return 0;
        } else {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    
    MPIInfo mpi = getMPIInfo(nz);
    
    if (mpi.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", mpi.size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
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
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    if (mpi.rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, nx, ny, nz, mpi);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    if (mpi.rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for boundary values (needed for correct Laplacian at subdomain boundaries)
        exchangeHalos(cold, nx, ny, nz, mpi);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, mpi);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz, mpi);
        
        // Swap buffers
        std::swap(cold, cnew);
        
        // Synchronize across MPI ranks
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi.rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Allgather all data for results printing and validation to ensure semantic equivalence
    std::vector<double> cold_full(gridSize);
    
    std::vector<int> sendcounts(mpi.size);
    std::vector<int> displs(mpi.size);
    
    for (int i = 0; i < mpi.size; ++i) {
        size_t z_per_rank = nz / mpi.size;
        size_t z_remainder = nz % mpi.size;
        size_t z_start_i = i * z_per_rank + std::min((size_t)i, z_remainder);
        size_t z_end_i = z_start_i + z_per_rank + (i < (int)z_remainder ? 1 : 0);
        size_t local_nz = z_end_i - z_start_i;
        
        sendcounts[i] = nx * ny * local_nz;
        displs[i] = z_start_i * nx * ny;
    }
    
    // Send only the computed portion of cold from this rank
    MPI_Allgatherv(cold.data() + displs[mpi.rank], sendcounts[mpi.rank], MPI_DOUBLE,
                   cold_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (printResults && mpi.rank == 0) {
        print_results(cold_full, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (mpi.rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateResult(cold_full, nx, ny, nz, mpi);
        
        if (mpi.rank == 0) {
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
