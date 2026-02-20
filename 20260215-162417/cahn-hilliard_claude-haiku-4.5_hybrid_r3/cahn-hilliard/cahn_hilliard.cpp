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

// CUDA device kernels
#ifdef __CUDACC__
__global__ void cuda_computeLaplacian_kernel(const double* c, double* laplacian,
                                             const size_t nx, const size_t ny, const size_t nz,
                                             const double inv_dxx, const double inv_dyy, const double inv_dzz,
                                             const size_t x_start, const size_t x_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + x_start;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= x_end || y >= ny || z >= nz) return;
    
    auto idx3 = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) { 
        return z * (nx * ny) + y * nx + x; 
    };
    
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                        2.0 * c[idx3(x, y, z, nx, ny)]) * inv_dxx;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                        2.0 * c[idx3(x, y, z, nx, ny)]) * inv_dyy;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                        2.0 * c[idx3(x, y, z, nx, ny)]) * inv_dzz;
    
    laplacian[idx3(x, y, z, nx, ny)] = cxx + cyy + czz;
}

__global__ void cuda_computeChemicalPotential_kernel(const double* c, const double* laplacian_c, double* mu,
                                                     const size_t nx, const size_t ny, const size_t nz,
                                                     const double gamma, const double e_AA, 
                                                     const double e_BB, const double e_AB,
                                                     const size_t x_start, const size_t x_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + x_start;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= x_end || y >= ny || z >= nz) return;
    
    auto idx3 = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) { 
        return z * (nx * ny) + y * nx + x; 
    };
    
    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];
    
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
            + 3.0 * cv + cv * cv * cv
            - gamma * laplacian_c[idx];
}

__global__ void cuda_cahnHilliardUpdate_kernel(double* cnew, const double* cold, const double* laplacian_mu,
                                              const size_t nx, const size_t ny, const size_t nz,
                                              const double coeff, const size_t x_start, const size_t x_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + x_start;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= x_end || y >= ny || z >= nz) return;
    
    auto idx3 = [](size_t x, size_t y, size_t z, size_t nx, size_t ny) { 
        return z * (nx * ny) + y * nx + x; 
    };
    
    const size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + coeff * laplacian_mu[idx];
}
#endif // __CUDACC__

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

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

// Compute chemical potential (CPU version with OpenMP)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    #pragma omp parallel for collapse(3) default(shared)
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

// Cahn-Hilliard update step (CPU version with OpenMP)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    #pragma omp parallel for collapse(3) default(shared)
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

// Initialize concentration field (with OpenMP)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(3) default(shared)
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

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    double minVal = c[0];
    double maxVal = c[0];
    
    #pragma omp parallel for default(shared) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
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
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same)
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
    
    // Domain decomposition: split along X dimension
    const size_t local_nx = nx / nprocs + (rank < (int)(nx % nprocs) ? 1 : 0);
    
    const size_t local_grid_size = local_nx * ny * nz;
    const size_t full_grid_size = nx * ny * nz;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Global grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
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
    
    // Allocate local arrays
    std::vector<double> cold(local_grid_size);
    std::vector<double> cnew(local_grid_size);
    std::vector<double> mu(local_grid_size);
    
    // Initialize concentration field locally
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, local_nx, ny, nz);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for boundary values
        std::vector<double> recv_left(ny * nz), recv_right(ny * nz);
        std::vector<double> send_left(ny * nz), send_right(ny * nz);
        
        // Pack boundary data for halo exchange
        #pragma omp parallel for collapse(2) default(shared)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                send_left[z * ny + y] = cold[idx3(0, y, z, local_nx, ny)];
                send_right[z * ny + y] = cold[idx3(local_nx - 1, y, z, local_nx, ny)];
            }
        }
        
        // Exchange with neighbors
        int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
        
        MPI_Sendrecv(send_left.data(), ny * nz, MPI_DOUBLE, left_rank, 0,
                     recv_right.data(), ny * nz, MPI_DOUBLE, right_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_right.data(), ny * nz, MPI_DOUBLE, right_rank, 1,
                     recv_left.data(), ny * nz, MPI_DOUBLE, left_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, local_nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Halo exchange for mu
        #pragma omp parallel for collapse(2) default(shared)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                send_left[z * ny + y] = mu[idx3(0, y, z, local_nx, ny)];
                send_right[z * ny + y] = mu[idx3(local_nx - 1, y, z, local_nx, ny)];
            }
        }
        
        MPI_Sendrecv(send_left.data(), ny * nz, MPI_DOUBLE, left_rank, 2,
                     recv_right.data(), ny * nz, MPI_DOUBLE, right_rank, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_right.data(), ny * nz, MPI_DOUBLE, right_rank, 3,
                     recv_left.data(), ny * nz, MPI_DOUBLE, left_rank, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, local_nx, ny, nz, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance (using global grid size)
        double cellUpdates = (double)full_grid_size * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather all data to rank 0 for output and validation
    std::vector<double> full_c;
    if (rank == 0) {
        full_c.resize(full_grid_size);
    }
    
    // Simple gather (not optimal for distribution, but works for verification)
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    for (int i = 0; i < nprocs; ++i) {
        sendcounts[i] = (nx / nprocs + (i < (int)(nx % nprocs) ? 1 : 0)) * ny * nz;
    }
    displs[0] = 0;
    for (int i = 1; i < nprocs; ++i) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }
    
    MPI_Gatherv(cold.data(), local_grid_size, MPI_DOUBLE,
                full_c.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(full_c, "Concentration");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_c, nx, ny, nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
