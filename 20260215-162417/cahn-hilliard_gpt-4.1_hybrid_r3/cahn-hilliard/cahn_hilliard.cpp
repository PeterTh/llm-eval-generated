#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__host__ __device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny, const size_t nz,
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

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < nx && y < ny && z < nz) {
        size_t idx = idx3(x, y, z, nx, ny);
        double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    double *d_c, *d_mu;
    size_t bytes = nx * ny * nz * sizeof(double);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_mu, bytes);
    cudaMemcpy(d_c, c.data(), bytes, cudaMemcpyHostToDevice);
    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);
    computeChemicalPotentialKernel<<<grid, block>>>(d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    cudaMemcpy(mu.data(), d_mu, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    cudaFree(d_mu);
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < nx && y < ny && z < nz) {
        size_t idx = idx3(x, y, z, nx, ny);
        cnew[idx] = cold[idx] + dt * D * computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along z
    size_t z_per_rank = nz / mpi_size;
    size_t z_start = mpi_rank * z_per_rank;
    size_t z_end = (mpi_rank == mpi_size - 1) ? nz : (mpi_rank + 1) * z_per_rank;
    size_t local_nz = z_end - z_start;
    size_t local_gridSize = nx * ny * local_nz;

    if (mpi_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", mpi_size);
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
    
    // Allocate arrays (with ghost layers)
    std::vector<double> cold(nx * ny * (local_nz + 2));
    std::vector<double> cnew(nx * ny * (local_nz + 2));
    std::vector<double> mu(nx * ny * (local_nz + 2));
    
    // Initialize concentration field (local only)
    if (mpi_rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz);
    
    // Run simulation
    if (mpi_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange ghost layers with neighbors
        if (mpi_size > 1) {
            // Send/recv lower ghost
            if (mpi_rank > 0) {
                MPI_Sendrecv(&cold[nx*ny], nx*ny, MPI_DOUBLE, mpi_rank-1, 0,
                             &cold[0], nx*ny, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // Send/recv upper ghost
            if (mpi_rank < mpi_size-1) {
                MPI_Sendrecv(&cold[nx*ny*local_nz], nx*ny, MPI_DOUBLE, mpi_rank+1, 1,
                             &cold[nx*ny*(local_nz+1)], nx*ny, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // Compute chemical potential (OpenMP+CUDA inside)
        computeChemicalPotential(cold, mu, nx, ny, local_nz+2, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        // Update concentration (OpenMP+CUDA inside)
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz+2, D, dt, dx, dy, dz);
        // Swap buffers
        std::swap(cold, cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (mpi_rank == 0) printf("Computation time: %ld ms\n", duration.count());
    // Gather results to rank 0 for output/validation
    std::vector<double> global_cold;
    if (mpi_rank == 0) global_cold.resize(nx*ny*nz);
    std::vector<int> recvcounts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t z0 = r * z_per_rank;
        size_t z1 = (r == mpi_size-1) ? nz : (r+1)*z_per_rank;
        recvcounts[r] = (z1-z0)*nx*ny;
        displs[r] = z0*nx*ny;
    }
    MPI_Gatherv(&cold[nx*ny], local_nz*nx*ny, MPI_DOUBLE,
                mpi_rank == 0 ? global_cold.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Calculate performance
    if (mpi_rank == 0) {
        double cellUpdates = (double)nx*ny*nz * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        print_results(global_cold, "Concentration");
    }
    // Validation
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(global_cold, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    MPI_Finalize();
    return 0;
}
