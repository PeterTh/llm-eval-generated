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

#ifndef DISABLE_CUDA
#include <cuda_runtime.h>

// CUDA kernel for computing Laplacian
__global__ void laplacian_kernel(const double* c, double* lap, 
                                  size_t nx, size_t ny, size_t nz,
                                  double dx_inv2, double dy_inv2, double dz_inv2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    
    if (idx >= total) return;
    
    size_t z = idx / (ny * nx);
    size_t remainder = idx % (ny * nx);
    size_t y = remainder / nx;
    size_t x = remainder % nx;
    
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;
    
    size_t idx_c = z * (nx * ny) + y * nx + x;
    size_t idx_xp = z * (nx * ny) + y * nx + xp;
    size_t idx_xn = z * (nx * ny) + y * nx + xn;
    size_t idx_yp = z * (nx * ny) + yp * nx + x;
    size_t idx_yn = z * (nx * ny) + yn * nx + x;
    size_t idx_zp = zp * (nx * ny) + y * nx + x;
    size_t idx_zn = zn * (nx * ny) + y * nx + x;
    
    double cxx = (c[idx_xp] + c[idx_xn] - 2.0 * c[idx_c]) * dx_inv2;
    double cyy = (c[idx_yp] + c[idx_yn] - 2.0 * c[idx_c]) * dy_inv2;
    double czz = (c[idx_zp] + c[idx_zn] - 2.0 * c[idx_c]) * dz_inv2;
    
    lap[idx_c] = cxx + cyy + czz;
}

// CUDA kernel for chemical potential
__global__ void chem_potential_kernel(const double* c, double* mu, const double* lap,
                                       size_t nx, size_t ny, size_t nz,
                                       double gamma, double e_AA, double e_BB, double e_AB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    
    if (idx >= total) return;
    
    double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
            + 3.0 * cv + cv * cv * cv - gamma * lap[idx];
}

// CUDA kernel for Cahn-Hilliard update
__global__ void update_kernel(double* cnew, const double* cold, const double* lap,
                               size_t nx, size_t ny, size_t nz,
                               double D, double dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    
    if (idx >= total) return;
    
    cnew[idx] = cold[idx] + dt * D * lap[idx];
}

// Compute Laplacian on GPU
void computeLaplacianGPU(const double* d_c, double* d_lap, 
                         size_t nx, size_t ny, size_t nz,
                         double dx, double dy, double dz) {
    double dx_inv2 = 1.0 / (dx * dx);
    double dy_inv2 = 1.0 / (dy * dy);
    double dz_inv2 = 1.0 / (dz * dz);
    
    size_t total = nx * ny * nz;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;
    
    laplacian_kernel<<<numBlocks, blockSize>>>(d_c, d_lap, nx, ny, nz, 
                                                dx_inv2, dy_inv2, dz_inv2);
    cudaDeviceSynchronize();
}

// Compute chemical potential with GPU acceleration
void computeChemicalPotentialGPU(const std::vector<double>& c, std::vector<double>& mu,
                                 const size_t nx, const size_t ny, const size_t nz,
                                 const double dx, const double dy, const double dz,
                                 const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                 double* d_c, double* d_mu, double* d_lap) {
    size_t gridSize = nx * ny * nz;
    cudaMemcpy(d_c, c.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice);
    
    computeLaplacianGPU(d_c, d_lap, nx, ny, nz, dx, dy, dz);
    
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    chem_potential_kernel<<<numBlocks, blockSize>>>(d_c, d_mu, d_lap, nx, ny, nz,
                                                      gamma, e_AA, e_BB, e_AB);
    cudaDeviceSynchronize();
    
    cudaMemcpy(mu.data(), d_mu, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
}

// Cahn-Hilliard update step with GPU acceleration
void cahnHilliardUpdateGPU(std::vector<double>& cnew, const std::vector<double>& cold,
                           const std::vector<double>& mu,
                           const size_t nx, const size_t ny, const size_t nz,
                           const double D, const double dt, const double dx, const double dy, const double dz,
                           double* d_cnew, double* d_cold, double* d_mu, double* d_lap) {
    size_t gridSize = nx * ny * nz;
    cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, mu.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice);
    
    computeLaplacianGPU(d_mu, d_lap, nx, ny, nz, dx, dy, dz);
    
    int blockSize = 256;
    int numBlocks = (gridSize + blockSize - 1) / blockSize;
    update_kernel<<<numBlocks, blockSize>>>(d_cnew, d_cold, d_lap, nx, ny, nz, D, dt);
    cudaDeviceSynchronize();
    
    cudaMemcpy(cnew.data(), d_cnew, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
}

#else

// CPU fallback versions when CUDA is disabled
void computeLaplacianGPU(const double*, double*, size_t, size_t, size_t, double, double, double) {
    fprintf(stderr, "Error: CUDA disabled at compile time\n");
    exit(1);
}

void computeChemicalPotentialGPU(const std::vector<double>&, std::vector<double>&,
                                 size_t, size_t, size_t, double, double, double,
                                 double, double, double, double,
                                 double*, double*, double*) {
    fprintf(stderr, "Error: CUDA disabled at compile time\n");
    exit(1);
}

void cahnHilliardUpdateGPU(std::vector<double>&, const std::vector<double>&,
                           const std::vector<double>&, size_t, size_t, size_t,
                           double, double, double, double, double,
                           double*, double*, double*, double*) {
    fprintf(stderr, "Error: CUDA disabled at compile time\n");
    exit(1);
}

#endif

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CPU fallback: Compute Laplacian
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

// Initialize concentration field with OpenMP parallelization
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(3) schedule(static)
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
    bool hasNaN = false;
    #pragma omp parallel for reduction(||:hasNaN)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            hasNaN = true;
        }
    }
    
    if (hasNaN) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
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
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
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
    
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    double *d_c [[maybe_unused]] = nullptr, *d_mu [[maybe_unused]] = nullptr, 
           *d_cnew [[maybe_unused]] = nullptr, *d_lap [[maybe_unused]] = nullptr;
    
#ifndef DISABLE_CUDA
    cudaSetDevice(rank % 1);
    cudaMalloc(&d_c, gridSize * sizeof(double));
    cudaMalloc(&d_mu, gridSize * sizeof(double));
    cudaMalloc(&d_cnew, gridSize * sizeof(double));
    cudaMalloc(&d_lap, gridSize * sizeof(double));
#endif
    
    if (rank == 0) {
        printf("Initializing concentration field...\n");
        initializeConcentration(cold, nx, ny, nz);
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    MPI_Bcast(cold.data(), gridSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
#ifndef DISABLE_CUDA
        computeChemicalPotentialGPU(cold, mu, nx, ny, nz, dx, dy, dz, 
                                    gamma, e_AA, e_BB, e_AB,
                                    d_c, d_mu, d_lap);
#else
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    const double cv = cold[idx];
                    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                             + 3.0 * cv + cv * cv * cv
                             - gamma * computeLaplacian(cold, nx, ny, nz, dx, dy, dz, x, y, z);
                }
            }
        }
#endif
        
        MPI_Bcast(mu.data(), gridSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
#ifndef DISABLE_CUDA
        cahnHilliardUpdateGPU(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz,
                              d_cnew, d_c, d_mu, d_lap);
#else
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    cnew[idx] = cold[idx] + dt * D * 
                               computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
                }
            }
        }
#endif
        
        MPI_Bcast(cnew.data(), gridSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cold = cnew;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        if (printResults) {
            print_results(cold, "Concentration");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
#ifndef DISABLE_CUDA
    cudaFree(d_c);
    cudaFree(d_mu);
    cudaFree(d_cnew);
    cudaFree(d_lap);
#endif
    
    MPI_Finalize();
    
    return 0;
}
