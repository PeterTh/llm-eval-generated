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

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

// MPI communicator info
int mpi_rank = 0;
int mpi_size = 1;
int mpi_coords[3] = {0, 0, 0};
int mpi_dims[3] = {1, 1, 1};
MPI_Comm cart_comm = MPI_COMM_WORLD;
int mpi_neighbors[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

// 3D index calculation
inline __host__ __device__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for computing Laplacian (global kernel)
__global__ void cuda_computeLaplacian_kernel(const double* c, double* laplacian, 
                                              const size_t nx, const size_t ny, const size_t nz,
                                              const double dx_inv2, const double dy_inv2, const double dz_inv2) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;
    
    size_t idx = idx3(x, y, z, nx, ny);
    double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * c[idx]) * dx_inv2;
    double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * c[idx]) * dy_inv2;
    double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * c[idx]) * dz_inv2;
    
    laplacian[idx] = cxx + cyy + czz;
}

// CUDA kernel for computing chemical potential
__global__ void cuda_computeChemicalPotential_kernel(const double* c, const double* laplacian, double* mu,
                                                      const size_t nx, const size_t ny, const size_t nz,
                                                      const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    size_t idx = idx3(x, y, z, nx, ny);
    double cv = c[idx];
    
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
            + 3.0 * cv + cv * cv * cv
            - gamma * laplacian[idx];
}

// CUDA kernel for Cahn-Hilliard update
__global__ void cuda_cahnHilliardUpdate_kernel(double* cnew, const double* cold, const double* mu_laplacian,
                                                const size_t nx, const size_t ny, const size_t nz,
                                                const double coeff) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    size_t idx = idx3(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + coeff * mu_laplacian[idx];
}

// Initialize concentration field (OpenMP parallelized)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
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

// Compute chemical potential with GPU acceleration
void computeChemicalPotential_Hybrid(const std::vector<double>& c, std::vector<double>& mu,
                                     const size_t nx, const size_t ny, const size_t nz,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                     double* d_c, double* d_laplacian, double* d_mu) {
    const size_t gridSize = nx * ny * nz;
    const double dx_inv2 = 1.0 / (dx * dx);
    const double dy_inv2 = 1.0 / (dy * dy);
    const double dz_inv2 = 1.0 / (dz * dz);
    
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockSize(8, 8, 4);
    dim3 gridDim((nx + blockSize.x - 1) / blockSize.x,
                 (ny + blockSize.y - 1) / blockSize.y,
                 (nz + blockSize.z - 1) / blockSize.z);
    
    cuda_computeLaplacian_kernel<<<gridDim, blockSize>>>(d_c, d_laplacian, nx, ny, nz, dx_inv2, dy_inv2, dz_inv2);
    CUDA_CHECK(cudaGetLastError());
    
    cuda_computeChemicalPotential_kernel<<<gridDim, blockSize>>>(d_c, d_laplacian, d_mu, nx, ny, nz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(mu.data(), d_mu, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
}

// Cahn-Hilliard update step with GPU acceleration
void cahnHilliardUpdate_Hybrid(std::vector<double>& cnew, const std::vector<double>& cold,
                              const std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double D, const double dt, const double dx, const double dy, const double dz,
                              double* d_cold, double* d_mu, double* d_laplacian, double* d_cnew) {
    const size_t gridSize = nx * ny * nz;
    const double dx_inv2 = 1.0 / (dx * dx);
    const double dy_inv2 = 1.0 / (dy * dy);
    const double dz_inv2 = 1.0 / (dz * dz);
    const double coeff = dt * D;
    
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, mu.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockSize(8, 8, 4);
    dim3 gridDim((nx + blockSize.x - 1) / blockSize.x,
                 (ny + blockSize.y - 1) / blockSize.y,
                 (nz + blockSize.z - 1) / blockSize.z);
    
    cuda_computeLaplacian_kernel<<<gridDim, blockSize>>>(d_mu, d_laplacian, nx, ny, nz, dx_inv2, dy_inv2, dz_inv2);
    CUDA_CHECK(cudaGetLastError());
    
    cuda_cahnHilliardUpdate_kernel<<<gridDim, blockSize>>>(d_cnew, d_cold, d_laplacian, nx, ny, nz, coeff);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(cnew.data(), d_cnew, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            if (mpi_rank == 0) printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    
    if (mpi_rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    if (maxVal > 10.0 || minVal < -10.0) {
        if (mpi_rank == 0) printf("Validation failed: values out of expected range\n");
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

// Helper to gather and reduce results across MPI ranks
void gatherResults(std::vector<double>& global_c, const std::vector<double>& local_c) {
    if (mpi_size == 1) {
        global_c = local_c;
        return;
    }
    
    size_t local_size = local_c.size();
    size_t total_size = 0;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, cart_comm);
    
    if (mpi_rank == 0) {
        global_c.resize(total_size);
        std::copy(local_c.begin(), local_c.end(), global_c.begin());
        
        size_t offset = local_size;
        std::vector<double> recv_buf(local_size);
        
        for (int rank = 1; rank < mpi_size; ++rank) {
            MPI_Recv(recv_buf.data(), recv_buf.size(), MPI_DOUBLE, rank, 0, cart_comm, MPI_STATUS_IGNORE);
            std::copy(recv_buf.begin(), recv_buf.end(), global_c.begin() + offset);
            offset += recv_buf.size();
        }
    } else {
        MPI_Send(local_c.data(), local_c.size(), MPI_DOUBLE, 0, 0, cart_comm);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    size_t global_nx = 64;
    size_t global_ny = 0;
    size_t global_nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                global_nx = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                global_ny = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                global_nz = atoi(argv[++i]);
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
    
    MPI_Bcast(&global_nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (global_ny == 0) global_ny = global_nx;
    if (global_nz == 0) global_nz = global_nx;
    
    mpi_dims[0] = (mpi_size >= 4) ? 2 : 1;
    mpi_dims[1] = (mpi_size >= 4) ? 2 : 1;
    mpi_dims[2] = (mpi_size >= 8) ? (mpi_size / 4) : 1;
    if (mpi_dims[0] * mpi_dims[1] * mpi_dims[2] != mpi_size) {
        mpi_dims[0] = mpi_size;
        mpi_dims[1] = 1;
        mpi_dims[2] = 1;
    }
    
    int periodic[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims, periodic, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &mpi_rank);
    MPI_Cart_coords(cart_comm, mpi_rank, 3, mpi_coords);
    
    for (int i = 0; i < 6; ++i) {
        int direction = i / 2;
        int displacement = (i % 2 == 0) ? -1 : 1;
        MPI_Cart_shift(cart_comm, direction, displacement, &mpi_neighbors[i ^ 1], &mpi_neighbors[i]);
    }
    
    if (mpi_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Global grid size: %zu x %zu x %zu\n", global_nx, global_ny, global_nz);
        printf("MPI layout: %d x %d x %d\n", mpi_dims[0], mpi_dims[1], mpi_dims[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t local_nx = global_nx / mpi_dims[0];
    size_t local_ny = global_ny / mpi_dims[1];
    size_t local_nz = global_nz / mpi_dims[2];
    
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t local_gridSize = local_nx * local_ny * local_nz;
    
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    if (mpi_rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, local_nx, local_ny, local_nz);
    
    double* d_c = nullptr;
    double* d_laplacian = nullptr;
    double* d_mu = nullptr;
    double* d_cnew = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_c, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_laplacian, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_gridSize * sizeof(double)));
    
    if (mpi_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential_Hybrid(cold, mu, local_nx, local_ny, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                        d_c, d_laplacian, d_mu);
        
        cahnHilliardUpdate_Hybrid(cnew, cold, mu, local_nx, local_ny, local_nz, D, dt, dx, dy, dz,
                                 d_c, d_mu, d_laplacian, d_cnew);
        
        std::swap(cold, cnew);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double total_cells = static_cast<double>(global_nx) * global_ny * global_nz;
        double cellUpdates = total_cells * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    if (printResults || validate) {
        std::vector<double> global_c;
        gatherResults(global_c, cold);
        
        if (mpi_rank == 0) {
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_c, global_nx, global_ny, global_nz);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_laplacian));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}
