#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Forward declaration if header not available or just include it
#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// 3D index calculation for device
__device__ inline size_t idx3_device(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ double computeLaplacianDevice(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz_local,
                                        const double dx, const double dy, const double dz, 
                                        const size_t x, const size_t y, const size_t z,
                                        const int rank, const int size) {
    double val_c = c[idx3_device(x, y, z, nx, ny)];
    
    // X direction
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    double val_xp = c[idx3_device(xp, y, z, nx, ny)];
    double val_xn = c[idx3_device(xn, y, z, nx, ny)];
    double cxx = (val_xp + val_xn - 2.0 * val_c) / (dx * dx);
    
    // Y direction
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    double val_yp = c[idx3_device(x, yp, z, nx, ny)];
    double val_yn = c[idx3_device(x, yn, z, nx, ny)];
    double cyy = (val_yp + val_yn - 2.0 * val_c) / (dy * dy);
    
    // Z direction
    // Neighbors are at z+1 and z-1 in the buffer (which includes halos)
    double val_zp = c[idx3_device(x, y, z + 1, nx, ny)];
    double val_zn = c[idx3_device(x, y, z - 1, nx, ny)];
    double czz = (val_zp + val_zn - 2.0 * val_c) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential kernel
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                              const size_t nx, const size_t ny, const size_t nz_local,
                                              const double dx, const double dy, const double dz,
                                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                              const int rank, const int size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z_local = blockIdx.z * blockDim.z + threadIdx.z; // 0 to nz_local-1

    if (x >= nx || y >= ny || z_local >= nz_local) return;

    // Shift z to account for halo at z=0 (buffer index 0 is halo, 1 is z_local=0)
    size_t z_mem = z_local + 1;

    double cv = c[idx3_device(x, y, z_mem, nx, ny)];
    double lap = computeLaplacianDevice(c, nx, ny, nz_local, dx, dy, dz, x, y, z_mem, rank, size);

    mu[idx3_device(x, y, z_mem, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                                         + 3.0 * cv + cv * cv * cv
                                         - gamma * lap;
}

// Cahn-Hilliard update kernel
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                        const double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nz_local,
                                        const double D, const double dt, const double dx, const double dy, const double dz,
                                        const int rank, const int size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z_local = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z_local >= nz_local) return;

    size_t z_mem = z_local + 1;

    double lap_mu = computeLaplacianDevice(mu, nx, ny, nz_local, dx, dy, dz, x, y, z_mem, rank, size);
    
    cnew[idx3_device(x, y, z_mem, nx, ny)] = cold[idx3_device(x, y, z_mem, nx, ny)] + dt * D * lap_mu;
}

// Helper to fill halos on device
__global__ void fillHalosKernel(double* __restrict__ data, 
                                const size_t nx, const size_t ny, const size_t nz_local,
                                const int rank, const int size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= nx || y >= ny) return;

    // Clamped boundary conditions for global boundaries
    // If rank == 0, halo bottom (index 0) should equal first real slice (index 1*nx*ny)
    if (rank == 0) {
        data[idx3_device(x, y, 0, nx, ny)] = data[idx3_device(x, y, 1, nx, ny)];
    }
    
    // If rank == size - 1, halo top (index nz_local+1) should equal last real slice (index nz_local)
    if (rank == size - 1) {
        data[idx3_device(x, y, nz_local + 1, nx, ny)] = data[idx3_device(x, y, nz_local, nx, ny)];
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Domain decomposition: Z-axis
    size_t nz_local = nz / size;
    size_t remainder = nz % size;
    size_t z_start = rank * nz_local;
    if (rank < (int)remainder) {
        nz_local++;
        z_start += rank;
    } else {
        z_start += remainder;
    }
    
    // Select GPU
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
    } else {
        if (rank == 0) printf("No CUDA devices found!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+CUDA+OpenMP)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", size);
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
    
    size_t slice_size = nx * ny;
    size_t alloc_vol = slice_size * (nz_local + 2); // +2 for halos
    
    std::vector<double> h_c(alloc_vol);
    
    if (rank == 0) printf("Initializing concentration field...\n");
    
    // Initialization with global consistency
    // Using OpenMP for CPU-side initialization
    #pragma omp parallel for
    for (size_t z_loc = 0; z_loc < nz_local; ++z_loc) {
        size_t z_global = z_start + z_loc;
        size_t z_buf = z_loc + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = z_buf * slice_size + y * nx + x;
                size_t linear_id = z_global * (nx * ny) + y * nx + x;
                size_t vol_global = nx * ny * nz;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol_global) / static_cast<double>(vol_global));
                h_c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    
    // Allocate device memory
    double *d_c, *d_mu, *d_cnew;
    CUDA_CHECK(cudaMalloc(&d_c, alloc_vol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, alloc_vol * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, alloc_vol * sizeof(double)));
    
    // Halo buffers on Host (safe MPI)
    std::vector<double> h_halo_send_top(slice_size);
    std::vector<double> h_halo_send_bot(slice_size);
    std::vector<double> h_halo_recv_top(slice_size);
    std::vector<double> h_halo_recv_bot(slice_size);
    
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), alloc_vol * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, 
                  (ny + blockSize.y - 1) / blockSize.y, 
                  (nz_local + blockSize.z - 1) / blockSize.z);
    
    dim3 haloBlockSize(16, 16);
    dim3 haloGridSize((nx + haloBlockSize.x - 1) / haloBlockSize.x, 
                      (ny + haloBlockSize.y - 1) / haloBlockSize.y);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // --- Step 1: Chemical Potential ---
        
        // Exchange Halos for C
        if (rank < size - 1) {
            CUDA_CHECK(cudaMemcpy(h_halo_send_top.data(), d_c + nz_local * slice_size, slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpy(h_halo_send_bot.data(), d_c + slice_size, slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        
        MPI_Request reqs[4];
        int num_reqs = 0;
        
        // Odd/Even comms pattern or Isend/Irecv with Waitall
        int top_rank = rank + 1;
        int bot_rank = rank - 1;
        int tag_top = 0;
        int tag_bot = 1;

        if (rank < size - 1) {
            MPI_Isend(h_halo_send_top.data(), slice_size, MPI_DOUBLE, top_rank, tag_bot, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_halo_recv_top.data(), slice_size, MPI_DOUBLE, top_rank, tag_top, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
        
        if (rank > 0) {
            MPI_Isend(h_halo_send_bot.data(), slice_size, MPI_DOUBLE, bot_rank, tag_top, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_halo_recv_bot.data(), slice_size, MPI_DOUBLE, bot_rank, tag_bot, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
        
        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        
        if (rank < size - 1) {
             CUDA_CHECK(cudaMemcpy(d_c + (nz_local + 1) * slice_size, h_halo_recv_top.data(), slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (rank > 0) {
             CUDA_CHECK(cudaMemcpy(d_c, h_halo_recv_bot.data(), slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        
        fillHalosKernel<<<haloGridSize, haloBlockSize>>>(d_c, nx, ny, nz_local, rank, size);
        computeChemicalPotentialKernel<<<gridSize, blockSize>>>(d_c, d_mu, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB, rank, size);
        
        // --- Step 2: Update C ---
        
        // Exchange Halos for Mu
        if (rank < size - 1) {
            CUDA_CHECK(cudaMemcpy(h_halo_send_top.data(), d_mu + nz_local * slice_size, slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpy(h_halo_send_bot.data(), d_mu + slice_size, slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        
        num_reqs = 0;
        if (rank < size - 1) {
            MPI_Isend(h_halo_send_top.data(), slice_size, MPI_DOUBLE, top_rank, tag_bot, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_halo_recv_top.data(), slice_size, MPI_DOUBLE, top_rank, tag_top, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
        if (rank > 0) {
            MPI_Isend(h_halo_send_bot.data(), slice_size, MPI_DOUBLE, bot_rank, tag_top, MPI_COMM_WORLD, &reqs[num_reqs++]);
            MPI_Irecv(h_halo_recv_bot.data(), slice_size, MPI_DOUBLE, bot_rank, tag_bot, MPI_COMM_WORLD, &reqs[num_reqs++]);
        }
        
        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        
        if (rank < size - 1) {
             CUDA_CHECK(cudaMemcpy(d_mu + (nz_local + 1) * slice_size, h_halo_recv_top.data(), slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (rank > 0) {
             CUDA_CHECK(cudaMemcpy(d_mu, h_halo_recv_bot.data(), slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        
        fillHalosKernel<<<haloGridSize, haloBlockSize>>>(d_mu, nx, ny, nz_local, rank, size);
        cahnHilliardUpdateKernel<<<gridSize, blockSize>>>(d_cnew, d_c, d_mu, nx, ny, nz_local, D, dt, dx, dy, dz, rank, size);
        
        std::swap(d_c, d_cnew);
    }
    
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // --- Validation Gather ---
    if (validate || printResults) {
        // Copy real part back to host
        CUDA_CHECK(cudaMemcpy(h_c.data() + slice_size, d_c + slice_size, nz_local * slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        
        std::vector<double> global_c;
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = nz_local * slice_size;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            global_c.resize(nx * ny * nz);
            displs[0] = 0;
            for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Gatherv(h_c.data() + slice_size, my_count, MPI_DOUBLE,
                    global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                double minVal = global_c[0];
                double maxVal = global_c[0];
                int hasNanInt = 0;
                size_t n = global_c.size();
                
                #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:hasNanInt)
                for (size_t i = 0; i < n; ++i) {
                    double val = global_c[i];
                    if (std::isnan(val) || std::isinf(val)) hasNanInt = 1;
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
                bool hasNan = (hasNanInt > 0);
                
                if (hasNan) {
                     printf("Validation failed: found NaN or Inf value\n");
                } else {
                    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                    if (maxVal > 10.0 || minVal < -10.0) {
                        printf("Validation failed: values out of expected range\n");
                    } else {
                        printf("Validation: PASSED\n");
                    }
                }
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    
    MPI_Finalize();
    return 0;
}
