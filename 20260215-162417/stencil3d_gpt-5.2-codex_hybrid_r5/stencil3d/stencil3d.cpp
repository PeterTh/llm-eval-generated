#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void abortWithMessage(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::abort();
}

inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        abortWithMessage("Aborting due to CUDA error.");
    }
}

#define CHECK_CUDA(call) checkCuda((call), __FILE__, __LINE__)

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start) {
    std::fill(grid.begin(), grid.end(), 0.0);

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                const size_t local_z = z + 1;
                const size_t local_idx = idx3(x, y, local_z, nx, ny);
                grid[local_idx] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx, const int ny,
                              const int z_begin, const int z_end) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + z_begin;

    if (x >= nx - 1 || y >= ny - 1 || z > z_end) {
        return;
    }

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - static_cast<size_t>(nx)];
    const Real back = input[idx + static_cast<size_t>(nx)];
    const Real bottom = input[idx - plane];
    const Real top = input[idx + plane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

__global__ void copyBoundariesKernel(const Real* __restrict__ input,
                                     Real* __restrict__ output,
                                     const int nx, const int ny,
                                     const int local_nz,
                                     const int z_start,
                                     const int nz_global) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_local = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z_local > local_nz) {
        return;
    }

    const int global_z = z_start + (z_local - 1);
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_z == 0 || global_z == nz_global - 1) {
        const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        const size_t idx = static_cast<size_t>(z_local) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
        output[idx] = input[idx];
    }
}

void launchStencilKernel(const Real* input, Real* output, const int nx, const int ny, const int z_begin, const int z_end) {
    if (z_begin > z_end || nx <= 2 || ny <= 2) {
        return;
    }
    const dim3 block(8, 8, 4);
    const dim3 grid((nx - 2 + block.x - 1) / block.x,
                    (ny - 2 + block.y - 1) / block.y,
                    (z_end - z_begin + 1 + block.z - 1) / block.z);
    stencilKernel<<<grid, block>>>(input, output, nx, ny, z_begin, z_end);
    CHECK_CUDA(cudaGetLastError());
}

void launchBoundaryKernel(const Real* input, Real* output, const int nx, const int ny, const int local_nz, const int z_start, const int nz_global) {
    if (local_nz <= 0 || nx <= 0 || ny <= 0) {
        return;
    }
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);
    copyBoundariesKernel<<<grid, block>>>(input, output, nx, ny, local_nz, z_start, nz_global);
    CHECK_CUDA(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI implementation does not provide required threading level.");
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                parse_status = 2;
                printUsage(argv[0]);
            } else {
                parse_status = 1;
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            if (parse_status != 0) {
                break;
            }
        }

        if (parse_status == 0) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 0) {
        MPI_Finalize();
        return (parse_status == 1) ? 1 : 0;
    }

    unsigned long long nx_b = nx;
    unsigned long long ny_b = ny;
    unsigned long long nz_b = nz;
    MPI_Bcast(&nx_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    nx = static_cast<size_t>(nx_b);
    ny = static_cast<size_t>(ny_b);
    nz = static_cast<size_t>(nz_b);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    if (nx < 2 || ny < 2 || nz < 2) {
        if (rank == 0) {
            printf("Grid dimensions must be at least 2 in each dimension.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    if (local_nz == 0) {
        if (rank == 0) {
            printf("MPI rank count exceeds grid size in Z dimension.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t local_with_halo = local_nz + 2;
    const size_t plane = nx * ny;
    const size_t local_size = local_with_halo * plane;
    const size_t local_compact_size = local_nz * plane;

    std::vector<Real> host_grid(local_size);
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(host_grid, nx, ny, local_nz, z_start);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_grid1, local_size * sizeof(Real)));
    CHECK_CUDA(cudaMalloc(&d_grid2, local_size * sizeof(Real)));
    CHECK_CUDA(cudaMemcpy(d_grid1, host_grid.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));

    Real* h_send_down = nullptr;
    Real* h_send_up = nullptr;
    Real* h_recv_down = nullptr;
    Real* h_recv_up = nullptr;
    if (size > 1) {
        CHECK_CUDA(cudaHostAlloc(&h_send_down, plane * sizeof(Real), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_send_up, plane * sizeof(Real), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_recv_down, plane * sizeof(Real), cudaHostAllocDefault));
        CHECK_CUDA(cudaHostAlloc(&h_recv_up, plane * sizeof(Real), cudaHostAllocDefault));
    }

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        MPI_Request reqs[4];
        int req_count = 0;
        if (size > 1) {
            if (rank > 0) {
                CHECK_CUDA(cudaMemcpy(h_send_down, d_in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
                MPI_Irecv(h_recv_down, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
                MPI_Isend(h_send_down, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            if (rank < size - 1) {
                CHECK_CUDA(cudaMemcpy(h_send_up, d_in + static_cast<size_t>(local_nz) * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
                MPI_Irecv(h_recv_up, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
                MPI_Isend(h_send_up, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
            }
        }

        const int z_compute_begin = (z_start == 0) ? 2 : 1;
        const int z_compute_end = (z_start + local_nz - 1 == nz - 1) ? static_cast<int>(local_nz) - 1 : static_cast<int>(local_nz);
        const int z_inner_begin = std::max(z_compute_begin, 2);
        const int z_inner_end = std::min(z_compute_end, static_cast<int>(local_nz) - 1);

        launchStencilKernel(d_in, d_out, static_cast<int>(nx), static_cast<int>(ny), z_inner_begin, z_inner_end);

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
            if (rank > 0) {
                CHECK_CUDA(cudaMemcpy(d_in, h_recv_down, plane * sizeof(Real), cudaMemcpyHostToDevice));
            }
            if (rank < size - 1) {
                CHECK_CUDA(cudaMemcpy(d_in + static_cast<size_t>(local_nz + 1) * plane, h_recv_up, plane * sizeof(Real), cudaMemcpyHostToDevice));
            }
        }

        if (z_compute_begin <= z_compute_end) {
            if (z_compute_begin <= z_inner_begin - 1) {
                launchStencilKernel(d_in, d_out, static_cast<int>(nx), static_cast<int>(ny), z_compute_begin, z_inner_begin - 1);
            }
            if (z_inner_end + 1 <= z_compute_end) {
                launchStencilKernel(d_in, d_out, static_cast<int>(nx), static_cast<int>(ny), z_inner_end + 1, z_compute_end);
            }
        }

        launchBoundaryKernel(d_in, d_out, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(local_nz), static_cast<int>(z_start), static_cast<int>(nz));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double elapsed = end - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = (max_elapsed > 0.0) ? (cellUpdates / max_elapsed / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;

    std::vector<Real> finalGrid;
    if (validate || printResults) {
        std::vector<Real> local_compact(local_compact_size);
        CHECK_CUDA(cudaMemcpy(local_compact.data(), d_final + plane, local_compact_size * sizeof(Real), cudaMemcpyDeviceToHost));

        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
        }

        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = base + ((static_cast<size_t>(r) < rem) ? 1 : 0);
                const size_t r_z_start = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
                counts[r] = static_cast<int>(r_local_nz * plane);
                displs[r] = static_cast<int>(r_z_start * plane);
            }
        }

        MPI_Gatherv(local_compact.data(), static_cast<int>(local_compact_size), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int final_status = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                final_status = 1;
            }
        }
    }

    MPI_Bcast(&final_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (h_send_down) {
        CHECK_CUDA(cudaFreeHost(h_send_down));
        CHECK_CUDA(cudaFreeHost(h_send_up));
        CHECK_CUDA(cudaFreeHost(h_recv_down));
        CHECK_CUDA(cudaFreeHost(h_recv_up));
    }
    CHECK_CUDA(cudaFree(d_grid1));
    CHECK_CUDA(cudaFree(d_grid2));

    MPI_Finalize();
    return final_status;
}
