#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void checkCuda(cudaError_t status, const char* msg, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d CUDA error: %s: %s\n", rank, msg, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline void decomposeZ(const size_t global_nz, const int rank, const int size, size_t& local_nz, size_t& z_start) {
    const size_t base = global_nz / static_cast<size_t>(size);
    const size_t rem = global_nz % static_cast<size_t>(size);
    local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start) {
    std::fill(grid.begin(), grid.end(), static_cast<Real>(0.0));
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t lz = 0; lz < local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + lz;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[idx3(x, y, lz + 1, nx, ny)] = static_cast<Real>((global_idx % 19) * 1.0);
            }
        }
    }
}

__global__ void stencilKernel(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t local_nz,
                              const size_t global_z_start, const size_t global_nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t lz = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lz >= local_nz) {
        return;
    }

    const size_t global_z = global_z_start + lz;
    const size_t z = lz + 1;
    const size_t idx = idx3(x, y, z, nx, ny);

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || global_z == 0 || global_z + 1 == global_nz) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx3(x - 1, y, z, nx, ny)];
    const Real right = input[idx3(x + 1, y, z, nx, ny)];
    const Real front = input[idx3(x, y - 1, z, nx, ny)];
    const Real back = input[idx3(x, y + 1, z, nx, ny)];
    const Real bottom = input[idx3(x, y, z - 1, nx, ny)];
    const Real top = input[idx3(x, y, z + 1, nx, ny)];

    output[idx] = (center + left + right + front + back + bottom + top) * static_cast<Real>(1.0 / 7.0);
}

void launchStencil(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t local_nz,
                   const size_t global_z_start, const size_t global_nz, int rank) {
    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);

    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        return;
    }

    stencilKernel<<<grid, block>>>(input, output, nx, ny, local_nz, global_z_start, global_nz);
    checkCuda(cudaGetLastError(), "stencilKernel launch", rank);
}

void exchangeHalos(Real* d_input, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size,
                   Real* h_send_lower, Real* h_recv_lower,
                   Real* h_send_upper, Real* h_recv_upper) {
    if (local_nz == 0) {
        return;
    }

    const size_t plane_elems = nx * ny;
    const size_t plane_bytes = plane_elems * sizeof(Real);
    const int plane_count = static_cast<int>(plane_elems);

    if (rank > 0) {
        checkCuda(cudaMemcpy(h_send_lower, d_input + idx3(0, 0, 1, nx, ny), plane_bytes, cudaMemcpyDeviceToHost),
                  "copy lower halo to host", rank);
        MPI_Sendrecv(h_send_lower, plane_count, MPI_DOUBLE, rank - 1, 0,
                     h_recv_lower, plane_count, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        checkCuda(cudaMemcpy(d_input + idx3(0, 0, 0, nx, ny), h_recv_lower, plane_bytes, cudaMemcpyHostToDevice),
                  "copy lower halo to device", rank);
    }

    if (rank < size - 1) {
        checkCuda(cudaMemcpy(h_send_upper, d_input + idx3(0, 0, local_nz, nx, ny), plane_bytes, cudaMemcpyDeviceToHost),
                  "copy upper halo to host", rank);
        MPI_Sendrecv(h_send_upper, plane_count, MPI_DOUBLE, rank + 1, 1,
                     h_recv_upper, plane_count, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        checkCuda(cudaMemcpy(d_input + idx3(0, 0, local_nz + 1, nx, ny), h_recv_upper, plane_bytes, cudaMemcpyHostToDevice),
                  "copy upper halo to device", rank);
    }
}

bool validateResultMPI(const std::vector<Real>& local_grid, MPI_Comm comm, int rank) {
    int local_finite = 1;
    Real local_min = std::numeric_limits<Real>::max();
    Real local_max = std::numeric_limits<Real>::lowest();

#pragma omp parallel for reduction(min : local_min) reduction(max : local_max) reduction(&& : local_finite)
    for (size_t i = 0; i < local_grid.size(); ++i) {
        const Real val = local_grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            local_finite = 0;
        }
        local_min = std::min(local_min, val);
        local_max = std::max(local_max, val);
    }

    int global_finite = 0;
    MPI_Allreduce(&local_finite, &global_finite, 1, MPI_INT, MPI_LAND, comm);

    Real global_min = 0.0;
    Real global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
    }

    if (!global_finite) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    if (global_max > 1e6 || global_min < -1e6) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
        return false;
    }

    return true;
}

void gatherAndPrintResults(const std::vector<Real>& local_grid, const size_t nx, const size_t ny, const size_t nz,
                           const int rank, const int size) {
    const size_t plane_elems = nx * ny;
    const size_t local_count = local_grid.size();

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<Real> global_grid;

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            size_t r_local_nz = 0;
            size_t r_z_start = 0;
            decomposeZ(nz, r, size, r_local_nz, r_z_start);
            counts[r] = static_cast<int>(r_local_nz * plane_elems);
            displs[r] = static_cast<int>(r_z_start * plane_elems);
        }
        global_grid.resize(nx * ny * nz);
    }

    MPI_Gatherv(local_grid.data(), static_cast<int>(local_count), MPI_DOUBLE,
                rank == 0 ? global_grid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_results(global_grid, "Grid");
    }
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t nx_u = 128;
    uint64_t ny_u = 0;
    uint64_t nz_u = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;
    int run = 1;
    int exit_code = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx_u = static_cast<uint64_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny_u = static_cast<uint64_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz_u = static_cast<uint64_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                run = 0;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                run = 0;
                exit_code = 1;
                break;
            }
        }

        if (run) {
            if (ny_u == 0) {
                ny_u = nx_u;
            }
            if (nz_u == 0) {
                nz_u = nx_u;
            }
        }
    }

    MPI_Bcast(&run, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!run) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&nx_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const size_t nx = static_cast<size_t>(nx_u);
    const size_t ny = static_cast<size_t>(ny_u);
    const size_t nz = static_cast<size_t>(nz_u);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank);
    if (device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(rank % device_count), "cudaSetDevice", rank);

    size_t local_nz = 0;
    size_t z_start = 0;
    decomposeZ(nz, rank, size, local_nz, z_start);

    const size_t plane_elems = nx * ny;
    const size_t local_total = (local_nz + 2) * plane_elems;

    std::vector<Real> host_grid(local_total);
    initializeGrid(host_grid, nx, ny, local_nz, z_start);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    checkCuda(cudaMalloc(&d_grid1, local_total * sizeof(Real)), "cudaMalloc grid1", rank);
    checkCuda(cudaMalloc(&d_grid2, local_total * sizeof(Real)), "cudaMalloc grid2", rank);
    checkCuda(cudaMemcpy(d_grid1, host_grid.data(), local_total * sizeof(Real), cudaMemcpyHostToDevice),
              "copy grid1 to device", rank);

    host_grid.clear();
    host_grid.shrink_to_fit();

    Real* h_send_lower = nullptr;
    Real* h_recv_lower = nullptr;
    Real* h_send_upper = nullptr;
    Real* h_recv_upper = nullptr;

    if (rank > 0) {
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&h_send_lower), plane_elems * sizeof(Real)),
                  "cudaMallocHost send lower", rank);
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&h_recv_lower), plane_elems * sizeof(Real)),
                  "cudaMallocHost recv lower", rank);
    }

    if (rank < size - 1) {
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&h_send_upper), plane_elems * sizeof(Real)),
                  "cudaMallocHost send upper", rank);
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&h_recv_upper), plane_elems * sizeof(Real)),
                  "cudaMallocHost recv upper", rank);
    }

    if (rank == 0) {
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    Real* d_current = d_grid1;
    Real* d_next = d_grid2;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(d_current, nx, ny, local_nz, rank, size, h_send_lower, h_recv_lower, h_send_upper, h_recv_upper);
        launchStencil(d_current, d_next, nx, ny, local_nz, z_start, nz, rank);
        std::swap(d_current, d_next);
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double elapsed = end - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_elapsed * 1000.0);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<Real> local_owned;
    if (printResults || validate) {
        const size_t local_count = local_nz * plane_elems;
        local_owned.resize(local_count);
        if (local_count > 0) {
            checkCuda(cudaMemcpy(local_owned.data(), d_current + idx3(0, 0, 1, nx, ny),
                                  local_count * sizeof(Real), cudaMemcpyDeviceToHost),
                      "copy final grid to host", rank);
        }
    }

    if (printResults) {
        gatherAndPrintResults(local_owned, nx, ny, nz, rank, size);
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResultMPI(local_owned, MPI_COMM_WORLD, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        if (!valid) {
            if (h_send_lower) {
                cudaFreeHost(h_send_lower);
            }
            if (h_recv_lower) {
                cudaFreeHost(h_recv_lower);
            }
            if (h_send_upper) {
                cudaFreeHost(h_send_upper);
            }
            if (h_recv_upper) {
                cudaFreeHost(h_recv_upper);
            }
            cudaFree(d_grid1);
            cudaFree(d_grid2);
            MPI_Finalize();
            return 1;
        }
    }

    if (h_send_lower) {
        cudaFreeHost(h_send_lower);
    }
    if (h_recv_lower) {
        cudaFreeHost(h_recv_lower);
    }
    if (h_send_upper) {
        cudaFreeHost(h_send_upper);
    }
    if (h_recv_upper) {
        cudaFreeHost(h_recv_upper);
    }

    cudaFree(d_grid1);
    cudaFree(d_grid2);

    MPI_Finalize();
    return 0;
}
