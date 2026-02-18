#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation for the global grid
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline constexpr size_t local_idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Partition {
    size_t local_z = 0;
    size_t z_start = 0;
};

Partition partitionZ(const int rank, const int size, const size_t nz) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    return {local, start};
}

inline void checkCuda(const cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void stencil7_kernel(const Real* __restrict__ input,
                                Real* __restrict__ output,
                                const size_t nx,
                                const size_t ny,
                                const size_t local_z,
                                const size_t z_start,
                                const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= local_z) {
        return;
    }

    const size_t local_z_idx = z + 1;
    const size_t global_z = z_start + z;
    const size_t stride = nx * ny;
    const size_t idx = local_z_idx * stride + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_z == 0 || global_z == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - stride];
    const Real top = input[idx + stride];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeLocalGrid(std::vector<Real>& grid,
                         const size_t nx,
                         const size_t ny,
                         const size_t local_z,
                         const size_t z_start) {
    std::fill(grid.begin(), grid.end(), 0.0);
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_z; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + z;
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                const size_t local_idx = local_idx3(x, y, z + 1, nx, ny);
                grid[local_idx] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    int bad = 0;
    #pragma omp parallel for reduction(|:bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            bad = 1;
        }
    }
    if (bad != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

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
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    if (world_rank == 0) {
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

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    uint64_t nx64 = static_cast<uint64_t>(nx);
    uint64_t ny64 = static_cast<uint64_t>(ny);
    uint64_t nz64 = static_cast<uint64_t>(nz);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;

    MPI_Bcast(&nx64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    nx = static_cast<size_t>(nx64);
    ny = static_cast<size_t>(ny64);
    nz = static_cast<size_t>(nz64);
    validate = validateFlag != 0;
    printResults = printFlag != 0;

    const int active_ranks = std::min(world_size, static_cast<int>(nz));
    const int color = world_rank < active_ranks ? 0 : MPI_UNDEFINED;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &comm);
    if (color == MPI_UNDEFINED) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device_id = rank % device_count;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    const Partition part = partitionZ(rank, size, nz);
    const size_t local_z = part.local_z;
    const size_t z_start = part.z_start;

    const size_t plane = nx * ny;
    const size_t local_nz = local_z + 2;
    const size_t total_elems = local_nz * plane;
    const size_t total_bytes = total_elems * sizeof(Real);
    const size_t plane_bytes = plane * sizeof(Real);

    std::vector<Real> host_grid(total_elems, 0.0);
    initializeLocalGrid(host_grid, nx, ny, local_z, z_start);

    Real* d_in = nullptr;
    Real* d_out = nullptr;
    checkCuda(cudaMalloc(&d_in, total_bytes), "cudaMalloc d_in");
    checkCuda(cudaMalloc(&d_out, total_bytes), "cudaMalloc d_out");
    checkCuda(cudaMemcpy(d_in, host_grid.data(), total_bytes, cudaMemcpyHostToDevice), "copy init");
    checkCuda(cudaMemset(d_out, 0, total_bytes), "memset d_out");

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_z + block.z - 1) / block.z);

    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (local_z > 0) {
            checkCuda(cudaMemcpy(host_grid.data() + plane, d_in + plane, plane_bytes, cudaMemcpyDeviceToHost), "copy lower plane");
            checkCuda(cudaMemcpy(host_grid.data() + plane * local_z, d_in + plane * local_z, plane_bytes, cudaMemcpyDeviceToHost), "copy upper plane");
        }

        if (size > 1 && local_z > 0) {
            if (rank > 0) {
                MPI_Sendrecv(host_grid.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                             host_grid.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                             comm, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) {
                MPI_Sendrecv(host_grid.data() + plane * local_z, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                             host_grid.data() + plane * (local_z + 1), static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                             comm, MPI_STATUS_IGNORE);
            }
        }

        if (rank > 0) {
            checkCuda(cudaMemcpy(d_in, host_grid.data(), plane_bytes, cudaMemcpyHostToDevice), "copy lower halo");
        }
        if (rank < size - 1) {
            checkCuda(cudaMemcpy(d_in + plane * (local_z + 1), host_grid.data() + plane * (local_z + 1), plane_bytes, cudaMemcpyHostToDevice), "copy upper halo");
        }

        if (local_z > 0) {
            stencil7_kernel<<<grid, block>>>(d_in, d_out, nx, ny, local_z, z_start, nz);
            checkCuda(cudaGetLastError(), "kernel launch");
        }

        std::swap(d_in, d_out);
    }

    checkCuda(cudaDeviceSynchronize(), "device sync");
    MPI_Barrier(comm);
    const double end = MPI_Wtime();

    const double duration = end - start;
    double maxDuration = 0.0;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const bool needGather = validate || printResults;
    std::vector<Real> fullGrid;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    int validation_ok = 1;

    if (needGather) {
        checkCuda(cudaMemcpy(host_grid.data(), d_in, total_bytes, cudaMemcpyDeviceToHost), "copy final grid");

        const size_t local_elems = local_z * plane;
        const int local_elems_i = static_cast<int>(local_elems);

        if (rank == 0) {
            fullGrid.resize(nx * ny * nz);
            recvcounts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const Partition pr = partitionZ(r, size, nz);
                const size_t elems = pr.local_z * plane;
                recvcounts[r] = static_cast<int>(elems);
                displs[r] = static_cast<int>(offset);
                offset += elems;
            }
        }

        MPI_Gatherv(host_grid.data() + plane, local_elems_i, MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);

        if (rank == 0 && printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
                validation_ok = validateResult(fullGrid, nx, ny, nz) ? 1 : 0;
                printf("Validation: %s\n", validation_ok ? "PASSED" : "FAILED");
            }
            MPI_Bcast(&validation_ok, 1, MPI_INT, 0, comm);
        }
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxDuration * 1000.0);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / maxDuration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    checkCuda(cudaFree(d_in), "cudaFree d_in");
    checkCuda(cudaFree(d_out), "cudaFree d_out");

    MPI_Comm_free(&comm);
    MPI_Finalize();

    if (validate && validation_ok == 0) {
        return 1;
    }
    return 0;
}
