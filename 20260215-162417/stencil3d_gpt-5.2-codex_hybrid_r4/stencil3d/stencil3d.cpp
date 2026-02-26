#include <algorithm>
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

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        const cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                 \
        }                                                                 \
    } while (0)

// 3D index calculation
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGridLocal(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_start) {
    if (local_nz == 0) {
        return;
    }
    const size_t plane = nx * ny;
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + (z - 1);
                const size_t global_idx = global_z * plane + y * nx + x;
                grid[idx3(x, y, z, nx, ny)] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

__global__ void stencilKernel(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t local_nz,
                              const size_t z_start, const size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) {
        return;
    }
    const size_t global_z = z_start + (z - 1);
    const size_t idx = idx3(x, y, z, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_z == 0 || global_z == global_nz - 1) {
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

    output[idx] = (center + left + right + front + back + bottom + top) / static_cast<Real>(7.0);
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
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    // Parse command line arguments
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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    int localFlag = (showHelp ? 1 : 0) | (parseError ? 2 : 0);
    int globalFlag = 0;
    MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    if (globalFlag != 0) {
        if (world_rank == 0) {
            if (parseError) {
                printf("Unknown option provided.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return (globalFlag & 2) ? 1 : 0;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (world_size > static_cast<int>(nz)) {
        if (world_rank == 0) {
            fprintf(stderr, "MPI ranks (%d) exceed global Z dimension (%zu).\n", world_size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (world_rank == 0) {
            fprintf(stderr, "Plane size too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int plane_count = static_cast<int>(plane);

    const size_t base = nz / static_cast<size_t>(world_size);
    const size_t remainder = nz % static_cast<size_t>(world_size);
    const size_t local_nz = base + (static_cast<size_t>(world_rank) < remainder ? 1 : 0);
    const size_t z_start = base * static_cast<size_t>(world_rank) + std::min(static_cast<size_t>(world_rank), remainder);

    const size_t local_with_halo = local_nz > 0 ? (local_nz + 2) * plane : 0;
    std::vector<Real> h_grid1(local_with_halo, 0.0);
    std::vector<Real> h_grid2(local_with_halo, 0.0);

    if (world_rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", world_size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (world_rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGridLocal(h_grid1, nx, ny, local_nz, z_start);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    if (local_with_halo > 0) {
        CUDA_CHECK(cudaMalloc(&d_grid1, local_with_halo * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_grid2, local_with_halo * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_grid1, h_grid1.data(), local_with_halo * sizeof(Real), cudaMemcpyHostToDevice));
    }

    std::vector<Real> send_lower;
    std::vector<Real> recv_lower;
    std::vector<Real> send_upper;
    std::vector<Real> recv_upper;
    if (world_size > 1 && local_nz > 0) {
        send_lower.resize(plane);
        recv_lower.resize(plane);
        send_upper.resize(plane);
        recv_upper.resize(plane);
    }

    const int TAG_DOWN = 100;
    const int TAG_UP = 101;
    auto exchangeHalos = [&](Real* d_grid) {
        if (world_size == 1 || local_nz == 0) {
            return;
        }
        if (world_rank > 0) {
            CUDA_CHECK(cudaMemcpy(send_lower.data(), d_grid + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(send_lower.data(), plane_count, MPI_DOUBLE, world_rank - 1, TAG_DOWN,
                         recv_lower.data(), plane_count, MPI_DOUBLE, world_rank - 1, TAG_UP,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(d_grid, recv_lower.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        }
        if (world_rank < world_size - 1) {
            CUDA_CHECK(cudaMemcpy(send_upper.data(), d_grid + plane * local_nz, plane * sizeof(Real), cudaMemcpyDeviceToHost));
            MPI_Sendrecv(send_upper.data(), plane_count, MPI_DOUBLE, world_rank + 1, TAG_UP,
                         recv_upper.data(), plane_count, MPI_DOUBLE, world_rank + 1, TAG_DOWN,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(d_grid + plane * (local_nz + 1), recv_upper.data(), plane * sizeof(Real), cudaMemcpyHostToDevice));
        }
    };

    if (world_rank == 0) {
        printf("Running stencil computation...\n");
    }

    Real* d_in = d_grid1;
    Real* d_out = d_grid2;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        exchangeHalos(d_in);
        if (local_nz > 0) {
            const dim3 block(8, 8, 4);
            const dim3 grid((nx + block.x - 1) / block.x,
                            (ny + block.y - 1) / block.y,
                            (local_nz + block.z - 1) / block.z);
            stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, local_nz, z_start, nz);
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_in, d_out);
        }
    }

    if (local_nz > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    const double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int result_code = 0;
    if (printResults || validate) {
        const size_t local_count = local_nz * plane;
        if (local_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (world_rank == 0) {
                fprintf(stderr, "Local grid too large for MPI gather.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<Real> h_local;
        if (local_nz > 0) {
            h_local.resize(local_count);
            CUDA_CHECK(cudaMemcpy(h_local.data(), d_in + plane, local_count * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        std::vector<Real> globalGrid;
        std::vector<int> counts;
        std::vector<int> displs;
        if (world_rank == 0) {
            const size_t global_count = nx * ny * nz;
            if (global_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                fprintf(stderr, "Global grid too large for MPI gather.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            globalGrid.resize(global_count);
            counts.resize(world_size);
            displs.resize(world_size);
            for (int r = 0; r < world_size; ++r) {
                const size_t r_local_nz = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t r_z_start = base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder);
                const size_t r_count = r_local_nz * plane;
                if (r_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    fprintf(stderr, "Rank %d grid too large for MPI gather.\n", r);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                counts[r] = static_cast<int>(r_count);
                displs[r] = static_cast<int>(r_z_start * plane);
            }
        }

        MPI_Gatherv(local_nz > 0 ? h_local.data() : nullptr, static_cast<int>(local_count), MPI_DOUBLE,
                    world_rank == 0 ? globalGrid.data() : nullptr,
                    world_rank == 0 ? counts.data() : nullptr,
                    world_rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (world_rank == 0 && printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            int valid_int = 1;
            if (world_rank == 0) {
                printf("Validating result...\n");
                const bool valid = validateResult(globalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
                valid_int = valid ? 1 : 0;
            }
            MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            result_code = valid_int ? 0 : 1;
        }
    }

    if (d_grid1) {
        CUDA_CHECK(cudaFree(d_grid1));
    }
    if (d_grid2) {
        CUDA_CHECK(cudaFree(d_grid2));
    }

    MPI_Finalize();
    return result_code;
}
