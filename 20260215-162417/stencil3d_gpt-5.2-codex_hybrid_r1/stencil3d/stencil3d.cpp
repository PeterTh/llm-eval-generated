#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                       \
        cudaError_t err = (call);                                                              \
        if (err != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

static void computeLocalZ(const size_t nz, const int rank, const int size, size_t& local_nz, size_t& z_offset) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    z_offset = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
}

static void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny,
                                const size_t local_nz, const size_t z_offset) {
    const size_t plane = nx * ny;
    std::fill(grid.begin(), grid.end(), 0.0);

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_z = z + 1;
                const size_t global_z = z_offset + z;
                const size_t global_idx = global_z * plane + y * nx + x;
                grid[idx3(x, y, local_z, nx, ny)] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

__global__ void stencilKernel(const Real* input, Real* output,
                              const size_t nx, const size_t ny,
                              const size_t local_nz, const size_t z_offset,
                              const size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= local_nz) {
        return;
    }

    const size_t local_z = z + 1;
    const size_t global_z = z_offset + z;
    const size_t plane = nx * ny;
    const size_t idx = local_z * plane + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || global_z == 0 || global_z == global_nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - plane];
    const Real top = input[idx + plane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void exchangeHalos(Real* d_input, const size_t nx, const size_t ny, const size_t local_nz,
                          const int rank, const int size, const int plane_count,
                          std::vector<Real>& send_down, std::vector<Real>& send_up,
                          std::vector<Real>& recv_down, std::vector<Real>& recv_up) {
    const size_t plane_bytes = static_cast<size_t>(plane_count) * sizeof(Real);

    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(send_down.data(), d_input + idx3(0, 0, 1, nx, ny), plane_bytes, cudaMemcpyDeviceToHost));
        MPI_Sendrecv(send_down.data(), plane_count, MPI_DOUBLE, rank - 1, 0,
                     recv_down.data(), plane_count, MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_input + idx3(0, 0, 0, nx, ny), recv_down.data(), plane_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_input + idx3(0, 0, 0, nx, ny), d_input + idx3(0, 0, 1, nx, ny), plane_bytes, cudaMemcpyDeviceToDevice));
    }

    if (rank < size - 1) {
        CUDA_CHECK(cudaMemcpy(send_up.data(), d_input + idx3(0, 0, local_nz, nx, ny), plane_bytes, cudaMemcpyDeviceToHost));
        MPI_Sendrecv(send_up.data(), plane_count, MPI_DOUBLE, rank + 1, 1,
                     recv_up.data(), plane_count, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_input + idx3(0, 0, local_nz + 1, nx, ny), recv_up.data(), plane_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_input + idx3(0, 0, local_nz + 1, nx, ny), d_input + idx3(0, 0, local_nz, nx, ny), plane_bytes, cudaMemcpyDeviceToDevice));
    }
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
    bool showHelp = false;
    bool parseOk = true;
    const char* unknownOption = nullptr;

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
            parseOk = false;
            unknownOption = argv[i];
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (!parseOk) {
        if (rank == 0) {
            printf("Unknown option: %s\n", unknownOption ? unknownOption : "");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    size_t local_nz = 0;
    size_t z_offset = 0;
    computeLocalZ(nz, rank, size, local_nz, z_offset);

    if (local_nz == 0) {
        if (rank == 0) {
            printf("Error: MPI size (%d) exceeds grid dimension nz (%zu).\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
    }

    const size_t plane_elems = nx * ny;
    if (plane_elems > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: plane size too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (local_nz > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: local_nz too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t local_count_elems = local_nz * plane_elems;
    if (local_count_elems > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            printf("Error: local count too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int plane_count = static_cast<int>(plane_elems);
    const size_t local_total_elems = (local_nz + 2) * plane_elems;

    // Allocate grids (double buffering) with halo layers
    std::vector<Real> host_grid(local_total_elems);
    initializeLocalGrid(host_grid, nx, ny, local_nz, z_offset);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_total_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_total_elems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, host_grid.data(), local_total_elems * sizeof(Real), cudaMemcpyHostToDevice));

    std::vector<Real> send_down(plane_elems);
    std::vector<Real> send_up(plane_elems);
    std::vector<Real> recv_down(plane_elems);
    std::vector<Real> recv_up(plane_elems);

    if (rank == 0) {
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (local_nz + block.z - 1) / block.z);

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* output = (iter % 2 == 0) ? d_grid2 : d_grid1;

        exchangeHalos(input, nx, ny, local_nz, rank, size, plane_count, send_down, send_up, recv_down, recv_up);

        stencilKernel<<<grid, block>>>(input, output, nx, ny, local_nz, z_offset, nz);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double elapsed = end - start;

    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);

        // Calculate performance metrics
        const double cellUpdates = static_cast<double>((nx > 2 && ny > 2 && nz > 2)
            ? (nx - 2) * (ny - 2) * (nz - 2)
            : 0) * static_cast<double>(iterations);
        const double mcups = (maxElapsed > 0.0) ? cellUpdates / maxElapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int validation_status = 0;
    const bool needFinalGrid = printResults || validate;

    if (needFinalGrid) {
        CUDA_CHECK(cudaMemcpy(host_grid.data(), (iterations % 2 == 0) ? d_grid1 : d_grid2,
                              local_total_elems * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> nz_counts;
        if (rank == 0) {
            nz_counts.resize(size);
        }
        const int local_nz_int = static_cast<int>(local_nz);
        MPI_Gather(&local_nz_int, 1, MPI_INT, rank == 0 ? nz_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<Real> full_grid;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            int running = 0;
            for (int r = 0; r < size; ++r) {
                const size_t count = static_cast<size_t>(nz_counts[r]) * plane_elems;
                if (count > static_cast<size_t>(INT_MAX)) {
                    printf("Error: gathered count too large for MPI.\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                counts[r] = static_cast<int>(count);
                displs[r] = running;
                running += counts[r];
            }
            full_grid.resize(static_cast<size_t>(nx) * ny * nz);
        }

        const int local_count = static_cast<int>(local_nz * plane_elems);
        MPI_Gatherv(host_grid.data() + idx3(0, 0, 1, nx, ny), local_count, MPI_DOUBLE,
                    rank == 0 ? full_grid.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const std::vector<Real>& finalGrid = full_grid;
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
                }
                validation_status = valid ? 0 : 1;
            }
        }
    }

    MPI_Bcast(&validation_status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();
    return validation_status;
}
