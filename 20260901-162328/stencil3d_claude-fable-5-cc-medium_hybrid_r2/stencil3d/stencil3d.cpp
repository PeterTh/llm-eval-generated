#include <algorithm>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil kernel over local planes [kFirst, kLast] of the slab
// (halo planes live at local z=0 and z=lnz+1). Local plane k corresponds to
// global plane z0+k-1. Global boundary points are copied through unchanged.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t kFirst, const size_t kLast, const size_t z0) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t k = blockIdx.z * blockDim.z + threadIdx.z + kFirst;  // local plane
    if (x >= nx || y >= ny || k > kLast) return;

    const size_t gz = z0 + k - 1;  // global z
    const size_t idx = idx3(x, y, k, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top    = input[idx + nx * ny];

    // Simple averaging stencil (same operand order as the original code)
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeGrid(Real* grid, const size_t nx, const size_t ny,
                    const size_t z0, const size_t lnz) {
    // Fill the local slab (halo-owning layout: local plane k = global z0+k-1)
    // with the same values the original serial initialization produced.
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t k = 1; k <= lnz; ++k) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = z0 + k - 1;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, gz, nx, ny);
                grid[idx3(x, y, k, nx, ny)] = (gidx % 19) * 1.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool hasBad = false;
#pragma omp parallel for schedule(static) reduction(||: hasBad)
    for (size_t i = 0; i < n; ++i) {
        hasBad = hasBad || std::isnan(grid[i]) || std::isinf(grid[i]);
    }
    if (hasBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min: minVal) reduction(max: maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
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
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", nranks, omp_get_max_threads());
    }

    // Bind each rank to a GPU (round-robin over devices visible on the node)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    }

    // 1D domain decomposition of the z dimension into contiguous slabs.
    // Ranks 0..(nz%nranks - 1) get one extra plane, so any empty ranks are
    // a contiguous tail.
    const size_t baseZ = nz / nranks;
    const size_t remZ = nz % nranks;
    const size_t lnz = baseZ + ((size_t)rank < remZ ? 1 : 0);
    const size_t z0 = (size_t)rank * baseZ + std::min((size_t)rank, remZ);
    const int activeRanks = (int)std::min((size_t)nranks, nz);
    const bool active = lnz > 0;
    const int prevRank = (rank > 0 && active) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localSize = plane * (lnz + 2);  // includes halo planes

    // Host slab (pinned for fast transfers) and device double buffers
    Real* h_slab = nullptr;
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    Real* h_sendLo = nullptr;
    Real* h_sendHi = nullptr;
    Real* h_recvLo = nullptr;
    Real* h_recvHi = nullptr;
    if (active) {
        CUDA_CHECK(cudaMallocHost(&h_slab, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_a, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvHi, plane * sizeof(Real)));
    }

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    if (active) {
        initializeGrid(h_slab, nx, ny, z0, lnz);
        CUDA_CHECK(cudaMemcpy(d_a, h_slab, localSize * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_b, 0, localSize * sizeof(Real)));
    }

    cudaStream_t sComp = nullptr, sHalo = nullptr;
    if (active) {
        CUDA_CHECK(cudaStreamCreate(&sComp));
        CUDA_CHECK(cudaStreamCreate(&sHalo));
    }

    const dim3 block(64, 4, 2);
    auto launchPlanes = [&](const Real* d_in, Real* d_out, size_t kFirst,
                            size_t kLast, cudaStream_t stream) {
        const size_t nk = kLast - kFirst + 1;
        const dim3 gridDim((unsigned)((nx + block.x - 1) / block.x),
                           (unsigned)((ny + block.y - 1) / block.y),
                           (unsigned)((nk + block.z - 1) / block.z));
        stencilKernel<<<gridDim, block, 0, stream>>>(d_in, d_out, nx, ny, nz,
                                                     kFirst, kLast, z0);
        CUDA_CHECK(cudaGetLastError());
    };

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_a : d_b;
        Real* d_out = (iter % 2 == 0) ? d_b : d_a;

        // Overlap: stage boundary planes of the input buffer for the halo
        // exchange while the interior planes are computed on sComp.
        if (active) {
            if (nextRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_in + plane * lnz,
                                           plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sHalo));
            }
            if (prevRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_in + plane,
                                           plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sHalo));
            }
            if (lnz > 2) {
                launchPlanes(d_in, d_out, 2, lnz - 1, sComp);
            }
            CUDA_CHECK(cudaStreamSynchronize(sHalo));
            MPI_Sendrecv(h_sendHi, (int)plane, MPI_DOUBLE, nextRank, 0,
                         h_recvLo, (int)plane, MPI_DOUBLE, prevRank, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(h_sendLo, (int)plane, MPI_DOUBLE, prevRank, 1,
                         h_recvHi, (int)plane, MPI_DOUBLE, nextRank, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (prevRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in, h_recvLo,
                                           plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, sHalo));
            }
            if (nextRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in + plane * (lnz + 1), h_recvHi,
                                           plane * sizeof(Real),
                                           cudaMemcpyHostToDevice, sHalo));
            }

            // Edge planes depend on the freshly received halos (same stream)
            launchPlanes(d_in, d_out, 1, 1, sHalo);
            if (lnz > 1) {
                launchPlanes(d_in, d_out, lnz, lnz, sHalo);
            }
            CUDA_CHECK(cudaStreamSynchronize(sHalo));
            CUDA_CHECK(cudaStreamSynchronize(sComp));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const double localElapsed = tEnd - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = (long)(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Bring the final buffer (matching the original's parity) back to the host
    if (active) {
        const Real* d_final = (iterations % 2 == 0) ? d_a : d_b;
        CUDA_CHECK(cudaMemcpy(h_slab, d_final, localSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Gather the full grid on rank 0 for result output and validation
    std::vector<Real> finalGrid;
    {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t rlnz = baseZ + ((size_t)r < remZ ? 1 : 0);
            const size_t rz0 = (size_t)r * baseZ + std::min((size_t)r, remZ);
            counts[r] = (int)(rlnz * plane);
            displs[r] = (int)(rz0 * plane);
        }
        if (rank == 0) finalGrid.resize(nx * ny * nz);
        MPI_Gatherv(active ? h_slab + plane : nullptr, (int)(lnz * plane), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (active) {
        CUDA_CHECK(cudaStreamDestroy(sComp));
        CUDA_CHECK(cudaStreamDestroy(sHalo));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFreeHost(h_slab));
        CUDA_CHECK(cudaFreeHost(h_sendLo));
        CUDA_CHECK(cudaFreeHost(h_sendHi));
        CUDA_CHECK(cudaFreeHost(h_recvLo));
        CUDA_CHECK(cudaFreeHost(h_recvHi));
    }

    MPI_Finalize();
    return exitCode;
}
