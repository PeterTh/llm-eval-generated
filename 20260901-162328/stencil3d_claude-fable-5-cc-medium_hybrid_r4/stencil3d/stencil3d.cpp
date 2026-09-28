// Hybrid MPI + OpenMP + CUDA 3D stencil benchmark.
//
// Parallelization strategy:
//  - MPI: 1D slab decomposition of the grid along Z; one GPU per rank
//    (selected round-robin by node-local rank). Halo planes are exchanged
//    every iteration, overlapped with the interior CUDA kernel.
//  - CUDA: the 7-point stencil update runs entirely on the GPU with double
//    buffering in device memory (interior kernel + boundary-plane kernel).
//  - OpenMP: host-side grid initialization and result validation.

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil update for one output point at local plane lz (global plane gz).
// Global boundary points are copied from the input, matching the serial code.
__device__ inline void stencilPoint(const Real* __restrict__ in, Real* __restrict__ out,
                                    const size_t x, const size_t y, const size_t lz,
                                    const size_t gz, const size_t nx, const size_t ny,
                                    const size_t gnz) {
    const size_t idx = idx3(x, y, lz, nx, ny);
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == gnz - 1) {
        out[idx] = in[idx];
        return;
    }
    const Real center = in[idx];
    const Real left   = in[idx - 1];
    const Real right  = in[idx + 1];
    const Real front  = in[idx - nx];
    const Real back   = in[idx + nx];
    const Real bottom = in[idx - nx * ny];
    const Real top    = in[idx + nx * ny];
    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// Updates owned planes lz in [2, nzl-1] (those that do not read ghost planes).
__global__ void stencilInterior(const Real* __restrict__ in, Real* __restrict__ out,
                                const size_t nx, const size_t ny, const size_t nzl,
                                const size_t gz0, const size_t gnz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 2;
    if (x >= nx || y >= ny || lz > nzl - 1) return;
    stencilPoint(in, out, x, y, lz, gz0 + lz - 1, nx, ny, gnz);
}

// Updates the first (lz = 1) and last (lz = nzl) owned planes, which depend on
// the ghost planes received from the MPI neighbors.
__global__ void stencilBoundary(const Real* __restrict__ in, Real* __restrict__ out,
                                const size_t nx, const size_t ny, const size_t nzl,
                                const size_t gz0, const size_t gnz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t lz = (blockIdx.z == 0) ? 1 : nzl;
    stencilPoint(in, out, x, y, lz, gz0 + lz - 1, nx, ny, gnz);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    const size_t n = grid.size();
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    // 1. No NaN or Inf values; 2. track value range
#pragma omp parallel for reduction(||:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        bad = bad || std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks, omp_get_max_threads());
    }

    // Select a GPU per rank, round-robin over the devices visible on this node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // 1D slab decomposition along Z.
    const size_t base = nz / (size_t)nranks;
    const size_t rem = nz % (size_t)nranks;
    const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);          // owned planes
    const size_t gz0 = (size_t)rank * base + std::min((size_t)rank, rem);  // first owned global plane
    const int nActive = (int)std::min((size_t)nranks, nz);          // ranks with nzl > 0

    const int downNeighbor = (rank > 0 && rank < nActive) ? rank - 1 : MPI_PROC_NULL;
    const int upNeighbor = (rank + 1 < nActive) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeSize = nx * ny;
    const size_t localSize = planeSize * (nzl + 2);  // owned planes + 2 ghost planes
    size_t gridSize = nx * ny * nz;

    // Initialize the local slab on the host (including ghost planes, so the
    // first iteration needs no exchange of initial data).
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> hostGrid(std::max(localSize, (size_t)1), 0.0);
    if (nzl > 0) {
        // Fill owned planes and ghost planes with the initial pattern, using
        // global indices so the result matches the serial initialization.
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t lz = 0; lz < nzl + 2; ++lz) {
            for (size_t y = 0; y < ny; ++y) {
                const long long gz = (long long)gz0 + (long long)lz - 1;
                for (size_t x = 0; x < nx; ++x) {
                    Real v = 0.0;
                    if (gz >= 0 && gz < (long long)nz) {
                        const size_t gidx = idx3(x, y, (size_t)gz, nx, ny);
                        v = (gidx % 19) * 1.0;
                    }
                    hostGrid[lz * planeSize + y * nx + x] = v;
                }
            }
        }
    }

    // Device buffers (double buffering).
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    if (nzl > 0) {
        CUDA_CHECK(cudaMalloc(&d_a, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_a, hostGrid.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, hostGrid.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice));
    }

    // Pinned host staging buffers for halo exchange.
    Real* h_sendLo = nullptr;
    Real* h_sendHi = nullptr;
    Real* h_recvLo = nullptr;
    Real* h_recvHi = nullptr;
    if (nzl > 0) {
        CUDA_CHECK(cudaMallocHost(&h_sendLo, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvLo, planeSize * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvHi, planeSize * sizeof(Real)));
    }

    cudaStream_t computeStream, commStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&commStream));

    const dim3 blockI(64, 4, 2);
    const dim3 blockB(64, 8, 1);
    const size_t nInterior = (nzl > 2) ? nzl - 2 : 0;
    const dim3 gridI((unsigned)((nx + blockI.x - 1) / blockI.x),
                     (unsigned)((ny + blockI.y - 1) / blockI.y),
                     (unsigned)((nInterior + blockI.z - 1) / blockI.z));
    const dim3 gridB((unsigned)((nx + blockB.x - 1) / blockB.x),
                     (unsigned)((ny + blockB.y - 1) / blockB.y),
                     (nzl > 1) ? 2u : 1u);

    Real* d_in = d_a;
    Real* d_out = d_b;

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (nzl > 0) {
            // Stage the outermost owned planes of the input for the halo exchange.
            if (downNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_in + planeSize, planeSize * sizeof(Real),
                                           cudaMemcpyDeviceToHost, commStream));
            }
            if (upNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_in + nzl * planeSize, planeSize * sizeof(Real),
                                           cudaMemcpyDeviceToHost, commStream));
            }
            CUDA_CHECK(cudaStreamSynchronize(commStream));

            // Interior planes do not touch ghosts: overlap them with the exchange.
            if (nInterior > 0) {
                stencilInterior<<<gridI, blockI, 0, computeStream>>>(d_in, d_out, nx, ny, nzl, gz0, nz);
                CUDA_CHECK(cudaGetLastError());
            }

            MPI_Sendrecv(h_sendLo, (int)planeSize, MPI_DOUBLE, downNeighbor, 0,
                         h_recvHi, (int)planeSize, MPI_DOUBLE, upNeighbor, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(h_sendHi, (int)planeSize, MPI_DOUBLE, upNeighbor, 1,
                         h_recvLo, (int)planeSize, MPI_DOUBLE, downNeighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (downNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in, h_recvLo, planeSize * sizeof(Real),
                                           cudaMemcpyHostToDevice, commStream));
            }
            if (upNeighbor != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in + (nzl + 1) * planeSize, h_recvHi, planeSize * sizeof(Real),
                                           cudaMemcpyHostToDevice, commStream));
            }

            // Ghost-dependent planes run after the ghosts have arrived.
            stencilBoundary<<<gridB, blockB, 0, commStream>>>(d_in, d_out, nx, ny, nzl, gz0, nz);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::swap(d_in, d_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const double localElapsed = tEnd - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = (long)(maxElapsed * 1000.0);

    // Download the owned planes of the final grid.
    if (nzl > 0) {
        CUDA_CHECK(cudaMemcpy(hostGrid.data() + planeSize, d_in + planeSize,
                              nzl * planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Gather the full grid on rank 0 for output/validation.
    std::vector<Real> finalGrid;
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t rNzl = base + ((size_t)r < rem ? 1 : 0);
        const size_t rGz0 = (size_t)r * base + std::min((size_t)r, rem);
        counts[r] = (int)(rNzl * planeSize);
        displs[r] = (int)(rGz0 * planeSize);
    }
    if (rank == 0) finalGrid.resize(gridSize);
    MPI_Gatherv(hostGrid.data() + planeSize, (int)(nzl * planeSize), MPI_DOUBLE,
                finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(commStream));
    if (nzl > 0) {
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFreeHost(h_sendLo));
        CUDA_CHECK(cudaFreeHost(h_sendHi));
        CUDA_CHECK(cudaFreeHost(h_recvLo));
        CUDA_CHECK(cudaFreeHost(h_recvHi));
    }

    MPI_Finalize();
    return exitCode;
}
