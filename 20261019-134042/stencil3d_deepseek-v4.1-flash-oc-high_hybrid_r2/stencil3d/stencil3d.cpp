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

using Real = double;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_),       \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 7-point stencil kernel. Each rank owns a contiguous range of global z planes.
// The local grid stores the owned planes in [1, localCount] plus one halo plane
// on each side (local indices 0 and localCount + 1).
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx, const int ny, const int nz,
                              const int zBase) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int lz = blockIdx.z + 1;
    const int gz = zBase + blockIdx.z;
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(lz) * plane +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
        gz == 0 || gz == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    // Preserve the original left-to-right summation order.
    const Real sum = input[idx] +
                     input[idx - 1] +
                     input[idx + 1] +
                     input[idx - static_cast<size_t>(nx)] +
                     input[idx + static_cast<size_t>(nx)] +
                     input[idx - plane] +
                     input[idx + plane];
    output[idx] = sum / 7.0;
}

static inline int localCountFor(const int nz, const int size, const int rank) {
    const int base = nz / size;
    const int rem = nz % size;
    return base + (rank < rem ? 1 : 0);
}

static inline int startFor(const int nz, const int size, const int rank) {
    const int base = nz / size;
    const int rem = nz % size;
    return rank * base + std::min(rank, rem);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    int bad = 0;
    Real minVal = grid.empty() ? Real(0) : grid[0];
    Real maxVal = grid.empty() ? Real(0) : grid[0];

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) reduction(| : bad)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = 1;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Select one GPU per rank on each node.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
    }
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    bool exitOk = false;
    bool exitError = false;
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
            exitOk = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitError = true;
            break;
        }
    }

    if (exitOk || exitError) {
        if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return exitError ? 1 : 0;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    const int zStart = startFor(inz, size, rank);
    const int localCount = localCountFor(inz, size, rank);
    const int localNz = localCount + 2;
    const size_t localSize = static_cast<size_t>(localNz) * plane;
    const bool hasData = localCount > 0;

    int lower = MPI_PROC_NULL;
    int upper = MPI_PROC_NULL;
    if (hasData) {
        if (rank > 0 && localCountFor(inz, size, rank - 1) > 0) lower = rank - 1;
        if (rank + 1 < size && localCountFor(inz, size, rank + 1) > 0) upper = rank + 1;
    }

    // Allocate grids (double buffering) on the device.
    Real* dGrid[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGrid[0]), localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGrid[1]), localSize * sizeof(Real)));

    // Initialize the local block including its halo planes.
    if (rank == 0) printf("Initializing grid...\n");
    {
        std::vector<Real> hostLocal(localSize);
#pragma omp parallel for collapse(2)
        for (int lz = 0; lz < localNz; ++lz) {
            for (int y = 0; y < iny; ++y) {
                const int gz = zStart + lz - 1;
                for (int x = 0; x < inx; ++x) {
                    const size_t lidx = static_cast<size_t>(lz) * plane +
                                        static_cast<size_t>(y) * nx +
                                        static_cast<size_t>(x);
                    if (gz < 0 || gz >= inz) {
                        hostLocal[lidx] = Real(0);
                    } else {
                        const size_t gidx = idx3(static_cast<size_t>(x),
                                                 static_cast<size_t>(y),
                                                 static_cast<size_t>(gz), nx, ny);
                        hostLocal[lidx] = (gidx % 19) * 1.0;
                    }
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(dGrid[0], hostLocal.data(), localSize * sizeof(Real),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dGrid[1], hostLocal.data(), localSize * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    // Pinned host buffers used for halo exchange.
    Real* sendTop = nullptr;
    Real* sendBottom = nullptr;
    Real* recvTop = nullptr;
    Real* recvBottom = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendTop), plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&sendBottom), plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvTop), plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&recvBottom), plane * sizeof(Real)));

    const int mpiPlaneCount = static_cast<int>(plane);
    const dim3 block(32, 8, 1);
    const dim3 grid((inx + block.x - 1) / block.x,
                    (iny + block.y - 1) / block.y,
                    localCount);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const int inIdx = iter & 1;
        const int outIdx = (iter + 1) & 1;
        Real* dIn = dGrid[inIdx];
        Real* dOut = dGrid[outIdx];

        if (hasData) {
            stencilKernel<<<grid, block>>>(dIn, dOut, inx, iny, inz, zStart);
            CUDA_CHECK(cudaGetLastError());
        }

        if (iter + 1 < iterations) {
            if (hasData) {
                // Extract the owned boundary planes that neighbors need next iteration.
                CUDA_CHECK(cudaMemcpy(sendTop,
                                      dOut + static_cast<size_t>(localCount) * plane,
                                      plane * sizeof(Real), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(sendBottom,
                                      dOut + plane,
                                      plane * sizeof(Real), cudaMemcpyDeviceToHost));

                // Exchange with the upper/lower neighbors.
                MPI_Sendrecv(sendTop, mpiPlaneCount, MPI_DOUBLE, upper, 201,
                             recvBottom, mpiPlaneCount, MPI_DOUBLE, lower, 201,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(sendBottom, mpiPlaneCount, MPI_DOUBLE, lower, 200,
                             recvTop, mpiPlaneCount, MPI_DOUBLE, upper, 200,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Place received planes into the output halos for the next iteration.
                if (lower != MPI_PROC_NULL) {
                    CUDA_CHECK(cudaMemcpy(dOut,
                                          recvBottom,
                                          plane * sizeof(Real), cudaMemcpyHostToDevice));
                }
                if (upper != MPI_PROC_NULL) {
                    CUDA_CHECK(cudaMemcpy(dOut + static_cast<size_t>(localCount + 1) * plane,
                                          recvTop,
                                          plane * sizeof(Real), cudaMemcpyHostToDevice));
                }
            } else {
                // Ranks without owned planes still participate in the exchange pattern.
                MPI_Sendrecv(sendTop, 0, MPI_DOUBLE, upper, 201,
                             recvBottom, 0, MPI_DOUBLE, lower, 201,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(sendBottom, 0, MPI_DOUBLE, lower, 200,
                             recvTop, 0, MPI_DOUBLE, upper, 200,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the final state when it is needed for output/validation.
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        std::vector<Real> localOwned(static_cast<size_t>(localCount) * plane);
        if (hasData) {
            const int finalIdx = iterations & 1;
            CUDA_CHECK(cudaMemcpy(localOwned.data(),
                                  dGrid[finalIdx] + plane,
                                  localOwned.size() * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(size, 0);
        std::vector<int> displs(size, 0);
        for (int r = 0; r < size; ++r) {
            counts[r] = localCountFor(inz, size, r) * mpiPlaneCount;
            displs[r] = startFor(inz, size, r) * mpiPlaneCount;
        }

        if (rank == 0) finalGrid.resize(gridSize);
        MPI_Gatherv(hasData ? localOwned.data() : nullptr,
                    hasData ? localCount * mpiPlaneCount : 0,
                    MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(finalGrid, "Grid");
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(finalGrid, nx, ny, nz) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaFreeHost(sendTop));
    CUDA_CHECK(cudaFreeHost(sendBottom));
    CUDA_CHECK(cudaFreeHost(recvTop));
    CUDA_CHECK(cudaFreeHost(recvBottom));
    CUDA_CHECK(cudaFree(dGrid[0]));
    CUDA_CHECK(cudaFree(dGrid[1]));
    if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);

    MPI_Finalize();
    return valid ? 0 : 1;
}
