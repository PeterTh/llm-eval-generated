#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx,
                              const int ny,
                              const int nzLocal,
                              const int startZ,
                              const int nzGlobal) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    for (int k = 1; k <= nzLocal; ++k) {
        const int gz = startZ + k - 1;
        const size_t idx = static_cast<size_t>(k) * plane + base;

        if (xyBoundary || gz == 0 || gz == nzGlobal - 1) {
            output[idx] = input[idx];
        } else {
            output[idx] = (input[idx] +
                           input[idx - 1] +
                           input[idx + 1] +
                           input[idx - nx] +
                           input[idx + nx] +
                           input[idx - plane] +
                           input[idx + plane]) / 7.0;
        }
    }
}

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static inline int ownerOfZ(const int z, const int nz, const int size) {
    return static_cast<int>(((static_cast<long long>(z) + 1) * size - 1) / nz);
}

static void exchangeHalos(Real* dGrid,
                          const int nx,
                          const int ny,
                          const int nzLocal,
                          const int bottomRank,
                          const int topRank,
                          Real* hSend,
                          Real* hRecv) {
    if (nzLocal <= 0) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t bytes = plane * sizeof(Real);

    checkCuda(cudaMemcpy(hSend, dGrid + plane, bytes, cudaMemcpyDeviceToHost), "halo pack");
    checkCuda(cudaMemcpy(hSend + plane, dGrid + static_cast<size_t>(nzLocal) * plane, bytes,
                         cudaMemcpyDeviceToHost), "halo pack");

    MPI_Request requests[4];
    MPI_Irecv(hRecv, static_cast<int>(plane), MPI_DOUBLE, topRank, 0,
              MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(hRecv + plane, static_cast<int>(plane), MPI_DOUBLE, bottomRank, 1,
              MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(hSend, static_cast<int>(plane), MPI_DOUBLE, bottomRank, 0,
              MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(hSend + plane, static_cast<int>(plane), MPI_DOUBLE, topRank, 1,
              MPI_COMM_WORLD, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    checkCuda(cudaMemcpy(dGrid + static_cast<size_t>(nzLocal + 1) * plane, hRecv, bytes,
                         cudaMemcpyHostToDevice), "halo unpack");
    checkCuda(cudaMemcpy(dGrid, hRecv + plane, bytes, cudaMemcpyHostToDevice), "halo unpack");
}

void initializeGrid(Real* hostGrid, const int nx, const int ny, const int nzLocal, const int startZ) {
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nzLocal + 2; ++k) {
        for (int y = 0; y < ny; ++y) {
            const size_t row = static_cast<size_t>(k) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx);
            if (k == 0 || k == nzLocal + 1) {
                for (int x = 0; x < nx; ++x) {
                    hostGrid[row + static_cast<size_t>(x)] = 0.0;
                }
            } else {
                const int gz = startZ + k - 1;
                for (int x = 0; x < nx; ++x) {
                    const size_t gidx = idx3(static_cast<size_t>(x), static_cast<size_t>(y),
                                             static_cast<size_t>(gz),
                                             static_cast<size_t>(nx), static_cast<size_t>(ny));
                    hostGrid[row + static_cast<size_t>(x)] = static_cast<Real>(gidx % 19);
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool bad = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    const long long n = static_cast<long long>(grid.size());
    #pragma omp parallel for reduction(||:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Real val = grid[static_cast<size_t>(i)];
        if (std::isnan(val) || std::isinf(val)) {
            bad = true;
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device found\n");
        MPI_Finalize();
        return 1;
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    int nx = 128;
    int ny = 0;
    int nz = 0;
    int iterations = 10;
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
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int plane = nx * ny;
    const int startZ = static_cast<int>((static_cast<long long>(rank) * nz) / size);
    const int endZ = static_cast<int>((static_cast<long long>(rank + 1) * nz) / size);
    const int nzLocal = endZ - startZ;
    const int bottomRank = (startZ > 0) ? ownerOfZ(startZ - 1, nz, size) : MPI_PROC_NULL;
    const int topRank = (endZ < nz) ? ownerOfZ(endZ, nz, size) : MPI_PROC_NULL;

    const size_t planeSz = static_cast<size_t>(plane);
    const size_t ownedCount = static_cast<size_t>(nzLocal) * planeSz;
    const size_t paddedCount = static_cast<size_t>(nzLocal + 2) * planeSz;
    const size_t allocCount = paddedCount > 0 ? paddedCount : 1;
    const size_t hostCount = paddedCount > 0 ? paddedCount : 1;

    Real* dGrid1 = nullptr;
    Real* dGrid2 = nullptr;
    Real* hGrid = nullptr;
    Real* hSend = nullptr;
    Real* hRecv = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dGrid1), allocCount * sizeof(Real)), "cudaMalloc");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dGrid2), allocCount * sizeof(Real)), "cudaMalloc");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hGrid), hostCount * sizeof(Real)), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hSend), 2 * planeSz * sizeof(Real)), "cudaMallocHost");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hRecv), 2 * planeSz * sizeof(Real)), "cudaMallocHost");
    std::memset(hRecv, 0, 2 * planeSz * sizeof(Real));

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(hGrid, nx, ny, nzLocal, startZ);
    if (paddedCount > 0) {
        checkCuda(cudaMemcpy(dGrid1, hGrid, paddedCount * sizeof(Real), cudaMemcpyHostToDevice), "initial upload");
    }

    if (iterations > 0 && nzLocal > 0) {
        exchangeHalos(dGrid1, nx, ny, nzLocal, bottomRank, topRank, hSend, hRecv);
    }

    if (rank == 0) printf("Running stencil computation...\n");

    const dim3 block(32, 8);
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y);

    auto start = std::chrono::high_resolution_clock::now();

    Real* dCurrent = dGrid1;
    Real* dNext = dGrid2;
    for (int iter = 0; iter < iterations; ++iter) {
        if (nzLocal > 0) {
            stencilKernel<<<grid, block>>>(dCurrent, dNext, nx, ny, nzLocal, startZ, nz);
            checkCuda(cudaGetLastError(), "stencilKernel");
        }
        if (iter + 1 < iterations) {
            exchangeHalos(dNext, nx, ny, nzLocal, bottomRank, topRank, hSend, hRecv);
        }
        Real* tmp = dCurrent;
        dCurrent = dNext;
        dNext = tmp;
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedSec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    long long elapsedMs = static_cast<long long>(elapsedSec * 1000.0);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", elapsedMs);
        double cellUpdates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                             static_cast<double>(nz - 2) * static_cast<double>(iterations);
        double mcups = elapsedSec > 0.0 ? cellUpdates / elapsedSec / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (ownedCount > 0) {
        checkCuda(cudaMemcpy(hGrid, dCurrent + planeSz, ownedCount * sizeof(Real),
                             cudaMemcpyDeviceToHost), "result download");
    }

    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const int rs = static_cast<int>((static_cast<long long>(r) * nz) / size);
        const int re = static_cast<int>((static_cast<long long>(r + 1) * nz) / size);
        recvCounts[r] = (re - rs) * plane;
        displs[r] = rs * plane;
    }

    std::vector<Real> fullGrid;
    if (rank == 0) {
        fullGrid.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));
    }

    MPI_Gatherv(hGrid, static_cast<int>(ownedCount), MPI_DOUBLE,
                rank == 0 ? fullGrid.data() : nullptr,
                recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullGrid, static_cast<size_t>(nx),
                                        static_cast<size_t>(ny), static_cast<size_t>(nz));
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    int globalExit = 0;
    MPI_Allreduce(&exitCode, &globalExit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    checkCuda(cudaFree(dGrid1), "cudaFree");
    checkCuda(cudaFree(dGrid2), "cudaFree");
    checkCuda(cudaFreeHost(hGrid), "cudaFreeHost");
    checkCuda(cudaFreeHost(hSend), "cudaFreeHost");
    checkCuda(cudaFreeHost(hRecv), "cudaFreeHost");

    MPI_Finalize();
    return globalExit;
}
