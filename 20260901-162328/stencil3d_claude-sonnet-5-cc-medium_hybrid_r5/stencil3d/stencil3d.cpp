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

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t _err = (call);                                                 \
        if (_err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_err));                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

// 3D index calculation (host, global-layout semantics preserved)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// GPU kernel: computes the 7-point averaging stencil for a range of local
// z-planes [zLocalLo, zLocalHi] (inclusive). Local array layout has one ghost
// plane at local z = 0 and one ghost plane at local z = nzLocal + 1. Points
// on the *global* domain boundary are copied through unchanged, exactly as
// in the original single-process implementation.
// ---------------------------------------------------------------------------
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               int nx, int ny,
                               long globalZStart,   // global z index of local plane z=1
                               long nzGlobal,
                               int zLocalLo, int zLocalHi) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zl = zLocalLo + blockIdx.z;

    if (x >= nx || y >= ny || zl > zLocalHi) return;

    const size_t planeSize = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(zl) * planeSize + static_cast<size_t>(y) * nx + x;
    const long globalZ = globalZStart + (zl - 1);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || globalZ == 0 || globalZ == nzGlobal - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - planeSize];
    const Real top = input[idx + planeSize];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// Initializes the local (owned, no ghosts) portion of the grid on the host,
// using global indices so results are bit-identical to a non-distributed run.
void initializeGridLocal(std::vector<Real>& hostLocal, size_t nx, size_t ny, size_t nzLocal, size_t globalZStart) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < nzLocal; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t z = globalZStart + zl;
                const size_t globalIdx = idx3(x, y, z, nx, ny);
                hostLocal[zl * (nx * ny) + y * nx + x] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// Local (per-rank) validation: checks the owned sub-grid (no ghosts) for
// NaN/Inf and computes the local min/max, using OpenMP for the reduction.
void validateLocal(const std::vector<Real>& hostLocal, bool& hasInvalid, Real& localMin, Real& localMax) {
    hasInvalid = false;
    localMin = hostLocal[0];
    localMax = hostLocal[0];

    #pragma omp parallel
    {
        bool threadInvalid = false;
        Real tMin = hostLocal[0];
        Real tMax = hostLocal[0];

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < hostLocal.size(); ++i) {
            const Real val = hostLocal[i];
            if (std::isnan(val) || std::isinf(val)) threadInvalid = true;
            tMin = std::min(tMin, val);
            tMax = std::max(tMax, val);
        }

        #pragma omp critical
        {
            if (threadInvalid) hasInvalid = true;
            localMin = std::min(localMin, tMin);
            localMax = std::max(localMax, tMax);
        }
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

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

    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid depth (nz=%zu)\n", nranks, nz);
        }
        MPI_Finalize();
        return 1;
    }

    // ---- Bind this rank to a GPU (node-local round robin) ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "Error: no CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA hybrid)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
        printf("CUDA devices visible per node: %d\n", deviceCount);
        printf("OpenMP max threads per rank: %d\n", omp_get_max_threads());
    }

    // ---- Domain decomposition along Z (contiguous block distribution) ----
    const size_t baseZ = nz / static_cast<size_t>(nranks);
    const size_t remZ = nz % static_cast<size_t>(nranks);
    auto zCountFor = [&](int r) -> size_t { return baseZ + (static_cast<size_t>(r) < remZ ? 1 : 0); };
    auto zStartFor = [&](int r) -> size_t {
        size_t start = 0;
        for (int i = 0; i < r; ++i) start += zCountFor(i);
        return start;
    };

    const size_t nzLocal = zCountFor(rank);
    const size_t zOffset = zStartFor(rank);
    const size_t planeSize = nx * ny;
    const size_t nzExt = nzLocal + 2;  // + 2 ghost planes

    const int neighborDown = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int neighborUp = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;

    // ---- Host pinned buffers ----
    std::vector<Real> hostLocal(nzLocal * planeSize);
    Real* pinnedSendDown = nullptr;
    Real* pinnedSendUp = nullptr;
    Real* pinnedRecvDown = nullptr;
    Real* pinnedRecvUp = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinnedSendDown, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&pinnedSendUp, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&pinnedRecvDown, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&pinnedRecvUp, planeSize * sizeof(Real)));

    // ---- Initialize (OpenMP on host, then upload to device) ----
    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(hostLocal, nx, ny, nzLocal, zOffset);

    Real *dGrid1 = nullptr, *dGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dGrid1, nzExt * planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dGrid2, nzExt * planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dGrid1 + planeSize, hostLocal.data(), nzLocal * planeSize * sizeof(Real), cudaMemcpyHostToDevice));

    cudaStream_t streamCompute, streamHalo;
    CUDA_CHECK(cudaStreamCreate(&streamCompute));
    CUDA_CHECK(cudaStreamCreate(&streamHalo));

    dim3 block(32, 8, 1);
    dim3 gridEdge((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    // ---- Run stencil iterations ----
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* dIn = (iter % 2 == 0) ? dGrid1 : dGrid2;
        Real* dOut = (iter % 2 == 0) ? dGrid2 : dGrid1;

        #pragma omp parallel num_threads(2)
        {
            CUDA_CHECK(cudaSetDevice(deviceId));

            #pragma omp sections
            {
                // ---- Section 1: interior points, no MPI dependency ----
                #pragma omp section
                {
                    const long zLo = 2;
                    const long zHi = static_cast<long>(nzLocal) - 1;
                    if (zLo <= zHi) {
                        dim3 gridInterior((nx + block.x - 1) / block.x,
                                           (ny + block.y - 1) / block.y,
                                           static_cast<unsigned int>(zHi - zLo + 1));
                        stencilKernel<<<gridInterior, block, 0, streamCompute>>>(
                            dIn, dOut, static_cast<int>(nx), static_cast<int>(ny),
                            static_cast<long>(zOffset), static_cast<long>(nz),
                            static_cast<int>(zLo), static_cast<int>(zHi));
                    }
                    CUDA_CHECK(cudaStreamSynchronize(streamCompute));
                }

                // ---- Section 2: halo exchange + edge-plane compute ----
                #pragma omp section
                {
                    // Pack: copy owned boundary planes (local z=1 and z=nzLocal) to host
                    if (neighborDown != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(pinnedSendDown, dIn + planeSize, planeSize * sizeof(Real),
                                                    cudaMemcpyDeviceToHost, streamHalo));
                    }
                    if (neighborUp != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(pinnedSendUp, dIn + nzLocal * planeSize, planeSize * sizeof(Real),
                                                    cudaMemcpyDeviceToHost, streamHalo));
                    }
                    CUDA_CHECK(cudaStreamSynchronize(streamHalo));

                    MPI_Sendrecv(pinnedSendDown, static_cast<int>(planeSize), MPI_DOUBLE, neighborDown, 0,
                                 pinnedRecvUp, static_cast<int>(planeSize), MPI_DOUBLE, neighborUp, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Sendrecv(pinnedSendUp, static_cast<int>(planeSize), MPI_DOUBLE, neighborUp, 1,
                                 pinnedRecvDown, static_cast<int>(planeSize), MPI_DOUBLE, neighborDown, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if (neighborDown != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(dIn, pinnedRecvDown, planeSize * sizeof(Real),
                                                    cudaMemcpyHostToDevice, streamHalo));
                    }
                    if (neighborUp != MPI_PROC_NULL) {
                        CUDA_CHECK(cudaMemcpyAsync(dIn + (nzLocal + 1) * planeSize, pinnedRecvUp, planeSize * sizeof(Real),
                                                    cudaMemcpyHostToDevice, streamHalo));
                    }
                    CUDA_CHECK(cudaStreamSynchronize(streamHalo));

                    // Compute the (at most two) boundary-adjacent local planes.
                    if (nzLocal == 1) {
                        dim3 gridEdge1(gridEdge.x, gridEdge.y, 1);
                        stencilKernel<<<gridEdge1, block, 0, streamHalo>>>(
                            dIn, dOut, static_cast<int>(nx), static_cast<int>(ny),
                            static_cast<long>(zOffset), static_cast<long>(nz), 1, 1);
                    } else {
                        stencilKernel<<<gridEdge, block, 0, streamHalo>>>(
                            dIn, dOut, static_cast<int>(nx), static_cast<int>(ny),
                            static_cast<long>(zOffset), static_cast<long>(nz), 1, 1);
                        stencilKernel<<<gridEdge, block, 0, streamHalo>>>(
                            dIn, dOut, static_cast<int>(nx), static_cast<int>(ny),
                            static_cast<long>(zOffset), static_cast<long>(nz),
                            static_cast<int>(nzLocal), static_cast<int>(nzLocal));
                    }
                    CUDA_CHECK(cudaStreamSynchronize(streamHalo));
                }
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = duration.count();
    long globalMs = 0;
    MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalMs);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (globalMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Download final owned data (no ghosts) ----
    Real* dFinal = (iterations % 2 == 0) ? dGrid1 : dGrid2;
    CUDA_CHECK(cudaMemcpy(hostLocal.data(), dFinal + planeSize, nzLocal * planeSize * sizeof(Real), cudaMemcpyDeviceToHost));

    // ---- Print results for external validation (requires full gather) ----
    if (printResults) {
        std::vector<int> recvCounts, displs;
        std::vector<Real> fullGrid;
        if (rank == 0) {
            recvCounts.resize(nranks);
            displs.resize(nranks);
            int offset = 0;
            for (int r = 0; r < nranks; ++r) {
                recvCounts[r] = static_cast<int>(zCountFor(r) * planeSize);
                displs[r] = offset;
                offset += recvCounts[r];
            }
            fullGrid.resize(nx * ny * nz);
        }
        MPI_Gatherv(hostLocal.data(), static_cast<int>(nzLocal * planeSize), MPI_DOUBLE,
                    rank == 0 ? fullGrid.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(fullGrid, "Grid");
        }
    }

    // ---- Validation (distributed: local check + reduction, no full gather needed) ----
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        bool hasInvalidLocal = false;
        Real localMin = 0.0, localMax = 0.0;
        validateLocal(hostLocal, hasInvalidLocal, localMin, localMax);

        int hasInvalidLocalInt = hasInvalidLocal ? 1 : 0;
        int hasInvalidGlobal = 0;
        Real globalMin = 0.0, globalMax = 0.0;
        MPI_Allreduce(&hasInvalidLocalInt, &hasInvalidGlobal, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        bool valid = true;
        if (hasInvalidGlobal) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        }

        if (rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);

        if (globalMax > 1e6 || globalMin < -1e6) {
            if (rank == 0) printf("Validation failed: values out of expected range\n");
            valid = false;
        }

        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(streamCompute));
    CUDA_CHECK(cudaStreamDestroy(streamHalo));
    CUDA_CHECK(cudaFree(dGrid1));
    CUDA_CHECK(cudaFree(dGrid2));
    CUDA_CHECK(cudaFreeHost(pinnedSendDown));
    CUDA_CHECK(cudaFreeHost(pinnedSendUp));
    CUDA_CHECK(cudaFreeHost(pinnedRecvDown));
    CUDA_CHECK(cudaFreeHost(pinnedRecvUp));

    MPI_Finalize();
    return exitCode;
}
