#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Local layout per rank: plane 0 and plane nzl+1 are halo (ghost) planes,
// planes 1..nzl are owned. Global z of local plane p is z0 + p - 1.

// 7-point stencil over local planes [pBegin, pEnd). Global boundary points are copied.
// The operation order matches the original host code exactly.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              size_t nx, size_t ny, size_t nz, size_t z0,
                              size_t pBegin, size_t pEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t nxy = nx * ny;
    for (size_t p = pBegin + blockIdx.z; p < pEnd; p += gridDim.z) {
        const size_t gz = z0 + p - 1;
        const size_t idx = p * nxy + y * nx + x;
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == nz - 1) {
            out[idx] = in[idx];
        } else {
            const Real center = in[idx];
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            const Real bottom = in[idx - nxy];
            const Real top = in[idx + nxy];
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

// Initialize owned planes with the same values as the original global index formula.
__global__ void initKernel(Real* __restrict__ buf, size_t nx, size_t ny, size_t z0, size_t nzl) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t nxy = nx * ny;
    for (size_t p = 1 + blockIdx.z; p <= nzl; p += gridDim.z) {
        const size_t globalIdx = (z0 + p - 1) * nxy + y * nx + x;
        buf[p * nxy + y * nx + x] = (globalIdx % 19) * 1.0;
    }
}

static dim3 planeBlock() { return dim3(32, 8, 1); }

static dim3 planeGrid(size_t nx, size_t ny, size_t numPlanes) {
    const dim3 block = planeBlock();
    return dim3(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                static_cast<unsigned int>((ny + block.y - 1) / block.y),
                static_cast<unsigned int>(std::min<size_t>(numPlanes, 65535)));
}

static void launchStencil(cudaStream_t s, const Real* in, Real* out,
                          size_t nx, size_t ny, size_t nz, size_t z0,
                          size_t pBegin, size_t pEnd) {
    if (pEnd <= pBegin) return;
    stencilKernel<<<planeGrid(nx, ny, pEnd - pBegin), planeBlock(), 0, s>>>(
        in, out, nx, ny, nz, z0, pBegin, pEnd);
    CUDA_CHECK(cudaGetLastError());
}

// Validation on this rank's owned slab (OpenMP), then reduced across ranks.
// Produces the same checks and messages as the original single-process validateResult.
static bool validateResult(const std::vector<Real>& local, MPI_Comm comm, bool root) {
    const long long n = static_cast<long long>(local.size());
    int nonFinite = 0;
    Real minVal = local[0];
    Real maxVal = local[0];

#pragma omp parallel for reduction(| : nonFinite) reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < n; ++i) {
        const Real val = local[i];
        nonFinite |= (std::isnan(val) || std::isinf(val)) ? 1 : 0;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    int globalNonFinite = 0;
    MPI_Allreduce(&nonFinite, &globalNonFinite, 1, MPI_INT, MPI_LOR, comm);
    if (globalNonFinite) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (root) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        if (root) printf("Validation failed: values out of expected range\n");
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

static int runBenchmark(int argc, char** argv, int rank, int numRanks) {
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Grid dimensions must be positive\n");
        return 1;
    }
    if (static_cast<size_t>(numRanks) > nz) {
        if (root) printf("Number of MPI ranks (%d) exceeds Z grid size (%zu)\n", numRanks, nz);
        return 1;
    }

    const size_t nxy = nx * ny;
    if (nxy > static_cast<size_t>(INT_MAX)) {
        if (root) printf("Grid plane size exceeds supported MPI message size\n");
        return 1;
    }
    const int nxyInt = static_cast<int>(nxy);

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind this rank to a GPU on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Slab decomposition along Z. Every rank owns at least one plane.
    const size_t base = nz / numRanks;
    const size_t rem = nz % numRanks;
    auto planesOf = [&](int r) { return base + (static_cast<size_t>(r) < rem ? 1 : 0); };
    auto startOf = [&](int r) {
        return static_cast<size_t>(r) * base + std::min<size_t>(static_cast<size_t>(r), rem);
    };
    const size_t nzl = planesOf(rank);
    const size_t z0 = startOf(rank);
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Datatype planeType;
    MPI_Type_contiguous(nxyInt, MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Device double buffers (owned planes plus two halo planes)
    const size_t localElems = (nzl + 2) * nxy;
    Real* bufA = nullptr;
    Real* bufB = nullptr;
    CUDA_CHECK(cudaMalloc(&bufA, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&bufB, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(bufA, 0, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(bufB, 0, localElems * sizeof(Real)));

    // Pinned host staging buffers for halo planes
    Real* sendLo = nullptr;
    Real* sendHi = nullptr;
    Real* recvLo = nullptr;
    Real* recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendLo, nxy * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendHi, nxy * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLo, nxy * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvHi, nxy * sizeof(Real)));

    cudaStream_t computeStream, haloStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    cudaEvent_t readyEvent, haloEvent;
    CUDA_CHECK(cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloEvent, cudaEventDisableTiming));

    // Initialize
    if (root) printf("Initializing grid...\n");
    initKernel<<<planeGrid(nx, ny, nzl), planeBlock(), 0, computeStream>>>(bufA, nx, ny, z0, nzl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(readyEvent, computeStream));

    Real* cur = bufA;
    Real* next = bufB;

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // Interior planes need no halo data; queue them first so the GPU stays busy
        // while the halo exchange is in flight.
        if (nzl >= 3) {
            launchStencil(computeStream, cur, next, nx, ny, nz, z0, 2, nzl);
        }

        // Halo exchange, staged through pinned host memory
        CUDA_CHECK(cudaStreamWaitEvent(haloStream, readyEvent, 0));
        CUDA_CHECK(cudaMemcpyAsync(sendLo, cur + nxy, nxy * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaMemcpyAsync(sendHi, cur + nzl * nxy, nxy * sizeof(Real), cudaMemcpyDeviceToHost, haloStream));
        CUDA_CHECK(cudaStreamSynchronize(haloStream));

        MPI_Request reqs[4];
        MPI_Irecv(recvLo, nxyInt, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recvHi, nxyInt, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(sendHi, nxyInt, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sendLo, nxyInt, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (down != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(cur, recvLo, nxy * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        }
        if (up != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(cur + (nzl + 1) * nxy, recvHi, nxy * sizeof(Real), cudaMemcpyHostToDevice, haloStream));
        }
        CUDA_CHECK(cudaEventRecord(haloEvent, haloStream));

        // Boundary-adjacent owned planes need the halo
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloEvent, 0));
        launchStencil(computeStream, cur, next, nx, ny, nz, z0, 1, 2);
        if (nzl > 1) {
            launchStencil(computeStream, cur, next, nx, ny, nz, z0, nzl, nzl + 1);
        }
        CUDA_CHECK(cudaEventRecord(readyEvent, computeStream));

        std::swap(cur, next);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy this rank's owned slab of the final grid to host (only needed for -v / -r)
    std::vector<Real> hostOwned;
    if (validate || printResults) {
        hostOwned.resize(nzl * nxy);
        CUDA_CHECK(cudaMemcpy(hostOwned.data(), cur + nxy, nzl * nxy * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        const size_t gridSize = nx * ny * nz;
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(planesOf(r));
            displs[r] = static_cast<int>(startOf(r));
        }
        std::vector<Real> finalGrid;
        if (root) finalGrid.resize(gridSize);
        MPI_Gatherv(hostOwned.data(), static_cast<int>(nzl), planeType,
                    finalGrid.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        if (root) print_results(finalGrid, "Grid");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateResult(hostOwned, MPI_COMM_WORLD, root);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
        } else {
            if (root) printf("Validation: FAILED\n");
            status = 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(readyEvent));
    CUDA_CHECK(cudaEventDestroy(haloEvent));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaFreeHost(sendLo));
    CUDA_CHECK(cudaFreeHost(sendHi));
    CUDA_CHECK(cudaFreeHost(recvLo));
    CUDA_CHECK(cudaFreeHost(recvHi));
    CUDA_CHECK(cudaFree(bufA));
    CUDA_CHECK(cudaFree(bufB));
    MPI_Type_free(&planeType);

    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int status = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return status;
}
