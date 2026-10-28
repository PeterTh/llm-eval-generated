#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 2);                                     \
        }                                                                     \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int kTagUp = 0;        // message travels toward higher rank
constexpr int kTagDown = 1;      // message travels toward lower rank
constexpr int kTagGather = 2;    // final-grid gather
constexpr size_t kMaxChunk = size_t(1) << 28;  // elements per MPI message in gather
constexpr size_t kTileZ = 32;    // z-planes processed per thread column in the kernel

// Contiguous z-slab owned by one MPI rank (global planes [z0, z0 + lz)).
struct Slab {
    size_t z0;
    size_t lz;
};

Slab slabFor(const size_t rank, const size_t nranks, const size_t nz) {
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    return Slab{rank * base + std::min(rank, rem), base + (rank < rem ? 1 : 0)};
}

// Initialize the owned slab (packed, lz planes) with the same values as the
// global grid initialization: grid[idx] = (idx % 19) * 1.0 with global idx.
void initializeSlab(std::vector<Real>& slab, const size_t nx, const size_t ny, const size_t z0, const size_t lz) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < lz; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z0 + zl, nx, ny);
                slab[idx3(x, y, zl, nx, ny)] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil over z-planes [zStart, zStart + zCount) of a local buffer whose
// plane 0 and plane (nz_local-1) are ghost planes. Each thread owns one (x, y)
// column and marches along z in tiles of kTileZ planes, keeping the z-neighbours
// in registers. The addition order matches the original CPU code exactly.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const size_t nx, const size_t ny,
                              const size_t zStart, const size_t zCount) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    if (x > nx - 2 || y > ny - 2) return;

    const size_t zBeg = zStart + static_cast<size_t>(blockIdx.z) * kTileZ;
    if (zBeg >= zStart + zCount) return;
    size_t zEnd = zBeg + kTileZ;
    if (zEnd > zStart + zCount) zEnd = zStart + zCount;

    const size_t plane = nx * ny;
    size_t base = zBeg * plane + y * nx + x;
    Real below = in[base - plane];
    Real center = in[base];
    for (size_t z = zBeg; z < zEnd; ++z, base += plane) {
        const Real top = in[base + plane];
        const Real left = in[base - 1];
        const Real right = in[base + 1];
        const Real front = in[base - nx];
        const Real back = in[base + nx];
        out[base] = (center + left + right + front + back + below + top) / 7.0;
        below = center;
        center = top;
    }
}

void launchStencil(cudaStream_t stream, const Real* in, Real* out,
                   const size_t nx, const size_t ny, const size_t zStart, const size_t zCount) {
    if (zCount == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                    static_cast<unsigned>((ny - 2 + block.y - 1) / block.y),
                    static_cast<unsigned>((zCount + kTileZ - 1) / kTileZ));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, zStart, zCount);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool nonFinite = false;
#pragma omp parallel for reduction(||:nonFinite)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) nonFinite = true;
    }
    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0 || nx * ny > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) printf("Error: grid dimensions must be positive and nx*ny must fit in int\n");
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each node-local rank to one GPU
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Each active rank must own at least one z-plane
    const int activeRanks = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    const size_t plane = nx * ny;
    const size_t gridSize = nx * ny * nz;
    const Slab mine = slabFor(static_cast<size_t>(rank), static_cast<size_t>(nranks), nz);
    const size_t z0 = mine.z0;
    const size_t lz = mine.lz;
    const int planeCount = static_cast<int>(plane);

    // Local double-buffered storage: planes [0, lz+1] are owned planes 1..lz
    // surrounded by ghost planes 0 and lz+1.
    const size_t bufElems = (lz + 2) * plane;
    Real* bufA = nullptr;
    Real* bufB = nullptr;
    CUDA_CHECK(cudaMalloc(&bufA, bufElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&bufB, bufElems * sizeof(Real)));
    CUDA_CHECK(cudaMemset(bufA, 0, bufElems * sizeof(Real)));

    // Pinned host staging buffers for halo exchange
    Real* sendLo = nullptr;
    Real* sendHi = nullptr;
    Real* recvLo = nullptr;
    Real* recvHi = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendHi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvLo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvHi, plane * sizeof(Real)));

    cudaStream_t sc;   // compute stream
    cudaStream_t scomm; // halo transfer stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&scomm, cudaStreamNonBlocking));
    cudaEvent_t evReady;
    cudaEvent_t evD2H;
    cudaEvent_t evHalo;
    CUDA_CHECK(cudaEventCreateWithFlags(&evReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evD2H, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

    // Initialize
    if (worldRank == 0) printf("Initializing grid...\n");
    {
        std::vector<Real> slab(lz * plane);
        initializeSlab(slab, nx, ny, z0, lz);
        CUDA_CHECK(cudaMemcpy(bufA + plane, slab.data(), lz * plane * sizeof(Real), cudaMemcpyHostToDevice));
    }
    // Second buffer starts as a copy of the first, including the fixed boundary
    // values, which the original code also carries in both buffers.
    CUDA_CHECK(cudaMemcpy(bufB, bufA, bufElems * sizeof(Real), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(evReady, sc));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    if (worldRank == 0) printf("Running stencil computation...\n");

    const bool hasLo = rank > 0;
    const bool hasHi = rank + 1 < nranks;
    // Local planes that are updated: global z in [1, nz-2]
    const size_t jlo = (z0 == 0) ? 2 : 1;
    const size_t jhi = (z0 + lz == nz) ? lz - 1 : lz;
    const bool updates = (nx >= 3) && (ny >= 3) && (jlo <= jhi);

    Real* in = bufA;
    Real* out = bufB;

    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        // Stage boundary planes of the input to the host once it is ready
        CUDA_CHECK(cudaStreamWaitEvent(scomm, evReady, 0));
        if (hasLo) CUDA_CHECK(cudaMemcpyAsync(sendLo, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, scomm));
        if (hasHi) CUDA_CHECK(cudaMemcpyAsync(sendHi, in + lz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, scomm));
        CUDA_CHECK(cudaEventRecord(evD2H, scomm));

        // Interior planes need no ghost data: compute them while the halo moves
        if (updates && jhi >= jlo + 2) {
            launchStencil(sc, in, out, nx, ny, jlo + 1, jhi - jlo - 1);
        }

        CUDA_CHECK(cudaEventSynchronize(evD2H));

        MPI_Request reqs[4];
        int nreq = 0;
        if (hasLo) {
            MPI_Irecv(recvLo, planeCount, MPI_DOUBLE, rank - 1, kTagUp, comm, &reqs[nreq++]);
            MPI_Isend(sendLo, planeCount, MPI_DOUBLE, rank - 1, kTagDown, comm, &reqs[nreq++]);
        }
        if (hasHi) {
            MPI_Irecv(recvHi, planeCount, MPI_DOUBLE, rank + 1, kTagDown, comm, &reqs[nreq++]);
            MPI_Isend(sendHi, planeCount, MPI_DOUBLE, rank + 1, kTagUp, comm, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        if (hasLo) CUDA_CHECK(cudaMemcpyAsync(in, recvLo, plane * sizeof(Real), cudaMemcpyHostToDevice, scomm));
        if (hasHi) CUDA_CHECK(cudaMemcpyAsync(in + (lz + 1) * plane, recvHi, plane * sizeof(Real), cudaMemcpyHostToDevice, scomm));
        CUDA_CHECK(cudaEventRecord(evHalo, scomm));

        // Boundary planes depend on the freshly received ghost planes
        CUDA_CHECK(cudaStreamWaitEvent(sc, evHalo, 0));
        if (updates) {
            launchStencil(sc, in, out, nx, ny, jlo, 1);
            if (jhi != jlo) launchStencil(sc, in, out, nx, ny, jhi, 1);
        }
        CUDA_CHECK(cudaEventRecord(evReady, sc));

        std::swap(in, out);
    }

    CUDA_CHECK(cudaStreamSynchronize(sc));
    CUDA_CHECK(cudaStreamSynchronize(scomm));
    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    // After the loop, 'in' holds the most recent result (grid1 for even iteration counts)
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid and run output/validation on rank 0
    int status = 0;
    if (printResults || validate) {
        if (rank == 0) {
            std::vector<Real> full(gridSize);
            CUDA_CHECK(cudaMemcpy(full.data(), in + plane, lz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
            for (int r = 1; r < nranks; ++r) {
                const Slab s = slabFor(static_cast<size_t>(r), static_cast<size_t>(nranks), nz);
                Real* dst = full.data() + s.z0 * plane;
                const size_t count = s.lz * plane;
                for (size_t off = 0; off < count; off += kMaxChunk) {
                    const size_t n = std::min(kMaxChunk, count - off);
                    MPI_Recv(dst + off, static_cast<int>(n), MPI_DOUBLE, r, kTagGather, comm, MPI_STATUS_IGNORE);
                }
            }

            if (printResults) {
                print_results(full, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        } else {
            std::vector<Real> owned(lz * plane);
            CUDA_CHECK(cudaMemcpy(owned.data(), in + plane, lz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
            const size_t count = lz * plane;
            for (size_t off = 0; off < count; off += kMaxChunk) {
                const size_t n = std::min(kMaxChunk, count - off);
                MPI_Send(owned.data() + off, static_cast<int>(n), MPI_DOUBLE, 0, kTagGather, comm);
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);

    CUDA_CHECK(cudaEventDestroy(evReady));
    CUDA_CHECK(cudaEventDestroy(evD2H));
    CUDA_CHECK(cudaEventDestroy(evHalo));
    CUDA_CHECK(cudaStreamDestroy(sc));
    CUDA_CHECK(cudaStreamDestroy(scomm));
    CUDA_CHECK(cudaFreeHost(sendLo));
    CUDA_CHECK(cudaFreeHost(sendHi));
    CUDA_CHECK(cudaFreeHost(recvLo));
    CUDA_CHECK(cudaFreeHost(recvHi));
    CUDA_CHECK(cudaFree(bufA));
    CUDA_CHECK(cudaFree(bufB));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
