#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Block distribution of the nz global z-planes over the active ranks.
// If there are fewer planes than ranks, the surplus ranks own no planes and stay idle.
void planeRange(const size_t nz, const int rank, const int size, size_t& z0, size_t& n) {
    const size_t active = std::min(static_cast<size_t>(size), nz);
    const size_t r = rank;
    if (r >= active) {
        z0 = nz;
        n = 0;
        return;
    }
    const size_t base = nz / active;
    const size_t rem = nz % active;
    n = base + (r < rem ? 1 : 0);
    z0 = r * base + std::min(r, rem);
}

// Host-side initialization of this rank's owned planes (OpenMP parallel).
// Values match the global index formula of the original program.
void initializeSlab(std::vector<Real>& slab, const size_t nx, const size_t ny, const size_t z0, const size_t nLocal) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nLocal; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z0 + z, nx, ny);
                slab[idx3(x, y, z, nx, ny)] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil over local planes [kBegin, kEnd).
// Local plane k holds global plane z0 + k - 1; planes 0 and n + 1 are halos.
// The summation order matches the original so results are bitwise identical.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t z0, const size_t kBegin, const size_t kEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t k = kBegin + blockIdx.z; k < kEnd; k += gridDim.z) {
        const size_t gz = z0 + k - 1;
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
            const size_t idx = idx3(x, y, k, nx, ny);
            if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || gz == 0 || gz == nz-1) {
                output[idx] = input[idx];
            } else {
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - nx * ny];
                const Real top = input[idx + nx * ny];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

void launchStencil(cudaStream_t stream, const Real* input, Real* output,
                   const size_t nx, const size_t ny, const size_t nz,
                   const size_t z0, const size_t kBegin, const size_t kEnd) {
    if (kBegin >= kEnd || nx == 0 || ny == 0) return;
    const dim3 block(64, 4);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>(std::min<size_t>((ny + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned>(std::min<size_t>(kEnd - kBegin, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(input, output, nx, ny, nz, z0, kBegin, kEnd);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool nonFinite = false;
    #pragma omp parallel for reduction(||:nonFinite)
    for (size_t i = 0; i < grid.size(); ++i) {
        nonFinite = nonFinite || std::isnan(grid[i]) || std::isinf(grid[i]);
    }
    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
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

// Per-rank state: a slab of z-planes with one halo plane on each side,
// resident on this rank's GPU, plus the streams/events that overlap work.
struct Slab {
    int rank = 0;
    int size = 1;
    int activeRanks = 1;                // ranks that own at least one z-plane
    size_t nx = 0, ny = 0, nz = 0;      // global grid
    size_t z0 = 0;                      // first global z-plane owned by this rank
    size_t n = 0;                       // number of owned z-planes
    size_t planeElems = 0;              // nx * ny
    Real* dev[2] = {nullptr, nullptr};  // double buffers, (n + 2) planes each
    Real* hostSendDown = nullptr;       // pinned staging for halo traffic
    Real* hostSendUp = nullptr;
    Real* hostRecvDown = nullptr;
    Real* hostRecvUp = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t transfer = nullptr;
    cudaEvent_t computeDone = nullptr;  // recorded after each sweep
    cudaEvent_t haloReady = nullptr;    // recorded after halos land in the input buffer
};

constexpr int kTagUp = 0;    // message travelling to the rank above
constexpr int kTagDown = 1;  // message travelling to the rank below

// Refresh the halo planes of `buf` from the neighbouring ranks' boundary planes.
// Host staging is used for the MPI transfers, so no CUDA-aware MPI is required.
void exchangeHalos(Slab& s, Real* buf) {
    const bool hasDown = s.n > 0 && s.rank > 0;
    const bool hasUp = s.n > 0 && s.rank + 1 < s.activeRanks;
    const size_t planeBytes = s.planeElems * sizeof(Real);

    // The input buffer must be complete before its boundary planes are read
    CUDA_CHECK(cudaStreamWaitEvent(s.transfer, s.computeDone, 0));
    if (hasDown) {
        CUDA_CHECK(cudaMemcpyAsync(s.hostSendDown, buf + s.planeElems, planeBytes, cudaMemcpyDeviceToHost, s.transfer));
    }
    if (hasUp) {
        CUDA_CHECK(cudaMemcpyAsync(s.hostSendUp, buf + s.n * s.planeElems, planeBytes, cudaMemcpyDeviceToHost, s.transfer));
    }
    CUDA_CHECK(cudaStreamSynchronize(s.transfer));

    MPI_Request req[4];
    int nreq = 0;
    if (hasDown) {
        MPI_Irecv(s.hostRecvDown, static_cast<int>(s.planeElems), MPI_DOUBLE, s.rank - 1, kTagUp, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(s.hostSendDown, static_cast<int>(s.planeElems), MPI_DOUBLE, s.rank - 1, kTagDown, MPI_COMM_WORLD, &req[nreq++]);
    }
    if (hasUp) {
        MPI_Irecv(s.hostRecvUp, static_cast<int>(s.planeElems), MPI_DOUBLE, s.rank + 1, kTagDown, MPI_COMM_WORLD, &req[nreq++]);
        MPI_Isend(s.hostSendUp, static_cast<int>(s.planeElems), MPI_DOUBLE, s.rank + 1, kTagUp, MPI_COMM_WORLD, &req[nreq++]);
    }
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    if (hasDown) {
        CUDA_CHECK(cudaMemcpyAsync(buf, s.hostRecvDown, planeBytes, cudaMemcpyHostToDevice, s.transfer));
    }
    if (hasUp) {
        CUDA_CHECK(cudaMemcpyAsync(buf + (s.n + 1) * s.planeElems, s.hostRecvUp, planeBytes, cudaMemcpyHostToDevice, s.transfer));
    }
    CUDA_CHECK(cudaEventRecord(s.haloReady, s.transfer));
}

// One Jacobi sweep: in -> out. Interior planes need no halo data, so they are
// queued first and run while the halo exchange proceeds on the host and transfer stream.
void sweep(Slab& s, Real* in, Real* out) {
    if (s.n == 0) return;  // idle rank
    launchStencil(s.compute, in, out, s.nx, s.ny, s.nz, s.z0, 2, s.n);
    exchangeHalos(s, in);
    CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.haloReady, 0));
    launchStencil(s.compute, in, out, s.nx, s.ny, s.nz, s.z0, 1, 2);
    if (s.n > 1) {
        launchStencil(s.compute, in, out, s.nx, s.ny, s.nz, s.z0, s.n, s.n + 1);
    }
    CUDA_CHECK(cudaEventRecord(s.computeDone, s.compute));
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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) {
            fprintf(stderr, "Grid dimensions must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank, round-robin over the GPUs visible on this node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Share host cores among the ranks on this node unless the user chose otherwise
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    Slab s;
    s.rank = rank;
    s.size = size;
    s.activeRanks = static_cast<int>(std::min(static_cast<size_t>(size), nz));
    s.nx = nx;
    s.ny = ny;
    s.nz = nz;
    planeRange(nz, rank, size, s.z0, s.n);
    s.planeElems = nx * ny;
    const size_t plane = s.planeElems;
    const size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering), zero-initialised like the original vectors
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&s.dev[b], (s.n + 2) * plane * sizeof(Real)));
        CUDA_CHECK(cudaMemset(s.dev[b], 0, (s.n + 2) * plane * sizeof(Real)));
    }
    CUDA_CHECK(cudaMallocHost(&s.hostSendDown, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&s.hostSendUp, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&s.hostRecvDown, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&s.hostRecvUp, plane * sizeof(Real)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.transfer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.computeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.haloReady, cudaEventDisableTiming));

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> initial(s.n * plane);
    initializeSlab(initial, nx, ny, s.z0, s.n);
    CUDA_CHECK(cudaMemcpy(s.dev[0] + plane, initial.data(), initial.size() * sizeof(Real), cudaMemcpyHostToDevice));

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(s.computeDone, s.compute));
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            sweep(s, s.dev[0], s.dev[1]);
        } else {
            sweep(s, s.dev[1], s.dev[0]);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(s.compute));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long maxMs = 0;
    long long localMs = duration.count();
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0 for output and validation
    const size_t finalBuf = (iterations % 2 == 0) ? 0 : 1;
    int status = 0;
    if (printResults || validate) {
        std::vector<Real> owned(s.n * plane);
        CUDA_CHECK(cudaMemcpy(owned.data(), s.dev[finalBuf] + plane, owned.size() * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<Real> finalGrid(rank == 0 ? gridSize : size_t(0));
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            size_t zr = 0;
            size_t nr = 0;
            planeRange(nz, r, size, zr, nr);
            counts[r] = static_cast<int>(nr);
            displs[r] = static_cast<int>(zr);
        }
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(owned.data(), static_cast<int>(s.n), planeType,
                    finalGrid.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(finalGrid, "Grid");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(finalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(s.dev[0]));
    CUDA_CHECK(cudaFree(s.dev[1]));
    CUDA_CHECK(cudaFreeHost(s.hostSendDown));
    CUDA_CHECK(cudaFreeHost(s.hostSendUp));
    CUDA_CHECK(cudaFreeHost(s.hostRecvDown));
    CUDA_CHECK(cudaFreeHost(s.hostRecvUp));
    CUDA_CHECK(cudaEventDestroy(s.computeDone));
    CUDA_CHECK(cudaEventDestroy(s.haloReady));
    CUDA_CHECK(cudaStreamDestroy(s.compute));
    CUDA_CHECK(cudaStreamDestroy(s.transfer));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return status;
}
