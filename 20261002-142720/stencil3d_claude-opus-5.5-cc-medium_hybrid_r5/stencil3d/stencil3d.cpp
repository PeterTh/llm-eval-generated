#include <algorithm>
#include <chrono>
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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;

// Initialize local slab (including halo planes). Local plane k maps to global z = z0 + k - 1.
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t nx, const size_t ny, const size_t nz,
                                     const long long z0, const long long nplanes) {
    const size_t plane = nx * ny;
    const size_t total = plane * (size_t)nplanes;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const long long k = (long long)(i / plane);
        const long long gz = z0 + k - 1;
        if (gz < 0 || gz >= (long long)nz) continue;
        const size_t gidx = (size_t)gz * plane + (i % plane);
        grid[i] = (gidx % 19) * 1.0;
    }
}

// Correctly rounded s / 7.0 (bit-identical to IEEE division for finite s):
// q = RN(s * RN(1/7)) is within 1 ulp, one FMA residual correction rounds it exactly (Markstein).
// Much cheaper than the generic double-precision division sequence.
__device__ __forceinline__ Real divideBy7(const Real s) {
    constexpr Real inv7 = 1.0 / 7.0;
    const Real q = __dmul_rn(s, inv7);
    const Real r = __fma_rn(-q, 7.0, s);
    return __fma_rn(r, inv7, q);
}

// 7-point stencil on local planes [kbeg, kend), interior x/y only.
// Each thread marches along z, keeping the z-neighbours in registers.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int kbeg, const int kend) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const int k0 = kbeg + blockIdx.z * Z_CHUNK;
    if (k0 >= kend) return;
    const int k1 = min(k0 + Z_CHUNK, kend);

    const size_t plane = (size_t)nx * ny;
    size_t idx = (size_t)k0 * plane + (size_t)y * nx + x;
    Real bottom = input[idx - plane];
    Real center = input[idx];
    for (int k = k0; k < k1; ++k) {
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        output[idx] = divideBy7(center + left + right + front + back + bottom + top);
        bottom = center;
        center = top;
        idx += plane;
    }
}

static void launchStencil(const Real* in, Real* out, const size_t nx, const size_t ny,
                          const long long kbeg, const long long kend, cudaStream_t stream) {
    if (kend <= kbeg || nx < 3 || ny < 3) return;
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    dim3 grid((unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X), (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y),
              (unsigned)((kend - kbeg + Z_CHUNK - 1) / Z_CHUNK));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, (int)nx, (int)ny, (int)kbeg, (int)kend);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks: single parallel pass for NaN/Inf detection and min/max
    const size_t n = grid.size();
    const Real* g = grid.data();
    int bad = 0;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = g[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    // 1. No NaN or Inf values
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // 1D domain decomposition along z: rank r owns global planes [z0, z0 + lnz)
    const int active = (int)std::min<size_t>((size_t)nranks, std::max<size_t>(nz, 1));
    const bool isActive = rank < active;
    long long lnz = 0, z0 = 0;
    if (isActive) {
        const long long base = (long long)nz / active, rem = (long long)nz % active;
        lnz = base + (rank < rem ? 1 : 0);
        z0 = rank * base + std::min<long long>(rank, rem);
    }
    const int down = (isActive && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (isActive && rank < active - 1) ? rank + 1 : MPI_PROC_NULL;

    // Select GPU by node-local rank
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    // Local slab with one halo plane on each side (double buffering)
    const long long nplanes = lnz + 2;
    const size_t localElems = isActive ? plane * (size_t)nplanes : 0;
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    Real* h_send = nullptr;  // [0, plane): to down, [plane, 2*plane): to up
    Real* h_recv = nullptr;  // [0, plane): from down, [plane, 2*plane): from up
    cudaStream_t sEdge, sInner;
    // Halo planes go on a high-priority stream so they are not queued behind the bulk kernel
    int prioLeast = 0, prioGreatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sEdge, cudaStreamNonBlocking, prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sInner, cudaStreamNonBlocking, prioLeast));

    // Initialize
    if (root) printf("Initializing grid...\n");
    if (isActive && localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_a, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_send, 2 * plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recv, 2 * plane * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_a, 0, localElems * sizeof(Real)));
        initializeGridKernel<<<1024, 256>>>(d_a, nx, ny, nz, z0, nplanes);
        CUDA_CHECK(cudaGetLastError());
        // Boundary values are only ever copied, so both buffers start with identical contents
        CUDA_CHECK(cudaMemcpy(d_b, d_a, localElems * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Local planes that are updated (global z in [1, nz-2])
    const long long kLo = std::max<long long>(1, 2 - z0);
    const long long kHi = std::min<long long>(lnz, (long long)nz - 1 - z0);  // inclusive
    const int planeCount = (int)plane;
    
    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    Real* in = d_a;
    Real* out = d_b;
    if (isActive && localElems > 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            // Edge planes first (needed by neighbours), then the bulk on a separate stream
            const bool needDown = (down != MPI_PROC_NULL), needUp = (up != MPI_PROC_NULL);
            long long innerLo = kLo, innerHi = kHi;
            if (needDown && innerLo <= 1 && 1 <= innerHi) {
                launchStencil(in, out, nx, ny, 1, 2, sEdge);
                innerLo = 2;
            }
            if (needUp && lnz >= innerLo && lnz <= innerHi) {
                launchStencil(in, out, nx, ny, lnz, lnz + 1, sEdge);
                innerHi = lnz - 1;
            }
            if (needDown)
                CUDA_CHECK(cudaMemcpyAsync(h_send, out + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
            if (needUp)
                CUDA_CHECK(cudaMemcpyAsync(h_send + plane, out + (size_t)lnz * plane, plane * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sEdge));
            launchStencil(in, out, nx, ny, innerLo, innerHi + 1, sInner);

            if (needDown || needUp) {
                MPI_Request reqs[4];
                MPI_Irecv(h_recv, planeCount, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[0]);
                MPI_Irecv(h_recv + plane, planeCount, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[1]);
                CUDA_CHECK(cudaStreamSynchronize(sEdge));
                MPI_Isend(h_send, planeCount, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[2]);
                MPI_Isend(h_send + plane, planeCount, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[3]);
                MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
                if (needDown)
                    CUDA_CHECK(cudaMemcpyAsync(out, h_recv, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
                if (needUp)
                    CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(lnz + 1) * plane, h_recv + plane, plane * sizeof(Real),
                                               cudaMemcpyHostToDevice, sEdge));
            }
            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            CUDA_CHECK(cudaStreamSynchronize(sInner));
            std::swap(in, out);
        }
        CUDA_CHECK(cudaGetLastError());
    }
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

    // Gather final grid on root (only when it is needed for output)
    int exitCode = 0;
    if (printResults || validate) {
        // Original semantics: a negative odd iteration count ends on the untouched second buffer (all zeros)
        const bool zeroResult = (iterations < 0) && (iterations % 2 != 0);
        std::vector<Real> localData(isActive ? (size_t)lnz * plane : 0);
        if (isActive && !localData.empty())
            CUDA_CHECK(cudaMemcpy(localData.data(), in + plane, localData.size() * sizeof(Real), cudaMemcpyDeviceToHost));

        MPI_Datatype planeType;
        MPI_Type_contiguous(planeCount, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        std::vector<int> counts, displs;
        std::vector<Real> finalGrid;
        if (root) {
            counts.resize(nranks, 0);
            displs.resize(nranks, 0);
            const long long base = (long long)nz / active, rem = (long long)nz % active;
            for (int r = 0; r < active; ++r) {
                counts[r] = (int)(base + (r < rem ? 1 : 0));
                displs[r] = (int)(r * base + std::min<long long>(r, rem));
            }
            finalGrid.resize(gridSize);
        }
        MPI_Gatherv(localData.data(), (int)lnz, planeType, finalGrid.data(), counts.data(), displs.data(), planeType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) {
            if (zeroResult) std::fill(finalGrid.begin(), finalGrid.end(), Real(0));

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
    }

    if (d_a) CUDA_CHECK(cudaFree(d_a));
    if (d_b) CUDA_CHECK(cudaFree(d_b));
    if (h_send) CUDA_CHECK(cudaFreeHost(h_send));
    if (h_recv) CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaStreamDestroy(sInner));
    MPI_Finalize();
    return exitCode;
}
