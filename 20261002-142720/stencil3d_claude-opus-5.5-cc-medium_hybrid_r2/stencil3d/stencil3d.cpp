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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread block shape for the stencil kernel (x,y tile; each thread marches along z)
constexpr int BX = 32;
constexpr int BY = 8;
constexpr int ZCHUNK = 16;

// Initialize local planes [0, nPlanes) which map to global planes starting at gz0.
__global__ void initializeKernel(Real* __restrict__ grid1, Real* __restrict__ grid2,
                                 const size_t planeSize, const size_t nPlanes, const size_t gz0) {
    const size_t total = planeSize * nPlanes;
    const size_t base = gz0 * planeSize;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const Real v = ((base + i) % 19) * 1.0;
        grid1[i] = v;
        grid2[i] = v;
    }
}

// 7-point stencil on local planes [zBegin, zEnd) (local indices into a buffer that
// includes halo planes). Only interior x/y are updated; boundary values never change,
// so they are kept identical in both buffers from initialization on.
__global__ void __launch_bounds__(BX * BY)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int zBegin, const int zEnd) {
    const int x = blockIdx.x * BX + threadIdx.x + 1;
    const int y = blockIdx.y * BY + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = zBegin + blockIdx.z * ZCHUNK;
    const int z1 = min(z0 + ZCHUNK, zEnd);
    if (z0 >= z1) return;

    const Real inv7 = 1.0 / 7.0;
    const size_t plane = (size_t)nx * ny;
    size_t idx = (size_t)z0 * plane + (size_t)y * nx + x;

    Real bottom = __ldg(&input[idx - plane]);
    Real center = __ldg(&input[idx]);
    for (int z = z0; z < z1; ++z) {
        const Real top = __ldg(&input[idx + plane]);
        const Real left = __ldg(&input[idx - 1]);
        const Real right = __ldg(&input[idx + 1]);
        const Real front = __ldg(&input[idx - nx]);
        const Real back = __ldg(&input[idx + nx]);
        // Same summation order as the reference for bitwise-identical results
        const Real sum = center + left + right + front + back + bottom + top;
        // Correctly rounded sum / 7.0 (Markstein: q = RN(s*r), q' = q + RN(s - q*7)*r),
        // avoiding the costly FP64 division on GPUs with reduced FP64 throughput
        const Real q = sum * inv7;
        output[idx] = fma(fma(-q, 7.0, sum), inv7, q);
        bottom = center;
        center = top;
        idx += plane;
    }
}

static void launchStencil(const Real* in, Real* out, int nx, int ny, int zBegin, int zEnd,
                          cudaStream_t stream) {
    if (zEnd <= zBegin || nx < 3 || ny < 3) return;
    dim3 block(BX, BY, 1);
    dim3 grid((nx - 2 + BX - 1) / BX, (ny - 2 + BY - 1) / BY,
              (zEnd - zBegin + ZCHUNK - 1) / ZCHUNK);
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, zBegin, zEnd);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Real val = grid[i];
        bad = bad || std::isnan(val) || std::isinf(val);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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
    const size_t planeSize = nx * ny;

    // ---- Domain decomposition along z (slabs). Ranks without planes are idle. ----
    const size_t nRanks = (size_t)worldSize;
    const size_t activeRanks = std::max<size_t>(1, std::min(nRanks, nz));
    const bool active = (size_t)worldRank < activeRanks;
    size_t nLocal = 0, gz0 = 0;
    if (active) {
        const size_t base = nz / activeRanks, rem = nz % activeRanks;
        const size_t r = (size_t)worldRank;
        nLocal = base + (r < rem ? 1 : 0);
        gz0 = r * base + std::min(r, rem);
    }
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    // ---- Select GPU by node-local rank ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const int rank = worldRank;
    const bool hasLower = active && gz0 > 0;                    // neighbor at rank-1
    const bool hasUpper = active && gz0 + nLocal < nz;          // neighbor at rank+1
    // Local buffer: [lower halo?][nLocal owned planes][upper halo?]
    const size_t off = hasLower ? 1 : 0;
    const size_t nAlloc = nLocal + off + (hasUpper ? 1 : 0);
    const size_t gAlloc0 = gz0 - off;  // global z of local plane 0

    Real* d1 = nullptr;
    Real* d2 = nullptr;
    Real* hSend[2] = {nullptr, nullptr};  // [0]=lower, [1]=upper
    Real* hRecv[2] = {nullptr, nullptr};
    cudaStream_t sInner = nullptr, sEdge = nullptr;
    const size_t planeBytes = planeSize * sizeof(Real);

    if (active && nAlloc > 0 && planeSize > 0) {
        CUDA_CHECK(cudaMalloc(&d1, nAlloc * planeBytes));
        CUDA_CHECK(cudaMalloc(&d2, nAlloc * planeBytes));
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaMallocHost(&hSend[k], planeBytes));
            CUDA_CHECK(cudaMallocHost(&hRecv[k], planeBytes));
        }
        // Edge (halo-producing) work gets high priority so it is not starved by the
        // inner kernel, keeping the halo exchange off the critical path.
        int prioLeast = 0, prioGreatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sInner, cudaStreamNonBlocking, prioLeast));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sEdge, cudaStreamNonBlocking, prioGreatest));
    }

    // Initialize (both buffers, including halos: boundaries & halos are then valid)
    if (root) printf("Initializing grid...\n");
    if (d1) {
        const size_t total = nAlloc * planeSize;
        const int threads = 256;
        const int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535 * 8);
        initializeKernel<<<blocks, threads, 0, sInner>>>(d1, d2, planeSize, nAlloc, gAlloc0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sInner));
    }

    // Global interior z range [1, nz-1) intersected with owned planes, in local indices
    int zLo = 0, zHi = 0;
    if (active && nz >= 3) {
        const size_t gLo = std::max<size_t>(gz0, 1);
        const size_t gHi = std::min<size_t>(gz0 + nLocal, nz - 1);
        if (gHi > gLo) {
            zLo = (int)(gLo - gAlloc0);
            zHi = (int)(gHi - gAlloc0);
        }
    }
    // Edge planes adjacent to neighbors are computed first (on sEdge) so the halo
    // exchange overlaps with the inner computation.
    const int ownLo = (int)off;                  // first owned local plane
    const int ownHi = (int)(off + nLocal) - 1;   // last owned local plane
    const bool doExchange = (hasLower || hasUpper) && d1;
    int innerLo = zLo, innerHi = zHi;
    if (hasLower && innerLo <= ownLo && innerHi > ownLo) innerLo = ownLo + 1;
    if (hasUpper && innerHi > ownHi && innerLo <= ownHi) innerHi = ownHi;
    const bool lowerEdgeComputed = hasLower && zLo <= ownLo && zHi > ownLo;
    const bool upperEdgeComputed = hasUpper && zLo <= ownHi && zHi > ownHi &&
                                   !(lowerEdgeComputed && ownHi == ownLo);

    const int ix = (int)nx, iy = (int)ny;
    const int lowerRank = rank - 1, upperRank = rank + 1;
    const int nxy = (int)planeSize;

    MPI_Barrier(MPI_COMM_WORLD);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (d1) {
        Real* in = d1;
        Real* out = d2;
        for (int iter = 0; iter < iterations; ++iter) {
            if (doExchange) {
                Real* recvLo = hRecv[0];
                Real* recvHi = hRecv[1];
                MPI_Request reqs[4];
                int nreq = 0;
                if (hasLower) MPI_Irecv(recvLo, nxy, MPI_DOUBLE, lowerRank, 1, comm, &reqs[nreq++]);
                if (hasUpper) MPI_Irecv(recvHi, nxy, MPI_DOUBLE, upperRank, 0, comm, &reqs[nreq++]);

                // Edge planes first (high-priority stream)
                if (lowerEdgeComputed) launchStencil(in, out, ix, iy, ownLo, ownLo + 1, sEdge);
                if (upperEdgeComputed) launchStencil(in, out, ix, iy, ownHi, ownHi + 1, sEdge);
                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(hSend[0], out + (size_t)ownLo * planeSize, planeBytes,
                                               cudaMemcpyDeviceToHost, sEdge));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(hSend[1], out + (size_t)ownHi * planeSize, planeBytes,
                                               cudaMemcpyDeviceToHost, sEdge));
                // Inner planes overlap with the exchange
                launchStencil(in, out, ix, iy, innerLo, innerHi, sInner);

                CUDA_CHECK(cudaStreamSynchronize(sEdge));
                if (hasLower) MPI_Isend(hSend[0], nxy, MPI_DOUBLE, lowerRank, 0, comm, &reqs[nreq++]);
                if (hasUpper) MPI_Isend(hSend[1], nxy, MPI_DOUBLE, upperRank, 1, comm, &reqs[nreq++]);
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

                if (hasLower)
                    CUDA_CHECK(cudaMemcpyAsync(out, recvLo, planeBytes, cudaMemcpyHostToDevice, sEdge));
                if (hasUpper)
                    CUDA_CHECK(cudaMemcpyAsync(out + (size_t)(ownHi + 1) * planeSize, recvHi, planeBytes,
                                               cudaMemcpyHostToDevice, sEdge));
                CUDA_CHECK(cudaStreamSynchronize(sEdge));
                CUDA_CHECK(cudaStreamSynchronize(sInner));
            } else {
                launchStencil(in, out, ix, iy, zLo, zHi, sInner);
            }
            std::swap(in, out);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sInner));
        CUDA_CHECK(cudaStreamSynchronize(sEdge));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(elapsedMs);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0 only if needed
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        if (root) finalGrid.resize(gridSize);
        Real* finalDev = (iterations % 2 == 0) ? d1 : d2;
        if (active && planeSize > 0) {
            MPI_Datatype planeType;
            MPI_Type_contiguous(nxy, MPI_DOUBLE, &planeType);
            MPI_Type_commit(&planeType);
            int nActive = 0;
            MPI_Comm_size(comm, &nActive);
            std::vector<Real> local;
            Real* sendBuf;
            if (root) {
                sendBuf = finalGrid.data() + gz0 * planeSize;
            } else {
                local.resize(nLocal * planeSize);
                sendBuf = local.data();
            }
            CUDA_CHECK(cudaMemcpy(sendBuf, finalDev + off * planeSize, nLocal * planeBytes,
                                  cudaMemcpyDeviceToHost));
            std::vector<int> counts, displs;
            if (root) {
                counts.resize(nActive);
                displs.resize(nActive);
                const size_t base = nz / activeRanks, rem = nz % activeRanks;
                for (int r = 0; r < nActive; ++r) {
                    counts[r] = (int)(base + ((size_t)r < rem ? 1 : 0));
                    displs[r] = (int)(r * base + std::min<size_t>(r, rem));
                }
            }
            MPI_Gatherv(root ? MPI_IN_PLACE : sendBuf, (int)nLocal, planeType,
                        root ? finalGrid.data() : nullptr, counts.data(), displs.data(), planeType, 0, comm);
            MPI_Type_free(&planeType);
        }

        if (root) {
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
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d1) {
        CUDA_CHECK(cudaFree(d1));
        CUDA_CHECK(cudaFree(d2));
        for (int k = 0; k < 2; ++k) {
            CUDA_CHECK(cudaFreeHost(hSend[k]));
            CUDA_CHECK(cudaFreeHost(hRecv[k]));
        }
        CUDA_CHECK(cudaStreamDestroy(sInner));
        CUDA_CHECK(cudaStreamDestroy(sEdge));
    }
    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
