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

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
                    __FILE__, __LINE__);                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

constexpr int BX = 32;
constexpr int BY = 8;
constexpr int ZCH = 16;  // planes marched per thread block

// Initial value of the global grid at linear index idx
__global__ void initKernel(Real* __restrict__ grid, const size_t nx, const size_t ny,
                           const long long zGlobal0, const long long nz, const int planes) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    const int k = blockIdx.z;
    if (x >= nx || y >= ny || k >= planes) return;
    const long long gz = zGlobal0 + k;
    if (gz < 0 || gz >= nz) return;
    const size_t gidx = (size_t)gz * (nx * ny) + y * nx + x;
    grid[(size_t)k * (nx * ny) + y * nx + x] = (Real)(gidx % 19);
}

// 7-point stencil over local planes [lbegin, lbegin+lcount). Local plane k corresponds to
// global plane zGlobal0 + k. Boundary points of the global grid are copied.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const size_t nx, const size_t ny, const long long nz,
                              const long long zGlobal0, const int lbegin, const int lcount) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int zl0 = lbegin + blockIdx.z * ZCH;
    const int zl1 = min(zl0 + ZCH, lbegin + lcount);
    if (zl0 >= zl1) return;

    const size_t P = nx * ny;
    size_t idx = (size_t)zl0 * P + y * nx + x;
    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);

    if (xyBoundary) {
        for (int z = zl0; z < zl1; ++z, idx += P) out[idx] = in[idx];
        return;
    }

    Real bottom = in[idx - P];
    Real center = in[idx];
    for (int z = zl0; z < zl1; ++z, idx += P) {
        const Real top = in[idx + P];
        const long long gz = zGlobal0 + z;
        if (gz == 0 || gz == nz - 1) {
            out[idx] = center;
        } else {
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny, long long nz,
                          long long zGlobal0, int lbegin, int lcount, cudaStream_t s) {
    if (lcount <= 0) return;
    dim3 block(BX, BY);
    dim3 grid((unsigned)((nx + BX - 1) / BX), (unsigned)((ny + BY - 1) / BY),
              (unsigned)((lcount + ZCH - 1) / ZCH));
    stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, nz, zGlobal0, lbegin, lcount);
    CUDA_CHECK(cudaGetLastError());
}

// Large-message helpers (chunked to stay below int limits)
static void sendLarge(const Real* buf, size_t count, int dest, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 26;
    for (size_t o = 0; o < count; o += chunk)
        MPI_Send(buf + o, (int)std::min(chunk, count - o), MPI_DOUBLE, dest, 77, comm);
}
static void recvLarge(Real* buf, size_t count, int src, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 26;
    for (size_t o = 0; o < count; o += chunk)
        MPI_Recv(buf + o, (int)std::min(chunk, count - o), MPI_DOUBLE, src, 77, comm,
                 MPI_STATUS_IGNORE);
}

// Distributed validation of the local slab; returns true if valid (same result on all ranks)
bool validateResult(const Real* data, const size_t n, const bool haveData, const int rank) {
    int bad = 0;
    Real minVal = INFINITY, maxVal = -INFINITY;
    if (haveData) {
        #pragma omp parallel for reduction(|:bad) reduction(min:minVal) reduction(max:maxVal)
        for (size_t i = 0; i < n; ++i) {
            const Real v = data[i];
            if (std::isnan(v) || std::isinf(v)) bad = 1;
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }
    }
    int badAll = 0;
    Real gmin, gmax;
    MPI_Allreduce(&bad, &badAll, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (badAll) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 1e6 || gmin < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
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

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank (selected by node-local rank)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lr;
        MPI_Comm_rank(local, &lr);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lr % ndev));
    }

    // 1D decomposition along z; ranks beyond nz planes stay idle
    const int active = (int)std::max<size_t>(1, std::min<size_t>((size_t)size, nz));
    int nl = 0;
    long long z0 = 0;
    {
        const size_t base = nz / active, rem = nz % active;
        if (rank < active) {
            nl = (int)(base + ((size_t)rank < rem ? 1 : 0));
            z0 = (long long)((size_t)rank * base + std::min<size_t>(rank, rem));
        }
    }
    const bool isActive = rank < active && nl > 0;
    const int down = (isActive && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (isActive && rank < active - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t P = nx * ny;
    const size_t localElems = isActive ? (size_t)(nl + 2) * P : 0;
    Real *d_a = nullptr, *d_b = nullptr;
    Real *h_sendDown = nullptr, *h_sendUp = nullptr, *h_recvDown = nullptr, *h_recvUp = nullptr;
    cudaStream_t sEdge = nullptr, sInt = nullptr;

    if (isActive) {
        CUDA_CHECK(cudaMalloc(&d_a, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_a, 0, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_b, 0, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendDown, P * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendUp, P * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvDown, P * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvUp, P * sizeof(Real)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sEdge, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sInt, cudaStreamNonBlocking));

        // Initialize (including halo planes, so the first iteration needs no exchange)
        if (rank == 0) printf("Initializing grid...\n");
        dim3 block(BX, BY);
        dim3 grid((unsigned)((nx + BX - 1) / BX), (unsigned)((ny + BY - 1) / BY), (unsigned)(nl + 2));
        initKernel<<<grid, block>>>(d_a, nx, ny, z0 - 1, (long long)nz, nl + 2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else if (rank == 0) {
        printf("Initializing grid...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    const double tStart = MPI_Wtime();

    Real* cur = d_a;
    Real* nxt = d_b;
    if (isActive) {
        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Request reqs[4];
            int nreq = 0;
            // Post receives for halos of the new grid
            MPI_Irecv(h_recvDown, (int)P, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(h_recvUp, (int)P, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &reqs[nreq++]);

            // Boundary planes first so communication overlaps with interior compute
            launchStencil(cur, nxt, nx, ny, (long long)nz, z0 - 1, 1, 1, sEdge);
            if (nl > 1) launchStencil(cur, nxt, nx, ny, (long long)nz, z0 - 1, nl, 1, sEdge);
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(h_sendDown, nxt + P, P * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sEdge));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(h_sendUp, nxt + (size_t)nl * P, P * sizeof(Real),
                                           cudaMemcpyDeviceToHost, sEdge));
            if (nl > 2) launchStencil(cur, nxt, nx, ny, (long long)nz, z0 - 1, 2, nl - 2, sInt);

            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            MPI_Isend(h_sendDown, (int)P, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(h_sendUp, (int)P, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(nxt, h_recvDown, P * sizeof(Real),
                                           cudaMemcpyHostToDevice, sEdge));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(nxt + (size_t)(nl + 1) * P, h_recvUp, P * sizeof(Real),
                                           cudaMemcpyHostToDevice, sEdge));
            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            CUDA_CHECK(cudaStreamSynchronize(sInt));
            std::swap(cur, nxt);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long long ms = (long long)((tEnd - tStart) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Final grid lives in `cur`
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> local;
        if (isActive) {
            local.resize((size_t)nl * P);
            CUDA_CHECK(cudaMemcpy(local.data(), cur + P, (size_t)nl * P * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }

        if (printResults) {
            if (rank == 0) {
                std::vector<Real> full(nz * P);
                std::copy(local.begin(), local.end(), full.begin());
                size_t off = local.size();
                for (int r = 1; r < active; ++r) {
                    const size_t base = nz / active, rem = nz % active;
                    const size_t cnt = (base + ((size_t)r < rem ? 1 : 0)) * P;
                    recvLarge(full.data() + off, cnt, r, MPI_COMM_WORLD);
                    off += cnt;
                }
                print_results(full, "Grid");
            } else if (isActive) {
                sendLarge(local.data(), local.size(), 0, MPI_COMM_WORLD);
            }
        }

        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateResult(local.data(), local.size(), isActive, rank);
            if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            if (!valid) exitCode = 1;
        }
    }

    if (isActive) {
        cudaStreamDestroy(sEdge);
        cudaStreamDestroy(sInt);
        cudaFreeHost(h_sendDown);
        cudaFreeHost(h_sendUp);
        cudaFreeHost(h_recvDown);
        cudaFreeHost(h_recvUp);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    MPI_Finalize();
    return exitCode;
}
