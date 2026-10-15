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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),    \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Domain decomposition: 1D along Z. Each rank owns nzl consecutive z-planes and
// stores them in local planes 1..nzl; local planes 0 and nzl+1 are halos.
// Local plane lz corresponds to global z = z0 - 1 + lz.

constexpr int BX = 32, BY = 8, ZB = 16;

// Initialize all local planes (including halos) on the device
__global__ void initKernel(Real* g, const size_t nx, const size_t ny, const long z0,
                           const long nz, const int nplanes) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const int lz = blockIdx.z;
    if (x >= nx || y >= ny || lz >= nplanes) return;
    const long gz = z0 - 1 + lz;
    const size_t li = (size_t)lz * nx * ny + y * nx + x;
    if (gz < 0 || gz >= nz) {
        g[li] = 0.0;
        return;
    }
    const size_t idx = (size_t)gz * nx * ny + y * nx + x;
    g[li] = (idx % 19) * 1.0;
}

// 7-point stencil on local planes [zbeg, zend). Each thread walks a column of
// up to ZB planes, reusing z-neighbours from registers.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const int nx, const int ny, const int zbeg, const int zend) {
    const int x = 1 + blockIdx.x * BX + threadIdx.x;
    const int y = 1 + blockIdx.y * BY + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;
    const int z0 = zbeg + blockIdx.z * ZB;
    const int z1 = min(z0 + ZB, zend);
    const size_t plane = (size_t)nx * ny;
    size_t i = (size_t)z0 * plane + (size_t)y * nx + x;
    Real bottom = in[i - plane];
    Real center = in[i];
    for (int z = z0; z < z1; ++z, i += plane) {
        const Real top = in[i + plane];
        const Real left = in[i - 1];
        const Real right = in[i + 1];
        const Real front = in[i - nx];
        const Real back = in[i + nx];
        out[i] = (center + left + right + front + back + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }
}

static void launchStencil(const Real* in, Real* out, int nx, int ny, int zbeg, int zend,
                          cudaStream_t s) {
    if (zend <= zbeg || nx < 3 || ny < 3) return;
    dim3 block(BX, BY);
    dim3 grid((nx - 2 + BX - 1) / BX, (ny - 2 + BY - 1) / BY, (zend - zbeg + ZB - 1) / ZB);
    stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, zbeg, zend);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    const size_t n = grid.size();
    int bad = 0;
    #pragma omp parallel for reduction(|:bad)
    for (size_t i = 0; i < n; ++i) {
        bad |= (std::isnan(grid[i]) || std::isinf(grid[i])) ? 1 : 0;
    }
    if (bad) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int wrank = 0, wsize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

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
            if (wrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (wrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (wrank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only ranks that own at least one plane participate
    const int nactive = (int)std::min<size_t>((size_t)wsize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < nactive ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Bind one GPU per rank (by node-local rank)
    {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(node, &lrank);
        MPI_Comm_free(&node);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Slab decomposition
    const size_t base = nz / size, rem = nz % size;
    const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min<size_t>(rank, rem);
    const int lo = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int hi = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localElems = plane * (nzl + 2);

    Real *d1 = nullptr, *d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d1, localElems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, localElems * sizeof(Real)));

    Real *hSend[2] = {nullptr, nullptr}, *hRecv[2] = {nullptr, nullptr};
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaMallocHost(&hSend[k], plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&hRecv[k], plane * sizeof(Real)));
    }

    cudaStream_t sB, sI;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));

    // Initialize (both buffers so fixed boundaries are present in each)
    if (rank == 0) printf("Initializing grid...\n");
    {
        dim3 block(BX, BY);
        dim3 grid((nx + BX - 1) / BX, (ny + BY - 1) / BY, (unsigned)(nzl + 2));
        initKernel<<<grid, block>>>(d1, nx, ny, (long)z0, (long)nz, (int)(nzl + 2));
        initKernel<<<grid, block>>>(d2, nx, ny, (long)z0, (long)nz, (int)(nzl + 2));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Local plane ranges (local indices) containing global-interior points
    const long lzMin = std::max<long>(1, 2 - (long)z0);
    const long lzMax = std::min<long>((long)nzl, (long)nz - 1 - (long)z0);  // inclusive
    const int ib = (int)lzMin, ie = (int)lzMax + 1;                         // [ib, ie)
    // Planes adjacent to neighbours are computed first (to overlap communication)
    const bool needLo = lo != MPI_PROC_NULL, needHi = hi != MPI_PROC_NULL;

    if (rank == 0) printf("Running stencil computation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    const int inx = (int)nx, iny = (int)ny;
    const int nzli = (int)nzl;
    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = (iter % 2 == 0) ? d1 : d2;
        Real* out = (iter % 2 == 0) ? d2 : d1;

        // 1. Planes adjacent to neighbours first (stream B)
        if (needLo) launchStencil(in, out, inx, iny, std::max(1, ib), std::min(2, ie), sB);
        if (needHi && !(needLo && nzli == 1))
            launchStencil(in, out, inx, iny, std::max(nzli, ib), std::min(nzli + 1, ie), sB);
        if (needLo) CUDA_CHECK(cudaMemcpyAsync(hSend[0], out + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sB));
        if (needHi) CUDA_CHECK(cudaMemcpyAsync(hSend[1], out + plane * nzl, plane * sizeof(Real), cudaMemcpyDeviceToHost, sB));
        // 2. Remaining planes (stream I), overlapped with the exchange
        launchStencil(in, out, inx, iny, needLo ? std::max(ib, 2) : ib, needHi ? std::min(ie, nzli) : ie, sI);
        CUDA_CHECK(cudaGetLastError());

        // 3. Halo exchange through pinned host buffers
        if (needLo || needHi) {
            CUDA_CHECK(cudaStreamSynchronize(sB));
            MPI_Request req[4];
            int nr = 0;
            if (needLo) {
                MPI_Irecv(hRecv[0], (int)plane, MPI_DOUBLE, lo, 0, comm, &req[nr++]);
                MPI_Isend(hSend[0], (int)plane, MPI_DOUBLE, lo, 1, comm, &req[nr++]);
            }
            if (needHi) {
                MPI_Irecv(hRecv[1], (int)plane, MPI_DOUBLE, hi, 1, comm, &req[nr++]);
                MPI_Isend(hSend[1], (int)plane, MPI_DOUBLE, hi, 0, comm, &req[nr++]);
            }
            MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
            if (needLo) CUDA_CHECK(cudaMemcpyAsync(out, hRecv[0], plane * sizeof(Real), cudaMemcpyHostToDevice, sB));
            if (needHi) CUDA_CHECK(cudaMemcpyAsync(out + plane * (nzl + 1), hRecv[1], plane * sizeof(Real), cudaMemcpyHostToDevice, sB));
        }
        CUDA_CHECK(cudaStreamSynchronize(sB));
        CUDA_CHECK(cudaStreamSynchronize(sI));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_ms);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int ret = 0;
    if (printResults || validate) {
        // Gather final grid on rank 0
        const Real* dfinal = (iterations % 2 == 0) ? d1 : d2;
        std::vector<Real> mine(plane * nzl);
        CUDA_CHECK(cudaMemcpy(mine.data(), dfinal + plane, mine.size() * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<Real> finalGrid;
        // Chunk the gather so counts fit in int
        const size_t maxChunk = (size_t)1 << 26;
        std::vector<size_t> cnt(size), dsp(size);
        for (int r = 0; r < size; ++r) {
            size_t n = base + ((size_t)r < rem ? 1 : 0);
            cnt[r] = n * plane;
            dsp[r] = ((size_t)r * base + std::min<size_t>(r, rem)) * plane;
        }
        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
            std::memcpy(finalGrid.data(), mine.data(), mine.size() * sizeof(Real));
            for (int r = 1; r < size; ++r) {
                for (size_t off = 0; off < cnt[r]; off += maxChunk) {
                    const int n = (int)std::min(maxChunk, cnt[r] - off);
                    MPI_Recv(finalGrid.data() + dsp[r] + off, n, MPI_DOUBLE, r, 7, comm, MPI_STATUS_IGNORE);
                }
            }
        } else {
            for (size_t off = 0; off < mine.size(); off += maxChunk) {
                const int n = (int)std::min(maxChunk, mine.size() - off);
                MPI_Send(mine.data() + off, n, MPI_DOUBLE, 0, 7, comm);
            }
        }
        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(finalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    ret = 1;
                }
            }
        }
    }

    for (int k = 0; k < 2; ++k) {
        cudaFreeHost(hSend[k]);
        cudaFreeHost(hRecv[k]);
    }
    cudaStreamDestroy(sB);
    cudaStreamDestroy(sI);
    cudaFree(d1);
    cudaFree(d2);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
