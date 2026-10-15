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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t e_ = (call);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Domain is decomposed along Z across MPI ranks. Each rank owns nzl planes and keeps
// them on its GPU as local planes 1..nzl, with halo planes 0 and nzl+1.
// Global boundary cells are never modified, so both buffers hold them from initialization.

// Fills all local planes (including halos) of both buffers with the initial pattern.
__global__ void initKernel(Real* a, Real* b, const size_t nx, const size_t ny, const size_t nzLocal,
                           const long long zStart) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z;
    if (x >= nx || y >= ny || lz >= nzLocal + 2) return;
    const long long gz = zStart + (long long)lz - 1;
    Real v = 0.0;
    if (gz >= 0) v = (Real)((((size_t)gz * nx * ny) + y * nx + x) % 19);
    const size_t i = lz * nx * ny + y * nx + x;
    a[i] = v;
    b[i] = v;
}

// Updates interior x/y points for local planes [zlo, zhi]; each thread marches along z.
constexpr int ZCHUNK = 32;
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const int nx, const int ny, const int zlo, const int zhi) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;
    const int z0 = zlo + blockIdx.z * ZCHUNK;
    const int z1 = min(z0 + ZCHUNK - 1, zhi);
    if (z0 > z1) return;
    const size_t plane = (size_t)nx * ny;
    size_t i = (size_t)z0 * plane + (size_t)y * nx + x;
    Real bottom = in[i - plane];
    Real center = in[i];
    for (int z = z0; z <= z1; ++z, i += plane) {
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

static void launchStencil(const Real* in, Real* out, int nx, int ny, int zlo, int zhi, cudaStream_t s) {
    if (zlo > zhi || nx < 3 || ny < 3) return;
    dim3 block(32, 8);
    dim3 grid((nx + 31) / 32, (ny + 7) / 8, (zhi - zlo + ZCHUNK) / ZCHUNK);
    stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, zlo, zhi);
    CUDA_CHECK(cudaGetLastError());
}

bool validateLocal(const Real* grid, const size_t n, Real& minVal, Real& maxVal) {
    bool bad = false;
    Real mn = grid[0], mx = grid[0];
#pragma omp parallel for reduction(min : mn) reduction(max : mx) reduction(|| : bad)
    for (long long i = 0; i < (long long)n; ++i) {
        const Real v = grid[i];
        if (std::isnan(v) || std::isinf(v)) bad = true;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    minVal = mn;
    maxVal = mx;
    return !bad;
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
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

    // Select a GPU per rank (round-robin over ranks sharing a node)
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &node);
        int lrank;
        MPI_Comm_rank(node, &lrank);
        MPI_Comm_free(&node);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Only ranks that own at least one plane participate
    const int active = (int)std::min<size_t>((size_t)wsize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = wrank;
    const size_t base = nz / active, rem = nz % active;
    auto planesOf = [&](int r) { return base + ((size_t)r < rem ? 1 : 0); };
    auto startOf = [&](int r) { return (size_t)r * base + std::min((size_t)r, rem); };
    const size_t nzl = planesOf(rank);
    const size_t zStart = startOf(rank);
    const int up = rank + 1 < active ? rank + 1 : MPI_PROC_NULL;
    const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t localSize = plane * (nzl + 2);
    Real *d1, *d2;
    CUDA_CHECK(cudaMalloc(&d1, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, localSize * sizeof(Real)));
    Real *hSendUp, *hSendDown, *hRecvUp, *hRecvDown;
    CUDA_CHECK(cudaMallocHost(&hSendUp, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hSendDown, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecvUp, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecvDown, plane * sizeof(Real)));
    cudaStream_t sEdge, sInner;
    CUDA_CHECK(cudaStreamCreate(&sEdge));
    CUDA_CHECK(cudaStreamCreate(&sInner));

    if (wrank == 0) printf("Initializing grid...\n");
    {
        dim3 block(32, 8);
        dim3 grid((nx + 31) / 32, (ny + 7) / 8, nzl + 2);
        initKernel<<<grid, block>>>(d1, d2, nx, ny, nzl, (long long)zStart);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (wrank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    // Local planes of global interior: global z in [1, nz-2]
    const long long gLo = 1, gHi = (long long)nz - 2;
    auto clampLo = [&](long long l) { return std::max<long long>(l, gLo - (long long)zStart + 1); };
    auto clampHi = [&](long long l) { return std::min<long long>(l, gHi - (long long)zStart + 1); };
    const int inx = (int)nx, iny = (int)ny;
    const long long L = (long long)nzl;

    Real* in = d1;
    Real* out = d2;
    for (int iter = 0; iter < iterations; ++iter) {
        // Edge planes first (needed by neighbors), then interior overlaps the halo exchange
        launchStencil(in, out, inx, iny, (int)clampLo(1), (int)clampHi(1), sEdge);
        if (L > 1) launchStencil(in, out, inx, iny, (int)clampLo(L), (int)clampHi(L), sEdge);
        if (active > 1) {
            CUDA_CHECK(cudaMemcpyAsync(hSendDown, out + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
            CUDA_CHECK(cudaMemcpyAsync(hSendUp, out + plane * nzl, plane * sizeof(Real), cudaMemcpyDeviceToHost, sEdge));
        }
        if (L > 2) launchStencil(in, out, inx, iny, (int)clampLo(2), (int)clampHi(L - 1), sInner);
        if (active > 1) {
            CUDA_CHECK(cudaStreamSynchronize(sEdge));
            MPI_Request req[4];
            MPI_Irecv(hRecvDown, (int)plane, MPI_DOUBLE, down, 0, comm, &req[0]);
            MPI_Irecv(hRecvUp, (int)plane, MPI_DOUBLE, up, 1, comm, &req[1]);
            MPI_Isend(hSendUp, (int)plane, MPI_DOUBLE, up, 0, comm, &req[2]);
            MPI_Isend(hSendDown, (int)plane, MPI_DOUBLE, down, 1, comm, &req[3]);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            if (down != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(out, hRecvDown, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
            if (up != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(out + plane * (nzl + 1), hRecvUp, plane * sizeof(Real), cudaMemcpyHostToDevice, sEdge));
        }
        CUDA_CHECK(cudaStreamSynchronize(sEdge));
        CUDA_CHECK(cudaStreamSynchronize(sInner));
        std::swap(in, out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (wrank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // After the swap, `in` holds the final grid
    std::vector<Real> local;
    if (printResults || validate) {
        local.resize(plane * nzl);
        CUDA_CHECK(cudaMemcpy(local.data(), in + plane, plane * nzl * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        std::vector<Real> finalGrid;
        std::vector<int> counts, displs;
        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
            counts.resize(active);
            displs.resize(active);
            for (int r = 0; r < active; ++r) {
                counts[r] = (int)planesOf(r);
                displs[r] = (int)startOf(r);
            }
        }
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(local.data(), (int)nzl, planeType, finalGrid.data(), counts.data(), displs.data(), planeType, 0, comm);
        MPI_Type_free(&planeType);
        if (rank == 0) print_results(finalGrid, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        if (wrank == 0) printf("Validating result...\n");
        Real mn, mx;
        int ok = validateLocal(local.data(), local.size(), mn, mx) ? 1 : 0;
        int allOk;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, comm);
        Real gmn, gmx;
        MPI_Allreduce(&mn, &gmn, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&mx, &gmx, 1, MPI_DOUBLE, MPI_MAX, comm);
        bool valid = true;
        if (!allOk) {
            if (wrank == 0) printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            if (wrank == 0) printf("Value range: [%.6f, %.6f]\n", gmn, gmx);
            if (gmx > 1e6 || gmn < -1e6) {
                if (wrank == 0) printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }
        if (wrank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        exitCode = valid ? 0 : 1;
    }

    cudaFree(d1);
    cudaFree(d2);
    cudaFreeHost(hSendUp);
    cudaFreeHost(hSendDown);
    cudaFreeHost(hRecvUp);
    cudaFreeHost(hRecvDown);
    cudaStreamDestroy(sEdge);
    cudaStreamDestroy(sInner);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return exitCode;
}
