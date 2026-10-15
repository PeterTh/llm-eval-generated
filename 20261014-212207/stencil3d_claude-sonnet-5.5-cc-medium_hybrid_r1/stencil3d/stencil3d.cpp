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

// Domain decomposition: 1D along z. Each rank owns planes [z0, z0+nl) of the global
// grid and stores them in local planes 1..nl; local planes 0 and nl+1 are halos.

// Initialization directly on the device (global linear index % 19)
__global__ void initKernel(Real* grid, const size_t nx, const size_t ny, const size_t nl, const size_t z0) {
    const size_t plane = nx * ny;
    const size_t total = plane * (nl + 2);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        // local plane p corresponds to global plane z0 + p - 1 (halos may be out of range: unused/overwritten)
        const long long gz = (long long)z0 + (long long)(i / plane) - 1;
        const size_t gidx = (size_t)(gz < 0 ? 0 : gz) * plane + (i % plane);
        grid[i] = (Real)(gidx % 19);
    }
}

// 7-point stencil on local planes [lz0, lz1] (inclusive). Global boundary points are never
// written; both buffers hold identical boundary data from initialization.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const int nx, const int ny, const int lz0, const int lz1,
                              const size_t z0, const size_t nz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = lz0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx - 1 || y >= ny - 1 || z > lz1) return;
    const size_t gz = z0 + z - 1;
    if (gz == 0 || gz == nz - 1) return;
    const size_t sx = 1, sy = nx, sz = (size_t)nx * ny;
    const size_t i = z * sz + y * sy + x;
    const Real c = __ldg(in + i);
    const Real l = __ldg(in + i - sx);
    const Real r = __ldg(in + i + sx);
    const Real f = __ldg(in + i - sy);
    const Real b = __ldg(in + i + sy);
    const Real d = __ldg(in + i - sz);
    const Real u = __ldg(in + i + sz);
    out[i] = (c + l + r + f + b + d + u) / 7.0;
}

static void launchStencil(const Real* in, Real* out, int nx, int ny, int lz0, int lz1,
                          size_t z0, size_t nz, cudaStream_t s) {
    if (lz1 < lz0 || nx < 3 || ny < 3) return;
    dim3 blk(32, 4, 2);
    dim3 grd((nx - 2 + blk.x - 1) / blk.x, (ny - 2 + blk.y - 1) / blk.y,
             (lz1 - lz0 + 1 + blk.z - 1) / blk.z);
    stencilKernel<<<grd, blk, 0, s>>>(in, out, nx, ny, lz0, lz1, z0, nz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateLocal(const Real* grid, const size_t n, Real& minVal, Real& maxVal) {
    int bad = 0;
    Real mn = grid[0], mx = grid[0];
#pragma omp parallel for reduction(|:bad) reduction(min:mn) reduction(max:mx)
    for (long long i = 0; i < (long long)n; ++i) {
        const Real v = grid[i];
        if (std::isnan(v) || std::isinf(v)) bad |= 1;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    minVal = mn;
    maxVal = mx;
    return bad == 0;
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
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU of its node
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // 1D slab decomposition along z; ranks beyond nz own nothing
    const int active = (int)std::min<size_t>((size_t)nranks, nz);
    size_t nl = 0, z0 = 0;
    if (rank < active) {
        const size_t base = nz / active, rem = nz % active;
        nl = base + ((size_t)rank < rem ? 1 : 0);
        z0 = (size_t)rank * base + std::min<size_t>(rank, rem);
    }
    const int lower = (rank > 0 && rank < active) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank + 1 < active) ? rank + 1 : MPI_PROC_NULL;

    Real *d_a = nullptr, *d_b = nullptr;
    Real *h_sendLo = nullptr, *h_sendHi = nullptr, *h_recvLo = nullptr, *h_recvHi = nullptr;
    cudaStream_t sB = nullptr, sI = nullptr;
    if (nl > 0) {
        const size_t bytes = plane * (nl + 2) * sizeof(Real);
        CUDA_CHECK(cudaMalloc(&d_a, bytes));
        CUDA_CHECK(cudaMalloc(&d_b, bytes));
        CUDA_CHECK(cudaMallocHost(&h_sendLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvLo, plane * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recvHi, plane * sizeof(Real)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));
    }

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    if (nl > 0) {
        initKernel<<<2048, 256>>>(d_a, nx, ny, nl, z0);
        initKernel<<<2048, 256>>>(d_b, nx, ny, nl, z0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (nl > 0) {
        const int inx = (int)nx, iny = (int)ny, inl = (int)nl;
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* in = (iter % 2 == 0) ? d_a : d_b;
            Real* out = (iter % 2 == 0) ? d_b : d_a;
            const bool lastIter = (iter == iterations - 1);

            // Border planes first (stream sB), interior concurrently (stream sI)
            launchStencil(in, out, inx, iny, 1, 1, z0, nz, sB);
            if (inl > 1) launchStencil(in, out, inx, iny, inl, inl, z0, nz, sB);
            if (inl > 2) launchStencil(in, out, inx, iny, 2, inl - 1, z0, nz, sI);

            if (!lastIter) {
                const size_t pb = plane * sizeof(Real);
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendLo, out + plane, pb, cudaMemcpyDeviceToHost, sB));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendHi, out + plane * nl, pb, cudaMemcpyDeviceToHost, sB));
                CUDA_CHECK(cudaStreamSynchronize(sB));

                MPI_Request req[4];
                int nreq = 0;
                if (lower != MPI_PROC_NULL) {
                    MPI_Irecv(h_recvLo, (int)plane, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req[nreq++]);
                    MPI_Isend(h_sendLo, (int)plane, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req[nreq++]);
                }
                if (upper != MPI_PROC_NULL) {
                    MPI_Irecv(h_recvHi, (int)plane, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req[nreq++]);
                    MPI_Isend(h_sendHi, (int)plane, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req[nreq++]);
                }
                MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(out, h_recvLo, pb, cudaMemcpyHostToDevice, sB));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(out + plane * (nl + 1), h_recvHi, pb, cudaMemcpyHostToDevice, sB));
            }
            CUDA_CHECK(cudaStreamSynchronize(sB));
            CUDA_CHECK(cudaStreamSynchronize(sI));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long elapsed_ms = duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration = std::chrono::milliseconds(elapsed_ms);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Bring owned planes of the final grid back to the host
    std::vector<Real> local;
    if (nl > 0 && (printResults || validate)) {
        const Real* fin = (iterations % 2 == 0) ? d_a : d_b;
        local.resize(plane * nl);
        CUDA_CHECK(cudaMemcpy(local.data(), fin + plane, plane * nl * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        std::vector<int> counts(nranks, 0), displs(nranks, 0);
        for (int r = 0; r < active; ++r) {
            const size_t base = nz / active, rem = nz % active;
            counts[r] = (int)(base + ((size_t)r < rem ? 1 : 0));
            displs[r] = (int)((size_t)r * base + std::min<size_t>(r, rem));
        }
        std::vector<Real> full;
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(local.data(), (int)nl, planeType, full.data(), counts.data(), displs.data(),
                    planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);
        if (rank == 0) print_results(full, "Grid");
    }
    
    int status = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int ok = 1;
        Real mn = INFINITY, mx = -INFINITY;
        if (nl > 0) ok = validateLocal(local.data(), local.size(), mn, mx) ? 1 : 0;
        int allOk = 1;
        Real gmn, gmx;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&mn, &gmn, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&mx, &gmx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        bool valid = allOk != 0;
        if (valid) {
            if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmn, gmx);
            if (gmx > 1e6 || gmn < -1e6) {
                if (rank == 0) printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        } else if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = valid ? 0 : 1;
    }

    if (nl > 0) {
        cudaStreamDestroy(sB);
        cudaStreamDestroy(sI);
        cudaFreeHost(h_sendLo); cudaFreeHost(h_sendHi);
        cudaFreeHost(h_recvLo); cudaFreeHost(h_recvHi);
        cudaFree(d_a);
        cudaFree(d_b);
    }
    MPI_Finalize();
    return status;
}
