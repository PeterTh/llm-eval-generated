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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Decomposition: the grid is split into contiguous slabs along z (one per MPI rank).
// Each rank holds its lz planes plus one halo plane on each side (local plane k
// corresponds to global plane z0 + k - 1).

// Initialize local planes (including halos) of both buffers with the global pattern.
__global__ void initKernel(Real* a, Real* b, size_t nx, size_t ny, size_t nz,
                           long long z0, size_t planes) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const size_t k = blockIdx.z;
    if (x >= nx || y >= ny || k >= planes) return;
    const long long gz = z0 + (long long)k - 1;
    if (gz < 0 || gz >= (long long)nz) return;
    const size_t gidx = (size_t)gz * (nx * ny) + y * nx + x;
    const Real v = (Real)(gidx % 19) * 1.0;
    const size_t lidx = k * (nx * ny) + y * nx + x;
    a[lidx] = v;
    b[lidx] = v;
}

// 7-point stencil over local planes [kbeg, kbeg + count). Global boundary cells are copied.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              size_t nx, size_t ny, size_t nz,
                              long long z0, size_t kbeg) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t k = kbeg + blockIdx.z;
    const long long gz = z0 + (long long)k - 1;
    const size_t plane = nx * ny;
    const size_t idx = k * plane + y * nx + x;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == (long long)nz - 1) {
        out[idx] = in[idx];
        return;
    }
    const Real center = __ldg(in + idx);
    const Real left = __ldg(in + idx - 1);
    const Real right = __ldg(in + idx + 1);
    const Real front = __ldg(in + idx - nx);
    const Real back = __ldg(in + idx + nx);
    const Real bottom = __ldg(in + idx - plane);
    const Real top = __ldg(in + idx + plane);
    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void launchStencil(const Real* in, Real* out, size_t nx, size_t ny, size_t nz,
                          long long z0, size_t kbeg, size_t count, cudaStream_t s) {
    if (count == 0) return;
    dim3 block(32, 4);
    dim3 grid((unsigned)((nx + block.x - 1) / block.x), (unsigned)((ny + block.y - 1) / block.y),
              (unsigned)count);
    stencilKernel<<<grid, block, 0, s>>>(in, out, nx, ny, nz, z0, kbeg);
    CUDA_CHECK(cudaGetLastError());
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
    int worldRank = 0, worldSize = 1;
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

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank, assigned round-robin among ranks on the same node
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

    // Ranks beyond the number of z-planes have no work
    const int active = (int)std::min<size_t>((size_t)worldSize, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    int exitCode = 0;
    if (comm != MPI_COMM_NULL) {
        const int rank = worldRank;
        const int P = active;
        const size_t base = nz / P, rem = nz % P;
        auto planesOf = [&](int r) { return base + ((size_t)r < rem ? 1 : 0); };
        auto startOf = [&](int r) { return (size_t)r * base + std::min((size_t)r, rem); };
        const size_t lz = planesOf(rank);
        const long long z0 = (long long)startOf(rank);
        const size_t plane = nx * ny;
        const size_t localSize = plane * (lz + 2);
        const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
        const int up = rank < P - 1 ? rank + 1 : MPI_PROC_NULL;

        Real *d_a = nullptr, *d_b = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, localSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_b, localSize * sizeof(Real)));
        Real* h_halo = nullptr;  // [send_lo | send_hi | recv_lo | recv_hi]
        CUDA_CHECK(cudaMallocHost(&h_halo, 4 * plane * sizeof(Real)));
        Real* h_send_lo = h_halo;
        Real* h_send_hi = h_halo + plane;
        Real* h_recv_lo = h_halo + 2 * plane;
        Real* h_recv_hi = h_halo + 3 * plane;

        cudaStream_t sB, sI;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));

        if (rank == 0) printf("Initializing grid...\n");
        {
            dim3 block(32, 4);
            dim3 grid((unsigned)((nx + 31) / 32), (unsigned)((ny + 3) / 4), (unsigned)(lz + 2));
            initKernel<<<grid, block>>>(d_a, d_b, nx, ny, nz, z0, lz + 2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        Real* cur = d_a;
        Real* nxt = d_b;
        const size_t bytes = plane * sizeof(Real);
        for (int iter = 0; iter < iterations; ++iter) {
            // Boundary planes first so the halo exchange overlaps with the interior update
            launchStencil(cur, nxt, nx, ny, nz, z0, 1, 1, sB);
            if (lz > 1) launchStencil(cur, nxt, nx, ny, nz, z0, lz, 1, sB);
            if (lz > 2) launchStencil(cur, nxt, nx, ny, nz, z0, 2, lz - 2, sI);

            if (iter + 1 < iterations && P > 1) {
                if (down != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_send_lo, nxt + plane, bytes, cudaMemcpyDeviceToHost, sB));
                if (up != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_send_hi, nxt + lz * plane, bytes, cudaMemcpyDeviceToHost, sB));
                CUDA_CHECK(cudaStreamSynchronize(sB));

                MPI_Request req[4];
                MPI_Irecv(h_recv_lo, (int)plane, MPI_DOUBLE, down, 1, comm, &req[0]);
                MPI_Irecv(h_recv_hi, (int)plane, MPI_DOUBLE, up, 2, comm, &req[1]);
                MPI_Isend(h_send_lo, (int)plane, MPI_DOUBLE, down, 2, comm, &req[2]);
                MPI_Isend(h_send_hi, (int)plane, MPI_DOUBLE, up, 1, comm, &req[3]);
                MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

                if (down != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(nxt, h_recv_lo, bytes, cudaMemcpyHostToDevice, sB));
                if (up != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(nxt + (lz + 1) * plane, h_recv_hi, bytes, cudaMemcpyHostToDevice, sB));
            }
            CUDA_CHECK(cudaStreamSynchronize(sB));
            CUDA_CHECK(cudaStreamSynchronize(sI));
            std::swap(cur, nxt);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long local_ms = duration.count(), global_ms = 0;
        MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, comm);

        if (rank == 0) {
            printf("Computation time: %ld ms\n", global_ms);

            // Calculate performance metrics
            double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
            double mcups = cellUpdates / (global_ms / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // cur holds the final grid
        std::vector<Real> local;
        if (printResults || validate) {
            local.resize(plane * lz);
            CUDA_CHECK(cudaMemcpy(local.data(), cur + plane, plane * lz * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        if (printResults) {
            std::vector<Real> full;
            std::vector<int> counts, displs;
            if (rank == 0) {
                full.resize(nx * ny * nz);
                counts.resize(P);
                displs.resize(P);
                for (int r = 0; r < P; ++r) {
                    counts[r] = (int)planesOf(r);
                    displs[r] = (int)startOf(r);
                }
            }
            MPI_Datatype planeType;
            MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
            MPI_Type_commit(&planeType);
            MPI_Gatherv(local.data(), (int)lz, planeType, full.data(), counts.data(), displs.data(),
                        planeType, 0, comm);
            MPI_Type_free(&planeType);
            if (rank == 0) print_results(full, "Grid");
        }

        if (validate) {
            const size_t n = local.size();
            int bad = 0;
            Real minVal = local[0];
            Real maxVal = local[0];
            #pragma omp parallel for reduction(max:bad) reduction(min:minVal) reduction(max:maxVal)
            for (size_t i = 0; i < n; ++i) {
                const Real v = local[i];
                if (std::isnan(v) || std::isinf(v)) bad = 1;
                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }
            int gbad = 0;
            MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
            Real gmin = 0, gmax = 0;
            MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);

            bool valid = true;
            if (rank == 0) printf("Validating result...\n");
            if (gbad) {
                if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
                valid = false;
            } else {
                if (rank == 0) printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
                if (gmax > 1e6 || gmin < -1e6) {
                    if (rank == 0) printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }
            if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            exitCode = valid ? 0 : 1;
        }

        CUDA_CHECK(cudaFreeHost(h_halo));
        CUDA_CHECK(cudaStreamDestroy(sB));
        CUDA_CHECK(cudaStreamDestroy(sI));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        MPI_Comm_free(&comm);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
