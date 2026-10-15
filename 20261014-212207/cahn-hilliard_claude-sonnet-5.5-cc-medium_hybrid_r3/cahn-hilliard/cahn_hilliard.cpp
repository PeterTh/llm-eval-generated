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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Domain is decomposed along Z over MPI ranks (one GPU per rank). Each rank stores its
// owned planes plus HALO ghost planes per side. With HALO = 2 a single exchange of the
// concentration field per step suffices: chemical potential is computed redundantly on
// one ghost plane, so no second exchange for mu is needed.
constexpr int HALO = 2;

struct Params {
    size_t nx, ny, nz;   // global grid
    size_t zoff;         // global z of first owned plane
    double dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB;
};

// Local plane lz (may be -HALO..nzl+HALO-1) lives at padded plane lz+HALO.
__device__ __forceinline__ size_t pidx(const size_t x, const size_t y, const long lz, const size_t nx, const size_t ny) {
    return (size_t)(lz + HALO) * (nx * ny) + y * nx + x;
}

// Laplacian with clamped boundary conditions (global boundaries), same op order as the reference
__device__ __forceinline__ double laplacian(const double* __restrict__ c, const Params& p, const size_t x, const size_t y,
                                            const long lz, const size_t gz) {
    const size_t nx = p.nx, ny = p.ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const long zp = (gz < p.nz - 1) ? lz + 1 : lz;
    const long zn = (gz > 0) ? lz - 1 : lz;

    const double cc = c[pidx(x, y, lz, nx, ny)];
    const double cxx = (c[pidx(xp, y, lz, nx, ny)] + c[pidx(xn, y, lz, nx, ny)] - 2.0 * cc) / (p.dx * p.dx);
    const double cyy = (c[pidx(x, yp, lz, nx, ny)] + c[pidx(x, yn, lz, nx, ny)] - 2.0 * cc) / (p.dy * p.dy);
    const double czz = (c[pidx(x, y, zp, nx, ny)] + c[pidx(x, y, zn, nx, ny)] - 2.0 * cc) / (p.dz * p.dz);
    return cxx + cyy + czz;
}

// mu on local planes [z0, z1)
__global__ void chemPotKernel(const double* __restrict__ c, double* __restrict__ mu, const Params p, const long z0, const long z1) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const long lz = z0 + (long)blockIdx.z;
    if (x >= p.nx || y >= p.ny || lz >= z1) return;
    const double cv = c[pidx(x, y, lz, p.nx, p.ny)];
    mu[pidx(x, y, lz, p.nx, p.ny)] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                                     + 3.0 * cv + cv * cv * cv
                                     - p.gamma * laplacian(c, p, x, y, lz, p.zoff + lz);
}

// cnew on local planes [z0, z1)
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                             const Params p, const long z0, const long z1) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const long lz = z0 + (long)blockIdx.z;
    if (x >= p.nx || y >= p.ny || lz >= z1) return;
    const size_t i = pidx(x, y, lz, p.nx, p.ny);
    cnew[i] = cold[i] + p.dt * p.D * laplacian(mu, p, x, y, lz, p.zoff + lz);
}

// Initialize concentration on local planes [z0, z1) (including ghost planes that exist globally)
__global__ void initKernel(double* __restrict__ c, const Params p, const long z0, const long z1) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const long lz = z0 + (long)blockIdx.z;
    if (x >= p.nx || y >= p.ny || lz >= z1) return;
    const size_t vol = p.nx * p.ny * p.nz;
    const size_t linear_id = (p.zoff + lz) * (p.nx * p.ny) + y * p.nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[pidx(x, y, lz, p.nx, p.ny)] = -1.0 + 2.0 * pseudo;
}

template <typename K, typename... Args>
static void run(K kernel, cudaStream_t s, const Params& p, long z0, long z1, Args... args) {
    if (z1 <= z0) return;
    const dim3 block(32, 4, 1);
    const dim3 grid((unsigned)((p.nx + block.x - 1) / block.x), (unsigned)((p.ny + block.y - 1) / block.y), (unsigned)(z1 - z0));
    kernel<<<grid, block, 0, s>>>(args..., p, z0, z1);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long n = (long)c.size();
    bool bad = false;
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal)
    for (long i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = true;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
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
    const bool root0 = (worldRank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            if (root0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (root0) printf("Cahn-Hilliard Phase Separation Benchmark\n");
    if (root0) printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    if (root0) printf("Time steps: %d\n", iterations);
    if (root0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    

    // Ranks that would own fewer than HALO planes sit out
    const int nact = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(1, nz / HALO));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < nact ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;

    // One GPU per rank, round-robin over the GPUs of the node
    {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(node, &lrank);
        MPI_Comm_free(&node);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    const size_t baseZ = nz / nact, remZ = nz % nact;
    const size_t nzl = baseZ + ((size_t)rank < remZ ? 1 : 0);
    const size_t zoff = (size_t)rank * baseZ + std::min<size_t>(rank, remZ);
    const bool hasLo = rank > 0, hasHi = rank < nact - 1;
    const int lo = hasLo ? rank - 1 : MPI_PROC_NULL;
    const int hi = hasHi ? rank + 1 : MPI_PROC_NULL;

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const size_t localElems = plane * (nzl + 2 * HALO);

    Params p{nx, ny, nz, zoff, dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB};

    double *d_a = nullptr, *d_b = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_a, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_b, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localElems * sizeof(double)));

    cudaStream_t sE, sI;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sE, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));
    cudaEvent_t evEdgeMu, evInterDone, evEdgeUpd;
    CUDA_CHECK(cudaEventCreateWithFlags(&evEdgeMu, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evInterDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evEdgeUpd, cudaEventDisableTiming));

    // Pinned staging buffers for halo exchange: 2 planes per direction
    const size_t haloElems = HALO * plane;
    double *h_sendLo, *h_sendHi, *h_recvLo, *h_recvHi;
    CUDA_CHECK(cudaMallocHost(&h_sendLo, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_sendHi, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvLo, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvHi, haloElems * sizeof(double)));
    MPI_Datatype planeType;
    MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Initialize concentration field (including ghost planes that exist globally)
    if (root0) printf("Initializing concentration field...\n");
    const long NZL = (long)nzl;
    run(initKernel, sE, p, hasLo ? -HALO : 0, NZL + (hasHi ? HALO : 0), d_a);
    CUDA_CHECK(cudaStreamSynchronize(sE));

    // Plane ranges (local plane indices). mu is needed on owned planes plus one ghost plane per existing neighbour.
    const long mlo = hasLo ? -1 : 0, mhi = NZL + (hasHi ? 1 : 0);
    const long muLoEnd = std::min<long>(3, mhi);
    const long muHiBeg = std::max<long>(3, NZL - 3);
    const long muIntEnd = muHiBeg;
    const long updLoEnd = std::min<long>(2, NZL);
    const long updHiBeg = std::max<long>(2, NZL - 2);

    double* cur = d_a;
    double* nxt = d_b;

    // Run simulation
    if (root0) printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t haloBytes = haloElems * sizeof(double);
    for (int t = 0; t < iterations; ++t) {
        // Edge stream: mu near both subdomain faces (needs ghost planes of cur)
        if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(sE, evInterDone, 0));
        run(chemPotKernel, sE, p, mlo, muLoEnd, cur, d_mu);
        run(chemPotKernel, sE, p, muHiBeg, mhi, cur, d_mu);
        CUDA_CHECK(cudaEventRecord(evEdgeMu, sE));

        // Interior stream: mu + update in the interior, independent of ghost data
        if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(sI, evEdgeUpd, 0));
        run(chemPotKernel, sI, p, 3, muIntEnd, cur, d_mu);
        CUDA_CHECK(cudaStreamWaitEvent(sI, evEdgeMu, 0));
        run(updateKernel, sI, p, 2, updHiBeg, nxt, cur, d_mu);
        CUDA_CHECK(cudaEventRecord(evInterDone, sI));

        // Edge stream: update near faces, then ship the new boundary planes to neighbours
        run(updateKernel, sE, p, 0, updLoEnd, nxt, cur, d_mu);
        run(updateKernel, sE, p, updHiBeg, NZL, nxt, cur, d_mu);
        CUDA_CHECK(cudaEventRecord(evEdgeUpd, sE));
        if (nact > 1) {
            if (hasLo) CUDA_CHECK(cudaMemcpyAsync(h_sendLo, nxt + HALO * plane, haloBytes, cudaMemcpyDeviceToHost, sE));
            if (hasHi) CUDA_CHECK(cudaMemcpyAsync(h_sendHi, nxt + (size_t)nzl * plane, haloBytes, cudaMemcpyDeviceToHost, sE));
            CUDA_CHECK(cudaStreamSynchronize(sE));
            MPI_Request req[4];
            int nr = 0;
            if (hasLo) {
                MPI_Irecv(h_recvLo, HALO, planeType, lo, 0, comm, &req[nr++]);
                MPI_Isend(h_sendLo, HALO, planeType, lo, 1, comm, &req[nr++]);
            }
            if (hasHi) {
                MPI_Irecv(h_recvHi, HALO, planeType, hi, 1, comm, &req[nr++]);
                MPI_Isend(h_sendHi, HALO, planeType, hi, 0, comm, &req[nr++]);
            }
            MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
            if (hasLo) CUDA_CHECK(cudaMemcpyAsync(nxt, h_recvLo, haloBytes, cudaMemcpyHostToDevice, sE));
            if (hasHi) CUDA_CHECK(cudaMemcpyAsync(nxt + (size_t)(nzl + HALO) * plane, h_recvHi, haloBytes, cudaMemcpyHostToDevice, sE));
        }
        std::swap(cur, nxt);
    }
    CUDA_CHECK(cudaStreamSynchronize(sE));
    CUDA_CHECK(cudaStreamSynchronize(sI));
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field on rank 0 only when it is needed
    std::vector<double> result;
    if (printResults || validate) {
        std::vector<int> counts(nact), displs(nact);
        for (int r = 0; r < nact; ++r) {
            counts[r] = (int)(baseZ + ((size_t)r < remZ ? 1 : 0));
            displs[r] = (int)((size_t)r * baseZ + std::min<size_t>(r, remZ));
        }
        double* h_local = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_local, nzl * plane * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(h_local, cur + HALO * plane, nzl * plane * sizeof(double), cudaMemcpyDeviceToHost));
        if (root0) result.resize(gridSize);
        MPI_Gatherv(h_local, (int)nzl, planeType, root0 ? result.data() : nullptr, counts.data(), displs.data(), planeType, 0, comm);
        CUDA_CHECK(cudaFreeHost(h_local));
    }

    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaEventDestroy(evEdgeMu));
    CUDA_CHECK(cudaEventDestroy(evInterDone));
    CUDA_CHECK(cudaEventDestroy(evEdgeUpd));
    CUDA_CHECK(cudaStreamDestroy(sE));
    CUDA_CHECK(cudaStreamDestroy(sI));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_Type_free(&planeType);

    int rc = 0;
    if (root0) {
        // Print results for external validation
        if (printResults) {
            print_results(result, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return rc;
}
