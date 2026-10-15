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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// Domain is decomposed in slabs along z over MPI ranks (one GPU per rank).
// Each local array holds HALO ghost planes on each side. Ghost planes of the
// concentration are exchanged once per step (depth 2): chemical potential is
// computed redundantly on one ghost plane, so no second exchange is needed.
constexpr int HALO = 2;

struct Params {
    size_t nx, ny, nz;   // global grid
    long z0;             // global z of first owned plane
    double dx, dy, dz, gamma, e_AA, e_BB, e_AB, D, dt;
};

// Laplacian with clamped boundary conditions. lz is the local plane, gz the global plane.
__device__ __forceinline__ double laplacian(const double* __restrict__ c, const Params& p,
                                            const size_t x, const size_t y, const size_t lz, const long gz) {
    const size_t nx = p.nx, ny = p.ny, plane = nx * ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (gz < (long)p.nz - 1) ? lz + 1 : lz;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (gz > 0) ? lz - 1 : lz;
    const size_t base = lz * plane + y * nx;
    const double cc = c[base + x];

    const double cxx = (c[base + xp] + c[base + xn] - 2.0 * cc) / (p.dx * p.dx);
    const double cyy = (c[lz * plane + yp * nx + x] + c[lz * plane + yn * nx + x] - 2.0 * cc) / (p.dy * p.dy);
    const double czz = (c[zp * plane + y * nx + x] + c[zn * plane + y * nx + x] - 2.0 * cc) / (p.dz * p.dz);
    return cxx + cyy + czz;
}

// Chemical potential on local planes [lzBeg, lzEnd)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const Params p, const int lzBeg) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;
    const size_t lz = lzBeg + blockIdx.z;
    const long gz = p.z0 + (long)lz - HALO;
    if (gz < 0 || gz >= (long)p.nz) return;
    const size_t idx = lz * (p.nx * p.ny) + y * p.nx + x;
    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
              + 3.0 * cv + cv * cv * cv
              - p.gamma * laplacian(c, p, x, y, lz, gz);
}

// Concentration update on local planes [lzBeg, lzEnd)
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, const Params p, const int lzBeg) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;
    const size_t lz = lzBeg + blockIdx.z;
    const long gz = p.z0 + (long)lz - HALO;
    const size_t idx = lz * (p.nx * p.ny) + y * p.nx + x;
    cnew[idx] = cold[idx] + p.dt * p.D * laplacian(mu, p, x, y, lz, gz);
}

// Initialize concentration (including ghost planes) from the global index
__global__ void initKernel(double* __restrict__ c, const Params p, const int lzBeg) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;
    const size_t lz = lzBeg + blockIdx.z;
    const long gz = p.z0 + (long)lz - HALO;
    if (gz < 0 || gz >= (long)p.nz) return;
    const size_t vol = p.nx * p.ny * p.nz;
    const size_t linear_id = (size_t)gz * (p.nx * p.ny) + y * p.nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[lz * (p.nx * p.ny) + y * p.nx + x] = -1.0 + 2.0 * pseudo;
}

#define BX 32
#define BY 4
static inline dim3 gridFor(const Params& p, int planes) {
    return dim3((unsigned)((p.nx + BX - 1) / BX), (unsigned)((p.ny + BY - 1) / BY), (unsigned)planes);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long n = (long)c.size();
    // Check for NaN or Inf
    int bad = 0;
    #pragma omp parallel for reduction(|:bad)
    for (long i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (long i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU: one per rank, round-robin over the GPUs on the node
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node);
        int lr; MPI_Comm_rank(node, &lr);
        MPI_Comm_free(&node);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lr % ndev));
    }

    // Only ranks that own at least HALO planes take part
    const size_t gridSize = nx * ny * nz;
    int active = (int)std::min<size_t>((size_t)worldSize, std::max<size_t>(1, nz / HALO));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const int size = active;

    const size_t zBase = nz / size, zRem = nz % size;
    const size_t nzl = zBase + ((size_t)rank < zRem ? 1 : 0);
    const size_t z0 = (size_t)rank * zBase + std::min((size_t)rank, zRem);

    Params p;
    p.nx = nx; p.ny = ny; p.nz = nz; p.z0 = (long)z0;
    p.dx = 1.0; p.dy = 1.0; p.dz = 1.0;
    p.dt = 0.01;
    p.e_AA = -(2.0 / 9.0);
    p.e_BB = -(2.0 / 9.0);
    p.e_AB = (2.0 / 9.0);
    p.gamma = 0.5;
    p.D = 1.0;

    const size_t planeSize = nx * ny;
    const size_t localElems = planeSize * (nzl + 2 * HALO);
    const size_t haloElems = planeSize * HALO;

    double *dA, *dB, *dMu;
    CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMu, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(dA, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(dB, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(dMu, 0, localElems * sizeof(double)));

    double *hSend[2], *hRecv[2];  // [0] = to/from lower neighbor, [1] = upper neighbor
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMallocHost(&hSend[i], haloElems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hRecv[i], haloElems * sizeof(double)));
    }

    cudaStream_t sB, sI;  // boundary/comm stream, interior stream
    CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));
    cudaEvent_t evMu, evInt;
    CUDA_CHECK(cudaEventCreateWithFlags(&evMu, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evInt, cudaEventDisableTiming));

    const dim3 block(BX, BY);
    const int L = (int)nzl;

    if (root) printf("Initializing concentration field...\n");
    initKernel<<<gridFor(p, L + 2 * HALO), block, 0, sB>>>(dA, p, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(sB));

    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
    const int bBeg0 = HALO, bEnd0 = HALO + std::min(HALO, L);     // bottom boundary planes
    const int tBeg = std::max(L, 2 * HALO), tEnd = L + HALO;      // top boundary planes (local idx L..L+1)
    const int iBeg = 2 * HALO, iEnd = L;                          // interior planes
    const size_t bytesHalo = haloElems * sizeof(double);

    double* cold = dA;
    double* cnew = dB;
    CUDA_CHECK(cudaEventRecord(evInt, sI));

    for (int t = 0; t < iterations; ++t) {
        // Chemical potential on owned planes plus one ghost plane on each side
        CUDA_CHECK(cudaStreamWaitEvent(sB, evInt, 0));
        chemicalPotentialKernel<<<gridFor(p, L + 2), block, 0, sB>>>(cold, dMu, p, HALO - 1);
        CUDA_CHECK(cudaEventRecord(evMu, sB));

        // Boundary planes first so the halo exchange overlaps with the interior update
        updateKernel<<<gridFor(p, bEnd0 - bBeg0), block, 0, sB>>>(cnew, cold, dMu, p, bBeg0);
        if (tEnd > tBeg)
            updateKernel<<<gridFor(p, tEnd - tBeg), block, 0, sB>>>(cnew, cold, dMu, p, tBeg);
        if (size > 1) {
            if (lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSend[0], cnew + HALO * planeSize, bytesHalo, cudaMemcpyDeviceToHost, sB));
            if (upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSend[1], cnew + (size_t)L * planeSize, bytesHalo, cudaMemcpyDeviceToHost, sB));
        }

        if (iEnd > iBeg) {
            CUDA_CHECK(cudaStreamWaitEvent(sI, evMu, 0));
            updateKernel<<<gridFor(p, iEnd - iBeg), block, 0, sI>>>(cnew, cold, dMu, p, iBeg);
        }
        CUDA_CHECK(cudaEventRecord(evInt, sI));

        if (size > 1) {
            CUDA_CHECK(cudaStreamSynchronize(sB));
            MPI_Request req[4];
            int nr = 0;
            if (lower != MPI_PROC_NULL) {
                MPI_Irecv(hRecv[0], (int)haloElems, MPI_DOUBLE, lower, 1, comm, &req[nr++]);
                MPI_Isend(hSend[0], (int)haloElems, MPI_DOUBLE, lower, 0, comm, &req[nr++]);
            }
            if (upper != MPI_PROC_NULL) {
                MPI_Irecv(hRecv[1], (int)haloElems, MPI_DOUBLE, upper, 0, comm, &req[nr++]);
                MPI_Isend(hSend[1], (int)haloElems, MPI_DOUBLE, upper, 1, comm, &req[nr++]);
            }
            MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
            if (lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(cnew, hRecv[0], bytesHalo, cudaMemcpyHostToDevice, sB));
            if (upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(cnew + (size_t)(L + HALO) * planeSize, hRecv[1], bytesHalo, cudaMemcpyHostToDevice, sB));
        }
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaStreamSynchronize(sB));
    CUDA_CHECK(cudaStreamSynchronize(sI));
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (root) {
        printf("Computation time: %ld ms\n", max_ms);
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the owned planes on rank 0
    std::vector<double> result;
    {
        std::vector<double> local(planeSize * nzl);
        CUDA_CHECK(cudaMemcpy(local.data(), cold + HALO * planeSize, local.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> counts(size), displs(size);
        if (root) {
            result.resize(gridSize);
            for (int r = 0; r < size; ++r) {
                size_t nr = zBase + ((size_t)r < zRem ? 1 : 0);
                size_t zr = (size_t)r * zBase + std::min((size_t)r, zRem);
                counts[r] = (int)(nr * planeSize);
                displs[r] = (int)(zr * planeSize);
            }
        }
        // Chunked to stay within int counts for very large grids
        if (gridSize < (size_t)1 << 30) {
            MPI_Gatherv(local.data(), (int)local.size(), MPI_DOUBLE, result.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, 0, comm);
        } else {
            for (int r = 0; r < size; ++r) {
                size_t nr = zBase + ((size_t)r < zRem ? 1 : 0);
                size_t zr = (size_t)r * zBase + std::min((size_t)r, zRem);
                size_t n = nr * planeSize, off = zr * planeSize;
                if (r == 0) {
                    if (root) std::memcpy(result.data(), local.data(), n * sizeof(double));
                    continue;
                }
                for (size_t o = 0; o < n; o += (size_t)1 << 28) {
                    int len = (int)std::min<size_t>((size_t)1 << 28, n - o);
                    if (rank == r) MPI_Send(local.data() + o, len, MPI_DOUBLE, 0, 7, comm);
                    else if (root) MPI_Recv(result.data() + off + o, len, MPI_DOUBLE, r, 7, comm, MPI_STATUS_IGNORE);
                }
            }
        }
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dMu);
    for (int i = 0; i < 2; ++i) { cudaFreeHost(hSend[i]); cudaFreeHost(hRecv[i]); }
    cudaStreamDestroy(sB); cudaStreamDestroy(sI);
    cudaEventDestroy(evMu); cudaEventDestroy(evInt);

    int ret = 0;
    if (root) {
        if (printResults) {
            print_results(result, "Concentration");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
