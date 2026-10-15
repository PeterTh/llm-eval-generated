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

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while (0)

// Domain decomposition: 1D slabs along Z, one MPI rank per GPU.
// Each rank stores its slab plus HALO ghost planes on either side. Because the
// chemical potential needs a 1-plane stencil of c and the update needs a 1-plane
// stencil of mu, a halo of 2 planes of c lets a single exchange per step suffice
// (mu is computed redundantly on one ghost plane).
static constexpr int HALO = 2;

struct Params {
    size_t nx, ny, nz;        // global sizes
    size_t z0;                // global z of local plane 0
    double dx, dy, dz;
    double gamma, e_AA, e_BB, e_AB;
    double D, dt;
};

// Local plane l (may be -HALO..nzl+HALO-1) lives at (l + HALO) * nx * ny.
// Laplacian with clamped boundary conditions (clamping w.r.t. global domain).
__device__ __forceinline__ double laplacian(const double* __restrict__ c, const size_t i,
                                            const size_t x, const size_t y, const size_t gz,
                                            const size_t nx, const size_t ny, const size_t nz,
                                            const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const size_t xp = (x < nx - 1) ? i + 1 : i;
    const size_t xn = (x > 0) ? i - 1 : i;
    const size_t yp = (y < ny - 1) ? i + nx : i;
    const size_t yn = (y > 0) ? i - nx : i;
    const size_t zp = (gz < nz - 1) ? i + plane : i;
    const size_t zn = (gz > 0) ? i - plane : i;
    const double cc = __ldg(&c[i]);
    const double cxx = (__ldg(&c[xp]) + __ldg(&c[xn]) - 2.0 * cc) / (dx * dx);
    const double cyy = (__ldg(&c[yp]) + __ldg(&c[yn]) - 2.0 * cc) / (dy * dy);
    const double czz = (__ldg(&c[zp]) + __ldg(&c[zn]) - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Initialize concentration field (local planes l0..l1-1, clipped to the global domain)
__global__ void initKernel(double* __restrict__ c, const Params p, const int l0, const int l1) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const int l = l0 + (int)blockIdx.z;
    if (x >= p.nx || y >= p.ny || l >= l1) return;
    const long long gzs = (long long)p.z0 + l;
    if (gzs < 0 || gzs >= (long long)p.nz) return;
    const size_t gz = (size_t)gzs;
    const size_t vol = p.nx * p.ny * p.nz;
    const size_t linear_id = gz * (p.nx * p.ny) + y * p.nx + x;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[(size_t)(l + HALO) * (p.nx * p.ny) + y * p.nx + x] = -1.0 + 2.0 * pseudo;
}

// Compute chemical potential on local planes l0..l1-1 (clipped to the global domain)
__global__ void chemPotKernel(const double* __restrict__ c, double* __restrict__ mu, const Params p,
                              const int l0, const int l1) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const int l = l0 + (int)blockIdx.z;
    if (x >= p.nx || y >= p.ny || l >= l1) return;
    const long long gzs = (long long)p.z0 + l;
    if (gzs < 0 || gzs >= (long long)p.nz) return;
    const size_t i = (size_t)(l + HALO) * (p.nx * p.ny) + y * p.nx + x;
    const double cv = __ldg(&c[i]);
    mu[i] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
            + 3.0 * cv + cv * cv * cv
            - p.gamma * laplacian(c, i, x, y, (size_t)gzs, p.nx, p.ny, p.nz, p.dx, p.dy, p.dz);
}

// Cahn-Hilliard update on local planes l0..l1-1
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu, const Params p, const int l0, const int l1) {
    const size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)blockDim.y + threadIdx.y;
    const int l = l0 + (int)blockIdx.z;
    if (x >= p.nx || y >= p.ny || l >= l1) return;
    const size_t gz = p.z0 + (size_t)l;
    const size_t i = (size_t)(l + HALO) * (p.nx * p.ny) + y * p.nx + x;
    cnew[i] = __ldg(&cold[i]) + p.dt * p.D *
              laplacian(mu, i, x, y, gz, p.nx, p.ny, p.nz, p.dx, p.dy, p.dz);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long long n = (long long)c.size();
    const double* d = c.data();

    // Check for NaN or Inf
    bool bad = false;
    #pragma omp parallel for reduction(||:bad)
    for (long long i = 0; i < n; ++i) {
        if (std::isnan(d[i]) || std::isinf(d[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    double minVal = d[0];
    double maxVal = d[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (long long i = 0; i < n; ++i) {
        minVal = std::min(minVal, d[i]);
        maxVal = std::max(maxVal, d[i]);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

static void printUsage(const char* progName) {
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
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

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
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

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

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // Select GPU by node-local rank
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &node);
        int lrank, ndev;
        MPI_Comm_rank(node, &lrank);
        MPI_Comm_free(&node);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Use at most nz/2 ranks so that every slab has at least HALO planes
    int active = (int)std::min<size_t>((size_t)wsize, std::max<size_t>(nz / HALO, 1));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);

    std::vector<double> result;   // full field, rank 0 only
    double elapsed = 0.0;

    if (comm != MPI_COMM_NULL) {
        const int rank = wrank;
        const int size = active;
        const int up = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
        const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

        auto slabStart = [&](int r) { return nz * (size_t)r / (size_t)size; };
        const size_t z0 = slabStart(rank);
        const int nzl = (int)(slabStart(rank + 1) - z0);

        Params p{nx, ny, nz, z0, dx, dy, dz, gamma, e_AA, e_BB, e_AB, D, dt};

        const size_t localElems = (size_t)(nzl + 2 * HALO) * plane;
        const size_t haloElems = (size_t)HALO * plane;
        double *d_a, *d_b, *d_mu;
        CUDA_CHECK(cudaMalloc(&d_a, localElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, localElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, localElems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_a, 0, localElems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_b, 0, localElems * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mu, 0, localElems * sizeof(double)));

        double *h_sendLo, *h_sendHi, *h_recvLo, *h_recvHi;
        CUDA_CHECK(cudaMallocHost(&h_sendLo, haloElems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_sendHi, haloElems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recvLo, haloElems * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recvHi, haloElems * sizeof(double)));

        cudaStream_t sB, sI;
        cudaEvent_t evMu, evB;
        CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sI, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMu, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evB, cudaEventDisableTiming));

        const dim3 block(32, 4, 1);
        const unsigned gx = (unsigned)((nx + block.x - 1) / block.x);
        const unsigned gy = (unsigned)((ny + block.y - 1) / block.y);
        auto grid = [&](int l0, int l1) { return dim3(gx, gy, (unsigned)std::max(l1 - l0, 0)); };

        if (wrank == 0) printf("Initializing concentration field...\n");
        // Initialize owned planes and ghost planes (ghosts computed directly, no exchange needed)
        initKernel<<<grid(-HALO, nzl + HALO), block, 0, sB>>>(d_a, p, -HALO, nzl + HALO);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sB));

        // Plane ranges for boundary / interior updates
        const int botEnd = std::min(HALO, nzl);
        const int topBeg = std::max(nzl - HALO, botEnd);
        const int intBeg = HALO, intEnd = nzl - HALO;

        double* cold = d_a;
        double* cnew = d_b;

        if (wrank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            const bool last = (t == iterations - 1);

            // Chemical potential on owned planes plus one ghost plane on each side
            chemPotKernel<<<grid(-1, nzl + 1), block, 0, sB>>>(cold, d_mu, p, -1, nzl + 1);
            CUDA_CHECK(cudaEventRecord(evMu, sB));

            // Boundary planes first so the halo exchange can overlap with the interior
            updateKernel<<<grid(0, botEnd), block, 0, sB>>>(cnew, cold, d_mu, p, 0, botEnd);
            updateKernel<<<grid(topBeg, nzl), block, 0, sB>>>(cnew, cold, d_mu, p, topBeg, nzl);
            if (!last) {
                if (down != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendLo, cnew + haloElems, haloElems * sizeof(double),
                                               cudaMemcpyDeviceToHost, sB));
                if (up != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_sendHi, cnew + (size_t)nzl * plane, haloElems * sizeof(double),
                                               cudaMemcpyDeviceToHost, sB));
            }
            CUDA_CHECK(cudaEventRecord(evB, sB));

            if (intEnd > intBeg) {
                CUDA_CHECK(cudaStreamWaitEvent(sI, evMu, 0));
                updateKernel<<<grid(intBeg, intEnd), block, 0, sI>>>(cnew, cold, d_mu, p, intBeg, intEnd);
            }
            CUDA_CHECK(cudaGetLastError());

            if (!last) {
                CUDA_CHECK(cudaEventSynchronize(evB));
                MPI_Request req[4];
                int nr = 0;
                if (down != MPI_PROC_NULL) {
                    MPI_Irecv(h_recvLo, (int)haloElems, MPI_DOUBLE, down, 0, comm, &req[nr++]);
                    MPI_Isend(h_sendLo, (int)haloElems, MPI_DOUBLE, down, 1, comm, &req[nr++]);
                }
                if (up != MPI_PROC_NULL) {
                    MPI_Irecv(h_recvHi, (int)haloElems, MPI_DOUBLE, up, 1, comm, &req[nr++]);
                    MPI_Isend(h_sendHi, (int)haloElems, MPI_DOUBLE, up, 0, comm, &req[nr++]);
                }
                MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
                if (down != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(cnew, h_recvLo, haloElems * sizeof(double),
                                               cudaMemcpyHostToDevice, sB));
                if (up != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(cnew + (size_t)(nzl + HALO) * plane, h_recvHi,
                                               haloElems * sizeof(double), cudaMemcpyHostToDevice, sB));
            }
            CUDA_CHECK(cudaStreamSynchronize(sB));
            CUDA_CHECK(cudaStreamSynchronize(sI));

            std::swap(cold, cnew);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);

        // Gather the result on rank 0 (only needed for output / validation)
        if (printResults || validate) {
            const size_t chunk = (size_t)1 << 24;
            if (rank == 0) {
                result.resize(gridSize);
                const size_t n0 = (size_t)nzl * plane;
                CUDA_CHECK(cudaMemcpy(result.data(), cold + haloElems, n0 * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                for (int r = 1; r < size; ++r) {
                    const size_t off = slabStart(r) * plane;
                    const size_t cnt = (slabStart(r + 1) - slabStart(r)) * plane;
                    for (size_t o = 0; o < cnt; o += chunk) {
                        const int m = (int)std::min(chunk, cnt - o);
                        MPI_Recv(result.data() + off + o, m, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                const size_t cnt = (size_t)nzl * plane;
                double* h_out;
                CUDA_CHECK(cudaMallocHost(&h_out, cnt * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(h_out, cold + haloElems, cnt * sizeof(double), cudaMemcpyDeviceToHost));
                for (size_t o = 0; o < cnt; o += chunk) {
                    const int m = (int)std::min(chunk, cnt - o);
                    MPI_Send(h_out + o, m, MPI_DOUBLE, 0, 2, comm);
                }
                CUDA_CHECK(cudaFreeHost(h_out));
            }
        }

        CUDA_CHECK(cudaEventDestroy(evMu));
        CUDA_CHECK(cudaEventDestroy(evB));
        CUDA_CHECK(cudaStreamDestroy(sB));
        CUDA_CHECK(cudaStreamDestroy(sI));
        CUDA_CHECK(cudaFreeHost(h_sendLo));
        CUDA_CHECK(cudaFreeHost(h_sendHi));
        CUDA_CHECK(cudaFreeHost(h_recvLo));
        CUDA_CHECK(cudaFreeHost(h_recvHi));
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_mu));
        MPI_Comm_free(&comm);
    }

    int ret = 0;
    if (wrank == 0) {
        printf("Computation time: %ld ms\n", (long)(elapsed * 1000.0));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

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
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
