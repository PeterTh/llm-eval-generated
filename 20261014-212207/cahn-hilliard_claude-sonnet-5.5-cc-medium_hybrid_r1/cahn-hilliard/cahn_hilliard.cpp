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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Domain is decomposed along Z across MPI ranks. Each rank stores its owned layers plus
// HALO ghost layers on each side. Local layer l corresponds to global z = z0 + l - HALO.
// One exchange of 2 layers of c per step lets mu be computed redundantly on 1 ghost layer,
// so only a single communication phase per time step is needed (overlapped with compute).
constexpr int HALO = 2;

__device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t l, const size_t nx, const size_t ny) {
    return l * (nx * ny) + y * nx + x;
}

// Laplacian with clamped boundary conditions (clamping applies at global Z boundaries only)
__device__ __forceinline__ double laplacian(const double* __restrict__ c, const size_t nx, const size_t ny,
                                            const long nz, const double dx, const double dy, const double dz,
                                            const size_t x, const size_t y, const size_t l, const long gz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (gz < nz - 1) ? l + 1 : l;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (gz > 0) ? l - 1 : l;

    const double cc = c[idx3(x, y, l, nx, ny)];
    const double cxx = (c[idx3(xp, y, l, nx, ny)] + c[idx3(xn, y, l, nx, ny)] - 2.0 * cc) / (dx * dx);
    const double cyy = (c[idx3(x, yp, l, nx, ny)] + c[idx3(x, yn, l, nx, ny)] - 2.0 * cc) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Chemical potential on local layers [lo, hi)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const long nz, const long z0,
                                        const int lo, const int hi,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    for (int l = lo + blockIdx.z; l < hi; l += gridDim.z) {
        const long gz = z0 + l - HALO;
        if (gz < 0 || gz >= nz) continue;
        const double cv = c[idx3(x, y, l, nx, ny)];
        mu[idx3(x, y, l, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                    + 3.0 * cv + cv * cv * cv
                                    - gamma * laplacian(c, nx, ny, nz, dx, dy, dz, x, y, l, gz);
    }
}

// Cahn-Hilliard update on owned layers
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             const size_t nx, const size_t ny, const long nz, const long z0, const int nzl,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    for (int l = HALO + blockIdx.z; l < HALO + nzl; l += gridDim.z) {
        const long gz = z0 + l - HALO;
        const size_t idx = idx3(x, y, l, nx, ny);
        cnew[idx] = cold[idx] + dt * D * laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, l, gz);
    }
}

__global__ void initKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nzTotal,
                           const long z0, const int nzl) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t vol = nx * ny * nzTotal;
    for (int l = blockIdx.z; l < nzl; l += gridDim.z) {
        const size_t z = static_cast<size_t>(z0 + l);
        const size_t linear_id = z * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[idx3(x, y, l + HALO, nx, ny)] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long n = static_cast<long>(c.size());
    const double* p = c.data();
    int bad = 0;
    double minVal = p[0];
    double maxVal = p[0];
#pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal)
    for (long i = 0; i < n; ++i) {
        const double v = p[i];
        if (std::isnan(v) || std::isinf(v)) bad = 1;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    // Bind each rank to a GPU (round-robin among ranks on the same node)
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

    // Active ranks: every rank needs at least 2 Z layers (unless the grid is a single layer)
    int nActive = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), std::max<size_t>(nz / 2, 1)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < nActive ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    const bool root = (rank == 0);

    if (root) {
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

    // Z decomposition
    const size_t base = nz / nActive, rem = nz % nActive;
    const size_t r_ = static_cast<size_t>(rank);
    const int nzl = static_cast<int>(base + (r_ < rem ? 1 : 0));
    const long z0 = static_cast<long>(r_ * base + std::min(r_, rem));
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank < nActive - 1) ? rank + 1 : MPI_PROC_NULL;
    const size_t localLayers = static_cast<size_t>(nzl) + 2 * HALO;
    const size_t localSize = localLayers * plane;
    const size_t haloSize = HALO * plane;

    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, localSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, localSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localSize * sizeof(double)));

    double *h_sendLo, *h_sendHi, *h_recvLo, *h_recvHi;
    CUDA_CHECK(cudaMallocHost(&h_sendLo, haloSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_sendHi, haloSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvLo, haloSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvHi, haloSize * sizeof(double)));

    cudaStream_t sCompute, sComm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    cudaEvent_t evStepDone, evHaloDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&evStepDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHaloDone, cudaEventDisableTiming));

    const dim3 block(32, 4, 1);
    auto gridFor = [&](int layers) {
        return dim3(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::max(1, std::min(layers, 64))));
    };
    auto launchMu = [&](const double* c, int lo, int hi) {
        if (hi <= lo) return;
        chemicalPotentialKernel<<<gridFor(hi - lo), block, 0, sCompute>>>(
            c, d_mu, nx, ny, static_cast<long>(nz), z0, lo, hi, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    };

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    initKernel<<<gridFor(nzl), block, 0, sCompute>>>(d_cold, nx, ny, nz, z0, nzl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(sCompute));

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Communication stream: stage boundary layers of c to host once previous step finished
        CUDA_CHECK(cudaEventRecord(evStepDone, sCompute));
        CUDA_CHECK(cudaStreamWaitEvent(sComm, evStepDone, 0));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendLo, d_cold + HALO * plane, haloSize * sizeof(double),
                                       cudaMemcpyDeviceToHost, sComm));
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendHi, d_cold + static_cast<size_t>(nzl) * plane, haloSize * sizeof(double),
                                       cudaMemcpyDeviceToHost, sComm));

        // Overlap: mu on layers that depend only on owned data
        if (nzl >= 2) launchMu(d_cold, HALO + 1, nzl + 1);

        CUDA_CHECK(cudaStreamSynchronize(sComm));
        MPI_Request req[4];
        int nreq = 0;
        if (down != MPI_PROC_NULL) {
            MPI_Isend(h_sendLo, static_cast<int>(haloSize), MPI_DOUBLE, down, 0, comm, &req[nreq++]);
            MPI_Irecv(h_recvLo, static_cast<int>(haloSize), MPI_DOUBLE, down, 1, comm, &req[nreq++]);
        }
        if (up != MPI_PROC_NULL) {
            MPI_Isend(h_sendHi, static_cast<int>(haloSize), MPI_DOUBLE, up, 1, comm, &req[nreq++]);
            MPI_Irecv(h_recvHi, static_cast<int>(haloSize), MPI_DOUBLE, up, 0, comm, &req[nreq++]);
        }
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_cold, h_recvLo, haloSize * sizeof(double), cudaMemcpyHostToDevice, sComm));
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_cold + static_cast<size_t>(nzl + HALO) * plane, h_recvHi,
                                       haloSize * sizeof(double), cudaMemcpyHostToDevice, sComm));
        CUDA_CHECK(cudaEventRecord(evHaloDone, sComm));
        CUDA_CHECK(cudaStreamWaitEvent(sCompute, evHaloDone, 0));

        // mu on remaining layers (ghost + boundary-adjacent)
        if (nzl >= 2) {
            launchMu(d_cold, 1, HALO + 1);
            launchMu(d_cold, nzl + 1, nzl + HALO + 1);
        } else {
            launchMu(d_cold, 1, nzl + HALO + 1);
        }

        // Update concentration
        updateKernel<<<gridFor(nzl), block, 0, sCompute>>>(d_cnew, d_cold, d_mu, nx, ny, static_cast<long>(nz), z0,
                                                            nzl, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(sCompute));
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field on root only if needed
    bool valid = true;
    if (printResults || validate) {
        const size_t ownedSize = static_cast<size_t>(nzl) * plane;
        const size_t chunk = size_t(1) << 26;
        if (root) {
            std::vector<double> cold(gridSize);
            CUDA_CHECK(cudaMemcpy(cold.data(), d_cold + haloSize, ownedSize * sizeof(double), cudaMemcpyDeviceToHost));
            size_t off = ownedSize;
            for (int r = 1; r < nActive; ++r) {
                const size_t rn = static_cast<size_t>(base + (static_cast<size_t>(r) < rem ? 1 : 0)) * plane;
                for (size_t done = 0; done < rn; done += chunk) {
                    const int cnt = static_cast<int>(std::min(chunk, rn - done));
                    MPI_Recv(cold.data() + off + done, cnt, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                }
                off += rn;
            }
            if (printResults) {
                print_results(cold, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(cold, nx, ny, nz);
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            }
        } else {
            std::vector<double> mine(ownedSize);
            CUDA_CHECK(cudaMemcpy(mine.data(), d_cold + haloSize, ownedSize * sizeof(double), cudaMemcpyDeviceToHost));
            for (size_t done = 0; done < ownedSize; done += chunk) {
                const int cnt = static_cast<int>(std::min(chunk, ownedSize - done));
                MPI_Send(mine.data() + done, cnt, MPI_DOUBLE, 0, 2, comm);
            }
        }
    }

    CUDA_CHECK(cudaEventDestroy(evStepDone));
    CUDA_CHECK(cudaEventDestroy(evHaloDone));
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(sComm));
    CUDA_CHECK(cudaFreeHost(h_sendLo));
    CUDA_CHECK(cudaFreeHost(h_sendHi));
    CUDA_CHECK(cudaFreeHost(h_recvLo));
    CUDA_CHECK(cudaFreeHost(h_recvHi));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
