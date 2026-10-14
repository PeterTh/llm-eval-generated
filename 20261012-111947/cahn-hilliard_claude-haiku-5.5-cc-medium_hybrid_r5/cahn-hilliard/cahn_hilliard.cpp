#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        const cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    } while (0)

// Physical parameters, passed by value to the device kernels
struct Physics {
    double dx, dy, dz, dt;
    double e_AA, e_BB, e_AB, gamma, D;
};

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions. Fields are stored as
// [ghost | owned planes | ghost] along z, so z is a local plane index and the
// z-neighbours are always present (ghost planes hold the clamped boundary values).
__host__ __device__ inline double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                                                   const double dx, const double dy, const double dz,
                                                   const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t center = idx3(x, y, z, nx, ny);
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[center]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[center]) / (dy * dy);
    const double czz = (c[center + nx * ny] + c[center - nx * ny] -
                  2.0 * c[center]) / (dz * dz);

    return cxx + cyy + czz;
}

// One thread per cell of planes [zBegin, zBegin + numCells / (nx*ny))
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t zBegin,
                                        const size_t numCells, const Physics p) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= numCells) return;

    const size_t nxy = nx * ny;
    const size_t z = zBegin + i / nxy;
    const size_t y = (i % nxy) / nx;
    const size_t x = i % nx;
    const size_t idx = idx3(x, y, z, nx, ny);

    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
             + 3.0 * cv + cv * cv * cv
             - p.gamma * computeLaplacian(c, nx, ny, p.dx, p.dy, p.dz, x, y, z);
}

// Cahn-Hilliard update step, same layout and thread mapping as the chemical potential kernel
__global__ void cahnHilliardUpdateKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t zBegin,
                                         const size_t numCells, const Physics p) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= numCells) return;

    const size_t nxy = nx * ny;
    const size_t z = zBegin + i / nxy;
    const size_t y = (i % nxy) / nx;
    const size_t x = i % nx;
    const size_t idx = idx3(x, y, z, nx, ny);

    cnew[idx] = cold[idx] + p.dt * p.D *
               computeLaplacian(mu, nx, ny, p.dx, p.dy, p.dz, x, y, z);
}

constexpr unsigned kThreadsPerBlock = 256;

// Launch on owned/ghost planes [zBegin, zEnd) of a rank-local field
void launchChemicalPotential(cudaStream_t s, const double* c, double* mu, const size_t nx, const size_t ny,
                             const size_t zBegin, const size_t zEnd, const Physics& p) {
    if (zEnd <= zBegin) return;
    const size_t numCells = nx * ny * (zEnd - zBegin);
    const unsigned blocks = static_cast<unsigned>((numCells + kThreadsPerBlock - 1) / kThreadsPerBlock);
    chemicalPotentialKernel<<<blocks, kThreadsPerBlock, 0, s>>>(c, mu, nx, ny, zBegin, numCells, p);
    CUDA_CHECK(cudaGetLastError());
}

void launchCahnHilliardUpdate(cudaStream_t s, const double* cold, double* cnew, const double* mu,
                              const size_t nx, const size_t ny, const size_t zBegin, const size_t zEnd,
                              const Physics& p) {
    if (zEnd <= zBegin) return;
    const size_t numCells = nx * ny * (zEnd - zBegin);
    const unsigned blocks = static_cast<unsigned>((numCells + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cahnHilliardUpdateKernel<<<blocks, kThreadsPerBlock, 0, s>>>(cold, cnew, mu, nx, ny, zBegin, numCells, p);
    CUDA_CHECK(cudaGetLastError());
}

// Refresh the two ghost planes of a field [ghost | nzl owned planes | ghost] from the
// neighbouring ranks. Halo planes are staged through pinned host buffers (no CUDA-aware
// MPI required). At the global z boundaries the ghost is a copy of the adjacent owned
// plane, which reproduces the clamped boundary conditions of the original code.
// `stream` must be ordered after all work that produces the owned planes of `field`.
void exchangeHalos(double* field, const size_t nxy, const size_t nzl, const int down, const int up,
                   MPI_Comm comm, cudaStream_t stream, double* halo) {
    double* sendLo = halo;
    double* sendHi = halo + nxy;
    double* recvLo = halo + 2 * nxy;
    double* recvHi = halo + 3 * nxy;

    double* lowerGhost = field;
    double* lowerOwned = field + nxy;
    double* upperOwned = field + nzl * nxy;
    double* upperGhost = field + (nzl + 1) * nxy;
    const size_t planeBytes = nxy * sizeof(double);
    const int count = static_cast<int>(nxy);

    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendLo, lowerOwned, planeBytes, cudaMemcpyDeviceToHost, stream));
    if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(sendHi, upperOwned, planeBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Request req[4];
    MPI_Irecv(recvLo, count, MPI_DOUBLE, down, 0, comm, &req[0]);
    MPI_Irecv(recvHi, count, MPI_DOUBLE, up, 1, comm, &req[1]);
    MPI_Isend(sendLo, count, MPI_DOUBLE, down, 1, comm, &req[2]);
    MPI_Isend(sendHi, count, MPI_DOUBLE, up, 0, comm, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

    if (down != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(lowerGhost, recvLo, planeBytes, cudaMemcpyHostToDevice, stream));
    else CUDA_CHECK(cudaMemcpyAsync(lowerGhost, lowerOwned, planeBytes, cudaMemcpyDeviceToDevice, stream));
    if (up != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpyAsync(upperGhost, recvHi, planeBytes, cudaMemcpyHostToDevice, stream));
    else CUDA_CHECK(cudaMemcpyAsync(upperGhost, upperOwned, planeBytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// z-slab owned by a rank: global start plane and number of planes
struct Slab {
    size_t z0;
    size_t nzl;
};

Slab slabFor(const int rank, const int size, const size_t nz) {
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Initialize the owned slab [z0, z0 + nzl) of the concentration field (OpenMP over rows)
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(2)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx3(x, y, z0 + z, nx, ny);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool nonFinite = false;
    #pragma omp parallel for reduction(||:nonFinite)
    for (size_t i = 0; i < c.size(); ++i) {
        nonFinite = nonFinite || std::isnan(c[i]) || std::isinf(c[i]);
    }
    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
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

int runBenchmark(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool root = worldRank == 0;

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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) fprintf(stderr, "Error: grid dimensions must be positive\n");
        return 1;
    }

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
    const Physics phys{dx, dy, dz, dt, e_AA, e_BB, e_AB, gamma, D};

    // Node-local rank: selects the GPU and partitions the host cores between ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: no CUDA devices visible to rank %d\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    // Each rank owns at least one z-plane; surplus ranks sit out the simulation
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const int simRanksAvailable = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), nz));
    MPI_Comm simComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < simRanksAvailable ? 0 : MPI_UNDEFINED, worldRank, &simComm);
    if (simComm == MPI_COMM_NULL) return 0;

    int simRank = 0;
    int simSize = 1;
    MPI_Comm_rank(simComm, &simRank);
    MPI_Comm_size(simComm, &simSize);

    const Slab slab = slabFor(simRank, simSize, nz);
    const size_t nzl = slab.nzl;
    const size_t nxy = nx * ny;
    const int down = (simRank > 0) ? simRank - 1 : MPI_PROC_NULL;
    const int up = (simRank < simSize - 1) ? simRank + 1 : MPI_PROC_NULL;

    // Device state: two concentration buffers (ping-pong), chemical potential, host staging
    const size_t fieldBytes = (nzl + 2) * nxy * sizeof(double);
    double* cA = nullptr;
    double* cB = nullptr;
    double* mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cA, fieldBytes));
    CUDA_CHECK(cudaMalloc(&cB, fieldBytes));
    CUDA_CHECK(cudaMalloc(&mu, fieldBytes));
    double* halo = nullptr;
    CUDA_CHECK(cudaMallocHost(&halo, 4 * nxy * sizeof(double)));

    cudaStream_t sComp;
    cudaStream_t sComm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    cudaEvent_t evReady;
    cudaEvent_t evMu;
    CUDA_CHECK(cudaEventCreateWithFlags(&evReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMu, cudaEventDisableTiming));

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    std::vector<double> owned(nzl * nxy);
    initializeConcentration(owned.data(), nx, ny, nz, slab.z0, nzl);
    // Stream-ordered (not the legacy default stream) so the halo and event ordering below covers it
    CUDA_CHECK(cudaMemcpyAsync(cA + nxy, owned.data(), nzl * nxy * sizeof(double), cudaMemcpyHostToDevice, sComp));
    CUDA_CHECK(cudaStreamSynchronize(sComp));

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(simComm);
    const double tStart = MPI_Wtime();

    double* cold = cA;
    double* cnew = cB;
    for (int t = 0; t < iterations; ++t) {
        // Chemical potential: interior planes need no halo, so they overlap the c exchange
        CUDA_CHECK(cudaEventRecord(evReady, sComp));
        CUDA_CHECK(cudaStreamWaitEvent(sComm, evReady, 0));
        launchChemicalPotential(sComp, cold, mu, nx, ny, 2, nzl, phys);
        exchangeHalos(cold, nxy, nzl, down, up, simComm, sComm, halo);
        launchChemicalPotential(sComp, cold, mu, nx, ny, 1, 2, phys);
        if (nzl > 1) launchChemicalPotential(sComp, cold, mu, nx, ny, nzl, nzl + 1, phys);

        // Update concentration: interior planes overlap the mu halo exchange
        CUDA_CHECK(cudaEventRecord(evMu, sComp));
        CUDA_CHECK(cudaStreamWaitEvent(sComm, evMu, 0));
        launchCahnHilliardUpdate(sComp, cold, cnew, mu, nx, ny, 2, nzl, phys);
        exchangeHalos(mu, nxy, nzl, down, up, simComm, sComm, halo);
        launchCahnHilliardUpdate(sComp, cold, cnew, mu, nx, ny, 1, 2, phys);
        if (nzl > 1) launchCahnHilliardUpdate(sComp, cold, cnew, mu, nx, ny, nzl, nzl + 1, phys);

        // Swap buffers
        std::swap(cold, cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(simComm);
    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, simComm);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)(nx * ny * nz) * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the owned slabs on rank 0 (z-major layout: each slab is contiguous)
    CUDA_CHECK(cudaMemcpy(owned.data(), cold + nxy, nzl * nxy * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> full;
    if (root) full.resize(nx * ny * nz);

    std::vector<int> counts(simSize);
    std::vector<int> displs(simSize);
    for (int r = 0; r < simSize; ++r) {
        const Slab s = slabFor(r, simSize, nz);
        counts[r] = static_cast<int>(s.nzl);
        displs[r] = static_cast<int>(s.z0);
    }
    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(nxy), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);
    MPI_Gatherv(owned.data(), static_cast<int>(nzl), planeType,
                root ? full.data() : nullptr, counts.data(), displs.data(), planeType, 0, simComm);
    MPI_Type_free(&planeType);

    CUDA_CHECK(cudaEventDestroy(evReady));
    CUDA_CHECK(cudaEventDestroy(evMu));
    CUDA_CHECK(cudaStreamDestroy(sComp));
    CUDA_CHECK(cudaStreamDestroy(sComm));
    CUDA_CHECK(cudaFreeHost(halo));
    CUDA_CHECK(cudaFree(cA));
    CUDA_CHECK(cudaFree(cB));
    CUDA_CHECK(cudaFree(mu));
    MPI_Comm_free(&simComm);

    if (!root) return 0;

    // Print results for external validation
    if (printResults) {
        print_results(full, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(full, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
