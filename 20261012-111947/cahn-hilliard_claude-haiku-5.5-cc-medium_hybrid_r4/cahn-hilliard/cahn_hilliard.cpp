#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

// Ghost planes on each side of a rank's z-slab. Two are needed because the
// chemical potential is evaluated one plane past the slab, and that in turn
// needs the concentration two planes past it.
constexpr int GHOST = 2;

// Thread tile for the (x, y) extent of each plane
constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 8;

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                 \
        }                                                                 \
    } while (0)

// Grid and physical parameters, passed to kernels by value
struct Params {
    int nx, ny;
    int lz;             // planes owned by this rank
    bool lowerEdge;     // slab starts at global z = 0 (clamped boundary)
    bool upperEdge;     // slab ends at global z = nz - 1 (clamped boundary)
    double dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB;
};

// Storage index of cell (x, y, p), where local plane p runs over [-GHOST, lz + GHOST)
__device__ __forceinline__ size_t cellIndex(const Params& q, const int x, const int y, const int p) {
    return static_cast<size_t>(p + GHOST) * q.nx * q.ny + static_cast<size_t>(y) * q.nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ __forceinline__ double laplacian(const double* f, const Params& q,
                                            const int x, const int y, const int p) {
    const int xp = (x < q.nx - 1) ? x + 1 : x;
    const int yp = (y < q.ny - 1) ? y + 1 : y;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    // Clamp only at the global domain faces; interior slab faces read ghost planes
    const int zp = (p == q.lz - 1 && q.upperEdge) ? p : p + 1;
    const int zn = (p == 0 && q.lowerEdge) ? p : p - 1;

    const double center = f[cellIndex(q, x, y, p)];
    const double cxx = (f[cellIndex(q, xp, y, p)] + f[cellIndex(q, xn, y, p)] -
                        2.0 * center) / (q.dx * q.dx);
    const double cyy = (f[cellIndex(q, x, yp, p)] + f[cellIndex(q, x, yn, p)] -
                        2.0 * center) / (q.dy * q.dy);
    const double czz = (f[cellIndex(q, x, y, zp)] + f[cellIndex(q, x, y, zn)] -
                        2.0 * center) / (q.dz * q.dz);
    return cxx + cyy + czz;
}

// Compute chemical potential on local planes [pBegin, pBegin + count)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const Params q, const int pBegin, const int count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= q.nx || y >= q.ny) return;
    for (int k = blockIdx.z; k < count; k += gridDim.z) {
        const int p = pBegin + k;
        const size_t i = cellIndex(q, x, y, p);
        const double cv = c[i];
        mu[i] = 4.5 * ((cv + 1.0) * q.e_AA + (cv - 1.0) * q.e_BB - 2.0 * cv * q.e_AB)
              + 3.0 * cv + cv * cv * cv
              - q.gamma * laplacian(c, q, x, y, p);
    }
}

// Cahn-Hilliard update step on local planes [pBegin, pBegin + count)
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu, const Params q,
                                         const int pBegin, const int count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= q.nx || y >= q.ny) return;
    for (int k = blockIdx.z; k < count; k += gridDim.z) {
        const int p = pBegin + k;
        const size_t i = cellIndex(q, x, y, p);
        cnew[i] = cold[i] + q.dt * q.D * laplacian(mu, q, x, y, p);
    }
}

dim3 planeLaunchGrid(const Params& q, const int count) {
    const unsigned zBlocks = static_cast<unsigned>(std::min(count, 65535));
    return dim3((q.nx + kBlockX - 1) / kBlockX, (q.ny + kBlockY - 1) / kBlockY, zBlocks);
}

void launchChemicalPotential(cudaStream_t s, const Params& q, const double* c, double* mu,
                             const int pBegin, const int count) {
    if (count <= 0) return;
    chemicalPotentialKernel<<<planeLaunchGrid(q, count), dim3(kBlockX, kBlockY), 0, s>>>(
        c, mu, q, pBegin, count);
    CUDA_CHECK(cudaGetLastError());
}

void launchCahnHilliardUpdate(cudaStream_t s, const Params& q, double* cnew, const double* cold,
                              const double* mu, const int pBegin, const int count) {
    if (count <= 0) return;
    cahnHilliardUpdateKernel<<<planeLaunchGrid(q, count), dim3(kBlockX, kBlockY), 0, s>>>(
        cnew, cold, mu, q, pBegin, count);
    CUDA_CHECK(cudaGetLastError());
}

// Initialize concentration field (host side, OpenMP over the local slab)
void initializeConcentration(std::vector<double>& c, const size_t firstCell, const size_t vol) {
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        // Generate pseudo-random value in [-1, 1] from the global linear index
        const size_t linear_id = firstCell + static_cast<size_t>(i);
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[static_cast<size_t>(i)] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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

// Blocked 1D decomposition of nz planes over the active ranks. Only ranks that can
// keep at least GHOST planes are active, so halo sends never reach past a slab.
// Inactive ranks own nothing and only take part in the final gather.
struct Slab {
    int z0;
    int lz;
};

Slab slabFor(const int rank, const int activeRanks, const int nz) {
    if (rank >= activeRanks) return {0, 0};
    const int base = nz / activeRanks;
    const int rem = nz % activeRanks;
    return {rank * base + std::min(rank, rem), base + (rank < rem ? 1 : 0)};
}

int runBenchmark(int argc, char** argv, const int rank, const int size) {
    const bool root = (rank == 0);

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

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // Domain decomposition along z
    const int nzEff = (gridSize > 0) ? static_cast<int>(nz) : 0;
    const int activeRanks = std::max(1, std::min(size, nzEff / GHOST));
    const Slab slab = slabFor(rank, activeRanks, nzEff);
    const int lz = slab.lz;
    const int lower = (lz > 0 && rank > 0) ? rank - 1 : -1;
    const int upper = (lz > 0 && rank + 1 < activeRanks) ? rank + 1 : -1;

    Params q{};
    q.nx = static_cast<int>(nx);
    q.ny = static_cast<int>(ny);
    q.lz = lz;
    q.lowerEdge = (slab.z0 == 0);
    q.upperEdge = (slab.z0 + lz == nzEff);
    q.dx = dx;
    q.dy = dy;
    q.dz = dz;
    q.dt = dt;
    q.D = D;
    q.gamma = gamma;
    q.e_AA = e_AA;
    q.e_BB = e_BB;
    q.e_AB = e_AB;

    // One GPU per rank, round-robin over the devices visible on this node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Device storage: two concentration buffers and mu, each with ghost planes
    const size_t planeBytes = plane * sizeof(double);
    const size_t storageBytes = static_cast<size_t>(lz + 2 * GHOST) * planeBytes;
    const size_t haloBytes = static_cast<size_t>(GHOST) * planeBytes;
    const int haloCount = static_cast<int>(GHOST * plane);

    double* bufA = nullptr;
    double* bufB = nullptr;
    double* mu = nullptr;
    double* hSendLo = nullptr;
    double* hSendHi = nullptr;
    double* hRecvLo = nullptr;
    double* hRecvHi = nullptr;
    cudaStream_t sCompute = nullptr;
    cudaStream_t sComm = nullptr;
    cudaEvent_t evStage = nullptr;
    cudaEvent_t evHalo = nullptr;

    if (lz > 0) {
        CUDA_CHECK(cudaMalloc(&bufA, storageBytes));
        CUDA_CHECK(cudaMalloc(&bufB, storageBytes));
        CUDA_CHECK(cudaMalloc(&mu, storageBytes));
        CUDA_CHECK(cudaMemset(bufA, 0, storageBytes));
        CUDA_CHECK(cudaMemset(bufB, 0, storageBytes));
        CUDA_CHECK(cudaMemset(mu, 0, storageBytes));

        CUDA_CHECK(cudaMallocHost(&hSendLo, haloBytes));
        CUDA_CHECK(cudaMallocHost(&hSendHi, haloBytes));
        CUDA_CHECK(cudaMallocHost(&hRecvLo, haloBytes));
        CUDA_CHECK(cudaMallocHost(&hRecvHi, haloBytes));

        CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evStage, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));
    }

    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    double* front = bufA;
    double* back = bufB;
    if (lz > 0) {
        std::vector<double> init(static_cast<size_t>(lz) * plane);
        initializeConcentration(init, static_cast<size_t>(slab.z0) * plane, gridSize);
        CUDA_CHECK(cudaMemcpy(front + GHOST * plane, init.data(), init.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (lz > 0) {
        for (int t = 0; t < iterations; ++t) {
            const bool haloNeeded = (lower >= 0 || upper >= 0);

            if (haloNeeded) {
                // Boundary planes of `front` are final once the previous step has finished
                CUDA_CHECK(cudaStreamWaitEvent(sComm, evStage, 0));
                if (lower >= 0)
                    CUDA_CHECK(cudaMemcpyAsync(hSendLo, front + GHOST * plane, haloBytes,
                                               cudaMemcpyDeviceToHost, sComm));
                if (upper >= 0)
                    CUDA_CHECK(cudaMemcpyAsync(hSendHi, front + lz * plane, haloBytes,
                                               cudaMemcpyDeviceToHost, sComm));
                CUDA_CHECK(cudaStreamSynchronize(sComm));
            }

            // Stage A: planes whose stencil stays inside the slab. Runs on the GPU while
            // the halo exchange proceeds on the host.
            launchChemicalPotential(sCompute, q, front, mu, 1, lz - 2);
            launchCahnHilliardUpdate(sCompute, q, back, front, mu, 2, lz - 4);

            if (haloNeeded) {
                MPI_Request reqs[4];
                int n = 0;
                if (lower >= 0) {
                    MPI_Irecv(hRecvLo, haloCount, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &reqs[n++]);
                    MPI_Isend(hSendLo, haloCount, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &reqs[n++]);
                }
                if (upper >= 0) {
                    MPI_Irecv(hRecvHi, haloCount, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &reqs[n++]);
                    MPI_Isend(hSendHi, haloCount, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &reqs[n++]);
                }
                MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);

                if (lower >= 0)
                    CUDA_CHECK(cudaMemcpyAsync(front, hRecvLo, haloBytes,
                                               cudaMemcpyHostToDevice, sComm));
                if (upper >= 0)
                    CUDA_CHECK(cudaMemcpyAsync(front + (lz + GHOST) * plane, hRecvHi, haloBytes,
                                               cudaMemcpyHostToDevice, sComm));
                CUDA_CHECK(cudaEventRecord(evHalo, sComm));
                CUDA_CHECK(cudaStreamWaitEvent(sCompute, evHalo, 0));
            }

            // Stage B: planes whose stencil reads ghost planes. Mu is needed on
            // [-1, lz + 1) when a neighbor exists, and on the edge planes otherwise.
            const int muLo = q.lowerEdge ? 0 : -1;
            const int muHi = q.upperEdge ? lz : lz + 1;
            const int muHiStart = std::max(1, lz - 1);
            launchChemicalPotential(sCompute, q, front, mu, muLo, 1 - muLo);
            launchChemicalPotential(sCompute, q, front, mu, muHiStart, muHi - muHiStart);
            launchCahnHilliardUpdate(sCompute, q, back, front, mu, 0, std::min(2, lz));
            const int updHiStart = std::max(0, lz - 2);
            launchCahnHilliardUpdate(sCompute, q, back, front, mu, updHiStart, lz - updHiStart);

            CUDA_CHECK(cudaEventRecord(evStage, sCompute));
            std::swap(front, back);
        }
        CUDA_CHECK(cudaStreamSynchronize(sCompute));
        CUDA_CHECK(cudaStreamSynchronize(sComm));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    duration = std::chrono::milliseconds(maxDuration);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field on rank 0 only when it is needed for output or validation
    int status = 0;
    if (printResults || validate) {
        std::vector<double> slabHost(static_cast<size_t>(lz) * plane);
        if (lz > 0)
            CUDA_CHECK(cudaMemcpy(slabHost.data(), front + GHOST * plane,
                                  slabHost.size() * sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<double> full;
        std::vector<int> counts;
        std::vector<int> displs;
        if (root) {
            full.resize(gridSize);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const Slab s = slabFor(r, activeRanks, nzEff);
                counts[r] = s.lz;
                displs[r] = s.z0;
            }
        }

        // Each z-plane is contiguous, so a slab is one contiguous run of planes
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(slabHost.data(), lz, planeType, full.data(), counts.data(), displs.data(),
                    planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
        if (validate) {
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (lz > 0) {
        CUDA_CHECK(cudaEventDestroy(evHalo));
        CUDA_CHECK(cudaEventDestroy(evStage));
        CUDA_CHECK(cudaStreamDestroy(sComm));
        CUDA_CHECK(cudaStreamDestroy(sCompute));
        CUDA_CHECK(cudaFreeHost(hRecvHi));
        CUDA_CHECK(cudaFreeHost(hRecvLo));
        CUDA_CHECK(cudaFreeHost(hSendHi));
        CUDA_CHECK(cudaFreeHost(hSendLo));
        CUDA_CHECK(cudaFree(mu));
        CUDA_CHECK(cudaFree(bufB));
        CUDA_CHECK(cudaFree(bufA));
    }

    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int status = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return status;
}
