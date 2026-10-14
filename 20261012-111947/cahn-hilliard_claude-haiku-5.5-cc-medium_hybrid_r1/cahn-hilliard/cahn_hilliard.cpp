#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Physical and grid constants shared with the device kernels
struct Physics {
    size_t nx, ny;
    double dxx, dyy, dzz;  // dx*dx, dy*dy, dz*dz
    double dt, D, gamma, e_AA, e_BB, e_AB;
};

// Laplacian with clamped boundary conditions on a padded slab. Plane p is a padded
// z-index: owned planes are [1, nzLocal] and the ghost planes 0 and nzLocal+1 always
// hold the clamped values, so z neighbours need no clamping here.
__device__ __forceinline__ double laplacian(const double* __restrict__ f, const Physics& P,
                                            const size_t x, const size_t y, const size_t p) {
    const size_t plane = P.nx * P.ny;
    const size_t row = y * P.nx;
    const size_t xp = (x < P.nx - 1) ? x + 1 : x;
    const size_t yp = (y < P.ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double fc = f[p * plane + row + x];
    const double cxx = (f[p * plane + row + xp] + f[p * plane + row + xn] - 2.0 * fc) / P.dxx;
    const double cyy = (f[p * plane + yp * P.nx + x] + f[p * plane + yn * P.nx + x] - 2.0 * fc) / P.dyy;
    const double czz = (f[(p + 1) * plane + row + x] + f[(p - 1) * plane + row + x] - 2.0 * fc) / P.dzz;

    return cxx + cyy + czz;
}

// Compute chemical potential on padded planes [pBegin, pEnd)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const Physics P, const size_t pBegin, const size_t pEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;

    for (size_t p = pBegin + blockIdx.z; p < pEnd; p += gridDim.z) {
        const size_t idx = p * (P.nx * P.ny) + y * P.nx + x;
        const double cv = c[idx];

        // Fused multiply-adds mirror the contraction applied to the host reference build,
        // so that results are bit-identical to it
        const double f = fma(cv + 1.0, P.e_AA, (cv - 1.0) * P.e_BB);
        const double g = fma(-(2.0 * cv), P.e_AB, f);
        mu[idx] = fma(-P.gamma, laplacian(c, P, x, y, p),
                      fma(cv * cv, cv, fma(4.5, g, 3.0 * cv)));
    }
}

// Cahn-Hilliard update step on padded planes [pBegin, pEnd)
__global__ void cahnHilliardKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                                   const double* __restrict__ mu, const Physics P,
                                   const size_t pBegin, const size_t pEnd) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;

    for (size_t p = pBegin + blockIdx.z; p < pEnd; p += gridDim.z) {
        const size_t idx = p * (P.nx * P.ny) + y * P.nx + x;
        cnew[idx] = fma(P.dt * P.D, laplacian(mu, P, x, y, p), cold[idx]);
    }
}

// Launch helpers: planes are processed by a grid-stride loop over blockIdx.z
void launchChemicalPotential(cudaStream_t s, const double* c, double* mu, const Physics& P,
                             const size_t pBegin, const size_t pEnd) {
    if (pBegin >= pEnd) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((P.nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((P.ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(std::min<size_t>(pEnd - pBegin, 65535)));
    chemicalPotentialKernel<<<grid, block, 0, s>>>(c, mu, P, pBegin, pEnd);
    CUDA_CHECK(cudaGetLastError());
}

void launchCahnHilliard(cudaStream_t s, const double* cold, double* cnew, const double* mu, const Physics& P,
                        const size_t pBegin, const size_t pEnd) {
    if (pBegin >= pEnd) return;
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((P.nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((P.ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(std::min<size_t>(pEnd - pBegin, 65535)));
    cahnHilliardKernel<<<grid, block, 0, s>>>(cold, cnew, mu, P, pBegin, pEnd);
    CUDA_CHECK(cudaGetLastError());
}

// Z-slab decomposition of this rank: nzLocal planes starting at global plane z0
struct Slab {
    size_t planeSize;  // nx * ny doubles per z-plane
    size_t nzLocal;
    int below;         // neighbour rank holding lower z-planes, or MPI_PROC_NULL
    int above;         // neighbour rank holding higher z-planes, or MPI_PROC_NULL
};

// Pinned host staging buffers for halo exchange (one plane each)
struct Halo {
    double* sendLo;
    double* sendHi;
    double* recvLo;
    double* recvHi;
};

// Refresh the ghost planes of padded field d_f. Owned planes must be finished on the
// compute stream (readyEv). Afterwards ghostEv marks the ghost planes as valid on the copy stream.
void exchangeGhosts(double* d_f, const Slab& S, const Halo& H, cudaStream_t copy,
                    cudaEvent_t readyEv, cudaEvent_t ghostEv) {
    const size_t plane = S.planeSize;
    const size_t bytes = plane * sizeof(double);

    CUDA_CHECK(cudaStreamWaitEvent(copy, readyEv, 0));
    CUDA_CHECK(cudaMemcpyAsync(H.sendLo, d_f + plane, bytes, cudaMemcpyDeviceToHost, copy));
    CUDA_CHECK(cudaMemcpyAsync(H.sendHi, d_f + S.nzLocal * plane, bytes, cudaMemcpyDeviceToHost, copy));
    CUDA_CHECK(cudaStreamSynchronize(copy));

    // MPI_PROC_NULL neighbours leave the receive buffers untouched at the physical boundary
    MPI_Sendrecv(H.sendHi, static_cast<int>(plane), MPI_DOUBLE, S.above, 0,
                 H.recvLo, static_cast<int>(plane), MPI_DOUBLE, S.below, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(H.sendLo, static_cast<int>(plane), MPI_DOUBLE, S.below, 1,
                 H.recvHi, static_cast<int>(plane), MPI_DOUBLE, S.above, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Physical boundary: the clamped neighbour is the plane itself
    if (S.below != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_f, H.recvLo, bytes, cudaMemcpyHostToDevice, copy));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_f, d_f + plane, bytes, cudaMemcpyDeviceToDevice, copy));
    }
    if (S.above != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_f + (S.nzLocal + 1) * plane, H.recvHi, bytes, cudaMemcpyHostToDevice, copy));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_f + (S.nzLocal + 1) * plane, d_f + S.nzLocal * plane, bytes,
                                   cudaMemcpyDeviceToDevice, copy));
    }
    CUDA_CHECK(cudaEventRecord(ghostEv, copy));
}

// Initialize the owned planes [z0, z0 + nzLocal) of the concentration field
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzLocal) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;

    #pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < nzLocal; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z0 + lz) * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    int status = -1;  // -1: keep running, otherwise exit with this status

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
            status = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }

    if (status < 0) {
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
        if (nx == 0 || ny == 0 || nz == 0) {
            if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
            status = 1;
        }
    }

    if (status >= 0) {
        MPI_Finalize();
        return status;
    }

    // One MPI rank per GPU on each node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (rank == 0) {
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

    const Physics P{nx, ny, dx * dx, dy * dy, dz * dz, dt, D, gamma, e_AA, e_BB, e_AB};

    // Slab decomposition along z. With fewer planes than ranks, the trailing ranks own
    // no planes and only take part in the collective operations.
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    auto planesOf = [base, rem](size_t q) { return base + (q < rem ? 1 : 0); };
    const size_t r = static_cast<size_t>(rank);
    Slab S{};
    S.planeSize = nx * ny;
    S.nzLocal = planesOf(r);
    const size_t z0 = r * base + std::min(r, rem);
    S.below = (rank > 0 && planesOf(r - 1) > 0) ? rank - 1 : MPI_PROC_NULL;
    S.above = (rank + 1 < nranks && planesOf(r + 1) > 0) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeSize = S.planeSize;
    const size_t paddedSize = planeSize * (S.nzLocal + 2);
    const size_t gridSize = nx * ny * nz;

    // Device storage: two padded concentration buffers and one chemical potential buffer
    double* d_c[2];
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c[0], paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c[1], paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_c[0], 0, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_c[1], 0, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, paddedSize * sizeof(double)));

    cudaStream_t compute;
    cudaStream_t copy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));

    cudaEvent_t evCReady, evCGhost, evMuReady, evMuGhost;
    CUDA_CHECK(cudaEventCreateWithFlags(&evCReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evCGhost, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMuReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMuGhost, cudaEventDisableTiming));

    Halo H{};
    CUDA_CHECK(cudaMallocHost(&H.sendLo, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&H.sendHi, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&H.recvLo, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&H.recvHi, planeSize * sizeof(double)));

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    if (S.nzLocal > 0) {
        std::vector<double> slab(planeSize * S.nzLocal);
        initializeConcentration(slab.data(), nx, ny, nz, z0, S.nzLocal);
        CUDA_CHECK(cudaMemcpy(d_c[0] + planeSize, slab.data(), slab.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(evCReady, compute));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Interior planes (owned planes not adjacent to a ghost plane) need no halo data
    const size_t interiorBegin = 2;
    const size_t interiorEnd = (S.nzLocal > 2) ? S.nzLocal : interiorBegin;
    const size_t lastPlane = S.nzLocal;

    int cs = 0;
    const int steps = (S.nzLocal > 0) ? iterations : 0;
    for (int t = 0; t < steps; ++t) {
        double* cur = d_c[cs];
        double* nxt = d_c[1 - cs];

        // Compute chemical potential: interior planes overlap the halo exchange of c
        launchChemicalPotential(compute, cur, d_mu, P, interiorBegin, interiorEnd);
        exchangeGhosts(cur, S, H, copy, evCReady, evCGhost);
        CUDA_CHECK(cudaStreamWaitEvent(compute, evCGhost, 0));
        launchChemicalPotential(compute, cur, d_mu, P, 1, 2);
        if (lastPlane > 1) launchChemicalPotential(compute, cur, d_mu, P, lastPlane, lastPlane + 1);
        CUDA_CHECK(cudaEventRecord(evMuReady, compute));

        // Update concentration: interior planes overlap the halo exchange of mu
        launchCahnHilliard(compute, cur, nxt, d_mu, P, interiorBegin, interiorEnd);
        exchangeGhosts(d_mu, S, H, copy, evMuReady, evMuGhost);
        CUDA_CHECK(cudaStreamWaitEvent(compute, evMuGhost, 0));
        launchCahnHilliard(compute, cur, nxt, d_mu, P, 1, 2);
        if (lastPlane > 1) launchCahnHilliard(compute, cur, nxt, d_mu, P, lastPlane, lastPlane + 1);
        CUDA_CHECK(cudaEventRecord(evCReady, compute));

        // Swap buffers
        cs = 1 - cs;
    }

    CUDA_CHECK(cudaStreamSynchronize(compute));
    CUDA_CHECK(cudaStreamSynchronize(copy));
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long durationMs = duration.count();
    long long slowestMs = 0;
    MPI_Reduce(&durationMs, &slowestMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", slowestMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (slowestMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the owned planes in global z-order (contiguous in memory) for output and validation
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<double> local(planeSize * S.nzLocal);
        if (S.nzLocal > 0) {
            CUDA_CHECK(cudaMemcpy(local.data(), cs == 0 ? d_c[0] + planeSize : d_c[1] + planeSize,
                                  local.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }

        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(planeSize), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);

        std::vector<int> counts(nranks), displs(nranks);
        for (int q = 0; q < nranks; ++q) {
            const size_t uq = static_cast<size_t>(q);
            counts[q] = static_cast<int>(planesOf(uq));
            displs[q] = static_cast<int>(uq * base + std::min(uq, rem));
        }

        std::vector<double> full;
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(local.data(), static_cast<int>(S.nzLocal), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (rank == 0) {
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
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaEventDestroy(evCReady));
    CUDA_CHECK(cudaEventDestroy(evCGhost));
    CUDA_CHECK(cudaEventDestroy(evMuReady));
    CUDA_CHECK(cudaEventDestroy(evMuGhost));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(copy));
    CUDA_CHECK(cudaFreeHost(H.sendLo));
    CUDA_CHECK(cudaFreeHost(H.sendHi));
    CUDA_CHECK(cudaFreeHost(H.recvLo));
    CUDA_CHECK(cudaFreeHost(H.recvHi));
    CUDA_CHECK(cudaFree(d_c[0]));
    CUDA_CHECK(cudaFree(d_c[1]));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exitCode;
}
