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

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA implementation.
//
// Domain decomposition: the grid is split along the z dimension across MPI
// ranks; every rank drives one GPU (ranks are mapped round-robin onto the GPUs
// of their node).  Each rank keeps its slab plus one halo plane on either side.
// Halo exchange overlaps with the computation of the slab interior, OpenMP is
// used for the host-side work (initialization, validation, halo packing).
//
// All floating point expressions are kept bit-identical to the serial version
// (no FMA contraction, no fast math), so results match the original exactly.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                             \
    do {                                                                                             \
        const cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                            \
        }                                                                                            \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local (halo-padded) index: plane 0 is the lower halo, planes 1..nzl the owned
// slab, plane nzl+1 the upper halo.
__host__ __device__ inline size_t lidx(const int x, const int y, const int zl, const int nx, const int ny) {
    return static_cast<size_t>(zl) * (static_cast<size_t>(nx) * ny) + static_cast<size_t>(y) * nx + x;
}

// Chemical potential kernel: mu = f(c) - gamma * lap(c) over local planes
// [zbeg, zbeg+zcount).  The z direction is rolled through registers, x/y
// neighbours are served from the caches.
// Division by a unit grid spacing is an exact identity in IEEE arithmetic, so
// the (slow, FP64) divisions are skipped when dx=dy=dz=1 -- results are
// bit-identical either way.
template <bool UNIT>
__device__ inline double divide(const double v, const double d) {
    return UNIT ? v : v / d;
}

template <bool UNIT>
__global__ void kChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                   const int nx, const int ny, const int zbeg, const int zcount,
                                   const int gz0, const int nzg,
                                   const double dx2, const double dy2, const double dz2,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    // Roll the z direction through registers: only the leading plane has to be
    // loaded in each iteration.
    double cv = c[lidx(x, y, zbeg, nx, ny)];
    double cprev = (gz0 + zbeg - 1 > 0) ? c[lidx(x, y, zbeg - 1, nx, ny)] : cv;

    for (int zl = zbeg; zl < zbeg + zcount; ++zl) {
        const int gz = gz0 + zl - 1;
        const double cnext = (gz < nzg - 1) ? c[lidx(x, y, zl + 1, nx, ny)] : cv;

        const double cxx = divide<UNIT>(c[lidx(xp, y, zl, nx, ny)] + c[lidx(xn, y, zl, nx, ny)] - 2.0 * cv, dx2);
        const double cyy = divide<UNIT>(c[lidx(x, yp, zl, nx, ny)] + c[lidx(x, yn, zl, nx, ny)] - 2.0 * cv, dy2);
        const double czz = divide<UNIT>(cnext + cprev - 2.0 * cv, dz2);

        mu[lidx(x, y, zl, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                     + 3.0 * cv + cv * cv * cv
                                     - gamma * (cxx + cyy + czz);

        cprev = cv;
        cv = cnext;
    }
}

// Concentration update: cnew = cold + dt * D * lap(mu)
template <bool UNIT>
__global__ void kUpdate(const double* __restrict__ mu, const double* __restrict__ cold, double* __restrict__ cnew,
                        const int nx, const int ny, const int zbeg, const int zcount,
                        const int gz0, const int nzg,
                        const double dx2, const double dy2, const double dz2,
                        const double D, const double dt) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    double mv = mu[lidx(x, y, zbeg, nx, ny)];
    double mprev = (gz0 + zbeg - 1 > 0) ? mu[lidx(x, y, zbeg - 1, nx, ny)] : mv;

    for (int zl = zbeg; zl < zbeg + zcount; ++zl) {
        const int gz = gz0 + zl - 1;
        const size_t idx = lidx(x, y, zl, nx, ny);
        const double mnext = (gz < nzg - 1) ? mu[lidx(x, y, zl + 1, nx, ny)] : mv;

        const double cxx = divide<UNIT>(mu[lidx(xp, y, zl, nx, ny)] + mu[lidx(xn, y, zl, nx, ny)] - 2.0 * mv, dx2);
        const double cyy = divide<UNIT>(mu[lidx(x, yp, zl, nx, ny)] + mu[lidx(x, yn, zl, nx, ny)] - 2.0 * mv, dy2);
        const double czz = divide<UNIT>(mnext + mprev - 2.0 * mv, dz2);

        cnew[idx] = cold[idx] + dt * D * (cxx + cyy + czz);

        mprev = mv;
        mv = mnext;
    }
}

// Initialize concentration field (local slab, global semantics preserved)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t gz0, const size_t nzl) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for schedule(static)
    for (size_t zl = 0; zl < nzl; ++zl) {
        const size_t z = gz0 + zl;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx3(x, y, zl, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
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

int main(int argc, char** argv) {
    int mpiProvided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiProvided);

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

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", worldSize, omp_get_max_threads());
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
    const size_t nxy = nx * ny;

    // ---- z decomposition over the ranks -----------------------------------
    const int usedRanks = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), nz));
    const bool active = worldRank < usedRanks;

    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);

    std::vector<int> counts(worldSize, 0), displs(worldSize, 0);
    {
        size_t off = 0;
        for (int r = 0; r < usedRanks; ++r) {
            const size_t planes = nz / usedRanks + (static_cast<size_t>(r) < nz % usedRanks ? 1 : 0);
            counts[r] = static_cast<int>(planes * nxy);
            displs[r] = static_cast<int>(off * nxy);
            off += planes;
        }
    }

    std::vector<double> cold;  // gathered result on rank 0 / local slab elsewhere

    if (active) {
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        const size_t nzl = nz / usedRanks + (static_cast<size_t>(rank) < nz % usedRanks ? 1 : 0);
        const size_t gz0 = static_cast<size_t>(displs[rank]) / nxy;

        // Pick a GPU: round-robin over the devices visible on this node.
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        CUDA_CHECK(cudaSetDevice(localRank % devCount));

        const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        const int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        // Initialize concentration field (host, OpenMP)
        if (rank == 0) printf("Initializing concentration field...\n");
        std::vector<double> hostSlab(nzl * nxy);
        initializeConcentration(hostSlab, nx, ny, nz, gz0, nzl);

        // Device buffers with one halo plane on each side
        const size_t padded = (nzl + 2) * nxy;
        double *d_c = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
        CUDA_CHECK(cudaMalloc(&d_c, padded * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, padded * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, padded * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_c + nxy, hostSlab.data(), nzl * nxy * sizeof(double), cudaMemcpyHostToDevice));

        // Pinned staging buffers for the halo exchange
        double *sendLo = nullptr, *sendHi = nullptr, *recvLo = nullptr, *recvHi = nullptr;
        CUDA_CHECK(cudaMallocHost(&sendLo, nxy * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&sendHi, nxy * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&recvLo, nxy * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&recvHi, nxy * sizeof(double)));

        cudaStream_t sCompute, sHalo;
        CUDA_CHECK(cudaStreamCreate(&sCompute));
        CUDA_CHECK(cudaStreamCreate(&sHalo));
        cudaEvent_t evCompute, evHalo;
        CUDA_CHECK(cudaEventCreateWithFlags(&evCompute, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

        const dim3 block(32, 8);
        const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                        static_cast<unsigned>((ny + block.y - 1) / block.y));

        const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
        const int inx = static_cast<int>(nx), iny = static_cast<int>(ny);
        const int inzl = static_cast<int>(nzl), inzg = static_cast<int>(nz), igz0 = static_cast<int>(gz0);
        const int interiorCount = (inzl > 2) ? inzl - 2 : 0;  // planes 2 .. nzl-1
        const bool unitSpacing = (dx2 == 1.0 && dy2 == 1.0 && dz2 == 1.0);

        auto launchMu = [&](cudaStream_t s, int zbeg, int zcount) {
            if (zcount <= 0) return;
            if (unitSpacing)
                kChemicalPotential<true><<<grid, block, 0, s>>>(d_c, d_mu, inx, iny, zbeg, zcount, igz0, inzg,
                                                                dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
            else
                kChemicalPotential<false><<<grid, block, 0, s>>>(d_c, d_mu, inx, iny, zbeg, zcount, igz0, inzg,
                                                                 dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
        };
        auto launchUpdate = [&](cudaStream_t s, int zbeg, int zcount) {
            if (zcount <= 0) return;
            if (unitSpacing)
                kUpdate<true><<<grid, block, 0, s>>>(d_mu, d_c, d_cnew, inx, iny, zbeg, zcount, igz0, inzg,
                                                     dx2, dy2, dz2, D, dt);
            else
                kUpdate<false><<<grid, block, 0, s>>>(d_mu, d_c, d_cnew, inx, iny, zbeg, zcount, igz0, inzg,
                                                      dx2, dy2, dz2, D, dt);
        };

        // Exchanges the boundary planes of `d` (local planes 1 and nzl) into the
        // halo planes of the neighbouring ranks. Runs on the halo stream.
        auto exchangeHalos = [&](double* d) {
            if (prev == MPI_PROC_NULL && next == MPI_PROC_NULL) return;
            if (prev != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendLo, d + nxy, nxy * sizeof(double), cudaMemcpyDeviceToHost, sHalo));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendHi, d + static_cast<size_t>(nzl) * nxy, nxy * sizeof(double),
                                           cudaMemcpyDeviceToHost, sHalo));
            CUDA_CHECK(cudaStreamSynchronize(sHalo));

            MPI_Request reqs[4];
            int nreq = 0;
            if (prev != MPI_PROC_NULL) {
                MPI_Irecv(recvLo, static_cast<int>(nxy), MPI_DOUBLE, prev, 0, comm, &reqs[nreq++]);
                MPI_Isend(sendLo, static_cast<int>(nxy), MPI_DOUBLE, prev, 1, comm, &reqs[nreq++]);
            }
            if (next != MPI_PROC_NULL) {
                MPI_Irecv(recvHi, static_cast<int>(nxy), MPI_DOUBLE, next, 1, comm, &reqs[nreq++]);
                MPI_Isend(sendHi, static_cast<int>(nxy), MPI_DOUBLE, next, 0, comm, &reqs[nreq++]);
            }
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

            if (prev != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d, recvLo, nxy * sizeof(double), cudaMemcpyHostToDevice, sHalo));
            if (next != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(d + static_cast<size_t>(nzl + 1) * nxy, recvHi, nxy * sizeof(double),
                                           cudaMemcpyHostToDevice, sHalo));
        };

        // Run simulation
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            // --- chemical potential -------------------------------------------
            // interior planes are computed while the c halos travel
            if (interiorCount > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(sCompute, evHalo, 0));
                launchMu(sCompute, 2, interiorCount);
            }
            exchangeHalos(d_c);
            launchMu(sHalo, 1, 1);
            if (inzl > 1) launchMu(sHalo, inzl, 1);
            CUDA_CHECK(cudaEventRecord(evHalo, sHalo));
            CUDA_CHECK(cudaEventRecord(evCompute, sCompute));

            // --- concentration update -----------------------------------------
            if (interiorCount > 0) {
                CUDA_CHECK(cudaStreamWaitEvent(sCompute, evHalo, 0));
                launchUpdate(sCompute, 2, interiorCount);
            }
            exchangeHalos(d_mu);
            CUDA_CHECK(cudaStreamWaitEvent(sHalo, evCompute, 0));
            launchUpdate(sHalo, 1, 1);
            if (inzl > 1) launchUpdate(sHalo, inzl, 1);
            CUDA_CHECK(cudaEventRecord(evHalo, sHalo));
            CUDA_CHECK(cudaEventRecord(evCompute, sCompute));
            CUDA_CHECK(cudaStreamWaitEvent(sHalo, evCompute, 0));

            // Swap buffers
            std::swap(d_c, d_cnew);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        long localDuration = duration.count();
        long maxDuration = 0;
        MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, comm);

        if (rank == 0) {
            printf("Computation time: %ld ms\n", maxDuration);

            // Calculate performance
            double cellUpdates = (double)gridSize * iterations;
            double mcups = cellUpdates / (maxDuration / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Collect the final field on rank 0 for output / validation
        CUDA_CHECK(cudaMemcpy(hostSlab.data(), d_c + nxy, nzl * nxy * sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults || validate) {
            if (rank == 0) cold.resize(gridSize);
            MPI_Gatherv(hostSlab.data(), static_cast<int>(nzl * nxy), MPI_DOUBLE,
                        rank == 0 ? cold.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        }

        CUDA_CHECK(cudaEventDestroy(evCompute));
        CUDA_CHECK(cudaEventDestroy(evHalo));
        CUDA_CHECK(cudaStreamDestroy(sCompute));
        CUDA_CHECK(cudaStreamDestroy(sHalo));
        CUDA_CHECK(cudaFreeHost(sendLo));
        CUDA_CHECK(cudaFreeHost(sendHi));
        CUDA_CHECK(cudaFreeHost(recvLo));
        CUDA_CHECK(cudaFreeHost(recvHi));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        MPI_Comm_free(&comm);
    }

    int exitCode = 0;
    if (worldRank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(cold, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(cold, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
