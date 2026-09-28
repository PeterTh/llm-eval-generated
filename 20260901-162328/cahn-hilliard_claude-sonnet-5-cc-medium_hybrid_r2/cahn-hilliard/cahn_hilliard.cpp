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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err__));                                     \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// 3D index calculation (global layout, z-major)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Device kernels operate on a local slab with a 1-plane halo on each side in
// the z-dimension (array layout: nx * ny * (nzl + 2), z index 0 and nzl+1 are
// the ghost/halo planes). x and y are never decomposed, so their boundaries
// are always clamped exactly as in the original single-process code. The z
// boundary is clamped only when this rank owns the corresponding global edge
// (firstRank / lastRank); otherwise the neighbor's data (exchanged via MPI)
// is used, which reproduces the exact same stencil as an undecomposed grid.
// ---------------------------------------------------------------------------

__device__ __forceinline__ double laplacianDevice(const double* __restrict__ f,
                                                   const size_t nx, const size_t ny, const size_t nzl,
                                                   const double dx, const double dy, const double dz,
                                                   const size_t x, const size_t y, const size_t z,
                                                   const bool firstRank, const bool lastRank) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t rz = z + 1; // index into halo-padded array
    const size_t rzp = (z < nzl - 1) ? rz + 1 : (lastRank ? rz : rz + 1);
    const size_t rzn = (z > 0) ? rz - 1 : (firstRank ? rz : rz - 1);

    const double c0 = f[idx3(x, y, rz, nx, ny)];

    const double cxx = (f[idx3(xp, y, rz, nx, ny)] + f[idx3(xn, y, rz, nx, ny)] - 2.0 * c0) / (dx * dx);
    const double cyy = (f[idx3(x, yp, rz, nx, ny)] + f[idx3(x, yn, rz, nx, ny)] - 2.0 * c0) / (dy * dy);
    const double czz = (f[idx3(x, y, rzp, nx, ny)] + f[idx3(x, y, rzn, nx, ny)] - 2.0 * c0) / (dz * dz);

    return cxx + cyy + czz;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nzl,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                        const bool firstRank, const bool lastRank) {
    const size_t total = nx * ny * nzl;
    const size_t lin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lin >= total) return;

    const size_t x = lin % nx;
    const size_t y = (lin / nx) % ny;
    const size_t z = lin / (nx * ny);

    const double cv = c[idx3(x, y, z + 1, nx, ny)];
    const double lap = laplacianDevice(c, nx, ny, nzl, dx, dy, dz, x, y, z, firstRank, lastRank);

    mu[idx3(x, y, z + 1, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                   + 3.0 * cv + cv * cv * cv
                                   - gamma * lap;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             const size_t nx, const size_t ny, const size_t nzl,
                             const double D, const double dt, const double dx, const double dy, const double dz,
                             const bool firstRank, const bool lastRank) {
    const size_t total = nx * ny * nzl;
    const size_t lin = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lin >= total) return;

    const size_t x = lin % nx;
    const size_t y = (lin / nx) % ny;
    const size_t z = lin / (nx * ny);

    const double lap = laplacianDevice(mu, nx, ny, nzl, dx, dy, dz, x, y, z, firstRank, lastRank);
    const size_t ridx = idx3(x, y, z + 1, nx, ny);
    cnew[ridx] = cold[ridx] + dt * D * lap;
}

// Initialize concentration field (host, OpenMP parallel, exact same pseudo-random
// sequence as the original single-process implementation, indexed by global id).
void initializeConcentrationLocal(double* c_local, const size_t nx, const size_t ny,
                                  const size_t nzl, const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = z + z_offset;
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                // z=0 halo plane is at local index z+1
                c_local[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validate the local slab (NaN/Inf check + range), reduced over OpenMP threads.
void validateResultLocal(const double* c_local, const size_t nx, const size_t ny, const size_t nzl,
                         bool& hasNanInf, double& minVal, double& maxVal) {
    bool localNanInf = false;
    double localMin = c_local[idx3(0, 0, 1, nx, ny)];
    double localMax = localMin;

    #pragma omp parallel for collapse(3) schedule(static) reduction(||:localNanInf) reduction(min:localMin) reduction(max:localMax)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const double val = c_local[idx3(x, y, z + 1, nx, ny)];
                if (std::isnan(val) || std::isinf(val)) {
                    localNanInf = true;
                }
                localMin = std::min(localMin, val);
                localMax = std::max(localMax, val);
            }
        }
    }

    hasNanInf = localNanInf;
    minVal = localMin;
    maxVal = localMax;
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

// Exchange one z-plane of halo data with the neighboring ranks (staged through
// pinned host buffers, since CUDA-aware MPI is not guaranteed to be available).
void haloExchange(double* d_field, double* h_send_top, double* h_send_bot,
                  double* h_recv_top, double* h_recv_bot,
                  const size_t nx, const size_t ny, const size_t nzl,
                  const int rank, const int size, MPI_Comm comm) {
    const size_t planeBytes = nx * ny * sizeof(double);
    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Stage local boundary planes: first interior plane (z local index 1) and
    // last interior plane (z local index nzl) into pinned host buffers.
    if (prev != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(h_send_top, d_field + 1 * nx * ny, planeBytes, cudaMemcpyDeviceToHost));
    }
    if (next != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(h_send_bot, d_field + nzl * nx * ny, planeBytes, cudaMemcpyDeviceToHost));
    }

    MPI_Request reqs[4];
    int nreq = 0;
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(h_recv_top, static_cast<int>(nx * ny), MPI_DOUBLE, prev, 0, comm, &reqs[nreq++]);
        MPI_Isend(h_send_top, static_cast<int>(nx * ny), MPI_DOUBLE, prev, 1, comm, &reqs[nreq++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(h_recv_bot, static_cast<int>(nx * ny), MPI_DOUBLE, next, 1, comm, &reqs[nreq++]);
        MPI_Isend(h_send_bot, static_cast<int>(nx * ny), MPI_DOUBLE, next, 0, comm, &reqs[nreq++]);
    }
    if (nreq > 0) {
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
    }

    if (prev != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_field + 0 * nx * ny, h_recv_top, planeBytes, cudaMemcpyHostToDevice));
    }
    if (next != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_field + (nzl + 1) * nx * ny, h_recv_bot, planeBytes, cudaMemcpyHostToDevice));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool argError = false;

    // Parse command line arguments (identical on every rank)
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
            argError = true;
            break;
        }
    }

    if (argError) {
        MPI_Finalize();
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (static_cast<size_t>(size) > nz) {
        if (rank == 0) {
            printf("Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    // Bind this rank to a GPU (round-robin across the GPUs visible on its node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) printf("Error: no CUDA-capable devices found\n");
        MPI_Finalize();
        return 1;
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               size, omp_get_max_threads(), deviceCount);
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

    // 1D domain decomposition along Z
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_offset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const bool firstRank = (rank == 0);
    const bool lastRank = (rank == size - 1);

    const size_t planeCells = nx * ny;
    const size_t localCellsPadded = planeCells * (nzl + 2);
    const size_t localCells = planeCells * nzl;

    // Host staging (pinned) buffers for halo exchange and I/O
    double* h_local = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_local, localCellsPadded * sizeof(double)));
    std::memset(h_local, 0, localCellsPadded * sizeof(double));

    double *h_send_top, *h_send_bot, *h_recv_top, *h_recv_bot;
    CUDA_CHECK(cudaMallocHost(&h_send_top, planeCells * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_bot, planeCells * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, planeCells * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bot, planeCells * sizeof(double)));

    // Initialize concentration field on host (OpenMP), then upload to device
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationLocal(h_local, nx, ny, nzl, z_offset, nz);

    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localCellsPadded * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold, h_local, localCellsPadded * sizeof(double), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = static_cast<int>((localCells + blockSize - 1) / blockSize);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos before computing the chemical potential
        haloExchange(d_cold, h_send_top, h_send_bot, h_recv_top, h_recv_bot, nx, ny, nzl, rank, size, MPI_COMM_WORLD);

        chemicalPotentialKernel<<<gridSize, blockSize>>>(d_cold, d_mu, nx, ny, nzl, dx, dy, dz,
                                                          gamma, e_AA, e_BB, e_AB, firstRank, lastRank);
        CUDA_CHECK(cudaGetLastError());

        // Exchange chemical-potential halos before the update step
        haloExchange(d_mu, h_send_top, h_send_bot, h_recv_top, h_recv_bot, nx, ny, nzl, rank, size, MPI_COMM_WORLD);

        updateKernel<<<gridSize, blockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nzl, D, dt, dx, dy, dz,
                                              firstRank, lastRank);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long ms = duration.count();
    long long msMax = 0;
    MPI_Reduce(&ms, &msMax, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", msMax);
        double cellUpdates = static_cast<double>(nx) * ny * nz * iterations;
        double mcups = cellUpdates / (msMax / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy final local result back to host
    CUDA_CHECK(cudaMemcpy(h_local, d_cold, localCellsPadded * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation (needs the full, globally-ordered array)
    if (printResults) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rnzl = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(rnzl * planeCells);
        }
        displs[0] = 0;
        for (int r = 1; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];

        std::vector<double> global;
        if (rank == 0) global.resize(nx * ny * nz);

        MPI_Gatherv(h_local + planeCells, static_cast<int>(localCells), MPI_DOUBLE,
                   rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        bool hasNanInf = false;
        double minVal = 0.0, maxVal = 0.0;
        validateResultLocal(h_local, nx, ny, nzl, hasNanInf, minVal, maxVal);

        int localBad = hasNanInf ? 1 : 0;
        int globalBad = 0;
        double globalMin = 0.0, globalMax = 0.0;
        MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        bool valid = true;
        if (rank == 0) {
            if (globalBad) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = false;
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        exitCode = valid ? 0 : 1;
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_local));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_send_bot));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bot));

    MPI_Finalize();
    return exitCode;
}
