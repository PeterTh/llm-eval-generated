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

// -----------------------------------------------------------------------------
// CUDA error checking
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err__ = (call);                                                            \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                    cudaGetErrorString(err__));                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// 3D index calculation for the local, z-padded grid
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t lz,
                             const size_t nx, const size_t ny) noexcept {
    return lz * (nx * ny) + y * nx + x;
}

// -----------------------------------------------------------------------------
// CUDA kernels.  The z dimension of each rank is padded with one ghost layer on
// each side, so the z stencil never needs a branch.  The global z boundaries are
// handled by replicating the boundary plane into the ghost layer, which exactly
// reproduces the clamped (zero-flux) boundary condition of the original code.
// -----------------------------------------------------------------------------

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const int nx, const int ny, const int nzLocal,
                                               const size_t plane,
                                               const double invDx2, const double invDy2,
                                               const double invDz2, const double gamma,
                                               const double e_AA, const double e_BB,
                                               const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lz = blockIdx.z * blockDim.z + threadIdx.z + 1; // skip lower ghost

    if (x >= nx || y >= ny || lz > nzLocal) return;

    const int xm = (x > 0) ? x - 1 : 0;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int ym = (y > 0) ? y - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;

    const size_t row = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * nx;
    const size_t rowYp = static_cast<size_t>(lz) * plane + static_cast<size_t>(yp) * nx;
    const size_t rowYm = static_cast<size_t>(lz) * plane + static_cast<size_t>(ym) * nx;
    const size_t rowZp = static_cast<size_t>(lz + 1) * plane + static_cast<size_t>(y) * nx;
    const size_t rowZm = static_cast<size_t>(lz - 1) * plane + static_cast<size_t>(y) * nx;

    const double cv = c[row + x];
    const double cxx = (c[row + xp] + c[row + xm] - 2.0 * cv) * invDx2;
    const double cyy = (c[rowYp + x] + c[rowYm + x] - 2.0 * cv) * invDy2;
    const double czz = (c[rowZp + x] + c[rowZm + x] - 2.0 * cv) * invDz2;
    const double lap = cxx + cyy + czz;

    mu[row + x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const int nzLocal,
                                         const size_t plane,
                                         const double invDx2, const double invDy2,
                                         const double invDz2, const double dtD) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lz = blockIdx.z * blockDim.z + threadIdx.z + 1; // skip lower ghost

    if (x >= nx || y >= ny || lz > nzLocal) return;

    const int xm = (x > 0) ? x - 1 : 0;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int ym = (y > 0) ? y - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;

    const size_t row = static_cast<size_t>(lz) * plane + static_cast<size_t>(y) * nx;
    const size_t rowYp = static_cast<size_t>(lz) * plane + static_cast<size_t>(yp) * nx;
    const size_t rowYm = static_cast<size_t>(lz) * plane + static_cast<size_t>(ym) * nx;
    const size_t rowZp = static_cast<size_t>(lz + 1) * plane + static_cast<size_t>(y) * nx;
    const size_t rowZm = static_cast<size_t>(lz - 1) * plane + static_cast<size_t>(y) * nx;

    const double center = mu[row + x];
    const double muxx = (mu[row + xp] + mu[row + xm] - 2.0 * center) * invDx2;
    const double muyy = (mu[rowYp + x] + mu[rowYm + x] - 2.0 * center) * invDy2;
    const double muzz = (mu[rowZp + x] + mu[rowZm + x] - 2.0 * center) * invDz2;

    cnew[row + x] = cold[row + x] + dtD * (muxx + muyy + muzz);
}

// -----------------------------------------------------------------------------
// Domain decomposition helpers
// -----------------------------------------------------------------------------

static void partitionZ(const size_t nz, const int nranks, const int rank,
                       size_t& z0, size_t& z1) {
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    z0 = static_cast<size_t>(rank) * base + std::min<size_t>(static_cast<size_t>(rank), rem);
    z1 = z0 + base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

// Exchange one ghost layer in z.  Host staging is used so that the code works
// with both CUDA-aware and non-CUDA-aware MPI implementations.
static void exchangeHalos(double* d_field, const int nx, const int ny,
                          const size_t nzLocal, const int rank, const int nranks,
                          MPI_Comm comm, double* hbuf) {
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const int count = static_cast<int>(plane);
    double* sendLo = hbuf;
    double* sendHi = hbuf + plane;
    double* recvLo = hbuf + 2 * plane;
    double* recvHi = hbuf + 3 * plane;
    const int lo = rank - 1;
    const int hi = rank + 1;

    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(sendLo, d_field + plane, plane * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    if (rank < nranks - 1) {
        CUDA_CHECK(cudaMemcpy(sendHi, d_field + nzLocal * plane, plane * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    if (rank > 0) {
        MPI_Sendrecv(sendLo, count, MPI_DOUBLE, lo, 0,
                     recvLo, count, MPI_DOUBLE, lo, 1, comm, MPI_STATUS_IGNORE);
    }
    if (rank < nranks - 1) {
        MPI_Sendrecv(sendHi, count, MPI_DOUBLE, hi, 1,
                     recvHi, count, MPI_DOUBLE, hi, 0, comm, MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(d_field, d_field + plane, plane * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field, recvLo, plane * sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    if (rank == nranks - 1) {
        CUDA_CHECK(cudaMemcpy(d_field + (nzLocal + 1) * plane, d_field + nzLocal * plane,
                              plane * sizeof(double), cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field + (nzLocal + 1) * plane, recvHi,
                              plane * sizeof(double), cudaMemcpyHostToDevice));
    }
}

// Initialize the local (z-padded) concentration field.  The pseudo-random values
// are computed from the global linear index, so the result is identical for any
// number of MPI ranks.
static void initializeConcentration(std::vector<double>& c, const size_t z0,
                                    const size_t nzLocal, const size_t nx,
                                    const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    const size_t vol = plane * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nzLocal; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = z0 + lz;
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = gz * plane + y * nx + x;
                const double pseudo =
                    (static_cast<double>(((linear_id + 1) * 1299709) % vol) /
                     static_cast<double>(vol));
                c[idx3(x, y, lz + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Pick a GPU based on the node-local rank, unless the launcher already
    // restricted the visible device set for this process.
    int localRank = 0;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &localRank);
    int deviceCount = 1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&nodeComm);

    omp_set_num_threads(std::max(1, omp_get_max_threads()));

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) printf("Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (static_cast<size_t>(nranks) > nz) {
        if (rank == 0) {
            printf("Number of MPI ranks (%d) exceeds the z grid size (%zu)\n", nranks, nz);
        }
        MPI_Finalize();
        return 1;
    }

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

    size_t z0 = 0;
    size_t z1 = 0;
    partitionZ(nz, nranks, rank, z0, z1);
    const size_t nzLocal = z1 - z0;
    const size_t plane = nx * ny;
    const size_t localPadded = plane * (nzLocal + 2);

    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    // Device buffers
    double* dCold = nullptr;
    double* dCnew = nullptr;
    double* dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dCold, localPadded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCnew, localPadded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMu, localPadded * sizeof(double)));

    double* hHalo = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hHalo), 4 * plane * sizeof(double),
                             cudaHostAllocDefault));

    // Initialize concentration field (host preparation + OpenMP)
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> hCold(localPadded, 0.0);
    initializeConcentration(hCold, z0, nzLocal, nx, ny, nz);
    CUDA_CHECK(cudaMemcpy(dCold, hCold.data(), localPadded * sizeof(double),
                          cudaMemcpyHostToDevice));
    hCold.clear();
    hCold.shrink_to_fit();

    exchangeHalos(dCold, static_cast<int>(nx), static_cast<int>(ny), nzLocal, rank, nranks,
                  MPI_COMM_WORLD, hHalo);

    const dim3 block(8, 8, 8);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>((nzLocal + block.z - 1) / block.z));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block>>>(
            dCold, dMu, static_cast<int>(nx), static_cast<int>(ny),
            static_cast<int>(nzLocal), plane, invDx2, invDy2, invDz2, gamma,
            e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        exchangeHalos(dMu, static_cast<int>(nx), static_cast<int>(ny), nzLocal, rank, nranks,
                      MPI_COMM_WORLD, hHalo);

        cahnHilliardUpdateKernel<<<grid, block>>>(
            dCnew, dCold, dMu, static_cast<int>(nx), static_cast<int>(ny),
            static_cast<int>(nzLocal), plane, invDx2, invDy2, invDz2, dtD);
        CUDA_CHECK(cudaGetLastError());

        // Exchange the new concentration for the next time step.
        if (t + 1 < iterations) {
            exchangeHalos(dCnew, static_cast<int>(nx), static_cast<int>(ny), nzLocal, rank,
                          nranks, MPI_COMM_WORLD, hHalo);
        }

        std::swap(dCold, dCnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const long localDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration = 0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration);

        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) *
                                static_cast<double>(nz);
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double seconds = static_cast<double>(duration) / 1000.0;
        const double mcups = (seconds > 0.0) ? cellUpdates / seconds / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the local interior to a compact linear buffer for optional output and
    // validation.  Ranks own contiguous z slabs, so concatenating the slabs in
    // rank order reconstructs the original linear layout.
    const size_t localSize = plane * nzLocal;
    std::vector<double> local;
    if (printResults || validate) {
        local.resize(localSize);
        CUDA_CHECK(cudaMemcpy2D(local.data(), plane * sizeof(double),
                                dCold + plane, plane * sizeof(double),
                                plane * sizeof(double), nzLocal,
                                cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> global;

        if (rank == 0) {
            counts.resize(nranks);
            displs.resize(nranks);
            size_t disp = 0;
            for (int r = 0; r < nranks; ++r) {
                size_t r0 = 0, r1 = 0;
                partitionZ(nz, nranks, r, r0, r1);
                counts[r] = static_cast<int>(plane * (r1 - r0));
                displs[r] = static_cast<int>(disp);
                disp += plane * (r1 - r0);
            }
            global.resize(static_cast<size_t>(disp));
        }

        MPI_Gatherv(local.data(), static_cast<int>(localSize), MPI_DOUBLE,
                    (rank == 0 ? global.data() : nullptr),
                    (rank == 0 ? counts.data() : nullptr),
                    (rank == 0 ? displs.data() : nullptr),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    // Validation
    if (validate) {
        double localMin = localSize > 0 ? local[0] : 0.0;
        double localMax = localSize > 0 ? local[0] : 0.0;
        bool localBad = false;

#pragma omp parallel for reduction(min : localMin) reduction(max : localMax) \
    reduction(|| : localBad) schedule(static)
        for (size_t i = 0; i < localSize; ++i) {
            const double val = local[i];
            if (std::isnan(val) || std::isinf(val)) localBad = true;
            localMin = std::min(localMin, val);
            localMax = std::max(localMax, val);
        }

        int badFlag = localBad ? 1 : 0;
        double globalMin = 0.0;
        double globalMax = 0.0;
        int globalBad = 0;
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&badFlag, &globalBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        bool valid = true;
        if (rank == 0) {
            printf("Validating result...\n");
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
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }

        int validFlag = valid ? 1 : 0;
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        CUDA_CHECK(cudaFreeHost(hHalo));
        CUDA_CHECK(cudaFree(dCold));
        CUDA_CHECK(cudaFree(dCnew));
        CUDA_CHECK(cudaFree(dMu));
        MPI_Finalize();
        return validFlag ? 0 : 1;
    }

    CUDA_CHECK(cudaFreeHost(hHalo));
    CUDA_CHECK(cudaFree(dCold));
    CUDA_CHECK(cudaFree(dCnew));
    CUDA_CHECK(cudaFree(dMu));
    MPI_Finalize();
    return 0;
}
