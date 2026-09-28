#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (call);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err),         \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                 \
    } while (0)

// 3D index calculation (z is the outermost/slowest-varying dimension, so a
// plane of constant z is contiguous in memory - this is what makes the
// halo exchange below a simple contiguous memcpy).
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian for a cell (x, y, zl) of the local sub-domain.
// The local buffer holds nzl interior planes plus one ghost plane below
// (zl == 0) and one ghost plane above (zl == nzl + 1); those ghost planes
// are always populated (via halo exchange with the MPI neighbor, or by a
// clamped copy of the adjacent interior plane at the global domain edge)
// before this is called, so no special-casing of the z direction is needed.
__device__ inline double deviceLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny,
                                         const double dx, const double dy, const double dz,
                                         const size_t x, const size_t y, const size_t zl) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = zl + 1;
    const size_t zn = zl - 1;

    const size_t plane = nx * ny;
    const size_t base = zl * plane + y * nx + x;
    const double cv = c[base];

    const double cxx = (c[zl * plane + y * nx + xp] + c[zl * plane + y * nx + xn] - 2.0 * cv) / (dx * dx);
    const double cyy = (c[zl * plane + yp * nx + x] + c[zl * plane + yn * nx + x] - 2.0 * cv) / (dy * dy);
    const double czz = (c[zp * plane + y * nx + x] + c[zn * plane + y * nx + x] - 2.0 * cv) / (dz * dz);

    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                const size_t nx, const size_t ny, const size_t nzl,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || zl > nzl) return;

    const size_t base = zl * (nx * ny) + y * nx + x;
    const double cv = c[base];

    mu[base] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * deviceLaplacian(c, nx, ny, dx, dy, dz, x, y, zl);
}

__global__ void cahnHilliardUpdateKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nzl,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || zl > nzl) return;

    const size_t base = zl * (nx * ny) + y * nx + x;
    cnew[base] = cold[base] + dt * D * deviceLaplacian(mu, nx, ny, dx, dy, dz, x, y, zl);
}

static void launchComputeChemicalPotential(const double* d_c, double* d_mu, size_t nx, size_t ny, size_t nzl,
                                           double dx, double dy, double dz,
                                           double gamma, double e_AA, double e_BB, double e_AB) {
    const dim3 block(16, 16, 4);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>((nzl + block.z - 1) / block.z));
    computeChemicalPotentialKernel<<<grid, block>>>(d_c, d_mu, nx, ny, nzl, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

static void launchCahnHilliardUpdate(const double* d_cold, double* d_cnew, const double* d_mu,
                                     size_t nx, size_t ny, size_t nzl,
                                     double D, double dt, double dx, double dy, double dz) {
    const dim3 block(16, 16, 4);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>((nzl + block.z - 1) / block.z));
    cahnHilliardUpdateKernel<<<grid, block>>>(d_cold, d_cnew, d_mu, nx, ny, nzl, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

// Initialize the local slab of the concentration field (global z in
// [z_start, z_start + nzl)). Uses the same pseudo-random formula as the
// original serial implementation, keyed off the *global* linear index, so
// the field is identical to the single-process result regardless of how
// the domain is decomposed across MPI ranks.
void initializeConcentrationLocal(std::vector<double>& c_local, const size_t nx, const size_t ny,
                                  const size_t nzl, const size_t z_start, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < nzl; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t z = z_start + zl;
                const size_t local_idx = zl * (nx * ny) + y * nx + x;
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c_local[local_idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    bool foundBad = false;
    #pragma omp parallel for reduction(||:foundBad) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            foundBad = true;
        }
    }
    if (foundBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
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

// Exchange the top/bottom ghost planes of a local field with the MPI
// neighbors above/below in the z decomposition. At the global domain edges
// (no neighbor) the ghost plane is instead filled with a clamped copy of the
// adjacent interior plane, reproducing the original code's clamped boundary
// condition exactly.
static void exchangeHalo(double* d_field, const size_t nx, const size_t ny, const size_t nzl,
                         int rank_below, int rank_above,
                         std::vector<double>& h_send_low, std::vector<double>& h_send_high,
                         std::vector<double>& h_recv_low, std::vector<double>& h_recv_high) {
    const size_t plane = nx * ny;
    const int count = static_cast<int>(plane);

    CUDA_CHECK(cudaMemcpy(h_send_low.data(), d_field + plane * 1, plane * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_high.data(), d_field + plane * nzl, plane * sizeof(double), cudaMemcpyDeviceToHost));

    MPI_Request reqs[4];
    MPI_Isend(h_send_low.data(), count, MPI_DOUBLE, rank_below, 100, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(h_send_high.data(), count, MPI_DOUBLE, rank_above, 200, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(h_recv_high.data(), count, MPI_DOUBLE, rank_above, 100, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(h_recv_low.data(), count, MPI_DOUBLE, rank_below, 200, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    if (rank_below != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_field + plane * 0, h_recv_low.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field + plane * 0, d_field + plane * 1, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    if (rank_above != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_field + plane * (nzl + 1), h_recv_high.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field + plane * (nzl + 1), d_field + plane * nzl, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank since argv is
    // the same everywhere under mpirun).
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

    if (nz < static_cast<size_t>(numRanks)) {
        if (rank == 0) {
            printf("Error: grid size in Z (%zu) must be >= number of MPI ranks (%d)\n", nz, numRanks);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
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

    // Decompose the z dimension into contiguous slabs, one per MPI rank.
    const size_t base = nz / static_cast<size_t>(numRanks);
    const size_t rem = nz % static_cast<size_t>(numRanks);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t nzl = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Bind this rank to a GPU: ranks sharing a node round-robin over that
    // node's visible devices.
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t plane = nx * ny;
    const size_t localBufElems = plane * (nzl + 2);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, localBufElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localBufElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localBufElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, localBufElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localBufElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localBufElems * sizeof(double)));

    // Initialize concentration field (host, OpenMP-parallel) then upload.
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> h_local(plane * nzl);
    initializeConcentrationLocal(h_local, nx, ny, nzl, z_start, nz);
    CUDA_CHECK(cudaMemcpy(d_cold + plane * 1, h_local.data(), plane * nzl * sizeof(double), cudaMemcpyHostToDevice));

    // Halo staging buffers (reused across cold/mu exchanges).
    std::vector<double> h_send_low(plane), h_send_high(plane), h_recv_low(plane), h_recv_high(plane);

    const int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rank_above = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos, then compute chemical potential.
        exchangeHalo(d_cold, nx, ny, nzl, rank_below, rank_above, h_send_low, h_send_high, h_recv_low, h_recv_high);
        launchComputeChemicalPotential(d_cold, d_mu, nx, ny, nzl, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exchange chemical-potential halos, then update concentration.
        exchangeHalo(d_mu, nx, ny, nzl, rank_below, rank_above, h_send_low, h_send_high, h_recv_low, h_recv_high);
        launchCahnHilliardUpdate(d_cold, d_cnew, d_mu, nx, ny, nzl, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMs = duration.count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full concentration field back to rank 0 in the original
    // (global, serial) layout so downstream reporting is bit-identical to
    // the single-process implementation.
    std::vector<double> h_result_local(plane * nzl);
    CUDA_CHECK(cudaMemcpy(h_result_local.data(), d_cold + plane * 1, plane * nzl * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<int> recvcounts, displs;
    if (rank == 0) {
        recvcounts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t rz_start = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            const size_t rnzl = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            recvcounts[r] = static_cast<int>(plane * rnzl);
            displs[r] = static_cast<int>(plane * rz_start);
        }
    }

    std::vector<double> full;
    if (rank == 0) full.resize(nx * ny * nz);
    MPI_Gatherv(h_result_local.data(), static_cast<int>(plane * nzl), MPI_DOUBLE,
                rank == 0 ? full.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxMs);

        // Calculate performance
        double cellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

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
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exitCode;
}
