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

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,  \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential for local z-planes [z0, z0 + numPlanes) in halo coordinates.
// The z direction is uniform (z-1 / z+1 always valid): halo planes carry either the
// neighbor rank's boundary plane or a copy of the local boundary plane (clamped BC).
__global__ void muKernel(const double* __restrict__ c, double* __restrict__ mu,
                         const int nx, const int ny, const int z0,
                         const double dx, const double dy, const double dz,
                         const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int z = z0 + blockIdx.z;

    const size_t plane = (size_t)nx * ny;
    const size_t idx = (size_t)z * plane + (size_t)y * nx + x;

    const int xp = (x < nx - 1) ? 1 : 0;
    const int xn = (x > 0) ? -1 : 0;
    const long yp = (y < ny - 1) ? nx : 0;
    const long yn = (y > 0) ? -nx : 0;

    const double cv = c[idx];
    const double cxx = (c[idx + xp] + c[idx + xn] - 2.0 * cv) / (dx * dx);
    const double cyy = (c[idx + yp] + c[idx + yn] - 2.0 * cv) / (dy * dy);
    const double czz = (c[idx + plane] + c[idx - plane] - 2.0 * cv) / (dz * dz);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * (cxx + cyy + czz);
}

// Cahn-Hilliard update step for local z-planes [z0, z0 + numPlanes) in halo coordinates
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             const int nx, const int ny, const int z0,
                             const double D, const double dt,
                             const double dx, const double dy, const double dz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int z = z0 + blockIdx.z;

    const size_t plane = (size_t)nx * ny;
    const size_t idx = (size_t)z * plane + (size_t)y * nx + x;

    const int xp = (x < nx - 1) ? 1 : 0;
    const int xn = (x > 0) ? -1 : 0;
    const long yp = (y < ny - 1) ? nx : 0;
    const long yn = (y > 0) ? -nx : 0;

    const double mv = mu[idx];
    const double mxx = (mu[idx + xp] + mu[idx + xn] - 2.0 * mv) / (dx * dx);
    const double myy = (mu[idx + yp] + mu[idx + yn] - 2.0 * mv) / (dy * dy);
    const double mzz = (mu[idx + plane] + mu[idx - plane] - 2.0 * mv) / (dz * dz);

    cnew[idx] = cold[idx] + dt * D * (mxx + myy + mzz);
}

// Initialize the local slab (planes [1, nzl] in halo coordinates) of the concentration
// field, matching the original global pseudo-random pattern
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t gz0, const size_t nzl) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                // Generate pseudo-random value in [-1, 1] from the global linear id
                const size_t linear_id = (gz0 + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf, and compute range
    bool bad = false;
    double minVal = c[0];
    double maxVal = c[0];
    const size_t n = c.size();

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        bad = bad || std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

// Halo exchange of one z-plane in each direction, staged through pinned host buffers.
// At global boundaries the halo plane is filled with a copy of the local boundary
// plane, which reproduces the clamped boundary condition of the serial code.
struct HaloExchanger {
    int prevRank = MPI_PROC_NULL;
    int nextRank = MPI_PROC_NULL;
    size_t planeElems = 0;
    size_t nzl = 0;
    double* h_sendLow = nullptr;
    double* h_sendHigh = nullptr;
    double* h_recvLow = nullptr;
    double* h_recvHigh = nullptr;
    cudaStream_t stream = nullptr;

    void exchange(double* d_field) const {
        const size_t bytes = planeElems * sizeof(double);
        double* lowPlane = d_field;                            // halo plane 0
        double* firstPlane = d_field + planeElems;             // local plane 1
        double* lastPlane = d_field + planeElems * nzl;        // local plane nzl
        double* highPlane = d_field + planeElems * (nzl + 1);  // halo plane nzl+1

        if (prevRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendLow, firstPlane, bytes, cudaMemcpyDeviceToHost, stream));
        if (nextRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendHigh, lastPlane, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Request reqs[4];
        int nreq = 0;
        if (prevRank != MPI_PROC_NULL) {
            MPI_Irecv(h_recvLow, (int)planeElems, MPI_DOUBLE, prevRank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(h_sendLow, (int)planeElems, MPI_DOUBLE, prevRank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (nextRank != MPI_PROC_NULL) {
            MPI_Irecv(h_recvHigh, (int)planeElems, MPI_DOUBLE, nextRank, 1, MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Isend(h_sendHigh, (int)planeElems, MPI_DOUBLE, nextRank, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        if (prevRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(lowPlane, h_recvLow, bytes, cudaMemcpyHostToDevice, stream));
        else
            CUDA_CHECK(cudaMemcpyAsync(lowPlane, firstPlane, bytes, cudaMemcpyDeviceToDevice, stream));
        if (nextRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(highPlane, h_recvHigh, bytes, cudaMemcpyHostToDevice, stream));
        else
            CUDA_CHECK(cudaMemcpyAsync(highPlane, lastPlane, bytes, cudaMemcpyDeviceToDevice, stream));
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (isRoot) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks, omp_get_max_threads());
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
    const size_t planeElems = nx * ny;

    // 1D slab decomposition along z
    const size_t base = nz / nranks;
    const size_t rem = nz % nranks;
    const size_t gz0 = rank * base + std::min<size_t>(rank, rem);
    const size_t nzl = base + ((size_t)rank < rem ? 1 : 0);
    const bool active = (nzl > 0);

    // Neighbors (ranks with zero planes only occur at the tail, so the previous
    // rank is always active and the last active rank has no next neighbor)
    const int prevRank = (active && gz0 > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (active && gz0 + nzl < nz) ? rank + 1 : MPI_PROC_NULL;

    // Bind one GPU per rank, round-robin over the node-local ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        if (isRoot) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devCount));

    // Local slab with one halo plane on each side
    const size_t localElems = planeElems * (nzl + 2);
    const size_t localBytes = localElems * sizeof(double);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    cudaStream_t streamCompute = nullptr;
    cudaStream_t streamComm = nullptr;
    HaloExchanger halo;

    if (active) {
        CUDA_CHECK(cudaMalloc(&d_cold, localBytes));
        CUDA_CHECK(cudaMalloc(&d_cnew, localBytes));
        CUDA_CHECK(cudaMalloc(&d_mu, localBytes));
        CUDA_CHECK(cudaMemset(d_cold, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_cnew, 0, localBytes));
        CUDA_CHECK(cudaMemset(d_mu, 0, localBytes));
        CUDA_CHECK(cudaStreamCreate(&streamCompute));
        CUDA_CHECK(cudaStreamCreate(&streamComm));

        halo.prevRank = prevRank;
        halo.nextRank = nextRank;
        halo.planeElems = planeElems;
        halo.nzl = nzl;
        halo.stream = streamComm;
        CUDA_CHECK(cudaHostAlloc(&halo.h_sendLow, planeElems * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&halo.h_sendHigh, planeElems * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&halo.h_recvLow, planeElems * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&halo.h_recvHigh, planeElems * sizeof(double), cudaHostAllocDefault));
    }

    // Initialize concentration field on the host (OpenMP) and upload
    if (isRoot) printf("Initializing concentration field...\n");
    std::vector<double> localField(active ? localElems : 0, 0.0);
    if (active) {
        initializeConcentration(localField, nx, ny, nz, gz0, nzl);
        CUDA_CHECK(cudaMemcpy(d_cold, localField.data(), localBytes, cudaMemcpyHostToDevice));
    }

    const dim3 block(32, 8, 1);
    const dim3 gridXY((unsigned)((nx + block.x - 1) / block.x),
                      (unsigned)((ny + block.y - 1) / block.y), 1);
    // Interior planes [2, nzl-1] can be computed without halo data; boundary
    // planes 1 and nzl need the exchanged halos
    const int interiorZ0 = 2;
    const int interiorCount = (nzl >= 3) ? (int)nzl - 2 : 0;

    auto launchMu = [&](int z0, int count, cudaStream_t s) {
        if (count <= 0) return;
        dim3 grid(gridXY.x, gridXY.y, (unsigned)count);
        muKernel<<<grid, block, 0, s>>>(d_cold, d_mu, (int)nx, (int)ny, z0,
                                        dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    };
    auto launchUpdate = [&](int z0, int count, cudaStream_t s) {
        if (count <= 0) return;
        dim3 grid(gridXY.x, gridXY.y, (unsigned)count);
        updateKernel<<<grid, block, 0, s>>>(d_cnew, d_cold, d_mu, (int)nx, (int)ny, z0,
                                            D, dt, dx, dy, dz);
    };

    // Run simulation
    if (isRoot) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // Compute chemical potential: overlap interior compute with c halo exchange
            launchMu(interiorZ0, interiorCount, streamCompute);
            halo.exchange(d_cold);
            launchMu(1, 1, streamComm);
            if (nzl > 1) launchMu((int)nzl, 1, streamComm);
            CUDA_CHECK(cudaStreamSynchronize(streamCompute));
            CUDA_CHECK(cudaStreamSynchronize(streamComm));

            // Update concentration: overlap interior compute with mu halo exchange
            launchUpdate(interiorZ0, interiorCount, streamCompute);
            halo.exchange(d_mu);
            launchUpdate(1, 1, streamComm);
            if (nzl > 1) launchUpdate((int)nzl, 1, streamComm);
            CUDA_CHECK(cudaStreamSynchronize(streamCompute));
            CUDA_CHECK(cudaStreamSynchronize(streamComm));
        }

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    if (active) CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxDuration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    if (printResults || validate) {
        // Download local slab and gather the full field on the root rank
        if (active) {
            CUDA_CHECK(cudaMemcpy(localField.data(), d_cold, localBytes, cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            const size_t rNzl = base + ((size_t)r < rem ? 1 : 0);
            const size_t rGz0 = r * base + std::min<size_t>((size_t)r, rem);
            counts[r] = (int)(rNzl * planeElems);
            displs[r] = (int)(rGz0 * planeElems);
        }

        std::vector<double> cfull(isRoot ? gridSize : 0);
        MPI_Gatherv(active ? localField.data() + planeElems : nullptr, (int)(nzl * planeElems),
                    MPI_DOUBLE, isRoot ? cfull.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (isRoot) {
            // Print results for external validation
            if (printResults) {
                print_results(cfull, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(cfull, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        if (validate) MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (active) {
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(halo.h_sendLow));
        CUDA_CHECK(cudaFreeHost(halo.h_sendHigh));
        CUDA_CHECK(cudaFreeHost(halo.h_recvLow));
        CUDA_CHECK(cudaFreeHost(halo.h_recvHigh));
        CUDA_CHECK(cudaStreamDestroy(streamCompute));
        CUDA_CHECK(cudaStreamDestroy(streamComm));
    }

    MPI_Finalize();
    return exitCode;
}
