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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),          \
                    __FILE__, __LINE__);                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions on a local slab.
// The local field has one halo plane on each side (planes 0 and lnz+1);
// zl is the local plane index in [1, lnz], gz the corresponding global z.
__device__ inline double computeLaplacianDev(const double* __restrict__ c,
                                             const size_t nx, const size_t ny, const size_t nz,
                                             const double dx, const double dy, const double dz,
                                             const size_t x, const size_t y, const size_t zl,
                                             const size_t gz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (gz < nz - 1) ? zl + 1 : zl;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (gz > 0) ? zl - 1 : zl;

    const double cxx = (c[idx3(xp, y, zl, nx, ny)] + c[idx3(xn, y, zl, nx, ny)] -
                  2.0 * c[idx3(x, y, zl, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, zl, nx, ny)] + c[idx3(x, yn, zl, nx, ny)] -
                  2.0 * c[idx3(x, y, zl, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, zl, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local planes [zl_start, zl_start + nplanes)
// (halo planes of c must be current if the range touches them)
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const size_t zl_start, const size_t nplanes,
                                               const size_t z_off,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const size_t n = plane * nplanes;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t zl = i / plane + zl_start;
        const size_t rem = i % plane;
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        const size_t gz = z_off + zl - 1;
        const size_t idx = idx3(x, y, zl, nx, ny);
        const double cv = c[idx];

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDev(c, nx, ny, nz, dx, dy, dz, x, y, zl, gz);
    }
}

// Cahn-Hilliard update step on local planes [zl_start, zl_start + nplanes)
// (halo planes of mu must be current if the range touches them)
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const size_t zl_start, const size_t nplanes,
                                         const size_t z_off,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const size_t n = plane * nplanes;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t zl = i / plane + zl_start;
        const size_t rem = i % plane;
        const size_t y = rem / nx;
        const size_t x = rem % nx;
        const size_t gz = z_off + zl - 1;
        const size_t idx = idx3(x, y, zl, nx, ny);
        cnew[idx] = cold[idx] + dt * D *
                   computeLaplacianDev(mu, nx, ny, nz, dx, dy, dz, x, y, zl, gz);
    }
}

// Halo exchange of one plane per z-neighbor, split into an asynchronous
// device-to-host stage (overlapped with interior computation on another
// stream) and a finish stage doing MPI transfer plus host-to-device copy.
// field points at the local slab including halos; plane = nx*ny doubles.
struct HaloExchange {
    size_t plane = 0;
    size_t lnz = 0;
    int prev = MPI_PROC_NULL;
    int next = MPI_PROC_NULL;
    double* h_send_lo = nullptr;
    double* h_recv_lo = nullptr;
    double* h_send_hi = nullptr;
    double* h_recv_hi = nullptr;
    cudaStream_t stream = nullptr;

    // Stage boundary planes to the host (async on the copy stream)
    void startD2H(const double* d_field) const {
        const size_t bytes = plane * sizeof(double);
        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_field + plane, bytes,
                                       cudaMemcpyDeviceToHost, stream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_field + lnz * plane, bytes,
                                       cudaMemcpyDeviceToHost, stream));
    }

    // Complete the MPI exchange and load received planes into the halos
    void finish(double* d_field) const {
        if (prev == MPI_PROC_NULL && next == MPI_PROC_NULL) return;
        const size_t bytes = plane * sizeof(double);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Sendrecv(h_send_lo, (int)plane, MPI_DOUBLE, prev, 0,
                     h_recv_hi, (int)plane, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_hi, (int)plane, MPI_DOUBLE, next, 1,
                     h_recv_lo, (int)plane, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (prev != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_field, h_recv_lo, bytes,
                                       cudaMemcpyHostToDevice, stream));
        if (next != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(d_field + (lnz + 1) * plane, h_recv_hi, bytes,
                                       cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// Initialize the local slab of the concentration field (OpenMP on the host)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t z_off, const size_t lnz) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < lnz; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z_off + zl) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf, and compute range (OpenMP on the host)
    bool bad = false;
    double minVal = c[0];
    double maxVal = c[0];
    const size_t n = c.size();

    #pragma omp parallel for schedule(static) reduction(||:bad) reduction(min:minVal) reduction(max:maxVal)
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

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", size, omp_get_max_threads());
    }

    // Bind each rank to a GPU on its node (round-robin over local ranks)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

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

    // 1D slab decomposition along z
    const size_t base = nz / size;
    const size_t extra = nz % size;
    const size_t lnz = base + ((size_t)rank < extra ? 1 : 0);
    const size_t z_off = (size_t)rank * base + std::min((size_t)rank, extra);

    // z-neighbors (ranks owning zero planes exchange nothing)
    const int prev = (lnz > 0 && z_off > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (lnz > 0 && z_off + lnz < nz) ? rank + 1 : MPI_PROC_NULL;

    const size_t localSize = plane * lnz;
    const size_t paddedSize = plane * (lnz + 2);  // slab plus one halo plane per side

    // Host copy of the local slab (no halos)
    std::vector<double> hostSlab(std::max<size_t>(localSize, 1));

    // Device arrays (with halos)
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold, 0, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, paddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, paddedSize * sizeof(double)));

    // Pinned staging buffers for halo exchange
    double *h_send_lo = nullptr, *h_recv_lo = nullptr, *h_send_hi = nullptr, *h_recv_hi = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, plane * sizeof(double)));

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(hostSlab, nx, ny, nz, z_off, lnz);
    if (lnz > 0)
        CUDA_CHECK(cudaMemcpy(d_cold + plane, hostSlab.data(), localSize * sizeof(double),
                              cudaMemcpyHostToDevice));

    cudaStream_t computeStream, copyStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    cudaEvent_t boundaryDone, stepDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&boundaryDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stepDone, cudaEventDisableTiming));

    HaloExchange halo{plane, lnz, prev, next, h_send_lo, h_recv_lo, h_send_hi, h_recv_hi, copyStream};

    // Planes whose stencil reads a halo are deferred until the exchange
    // completes; everything else ("interior") overlaps with communication.
    const size_t loDefer = (prev != MPI_PROC_NULL) ? 1 : 0;
    const size_t hiDefer = (next != MPI_PROC_NULL && lnz > loDefer) ? 1 : 0;
    const size_t intStart = 1 + loDefer;
    const size_t intPlanes = lnz - loDefer - hiDefer;

    const int blockSize = 256;
    auto blocksFor = [&](size_t nplanes) {
        const size_t n = (plane * nplanes + blockSize - 1) / blockSize;
        return (int)std::min<size_t>(std::max<size_t>(n, 1), 65535);
    };

    auto launchMu = [&](size_t zl_start, size_t nplanes) {
        if (nplanes == 0) return;
        computeChemicalPotentialKernel<<<blocksFor(nplanes), blockSize, 0, computeStream>>>(
            d_cold, d_mu, nx, ny, nz, zl_start, nplanes, z_off, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchUpdate = [&](size_t zl_start, size_t nplanes) {
        if (nplanes == 0) return;
        cahnHilliardUpdateKernel<<<blocksFor(nplanes), blockSize, 0, computeStream>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, zl_start, nplanes, z_off, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
    };

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Stage concentration boundary planes while computing interior mu
        halo.startD2H(d_cold);
        launchMu(intStart, intPlanes);
        halo.finish(d_cold);

        // Compute mu on the halo-dependent boundary planes
        launchMu(1, loDefer);
        if (lnz > loDefer) launchMu(lnz, hiDefer);

        // Stage mu boundary planes (after they are computed) while
        // updating the interior concentration
        CUDA_CHECK(cudaEventRecord(boundaryDone, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(copyStream, boundaryDone, 0));
        halo.startD2H(d_mu);
        launchUpdate(intStart, intPlanes);
        halo.finish(d_mu);

        // Update the halo-dependent boundary planes
        launchUpdate(1, loDefer);
        if (lnz > loDefer) launchUpdate(lnz, hiDefer);

        // Next step's staging must see this step's boundary results
        CUDA_CHECK(cudaEventRecord(stepDone, computeStream));
        CUDA_CHECK(cudaStreamWaitEvent(copyStream, stepDone, 0));

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy the local slab back and gather the full field on rank 0
    if (lnz > 0)
        CUDA_CHECK(cudaMemcpy(hostSlab.data(), d_cold + plane, localSize * sizeof(double),
                              cudaMemcpyDeviceToHost));

    std::vector<double> cold;
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rlnz = base + ((size_t)r < extra ? 1 : 0);
        const size_t roff = (size_t)r * base + std::min((size_t)r, extra);
        counts[r] = (int)(rlnz * plane);
        displs[r] = (int)(roff * plane);
    }
    if (rank == 0) cold.resize(gridSize);
    MPI_Gatherv(hostSlab.data(), (int)localSize, MPI_DOUBLE,
                rank == 0 ? cold.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxDuration / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

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

    CUDA_CHECK(cudaEventDestroy(boundaryDone));
    CUDA_CHECK(cudaEventDestroy(stepDone));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));

    MPI_Finalize();
    return exitCode;
}
