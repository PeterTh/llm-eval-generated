#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CHECK_CUDA(call)                                                                              \
    do {                                                                                              \
        const cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    } while (0)

// Physical and numerical parameters, passed by value to the kernels
struct Params {
    double dx, dy, dz;
    double dtD;  // dt * D
    double gamma, e_AA, e_BB, e_AB;
};

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian with clamped boundary conditions in x and y. The z neighbours come from the
// ghost planes, which hold the clamped values at the global z boundaries.
__device__ double computeLaplacian(const double* c, const size_t nx, const size_t ny,
                                   const size_t x, const size_t y, const size_t z, const Params& p) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cc = 2.0 * c[idx3(x, y, z, nx, ny)];
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - cc) / (p.dx * p.dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - cc) / (p.dy * p.dy);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] - cc) / (p.dz * p.dz);

    return cxx + cyy + czz;
}

// Chemical potential over planes [zFirst, zFirst + zCount) of a ghosted slab.
// The fma() calls and operation order reproduce the fused multiply-adds of the reference
// build, so the result is bit-identical to it (nvcc is run with --fmad=false).
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t zFirst, const size_t zCount,
                                        const Params p) {
    for (size_t z = zFirst + blockIdx.z; z < zFirst + zCount; z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx; x += gridDim.x * blockDim.x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                const double lap = computeLaplacian(c, nx, ny, x, y, z, p);

                const double s1 = fma(cv + 1.0, p.e_AA, (cv - 1.0) * p.e_BB);
                const double s2 = fma(-(2.0 * cv), p.e_AB, s1);
                const double s3 = fma(s2, 4.5, 3.0 * cv);
                const double s4 = fma(cv, cv * cv, s3);
                mu[idx] = s4 - p.gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update over planes [zFirst, zFirst + zCount) of a ghosted slab
__global__ void cahnHilliardKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                                   const double* __restrict__ mu,
                                   const size_t nx, const size_t ny, const size_t zFirst, const size_t zCount,
                                   const Params p) {
    for (size_t z = zFirst + blockIdx.z; z < zFirst + zCount; z += gridDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
            for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < nx; x += gridDim.x * blockDim.x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = fma(p.dtD, computeLaplacian(mu, nx, ny, x, y, z, p), cold[idx]);
            }
        }
    }
}

// Initialize the owned z-slab [z0, z0 + nzl) using global indices, so the field does not depend on the decomposition
void initializeConcentration(std::vector<double>& owned, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t nzl) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (z0 + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                owned[z * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Contiguous z-range [z0, z0 + nzl) of a rank; the first (nz % nprocs) ranks get one extra plane
void slabOf(const int rank, const int nprocs, const size_t nz, size_t& z0, size_t& nzl) {
    const size_t base = nz / nprocs;
    const size_t extra = nz % nprocs;
    const size_t r = rank;
    nzl = base + (r < extra ? 1 : 0);
    z0 = r * base + std::min(r, extra);
}

// Checks on the full field, reduced over all ranks of comm. Same checks and output as the serial reference.
bool validateField(const std::vector<double>& owned, MPI_Comm comm, const bool root) {
    long long bad = 0;
    double minVal = INFINITY;
    double maxVal = -INFINITY;

    #pragma omp parallel for reduction(+ : bad) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < owned.size(); ++i) {
        const double val = owned[i];
        if (std::isnan(val) || std::isinf(val)) ++bad;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    long long totalBad = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Allreduce(&bad, &totalBad, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    // Check for NaN or Inf
    if (totalBad > 0) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    if (root) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);

    // Values should generally stay within reasonable bounds
    if (globalMax > 10.0 || globalMin < -10.0) {
        if (root) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Gather all owned slabs on rank 0 in global z order. Slabs are contiguous in memory, so no reordering is needed.
void gatherField(const std::vector<double>& owned, MPI_Comm comm, const size_t nz, const size_t plane,
                 std::vector<double>& full) {
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t z0 = 0;
        size_t nzl = 0;
        slabOf(r, nprocs, nz, z0, nzl);
        if (nzl * plane > static_cast<size_t>(INT_MAX)) {
            fprintf(stderr, "Grid too large for a single MPI gather\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[r] = static_cast<int>(nzl * plane);
        displs[r] = static_cast<int>(z0 * plane);
    }

    if (rank == 0) full.resize(nz * plane);
    MPI_Gatherv(owned.data(), static_cast<int>(owned.size()), MPI_DOUBLE,
                full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
}

// One rank's z-slab on one GPU. Device fields hold nzl owned planes plus a ghost plane on each side
// (plane 0 and plane nzl + 1). A ghost plane holds the neighbour's boundary plane or, at the global
// z boundaries, a copy of this rank's own boundary plane (the clamped boundary condition).
class Slab {
  public:
    Slab(MPI_Comm comm, int rank, int nprocs, size_t nx, size_t ny, size_t nzl);
    ~Slab();

    Slab(const Slab&) = delete;
    Slab& operator=(const Slab&) = delete;

    void initialize(const std::vector<double>& owned);
    void step(const Params& p);
    void sync();
    void download(std::vector<double>& owned);

  private:
    void exchange(double* field, cudaEvent_t fieldReady);
    void chemPot(const double* c, size_t zFirst, size_t zCount, const Params& p);
    void update(const double* cold, double* cnew, size_t zFirst, size_t zCount, const Params& p);

    MPI_Comm comm;
    int down;  // neighbour below, MPI_PROC_NULL at the global lower boundary
    int up;    // neighbour above, MPI_PROC_NULL at the global upper boundary
    size_t nx;
    size_t ny;
    size_t nzl;
    size_t plane;  // nx * ny
    int k = 0;     // index of the current concentration buffer in c[]
    double* c[2];
    double* mu;

    // Pinned host buffers for the halo planes; shared by the concentration and chemical potential exchanges
    double* hSendLo;
    double* hSendHi;
    double* hRecvLo;
    double* hRecvHi;

    cudaStream_t compute;
    cudaStream_t copy;
    cudaEvent_t ready;     // end of step: concentration is up to date
    cudaEvent_t muReady;   // owned chemical potential planes are up to date
    cudaEvent_t haloIn;    // host-to-device halo copies have finished
};

Slab::Slab(MPI_Comm comm_, int rank, int nprocs, size_t nx_, size_t ny_, size_t nzl_)
    : comm(comm_),
      down(rank > 0 ? rank - 1 : MPI_PROC_NULL),
      up(rank + 1 < nprocs ? rank + 1 : MPI_PROC_NULL),
      nx(nx_),
      ny(ny_),
      nzl(nzl_),
      plane(nx_ * ny_) {
    if (plane > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Grid plane too large for MPI halo messages\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t fieldBytes = (nzl + 2) * plane * sizeof(double);
    const size_t haloBytes = plane * sizeof(double);
    CHECK_CUDA(cudaMalloc(&c[0], fieldBytes));
    CHECK_CUDA(cudaMalloc(&c[1], fieldBytes));
    CHECK_CUDA(cudaMalloc(&mu, fieldBytes));
    CHECK_CUDA(cudaMallocHost(&hSendLo, haloBytes));
    CHECK_CUDA(cudaMallocHost(&hSendHi, haloBytes));
    CHECK_CUDA(cudaMallocHost(&hRecvLo, haloBytes));
    CHECK_CUDA(cudaMallocHost(&hRecvHi, haloBytes));

    CHECK_CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&muReady, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&haloIn, cudaEventDisableTiming));
    CHECK_CUDA(cudaEventRecord(ready, compute));
}

Slab::~Slab() {
    cudaEventDestroy(haloIn);
    cudaEventDestroy(muReady);
    cudaEventDestroy(ready);
    cudaStreamDestroy(copy);
    cudaStreamDestroy(compute);
    cudaFreeHost(hRecvHi);
    cudaFreeHost(hRecvLo);
    cudaFreeHost(hSendHi);
    cudaFreeHost(hSendLo);
    cudaFree(mu);
    cudaFree(c[1]);
    cudaFree(c[0]);
}

void Slab::initialize(const std::vector<double>& owned) {
    CHECK_CUDA(cudaMemcpy(c[k] + plane, owned.data(), nzl * plane * sizeof(double), cudaMemcpyHostToDevice));
}

void Slab::sync() {
    CHECK_CUDA(cudaStreamSynchronize(compute));
}

void Slab::download(std::vector<double>& owned) {
    sync();
    owned.resize(nzl * plane);
    CHECK_CUDA(cudaMemcpy(owned.data(), c[k] + plane, nzl * plane * sizeof(double), cudaMemcpyDeviceToHost));
}

void Slab::chemPot(const double* cur, const size_t zFirst, const size_t zCount, const Params& p) {
    if (zCount == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>(std::min<size_t>((nx + block.x - 1) / block.x, 65535)),
                    static_cast<unsigned int>(std::min<size_t>((ny + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(zCount, 65535)));
    chemicalPotentialKernel<<<grid, block, 0, compute>>>(cur, mu, nx, ny, zFirst, zCount, p);
    CHECK_CUDA(cudaGetLastError());
}

void Slab::update(const double* cold, double* cnew, const size_t zFirst, const size_t zCount, const Params& p) {
    if (zCount == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>(std::min<size_t>((nx + block.x - 1) / block.x, 65535)),
                    static_cast<unsigned int>(std::min<size_t>((ny + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned int>(std::min<size_t>(zCount, 65535)));
    cahnHilliardKernel<<<grid, block, 0, compute>>>(cold, cnew, mu, nx, ny, zFirst, zCount, p);
    CHECK_CUDA(cudaGetLastError());
}

// Refresh the ghost planes of a device field. Waits until the owned planes are ready on the
// compute stream, stages the boundary planes through pinned host memory and exchanges them with MPI.
void Slab::exchange(double* field, cudaEvent_t fieldReady) {
    const size_t bytes = plane * sizeof(double);
    const int count = static_cast<int>(plane);

    CHECK_CUDA(cudaStreamWaitEvent(copy, fieldReady, 0));
    CHECK_CUDA(cudaMemcpyAsync(hSendLo, field + plane, bytes, cudaMemcpyDeviceToHost, copy));
    CHECK_CUDA(cudaMemcpyAsync(hSendHi, field + nzl * plane, bytes, cudaMemcpyDeviceToHost, copy));
    CHECK_CUDA(cudaStreamSynchronize(copy));

    MPI_Sendrecv(hSendHi, count, MPI_DOUBLE, up, 0, hRecvLo, count, MPI_DOUBLE, down, 0, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(hSendLo, count, MPI_DOUBLE, down, 1, hRecvHi, count, MPI_DOUBLE, up, 1, comm, MPI_STATUS_IGNORE);

    // Clamped boundary at the global z ends: the ghost plane is a copy of the owned boundary plane
    if (down == MPI_PROC_NULL) memcpy(hRecvLo, hSendLo, bytes);
    if (up == MPI_PROC_NULL) memcpy(hRecvHi, hSendHi, bytes);

    CHECK_CUDA(cudaMemcpyAsync(field, hRecvLo, bytes, cudaMemcpyHostToDevice, copy));
    CHECK_CUDA(cudaMemcpyAsync(field + (nzl + 1) * plane, hRecvHi, bytes, cudaMemcpyHostToDevice, copy));
    CHECK_CUDA(cudaEventRecord(haloIn, copy));
}

// One Cahn-Hilliard time step. Interior planes only read owned data, so they overlap with the halo exchanges.
void Slab::step(const Params& p) {
    const double* cur = c[k];
    double* nxt = c[1 - k];
    const size_t interior = nzl > 2 ? nzl - 2 : 0;

    chemPot(cur, 2, interior, p);

    exchange(c[k], ready);
    CHECK_CUDA(cudaStreamWaitEvent(compute, haloIn, 0));
    chemPot(cur, 1, 1, p);
    if (nzl > 1) chemPot(cur, nzl, 1, p);
    CHECK_CUDA(cudaEventRecord(muReady, compute));

    update(cur, nxt, 2, interior, p);

    exchange(mu, muReady);
    CHECK_CUDA(cudaStreamWaitEvent(compute, haloIn, 0));
    update(cur, nxt, 1, 1, p);
    if (nzl > 1) update(cur, nxt, nzl, 1, p);
    CHECK_CUDA(cudaEventRecord(ready, compute));

    k = 1 - k;
}

// Runs the simulation on the ranks of comm. Returns the slowest rank's time in ms (valid on rank 0 of comm).
// If keepField is set, the owned slab of the final field is returned in owned.
long long runSimulation(MPI_Comm comm, const size_t nx, const size_t ny, const size_t nz, const int iterations,
                        const Params& p, const bool keepField, std::vector<double>& owned) {
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    size_t z0 = 0;
    size_t nzl = 0;
    slabOf(rank, nprocs, nz, z0, nzl);

    // One GPU per rank: ranks on a node are spread round-robin over its devices
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int numDevices = 0;
    CHECK_CUDA(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK_CUDA(cudaSetDevice(localRank % numDevices));

    owned.resize(nzl * nx * ny);
    initializeConcentration(owned, nx, ny, nz, z0, nzl);

    long long elapsedMs = 0;
    {
        Slab slab(comm, rank, nprocs, nx, ny, nzl);
        slab.initialize(owned);

        MPI_Barrier(comm);
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            slab.step(p);
        }
        slab.sync();

        auto end = std::chrono::high_resolution_clock::now();
        const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        MPI_Reduce(&ms, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

        if (keepField) slab.download(owned);
    }
    return elapsedMs;
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
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = worldRank == 0;

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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
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
    const Params p{dx, dy, dz, dt * D, gamma, e_AA, e_BB, e_AB};

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // Decompose along z into one slab per active rank
    const size_t nActive = gridSize == 0 ? 0 : std::min<size_t>(worldSize, nz);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < static_cast<int>(nActive) ? 0 : MPI_UNDEFINED, worldRank, &comm);

    if (root) printf("Initializing concentration field...\n");
    if (root) printf("Running Cahn-Hilliard simulation...\n");

    std::vector<double> owned;
    long long elapsedMs = 0;
    if (comm != MPI_COMM_NULL) {
        elapsedMs = runSimulation(comm, nx, ny, nz, iterations, p, printResults || validate, owned);
    }

    if (root) printf("Computation time: %lld ms\n", elapsedMs);

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
    if (root) printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        if (comm != MPI_COMM_NULL) gatherField(owned, comm, nz, plane, full);
        if (root) print_results(full, "Concentration");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        // Ranks outside comm (only possible for an empty field) have nothing to check
        const bool valid = comm == MPI_COMM_NULL || validateField(owned, comm, root);

        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = valid ? 0 : 1;
    }

    if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
