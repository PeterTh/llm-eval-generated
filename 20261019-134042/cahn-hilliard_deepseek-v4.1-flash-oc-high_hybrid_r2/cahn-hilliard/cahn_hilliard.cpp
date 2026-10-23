#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err__),       \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential on one z-slab (interior planes only).
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const int nx, const int ny, const int kbegin,
                                               const int kend, const double invdx2,
                                               const double invdy2, const double invdz2,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = kbegin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || k > kend) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xm = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int ym = (y > 0) ? y - 1 : 0;

    const size_t row = static_cast<size_t>(y) * static_cast<size_t>(nx);
    const size_t base = static_cast<size_t>(k) * plane + row + static_cast<size_t>(x);
    const double cv = c[base];

    const double cxx = (c[static_cast<size_t>(k) * plane + row + static_cast<size_t>(xp)] +
                        c[static_cast<size_t>(k) * plane + row + static_cast<size_t>(xm)] -
                        2.0 * cv) * invdx2;
    const double cyy = (c[static_cast<size_t>(k) * plane + static_cast<size_t>(yp) * static_cast<size_t>(nx) + static_cast<size_t>(x)] +
                        c[static_cast<size_t>(k) * plane + static_cast<size_t>(ym) * static_cast<size_t>(nx) + static_cast<size_t>(x)] -
                        2.0 * cv) * invdy2;
    const double czz = (c[static_cast<size_t>(k - 1) * plane + row + static_cast<size_t>(x)] +
                        c[static_cast<size_t>(k + 1) * plane + row + static_cast<size_t>(x)] -
                        2.0 * cv) * invdz2;

    mu[base] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * (cxx + cyy + czz);
}

// Cahn-Hilliard concentration update on one z-slab (interior planes only).
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const int kbegin,
                                         const int kend, const double invdx2,
                                         const double invdy2, const double invdz2,
                                         const double D, const double dt) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = kbegin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || k > kend) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xm = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int ym = (y > 0) ? y - 1 : 0;

    const size_t row = static_cast<size_t>(y) * static_cast<size_t>(nx);
    const size_t base = static_cast<size_t>(k) * plane + row + static_cast<size_t>(x);
    const double muv = mu[base];

    const double mxx = (mu[static_cast<size_t>(k) * plane + row + static_cast<size_t>(xp)] +
                        mu[static_cast<size_t>(k) * plane + row + static_cast<size_t>(xm)] -
                        2.0 * muv) * invdx2;
    const double myy = (mu[static_cast<size_t>(k) * plane + static_cast<size_t>(yp) * static_cast<size_t>(nx) + static_cast<size_t>(x)] +
                        mu[static_cast<size_t>(k) * plane + static_cast<size_t>(ym) * static_cast<size_t>(nx) + static_cast<size_t>(x)] -
                        2.0 * muv) * invdy2;
    const double mzz = (mu[static_cast<size_t>(k - 1) * plane + row + static_cast<size_t>(x)] +
                        mu[static_cast<size_t>(k + 1) * plane + row + static_cast<size_t>(x)] -
                        2.0 * muv) * invdz2;

    cnew[base] = cold[base] + dt * D * (mxx + myy + mzz);
}

// Exchange the two z ghost planes of a device field between neighbouring ranks.
// The device field is laid out as (lnz + 2) contiguous x-y planes.
static void exchangeHalos(double* d_field, const int lnz, const size_t plane,
                          MPI_Comm comm, const int lower, const int upper,
                          const bool atBottom, const bool atTop, double* hbuf,
                          cudaStream_t streamA, cudaStream_t streamB, cudaEvent_t ready) {
    const size_t bytes = plane * sizeof(double);
    double* sendFirst = hbuf;
    double* sendLast = hbuf + plane;
    double* recvBottom = hbuf + 2 * plane;
    double* recvTop = hbuf + 3 * plane;

    // The data was just produced on the default stream; make the copy streams wait.
    CUDA_CHECK(cudaEventRecord(ready, 0));
    CUDA_CHECK(cudaStreamWaitEvent(streamA, ready, 0));
    CUDA_CHECK(cudaStreamWaitEvent(streamB, ready, 0));

    // Single-slab case: both "neighbours" are clamped boundaries, so the ghosts are
    // just device-to-device copies of the adjacent interior planes.
    if (lower == MPI_PROC_NULL && upper == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(d_field, d_field + plane, bytes,
                                   cudaMemcpyDeviceToDevice, streamA));
        CUDA_CHECK(cudaMemcpyAsync(d_field + static_cast<size_t>(lnz + 1) * plane,
                                   d_field + static_cast<size_t>(lnz) * plane, bytes,
                                   cudaMemcpyDeviceToDevice, streamB));
        CUDA_CHECK(cudaStreamSynchronize(streamA));
        CUDA_CHECK(cudaStreamSynchronize(streamB));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(sendFirst, d_field + plane, bytes,
                               cudaMemcpyDeviceToHost, streamA));
    CUDA_CHECK(cudaMemcpyAsync(sendLast, d_field + static_cast<size_t>(lnz) * plane, bytes,
                               cudaMemcpyDeviceToHost, streamB));
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamB));

    const int count = static_cast<int>(plane);
    constexpr int TAG_FIRST = 100;
    constexpr int TAG_LAST = 101;

    // Exchange both directions concurrently. Sending the first plane down makes it
    // the lower neighbour's top ghost; sending the last plane up makes it the upper
    // neighbour's bottom ghost.
    MPI_Request reqs[4];
    MPI_Irecv(recvTop, count, MPI_DOUBLE, upper, TAG_FIRST, comm, &reqs[0]);
    MPI_Irecv(recvBottom, count, MPI_DOUBLE, lower, TAG_LAST, comm, &reqs[1]);
    MPI_Isend(sendFirst, count, MPI_DOUBLE, lower, TAG_FIRST, comm, &reqs[2]);
    MPI_Isend(sendLast, count, MPI_DOUBLE, upper, TAG_LAST, comm, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Clamped boundaries: the missing ghost is a copy of the adjacent interior plane.
    if (atBottom) std::memcpy(recvBottom, sendFirst, bytes);
    if (atTop) std::memcpy(recvTop, sendLast, bytes);

    CUDA_CHECK(cudaMemcpyAsync(d_field, recvBottom, bytes, cudaMemcpyHostToDevice, streamA));
    CUDA_CHECK(cudaMemcpyAsync(d_field + static_cast<size_t>(lnz + 1) * plane, recvTop, bytes,
                               cudaMemcpyHostToDevice, streamB));
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamB));
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
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind each MPI rank on a node to a distinct GPU (round-robin if needed).
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA-capable device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    MPI_Comm_free(&nodeComm);

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
    const size_t plane = nx * ny;

    // Contiguous block distribution of z planes over MPI ranks.
    const size_t baseCount = (nranks > 0) ? nz / static_cast<size_t>(nranks) : nz;
    const size_t remainder = (nranks > 0) ? nz % static_cast<size_t>(nranks) : 0;
    const size_t rankU = static_cast<size_t>(rank);
    const size_t lnzSize = baseCount + ((rankU < remainder) ? 1 : 0);
    const size_t z0 = rankU * baseCount + std::min(rankU, remainder);
    const int lnz = static_cast<int>(lnzSize);

    // Ranks without work still participate in global collectives, but not in the
    // neighbouring-rank halo exchanges. Build a communicator of active ranks.
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, (lnz > 0) ? 1 : MPI_UNDEFINED, rank, &activeComm);
    int activeRank = 0;
    int activeSize = 0;
    if (activeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(activeComm, &activeRank);
        MPI_Comm_size(activeComm, &activeSize);
    }
    int lower = (activeComm != MPI_COMM_NULL && activeRank > 0) ? activeRank - 1 : MPI_PROC_NULL;
    int upper = (activeComm != MPI_COMM_NULL && activeRank + 1 < activeSize) ? activeRank + 1 : MPI_PROC_NULL;
    const bool atBottom = (z0 == 0);
    const bool atTop = (z0 + lnzSize == nz);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    double* h_halo = nullptr;
    cudaStream_t copyStreamA = nullptr;
    cudaStream_t copyStreamB = nullptr;
    cudaEvent_t halosReady = nullptr;
    const size_t localPlanes = lnzSize + (lnz > 0 ? 2 : 0);
    const size_t localSize = localPlanes * plane;

    if (lnz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), localSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), localSize * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_halo), 4 * plane * sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStreamA, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStreamB, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&halosReady, cudaEventDisableTiming));
    }

    // Initialize concentration field. OpenMP parallelizes the host-side setup.
    if (rank == 0) printf("Initializing concentration field...\n");
    if (lnz > 0) {
        std::vector<double> h_init(localSize, 0.0);
        const size_t vol = gridSize;
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 1; k <= lnz; ++k) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t globalZ = z0 + static_cast<size_t>(k - 1);
                    const size_t linear_id = globalZ * plane + y * nx + x;
                    const double pseudo =
                        ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    h_init[static_cast<size_t>(k) * plane + y * nx + x] =
                        -1.0 + 2.0 * pseudo;
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_cold, h_init.data(), localSize * sizeof(double),
                              cudaMemcpyHostToDevice));

        exchangeHalos(d_cold, lnz, plane, activeComm, lower, upper, atBottom, atTop, h_halo,
                      copyStreamA, copyStreamB, halosReady);
    }

    // CUDA kernels operate on z slabs; each MPI rank owns one local slab.
    const dim3 block(32, 4, 2);
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        if (lnz > 0) {
            const dim3 gridMu((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                              (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                              (static_cast<unsigned>(lnz) + block.z - 1) / block.z);
            computeChemicalPotentialKernel<<<gridMu, block>>>(
                d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), 1, lnz,
                invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());

            exchangeHalos(d_mu, lnz, plane, activeComm, lower, upper, atBottom, atTop, h_halo,
                          copyStreamA, copyStreamB, halosReady);

            const dim3 gridC((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                             (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                             (static_cast<unsigned>(lnz) + block.z - 1) / block.z);
            cahnHilliardUpdateKernel<<<gridC, block>>>(
                d_cnew, d_cold, d_mu, static_cast<int>(nx), static_cast<int>(ny), 1, lnz,
                invdx2, invdy2, invdz2, D, dt);
            CUDA_CHECK(cudaGetLastError());

            exchangeHalos(d_cnew, lnz, plane, activeComm, lower, upper, atBottom, atTop, h_halo,
                          copyStreamA, copyStreamB, halosReady);
        }

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double endTime = MPI_Wtime();

    const double localElapsed = endTime - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = (elapsed > 0.0) ? (cellUpdates / elapsed / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the local interior slab back to the host for gathering / validation.
    const size_t localCount = lnzSize * plane;
    std::vector<double> local;
    if (printResults || validate) {
        local.resize(std::max<size_t>(1, localCount));
        if (localCount > 0) {
            CUDA_CHECK(cudaMemcpy(local.data(), d_cold + plane,
                                  localCount * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    // Print results for external validation (gather all slabs on rank 0).
    if (printResults) {
        std::vector<int> counts(static_cast<size_t>(nranks), 0);
        std::vector<int> displs(static_cast<size_t>(nranks), 0);
        for (int r = 0; r < nranks; ++r) {
            const size_t rr = static_cast<size_t>(r);
            const size_t cnt = baseCount + ((rr < remainder) ? 1 : 0);
            const size_t off = rr * baseCount + std::min(rr, remainder);
            counts[static_cast<size_t>(r)] = static_cast<int>(cnt * plane);
            displs[static_cast<size_t>(r)] = static_cast<int>(off * plane);
        }

        std::vector<double> global;
        if (rank == 0) global.resize(gridSize);
        MPI_Gatherv(local.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    (rank == 0) ? global.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global, "Concentration");
    }

    // Validation (global reductions over all ranks).
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");

        double localMin = HUGE_VAL;
        double localMax = -HUGE_VAL;
        int localBad = 0;
#pragma omp parallel for reduction(min : localMin) reduction(max : localMax) reduction(| : localBad)
        for (size_t i = 0; i < localCount; ++i) {
            const double v = local[i];
            if (std::isnan(v) || std::isinf(v)) localBad = 1;
            localMin = std::min(localMin, v);
            localMax = std::max(localMax, v);
        }

        int globalBad = 0;
        double globalMin = 0.0;
        double globalMax = 0.0;
        MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (globalBad) {
            if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            if (rank == 0) {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                }
            }
            if (globalMax > 10.0 || globalMin < -10.0) valid = false;
        }

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    if (lnz > 0) {
        cudaFree(d_cold);
        cudaFree(d_cnew);
        cudaFree(d_mu);
        cudaFreeHost(h_halo);
        cudaStreamDestroy(copyStreamA);
        cudaStreamDestroy(copyStreamB);
        cudaEventDestroy(halosReady);
    }
    if (activeComm != MPI_COMM_NULL) MPI_Comm_free(&activeComm);

    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
