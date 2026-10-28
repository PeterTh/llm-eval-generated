// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization:
//   MPI    - 1D slab decomposition along z, one rank per GPU. Each rank owns a
//            contiguous block of z-planes (the z-major layout makes each slab a
//            contiguous memory block) and exchanges one-plane halos with its
//            neighbours every time step.
//   CUDA   - the stencil kernels (chemical potential and concentration update)
//            run on the rank's GPU. Interior planes are computed concurrently
//            with the halo exchanges (compute and communication streams).
//   OpenMP - host-side work: parallel field initialization and the validation
//            scan (min/max/non-finite check; these are order-independent, so the
//            results are identical to a serial scan).
//
// The per-cell arithmetic and evaluation order match the original serial
// formulation, including the fused multiply-adds the reference build produces
// (written out explicitly with fma()), so the final field and its hash are
// bit-identical to the original and independent of the number of ranks.

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr size_t kMaxGridZ = 65535;

// Physical / numerical parameters shared by host and device.
struct StencilParams {
    int nx;
    int ny;
    double dx2;     // dx * dx
    double dy2;     // dy * dy
    double dz2;     // dz * dz
    double gamma;
    double e_AA;
    double e_BB;
    double e_AB;
    double dtD;     // dt * D
};

// Laplacian with the same arithmetic as the reference formulation.
__device__ __forceinline__ double laplacian(double c, double xp, double xn, double yp, double yn,
                                            double zp, double zn, const StencilParams& p) {
    const double cxx = (xp + xn - 2.0 * c) / p.dx2;
    const double cyy = (yp + yn - 2.0 * c) / p.dy2;
    const double czz = (zp + zn - 2.0 * c) / p.dz2;
    return cxx + cyy + czz;
}

// Local chemical potential f'(c) - gamma * lap(c) without the Laplacian term.
// The multiply-adds are written as explicit fma() calls, mirroring the fused
// operations of the reference formulation, so results are bit-identical to it.
__device__ __forceinline__ double chemicalPotentialFromLap(double cv, double lap, const StencilParams& p) {
    const double a1 = fma(cv + 1.0, p.e_AA, (cv - 1.0) * p.e_BB);
    const double a = fma(-(2.0 * cv), p.e_AB, a1);
    const double m1 = fma(4.5, a, 3.0 * cv);
    const double m2 = fma(cv * cv, cv, m1);
    return fma(-p.gamma, lap, m2);
}

// Chemical potential mu(c) = f'(c) - gamma * lap(c).
// Buffers are laid out with one ghost plane below (plane 0) and above (plane nzl+1),
// so the z-neighbours are always available; x/y clamping is applied in-plane.
// Plane k of a buffer starts at element k * (nx*ny).
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const StencilParams p, const size_t zBegin, const size_t zCount) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const size_t P = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t rowC = static_cast<size_t>(y) * p.nx;
    const size_t rowYp = static_cast<size_t>(y < p.ny - 1 ? y + 1 : y) * p.nx;
    const size_t rowYn = static_cast<size_t>(y > 0 ? y - 1 : 0) * p.nx;
    const size_t xp = static_cast<size_t>(x < p.nx - 1 ? x + 1 : x);
    const size_t xn = static_cast<size_t>(x > 0 ? x - 1 : 0);
    const size_t xu = static_cast<size_t>(x);

    for (size_t k = blockIdx.z; k < zCount; k += gridDim.z) {
        const size_t base = (zBegin + k) * P;
        const size_t i = base + rowC + xu;
        const double cv = c[i];
        const double lap = laplacian(cv,
                                     c[base + rowC + xp], c[base + rowC + xn],
                                     c[base + rowYp + xu], c[base + rowYn + xu],
                                     c[i + P], c[i - P], p);
        mu[i] = chemicalPotentialFromLap(cv, lap, p);
    }
}

// Concentration update: cnew = cold + dt * D * lap(mu).
__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                             double* __restrict__ cnew, const StencilParams p,
                             const size_t zBegin, const size_t zCount) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;

    const size_t P = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
    const size_t rowC = static_cast<size_t>(y) * p.nx;
    const size_t rowYp = static_cast<size_t>(y < p.ny - 1 ? y + 1 : y) * p.nx;
    const size_t rowYn = static_cast<size_t>(y > 0 ? y - 1 : 0) * p.nx;
    const size_t xp = static_cast<size_t>(x < p.nx - 1 ? x + 1 : x);
    const size_t xn = static_cast<size_t>(x > 0 ? x - 1 : 0);
    const size_t xu = static_cast<size_t>(x);

    for (size_t k = blockIdx.z; k < zCount; k += gridDim.z) {
        const size_t base = (zBegin + k) * P;
        const size_t i = base + rowC + xu;
        const double lap = laplacian(mu[i],
                                     mu[base + rowC + xp], mu[base + rowC + xn],
                                     mu[base + rowYp + xu], mu[base + rowYn + xu],
                                     mu[i + P], mu[i - P], p);
        cnew[i] = fma(p.dtD, lap, cold[i]);
    }
}

dim3 gridFor(const StencilParams& p, size_t zCount) {
    return dim3(static_cast<unsigned>((p.nx + kBlockX - 1) / kBlockX),
                static_cast<unsigned>((p.ny + kBlockY - 1) / kBlockY),
                static_cast<unsigned>(std::min(zCount, kMaxGridZ)));
}

// Launch helpers operate on buffer planes [zBegin, zBegin + zCount).
void launchChemicalPotential(const double* c, double* mu, const StencilParams& p,
                             size_t zBegin, size_t zCount, cudaStream_t s) {
    if (zCount == 0) return;
    chemicalPotentialKernel<<<gridFor(p, zCount), dim3(kBlockX, kBlockY), 0, s>>>(c, mu, p, zBegin, zCount);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdate(const double* cold, const double* mu, double* cnew, const StencilParams& p,
                  size_t zBegin, size_t zCount, cudaStream_t s) {
    if (zCount == 0) return;
    updateKernel<<<gridFor(p, zCount), dim3(kBlockX, kBlockY), 0, s>>>(cold, mu, cnew, p, zBegin, zCount);
    CUDA_CHECK(cudaGetLastError());
}

// Per-rank slab description.
struct Slab {
    size_t nzl = 0;                      // owned planes on this rank
    size_t zStart = 0;                   // global index of first owned plane
    size_t P = 0;                        // plane size (nx * ny)
    int lower = MPI_PROC_NULL;           // neighbour holding lower z planes
    int upper = MPI_PROC_NULL;           // neighbour holding upper z planes
    MPI_Datatype planeType = MPI_DATATYPE_NULL;
};

// Pinned host staging buffers for one halo exchange.
struct HaloStage {
    double* sendLo = nullptr;
    double* sendHi = nullptr;
    double* recvLo = nullptr;
    double* recvHi = nullptr;
};

void allocHaloStage(HaloStage& h, size_t P) {
    const size_t bytes = P * sizeof(double);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h.sendLo), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h.sendHi), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h.recvLo), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h.recvHi), bytes));
}

void freeHaloStage(HaloStage& h) {
    cudaFreeHost(h.sendLo);
    cudaFreeHost(h.sendHi);
    cudaFreeHost(h.recvLo);
    cudaFreeHost(h.recvHi);
}

// Exchange one-plane halos of a ghosted device field (layout: plane 0 = lower ghost,
// planes 1..nzl = owned, plane nzl+1 = upper ghost). Waits until the owned boundary
// planes are produced (bndReady), then fills the ghost planes and records ghostDone.
// Physical domain boundaries use a copy of the own boundary plane, which reproduces
// the clamped (replicated) boundary condition exactly.
void exchangeHalo(double* d_field, const Slab& s, HaloStage& h,
                  cudaEvent_t bndReady, cudaEvent_t ghostDone, cudaStream_t comm) {
    const size_t bytes = s.P * sizeof(double);
    CUDA_CHECK(cudaStreamWaitEvent(comm, bndReady, 0));
    CUDA_CHECK(cudaMemcpyAsync(h.sendLo, d_field + s.P, bytes, cudaMemcpyDeviceToHost, comm));
    CUDA_CHECK(cudaMemcpyAsync(h.sendHi, d_field + s.nzl * s.P, bytes, cudaMemcpyDeviceToHost, comm));
    CUDA_CHECK(cudaStreamSynchronize(comm));

    MPI_Sendrecv(h.sendHi, 1, s.planeType, s.upper, 0,
                 h.recvLo, 1, s.planeType, s.lower, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h.sendLo, 1, s.planeType, s.lower, 1,
                 h.recvHi, 1, s.planeType, s.upper, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (s.lower == MPI_PROC_NULL) std::memcpy(h.recvLo, h.sendLo, bytes);
    if (s.upper == MPI_PROC_NULL) std::memcpy(h.recvHi, h.sendHi, bytes);

    CUDA_CHECK(cudaMemcpyAsync(d_field, h.recvLo, bytes, cudaMemcpyHostToDevice, comm));
    CUDA_CHECK(cudaMemcpyAsync(d_field + (s.nzl + 1) * s.P, h.recvHi, bytes, cudaMemcpyHostToDevice, comm));
    CUDA_CHECK(cudaEventRecord(ghostDone, comm));
}

// Initial concentration field for owned planes (parallel over host threads).
void initializeSlab(double* out, size_t nx, size_t ny, size_t vol, size_t zStart, size_t nzl) {
    const size_t P = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Pseudo-random value in [-1, 1], based on the global linear index.
                const size_t linear_id = (zStart + z) * P + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                out[z * P + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    if (c.empty()) return true;

    // Check for NaN or Inf
    int nonFinite = 0;
    double minVal = c[0];
    double maxVal = c[0];
    const long long n = static_cast<long long>(c.size());
    #pragma omp parallel for reduction(|: nonFinite) reduction(min: minVal) reduction(max: maxVal) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const double val = c[static_cast<size_t>(i)];
        nonFinite |= (std::isnan(val) || std::isinf(val)) ? 1 : 0;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (nonFinite) {
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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

    const size_t P = nx * ny;
    const size_t gridSize = P * nz;

    // Device selection: one GPU per rank, round-robin over the devices on the node.
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    // Domain decomposition along z. Ranks beyond nz receive no planes.
    if (P > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Plane size too large for MPI counts\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t active = (P > 0) ? std::min<size_t>(static_cast<size_t>(size), nz) : 0;
    auto countOf = [&](size_t r) -> size_t {
        if (r >= active) return 0;
        return nz / active + ((r < nz % active) ? 1 : 0);
    };
    std::vector<int> counts(size), displs(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(countOf(static_cast<size_t>(r)));
        displs[r] = static_cast<int>(offset);
        offset += static_cast<size_t>(counts[r]);
    }

    Slab slab;
    slab.nzl = static_cast<size_t>(counts[rank]);
    slab.zStart = static_cast<size_t>(displs[rank]);
    slab.P = P;
    if (slab.nzl > 0) {
        if (rank > 0) slab.lower = rank - 1;
        if (rank + 1 < size && counts[rank + 1] > 0) slab.upper = rank + 1;
    }
    MPI_Type_contiguous(static_cast<int>(P), MPI_DOUBLE, &slab.planeType);
    MPI_Type_commit(&slab.planeType);

    StencilParams sp{};
    sp.nx = static_cast<int>(nx);
    sp.ny = static_cast<int>(ny);
    sp.dx2 = dx * dx;
    sp.dy2 = dy * dy;
    sp.dz2 = dz * dz;
    sp.gamma = gamma;
    sp.e_AA = e_AA;
    sp.e_BB = e_BB;
    sp.e_AB = e_AB;
    sp.dtD = dt * D;

    // Device state (only ranks owning planes allocate).
    double* d_c = nullptr;      // current concentration (ghosted)
    double* d_cn = nullptr;     // next concentration (ghosted)
    double* d_mu = nullptr;     // chemical potential (ghosted)
    cudaStream_t sComp = nullptr;
    cudaStream_t sComm = nullptr;
    cudaEvent_t evCBnd = nullptr, evCGhost = nullptr, evMuBnd = nullptr, evMuGhost = nullptr;
    HaloStage hC{}, hMu{};

    if (root) printf("Initializing concentration field...\n");
    if (slab.nzl > 0) {
        const size_t fieldElems = (slab.nzl + 2) * P;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), fieldElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cn), fieldElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), fieldElems * sizeof(double)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evCBnd, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evCGhost, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMuBnd, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evMuGhost, cudaEventDisableTiming));
        allocHaloStage(hC, P);
        allocHaloStage(hMu, P);

        // Initialize owned planes on the host (OpenMP) and upload.
        std::vector<double> init(slab.nzl * P);
        initializeSlab(init.data(), nx, ny, gridSize, slab.zStart, slab.nzl);
        // The upload is issued on the compute stream and completed before the
        // halo exchange starts (the comm stream is not ordered after the legacy
        // default stream, and pageable copies may return before the DMA finishes).
        CUDA_CHECK(cudaMemcpyAsync(d_c + P, init.data(), slab.nzl * P * sizeof(double),
                                   cudaMemcpyHostToDevice, sComp));
        CUDA_CHECK(cudaStreamSynchronize(sComp));
        CUDA_CHECK(cudaEventRecord(evCBnd, sComp));
        // Fill the initial ghost planes.
        exchangeHalo(d_c, slab, hC, evCBnd, evCGhost, sComm);
    }

    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (slab.nzl > 0) {
        double* cur = d_c;
        double* nxt = d_cn;
        const size_t nzl = slab.nzl;
        // Start the interior chemical potential for the first step.
        if (iterations > 0 && slab.nzl > 2) {
            launchChemicalPotential(d_c, d_mu, sp, 2, slab.nzl - 2, sComp);
        }
        for (int t = 0; t < iterations; ++t) {
            // (1) chemical potential on the boundary planes of the current field.
            CUDA_CHECK(cudaStreamWaitEvent(sComp, evCGhost, 0));
            launchChemicalPotential(cur, d_mu, sp, 1, 1, sComp);
            if (nzl > 1) launchChemicalPotential(cur, d_mu, sp, nzl, 1, sComp);
            CUDA_CHECK(cudaEventRecord(evMuBnd, sComp));

            // (2) concentration update on interior planes (needs only owned mu);
            //     overlaps with the mu halo exchange below.
            if (nzl > 2) launchUpdate(cur, d_mu, nxt, sp, 2, nzl - 2, sComp);

            // (3) mu halo exchange.
            exchangeHalo(d_mu, slab, hMu, evMuBnd, evMuGhost, sComm);

            // (4) concentration update on boundary planes.
            CUDA_CHECK(cudaStreamWaitEvent(sComp, evMuGhost, 0));
            launchUpdate(cur, d_mu, nxt, sp, 1, 1, sComp);
            if (nzl > 1) launchUpdate(cur, d_mu, nxt, sp, nzl, 1, sComp);
            CUDA_CHECK(cudaEventRecord(evCBnd, sComp));

            if (t + 1 < iterations) {
                // (5) interior chemical potential of the next field; overlaps with
                //     the concentration halo exchange below.
                if (nzl > 2) launchChemicalPotential(nxt, d_mu, sp, 2, nzl - 2, sComp);
                // (6) concentration halo exchange for the next step.
                exchangeHalo(nxt, slab, hC, evCBnd, evCGhost, sComm);
            }
            std::swap(cur, nxt);
        }
        CUDA_CHECK(cudaStreamSynchronize(sComp));
        CUDA_CHECK(cudaStreamSynchronize(sComm));
        // Final field is in `cur`; keep the pointer for output below.
        d_c = cur;
    }

    auto end = std::chrono::high_resolution_clock::now();
    const long long localUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long maxUs = 0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const long msElapsed = static_cast<long>(maxUs / 1000);

    if (root) {
        printf("Computation time: %ld ms\n", msElapsed);

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (msElapsed / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field on rank 0 (only needed for output / validation).
    const bool needField = printResults || validate;
    std::vector<double> full;
    if (needField) {
        std::vector<double> local(slab.nzl * P);
        if (slab.nzl > 0) {
            CUDA_CHECK(cudaMemcpy(local.data(), d_c + P, slab.nzl * P * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (root) full.resize(gridSize);
        MPI_Gatherv(local.data(), static_cast<int>(slab.nzl), slab.planeType,
                    root ? full.data() : nullptr, counts.data(), displs.data(), slab.planeType,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (root && printResults) {
        print_results(full, "Concentration");
    }

    // Validation
    if (validate && root) {
        printf("Validating result...\n");
        bool valid = validateResult(full);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    if (slab.nzl > 0) {
        cudaFree(d_c);
        cudaFree(d_cn);
        cudaFree(d_mu);
        freeHaloStage(hC);
        freeHaloStage(hMu);
        cudaEventDestroy(evCBnd);
        cudaEventDestroy(evCGhost);
        cudaEventDestroy(evMuBnd);
        cudaEventDestroy(evMuGhost);
        cudaStreamDestroy(sComp);
        cudaStreamDestroy(sComm);
    }
    MPI_Type_free(&slab.planeType);
    MPI_Finalize();
    return exitCode;
}
