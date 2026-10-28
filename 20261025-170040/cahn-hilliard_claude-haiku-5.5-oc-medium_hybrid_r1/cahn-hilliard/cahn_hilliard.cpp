// Cahn-Hilliard phase separation benchmark.
//
// Hybrid parallelization:
//   * MPI    - 1D slab decomposition along z across ranks (one rank per GPU).
//              Halo planes are exchanged every half-step (c before the chemical
//              potential, mu before the update).
//   * CUDA   - all stencil sweeps run on the device; fields stay resident on the
//              GPU for the whole simulation. Interior planes are computed while
//              halo planes are transferred and exchanged.
//   * OpenMP - host-side work: field initialization and the validation scan.
//
// The slab layout is z-major, so the global linear order used by the original
// code is the concatenation of the per-rank slabs. Results (including the
// printed hash/samples) are gathered in that order and are therefore identical
// in layout to the serial program.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation (z-major)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Parameters shared by the device kernels. Plane indices are local: planes
// 1..nlocal are owned by this rank, planes 0 and nlocal+1 are halo planes.
struct StencilParams {
    size_t nx;
    size_t ny;
    size_t plane;       // nx * ny
    size_t nlocal;      // owned planes on this rank
    bool lowClamp;      // this rank owns global z = 0 (clamped boundary)
    bool highClamp;     // this rank owns global z = nz-1 (clamped boundary)
    double dx2, dy2, dz2;
    double gamma, e_AA, e_BB, e_AB;
    double Dt;          // dt * D
};

// Clamped-boundary Laplacian, same operation order as the original code.
__device__ __forceinline__ double laplacianAt(const double* __restrict__ f, const StencilParams p,
                                              const size_t x, const size_t y, const size_t lz) {
    const size_t xp = (x < p.nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < p.ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (lz == p.nlocal && p.highClamp) ? lz : lz + 1;
    const size_t zn = (lz == 1 && p.lowClamp) ? lz : lz - 1;

    const double* cur = f + lz * p.plane;
    const double* up = f + zp * p.plane;
    const double* dn = f + zn * p.plane;
    const size_t row = y * p.nx;

    const double center = cur[row + x];
    const double cxx = (cur[row + xp] + cur[row + xn] - 2.0 * center) / p.dx2;
    const double cyy = (cur[yp * p.nx + x] + cur[yn * p.nx + x] - 2.0 * center) / p.dy2;
    const double czz = (up[row + x] + dn[row + x] - 2.0 * center) / p.dz2;
    return cxx + cyy + czz;
}

// Chemical potential on planes [lz0, lz0+count).
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const StencilParams p, const size_t lz0, const size_t count) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;
    for (size_t k = blockIdx.z; k < count; k += gridDim.z) {
        const size_t lz = lz0 + k;
        const size_t idx = lz * p.plane + y * p.nx + x;
        const double cv = c[idx];
        const double lap = laplacianAt(c, p, x, y, lz);
        mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - p.gamma * lap;
    }
}

// Cahn-Hilliard update on planes [lz0, lz0+count).
__global__ void cahnHilliardKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                                   double* __restrict__ cnew, const StencilParams p,
                                   const size_t lz0, const size_t count) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.nx || y >= p.ny) return;
    for (size_t k = blockIdx.z; k < count; k += gridDim.z) {
        const size_t lz = lz0 + k;
        const size_t idx = lz * p.plane + y * p.nx + x;
        cnew[idx] = cold[idx] + p.Dt * laplacianAt(mu, p, x, y, lz);
    }
}

void launchChemicalPotential(cudaStream_t s, const double* c, double* mu, const StencilParams& p,
                             size_t lz0, size_t count) {
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((p.nx + block.x - 1) / block.x),
                    static_cast<unsigned>((p.ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(count, 65535)));
    chemicalPotentialKernel<<<grid, block, 0, s>>>(c, mu, p, lz0, count);
    CUDA_CHECK(cudaGetLastError());
}

void launchCahnHilliard(cudaStream_t s, const double* cold, const double* mu, double* cnew,
                        const StencilParams& p, size_t lz0, size_t count) {
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((p.nx + block.x - 1) / block.x),
                    static_cast<unsigned>((p.ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(count, 65535)));
    cahnHilliardKernel<<<grid, block, 0, s>>>(cold, mu, cnew, p, lz0, count);
    CUDA_CHECK(cudaGetLastError());
}

// Planes [z0, z0+nl) owned by rank r; nz is split as evenly as possible.
void planeRange(const size_t r, const size_t nz, const size_t nActive, size_t& z0, size_t& nl) {
    if (r >= nActive) {
        z0 = 0;
        nl = 0;
        return;
    }
    const size_t base = nz / nActive;
    const size_t rem = nz % nActive;
    nl = base + (r < rem ? 1 : 0);
    z0 = r * base + std::min(r, rem);
}

// Validation scan (OpenMP). Chunks are combined in order, which preserves the
// first-occurrence semantics of the sequential min/max scan exactly.
bool validateResult(const std::vector<double>& c) {
    if (c.empty()) return true;

    struct Partial {
        double lo = 0.0;
        double hi = 0.0;
        bool hasValue = false;
        bool bad = false;
    };

    const size_t n = c.size();
    const long long nChunks = std::max(1, omp_get_max_threads());
    std::vector<Partial> part(static_cast<size_t>(nChunks));

    #pragma omp parallel for schedule(static)
    for (long long ch = 0; ch < nChunks; ++ch) {
        const size_t b = n * static_cast<size_t>(ch) / static_cast<size_t>(nChunks);
        const size_t e = n * static_cast<size_t>(ch + 1) / static_cast<size_t>(nChunks);
        Partial p;
        for (size_t i = b; i < e; ++i) {
            const double v = c[i];
            if (std::isnan(v) || std::isinf(v)) {
                p.bad = true;
                break;
            }
            if (!p.hasValue) {
                p.lo = v;
                p.hi = v;
                p.hasValue = true;
            } else {
                if (v < p.lo) p.lo = v;
                if (v > p.hi) p.hi = v;
            }
        }
        part[static_cast<size_t>(ch)] = p;
    }

    for (const auto& p : part) {
        if (p.bad) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    bool have = false;
    double minVal = 0.0;
    double maxVal = 0.0;
    for (const auto& p : part) {
        if (!p.hasValue) continue;
        if (!have) {
            minVal = p.lo;
            maxVal = p.hi;
            have = true;
        } else {
            if (p.lo < minVal) minVal = p.lo;
            if (p.hi > maxVal) maxVal = p.hi;
        }
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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

int run(int argc, char** argv, int rank, int size) {
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    if (plane > static_cast<size_t>(INT_MAX) || nz > static_cast<size_t>(INT_MAX)) {
        if (root) fprintf(stderr, "Grid too large for MPI datatype counts\n");
        return 1;
    }

    // Device selection: one GPU per rank, round-robin over the GPUs of the node.
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    }

    // Domain decomposition along z. Ranks beyond the number of planes idle.
    const size_t nActive = (plane == 0) ? 0 : std::min<size_t>(static_cast<size_t>(size), nz);
    size_t z0 = 0;
    size_t nl = 0;
    planeRange(static_cast<size_t>(rank), nz, nActive, z0, nl);
    const bool hasLower = (nl > 0) && rank > 0;
    const bool hasUpper = (nl > 0) && static_cast<size_t>(rank) + 1 < nActive;
    const int lowerRank = hasLower ? rank - 1 : MPI_PROC_NULL;
    const int upperRank = hasUpper ? rank + 1 : MPI_PROC_NULL;

    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(plane), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // Host copy of this rank's owned planes (used for initialization and gather)
    std::vector<double> owned(nl * plane);

    if (root) printf("Initializing concentration field...\n");
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long lzi = 0; lzi < static_cast<long long>(nl); ++lzi) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t lz = static_cast<size_t>(lzi);
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = idx3(x, y, z0 + lz, nx, ny);
                // Pseudo-random value in [-1, 1]
                const double pseudo = ((((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize));
                owned[lz * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Device state (planes 0 and nl+1 are halo planes)
    double* dCold = nullptr;
    double* dCnew = nullptr;
    double* dMu = nullptr;
    double* hHalo = nullptr;  // pinned staging buffers: 4 c-planes then 4 mu-planes
    cudaStream_t sCompute = nullptr;
    cudaStream_t sComm = nullptr;
    cudaEvent_t eReady = nullptr;   // cold field (owned planes) is up to date
    cudaEvent_t eCD2H = nullptr;
    cudaEvent_t eCHalo = nullptr;   // cold halo planes are on the device
    cudaEvent_t eMu = nullptr;      // mu boundary planes are computed
    cudaEvent_t eMuD2H = nullptr;
    cudaEvent_t eMuHalo = nullptr;  // mu halo planes are on the device

    const size_t planeBytes = plane * sizeof(double);
    if (nl > 0) {
        const size_t bytes = (nl + 2) * planeBytes;
        CUDA_CHECK(cudaMalloc(&dCold, bytes));
        CUDA_CHECK(cudaMalloc(&dCnew, bytes));
        CUDA_CHECK(cudaMalloc(&dMu, bytes));
        CUDA_CHECK(cudaMemset(dCold, 0, bytes));
        CUDA_CHECK(cudaMemset(dCnew, 0, bytes));
        CUDA_CHECK(cudaMemset(dMu, 0, bytes));
        CUDA_CHECK(cudaMemcpy(dCold + plane, owned.data(), nl * planeBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMallocHost(&hHalo, 8 * planeBytes));

        CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
        const unsigned int evFlags = cudaEventDisableTiming;
        CUDA_CHECK(cudaEventCreateWithFlags(&eReady, evFlags));
        CUDA_CHECK(cudaEventCreateWithFlags(&eCD2H, evFlags));
        CUDA_CHECK(cudaEventCreateWithFlags(&eCHalo, evFlags));
        CUDA_CHECK(cudaEventCreateWithFlags(&eMu, evFlags));
        CUDA_CHECK(cudaEventCreateWithFlags(&eMuD2H, evFlags));
        CUDA_CHECK(cudaEventCreateWithFlags(&eMuHalo, evFlags));
        CUDA_CHECK(cudaEventRecord(eReady, sCompute));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    StencilParams params{};
    params.nx = nx;
    params.ny = ny;
    params.plane = plane;
    params.nlocal = nl;
    params.lowClamp = (rank == 0);
    params.highClamp = (nActive > 0) && (static_cast<size_t>(rank) + 1 == nActive);
    params.dx2 = dx * dx;
    params.dy2 = dy * dy;
    params.dz2 = dz * dz;
    params.gamma = gamma;
    params.e_AA = e_AA;
    params.e_BB = e_BB;
    params.e_AB = e_AB;
    params.Dt = dt * D;

    double* hCLoSend = hHalo + 0 * plane;
    double* hCHiSend = hHalo + 1 * plane;
    double* hCLoRecv = hHalo + 2 * plane;
    double* hCHiRecv = hHalo + 3 * plane;
    double* hMuLoSend = hHalo + 4 * plane;
    double* hMuHiSend = hHalo + 5 * plane;
    double* hMuLoRecv = hHalo + 6 * plane;
    double* hMuHiRecv = hHalo + 7 * plane;

    // Exchange one plane in each direction with the z-neighbours (MPI_PROC_NULL at the ends).
    auto exchangeHalos = [&](const double* loSend, const double* hiSend, double* loRecv, double* hiRecv) {
        MPI_Request req[4];
        MPI_Irecv(loRecv, 1, planeType, lowerRank, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(hiRecv, 1, planeType, upperRank, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(hiSend, 1, planeType, upperRank, 0, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(loSend, 1, planeType, lowerRank, 1, MPI_COMM_WORLD, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    };

    if (root) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    if (nl > 0) {
        for (int t = 0; t < iterations; ++t) {
            // (1) Interior chemical potential needs no halo data; overlap with the c halo exchange.
            if (nl > 2) launchChemicalPotential(sCompute, dCold, dMu, params, 2, nl - 2);

            // (2) Exchange boundary planes of c.
            CUDA_CHECK(cudaStreamWaitEvent(sComm, eReady, 0));
            if (hasLower) {
                CUDA_CHECK(cudaMemcpyAsync(hCLoSend, dCold + 1 * plane, planeBytes, cudaMemcpyDeviceToHost, sComm));
            }
            if (hasUpper) {
                CUDA_CHECK(cudaMemcpyAsync(hCHiSend, dCold + nl * plane, planeBytes, cudaMemcpyDeviceToHost, sComm));
            }
            CUDA_CHECK(cudaEventRecord(eCD2H, sComm));
            CUDA_CHECK(cudaEventSynchronize(eCD2H));

            exchangeHalos(hCLoSend, hCHiSend, hCLoRecv, hCHiRecv);

            if (hasLower) {
                CUDA_CHECK(cudaMemcpyAsync(dCold + 0 * plane, hCLoRecv, planeBytes, cudaMemcpyHostToDevice, sComm));
            }
            if (hasUpper) {
                CUDA_CHECK(cudaMemcpyAsync(dCold + (nl + 1) * plane, hCHiRecv, planeBytes, cudaMemcpyHostToDevice, sComm));
            }
            CUDA_CHECK(cudaEventRecord(eCHalo, sComm));

            // (3) Chemical potential on the boundary planes (needs c halos).
            CUDA_CHECK(cudaStreamWaitEvent(sCompute, eCHalo, 0));
            launchChemicalPotential(sCompute, dCold, dMu, params, 1, 1);
            if (nl > 1) launchChemicalPotential(sCompute, dCold, dMu, params, nl, 1);
            CUDA_CHECK(cudaEventRecord(eMu, sCompute));

            // (4) Send boundary planes of mu; update interior planes while exchanging.
            CUDA_CHECK(cudaStreamWaitEvent(sComm, eMu, 0));
            if (hasLower) {
                CUDA_CHECK(cudaMemcpyAsync(hMuLoSend, dMu + 1 * plane, planeBytes, cudaMemcpyDeviceToHost, sComm));
            }
            if (hasUpper) {
                CUDA_CHECK(cudaMemcpyAsync(hMuHiSend, dMu + nl * plane, planeBytes, cudaMemcpyDeviceToHost, sComm));
            }
            CUDA_CHECK(cudaEventRecord(eMuD2H, sComm));

            if (nl > 2) launchCahnHilliard(sCompute, dCold, dMu, dCnew, params, 2, nl - 2);

            CUDA_CHECK(cudaEventSynchronize(eMuD2H));
            exchangeHalos(hMuLoSend, hMuHiSend, hMuLoRecv, hMuHiRecv);

            if (hasLower) {
                CUDA_CHECK(cudaMemcpyAsync(dMu + 0 * plane, hMuLoRecv, planeBytes, cudaMemcpyHostToDevice, sComm));
            }
            if (hasUpper) {
                CUDA_CHECK(cudaMemcpyAsync(dMu + (nl + 1) * plane, hMuHiRecv, planeBytes, cudaMemcpyHostToDevice, sComm));
            }
            CUDA_CHECK(cudaEventRecord(eMuHalo, sComm));

            // (5) Update boundary planes (needs mu halos).
            CUDA_CHECK(cudaStreamWaitEvent(sCompute, eMuHalo, 0));
            launchCahnHilliard(sCompute, dCold, dMu, dCnew, params, 1, 1);
            if (nl > 1) launchCahnHilliard(sCompute, dCold, dMu, dCnew, params, nl, 1);
            CUDA_CHECK(cudaEventRecord(eReady, sCompute));

            std::swap(dCold, dCnew);
        }
    }

    if (nl > 0) CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - tStart;

    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (static_cast<double>(ms) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field in global (z-major) order on the root.
    const bool needFull = printResults || validate;
    std::vector<double> full;
    if (needFull) {
        if (nl > 0) {
            CUDA_CHECK(cudaMemcpy(owned.data(), dCold + plane, nl * planeBytes, cudaMemcpyDeviceToHost));
        }
        if (root) full.resize(gridSize);

        std::vector<int> counts(static_cast<size_t>(size));
        std::vector<int> displs(static_cast<size_t>(size));
        for (int r = 0; r < size; ++r) {
            size_t zr = 0;
            size_t nr = 0;
            planeRange(static_cast<size_t>(r), nz, nActive, zr, nr);
            counts[static_cast<size_t>(r)] = static_cast<int>(nr);
            displs[static_cast<size_t>(r)] = static_cast<int>(zr);
        }
        MPI_Gatherv(owned.data(), static_cast<int>(nl), planeType,
                    full.data(), counts.data(), displs.data(), planeType, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(full, "Concentration");
    }

    int exitCode = 0;
    if (validate) {
        int valid = 1;
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(full) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (root) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    // Release resources
    if (nl > 0) {
        CUDA_CHECK(cudaEventDestroy(eReady));
        CUDA_CHECK(cudaEventDestroy(eCD2H));
        CUDA_CHECK(cudaEventDestroy(eCHalo));
        CUDA_CHECK(cudaEventDestroy(eMu));
        CUDA_CHECK(cudaEventDestroy(eMuD2H));
        CUDA_CHECK(cudaEventDestroy(eMuHalo));
        CUDA_CHECK(cudaStreamDestroy(sCompute));
        CUDA_CHECK(cudaStreamDestroy(sComm));
        CUDA_CHECK(cudaFreeHost(hHalo));
        CUDA_CHECK(cudaFree(dCold));
        CUDA_CHECK(cudaFree(dCnew));
        CUDA_CHECK(cudaFree(dMu));
    }
    MPI_Type_free(&planeType);

    return exitCode;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = run(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
