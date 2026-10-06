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

// Hybrid MPI + CUDA + OpenMP implementation.
//  - MPI: 1D slab decomposition of the domain along z; one rank per GPU.
//  - CUDA: all stencil computation happens on the device.
//  - OpenMP: host-side post-processing (validation reductions).
// Each time step exchanges a 2-plane halo of the concentration field, so the
// chemical potential can be computed redundantly on one halo plane and the
// update step needs no further communication. The exchange is overlapped with
// the computation of the chemical potential in the slab interior.

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

constexpr int BX = 32;
constexpr int BY = 8;
constexpr long CHALO = 2; // halo planes of c on each side
constexpr long MHALO = 1; // halo planes of mu on each side

// Squared grid spacings. With RECIP the members hold 1/(d*d) instead of d*d;
// this is only used when 1/(d*d) is exact (power of two), so multiplying by it
// is bitwise identical to dividing by d*d while avoiding slow FP64 divisions.
struct Spacing2 {
    double x, y, z;
};

template <bool RECIP>
__device__ __forceinline__ double scaleBySpacing(const double v, const double h2) {
    if constexpr (RECIP) return v * h2;
    else return v / h2;
}

// Laplacian with clamped boundary conditions. 'p' points to the centre value,
// in-plane clamping uses global x/y, z clamping is resolved by the caller via
// the zp/zn plane offsets (0 at the global boundary).
template <bool RECIP>
__device__ __forceinline__ double laplacian(const double* __restrict__ p, const int x, const int y,
                                            const int nx, const int ny, const long zpOff, const long znOff,
                                            const Spacing2 h2) {
    const int xpOff = (x < nx - 1) ? 1 : 0;
    const int xnOff = (x > 0) ? -1 : 0;
    const int ypOff = (y < ny - 1) ? nx : 0;
    const int ynOff = (y > 0) ? -nx : 0;
    const double cv = __ldg(p);
    const double cxx = scaleBySpacing<RECIP>(__ldg(p + xpOff) + __ldg(p + xnOff) - 2.0 * cv, h2.x);
    const double cyy = scaleBySpacing<RECIP>(__ldg(p + ypOff) + __ldg(p + ynOff) - 2.0 * cv, h2.y);
    const double czz = scaleBySpacing<RECIP>(__ldg(p + zpOff) + __ldg(p + znOff) - 2.0 * cv, h2.z);
    return cxx + cyy + czz;
}

// Compute chemical potential for local planes [k0, k1). Local plane k is
// global plane z0 + k. c has CHALO halo planes, mu has MHALO halo planes.
template <bool RECIP>
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const int nx, const int ny, const long nz, const long z0,
                                        const long k0, const long k1,
                                        const Spacing2 h2, const double gamma, const double e_AA, const double e_BB,
                                        const double e_AB) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long plane = (long)nx * ny;
    for (long k = k0 + blockIdx.z; k < k1; k += gridDim.z) {
        const long gz = z0 + k;
        const long zpOff = (gz < nz - 1) ? plane : 0;
        const long znOff = (gz > 0) ? -plane : 0;
        const long inPlane = (long)y * nx + x;
        const double* p = c + (k + CHALO) * plane + inPlane;
        const double cv = *p;
        // 4.5 * ((cv + 1) * e_AA + (cv - 1) * e_BB - 2 * cv * e_AB) + 3 * cv + cv^3 - gamma * lap,
        // with the fused multiply-adds the host compiler generates for the reference build
        const double energy = fma(-(2.0 * cv), e_AB, fma(cv + 1.0, e_AA, (cv - 1.0) * e_BB));
        const double local = fma(cv * cv, cv, fma(4.5, energy, 3.0 * cv));
        mu[(k + MHALO) * plane + inPlane] =
            fma(-gamma, laplacian<RECIP>(p, x, y, nx, ny, zpOff, znOff, h2), local);
    }
}

// Cahn-Hilliard update for local planes [k0, k1)
template <bool RECIP>
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const int nx, const int ny, const long nz, const long z0,
                                         const long k0, const long k1,
                                         const double D, const double dt, const Spacing2 h2) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long plane = (long)nx * ny;
    for (long k = k0 + blockIdx.z; k < k1; k += gridDim.z) {
        const long gz = z0 + k;
        const long zpOff = (gz < nz - 1) ? plane : 0;
        const long znOff = (gz > 0) ? -plane : 0;
        const long inPlane = (long)y * nx + x;
        const long ci = (k + CHALO) * plane + inPlane;
        cnew[ci] = fma(dt * D,
                       laplacian<RECIP>(mu + (k + MHALO) * plane + inPlane, x, y, nx, ny, zpOff, znOff, h2),
                       cold[ci]);
    }
}

// Initialize local planes of the concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const int nx, const int ny,
                                              const size_t vol, const long z0, const long nzl) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const long plane = (long)nx * ny;
    for (long k = blockIdx.z; k < nzl; k += gridDim.z) {
        // Generate pseudo-random value in [-1, 1]
        const size_t linear_id = idx3(x, y, z0 + k, nx, ny);
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[(k + CHALO) * plane + (long)y * nx + x] = -1.0 + 2.0 * pseudo;
    }
}

static dim3 gridFor(const int nx, const int ny, const long nplanes) {
    return dim3((nx + BX - 1) / BX, (ny + BY - 1) / BY,
                (unsigned)std::max(1L, std::min(nplanes, 65535L)));
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long n = static_cast<long>(c.size());
    const double* data = c.data();

    // Check for NaN or Inf
    bool bad = false;
    #pragma omp parallel for reduction(||: bad) schedule(static)
    for (long i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min: minVal) reduction(max: maxVal) schedule(static)
    for (long i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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
    
    size_t gridSize = nx * ny * nz;
    const long plane = static_cast<long>(nx * ny);
    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const long lnz = static_cast<long>(nz);

    // Bind each rank to a GPU of its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (root) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % ndev));

    // Slab decomposition along z. Every active rank owns at least CHALO planes
    // so that a 2-plane halo always comes from the direct neighbour.
    const int nactive = static_cast<int>(std::max<long>(1, std::min<long>(nranks, lnz / CHALO)));
    const bool active = rank < nactive && gridSize > 0;
    std::vector<int> planeCounts(nranks, 0), planeDispls(nranks, 0);
    for (int r = 0, off = 0; r < nranks; ++r) {
        planeCounts[r] = (r < nactive) ? static_cast<int>(lnz / nactive + (r < lnz % nactive ? 1 : 0)) : 0;
        planeDispls[r] = off;
        off += planeCounts[r];
    }
    const long nzl = active ? planeCounts[rank] : 0;
    const long z0 = planeDispls[rank];
    const int lower = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (active && rank < nactive - 1) ? rank + 1 : MPI_PROC_NULL;

    // Device allocations (c with CHALO halo planes, mu with MHALO halo planes)
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    double* h_send = nullptr; // [lower 2 planes | upper 2 planes]
    double* h_recvBuf = nullptr; // two alternating receive buffers, same layout as h_send
    const size_t haloCount = static_cast<size_t>(CHALO * plane);
    const size_t cBytes = static_cast<size_t>((nzl + 2 * CHALO) * plane) * sizeof(double);
    const size_t muBytes = static_cast<size_t>((nzl + 2 * MHALO) * plane) * sizeof(double);
    cudaStream_t sComp, sComm;
    cudaEvent_t evStepDone, evHaloIn;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&evStepDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHaloIn, cudaEventDisableTiming));
    if (active) {
        CUDA_CHECK(cudaMalloc(&d_cold, cBytes));
        CUDA_CHECK(cudaMalloc(&d_cnew, cBytes));
        CUDA_CHECK(cudaMalloc(&d_mu, muBytes));
        CUDA_CHECK(cudaMemsetAsync(d_cold, 0, cBytes, sComp));
        CUDA_CHECK(cudaMemsetAsync(d_cnew, 0, cBytes, sComp));
        CUDA_CHECK(cudaMemsetAsync(d_mu, 0, muBytes, sComp));
        CUDA_CHECK(cudaMallocHost(&h_send, 2 * haloCount * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_recvBuf, 4 * haloCount * sizeof(double)));
    }
    
    // Initialize concentration field
    if (root) printf("Initializing concentration field...\n");
    if (active) {
        initializeConcentrationKernel<<<gridFor(inx, iny, nzl), dim3(BX, BY), 0, sComp>>>(
            d_cold, inx, iny, gridSize, z0, nzl);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sComp));
    }

    // Plane ranges for the chemical potential: [muLo, muHi) covers the owned
    // planes plus one halo plane towards each existing neighbour.
    const long muLo = (lower != MPI_PROC_NULL) ? -1 : 0;
    const long muHi = (upper != MPI_PROC_NULL) ? nzl + 1 : nzl;
    const bool hasNeighbours = (lower != MPI_PROC_NULL) || (upper != MPI_PROC_NULL);
    // Interior planes only depend on owned planes of c
    const long inLo = hasNeighbours ? std::min(1L, nzl) : 0;
    const long inHi = hasNeighbours ? std::max(inLo, nzl - 1) : nzl;
    // Interior update planes only depend on interior planes of mu
    const long upLo = hasNeighbours ? std::min(2L, nzl) : 0;
    const long upHi = hasNeighbours ? std::max(upLo, nzl - 2) : nzl;

    // Use multiplication by 1/(d*d) when it is exactly equivalent to division
    auto exactReciprocal = [](const double h) {
        int e = 0;
        return std::isnormal(h) && std::frexp(h, &e) == 0.5 && std::isnormal(1.0 / h);
    };
    const Spacing2 h2div{dx * dx, dy * dy, dz * dz};
    const bool useRecip = exactReciprocal(h2div.x) && exactReciprocal(h2div.y) && exactReciprocal(h2div.z);
    const Spacing2 h2 = useRecip ? Spacing2{1.0 / h2div.x, 1.0 / h2div.y, 1.0 / h2div.z} : h2div;

    auto launchMu = [&](const long k0, const long k1) {
        if (k1 <= k0) return;
        const dim3 grid = gridFor(inx, iny, k1 - k0);
        if (useRecip)
            chemicalPotentialKernel<true><<<grid, dim3(BX, BY), 0, sComp>>>(
                d_cold, d_mu, inx, iny, lnz, z0, k0, k1, h2, gamma, e_AA, e_BB, e_AB);
        else
            chemicalPotentialKernel<false><<<grid, dim3(BX, BY), 0, sComp>>>(
                d_cold, d_mu, inx, iny, lnz, z0, k0, k1, h2, gamma, e_AA, e_BB, e_AB);
    };
    auto launchUpdate = [&](const long k0, const long k1) {
        if (k1 <= k0) return;
        const dim3 grid = gridFor(inx, iny, k1 - k0);
        if (useRecip)
            cahnHilliardUpdateKernel<true><<<grid, dim3(BX, BY), 0, sComp>>>(
                d_cnew, d_cold, d_mu, inx, iny, lnz, z0, k0, k1, D, dt, h2);
        else
            cahnHilliardUpdateKernel<false><<<grid, dim3(BX, BY), 0, sComp>>>(
                d_cnew, d_cold, d_mu, inx, iny, lnz, z0, k0, k1, D, dt, h2);
    };
    
    // Run simulation
    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (active) {
        const size_t haloBytes = haloCount * sizeof(double);
        for (int t = 0; t < iterations; ++t) {
            if (hasNeighbours) {
                // Stage the outgoing boundary planes once the previous step is complete
                CUDA_CHECK(cudaEventRecord(evStepDone, sComp));
                CUDA_CHECK(cudaStreamWaitEvent(sComm, evStepDone, 0));
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_send, d_cold + CHALO * plane, haloBytes,
                                               cudaMemcpyDeviceToHost, sComm));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(h_send + haloCount, d_cold + nzl * plane, haloBytes,
                                               cudaMemcpyDeviceToHost, sComm));
            }

            // Chemical potential and update in the interior, overlapping the halo exchange
            launchMu(inLo, inHi);
            launchUpdate(upLo, upHi);

            if (hasNeighbours) {
                // Alternate receive buffers so that new messages never overwrite
                // data the previous step's host-to-device copy may still read
                double* h_recv = h_recvBuf + (t & 1) * 2 * haloCount;
                MPI_Request reqs[4];
                int nreq = 0;
                if (lower != MPI_PROC_NULL)
                    MPI_Irecv(h_recv, static_cast<int>(haloCount), MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &reqs[nreq++]);
                if (upper != MPI_PROC_NULL)
                    MPI_Irecv(h_recv + haloCount, static_cast<int>(haloCount), MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &reqs[nreq++]);
                CUDA_CHECK(cudaStreamSynchronize(sComm));
                if (lower != MPI_PROC_NULL)
                    MPI_Isend(h_send, static_cast<int>(haloCount), MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &reqs[nreq++]);
                if (upper != MPI_PROC_NULL)
                    MPI_Isend(h_send + haloCount, static_cast<int>(haloCount), MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &reqs[nreq++]);
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
                if (lower != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(d_cold, h_recv, haloBytes, cudaMemcpyHostToDevice, sComm));
                if (upper != MPI_PROC_NULL)
                    CUDA_CHECK(cudaMemcpyAsync(d_cold + (nzl + CHALO) * plane, h_recv + haloCount, haloBytes,
                                               cudaMemcpyHostToDevice, sComm));
                CUDA_CHECK(cudaEventRecord(evHaloIn, sComm));
                CUDA_CHECK(cudaStreamWaitEvent(sComp, evHaloIn, 0));

                // Chemical potential on the planes next to (and in) the halo
                launchMu(muLo, inLo);
                launchMu(std::max(inHi, inLo), muHi);

                // Update the planes next to the slab boundaries
                launchUpdate(0, upLo);
                launchUpdate(upHi, nzl);
            }
            
            // Swap buffers
            std::swap(d_cold, d_cnew);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sComp));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());
    
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full field on rank 0
        std::vector<double> localField(static_cast<size_t>(nzl * plane));
        if (active && nzl > 0)
            CUDA_CHECK(cudaMemcpy(localField.data(), d_cold + CHALO * plane, localField.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        std::vector<double> cold(root ? gridSize : 0);
        MPI_Datatype planeType;
        MPI_Type_contiguous(static_cast<int>(plane > 0 ? plane : 1), MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        MPI_Gatherv(localField.data(), static_cast<int>(nzl), planeType, cold.data(), planeCounts.data(),
                    planeDispls.data(), planeType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&planeType);

        if (root) {
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
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (active) {
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFreeHost(h_send));
        CUDA_CHECK(cudaFreeHost(h_recvBuf));
    }
    CUDA_CHECK(cudaEventDestroy(evStepDone));
    CUDA_CHECK(cudaEventDestroy(evHaloIn));
    CUDA_CHECK(cudaStreamDestroy(sComp));
    CUDA_CHECK(cudaStreamDestroy(sComm));
    
    MPI_Finalize();
    return exitCode;
}
