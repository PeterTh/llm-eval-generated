#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                              \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

// Thread block shape: x is the contiguous dimension (one warp wide)
constexpr int BX = 32;
constexpr int BY = 4;
constexpr int BZ = 2;

// Number of z ghost planes held by each GPU's slab on each side. Two planes of c
// let every GPU compute mu on one ghost plane itself, so only c has to be
// exchanged (once per time step).
constexpr int HALO = 2;

// Multi-GPU decomposition is only worthwhile when every GPU gets enough work
// to hide launch and halo-exchange latency
constexpr size_t MIN_CELLS_PER_GPU = 1600000;
constexpr int MIN_PLANES_PER_GPU = 16;

struct Params {
    int nx, ny, nz;
    double dx2, dy2, dz2;
    double inv_dx2, inv_dy2, inv_dz2;
    double gamma, e_AA, e_BB, e_AB;
    double dtD;
};

// Clamped 7-point Laplacian; the caller supplies the neighbour offsets
// (zero where the clamped neighbour equals the centre point).
// When the squared spacings are powers of two, dividing by them is exactly
// equivalent to multiplying by their (exact) reciprocals, which avoids slow
// FP64 divisions without changing results.
template <bool kPow2, typename I>
__device__ __forceinline__ double laplacian(const double* __restrict__ f, const I idx, const I ox_p,
                                            const I ox_n, const I oy_p, const I oy_n, const I oz_p,
                                            const I oz_n, const Params& p) {
    const double fc = __ldg(f + idx);
    const double sxx = __ldg(f + idx + ox_p) + __ldg(f + idx - ox_n) - 2.0 * fc;
    const double syy = __ldg(f + idx + oy_p) + __ldg(f + idx - oy_n) - 2.0 * fc;
    const double szz = __ldg(f + idx + oz_p) + __ldg(f + idx - oz_n) - 2.0 * fc;
    if constexpr (kPow2) {
        return sxx * p.inv_dx2 + syy * p.inv_dy2 + szz * p.inv_dz2;
    } else {
        return sxx / p.dx2 + syy / p.dy2 + szz / p.dz2;
    }
}

// Common index setup. Computes global z in [zBegin, zEnd); buffers hold global
// planes starting at zBase. Neighbours are clamped against the global domain.
#define STENCIL_SETUP()                                                       \
    const int x = blockIdx.x * BX + threadIdx.x;                              \
    const int y = blockIdx.y * BY + threadIdx.y;                              \
    const int z = zBegin + static_cast<int>(blockIdx.z * BZ + threadIdx.z);   \
    if (x >= p.nx || y >= p.ny || z >= zEnd) return;                          \
    const I sy = static_cast<I>(p.nx);                                        \
    const I sz = static_cast<I>(p.nx) * static_cast<I>(p.ny);                 \
    const I idx = static_cast<I>(z - zBase) * sz + static_cast<I>(y) * sy + x; \
    const I ox_p = (x < p.nx - 1) ? 1 : 0;                                    \
    const I ox_n = (x > 0) ? 1 : 0;                                           \
    const I oy_p = (y < p.ny - 1) ? sy : 0;                                   \
    const I oy_n = (y > 0) ? sy : 0;                                          \
    const I oz_p = (z < p.nz - 1) ? sz : 0;                                   \
    const I oz_n = (z > 0) ? sz : 0;

// Compute chemical potential
template <bool kPow2, typename I>
__global__ void __launch_bounds__(BX * BY * BZ)
computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu, const Params p,
                               const int zBegin, const int zEnd, const int zBase) {
    STENCIL_SETUP();
    const double cv = __ldg(c + idx);
    mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
             + 3.0 * cv + cv * cv * cv
             - p.gamma * laplacian<kPow2>(c, idx, ox_p, ox_n, oy_p, oy_n, oz_p, oz_n, p);
}

// Cahn-Hilliard update step
template <bool kPow2, typename I>
__global__ void __launch_bounds__(BX * BY * BZ)
cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                         const double* __restrict__ mu, const Params p,
                         const int zBegin, const int zEnd, const int zBase) {
    STENCIL_SETUP();
    cnew[idx] = __ldg(cold + idx) + p.dtD * laplacian<kPow2>(mu, idx, ox_p, ox_n, oy_p, oy_n, oz_p, oz_n, p);
}

// Initialize concentration field (cells [first, first + count) of the global grid)
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t first, const size_t count,
                                              const size_t vol) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        // Generate pseudo-random value in [-1, 1]
        const size_t linear_id = first + i;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Per-GPU slab of the domain: owns global planes [z0, z1), stores [lo, hi)
struct Slab {
    int dev = 0;
    int z0 = 0, z1 = 0;
    int lo = 0, hi = 0;
    double* c = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t comm = nullptr;
    cudaEvent_t boundaryDone = nullptr;  // owned boundary planes of cnew updated
    cudaEvent_t sent = nullptr;          // boundary planes staged in host memory
    cudaEvent_t received = nullptr;      // ghost planes of cnew filled from neighbours
    double* sendLo = nullptr;            // pinned staging: planes [z0, z0+HALO)
    double* sendHi = nullptr;            // pinned staging: planes [z1-HALO, z1)
};

class Simulation {
  public:
    Simulation(const Params& p, const int ngpu) : p_(p) {
        planeSize_ = static_cast<size_t>(p.nx) * static_cast<size_t>(p.ny);
        const size_t maxLocal = (static_cast<size_t>(p.nz) / ngpu + 1 + 2 * HALO) * planeSize_;
        smallIndex_ = maxLocal < static_cast<size_t>(0x7fffffff);
        auto isPow2 = [](const double v) {
            int e = 0;
            return std::isfinite(v) && v > 0.0 && std::frexp(v, &e) == 0.5 && e > -1000 && e < 1000;
        };
        pow2_ = isPow2(p.dx2) && isPow2(p.dy2) && isPow2(p.dz2);

        slabs_.resize(ngpu);
        for (int g = 0; g < ngpu; ++g) {
            Slab& s = slabs_[g];
            s.dev = g;
            s.z0 = static_cast<int>(static_cast<long long>(p.nz) * g / ngpu);
            s.z1 = static_cast<int>(static_cast<long long>(p.nz) * (g + 1) / ngpu);
            s.lo = (ngpu > 1) ? std::max(0, s.z0 - HALO) : 0;
            s.hi = (ngpu > 1) ? std::min(p.nz, s.z1 + HALO) : p.nz;
            const size_t n = static_cast<size_t>(s.hi - s.lo) * planeSize_;
            CUDA_CHECK(cudaSetDevice(s.dev));
            CUDA_CHECK(cudaMalloc(&s.c, n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&s.cnew, n * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&s.mu, n * sizeof(double)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
            if (ngpu > 1) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&s.comm, cudaStreamNonBlocking));
                CUDA_CHECK(cudaEventCreateWithFlags(&s.boundaryDone, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&s.sent, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&s.received, cudaEventDisableTiming));
                CUDA_CHECK(cudaMallocHost(&s.sendLo, HALO * planeSize_ * sizeof(double)));
                CUDA_CHECK(cudaMallocHost(&s.sendHi, HALO * planeSize_ * sizeof(double)));
            }
        }
    }

    ~Simulation() {
        if (pairExec_) cudaGraphExecDestroy(pairExec_);
        for (Slab& s : slabs_) {
            cudaSetDevice(s.dev);
            cudaFree(s.c);
            cudaFree(s.cnew);
            cudaFree(s.mu);
            if (s.sendLo) cudaFreeHost(s.sendLo);
            if (s.sendHi) cudaFreeHost(s.sendHi);
            if (s.boundaryDone) cudaEventDestroy(s.boundaryDone);
            if (s.sent) cudaEventDestroy(s.sent);
            if (s.received) cudaEventDestroy(s.received);
            if (s.comm) cudaStreamDestroy(s.comm);
            cudaStreamDestroy(s.compute);
        }
    }

    void initialize() {
        const size_t vol = planeSize_ * static_cast<size_t>(p_.nz);
        for (Slab& s : slabs_) {
            CUDA_CHECK(cudaSetDevice(s.dev));
            int numSMs = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, s.dev));
            const size_t count = static_cast<size_t>(s.hi - s.lo) * planeSize_;
            const size_t blocksNeeded = (count + 255) / 256;
            const unsigned int blocks = static_cast<unsigned int>(
                std::max<size_t>(1, std::min<size_t>(blocksNeeded, static_cast<size_t>(numSMs) * 32)));
            initializeConcentrationKernel<<<blocks, 256, 0, s.compute>>>(s.c, static_cast<size_t>(s.lo) * planeSize_,
                                                                         count, vol);
            CUDA_CHECK(cudaGetLastError());
        }
        // Force (lazy) loading of the stencil kernels on every device outside the timed region
        for (Slab& s : slabs_) {
            CUDA_CHECK(cudaSetDevice(s.dev));
            cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, computeChemicalPotentialKernel<true, int>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, computeChemicalPotentialKernel<true, long long>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, computeChemicalPotentialKernel<false, int>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, computeChemicalPotentialKernel<false, long long>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, cahnHilliardUpdateKernel<true, int>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, cahnHilliardUpdateKernel<true, long long>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, cahnHilliardUpdateKernel<false, int>));
            CUDA_CHECK(cudaFuncGetAttributes(&attr, cahnHilliardUpdateKernel<false, long long>));
        }
        synchronize();
        // Single GPU: capture two time steps (buffers return to their roles)
        // into a CUDA graph to minimise launch overhead.
        if (slabs_.size() == 1) {
            Slab& s = slabs_[0];
            CUDA_CHECK(cudaSetDevice(s.dev));
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(s.compute, cudaStreamCaptureModeThreadLocal));
            enqueueSingle(s, s.cnew, s.c);
            enqueueSingle(s, s.c, s.cnew);
            CUDA_CHECK(cudaStreamEndCapture(s.compute, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&pairExec_, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaGraphUpload(pairExec_, s.compute));
            CUDA_CHECK(cudaStreamSynchronize(s.compute));
        }
    }

    void run(const int iterations) {
        if (slabs_.size() == 1) {
            Slab& s = slabs_[0];
            CUDA_CHECK(cudaSetDevice(s.dev));
            int t = 0;
            for (; t + 1 < iterations; t += 2) {
                CUDA_CHECK(cudaGraphLaunch(pairExec_, s.compute));
            }
            if (t < iterations) {
                enqueueSingle(s, s.cnew, s.c);
                std::swap(s.c, s.cnew);
            }
            CUDA_CHECK(cudaGetLastError());
        } else {
            for (int t = 0; t < iterations; ++t) stepMulti();
        }
        synchronize();
    }

    void download(std::vector<double>& out) {
        for (Slab& s : slabs_) {
            CUDA_CHECK(cudaSetDevice(s.dev));
            const size_t offset = static_cast<size_t>(s.z0 - s.lo) * planeSize_;
            CUDA_CHECK(cudaMemcpy(out.data() + static_cast<size_t>(s.z0) * planeSize_, s.c + offset,
                                  static_cast<size_t>(s.z1 - s.z0) * planeSize_ * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
    }

    void synchronize() {
        for (Slab& s : slabs_) {
            CUDA_CHECK(cudaSetDevice(s.dev));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

  private:
    dim3 gridFor(const int zBegin, const int zEnd) const {
        return dim3(static_cast<unsigned int>((p_.nx + BX - 1) / BX), static_cast<unsigned int>((p_.ny + BY - 1) / BY),
                    static_cast<unsigned int>((zEnd - zBegin + BZ - 1) / BZ));
    }

    template <bool kPow2, typename I>
    void launchMuT(const Slab& s, const double* c, const int zb, const int ze) {
        computeChemicalPotentialKernel<kPow2, I>
            <<<gridFor(zb, ze), dim3(BX, BY, BZ), 0, s.compute>>>(c, s.mu, p_, zb, ze, s.lo);
    }
    template <bool kPow2, typename I>
    void launchUpdateT(const Slab& s, double* cn, const double* co, const int zb, const int ze) {
        cahnHilliardUpdateKernel<kPow2, I>
            <<<gridFor(zb, ze), dim3(BX, BY, BZ), 0, s.compute>>>(cn, co, s.mu, p_, zb, ze, s.lo);
    }
    void launchMu(const Slab& s, const double* c, const int zb, const int ze) {
        if (ze <= zb) return;
        if (pow2_) {
            smallIndex_ ? launchMuT<true, int>(s, c, zb, ze) : launchMuT<true, long long>(s, c, zb, ze);
        } else {
            smallIndex_ ? launchMuT<false, int>(s, c, zb, ze) : launchMuT<false, long long>(s, c, zb, ze);
        }
    }
    void launchUpdate(const Slab& s, double* cn, const double* co, const int zb, const int ze) {
        if (ze <= zb) return;
        if (pow2_) {
            smallIndex_ ? launchUpdateT<true, int>(s, cn, co, zb, ze)
                        : launchUpdateT<true, long long>(s, cn, co, zb, ze);
        } else {
            smallIndex_ ? launchUpdateT<false, int>(s, cn, co, zb, ze)
                        : launchUpdateT<false, long long>(s, cn, co, zb, ze);
        }
    }

    // One time step on a single GPU (whole domain)
    void enqueueSingle(const Slab& s, double* cn, const double* co) {
        launchMu(s, co, 0, p_.nz);
        launchUpdate(s, cn, co, 0, p_.nz);
    }

    double* planePtr(double* base, const Slab& s, const int z) const {
        return base + static_cast<size_t>(z - s.lo) * planeSize_;
    }

    // One time step on multiple GPUs with ghost-plane exchange through pinned
    // host memory, overlapped with the interior update.
    void stepMulti() {
        const int ngpu = static_cast<int>(slabs_.size());
        const size_t haloBytes = HALO * planeSize_ * sizeof(double);
        for (int g = 0; g < ngpu; ++g) {
            Slab& s = slabs_[g];
            CUDA_CHECK(cudaSetDevice(s.dev));
            // mu on owned planes plus one ghost plane on each interior side. Planes
            // whose stencil does not touch ghost planes of c are computed first,
            // overlapping the arrival of the previous step's exchange.
            const int muBegin = std::max(0, s.z0 - 1);
            const int muEnd = std::min(p_.nz, s.z1 + 1);
            const int muInBegin = std::min(muEnd, s.lo == s.z0 ? muBegin : s.z0 + 1);
            const int muInEnd = std::max(muInBegin, s.hi == s.z1 ? muEnd : s.z1 - 1);
            launchMu(s, s.c, muInBegin, muInEnd);
            CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.received, 0));
            launchMu(s, s.c, muBegin, muInBegin);
            launchMu(s, s.c, muInEnd, muEnd);
            // Boundary planes first so they can be shipped while the interior computes
            const bool hasLower = g > 0;
            const bool hasUpper = g + 1 < ngpu;
            const int bLoEnd = hasLower ? s.z0 + HALO : s.z0;
            const int bHiBegin = hasUpper ? s.z1 - HALO : s.z1;
            launchUpdate(s, s.cnew, s.c, s.z0, bLoEnd);
            launchUpdate(s, s.cnew, s.c, bHiBegin, s.z1);
            CUDA_CHECK(cudaEventRecord(s.boundaryDone, s.compute));
            launchUpdate(s, s.cnew, s.c, bLoEnd, bHiBegin);
            CUDA_CHECK(cudaGetLastError());

            // Stage boundary planes to host once neighbours consumed the previous ones
            CUDA_CHECK(cudaStreamWaitEvent(s.comm, s.boundaryDone, 0));
            if (hasLower) CUDA_CHECK(cudaStreamWaitEvent(s.comm, slabs_[g - 1].received, 0));
            if (hasUpper) CUDA_CHECK(cudaStreamWaitEvent(s.comm, slabs_[g + 1].received, 0));
            if (hasLower) {
                CUDA_CHECK(cudaMemcpyAsync(s.sendLo, planePtr(s.cnew, s, s.z0), haloBytes, cudaMemcpyDeviceToHost,
                                           s.comm));
            }
            if (hasUpper) {
                CUDA_CHECK(cudaMemcpyAsync(s.sendHi, planePtr(s.cnew, s, s.z1 - HALO), haloBytes,
                                           cudaMemcpyDeviceToHost, s.comm));
            }
            CUDA_CHECK(cudaEventRecord(s.sent, s.comm));
        }
        for (int g = 0; g < ngpu; ++g) {
            Slab& s = slabs_[g];
            CUDA_CHECK(cudaSetDevice(s.dev));
            if (g > 0) {
                const Slab& lower = slabs_[g - 1];
                CUDA_CHECK(cudaStreamWaitEvent(s.comm, lower.sent, 0));
                CUDA_CHECK(cudaMemcpyAsync(planePtr(s.cnew, s, s.z0 - HALO), lower.sendHi, haloBytes,
                                           cudaMemcpyHostToDevice, s.comm));
            }
            if (g + 1 < ngpu) {
                const Slab& upper = slabs_[g + 1];
                CUDA_CHECK(cudaStreamWaitEvent(s.comm, upper.sent, 0));
                CUDA_CHECK(cudaMemcpyAsync(planePtr(s.cnew, s, s.z1), upper.sendLo, haloBytes,
                                           cudaMemcpyHostToDevice, s.comm));
            }
            CUDA_CHECK(cudaEventRecord(s.received, s.comm));
            std::swap(s.c, s.cnew);
        }
    }

    Params p_;
    size_t planeSize_ = 0;
    bool smallIndex_ = true;
    bool pow2_ = false;
    std::vector<Slab> slabs_;
    cudaGraphExec_t pairExec_ = nullptr;
};

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
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
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    
    Params p;
    p.nx = static_cast<int>(nx);
    p.ny = static_cast<int>(ny);
    p.nz = static_cast<int>(nz);
    p.dx2 = dx * dx;
    p.dy2 = dy * dy;
    p.dz2 = dz * dz;
    p.inv_dx2 = 1.0 / p.dx2;
    p.inv_dy2 = 1.0 / p.dy2;
    p.inv_dz2 = 1.0 / p.dz2;
    p.gamma = gamma;
    p.e_AA = e_AA;
    p.e_BB = e_BB;
    p.e_AB = e_AB;
    p.dtD = dt * D;
    
    // Choose the number of GPUs (z-slab decomposition for large grids)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int ngpu = 1;
    {
        const size_t byWork = gridSize / MIN_CELLS_PER_GPU;
        const size_t byPlanes = nz / MIN_PLANES_PER_GPU;
        ngpu = static_cast<int>(std::max<size_t>(1, std::min<size_t>({static_cast<size_t>(deviceCount), byWork, byPlanes})));
    }
    
    Simulation sim(p, ngpu);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) sim.initialize();
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (gridSize > 0) sim.run(iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    if (gridSize > 0) sim.download(cold);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
