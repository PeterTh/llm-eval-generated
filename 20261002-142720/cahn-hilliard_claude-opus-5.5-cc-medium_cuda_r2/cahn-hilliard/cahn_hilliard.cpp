#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cctype>
#include <string>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error checking
#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                             \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// Physical parameters (compile-time constants so divisions by dx*dx etc. fold away)
constexpr double kDx = 1.0;
constexpr double kDy = 1.0;
constexpr double kDz = 1.0;
constexpr double kDt = 0.01;
constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = (2.0 / 9.0);
constexpr double kGamma = 0.5;
constexpr double kD = 1.0;

// The bulk free-energy derivative 4.5*((c+1)*e_AA + (c-1)*e_BB - 2*c*e_AB) + 3*c + c^3 is
// affine in c apart from the cubic term; pre-combine its coefficients at compile time.
constexpr double kMuLinear = 4.5 * (kEAA + kEBB - 2.0 * kEAB) + 3.0;
constexpr double kMuConst = 4.5 * (kEAA - kEBB);

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

// Each GPU owns a slab of z-planes plus kHalo ghost planes of the concentration field on each
// side. The chemical potential is recomputed one plane into the ghost region, so only a
// single halo exchange of the concentration field is needed per time step.
constexpr int kHalo = 2;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian with clamped boundary conditions. The center value and its z-neighbours are
// supplied by the caller (kept in registers while marching along z); x/y neighbours are
// read from global memory (served mostly by L1/L2).
__device__ __forceinline__ double laplacian(const double* __restrict__ p, const size_t i,
                                            const int x, const int y, const int nx, const int ny,
                                            const double cc, const double czn, const double czp) {
    const double cxp = (x < nx - 1) ? __ldg(p + i + 1) : cc;
    const double cxn = (x > 0) ? __ldg(p + i - 1) : cc;
    const double cyp = (y < ny - 1) ? __ldg(p + i + nx) : cc;
    const double cyn = (y > 0) ? __ldg(p + i - nx) : cc;

    const double cxx = (cxp + cxn - 2.0 * cc) / (kDx * kDx);
    const double cyy = (cyp + cyn - 2.0 * cc) / (kDy * kDy);
    const double czz = (czp + czn - 2.0 * cc) / (kDz * kDz);
    return cxx + cyy + czz;
}

// Compute chemical potential for global planes [zBegin, zEnd). Each thread owns one (x,y)
// column and marches over a chunk of zChunk planes. Local arrays start at global plane zBase.
__global__ void __launch_bounds__(kBlockX * kBlockY)
computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                               const int nx, const int ny, const int nz, const int zBase,
                               const int zBegin, const int zEnd, const int zChunk) {
    const int x = blockIdx.x * kBlockX + threadIdx.x;
    const int y = blockIdx.y * kBlockY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    const size_t plane = static_cast<size_t>(nx) * ny;

    size_t i = idx3(x, y, z0 - zBase, nx, ny);
    double cc = __ldg(c + i);
    double czn = (z0 > 0) ? __ldg(c + i - plane) : cc;
    for (int z = z0; z < z1; ++z, i += plane) {
        const double czp = (z < nz - 1) ? __ldg(c + i + plane) : cc;
        const double cv = cc;
        mu[i] = (kMuConst + kMuLinear * cv + cv * cv * cv)
               - kGamma * laplacian(c, i, x, y, nx, ny, cc, czn, czp);
        czn = cc;
        cc = czp;
    }
}

// Cahn-Hilliard update step for global planes [zBegin, zEnd)
__global__ void __launch_bounds__(kBlockX * kBlockY)
cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                         const double* __restrict__ mu,
                         const int nx, const int ny, const int nz, const int zBase,
                         const int zBegin, const int zEnd, const int zChunk) {
    const int x = blockIdx.x * kBlockX + threadIdx.x;
    const int y = blockIdx.y * kBlockY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    const size_t plane = static_cast<size_t>(nx) * ny;

    size_t i = idx3(x, y, z0 - zBase, nx, ny);
    double mc = __ldg(mu + i);
    double mzn = (z0 > 0) ? __ldg(mu + i - plane) : mc;
    for (int z = z0; z < z1; ++z, i += plane) {
        const double mzp = (z < nz - 1) ? __ldg(mu + i + plane) : mc;
        cnew[i] = __ldg(cold + i) + kDt * kD * laplacian(mu, i, x, y, nx, ny, mc, mzn, mzp);
        mzn = mc;
        mc = mzp;
    }
}

// Initialize concentration field for global planes [zBegin, zEnd)
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol,
                                              const size_t plane, const int zBase,
                                              const int zBegin, const int zEnd) {
    const size_t first = static_cast<size_t>(zBegin) * plane;
    const size_t count = static_cast<size_t>(zEnd - zBegin) * plane;
    const size_t offset = static_cast<size_t>(zBegin - zBase) * plane;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += stride) {
        // Generate pseudo-random value in [-1, 1]
        const size_t linear_id = first + k;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[offset + k] = -1.0 + 2.0 * pseudo;
    }
}

// NUMA node of the PCIe root of a GPU (-1 if unknown)
static int gpuNumaNode(const int device) {
    char busId[32] = {0};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) != cudaSuccess) return -1;
    std::string path = "/sys/bus/pci/devices/";
    for (const char* p = busId; *p; ++p) path += static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
    path += "/numa_node";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return -1;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node;
}

// Pinned host staging buffer placed on the given NUMA node (falls back to cudaMallocHost).
// Halo staging through memory on a remote socket can halve the exchange bandwidth.
struct PinnedBuffer {
    double* ptr = nullptr;
    size_t bytes = 0;
    bool registered = false;
};

static PinnedBuffer allocPinnedOnNode(const size_t bytes, const int node) {
    PinnedBuffer b;
    b.bytes = bytes;
    if (node >= 0 && node < 64 && bytes > 0) {
        void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem != MAP_FAILED) {
            const unsigned long mask = 1UL << node;
            constexpr int kMpolBind = 2;
            if (syscall(SYS_mbind, mem, bytes, kMpolBind, &mask, sizeof(mask) * 8, 0) == 0) {
                memset(mem, 0, bytes);  // first touch on the bound node
                if (cudaHostRegister(mem, bytes, cudaHostRegisterPortable) == cudaSuccess) {
                    b.ptr = static_cast<double*>(mem);
                    b.registered = true;
                    return b;
                }
                cudaGetLastError();  // clear the registration error
            }
            munmap(mem, bytes);
        }
    }
    CUDA_CHECK(cudaMallocHost(&b.ptr, bytes));
    return b;
}

static void freePinned(PinnedBuffer& b) {
    if (!b.ptr) return;
    if (b.registered) {
        CUDA_CHECK(cudaHostUnregister(b.ptr));
        munmap(b.ptr, b.bytes);
    } else {
        CUDA_CHECK(cudaFreeHost(b.ptr));
    }
    b.ptr = nullptr;
}

// Per-GPU state for the z-slab domain decomposition
struct Slab {
    int device = 0;
    int zs = 0;           // first owned global plane
    int ze = 0;           // one past last owned global plane
    int zBase = 0;        // global plane of local plane 0 (zs - kHalo)
    int zLo = 0;          // first valid global plane held locally (incl. halo)
    int zHi = 0;          // one past last valid global plane held locally (incl. halo)
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    PinnedBuffer sendLo[2];  // pinned host staging: first kHalo owned planes
    PinnedBuffer sendHi[2];  // pinned host staging: last kHalo owned planes
    cudaStream_t compute = nullptr;
    cudaStream_t copy = nullptr;
    cudaEvent_t boundaryDone = nullptr;      // boundary planes of cnew written
    cudaEvent_t sendDone = nullptr;          // boundary planes staged to host
    cudaEvent_t recvDone = nullptr;          // halo planes of cnew received
};

// Launch geometry: 2D tiles in x/y; z is split into chunks to provide enough blocks
static dim3 makeGrid(const int nx, const int ny, const int planes, const int numSMs, int& zChunk) {
    const unsigned gx = static_cast<unsigned>((nx + kBlockX - 1) / kBlockX);
    const unsigned gy = static_cast<unsigned>((ny + kBlockY - 1) / kBlockY);
    const long long tilesXY = static_cast<long long>(gx) * gy;
    const long long targetBlocks = static_cast<long long>(numSMs) * 16;
    long long zSplits = (targetBlocks + tilesXY - 1) / tilesXY;
    zSplits = std::max<long long>(1, std::min<long long>(zSplits, std::max(1, planes / 8)));
    zChunk = static_cast<int>((planes + zSplits - 1) / zSplits);
    return dim3(gx, gy, static_cast<unsigned>((planes + zChunk - 1) / zChunk));
}

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
    
    size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    // Domain decomposition over all visible GPUs (z-slabs). Small problems use fewer GPUs
    // since halo exchange and multi-device synchronization would dominate.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device found\n");
        return 1;
    }
    // Thresholds were tuned on a 4-GPU, 2-socket PCIe system without peer access.
    constexpr size_t kMinCellsTwoGpus = size_t(1) << 19;  // ~0.5M cells
    constexpr size_t kMinCellsAllGpus = size_t(3) << 20;  // ~3M cells
    constexpr int kMinPlanesPerGpu = 8;
    int numGpus = 1;
    if (gridSize >= kMinCellsAllGpus) {
        numGpus = deviceCount;
    } else if (gridSize >= kMinCellsTwoGpus) {
        numGpus = std::min(deviceCount, 2);
    }
    numGpus = std::max(1, std::min(numGpus, inz / kMinPlanesPerGpu));

    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

    std::vector<Slab> slabs(numGpus);
    const size_t haloBytes = static_cast<size_t>(kHalo) * plane * sizeof(double);
    for (int g = 0; g < numGpus; ++g) {
        Slab& s = slabs[g];
        s.device = g;
        s.zs = static_cast<int>(static_cast<long long>(nz) * g / numGpus);
        s.ze = static_cast<int>(static_cast<long long>(nz) * (g + 1) / numGpus);
        s.zBase = s.zs - kHalo;
        s.zLo = std::max(0, s.zs - kHalo);
        s.zHi = std::min(inz, s.ze + kHalo);
        const size_t localBytes = static_cast<size_t>(s.ze - s.zs + 2 * kHalo) * plane * sizeof(double);
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMalloc(&s.cold, localBytes));
        CUDA_CHECK(cudaMalloc(&s.cnew, localBytes));
        CUDA_CHECK(cudaMalloc(&s.mu, localBytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.boundaryDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.sendDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.recvDone, cudaEventDisableTiming));
        if (numGpus > 1) {
            // Staging buffers live on the sending GPU's NUMA node
            const int node = gpuNumaNode(s.device);
            for (int b = 0; b < 2; ++b) {
                if (g > 0) s.sendLo[b] = allocPinnedOnNode(haloBytes, node);
                if (g < numGpus - 1) s.sendHi[b] = allocPinnedOnNode(haloBytes, node);
            }
        }
    }
    // Pointer to a global plane within a slab's local arrays
    auto planePtr = [plane](const Slab& s, double* base, int z) { return base + static_cast<size_t>(z - s.zBase) * plane; };

    // Initialize concentration field (owned planes and halos)
    printf("Initializing concentration field...\n");
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        if (gridSize > 0) {
            initializeConcentrationKernel<<<static_cast<unsigned>(numSMs) * 32, 256, 0, s.compute>>>(
                s.cold, gridSize, plane, s.zBase, s.zLo, s.zHi);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0) {
        for (int t = 0; t < iterations; ++t) {
            const int buf = t & 1;
            for (int g = 0; g < numGpus; ++g) {
                Slab& s = slabs[g];
                CUDA_CHECK(cudaSetDevice(s.device));
                const bool hasLo = g > 0;
                const bool hasHi = g < numGpus - 1;

                // Compute chemical potential (owned planes plus one ghost plane each side).
                // Planes that do not depend on halo data are computed first, overlapping with
                // the halo exchange of the previous step.
                const int muLo = std::max(0, s.zs - 1);
                const int muHi = std::min(inz, s.ze + 1);
                const int muIntLo = hasLo ? s.zs + 1 : muLo;
                const int muIntHi = hasHi ? s.ze - 1 : muHi;
                int zChunk = 1;
                dim3 grid = makeGrid(inx, iny, muIntHi - muIntLo, numSMs, zChunk);
                computeChemicalPotentialKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                    s.cold, s.mu, inx, iny, inz, s.zBase, muIntLo, muIntHi, zChunk);
                if (numGpus > 1) {
                    // Halo planes of cold must have arrived before they are read
                    if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.recvDone, 0));
                    grid = makeGrid(inx, iny, 2, numSMs, zChunk);
                    if (hasLo) {
                        computeChemicalPotentialKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                            s.cold, s.mu, inx, iny, inz, s.zBase, s.zs - 1, s.zs + 1, zChunk);
                    }
                    if (hasHi) {
                        computeChemicalPotentialKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                            s.cold, s.mu, inx, iny, inz, s.zBase, s.ze - 1, s.ze + 1, zChunk);
                    }
                }

                // Update concentration: boundary planes first so the halo exchange can
                // overlap with the interior update
                const int intLo = hasLo ? s.zs + kHalo : s.zs;
                const int intHi = hasHi ? s.ze - kHalo : s.ze;
                if (numGpus > 1) {
                    grid = makeGrid(inx, iny, kHalo, numSMs, zChunk);
                    if (hasLo) {
                        cahnHilliardUpdateKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                            s.cnew, s.cold, s.mu, inx, iny, inz, s.zBase, s.zs, s.zs + kHalo, zChunk);
                    }
                    if (hasHi) {
                        cahnHilliardUpdateKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                            s.cnew, s.cold, s.mu, inx, iny, inz, s.zBase, s.ze - kHalo, s.ze, zChunk);
                    }
                    CUDA_CHECK(cudaEventRecord(s.boundaryDone, s.compute));
                    CUDA_CHECK(cudaStreamWaitEvent(s.copy, s.boundaryDone, 0));
                    if (hasLo) {
                        CUDA_CHECK(cudaMemcpyAsync(s.sendLo[buf].ptr, planePtr(s, s.cnew, s.zs), haloBytes,
                                                   cudaMemcpyDeviceToHost, s.copy));
                    }
                    if (hasHi) {
                        CUDA_CHECK(cudaMemcpyAsync(s.sendHi[buf].ptr, planePtr(s, s.cnew, s.ze - kHalo), haloBytes,
                                                   cudaMemcpyDeviceToHost, s.copy));
                    }
                    CUDA_CHECK(cudaEventRecord(s.sendDone, s.copy));
                }
                if (intHi > intLo) {
                    grid = makeGrid(inx, iny, intHi - intLo, numSMs, zChunk);
                    cahnHilliardUpdateKernel<<<grid, dim3(kBlockX, kBlockY), 0, s.compute>>>(
                        s.cnew, s.cold, s.mu, inx, iny, inz, s.zBase, intLo, intHi, zChunk);
                }
            }

            // Receive halo planes of cnew from neighbouring slabs
            if (numGpus > 1) {
                for (int g = 0; g < numGpus; ++g) {
                    Slab& s = slabs[g];
                    CUDA_CHECK(cudaSetDevice(s.device));
                    if (g > 0) {
                        const Slab& lo = slabs[g - 1];
                        CUDA_CHECK(cudaStreamWaitEvent(s.copy, lo.sendDone, 0));
                        CUDA_CHECK(cudaMemcpyAsync(planePtr(s, s.cnew, s.zs - kHalo), lo.sendHi[buf].ptr, haloBytes,
                                                   cudaMemcpyHostToDevice, s.copy));
                    }
                    if (g < numGpus - 1) {
                        const Slab& hi = slabs[g + 1];
                        CUDA_CHECK(cudaStreamWaitEvent(s.copy, hi.sendDone, 0));
                        CUDA_CHECK(cudaMemcpyAsync(planePtr(s, s.cnew, s.ze), hi.sendLo[buf].ptr, haloBytes,
                                                   cudaMemcpyHostToDevice, s.copy));
                    }
                    CUDA_CHECK(cudaEventRecord(s.recvDone, s.copy));
                }
            }

            // Swap buffers
            for (Slab& s : slabs) std::swap(s.cold, s.cnew);
        }
    }
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Gather result from all GPUs and release resources
    std::vector<double> cold(gridSize);
    for (Slab& s : slabs) {
        CUDA_CHECK(cudaSetDevice(s.device));
        if (gridSize > 0) {
            CUDA_CHECK(cudaMemcpy(cold.data() + static_cast<size_t>(s.zs) * plane, planePtr(s, s.cold, s.zs),
                                  static_cast<size_t>(s.ze - s.zs) * plane * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(s.cold));
        CUDA_CHECK(cudaFree(s.cnew));
        CUDA_CHECK(cudaFree(s.mu));
        for (int b = 0; b < 2; ++b) {
            freePinned(s.sendLo[b]);
            freePinned(s.sendHi[b]);
        }
        CUDA_CHECK(cudaEventDestroy(s.boundaryDone));
        CUDA_CHECK(cudaEventDestroy(s.sendDone));
        CUDA_CHECK(cudaEventDestroy(s.recvDone));
        CUDA_CHECK(cudaStreamDestroy(s.compute));
        CUDA_CHECK(cudaStreamDestroy(s.copy));
    }
    
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
