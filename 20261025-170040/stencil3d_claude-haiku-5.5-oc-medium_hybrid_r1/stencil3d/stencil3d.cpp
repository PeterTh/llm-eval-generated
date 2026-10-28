#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Hybrid parallelization:
//   MPI    : 3D Cartesian domain decomposition, one rank per sub-domain, halo exchange.
//   CUDA   : each rank drives one GPU; stencil and halo pack/unpack run as kernels.
//            The interior is computed while halos are exchanged, then the boundary shell.
//   OpenMP : host-side initialization, gather and scatter for result output/validation.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation (global grid, same linear layout as the serial code)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Index into a locally padded block: owned cells are [0,lx)x[0,ly)x[0,lz),
// with one halo layer at coordinates -1 and l.
__host__ __device__ inline size_t padIdx(const int x, const int y, const int z, const int sx, const int sy) noexcept {
    return (static_cast<size_t>(z + 1) * sy + static_cast<size_t>(y + 1)) * sx + static_cast<size_t>(x + 1);
}

// Geometry of one rank's sub-domain
struct Geom {
    int nx, ny, nz;  // global grid size
    int x0, y0, z0;  // global origin of owned block
    int lx, ly, lz;  // owned block extents
    int sx, sy;      // padded strides (lx+2, ly+2)
};

// Rectangular box of owned cells (local coordinates)
struct Box {
    int ox, oy, oz;
    int ex, ey, ez;
};

struct Range {
    int start;
    int len;
};

// Face numbering: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z ; opposite face is d^1

static Range blockRange(const int n, const int p, const int c) {
    const int base = n / p;
    const int rem = n % p;
    Range r;
    r.len = base + (c < rem ? 1 : 0);
    r.start = c * base + std::min(c, rem);
    return r;
}

// Choose a 3D process grid px*py*pz == P minimizing total halo volume.
// Each dimension must not exceed the grid size so every block is non-empty.
static bool chooseDims(const int P, const int nx, const int ny, const int nz, int dims[3]) {
    bool found = false;
    double best = 0.0;
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0 || px > nx) continue;
        const int rest = P / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0 || py > ny) continue;
            const int pz = rest / py;
            if (pz > nz) continue;
            const double cost = double(px - 1) * ny * nz + double(py - 1) * nx * nz + double(pz - 1) * nx * ny;
            if (!found || cost < best) {
                found = true;
                best = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
    return found;
}

// Packed halo layout: per face d, number of cells and offset in the packed buffer
static size_t faceLayout(const Geom& g, size_t off[6], size_t cnt[6]) {
    const int L[3] = {g.lx, g.ly, g.lz};
    size_t total = 0;
    for (int d = 0; d < 6; ++d) {
        const int axis = d >> 1;
        const int a0 = (axis == 0) ? 1 : 0;
        const int a1 = (axis == 2) ? 1 : 2;
        cnt[d] = static_cast<size_t>(L[a0]) * static_cast<size_t>(L[a1]);
        off[d] = total;
        total += cnt[d];
    }
    return total;
}

// Partition the owned block into disjoint boxes covering all cells whose
// 7-point neighborhood touches the halo (the "shell"). Empty boxes are skipped.
static std::vector<Box> buildShells(const int lx, const int ly, const int lz) {
    const int iy = std::max(ly - 2, 0);
    const int iz = std::max(lz - 2, 0);
    std::vector<Box> s;
    s.push_back({0, 0, 0, lx, ly, 1});                    // z = 0
    if (lz >= 2) s.push_back({0, 0, lz - 1, lx, ly, 1});  // z = lz-1
    s.push_back({0, 0, 1, lx, 1, iz});                    // y = 0
    if (ly >= 2) s.push_back({0, ly - 1, 1, lx, 1, iz});  // y = ly-1
    s.push_back({0, 1, 1, 1, iy, iz});                    // x = 0
    if (lx >= 2) s.push_back({lx - 1, 1, 1, 1, iy, iz});  // x = lx-1
    return s;
}

// Interior: owned cells whose neighbors are all owned (no halo access)
static Box buildInterior(const int lx, const int ly, const int lz) {
    return {1, 1, 1, std::max(lx - 2, 0), std::max(ly - 2, 0), std::max(lz - 2, 0)};
}

// 7-point stencil on a box of owned cells. Global boundary cells are copied.
__global__ void stencilBoxKernel(const Real* __restrict__ in, Real* __restrict__ out, const Box b, const Geom g) {
    const size_t sz = static_cast<size_t>(g.sx) * static_cast<size_t>(g.sy);
    const size_t sy = static_cast<size_t>(g.sx);
    const int x = b.ox + blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= b.ox + b.ex) return;
    const int gx = g.x0 + x;
    const bool xBoundary = (gx == 0) || (gx == g.nx - 1);
    for (int y = b.oy + blockIdx.y * blockDim.y + threadIdx.y; y < b.oy + b.ey; y += gridDim.y * blockDim.y) {
        const int gy = g.y0 + y;
        const bool xyBoundary = xBoundary || (gy == 0) || (gy == g.ny - 1);
        for (int z = b.oz + blockIdx.z; z < b.oz + b.ez; z += gridDim.z) {
            const size_t p = padIdx(x, y, z, g.sx, g.sy);
            const int gz = g.z0 + z;
            if (xyBoundary || gz == 0 || gz == g.nz - 1) {
                out[p] = in[p];
            } else {
                const Real center = in[p];
                const Real left = in[p - 1];
                const Real right = in[p + 1];
                const Real front = in[p - sy];
                const Real back = in[p + sy];
                const Real bottom = in[p - sz];
                const Real top = in[p + sz];
                // Simple averaging stencil (same operation order as the serial version)
                out[p] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Copy one owned face layer into a contiguous buffer (UNPACK=false),
// or write a received buffer into the halo layer (UNPACK=true).
template <bool UNPACK>
__global__ void faceKernel(Real* __restrict__ field, Real* __restrict__ buf, const int d, const Geom g) {
    const int axis = d >> 1;
    const bool hi = (d & 1) != 0;
    const int L[3] = {g.lx, g.ly, g.lz};
    const int a0 = (axis == 0) ? 1 : 0;
    const int a1 = (axis == 2) ? 1 : 2;
    const int nu = L[a0];
    const int nv = L[a1];
    int c[3];
    if (UNPACK) {
        c[axis] = hi ? L[axis] : -1;
    } else {
        c[axis] = hi ? L[axis] - 1 : 0;
    }
    for (int v = blockIdx.y * blockDim.y + threadIdx.y; v < nv; v += gridDim.y * blockDim.y) {
        for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < nu; u += gridDim.x * blockDim.x) {
            c[a0] = u;
            c[a1] = v;
            const size_t p = padIdx(c[0], c[1], c[2], g.sx, g.sy);
            const size_t q = static_cast<size_t>(v) * nu + u;
            if (UNPACK) {
                field[p] = buf[q];
            } else {
                buf[q] = field[p];
            }
        }
    }
}

static void launchStencil(const Real* in, Real* out, const Box& b, const Geom& g, cudaStream_t s) {
    if (b.ex <= 0 || b.ey <= 0 || b.ez <= 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid((b.ex + 31) / 32, std::min((b.ey + 7) / 8, 65535), std::min(b.ez, 65535));
    stencilBoxKernel<<<grid, block, 0, s>>>(in, out, b, g);
    CUDA_CHECK(cudaGetLastError());
}

template <bool UNPACK>
static void launchFaces(Real* field, Real* packed, const int nb[6], const size_t off[6], const Geom& g,
                        cudaStream_t s) {
    const int L[3] = {g.lx, g.ly, g.lz};
    for (int d = 0; d < 6; ++d) {
        if (nb[d] == MPI_PROC_NULL) continue;  // no neighbor on this side
        const int axis = d >> 1;
        const int a0 = (axis == 0) ? 1 : 0;
        const int a1 = (axis == 2) ? 1 : 2;
        const dim3 block(32, 8, 1);
        const dim3 grid((L[a0] + 31) / 32, std::min((L[a1] + 7) / 8, 65535), 1);
        faceKernel<UNPACK><<<grid, block, 0, s>>>(field, packed + off[d], d, g);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Exchange packed faces with the 6 neighbors (MPI_PROC_NULL sides are no-ops).
static void exchangeHalos(const int nb[6], const size_t off[6], const size_t cnt[6], Real* hSend, Real* hRecv,
                          MPI_Comm comm) {
    MPI_Request req[12];
    int n = 0;
    for (int d = 0; d < 6; ++d) {
        // Data arriving on my side d was sent by neighbor d as its face d^1.
        MPI_Irecv(hRecv + off[d], static_cast<int>(cnt[d]), MPI_DOUBLE, nb[d], d ^ 1, comm, &req[n++]);
    }
    for (int d = 0; d < 6; ++d) {
        MPI_Isend(hSend + off[d], static_cast<int>(cnt[d]), MPI_DOUBLE, nb[d], d, comm, &req[n++]);
    }
    MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
}

// OpenMP-parallel initialization of owned cells (global index formula from the serial code)
static void initializeLocal(std::vector<Real>& pad, const Geom& g) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < g.lz; ++z) {
        for (int y = 0; y < g.ly; ++y) {
            for (int x = 0; x < g.lx; ++x) {
                const size_t idx = idx3(static_cast<size_t>(g.x0 + x), static_cast<size_t>(g.y0 + y),
                                        static_cast<size_t>(g.z0 + z), static_cast<size_t>(g.nx),
                                        static_cast<size_t>(g.ny));
                pad[padIdx(x, y, z, g.sx, g.sy)] = (idx % 19) * 1.0;
            }
        }
    }
}

// OpenMP-parallel extraction of the owned block into a contiguous (z,y,x) array
static void extractOwned(const std::vector<Real>& pad, std::vector<Real>& block, const Geom& g) {
    block.resize(static_cast<size_t>(g.lx) * g.ly * g.lz);
#pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < g.lz; ++z) {
        for (int y = 0; y < g.ly; ++y) {
            for (int x = 0; x < g.lx; ++x) {
                block[(static_cast<size_t>(z) * g.ly + y) * g.lx + x] = pad[padIdx(x, y, z, g.sx, g.sy)];
            }
        }
    }
}

// OpenMP-parallel scatter of a contiguous block into the global grid
static void scatterBlock(std::vector<Real>& global, const std::vector<Real>& block, const int x0, const int y0,
                         const int z0, const int lx, const int ly, const int lz, const int nx, const int ny) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < lz; ++z) {
        for (int y = 0; y < ly; ++y) {
            for (int x = 0; x < lx; ++x) {
                const size_t gi = idx3(static_cast<size_t>(x0 + x), static_cast<size_t>(y0 + y),
                                       static_cast<size_t>(z0 + z), static_cast<size_t>(nx), static_cast<size_t>(ny));
                global[gi] = block[(static_cast<size_t>(z) * ly + y) * lx + x];
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

struct Options {
    size_t nx;
    size_t ny;
    size_t nz;
    int iterations;
    bool validate;
    bool printResults;
};

// Runs the distributed benchmark on the communicator `cart` (3D Cartesian, dims given).
// Returns the process exit code (identical on all active ranks).
static int runBenchmark(MPI_Comm cart, const Options& opt, const int dims[3], const int localRank) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &size);
    const size_t nx = opt.nx;
    const size_t ny = opt.ny;
    const size_t nz = opt.nz;
    const int iterations = opt.iterations;
    const bool validate = opt.validate;
    const bool printResults = opt.printResults;

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    int nb[6];  // neighbor ranks per face (MPI_PROC_NULL at domain edges)
    for (int a = 0; a < 3; ++a) {
        int src = MPI_PROC_NULL;
        int dst = MPI_PROC_NULL;
        MPI_Cart_shift(cart, a, 1, &src, &dst);
        nb[2 * a] = src;
        nb[2 * a + 1] = dst;
    }

    const Range rx = blockRange(static_cast<int>(nx), dims[0], coords[0]);
    const Range ry = blockRange(static_cast<int>(ny), dims[1], coords[1]);
    const Range rz = blockRange(static_cast<int>(nz), dims[2], coords[2]);
    Geom g{};
    g.nx = static_cast<int>(nx);
    g.ny = static_cast<int>(ny);
    g.nz = static_cast<int>(nz);
    g.x0 = rx.start;
    g.y0 = ry.start;
    g.z0 = rz.start;
    g.lx = rx.len;
    g.ly = ry.len;
    g.lz = rz.len;
    g.sx = g.lx + 2;
    g.sy = g.ly + 2;

    // One GPU per rank: select device by node-local rank
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Device buffers (double buffering, padded with one halo layer)
    const size_t padTotal = static_cast<size_t>(g.sx) * g.sy * (g.lz + 2);
    Real* dGrid[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dGrid[b], padTotal * sizeof(Real)));
        CUDA_CHECK(cudaMemset(dGrid[b], 0, padTotal * sizeof(Real)));
    }

    // Packed halo buffers: device pack/unpack, pinned host staging for MPI
    size_t faceOff[6];
    size_t faceCnt[6];
    const size_t faceTotal = faceLayout(g, faceOff, faceCnt);
    Real *dPack, *dHalo, *hSend, *hRecv;
    CUDA_CHECK(cudaMalloc(&dPack, faceTotal * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dHalo, faceTotal * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hSend, faceTotal * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecv, faceTotal * sizeof(Real)));
    memset(hSend, 0, faceTotal * sizeof(Real));
    memset(hRecv, 0, faceTotal * sizeof(Real));

    cudaStream_t sComp, sComm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    cudaEvent_t evHalo, evShell;
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evShell, cudaEventDisableTiming));

    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    {
        std::vector<Real> pad(padTotal, 0.0);
        initializeLocal(pad, g);
        CUDA_CHECK(cudaMemcpy(dGrid[0], pad.data(), padTotal * sizeof(Real), cudaMemcpyHostToDevice));
    }

    const std::vector<Box> shells = buildShells(g.lx, g.ly, g.lz);
    const Box interior = buildInterior(g.lx, g.ly, g.lz);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = dGrid[cur];
        Real* out = dGrid[1 - cur];

        // (1) Interior update: needs no halo data (overlaps with communication)
        launchStencil(in, out, interior, g, sComp);

        // (2) Halo exchange on the communication stream
        if (iter > 0) CUDA_CHECK(cudaStreamWaitEvent(sComm, evShell, 0));  // previous iteration's output ready
        launchFaces<false>(dGrid[cur], dPack, nb, faceOff, g, sComm);
        CUDA_CHECK(cudaMemcpyAsync(hSend, dPack, faceTotal * sizeof(Real), cudaMemcpyDeviceToHost, sComm));
        CUDA_CHECK(cudaStreamSynchronize(sComm));
        exchangeHalos(nb, faceOff, faceCnt, hSend, hRecv, cart);
        CUDA_CHECK(cudaMemcpyAsync(dHalo, hRecv, faceTotal * sizeof(Real), cudaMemcpyHostToDevice, sComm));
        launchFaces<true>(dGrid[cur], dHalo, nb, faceOff, g, sComm);
        CUDA_CHECK(cudaEventRecord(evHalo, sComm));

        // (3) Boundary shell: needs fresh halos
        CUDA_CHECK(cudaStreamWaitEvent(sComp, evHalo, 0));
        for (const Box& b : shells) {
            launchStencil(in, out, b, g, sComp);
        }
        CUDA_CHECK(cudaEventRecord(evShell, sComp));

        cur = 1 - cur;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    long long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, cart);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxMs));

        // Calculate performance metrics
        const double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        const double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid (only needed for result printing / validation)
    const int finalIdx = (iterations % 2 == 0) ? 0 : 1;
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> pad(padTotal);
        CUDA_CHECK(cudaMemcpy(pad.data(), dGrid[finalIdx], padTotal * sizeof(Real), cudaMemcpyDeviceToHost));
        std::vector<Real> local;
        extractOwned(pad, local, g);

        std::vector<Real> global;
        if (rank == 0) {
            global.resize(nx * ny * nz);
            scatterBlock(global, local, g.x0, g.y0, g.z0, g.lx, g.ly, g.lz, g.nx, g.ny);
            std::vector<Real> tmp;
            for (int r = 1; r < size; ++r) {
                int c[3];
                MPI_Cart_coords(cart, r, 3, c);
                const Range qx = blockRange(g.nx, dims[0], c[0]);
                const Range qy = blockRange(g.ny, dims[1], c[1]);
                const Range qz = blockRange(g.nz, dims[2], c[2]);
                tmp.resize(static_cast<size_t>(qx.len) * qy.len * qz.len);
                MPI_Recv(tmp.data(), static_cast<int>(tmp.size()), MPI_DOUBLE, r, 0, cart, MPI_STATUS_IGNORE);
                scatterBlock(global, tmp, qx.start, qy.start, qz.start, qx.len, qy.len, qz.len, g.nx, g.ny);
            }
        } else {
            MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 0, cart);
        }

        if (rank == 0) {
            if (printResults) {
                print_results(global, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, cart);
    }

    CUDA_CHECK(cudaEventDestroy(evHalo));
    CUDA_CHECK(cudaEventDestroy(evShell));
    CUDA_CHECK(cudaStreamDestroy(sComp));
    CUDA_CHECK(cudaStreamDestroy(sComm));
    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFreeHost(hRecv));
    CUDA_CHECK(cudaFree(dPack));
    CUDA_CHECK(cudaFree(dHalo));
    CUDA_CHECK(cudaFree(dGrid[0]));
    CUDA_CHECK(cudaFree(dGrid[1]));
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Options opt{};
    opt.nx = 128;
    opt.ny = 0;  // Will be set to nx if not specified
    opt.nz = 0;  // Will be set to nx if not specified
    opt.iterations = 10;
    opt.validate = false;
    opt.printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            opt.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            opt.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            opt.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            opt.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = true;
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

    if (opt.ny == 0) opt.ny = opt.nx;
    if (opt.nz == 0) opt.nz = opt.nx;

    if (opt.nx < 1 || opt.ny < 1 || opt.nz < 1 || opt.nx > (1u << 30) || opt.ny > (1u << 30) ||
        opt.nz > (1u << 30)) {
        if (rank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", opt.nx, opt.ny, opt.nz);
        printf("Iterations: %d\n", opt.iterations);
        printf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
    }

    // Node-local rank (for GPU selection), computed over all ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    // 3D Cartesian decomposition over the largest rank subset that can host the grid
    // (ranks beyond that subset stay idle). A single rank always fits.
    int active = size;
    int dims[3] = {1, 1, 1};
    while (!chooseDims(active, static_cast<int>(opt.nx), static_cast<int>(opt.ny), static_cast<int>(opt.nz), dims)) {
        --active;
    }
    if (active < size && rank == 0) {
        fprintf(stderr, "Note: using %d of %d ranks (grid too small to split further)\n", active, size);
    }

    const bool isActive = rank < active;
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, isActive ? 0 : MPI_UNDEFINED, rank, &activeComm);

    int exitCode = 0;
    if (isActive) {
        int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(activeComm, 3, dims, periods, 0, &cart);
        exitCode = runBenchmark(cart, opt, dims, localRank);
        MPI_Comm_free(&cart);
        MPI_Comm_free(&activeComm);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
