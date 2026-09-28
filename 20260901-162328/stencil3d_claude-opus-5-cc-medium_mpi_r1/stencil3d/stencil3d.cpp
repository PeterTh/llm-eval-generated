#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Block distribution of n points over p parts: part i has size n/p (+1 for the first n%p parts)
inline void blockRange(const size_t n, const int p, const int i, size_t& begin, size_t& count) noexcept {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    const size_t ii = static_cast<size_t>(i);
    count = base + (ii < rem ? 1 : 0);
    begin = ii * base + std::min(ii, rem);
}

// Local (per-rank) grid geometry: interior extents lx/ly/lz plus one halo layer on each side
struct Domain {
    size_t lx = 0, ly = 0, lz = 0;     // owned points per dimension
    size_t gx0 = 0, gy0 = 0, gz0 = 0;  // global index of first owned point
    size_t dx = 0, dy = 0, dz = 0;     // padded extents (l* + 2)
    size_t slice = 0;                  // dx * dy

    inline size_t index(const size_t i, const size_t j, const size_t k) const noexcept { return k * slice + j * dx + i; }
    inline size_t paddedSize() const noexcept { return dx * dy * dz; }
};

// Initialize the local grid (including halo layers) with the global initialization formula.
// The formula is defined for global points only; halo cells outside the global grid keep 0.
void initializeGrid(std::vector<Real>& grid, const Domain& d, const size_t nx, const size_t ny, const size_t nz) {
    std::fill(grid.begin(), grid.end(), Real(0));

    // Halo cells that lie inside the global grid are filled as well, so that the very first
    // iteration does not depend on the initial halo exchange.
    const size_t kBegin = (d.gz0 == 0) ? 1 : 0;
    const size_t kEnd = (d.gz0 + d.lz == nz) ? d.lz : d.lz + 1;
    const size_t jBegin = (d.gy0 == 0) ? 1 : 0;
    const size_t jEnd = (d.gy0 + d.ly == ny) ? d.ly : d.ly + 1;
    const size_t iBegin = (d.gx0 == 0) ? 1 : 0;
    const size_t iEnd = (d.gx0 + d.lx == nx) ? d.lx : d.lx + 1;

    for (size_t k = kBegin; k <= kEnd; ++k) {
        const size_t gz = d.gz0 + k - 1;
        for (size_t j = jBegin; j <= jEnd; ++j) {
            const size_t gy = d.gy0 + j - 1;
            const size_t rowBase = idx3(d.gx0 - 1 + iBegin, gy, gz, nx, ny);
            Real* __restrict__ out = &grid[d.index(iBegin, j, k)];
            for (size_t i = 0; i <= iEnd - iBegin; ++i) {
                out[i] = static_cast<Real>((rowBase + i) % 19) * 1.0;
            }
        }
    }
}

// Streaming (non-temporal) stores avoid write-allocate traffic and pay off once enough ranks
// share a memory system; on few cores they are slower than regular stores, so they are enabled
// per run (see main()).
static bool g_streamStores = false;

// 7-point stencil over the local sub-box [i0,i1] x [j0,j1] x [k0,k1] (inclusive, local indices)
static inline void stencilBox(const Real* __restrict__ input, Real* __restrict__ output, const Domain& d,
                              const size_t i0, const size_t i1, const size_t j0, const size_t j1,
                              const size_t k0, const size_t k1) {
    if (i0 > i1 || j0 > j1 || k0 > k1) return;

    const size_t dx = d.dx;
    const size_t slice = d.slice;

    for (size_t k = k0; k <= k1; ++k) {
        for (size_t j = j0; j <= j1; ++j) {
            const size_t base = d.index(i0, j, k);
            const Real* __restrict__ c = input + base;
            Real* __restrict__ o = output + base;
            const size_t n = i1 - i0 + 1;
            size_t i = 0;
#if defined(__AVX__)
            // Streaming stores for the aligned bulk of the row: the output is not read again in
            // this iteration, so bypassing the cache avoids the write-allocate traffic.
            // The arithmetic is identical (per-lane, same order) as the scalar path.
            if (g_streamStores) {
                const size_t misalign = (reinterpret_cast<uintptr_t>(o) / sizeof(Real)) & 3u;
                const size_t prefix = std::min<size_t>(n, misalign == 0 ? 0 : 4 - misalign);
                const __m256d v7 = _mm256_set1_pd(7.0);
                for (; i < prefix; ++i) {
                    o[i] = (c[i] + c[i - 1] + c[i + 1] + c[i - dx] + c[i + dx] + c[i - slice] + c[i + slice]) / 7.0;
                }
                for (; i + 4 <= n; i += 4) {
                    __m256d v = _mm256_loadu_pd(c + i);
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i - 1));
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i + 1));
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i - dx));
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i + dx));
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i - slice));
                    v = _mm256_add_pd(v, _mm256_loadu_pd(c + i + slice));
                    _mm256_stream_pd(o + i, _mm256_div_pd(v, v7));
                }
            }
#endif
            for (; i < n; ++i) {
                o[i] = (c[i] + c[i - 1] + c[i + 1] + c[i - dx] + c[i + dx] + c[i - slice] + c[i + slice]) / 7.0;
            }
        }
    }
#if defined(__AVX__)
    if (g_streamStores) _mm_sfence();  // make streaming stores visible to later loads / MPI sends
#endif
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

// Choose a process grid (dims[0]=Z, dims[1]=Y, dims[2]=X) that uses as many of the `size` ranks
// as possible without splitting a dimension into more parts than it has grid points, minimizing
// the halo surface area. Returns the number of ranks that are actually used.
int chooseDims(const int size, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    for (int m = size; m >= 1; --m) {
        double bestCost = 0.0;
        bool found = false;
        for (int a = 1; a <= m; ++a) {
            if (m % a != 0 || static_cast<size_t>(a) > nz) continue;
            const int rest = m / a;
            for (int b = 1; b <= rest; ++b) {
                if (rest % b != 0 || static_cast<size_t>(b) > ny) continue;
                const int c = rest / b;
                if (static_cast<size_t>(c) > nx) continue;
                const double lx = static_cast<double>(nx) / c;
                const double ly = static_cast<double>(ny) / b;
                const double lz = static_cast<double>(nz) / a;
                // Weights break ties in favour of long, contiguous X lines and cheap halo faces:
                // X faces need element-wise packing, Y faces row-wise, Z planes none.
                const double cost = 1.20 * (ly * lz) + 1.05 * (lx * lz) + 1.00 * (lx * ly);
                if (!found || cost < bestCost) {
                    bestCost = cost;
                    found = true;
                    dims[0] = a;
                    dims[1] = b;
                    dims[2] = c;
                }
            }
        }
        if (found) return m;
    }
    dims[0] = dims[1] = dims[2] = 1;
    return 1;
}

// Gather the distributed grid into a single global array on rank 0 (used only for output/validation)
void gatherGrid(const std::vector<Real>& local, const Domain& d, std::vector<Real>& global,
                const size_t nx, const size_t ny, const size_t nz, const int rank, const int size,
                const int dims[3], MPI_Comm comm) {
    // Pack owned points contiguously
    std::vector<Real> sendBuf(d.lx * d.ly * d.lz);
    for (size_t k = 0; k < d.lz; ++k) {
        for (size_t j = 0; j < d.ly; ++j) {
            std::memcpy(&sendBuf[(k * d.ly + j) * d.lx], &local[d.index(1, j + 1, k + 1)], d.lx * sizeof(Real));
        }
    }

    if (rank != 0) {
        MPI_Send(sendBuf.data(), static_cast<int>(sendBuf.size()), MPI_DOUBLE, 0, 7, comm);
        return;
    }

    global.assign(nx * ny * nz, Real(0));

    std::vector<Real> recvBuf;
    for (int r = 0; r < size; ++r) {
        int coords[3];
        MPI_Cart_coords(comm, r, 3, coords);
        size_t bx0, bxn, by0, byn, bz0, bzn;
        blockRange(nx, dims[2], coords[2], bx0, bxn);
        blockRange(ny, dims[1], coords[1], by0, byn);
        blockRange(nz, dims[0], coords[0], bz0, bzn);

        const Real* src;
        if (r == 0) {
            src = sendBuf.data();
        } else {
            recvBuf.resize(bxn * byn * bzn);
            MPI_Recv(recvBuf.data(), static_cast<int>(recvBuf.size()), MPI_DOUBLE, r, 7, comm, MPI_STATUS_IGNORE);
            src = recvBuf.data();
        }

        for (size_t k = 0; k < bzn; ++k) {
            for (size_t j = 0; j < byn; ++j) {
                if (bxn > 0) {
                    std::memcpy(&global[idx3(bx0, by0 + j, bz0 + k, nx, ny)], &src[(k * byn + j) * bxn], bxn * sizeof(Real));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Build a 3D Cartesian topology. dims[0] splits Z, dims[1] splits Y, dims[2] splits X.
    int dims[3] = {1, 1, 1};
    const int active = chooseDims(size, nx, ny, nz, dims);

    // Ranks beyond `active` (only possible for very small grids) do not take part in the computation
    MPI_Comm work;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : 1, rank, &work);
    if (rank >= active) {
        MPI_Comm_free(&work);
        MPI_Finalize();
        return 0;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(work, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&work);

    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &size);

    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, rank, 3, coords);

    int zDown, zUp, yDown, yUp, xDown, xUp;
    MPI_Cart_shift(cart, 0, 1, &zDown, &zUp);
    MPI_Cart_shift(cart, 1, 1, &yDown, &yUp);
    MPI_Cart_shift(cart, 2, 1, &xDown, &xUp);

    Domain d;
    blockRange(nx, dims[2], coords[2], d.gx0, d.lx);
    blockRange(ny, dims[1], coords[1], d.gy0, d.ly);
    blockRange(nz, dims[0], coords[0], d.gz0, d.lz);
    d.dx = d.lx + 2;
    d.dy = d.ly + 2;
    d.dz = d.lz + 2;
    d.slice = d.dx * d.dy;

    // Allocate grids (double buffering). Both buffers are initialized identically: global
    // boundary points are never written by the stencil, so they retain their initial value
    // in either buffer (equivalent to the boundary copy of the serial version).
    std::vector<Real> gridA(d.paddedSize());
    std::vector<Real> gridB;

    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(gridA, d, nx, ny, nz);
    gridB = gridA;

    // Local index range of points that are updated (global boundary points are excluded)
    const size_t i0 = (d.gx0 == 0) ? 2 : 1;
    const size_t i1 = (d.gx0 + d.lx == nx) ? d.lx - 1 : d.lx;
    const size_t j0 = (d.gy0 == 0) ? 2 : 1;
    const size_t j1 = (d.gy0 + d.ly == ny) ? d.ly - 1 : d.ly;
    const size_t k0 = (d.gz0 == 0) ? 2 : 1;
    const size_t k1 = (d.gz0 + d.lz == nz) ? d.lz - 1 : d.lz;
    const bool hasWork = d.lx > 0 && d.ly > 0 && d.lz > 0 && i0 <= i1 && j0 <= j1 && k0 <= k1;

    // Inner sub-box that does not depend on halo data (can be computed during communication)
    const size_t ii0 = std::max<size_t>(i0, 2), ii1 = (d.lx >= 1) ? std::min<size_t>(i1, d.lx - 1) : 0;
    const size_t jj0 = std::max<size_t>(j0, 2), jj1 = (d.ly >= 1) ? std::min<size_t>(j1, d.ly - 1) : 0;
    const size_t kk0 = std::max<size_t>(k0, 2), kk1 = (d.lz >= 1) ? std::min<size_t>(k1, d.lz - 1) : 0;
    const bool overlap = hasWork && ii0 <= ii1 && jj0 <= jj1 && kk0 <= kk1;

    // Halo exchange buffers. Z planes are contiguous and are sent in place; X and Y faces are
    // strided and get packed explicitly.
    const size_t xFace = d.ly * d.lz;
    const size_t yFace = d.lx * d.lz;
    const size_t zPlane = d.slice;
    std::vector<Real> xSendLo(xFace), xSendHi(xFace), xRecvLo(xFace), xRecvHi(xFace);
    std::vector<Real> ySendLo(yFace), ySendHi(yFace), yRecvLo(yFace), yRecvHi(yFace);

    // Enable streaming stores once enough ranks share a node's memory system: they remove the
    // write-allocate traffic of the output array, which helps when the node's memory bandwidth
    // is saturated but costs throughput when only a few cores are active.
    {
        MPI_Comm node;
        int nodeRanks = 1;
        MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        MPI_Comm_size(node, &nodeRanks);
        MPI_Comm_free(&node);
        g_streamStores = nodeRanks >= 16;
    }

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    Real* in = gridA.data();
    Real* out = gridB.data();

    MPI_Request reqs[12];

    for (int iter = 0; iter < iterations; ++iter) {
        int nreq = 0;

        // --- Z direction: whole padded planes are contiguous and are exchanged in place.
        // The halo ring included in such a plane is never read by the 7-point stencil.
        if (zPlane > 0) {
            if (zDown != MPI_PROC_NULL) {
                MPI_Irecv(&in[d.index(0, 0, 0)], static_cast<int>(zPlane), MPI_DOUBLE, zDown, 0, cart, &reqs[nreq++]);
            }
            if (zUp != MPI_PROC_NULL) {
                MPI_Irecv(&in[d.index(0, 0, d.lz + 1)], static_cast<int>(zPlane), MPI_DOUBLE, zUp, 1, cart, &reqs[nreq++]);
            }
            if (zDown != MPI_PROC_NULL) {
                MPI_Isend(&in[d.index(0, 0, 1)], static_cast<int>(zPlane), MPI_DOUBLE, zDown, 1, cart, &reqs[nreq++]);
            }
            if (zUp != MPI_PROC_NULL) {
                MPI_Isend(&in[d.index(0, 0, d.lz)], static_cast<int>(zPlane), MPI_DOUBLE, zUp, 0, cart, &reqs[nreq++]);
            }
        }

        // --- Y direction: rows of lx contiguous elements ---
        if (yFace > 0) {
            if (yDown != MPI_PROC_NULL) MPI_Irecv(yRecvLo.data(), static_cast<int>(yFace), MPI_DOUBLE, yDown, 2, cart, &reqs[nreq++]);
            if (yUp != MPI_PROC_NULL) MPI_Irecv(yRecvHi.data(), static_cast<int>(yFace), MPI_DOUBLE, yUp, 3, cart, &reqs[nreq++]);
            for (size_t k = 0; k < d.lz; ++k) {
                if (yDown != MPI_PROC_NULL) std::memcpy(&ySendLo[k * d.lx], &in[d.index(1, 1, k + 1)], d.lx * sizeof(Real));
                if (yUp != MPI_PROC_NULL) std::memcpy(&ySendHi[k * d.lx], &in[d.index(1, d.ly, k + 1)], d.lx * sizeof(Real));
            }
            if (yDown != MPI_PROC_NULL) MPI_Isend(ySendLo.data(), static_cast<int>(yFace), MPI_DOUBLE, yDown, 3, cart, &reqs[nreq++]);
            if (yUp != MPI_PROC_NULL) MPI_Isend(ySendHi.data(), static_cast<int>(yFace), MPI_DOUBLE, yUp, 2, cart, &reqs[nreq++]);
        }

        // --- X direction: single elements, fully strided ---
        if (xFace > 0) {
            if (xDown != MPI_PROC_NULL) MPI_Irecv(xRecvLo.data(), static_cast<int>(xFace), MPI_DOUBLE, xDown, 4, cart, &reqs[nreq++]);
            if (xUp != MPI_PROC_NULL) MPI_Irecv(xRecvHi.data(), static_cast<int>(xFace), MPI_DOUBLE, xUp, 5, cart, &reqs[nreq++]);
            for (size_t k = 0; k < d.lz; ++k) {
                for (size_t j = 0; j < d.ly; ++j) {
                    if (xDown != MPI_PROC_NULL) xSendLo[k * d.ly + j] = in[d.index(1, j + 1, k + 1)];
                    if (xUp != MPI_PROC_NULL) xSendHi[k * d.ly + j] = in[d.index(d.lx, j + 1, k + 1)];
                }
            }
            if (xDown != MPI_PROC_NULL) MPI_Isend(xSendLo.data(), static_cast<int>(xFace), MPI_DOUBLE, xDown, 5, cart, &reqs[nreq++]);
            if (xUp != MPI_PROC_NULL) MPI_Isend(xSendHi.data(), static_cast<int>(xFace), MPI_DOUBLE, xUp, 4, cart, &reqs[nreq++]);
        }

        // Overlap: compute the halo-independent inner box while messages are in flight
        if (overlap) {
            stencilBox(in, out, d, ii0, ii1, jj0, jj1, kk0, kk1);
        }

        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Unpack Y and X halos
        if (yFace > 0) {
            for (size_t k = 0; k < d.lz; ++k) {
                if (yDown != MPI_PROC_NULL) std::memcpy(&in[d.index(1, 0, k + 1)], &yRecvLo[k * d.lx], d.lx * sizeof(Real));
                if (yUp != MPI_PROC_NULL) std::memcpy(&in[d.index(1, d.ly + 1, k + 1)], &yRecvHi[k * d.lx], d.lx * sizeof(Real));
            }
        }
        if (xFace > 0) {
            for (size_t k = 0; k < d.lz; ++k) {
                for (size_t j = 0; j < d.ly; ++j) {
                    if (xDown != MPI_PROC_NULL) in[d.index(0, j + 1, k + 1)] = xRecvLo[k * d.ly + j];
                    if (xUp != MPI_PROC_NULL) in[d.index(d.lx + 1, j + 1, k + 1)] = xRecvHi[k * d.ly + j];
                }
            }
        }

        if (hasWork) {
            if (!overlap) {
                stencilBox(in, out, d, i0, i1, j0, j1, k0, k1);
            } else {
                // Remaining shell around the already computed inner box
                if (i0 < ii0) stencilBox(in, out, d, i0, ii0 - 1, j0, j1, k0, k1);
                if (ii1 < i1) stencilBox(in, out, d, ii1 + 1, i1, j0, j1, k0, k1);
                if (j0 < jj0) stencilBox(in, out, d, ii0, ii1, j0, jj0 - 1, k0, k1);
                if (jj1 < j1) stencilBox(in, out, d, ii0, ii1, jj1 + 1, j1, k0, k1);
                if (k0 < kk0) stencilBox(in, out, d, ii0, ii1, jj0, jj1, k0, kk0 - 1);
                if (kk1 < k1) stencilBox(in, out, d, ii0, ii1, jj0, jj1, kk1 + 1, k1);
            }
        }

        std::swap(in, out);
    }

    MPI_Barrier(cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // `in` points to the buffer holding the result of the last iteration
    const std::vector<Real>& localFinal = (in == gridA.data()) ? gridA : gridB;

    int status = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        gatherGrid(localFinal, d, finalGrid, nx, ny, nz, rank, size, dims, cart);

        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(finalGrid, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, cart);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return status;
}
