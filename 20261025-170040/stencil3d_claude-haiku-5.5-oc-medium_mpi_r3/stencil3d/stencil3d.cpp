#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Domain decomposition
// ---------------------------------------------------------------------------

// Block distribution of n cells over p parts: part c owns [start, start + len).
struct Block {
    size_t start;
    size_t len;
};

inline Block blockOf(const size_t n, const int p, const int c) {
    const size_t lo = n * static_cast<size_t>(c) / static_cast<size_t>(p);
    const size_t hi = n * static_cast<size_t>(c + 1) / static_cast<size_t>(p);
    return {lo, hi - lo};
}

// Per-rank subdomain. Storage is padded by one ghost cell on every side:
// padded index = i + j*stride[1] + k*stride[2], owned cells have i,j,k in [1, len].
struct Subdomain {
    size_t start[3];   // global coordinate of the first owned cell
    size_t len[3];     // number of owned cells per axis
    size_t stride[3];  // strides of the padded storage
    size_t total;      // padded allocation size
    long long gLo[3];  // inclusive padded range of owned cells that are global interior
    long long gHi[3];
};

inline Subdomain makeSubdomain(const size_t N[3], const int dims[3], const int coords[3]) {
    Subdomain sd{};
    for (int d = 0; d < 3; ++d) {
        const Block b = blockOf(N[d], dims[d], coords[d]);
        sd.start[d] = b.start;
        sd.len[d] = b.len;
    }
    sd.stride[0] = 1;
    sd.stride[1] = sd.len[0] + 2;
    sd.stride[2] = sd.stride[1] * (sd.len[1] + 2);
    sd.total = sd.stride[2] * (sd.len[2] + 2);
    for (int d = 0; d < 3; ++d) {
        const long long s = static_cast<long long>(sd.start[d]);
        const long long L = static_cast<long long>(sd.len[d]);
        const long long n = static_cast<long long>(N[d]);
        // Global interior is [1, n-2]; padded local i maps to global s + i - 1.
        sd.gLo[d] = std::max(1LL, 2LL - s);
        sd.gHi[d] = std::min(L, n - 1 - s);
    }
    return sd;
}

// Choose a 3D process grid with px*py*pz == nprocs, each factor not exceeding the
// grid extent on its axis, minimizing the halo surface per rank.
inline bool chooseProcessGrid(const int nprocs, const size_t N[3], int dims[3]) {
    bool found = false;
    double bestCost = std::numeric_limits<double>::max();
    for (int px = 1; px <= nprocs; ++px) {
        if (nprocs % px != 0 || static_cast<size_t>(px) > N[0]) continue;
        const int rest = nprocs / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0 || static_cast<size_t>(py) > N[1]) continue;
            const int pz = rest / py;
            if (static_cast<size_t>(pz) > N[2]) continue;
            const double bx = static_cast<double>((N[0] + px - 1) / px);
            const double by = static_cast<double>((N[1] + py - 1) / py);
            const double bz = static_cast<double>((N[2] + pz - 1) / pz);
            const double cost = bx * by + by * bz + bx * bz;
            if (cost < bestCost) {
                bestCost = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
                found = true;
            }
        }
    }
    return found;
}

// ---------------------------------------------------------------------------
// Halo packing
// ---------------------------------------------------------------------------

// Number of owned cells on a face orthogonal to `dim`.
inline size_t faceCells(const Subdomain& sd, const int dim) {
    size_t n = 1;
    for (int d = 0; d < 3; ++d) {
        if (d != dim) n *= sd.len[d];
    }
    return n;
}

// Visit every owned cell of the face at padded layer `layer` along `dim`.
// Buffer order: the lower-numbered free axis varies fastest (identical on both sides).
template <typename F>
inline void forEachFaceCell(const Subdomain& sd, const int dim, const size_t layer, F&& f) {
    int a = -1, b = -1;
    for (int d = 0; d < 3; ++d) {
        if (d == dim) continue;
        if (a < 0) a = d; else b = d;
    }
    const size_t base = layer * sd.stride[dim] + sd.stride[a] + sd.stride[b];
    size_t q = 0;
    for (size_t v = 0; v < sd.len[b]; ++v) {
        const size_t rowBase = base + v * sd.stride[b];
        for (size_t u = 0; u < sd.len[a]; ++u, ++q) {
            f(q, rowBase + u * sd.stride[a]);
        }
    }
}

// Direction d: 2*dim (minus side) or 2*dim+1 (plus side).
inline size_t sendLayer(const Subdomain& sd, const int d) {
    return (d & 1) ? sd.len[d / 2] : 1;
}
inline size_t recvLayer(const Subdomain& sd, const int d) {
    return (d & 1) ? sd.len[d / 2] + 1 : 0;
}

// ---------------------------------------------------------------------------
// Stencil update
// ---------------------------------------------------------------------------

// Inclusive box in padded local coordinates.
struct Box {
    long long lo[3];
    long long hi[3];
};

// out[p0..p1] = 7-point average of in (same operation order as the serial code).
inline void stencilCells(const Real* __restrict in, Real* __restrict out,
                         const long long p0, const long long p1,
                         const size_t sy, const size_t sz) {
    const long long ly = static_cast<long long>(sy);
    const long long lz = static_cast<long long>(sz);
    for (long long p = p0; p <= p1; ++p) {
        const Real center = in[p];
        const Real left = in[p - 1];
        const Real right = in[p + 1];
        const Real front = in[p - ly];
        const Real back = in[p + ly];
        const Real bottom = in[p - lz];
        const Real top = in[p + lz];
        out[p] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

inline void copyCells(const Real* in, Real* out, const long long p0, const long long p1) {
    if (p0 > p1) return;
    std::copy(in + p0, in + p1 + 1, out + p0);
}

// Apply the update rule to every cell of `box`: global interior cells get the stencil,
// global boundary cells are copied unchanged. `in` and `out` must be distinct buffers.
void updateBox(const Real* in, Real* out, const Subdomain& sd, const Box& box) {
    const size_t sy = sd.stride[1];
    const size_t sz = sd.stride[2];
    for (long long k = box.lo[2]; k <= box.hi[2]; ++k) {
        for (long long j = box.lo[1]; j <= box.hi[1]; ++j) {
            const long long rowBase = k * static_cast<long long>(sz) + j * static_cast<long long>(sy);
            const long long rowLo = rowBase + box.lo[0];
            const long long rowHi = rowBase + box.hi[0];
            const bool yzInterior = j >= sd.gLo[1] && j <= sd.gHi[1] && k >= sd.gLo[2] && k <= sd.gHi[2];
            const long long stLo = std::max(box.lo[0], sd.gLo[0]);
            const long long stHi = std::min(box.hi[0], sd.gHi[0]);
            if (!yzInterior || stLo > stHi) {
                copyCells(in, out, rowLo, rowHi);
                continue;
            }
            copyCells(in, out, rowLo, rowBase + stLo - 1);
            stencilCells(in, out, rowBase + stLo, rowBase + stHi, sy, sz);
            copyCells(in, out, rowBase + stHi + 1, rowHi);
        }
    }
}

// ---------------------------------------------------------------------------
// Halo exchange (nonblocking, 6 faces, MPI_PROC_NULL at the global domain edge)
// ---------------------------------------------------------------------------

class HaloExchange {
  public:
    HaloExchange(MPI_Comm comm, const Subdomain& sd, const int* nbr) : comm_(comm), sd_(sd) {
        for (int d = 0; d < 6; ++d) {
            nbr_[d] = nbr[d];
            const size_t n = faceCells(sd, d / 2);
            send_[d].resize(n);
            recv_[d].resize(n);
        }
    }

    // Post receives and sends for the owned boundary layers of `in`.
    void begin(const Real* in) {
        nreq_ = 0;
        for (int d = 0; d < 6; ++d) {
            if (nbr_[d] == MPI_PROC_NULL) continue;
            // Data arriving from direction d was sent by that neighbor toward its opposite side.
            MPI_Irecv(recv_[d].data(), static_cast<int>(recv_[d].size()), MPI_DOUBLE,
                      nbr_[d], d ^ 1, comm_, &req_[nreq_++]);
        }
        for (int d = 0; d < 6; ++d) {
            if (nbr_[d] == MPI_PROC_NULL) continue;
            Real* buf = send_[d].data();
            forEachFaceCell(sd_, d / 2, sendLayer(sd_, d),
                            [&](size_t q, size_t p) { buf[q] = in[p]; });
            MPI_Isend(buf, static_cast<int>(send_[d].size()), MPI_DOUBLE,
                      nbr_[d], d, comm_, &req_[nreq_++]);
        }
    }

    // Complete all transfers and write received faces into the ghost layers of `in`.
    void end(Real* in) {
        if (nreq_ > 0) MPI_Waitall(nreq_, req_, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 6; ++d) {
            if (nbr_[d] == MPI_PROC_NULL) continue;
            const Real* buf = recv_[d].data();
            forEachFaceCell(sd_, d / 2, recvLayer(sd_, d),
                            [&](size_t q, size_t p) { in[p] = buf[q]; });
        }
    }

  private:
    MPI_Comm comm_;
    Subdomain sd_;
    int nbr_[6]{};
    std::vector<Real> send_[6];
    std::vector<Real> recv_[6];
    MPI_Request req_[12]{};
    int nreq_ = 0;
};

// ---------------------------------------------------------------------------
// Grid setup, validation, output
// ---------------------------------------------------------------------------

void initializeGrid(Real* grid, const Subdomain& sd, const size_t N[3]) {
    const size_t sy = sd.stride[1];
    const size_t sz = sd.stride[2];
    for (size_t k = 1; k <= sd.len[2]; ++k) {
        for (size_t j = 1; j <= sd.len[1]; ++j) {
            for (size_t i = 1; i <= sd.len[0]; ++i) {
                const size_t gx = sd.start[0] + i - 1;
                const size_t gy = sd.start[1] + j - 1;
                const size_t gz = sd.start[2] + k - 1;
                const size_t idx = idx3(gx, gy, gz, N[0], N[1]);
                grid[i + j * sy + k * sz] = (idx % 19) * 1.0;
            }
        }
    }
}

// Global validation over distributed data; prints on rank 0 only.
bool validateResult(const Real* grid, const Subdomain& sd, MPI_Comm comm, const bool root) {
    int localBad = 0;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();
    for (size_t k = 1; k <= sd.len[2]; ++k) {
        for (size_t j = 1; j <= sd.len[1]; ++j) {
            for (size_t i = 1; i <= sd.len[0]; ++i) {
                const Real v = grid[i + j * sd.stride[1] + k * sd.stride[2]];
                if (std::isnan(v) || std::isinf(v)) {
                    ++localBad;
                    continue;
                }
                localMin = std::min(localMin, v);
                localMax = std::max(localMax, v);
            }
        }
    }

    // 1. No NaN or Inf values
    int bad = 0;
    MPI_Allreduce(&localBad, &bad, 1, MPI_INT, MPI_SUM, comm);
    if (bad > 0) {
        if (root) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = 0.0, maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (root) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        if (root) printf("Validation failed: values out of expected range\n");
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

// Copy the owned cells of `grid` into a contiguous block (x fastest).
void packOwned(const Real* grid, const Subdomain& sd, Real* dst) {
    const size_t lx = sd.len[0];
    for (size_t k = 1; k <= sd.len[2]; ++k) {
        for (size_t j = 1; j <= sd.len[1]; ++j) {
            const Real* src = grid + j * sd.stride[1] + k * sd.stride[2] + 1;
            Real* out = dst + ((k - 1) * sd.len[1] + (j - 1)) * lx;
            std::copy(src, src + lx, out);
        }
    }
}

// Place a contiguous block of owned cells at its global position.
void placeBlock(std::vector<Real>& global, const size_t N[3], const size_t start[3],
                const size_t len[3], const Real* src) {
    for (size_t k = 0; k < len[2]; ++k) {
        for (size_t j = 0; j < len[1]; ++j) {
            const size_t dst = ((start[2] + k) * N[1] + (start[1] + j)) * N[0] + start[0];
            std::copy(src + (k * len[1] + j) * len[0], src + (k * len[1] + j + 1) * len[0],
                      global.begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool root = (worldRank == 0);

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

    const size_t N[3] = {nx, ny, nz};
    const size_t gridSize = nx * ny * nz;

    int dims[3] = {1, 1, 1};
    if (!chooseProcessGrid(worldSize, N, dims)) {
        if (root) {
            printf("Error: cannot decompose a %zu x %zu x %zu grid over %d MPI processes "
                   "(every process needs at least one cell per axis)\n",
                   nx, ny, nz, worldSize);
        }
        MPI_Finalize();
        return 1;
    }

    // Cartesian communicator; reorder=0 keeps cart rank == world rank.
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int coords[3] = {0, 0, 0};
    MPI_Cart_coords(cart, rank, 3, coords);
    int nbr[6];
    for (int d = 0; d < 3; ++d) {
        // nbr[2d] is the neighbor at -1 along d, nbr[2d+1] at +1 (MPI_PROC_NULL at edges).
        MPI_Cart_shift(cart, d, 1, &nbr[2 * d], &nbr[2 * d + 1]);
    }

    const Subdomain sd = makeSubdomain(N, dims, coords);

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (process grid %d x %d x %d)\n", worldSize, dims[0], dims[1], dims[2]);
    }

    // Allocate local grids (double buffering), padded with ghost cells
    std::vector<Real> grid1(sd.total, 0.0);
    std::vector<Real> grid2(sd.total, 0.0);

    // Initialize
    if (root) printf("Initializing grid...\n");
    initializeGrid(grid1.data(), sd, N);

    // Run stencil iterations
    if (root) printf("Running stencil computation...\n");
    HaloExchange halo(cart, sd, nbr);

    // Surface boxes: owned cells outside the inner box [2, len-1]^3 (they need ghost data).
    // Boxes may overlap; recomputing a cell is idempotent since input and output differ.
    const long long Lx = static_cast<long long>(sd.len[0]);
    const long long Ly = static_cast<long long>(sd.len[1]);
    const long long Lz = static_cast<long long>(sd.len[2]);
    const Box inner{{2, 2, 2}, {Lx - 1, Ly - 1, Lz - 1}};
    const Box surface[6] = {
        {{1, 1, 1}, {Lx, Ly, 1}},             // z low
        {{1, 1, Lz}, {Lx, Ly, Lz}},           // z high
        {{1, 1, 2}, {Lx, 1, Lz - 1}},         // y low
        {{1, Ly, 2}, {Lx, Ly, Lz - 1}},       // y high
        {{1, 2, 2}, {1, Ly - 1, Lz - 1}},     // x low
        {{Lx, 2, 2}, {Lx, Ly - 1, Lz - 1}},   // x high
    };

    MPI_Barrier(cart);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* out = (iter % 2 == 0) ? grid2.data() : grid1.data();
        Real* inMut = (iter % 2 == 0) ? grid1.data() : grid2.data();

        // Overlap: start halo exchange, compute cells that need no ghost data,
        // then finish the exchange and compute the remaining surface cells.
        halo.begin(in);
        updateBox(in, out, sd, inner);
        halo.end(inMut);
        for (const Box& b : surface) {
            updateBox(in, out, sd, b);
        }
    }

    const double localElapsed = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Allreduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, cart);

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

        // Calculate performance metrics
        const double ix = static_cast<double>(nx > 2 ? nx - 2 : 0);
        const double iy = static_cast<double>(ny > 2 ? ny - 2 : 0);
        const double iz = static_cast<double>(nz > 2 ? nz - 2 : 0);
        const double cellUpdates = ix * iy * iz * iterations;
        const double mcups = cellUpdates / elapsed / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Final grid is the one written by the last iteration (grid1 if none ran or even count)
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.data() : grid2.data();

    // Print results for external validation (gathered on rank 0 in global order)
    if (printResults) {
        std::vector<Real> local(sd.len[0] * sd.len[1] * sd.len[2]);
        packOwned(finalGrid, sd, local.data());
        if (rank == 0) {
            std::vector<Real> global(gridSize);
            placeBlock(global, N, sd.start, sd.len, local.data());
            for (int r = 1; r < worldSize; ++r) {
                int rc[3];
                MPI_Cart_coords(cart, r, 3, rc);
                size_t start[3], len[3];
                size_t cnt = 1;
                for (int d = 0; d < 3; ++d) {
                    const Block b = blockOf(N[d], dims[d], rc[d]);
                    start[d] = b.start;
                    len[d] = b.len;
                    cnt *= b.len;
                }
                std::vector<Real> tmp(cnt);
                MPI_Recv(tmp.data(), static_cast<int>(cnt), MPI_DOUBLE, r, 0, cart, MPI_STATUS_IGNORE);
                placeBlock(global, N, start, len, tmp.data());
            }
            print_results(global, "Grid");
        } else {
            MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 0, cart);
        }
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, sd, cart, root);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return exitCode;
}
