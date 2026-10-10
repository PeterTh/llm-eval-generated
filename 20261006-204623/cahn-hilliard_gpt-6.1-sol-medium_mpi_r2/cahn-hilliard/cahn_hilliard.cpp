#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
// Use the MPI C interface; deprecated MPI C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

// Choose a processor grid that minimizes halo surface for the actual domain.
// Use fewer ranks only when the grid cannot be split into nonempty blocks.
std::array<int, 3> processorGrid(const size_t n[3], int processes, int& active) {
    for (active = processes; active > 0; --active) {
        double best = std::numeric_limits<double>::infinity();
        std::array<int, 3> result{};
        for (int px = 1; px <= active; ++px) {
            if (active % px || static_cast<size_t>(px) > n[0]) continue;
            const int rest = active / px;
            for (int py = 1; py <= rest; ++py) {
                if (rest % py || static_cast<size_t>(py) > n[1]) continue;
                const int pz = rest / py;
                if (static_cast<size_t>(pz) > n[2]) continue;
                const double cost = (px - 1.0) / n[0] + (py - 1.0) / n[1]
                                  + (pz - 1.0) / n[2];
                if (cost < best) {
                    best = cost;
                    result = {px, py, pz};
                }
            }
        }
        if (result[0]) return result;
    }
    return {1, 1, 1};
}

struct Domain {
    MPI_Comm comm;
    size_t global[3], begin[3], n[3], sx, plane, storage, cells;
    int neighbor[6];
    MPI_Datatype sendFace[6], recvFace[6];
    MPI_Request requests[12];

    Domain(MPI_Comm cart, const size_t dimensions[3]) : comm(cart) {
        int dims[3], periods[3], coords[3];
        MPI_Cart_get(comm, 3, dims, periods, coords);
        for (int a = 0; a < 3; ++a) {
            global[a] = dimensions[a];
            const size_t base = global[a] / dims[a], extra = global[a] % dims[a];
            n[a] = base + (static_cast<size_t>(coords[a]) < extra);
            begin[a] = base * coords[a] + std::min(extra, static_cast<size_t>(coords[a]));
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
            if (n[a] > INT_MAX - 2) {
                fprintf(stderr, "Local grid dimension exceeds MPI datatype limits\n");
                MPI_Abort(comm, 1);
            }
        }
        const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
        if (n[0]+2 > limit/(n[1]+2) || (n[0]+2)*(n[1]+2) > limit/(n[2]+2)) {
            fprintf(stderr, "Local grid including halos exceeds addressable memory\n");
            MPI_Abort(comm, 1);
        }
        sx = n[0] + 2;
        plane = sx * (n[1] + 2);
        storage = plane * (n[2] + 2);
        cells = n[0] * n[1] * n[2];
        // MPI subarrays have int dimensions, but address displacements use MPI_Aint.
        int sizes[3] = {static_cast<int>(n[2]+2), static_cast<int>(n[1]+2), static_cast<int>(n[0]+2)};
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                const int face = 2*a + side;
                int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
                int starts[3] = {1, 1, 1};
                subs[2-a] = 1;
                starts[2-a] = side ? static_cast<int>(n[a]) : 1;
                MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &sendFace[face]);
                MPI_Type_commit(&sendFace[face]);
                starts[2-a] = side ? static_cast<int>(n[a]+1) : 0;
                MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &recvFace[face]);
                MPI_Type_commit(&recvFace[face]);
            }
        }
    }

    ~Domain() {
        for (int f = 0; f < 6; ++f) {
            MPI_Type_free(&sendFace[f]);
            MPI_Type_free(&recvFace[f]);
        }
    }

    size_t index(size_t x, size_t y, size_t z) const { return z*plane + y*sx + x; }

    void exchange(std::vector<double>& field) {
        for (int f = 0; f < 6; ++f)
            MPI_Irecv(field.data(), 1, recvFace[f], neighbor[f], f^1, comm, &requests[f]);
        for (int f = 0; f < 6; ++f)
            MPI_Isend(field.data(), 1, sendFace[f], neighbor[f], f, comm, &requests[6+f]);
    }

    void finishExchange() { MPI_Waitall(12, requests, MPI_STATUSES_IGNORE); }
};

// The physical boundary is clamped independently for both stencils. Internal
// block boundaries instead use halos; edge/corner halo cells are never needed.
inline double laplacian(const double* field, size_t i, size_t xm, size_t xp,
                        size_t ym, size_t yp, size_t zm, size_t zp) {
    const double center = 2.0 * field[i];
    const double cxx = field[i+xp] + field[i-xm] - center;
    const double cyy = field[i+yp] + field[i-ym] - center;
    const double czz = field[i+zp] + field[i-zm] - center;
    return cxx + cyy + czz;
}

template<bool Chemical>
void computeBox(const Domain& d, const std::vector<double>& source,
                const std::vector<double>& cold, std::vector<double>& dest,
                size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double* __restrict__ field = source.data();
    const double* __restrict__ previous = cold.data();
    double* __restrict__ output = dest.data();
    for (size_t z = z0; z < z1; ++z) {
        const size_t zm = (z == 1 && d.neighbor[4] == MPI_PROC_NULL) ? 0 : d.plane;
        const size_t zp = (z == d.n[2] && d.neighbor[5] == MPI_PROC_NULL) ? 0 : d.plane;
        for (size_t y = y0; y < y1; ++y) {
            const size_t ym = (y == 1 && d.neighbor[2] == MPI_PROC_NULL) ? 0 : d.sx;
            const size_t yp = (y == d.n[1] && d.neighbor[3] == MPI_PROC_NULL) ? 0 : d.sx;
            const size_t row = d.index(0, y, z);
            const auto point = [&](size_t x, size_t xm, size_t xp) {
                const size_t i = row + x;
                const double lap = laplacian(field, i, xm, xp, ym, yp, zm, zp);
                if constexpr (Chemical) {
                    const double cv = field[i];
                    output[i] = 4.5 * ((cv + 1.0)*e_AA + (cv - 1.0)*e_BB - 2.0*cv*e_AB)
                              + 3.0*cv + cv*cv*cv - 0.5*lap;
                } else {
                    output[i] = previous[i] + 0.01*lap;
                }
            };
            // Handle the two x edges separately so the contiguous inner loop
            // has fixed neighbor offsets and can use SIMD without gathers.
            size_t first = x0, last = x1;
            if (first == 1 && first < last) {
                point(first, d.neighbor[0] != MPI_PROC_NULL,
                      !(first == d.n[0] && d.neighbor[1] == MPI_PROC_NULL));
                ++first;
            }
            if (last == d.n[0]+1 && first < last) {
                --last;
                point(last, 1, d.neighbor[1] != MPI_PROC_NULL);
            }
            for (size_t x = first; x < last; ++x) point(x, 1, 1);
        }
    }
}

template<bool Chemical>
void stencil(Domain& d, std::vector<double>& source,
             const std::vector<double>& cold, std::vector<double>& dest) {
    d.exchange(source);
    const size_t ex = d.n[0]+1, ey = d.n[1]+1, ez = d.n[2]+1;
    // Only cells touching a remote face must wait for communication.
    const size_t x0 = d.neighbor[0] == MPI_PROC_NULL ? 1 : 2;
    const size_t y0 = d.neighbor[2] == MPI_PROC_NULL ? 1 : 2;
    const size_t z0 = d.neighbor[4] == MPI_PROC_NULL ? 1 : 2;
    const size_t x1 = std::max(x0, ex - (d.neighbor[1] != MPI_PROC_NULL));
    const size_t y1 = std::max(y0, ey - (d.neighbor[3] != MPI_PROC_NULL));
    const size_t z1 = std::max(z0, ez - (d.neighbor[5] != MPI_PROC_NULL));
    if (x0 < ex && y0 < ey && z0 < ez)
        computeBox<Chemical>(d, source, cold, dest, x0, x1, y0, y1, z0, z1);
    d.finishExchange();
    // Thin domains have no interior. Compute them once after the exchange.
    if (x0 >= x1 || y0 >= y1 || z0 >= z1 || x0 >= ex || y0 >= ey || z0 >= ez) {
        computeBox<Chemical>(d, source, cold, dest, 1, ex, 1, ey, 1, ez);
        return;
    }
    // Six disjoint boxes cover the boundary without per-cell branching.
    computeBox<Chemical>(d, source, cold, dest, 1, ex, 1, ey, 1, z0);
    computeBox<Chemical>(d, source, cold, dest, 1, ex, 1, ey, z1, ez);
    computeBox<Chemical>(d, source, cold, dest, 1, ex, 1, y0, z0, z1);
    computeBox<Chemical>(d, source, cold, dest, 1, ex, y1, ey, z0, z1);
    computeBox<Chemical>(d, source, cold, dest, 1, x0, y0, y1, z0, z1);
    computeBox<Chemical>(d, source, cold, dest, x1, ex, y0, y1, z0, z1);
}

void initializeConcentration(const Domain& d, std::vector<double>& c) {
    const size_t vol = d.global[0]*d.global[1]*d.global[2];
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const size_t id = ((d.begin[2]+z-1)*d.global[1] + d.begin[1]+y-1)*d.global[0] + d.begin[0]+x-1;
                const double pseudo = (((id+1)*1299709) % vol) / static_cast<double>(vol);
                c[d.index(x,y,z)] = -1.0 + 2.0*pseudo;
            }
}

// Gather only when requested, preserving the serial order (including its hash
// and Kahan sum). Chunked messages avoid MPI's int count limit.
void printDistributedResults(const Domain& d, const std::vector<double>& c, int rank, int processes) {
    std::vector<double> all;
    if (rank == 0) all.resize(d.global[0]*d.global[1]*d.global[2]);
    constexpr size_t chunkLimit = 1048576;
    std::vector<double> buffer(std::min(chunkLimit, d.cells));
    for (int r = 0; r < processes; ++r) {
        unsigned long long meta[6];
        if (rank == r) {
            for (int a = 0; a < 3; ++a) { meta[a] = d.begin[a]; meta[a+3] = d.n[a]; }
            if (r != 0) MPI_Send(meta, 6, MPI_UNSIGNED_LONG_LONG, 0, 20, d.comm);
        }
        if (rank == 0 && r != 0) MPI_Recv(meta, 6, MPI_UNSIGNED_LONG_LONG, r, 20, d.comm, MPI_STATUS_IGNORE);
        if (rank != 0 && rank != r) continue;
        const size_t cells = static_cast<size_t>(meta[3])*meta[4]*meta[5];
        for (size_t offset = 0; offset < cells; ) {
            const size_t count = std::min(chunkLimit, cells-offset);
            if (buffer.size() < count) buffer.resize(count);
            if (rank == r) {
                for (size_t j = 0; j < count; ++j) {
                    const size_t k = offset+j;
                    buffer[j] = c[d.index(k % d.n[0]+1, (k/d.n[0]) % d.n[1]+1, k/(d.n[0]*d.n[1])+1)];
                }
                if (r != 0) MPI_Send(buffer.data(), static_cast<int>(count), MPI_DOUBLE, 0, 21, d.comm);
            } else {
                MPI_Recv(buffer.data(), static_cast<int>(count), MPI_DOUBLE, r, 21, d.comm, MPI_STATUS_IGNORE);
            }
            if (rank == 0) {
                for (size_t j = 0; j < count; ++j) {
                    const size_t k = offset+j;
                    const size_t x = meta[0]+k % meta[3];
                    const size_t y = meta[1]+(k/meta[3]) % meta[4];
                    const size_t z = meta[2]+k/(meta[3]*meta[4]);
                    all[(z*d.global[1]+y)*d.global[0]+x] = buffer[j];
                }
            }
            offset += count;
        }
    }
    if (rank == 0) print_results(all, "Concentration");
}

bool validateResult(const Domain& d, const std::vector<double>& c, int rank) {
    int invalid = 0, anyInvalid;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const double v = c[d.index(x,y,z)];
                invalid |= !std::isfinite(v);
                low = std::min(low, v);
                high = std::max(high, v);
            }
    double globalLow, globalHigh;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (rank == 0) {
        if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", globalLow, globalHigh);
            if (globalHigh > 10.0 || globalLow < -10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !anyInvalid && globalHigh <= 10.0 && globalLow >= -10.0;
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
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
    

    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > limit/ny || nx*ny > limit/nz) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive and fit in memory and MPI datatypes\n");
        MPI_Finalize();
        return 1;
    }
    const size_t dimensions[3] = {nx, ny, nz};
    int active;
    const auto dims = processorGrid(dimensions, processes, active);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    int status = 0;
    if (rank < active) {
        int cartDims[3] = {dims[0], dims[1], dims[2]}, periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(workers, 3, cartDims, periods, 0, &cart);
        {
            Domain d(cart, dimensions);
            std::vector<double> cold(d.storage), cnew(d.storage), mu(d.storage);
            if (rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("Initializing concentration field...\n");
            }
            initializeConcentration(d, cold);
            if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                stencil<true>(d, cold, cold, mu);
                stencil<false>(d, mu, cold, cnew);
                cold.swap(cnew);
            }
            const double elapsed = MPI_Wtime() - start;
            double duration;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (rank == 0) {
                printf("Computation time: %ld ms\n", static_cast<long>(duration*1000.0));
                const double cellUpdates = static_cast<double>(nx)*ny*nz*iterations;
                printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / duration / 1e6);
            }
            if (printResults) printDistributedResults(d, cold, rank, active);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateResult(d, cold, rank);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    // Idle ranks participate in shutdown and return the same validation status.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
