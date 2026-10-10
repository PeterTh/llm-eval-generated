#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Shape = std::array<size_t, 3>;
using ProcessGrid = std::array<int, 3>;

struct Box {
    Shape lo, hi; // Half-open local coordinates, with owned cells starting at 1.
};

struct Domain {
    Shape global, n, origin;
    ProcessGrid dims, coords;
    std::array<int, 6> neighbors;
    MPI_Comm comm;
    size_t row, plane, storage;
    int rank, ranks;

    Domain(const Shape& grid, const ProcessGrid& processGrid, MPI_Comm active)
        : global(grid), dims(processGrid) {
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims.data(), periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        MPI_Cart_coords(comm, rank, 3, coords.data());
        for (int a = 0; a < 3; ++a) {
            const size_t base = global[a] / dims[a];
            const size_t extra = global[a] % dims[a];
            n[a] = base + (static_cast<size_t>(coords[a]) < extra);
            origin[a] = base * coords[a] + std::min(static_cast<size_t>(coords[a]), extra);
            MPI_Cart_shift(comm, a, 1, &neighbors[2 * a], &neighbors[2 * a + 1]);
        }
        row = n[0] + 2;
        if (n[1] + 2 > std::numeric_limits<size_t>::max() / row)
            throw std::overflow_error("Local grid is too large");
        plane = row * (n[1] + 2);
        if (n[2] + 2 > std::numeric_limits<size_t>::max() / plane)
            throw std::overflow_error("Local grid is too large");
        storage = plane * (n[2] + 2);
    }

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }

    Box interior() const {
        Box b;
        for (int a = 0; a < 3; ++a) {
            b.lo[a] = neighbors[2 * a] == MPI_PROC_NULL ? 1 : 2;
            b.hi[a] = neighbors[2 * a + 1] == MPI_PROC_NULL ? n[a] + 1 : n[a];
            b.hi[a] = std::max(b.lo[a], b.hi[a]);
        }
        return b;
    }
};

// Favor balanced blocks with little inter-rank surface area. Enumerating
// factorizations also handles thin grids and prime process counts safely.
ProcessGrid chooseProcessGrid(const Shape& n, int available, int& active) {
    for (active = available; active > 0; --active) {
        ProcessGrid best = {0, 0, 0};
        long double bestCost = std::numeric_limits<long double>::infinity();
        for (int px = 1; px <= active && static_cast<size_t>(px) <= n[0]; ++px) {
            if (active % px != 0) continue;
            const int rest = active / px;
            for (int py = 1; py <= rest && static_cast<size_t>(py) <= n[1]; ++py) {
                if (rest % py != 0) continue;
                const int pz = rest / py;
                if (static_cast<size_t>(pz) > n[2]) continue;
                const long double maxVolume =
                    static_cast<long double>(n[0] / px + (n[0] % px != 0)) *
                    (n[1] / py + (n[1] % py != 0)) * (n[2] / pz + (n[2] % pz != 0));
                // Splitting X costs slightly more because its faces are strided.
                const long double surface =
                    1.1L * (px - 1) * n[1] * n[2] +
                    static_cast<long double>(py - 1) * n[0] * n[2] +
                    static_cast<long double>(pz - 1) * n[0] * n[1];
                const long double cost = maxVolume + 2.0L * surface / active;
                if (cost < bestCost) {
                    bestCost = cost;
                    best = {px, py, pz};
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

// Contiguous face buffers avoid expensive MPI packing of strided X/Y faces.
// Persistent requests and buffers are reused for both stencil passes and all
// time steps, including when the concentration vectors swap storage.
class HaloExchange {
    const Domain& d;
    std::array<std::vector<double>, 6> send, receive;
    std::vector<MPI_Request> requests;

    void copyFace(double* field, double* buffer, int face, bool unpack) const {
        const int axis = face / 2;
        const size_t coordinate = unpack ? ((face % 2) ? d.n[axis] + 1 : 0)
                                         : ((face % 2) ? d.n[axis] : 1);
        if (axis == 0) {
            size_t k = 0;
            for (size_t z = 1; z <= d.n[2]; ++z)
                for (size_t y = 1; y <= d.n[1]; ++y) {
                    double& v = field[d.index(coordinate, y, z)];
                    if (unpack) v = buffer[k++];
                    else buffer[k++] = v;
                }
        } else if (axis == 1) {
            for (size_t z = 1; z <= d.n[2]; ++z) {
                double* row = field + d.index(1, coordinate, z);
                double* packed = buffer + (z - 1) * d.n[0];
                if (unpack) std::copy_n(packed, d.n[0], row);
                else std::copy_n(row, d.n[0], packed);
            }
        } else {
            for (size_t y = 1; y <= d.n[1]; ++y) {
                double* row = field + d.index(1, y, coordinate);
                double* packed = buffer + (y - 1) * d.n[0];
                if (unpack) std::copy_n(packed, d.n[0], row);
                else std::copy_n(row, d.n[0], packed);
            }
        }
    }

    void clampFace(double* field, int face) const {
        const int axis = face / 2;
        const size_t ghost = (face % 2) ? d.n[axis] + 1 : 0;
        const size_t owned = (face % 2) ? d.n[axis] : 1;
        if (axis == 0) {
            for (size_t z = 1; z <= d.n[2]; ++z)
                for (size_t y = 1; y <= d.n[1]; ++y)
                    field[d.index(ghost, y, z)] = field[d.index(owned, y, z)];
        } else if (axis == 1) {
            for (size_t z = 1; z <= d.n[2]; ++z)
                std::copy_n(field + d.index(1, owned, z), d.n[0], field + d.index(1, ghost, z));
        } else {
            for (size_t y = 1; y <= d.n[1]; ++y)
                std::copy_n(field + d.index(1, y, owned), d.n[0], field + d.index(1, y, ghost));
        }
    }

public:
    explicit HaloExchange(const Domain& domain) : d(domain) {
        // Post receives before sends. Split faces if they exceed MPI's int count.
        int* tagLimit = nullptr;
        int hasTagLimit = 0;
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tagLimit, &hasTagLimit);
        const int maxTag = hasTagLimit ? *tagLimit : 32767;
        auto tag = [maxTag](int face, size_t offset) {
            const size_t chunkIndex = offset / static_cast<size_t>(INT_MAX);
            if (chunkIndex > static_cast<size_t>((maxTag - face) / 6))
                throw std::overflow_error("Halo requires too many MPI messages");
            // Startall may activate requests in any order, so chunks need
            // distinct tags rather than relying on message ordering.
            return face + 6 * static_cast<int>(chunkIndex);
        };
        for (int face = 0; face < 6; ++face) {
            if (d.neighbors[face] == MPI_PROC_NULL) continue;
            const int axis = face / 2;
            const size_t count = d.n[(axis + 1) % 3] * d.n[(axis + 2) % 3];
            send[face].resize(count);
            receive[face].resize(count);
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                MPI_Request r;
                MPI_Recv_init(receive[face].data() + offset, chunk, MPI_DOUBLE,
                              d.neighbors[face], tag(face ^ 1, offset), d.comm, &r);
                requests.push_back(r);
                offset += chunk;
            }
        }
        for (int face = 0; face < 6; ++face) {
            if (d.neighbors[face] == MPI_PROC_NULL) continue;
            for (size_t offset = 0; offset < send[face].size();) {
                const int chunk = static_cast<int>(std::min(send[face].size() - offset,
                                                           static_cast<size_t>(INT_MAX)));
                MPI_Request r;
                MPI_Send_init(send[face].data() + offset, chunk, MPI_DOUBLE,
                              d.neighbors[face], tag(face, offset), d.comm, &r);
                requests.push_back(r);
                offset += chunk;
            }
        }
    }

    ~HaloExchange() {
        for (auto& r : requests) MPI_Request_free(&r);
    }

    void begin(std::vector<double>& field) {
        for (int face = 0; face < 6; ++face) {
            if (d.neighbors[face] == MPI_PROC_NULL) clampFace(field.data(), face);
            else copyFace(field.data(), send[face].data(), face, false);
        }
        if (!requests.empty()) MPI_Startall(static_cast<int>(requests.size()), requests.data());
    }

    void finish(std::vector<double>& field) {
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        for (int face = 0; face < 6; ++face)
            if (d.neighbors[face] != MPI_PROC_NULL)
                copyFace(field.data(), receive[face].data(), face, true);
    }
};

inline double laplacian(const double* c, size_t i, size_t row, size_t plane) {
    // All three grid spacings are 1. Preserve the original arithmetic order.
    const double cxx = c[i + 1] + c[i - 1] - 2.0 * c[i];
    const double cyy = c[i + row] + c[i - row] - 2.0 * c[i];
    const double czz = c[i + plane] + c[i - plane] - 2.0 * c[i];
    return cxx + cyy + czz;
}

// Separate template instantiations give branch-free, vectorizable inner loops.
template<bool chemical>
void computeBox(double* __restrict output, const double* __restrict cold,
                const double* __restrict mu, const Domain& d, const Box& b) {
    if (b.lo[0] == b.hi[0] || b.lo[1] == b.hi[1] || b.lo[2] == b.hi[2]) return;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t z = b.lo[2]; z < b.hi[2]; ++z) {
        for (size_t y = b.lo[1]; y < b.hi[1]; ++y) {
            const size_t base = d.index(0, y, z);
            for (size_t x = b.lo[0]; x < b.hi[0]; ++x) {
                const size_t i = base + x;
                if constexpr (chemical) {
                    const double cv = cold[i];
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv
                              - 0.5 * laplacian(cold, i, d.row, d.plane);
                } else {
                    output[i] = cold[i] + 0.01 * laplacian(mu, i, d.row, d.plane);
                }
            }
        }
    }
}

template<bool chemical>
void computeBoundary(double* output, const double* cold, const double* mu,
                     const Domain& d, const Box& interior) {
    // Six disjoint boxes cover exactly the cells excluded from the interior,
    // including blocks only one or two cells wide. No per-cell boundary branch.
    Box remaining{{1, 1, 1}, {d.n[0] + 1, d.n[1] + 1, d.n[2] + 1}};
    for (int axis = 2; axis >= 0; --axis) {
        Box face = remaining;
        face.hi[axis] = interior.lo[axis];
        computeBox<chemical>(output, cold, mu, d, face);
        face = remaining;
        face.lo[axis] = interior.hi[axis];
        computeBox<chemical>(output, cold, mu, d, face);
        remaining.lo[axis] = interior.lo[axis];
        remaining.hi[axis] = interior.hi[axis];
    }
}

void initializeConcentration(std::vector<double>& c, const Domain& d, size_t volume) {
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const size_t globalId = ((d.origin[2] + z - 1) * d.global[1] +
                                        d.origin[1] + y - 1) * d.global[0] + d.origin[0] + x - 1;
                const double pseudo = (((globalId + 1) * 1299709) % volume) / static_cast<double>(volume);
                c[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
}

bool validateResult(const std::vector<double>& c, const Domain& d) {
    int invalid = 0;
    double extrema[2] = {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::infinity()};
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const double v = c[d.index(x, y, z)];
                invalid |= !std::isfinite(v);
                extrema[0] = std::min(extrema[0], v);
                extrema[1] = std::min(extrema[1], -v);
            }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MIN, d.comm);
    if (invalid) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (d.rank == 0) printf("Concentration range: [%.6f, %.6f]\n", extrema[0], -extrema[1]);
    if (extrema[0] < -10.0 || -extrema[1] > 10.0) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printDistributedResults(const std::vector<double>& c, const Domain& d, size_t volume) {
    // Gather only on request, outside the timed solver. Reassemble the original
    // X-fastest order so Kahan summation, samples, and the byte hash are unchanged.
    std::vector<double> packed(d.n[0] * d.n[1] * d.n[2]);
    size_t offset = 0;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y) {
            std::copy_n(c.data() + d.index(1, y, z), d.n[0], packed.data() + offset);
            offset += d.n[0];
        }
    if (d.rank != 0) {
        for (offset = 0; offset < packed.size();) {
            const int count = static_cast<int>(std::min(packed.size() - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(packed.data() + offset, count, MPI_DOUBLE, 0, 42, d.comm);
            offset += count;
        }
        return;
    }
    std::vector<double> global(volume);
    for (int source = 0; source < d.ranks; ++source) {
        ProcessGrid coords;
        MPI_Cart_coords(d.comm, source, 3, coords.data());
        Shape n, origin;
        for (int a = 0; a < 3; ++a) {
            const size_t base = d.global[a] / d.dims[a], extra = d.global[a] % d.dims[a];
            n[a] = base + (static_cast<size_t>(coords[a]) < extra);
            origin[a] = base * coords[a] + std::min(static_cast<size_t>(coords[a]), extra);
        }
        if (source != 0) {
            packed.resize(n[0] * n[1] * n[2]);
            for (offset = 0; offset < packed.size();) {
                const int count = static_cast<int>(std::min(packed.size() - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(packed.data() + offset, count, MPI_DOUBLE, source, 42, d.comm, MPI_STATUS_IGNORE);
                offset += count;
            }
        }
        offset = 0;
        for (size_t z = 0; z < n[2]; ++z)
            for (size_t y = 0; y < n[1]; ++y) {
                const size_t globalId = ((origin[2] + z) * d.global[1] + origin[1] + y) * d.global[0] + origin[0];
                std::copy_n(packed.data() + offset, n[0], global.data() + globalId);
                offset += n[0];
            }
    }
    print_results(global, "Concentration");
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

bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

int run(int argc, char** argv, int worldRank, int worldSize) {
    Shape grid = {64, 0, 0};
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0 ||
             strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            const char* argument = argv[++i];
            bool valid;
            if (option == 'i') {
                errno = 0;
                char* end = nullptr;
                const long value = std::strtol(argument, &end, 10);
                valid = errno == 0 && end != argument && *end == '\0' && value >= INT_MIN && value <= INT_MAX;
                if (valid) iterations = static_cast<int>(value);
            } else {
                size_t value;
                valid = parseSize(argument, value);
                if (valid) grid[option == 'x' ? 0 : option == 'y' ? 1 : 2] = value;
            }
            if (!valid) {
                if (worldRank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (grid[1] == 0) grid[1] = grid[0];
    if (grid[2] == 0) grid[2] = grid[0];
    size_t volume = 1;
    for (const size_t n : grid) {
        if (n == 0 || n > std::numeric_limits<size_t>::max() - 2 ||
            n > std::numeric_limits<size_t>::max() / volume) {
            if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive and fit in memory\n");
            return 1;
        }
        volume *= n;
    }
    int activeRanks;
    const ProcessGrid dims = chooseProcessGrid(grid, static_cast<int>(std::min(volume, static_cast<size_t>(worldSize))), activeRanks);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (active == MPI_COMM_NULL) return 0;
    Domain d(grid, dims, active);
    MPI_Comm_free(&active);
    if (d.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", grid[0], grid[1], grid[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold(d.storage), cnew(d.storage), mu(d.storage);
    initializeConcentration(cold, d, volume);
    double elapsed;
    {
        HaloExchange halo(d);
        const Box interior = d.interior();
        if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(d.comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            halo.begin(cold);
            computeBox<true>(mu.data(), cold.data(), nullptr, d, interior);
            halo.finish(cold);
            computeBoundary<true>(mu.data(), cold.data(), nullptr, d, interior);
            halo.begin(mu);
            computeBox<false>(cnew.data(), cold.data(), mu.data(), d, interior);
            halo.finish(mu);
            computeBoundary<false>(cnew.data(), cold.data(), mu.data(), d, interior);
            cold.swap(cnew);
        }
        elapsed = MPI_Wtime() - start;
    }
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    if (d.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(maximumElapsed * 1000.0));
        const double mcups = maximumElapsed > 0.0 ? static_cast<double>(volume) * iterations / maximumElapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) printDistributedResults(cold, d, volume);
    bool valid = true;
    if (validate) {
        if (d.rank == 0) printf("Validating result...\n");
        valid = validateResult(cold, d);
        if (d.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&d.comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
