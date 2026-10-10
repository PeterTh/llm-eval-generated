#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Sizes = std::array<size_t, 3>;
using ProcessGrid = std::array<int, 3>;

struct Domain {
    Sizes global, local, offset;
    ProcessGrid processes, coordinates, lower, upper;
    MPI_Comm comm;
    int rank, ranks;
    size_t row, plane;

    size_t index(size_t x, size_t y, size_t z) const noexcept {
        return z * plane + y * row + x;
    }
};

// Search factorizations rather than imposing a cubic process grid on a
// rectangular problem. Prefer contiguous faces when communication costs tie.
ProcessGrid chooseProcessGrid(const Sizes& global, int available) {
    const size_t cells = global[0] * global[1] * global[2];
    for (int ranks = static_cast<int>(std::min(cells, static_cast<size_t>(available)));
         ranks > 0; --ranks) {
        std::vector<int> divisors;
        for (int d = 1; d <= ranks / d; ++d) {
            if (ranks % d == 0) {
                divisors.push_back(d);
                if (d != ranks / d) divisors.push_back(ranks / d);
            }
        }
        std::sort(divisors.begin(), divisors.end());
        double best = std::numeric_limits<double>::infinity();
        ProcessGrid result{0, 0, 0};
        for (int px : divisors) {
            if (static_cast<size_t>(px) > global[0]) continue;
            const int remaining = ranks / px;
            for (int py : divisors) {
                if (remaining % py != 0 || static_cast<size_t>(py) > global[1]) continue;
                const int pz = remaining / py;
                if (static_cast<size_t>(pz) > global[2]) continue;
                const double maxVolume = static_cast<double>((global[0] + px - 1) / px) *
                    ((global[1] + py - 1) / py) * ((global[2] + pz - 1) / pz);
                const double traffic =
                    1.05 * (px - 1) * static_cast<double>(global[1]) * global[2] +
                    1.01 * (py - 1) * static_cast<double>(global[0]) * global[2] +
                    (pz - 1) * static_cast<double>(global[0]) * global[1];
                const double cost = maxVolume + 4.0 * traffic / ranks;
                if (cost < best) {
                    best = cost;
                    result = {px, py, pz};
                }
            }
        }
        // Excess ranks are idle only when no nonempty factorization fits.
        if (result[0] != 0) return result;
    }
    return {1, 1, 1};
}

void partition(const Sizes& global, const ProcessGrid& processes,
               const ProcessGrid& coordinates, Sizes& local, Sizes& offset) {
    for (int axis = 0; axis < 3; ++axis) {
        const size_t base = global[axis] / processes[axis];
        const size_t extra = global[axis] % processes[axis];
        local[axis] = base + (static_cast<size_t>(coordinates[axis]) < extra);
        offset[axis] = coordinates[axis] * base +
            std::min(static_cast<size_t>(coordinates[axis]), extra);
    }
}

MPI_Datatype subarrayType(const Sizes& sizes, const Sizes& extent, const Sizes& offset) {
    // MPI_ORDER_C has its fastest-changing coordinate last; our X is first.
    int dimensions[3], subsizes[3], starts[3];
    for (int axis = 0; axis < 3; ++axis) {
        dimensions[2 - axis] = static_cast<int>(sizes[axis]);
        subsizes[2 - axis] = static_cast<int>(extent[axis]);
        starts[2 - axis] = static_cast<int>(offset[axis]);
    }
    MPI_Datatype type;
    MPI_Type_create_subarray(3, dimensions, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Each buffer has its own persistent requests. Face datatypes transfer directly
// between the owned cells and ghost cells, without temporary packing buffers.
class HaloExchange {
    std::array<MPI_Datatype, 3> faces{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    std::array<std::array<MPI_Request, 12>, 2> requests{};
    std::array<bool, 2> complete{true, true};
    int count = 0;

public:
    HaloExchange(const Domain& d, std::vector<Real>& first, std::vector<Real>& second) {
        const Sizes padded{d.local[0] + 2, d.local[1] + 2, d.local[2] + 2};
        for (int axis = 0; axis < 3; ++axis) {
            if (d.processes[axis] == 1) continue;
            Sizes face = d.local;
            face[axis] = 1;
            faces[axis] = subarrayType(padded, face, {0, 0, 0});
        }
        std::array<Real*, 2> buffers{first.data(), second.data()};
        // Post receives before sends when starting the persistent request set.
        for (int receiving = 1; receiving >= 0; --receiving) {
            for (int axis = 0; axis < 3; ++axis) {
                for (int side = 0; side < 2; ++side) {
                    const int neighbor = side == 0 ? d.lower[axis] : d.upper[axis];
                    if (neighbor == MPI_PROC_NULL) continue;
                    Sizes position{1, 1, 1};
                    position[axis] = receiving ? (side == 0 ? 0 : d.local[axis] + 1)
                                              : (side == 0 ? 1 : d.local[axis]);
                    const size_t location = d.index(position[0], position[1], position[2]);
                    const int tag = 2 * axis + (receiving ? 1 - side : side);
                    for (int buffer = 0; buffer < 2; ++buffer) {
                        if (receiving) {
                            MPI_Recv_init(buffers[buffer] + location, 1, faces[axis],
                                          neighbor, tag, d.comm, &requests[buffer][count]);
                        } else {
                            MPI_Send_init(buffers[buffer] + location, 1, faces[axis],
                                          neighbor, tag, d.comm, &requests[buffer][count]);
                        }
                    }
                    ++count;
                }
            }
        }
    }

    void start(int buffer) {
        if (count != 0) {
            MPI_Startall(count, requests[buffer].data());
            complete[buffer] = false;
        }
    }

    // Many MPI implementations need MPI calls to advance large transfers.
    // Stop polling once Testall has completed and deactivated the requests.
    void progress(int buffer) {
        if (!complete[buffer]) {
            int finished;
            MPI_Testall(count, requests[buffer].data(), &finished, MPI_STATUSES_IGNORE);
            complete[buffer] = finished != 0;
        }
    }

    void wait(int buffer) {
        if (!complete[buffer]) {
            MPI_Waitall(count, requests[buffer].data(), MPI_STATUSES_IGNORE);
            complete[buffer] = true;
        }
    }

    ~HaloExchange() {
        for (auto& set : requests) {
            for (int i = 0; i < count; ++i) MPI_Request_free(&set[i]);
        }
        for (auto& face : faces) {
            if (face != MPI_DATATYPE_NULL) MPI_Type_free(&face);
        }
    }
};

void initializeGrid(std::vector<Real>& first, std::vector<Real>& second, const Domain& d) {
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t globalIndex = ((d.offset[2] + z - 1) * d.global[1] +
                                        d.offset[1] + y - 1) * d.global[0] + d.offset[0];
            size_t value = globalIndex % 19;
            const size_t row = d.index(1, y, z);
            for (size_t x = 0; x < d.local[0]; ++x) {
                first[row + x] = second[row + x] = static_cast<Real>(value);
                if (++value == 19) value = 0;
            }
        }
    }
}

struct Box {
    Sizes begin, end; // half-open local coordinates
    bool empty() const noexcept {
        return begin[0] >= end[0] || begin[1] >= end[1] || begin[2] >= end[2];
    }
};

// Restrict-qualified arrays permit SIMD without changing floating-point
// expression ordering. Both buffers retain the fixed physical boundaries.
void stencilBox(const Real* __restrict input, Real* __restrict output,
                const Domain& d, const Box& box,
                HaloExchange* exchange = nullptr, int buffer = 0) {
    if (box.empty()) return;
    const size_t row = d.row, plane = d.plane;
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        if (exchange != nullptr && (z & 15) == 0) exchange->progress(buffer);
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t base = z * plane + y * row;
            for (size_t x = box.begin[0]; x < box.end[0]; ++x) {
                const size_t i = base + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - row] + input[i + row] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

Box updateBox(const Domain& d) {
    Box box{{1, 1, 1}, {d.local[0] + 1, d.local[1] + 1, d.local[2] + 1}};
    for (int axis = 0; axis < 3; ++axis) {
        if (d.offset[axis] == 0) ++box.begin[axis];
        if (d.offset[axis] + d.local[axis] == d.global[axis]) --box.end[axis];
    }
    return box;
}

// Split the halo-dependent shell into disjoint boxes, keeping branches out of
// the cell loop and avoiding repeated updates of edges and corners.
void stencilShell(const Real* input, Real* output, const Domain& d,
                  const Box& whole, const Box& core) {
    if (core.empty()) {
        stencilBox(input, output, d, whole);
        return;
    }
    Box remaining = whole;
    for (int axis = 2; axis >= 0; --axis) {
        Box face = remaining;
        face.end[axis] = core.begin[axis];
        stencilBox(input, output, d, face);
        face = remaining;
        face.begin[axis] = core.end[axis];
        stencilBox(input, output, d, face);
        remaining.begin[axis] = core.begin[axis];
        remaining.end[axis] = core.end[axis];
    }
}

void printDistributedResults(const std::vector<Real>& local, const Domain& d) {
    const Sizes padded{d.local[0] + 2, d.local[1] + 2, d.local[2] + 2};
    MPI_Datatype owned = subarrayType(padded, d.local, {1, 1, 1});
    if (d.rank == 0) {
        std::vector<Real> global(d.global[0] * d.global[1] * d.global[2]);
        for (size_t z = 0; z < d.local[2]; ++z) {
            for (size_t y = 0; y < d.local[1]; ++y) {
                const size_t destination = ((z + d.offset[2]) * d.global[1] +
                                             y + d.offset[1]) * d.global[0] + d.offset[0];
                std::copy_n(local.data() + d.index(1, y + 1, z + 1), d.local[0],
                            global.data() + destination);
            }
        }
        // Receive directly into global XYZ order so Kahan summation and the
        // order-sensitive result hash remain identical to the serial program.
        for (int rank = 1; rank < d.ranks; ++rank) {
            ProcessGrid coordinates;
            Sizes extent, offset;
            MPI_Cart_coords(d.comm, rank, 3, coordinates.data());
            partition(d.global, d.processes, coordinates, extent, offset);
            MPI_Datatype destination = subarrayType(d.global, extent, offset);
            MPI_Recv(global.data(), 1, destination, rank, 6, d.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&destination);
        }
        print_results(global, "Grid");
    } else {
        MPI_Send(local.data(), 1, owned, 0, 6, d.comm);
    }
    MPI_Type_free(&owned);
}

bool validateResult(const std::vector<Real>& grid, const Domain& d) {
    int nonfinite = 0, globalNonfinite;
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t base = d.index(1, y, z);
            for (size_t x = 0; x < d.local[0]; ++x) {
                const Real value = grid[base + x];
                nonfinite |= !std::isfinite(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
        }
    }
    Real globalMinimum, globalMaximum;
    MPI_Allreduce(&nonfinite, &globalNonfinite, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (globalNonfinite) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
    if (globalMaximum > 1e6 || globalMinimum < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
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

bool parseNumber(const char* text, size_t& number) {
    char* end;
    errno = 0;
    if (*text == '-' || *text == '\0') return false;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value > static_cast<unsigned long long>(
            std::numeric_limits<int>::max() - 2)) return false;
    number = static_cast<size_t>(value);
    return true;
}

int runBenchmark(const Sizes& global, const ProcessGrid& processes, MPI_Comm active,
                 int iterations, bool validate, bool printResults) {
    Domain d{};
    d.global = global;
    d.processes = processes;
    int periodic[3] = {0, 0, 0};
    MPI_Cart_create(active, 3, processes.data(), periodic, 0, &d.comm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.ranks);
    MPI_Cart_coords(d.comm, d.rank, 3, d.coordinates.data());
    partition(global, processes, d.coordinates, d.local, d.offset);
    for (int axis = 0; axis < 3; ++axis) {
        MPI_Cart_shift(d.comm, axis, 1, &d.lower[axis], &d.upper[axis]);
    }
    d.row = d.local[0] + 2;
    const size_t maxElements = std::vector<Real>().max_size();
    if (d.row > maxElements / (d.local[1] + 2) ||
        (d.plane = d.row * (d.local[1] + 2)) > maxElements / (d.local[2] + 2)) {
        throw std::length_error("Local grid is too large");
    }
    int result = 0;
    {
        std::vector<Real> first(d.plane * (d.local[2] + 2));
        std::vector<Real> second(first.size());
        if (d.rank == 0) printf("Initializing grid...\n");
        initializeGrid(first, second, d);
        HaloExchange exchange(d, first, second);
        const Box whole = updateBox(d);
        Box core = whole;
        for (int axis = 0; axis < 3; ++axis) {
            core.begin[axis] = std::max(core.begin[axis], size_t{2});
            core.end[axis] = std::min(core.end[axis], d.local[axis]);
        }
        const bool hasInterior = global[0] > 2 && global[1] > 2 && global[2] > 2;
        if (d.rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(d.comm);
        const double start = MPI_Wtime();
        if (hasInterior) {
            for (int iter = 0; iter < iterations; ++iter) {
                const int buffer = iter & 1;
                const auto& input = buffer == 0 ? first : second;
                auto& output = buffer == 0 ? second : first;
                exchange.start(buffer);
                stencilBox(input.data(), output.data(), d, core, &exchange, buffer);
                exchange.wait(buffer);
                stencilShell(input.data(), output.data(), d, whole, core);
            }
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
        if (d.rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            const double cellUpdates = hasInterior ? static_cast<double>(global[0] - 2) *
                (global[1] - 2) * (global[2] - 2) * iterations : 0.0;
            const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }
        const auto& finalGrid = (iterations & 1) == 0 ? first : second;
        if (printResults) printDistributedResults(finalGrid, d);
        if (validate) {
            if (d.rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(finalGrid, d);
            if (d.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Comm_free(&d.comm);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    Sizes global{128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0 ||
             strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value;
            if (!parseNumber(argv[++i], value)) {
                if (rank == 0) printf("Invalid value for -%c: %s\n", option, argv[i]);
                parseStatus = 1;
                break;
            }
            if (option == 'i') iterations = static_cast<int>(value);
            else global[option == 'x' ? 0 : option == 'y' ? 1 : 2] = value;
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
            parseStatus = 1;
            break;
        }
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    const size_t maxElements = std::vector<Real>().max_size();
    if (global[0] == 0 || global[1] == 0 || global[2] == 0 ||
        global[0] > maxElements / global[1] ||
        global[0] * global[1] > maxElements / global[2]) {
        if (rank == 0) printf("Invalid or excessively large grid dimensions\n");
        parseStatus = 1;
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    int result = 0;
    try {
        ProcessGrid processes{};
        if (rank == 0) processes = chooseProcessGrid(global, ranks);
        MPI_Bcast(processes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
        const int activeRanks = processes[0] * processes[1] * processes[2];
        MPI_Comm active;
        MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
        if (active != MPI_COMM_NULL) {
            result = runBenchmark(global, processes, active, iterations, validate, printResults);
            MPI_Comm_free(&active);
        }
        // Idle ranks also receive the validation status and finalize cleanly.
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
