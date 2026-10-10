// Use the MPI C interface without the deprecated MPI C++ bindings.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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

namespace {

size_t checkedProduct(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("Grid size is too large");
    return a * b;
}

bool parseSize(const char* text, size_t& value) {
    if (*text == '-' || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long number = std::strtoull(text, &end, 10);
    if (errno || *end != '\0' || number > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(number);
    return true;
}

bool parseIterations(const char* text, int& value) {
    char* end = nullptr;
    errno = 0;
    const long number = std::strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || number < INT_MIN || number > INT_MAX) return false;
    value = static_cast<int>(number);
    return true;
}

// Minimize internal face area, with a small penalty for packing strided faces
// and for uneven blocks. Prefer Z, then Y, to keep long contiguous X rows.
std::array<int, 3> processGrid(const std::array<size_t, 3>& global, int available) {
    const size_t volume = checkedProduct(checkedProduct(global[0], global[1]), global[2]);
    const int limit = static_cast<int>(std::min(volume, static_cast<size_t>(available)));
    for (int processes = limit; processes > 0; --processes) {
        std::array<int, 3> best{0, 0, 0};
        long double bestCost = std::numeric_limits<long double>::infinity();
        for (int px = 1; px <= processes && static_cast<size_t>(px) <= global[0]; ++px) {
            if (processes % px != 0) continue;
            const int rest = processes / px;
            for (int py = 1; py <= rest && static_cast<size_t>(py) <= global[1]; ++py) {
                if (rest % py != 0) continue;
                const int pz = rest / py;
                if (static_cast<size_t>(pz) > global[2]) continue;
                const long double area =
                    1.15L * (px - 1) * global[1] * global[2] +
                    1.05L * (py - 1) * global[0] * global[2] +
                    (pz - 1) * static_cast<long double>(global[0]) * global[1];
                const long double largestBlock =
                    static_cast<long double>(global[0] / px + (global[0] % px != 0)) *
                    (global[1] / py + (global[1] % py != 0)) *
                    (global[2] / pz + (global[2] % pz != 0));
                const long double imbalance = largestBlock * processes / volume;
                const long double cost = area * imbalance;
                if (cost < bestCost) {
                    bestCost = cost;
                    best = {px, py, pz};
                }
            }
        }
        // Tiny grids may have fewer nonempty Cartesian blocks than MPI ranks.
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

void blockExtent(size_t global, int divisions, int coordinate, size_t& count, size_t& start) {
    const size_t base = global / divisions;
    const size_t remainder = global % divisions;
    count = base + (static_cast<size_t>(coordinate) < remainder);
    start = static_cast<size_t>(coordinate) * base +
            std::min(static_cast<size_t>(coordinate), remainder);
}

struct Box {
    size_t x0, x1, y0, y1, z0, z1; // Half-open ranges, excluding padding.
    bool empty() const { return x0 >= x1 || y0 >= y1 || z0 >= z1; }
};

struct Domain {
    MPI_Comm comm;
    std::array<size_t, 3> global, size{}, origin{};
    std::array<int, 3> divisions;
    std::array<int, 6> neighbor{};
    size_t row, plane, storage;
    Box whole, interior;

    Domain(MPI_Comm communicator, const std::array<size_t, 3>& dimensions,
           const std::array<int, 3>& grid) : comm(communicator), global(dimensions), divisions(grid) {
        int rank, coordinates[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coordinates);
        for (int axis = 0; axis < 3; ++axis) {
            blockExtent(global[axis], divisions[axis], coordinates[axis], size[axis], origin[axis]);
            MPI_Cart_shift(comm, axis, 1, &neighbor[2 * axis], &neighbor[2 * axis + 1]);
            if (size[axis] > std::numeric_limits<size_t>::max() - 2)
                throw std::overflow_error("Local grid size is too large");
        }
        row = size[0] + 2;
        plane = checkedProduct(row, size[1] + 2);
        storage = checkedProduct(plane, size[2] + 2);
        whole = {1, size[0] + 1, 1, size[1] + 1, 1, size[2] + 1};
        interior = {1 + size_t(neighbor[0] != MPI_PROC_NULL),
                    size[0] + 1 - size_t(neighbor[1] != MPI_PROC_NULL),
                    1 + size_t(neighbor[2] != MPI_PROC_NULL),
                    size[1] + 1 - size_t(neighbor[3] != MPI_PROC_NULL),
                    1 + size_t(neighbor[4] != MPI_PROC_NULL),
                    size[2] + 1 - size_t(neighbor[5] != MPI_PROC_NULL)};
    }

    size_t index(size_t x, size_t y, size_t z) const { return z * plane + y * row + x; }
};

// Contiguous Z faces travel directly from/to the field. Pack only the strided
// X/Y faces into reusable buffers. No edges/corners are needed by the stencil.
class HaloExchange {
    const Domain& d;
    std::array<std::vector<double>, 4> sendBuffer, receiveBuffer;
    std::vector<MPI_Request> requests;

    void post(double* data, size_t count, int peer, int tag, bool receive) {
        for (size_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
            requests.push_back(MPI_REQUEST_NULL);
            if (receive)
                MPI_Irecv(data + offset, chunk, MPI_DOUBLE, peer, tag, d.comm, &requests.back());
            else
                MPI_Isend(data + offset, chunk, MPI_DOUBLE, peer, tag, d.comm, &requests.back());
            offset += chunk;
        }
    }

public:
    explicit HaloExchange(const Domain& domain) : d(domain) {
        requests.reserve(12);
        for (int face = 0; face < 4; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            const size_t count = checkedProduct(face < 2 ? d.size[1] : d.size[0], d.size[2]);
            sendBuffer[face].resize(count);
            receiveBuffer[face].resize(count);
        }
    }

    void begin(std::vector<double>& field) {
        double* a = field.data();
        requests.clear();
        // Clamping at global boundaries is independent for each field.
        for (size_t z = 1; z <= d.size[2]; ++z) {
            for (size_t y = 1; y <= d.size[1]; ++y) {
                const size_t i = d.index(1, y, z);
                if (d.neighbor[0] == MPI_PROC_NULL) a[i - 1] = a[i];
                if (d.neighbor[1] == MPI_PROC_NULL) a[i + d.size[0]] = a[i + d.size[0] - 1];
            }
            if (d.neighbor[2] == MPI_PROC_NULL)
                std::copy_n(a + d.index(1, 1, z), d.size[0], a + d.index(1, 0, z));
            if (d.neighbor[3] == MPI_PROC_NULL)
                std::copy_n(a + d.index(1, d.size[1], z), d.size[0], a + d.index(1, d.size[1] + 1, z));
        }
        if (d.neighbor[4] == MPI_PROC_NULL) std::copy_n(a + d.plane, d.plane, a);
        if (d.neighbor[5] == MPI_PROC_NULL)
            std::copy_n(a + d.size[2] * d.plane, d.plane, a + (d.size[2] + 1) * d.plane);

        // Post every receive before packing/sending to allow early progress.
        for (int face = 0; face < 6; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            double* destination = face < 4 ? receiveBuffer[face].data() :
                                  a + (face == 4 ? 0 : (d.size[2] + 1) * d.plane);
            const size_t count = face < 4 ? receiveBuffer[face].size() : d.plane;
            post(destination, count, d.neighbor[face], face ^ 1, true);
        }
        for (int face = 0; face < 6; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            if (face < 2) {
                const size_t x = face == 0 ? 1 : d.size[0];
                size_t offset = 0;
                for (size_t z = 1; z <= d.size[2]; ++z)
                    for (size_t y = 1; y <= d.size[1]; ++y)
                        sendBuffer[face][offset++] = a[d.index(x, y, z)];
            } else if (face < 4) {
                const size_t y = face == 2 ? 1 : d.size[1];
                for (size_t z = 1; z <= d.size[2]; ++z)
                    std::copy_n(a + d.index(1, y, z), d.size[0],
                                sendBuffer[face].data() + (z - 1) * d.size[0]);
            }
            double* source = face < 4 ? sendBuffer[face].data() :
                             a + (face == 4 ? 1 : d.size[2]) * d.plane;
            const size_t count = face < 4 ? sendBuffer[face].size() : d.plane;
            post(source, count, d.neighbor[face], face, false);
        }
    }

    void finish(std::vector<double>& field) {
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        for (int face = 0; face < 4; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            if (face < 2) {
                const size_t x = face == 0 ? 0 : d.size[0] + 1;
                size_t offset = 0;
                for (size_t z = 1; z <= d.size[2]; ++z)
                    for (size_t y = 1; y <= d.size[1]; ++y)
                        field[d.index(x, y, z)] = receiveBuffer[face][offset++];
            } else {
                const size_t y = face == 2 ? 0 : d.size[1] + 1;
                for (size_t z = 1; z <= d.size[2]; ++z)
                    std::copy_n(receiveBuffer[face].data() + (z - 1) * d.size[0], d.size[0],
                                field.data() + d.index(1, y, z));
            }
        }
    }
};

// Unit grid spacing and the physical parameters are unchanged. Keep the
// original arithmetic ordering; do not enable reassociation or fast-math.
template<bool chemical>
void computeBox(const Domain& d, const Box& box, const double* __restrict__ input,
                const double* __restrict__ cold, double* __restrict__ output) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5, dt = 0.01, D = 1.0;
    for (size_t z = box.z0; z < box.z1; ++z) {
        for (size_t y = box.y0; y < box.y1; ++y) {
            const size_t base = d.index(0, y, z);
            for (size_t x = box.x0; x < box.x1; ++x) {
                const size_t i = base + x;
                const double cv = input[i];
                const double cxx = input[i + 1] + input[i - 1] - 2.0 * cv;
                const double cyy = input[i + d.row] + input[i - d.row] - 2.0 * cv;
                const double czz = input[i + d.plane] + input[i - d.plane] - 2.0 * cv;
                const double laplacian = cxx + cyy + czz;
                if constexpr (chemical)
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                else
                    output[i] = cold[i] + dt * D * laplacian;
            }
        }
    }
}

template<bool chemical>
void computeBoundary(const Domain& d, const double* input, const double* cold, double* output) {
    const Box& b = d.interior;
    const Box& w = d.whole;
    if (b.empty()) {
        computeBox<chemical>(d, w, input, cold, output);
        return;
    }
    // Six disjoint boxes cover precisely the cells omitted by the interior.
    const Box faces[] = {
        {w.x0, w.x1, w.y0, w.y1, w.z0, b.z0},
        {w.x0, w.x1, w.y0, w.y1, b.z1, w.z1},
        {w.x0, w.x1, w.y0, b.y0, b.z0, b.z1},
        {w.x0, w.x1, b.y1, w.y1, b.z0, b.z1},
        {w.x0, b.x0, b.y0, b.y1, b.z0, b.z1},
        {b.x1, w.x1, b.y0, b.y1, b.z0, b.z1}
    };
    for (const Box& face : faces)
        if (!face.empty()) computeBox<chemical>(d, face, input, cold, output);
}

void initializeConcentration(std::vector<double>& c, const Domain& d, size_t volume) {
    for (size_t z = 1; z <= d.size[2]; ++z) {
        for (size_t y = 1; y <= d.size[1]; ++y) {
            const size_t globalRow = ((d.origin[2] + z - 1) * d.global[1] + d.origin[1] + y - 1) * d.global[0];
            for (size_t x = 1; x <= d.size[0]; ++x) {
                const size_t linear_id = globalRow + d.origin[0] + x - 1;
                const double pseudo = ((linear_id + 1) * size_t(1299709) % volume) / static_cast<double>(volume);
                c[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather only on request, restoring global X-fastest order so the serial
// Kahan sum, samples and byte-wise hash remain independent of rank count.
void printDistributedResults(const std::vector<double>& c, const Domain& d, size_t volume) {
    const size_t localVolume = d.size[0] * d.size[1] * d.size[2];
    std::vector<double> packed(localVolume);
    for (size_t z = 1; z <= d.size[2]; ++z)
        for (size_t y = 1; y <= d.size[1]; ++y)
            std::copy_n(c.data() + d.index(1, y, z), d.size[0],
                        packed.data() + ((z - 1) * d.size[1] + y - 1) * d.size[0]);
    int rank, processes;
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &processes);
    constexpr int tag = 6;
    if (rank != 0) {
        for (size_t offset = 0; offset < localVolume;) {
            const int chunk = static_cast<int>(std::min(localVolume - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(packed.data() + offset, chunk, MPI_DOUBLE, 0, tag, d.comm);
            offset += chunk;
        }
        return;
    }
    std::vector<double> global(volume), received;
    for (int source = 0; source < processes; ++source) {
        int coordinates[3];
        MPI_Cart_coords(d.comm, source, 3, coordinates);
        std::array<size_t, 3> count, start;
        for (int axis = 0; axis < 3; ++axis)
            blockExtent(d.global[axis], d.divisions[axis], coordinates[axis], count[axis], start[axis]);
        const size_t elements = count[0] * count[1] * count[2];
        const double* values = packed.data();
        if (source != 0) {
            received.resize(elements);
            for (size_t offset = 0; offset < elements;) {
                const int chunk = static_cast<int>(std::min(elements - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, tag, d.comm, MPI_STATUS_IGNORE);
                offset += chunk;
            }
            values = received.data();
        }
        for (size_t z = 0; z < count[2]; ++z)
            for (size_t y = 0; y < count[1]; ++y)
                std::copy_n(values + (z * count[1] + y) * count[0], count[0],
                            global.data() + ((start[2] + z) * d.global[1] + start[1] + y) * d.global[0] + start[0]);
    }
    print_results(global, "Concentration");
}

bool validateResult(const std::vector<double>& c, const Domain& d, int rank) {
    int invalid = 0, globalInvalid = 0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= d.size[2]; ++z)
        for (size_t y = 1; y <= d.size[1]; ++y)
            for (size_t x = 1; x <= d.size[0]; ++x) {
                const double value = c[d.index(x, y, z)];
                invalid |= !std::isfinite(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
    double globalMinimum = 0.0, globalMaximum = 0.0;
    MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, 0, d.comm);
    MPI_Reduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, 0, d.comm);
    MPI_Reduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    if (rank != 0) return true;
    if (globalInvalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
    if (globalMaximum > 10.0 || globalMinimum < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

} // namespace

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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        bool validOption = true;
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)
            validOption = parseSize(argv[++i], nx);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc)
            validOption = parseSize(argv[++i], ny);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc)
            validOption = parseSize(argv[++i], nz);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            validOption = parseIterations(argv[++i], iterations);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
        if (!validOption) {
            if (worldRank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0) {
        if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    int status = 0;
    try {
        const std::array<size_t, 3> global{nx, ny, nz};
        const size_t gridSize = checkedProduct(checkedProduct(nx, ny), nz);
        auto divisions = processGrid(global, worldSize);
        const int activeSize = divisions[0] * divisions[1] * divisions[2];
        MPI_Comm active = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &active);
        if (active != MPI_COMM_NULL) {
            MPI_Comm cart = MPI_COMM_NULL;
            int periodic[3] = {0, 0, 0};
            MPI_Cart_create(active, 3, divisions.data(), periodic, 0, &cart);
            int rank;
            MPI_Comm_rank(cart, &rank);
            Domain d(cart, global, divisions);
            HaloExchange halo(d);
            std::vector<double> cold(d.storage), cnew(d.storage), mu(d.storage);
            if (rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("MPI ranks: %d; process grid: %d x %d x %d\n", activeSize, divisions[0], divisions[1], divisions[2]);
                printf("Initializing concentration field...\n");
            }
            initializeConcentration(cold, d, gridSize);
            if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                halo.begin(cold);
                if (!d.interior.empty())
                    computeBox<true>(d, d.interior, cold.data(), nullptr, mu.data());
                halo.finish(cold);
                computeBoundary<true>(d, cold.data(), nullptr, mu.data());

                halo.begin(mu);
                if (!d.interior.empty())
                    computeBox<false>(d, d.interior, mu.data(), cold.data(), cnew.data());
                halo.finish(mu);
                computeBoundary<false>(d, mu.data(), cold.data(), cnew.data());
                cold.swap(cnew);
            }
            const double elapsed = MPI_Wtime() - start;
            double duration = 0.0;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (rank == 0) {
                printf("Computation time: %.3f ms\n", duration * 1000.0);
                const double cellUpdates = static_cast<double>(gridSize) * iterations;
                printf("Performance: %.3f MCellUpdates/s\n", duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0);
            }
            if (printResults) printDistributedResults(cold, d, gridSize);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateResult(cold, d, rank);
                if (rank == 0) {
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
            MPI_Comm_free(&cart);
            MPI_Comm_free(&active);
        }
        // Also deliver the root's validation status to inactive ranks.
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
