#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
// The generated mesh has at most four connections per element.
constexpr int MAX_CONNECTIONS = 4;
struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    size_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
    unsigned char material_idx, num_connections;
};
constexpr unsigned char DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

// Quotient/remainder partitioning gives every active rank a nonempty rectangle.
int blockSize(int n, int parts, int coord) {
    return n / parts + (coord < n % parts);
}
int blockStart(int n, int parts, int coord) {
    return coord * (n / parts) + std::min(coord, n % parts);
}

struct World {
    MPI_Comm comm;
    int n, nx, ny, x0, y0;
    size_t pitch;
    std::array<Material, 3> materials{{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}}};
    std::vector<ElementStatic> elements_static;
    // Only energies need halos. Accumulated absolute flux is purely local.
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> total_flux;
    std::array<std::vector<MPI_Request>, 2> exchanges;
    MPI_Datatype column = MPI_DATATYPE_NULL;
    int current = 0;

    World(MPI_Comm c, int size) : comm(c), n(size) {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(comm, 2, dims, periods, coords);
        nx = blockSize(n, dims[0], coords[0]);
        ny = blockSize(n, dims[1], coords[1]);
        x0 = blockStart(n, dims[0], coords[0]);
        y0 = blockStart(n, dims[1], coords[1]);
        pitch = static_cast<size_t>(ny) + 2;
        const size_t count = static_cast<size_t>(nx) * ny;
        elements_static.resize(count);
        total_flux.resize(count, 0.0);
        for (auto& buffer : energy)
            buffer.resize((static_cast<size_t>(nx) + 2) * pitch, 0.0);

        // Map global neighbors to owned cells or one-cell halos. Preserve the
        // serial neighbor order (+x, -x, +y, -y), including physical boundaries.
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                auto& e = elements_static[static_cast<size_t>(x) * ny + y];
                const size_t local = (static_cast<size_t>(x) + 1) * pitch + y + 1;
                e.material_idx = DEFAULT_MAT_ID;
                e.num_connections = 0;
                auto connect = [&](size_t neighbor) {
                    e.connected_idx[e.num_connections] = neighbor;
                    e.connected_flux[e.num_connections++] = 1.0;
                };
                if (x0 + x + 1 < n) connect(local + pitch);
                if (x0 + x > 0) connect(local - pitch);
                if (y0 + y + 1 < n) connect(local + 1);
                if (y0 + y > 0) connect(local - 1);
                // Preserve assignment order even for the degenerate 1x1 grid.
                if (x0 + x == 0 && y0 + y == 0) e.material_idx = INFLOW_MAT_ID;
                if (x0 + x == 0 && y0 + y == n - 1) e.material_idx = OUTFLOW_MAT_ID;
                if (x0 + x == n - 1 && y0 + y == 0) e.material_idx = OUTFLOW_MAT_ID;
                if (x0 + x == n - 1 && y0 + y == n - 1) e.material_idx = INFLOW_MAT_ID;
            }
        }

        MPI_Type_vector(nx, 1, static_cast<int>(pitch), MPI_DOUBLE, &column);
        MPI_Type_commit(&column);
        int minusX, plusX, minusY, plusY;
        MPI_Cart_shift(comm, 0, 1, &minusX, &plusX);
        MPI_Cart_shift(comm, 1, 1, &minusY, &plusY);
        for (int b = 0; b < 2; ++b) {
            auto* data = energy[b].data();
            // Persistent operations avoid rebuilding the same exchange each step.
            auto receive = [&](size_t offset, int count, MPI_Datatype type, int peer, int tag) {
                if (peer == MPI_PROC_NULL) return;
                MPI_Request request;
                MPI_Recv_init(data + offset, count, type, peer, tag, comm, &request);
                exchanges[b].push_back(request);
            };
            auto send = [&](size_t offset, int count, MPI_Datatype type, int peer, int tag) {
                if (peer == MPI_PROC_NULL) return;
                MPI_Request request;
                MPI_Send_init(data + offset, count, type, peer, tag, comm, &request);
                exchanges[b].push_back(request);
            };
            receive(1, ny, MPI_DOUBLE, minusX, 0);
            receive((static_cast<size_t>(nx) + 1) * pitch + 1, ny, MPI_DOUBLE, plusX, 1);
            receive(pitch, 1, column, minusY, 2);
            receive(pitch + ny + 1, 1, column, plusY, 3);
            send(static_cast<size_t>(nx) * pitch + 1, ny, MPI_DOUBLE, plusX, 0);
            send(pitch + 1, ny, MPI_DOUBLE, minusX, 1);
            send(pitch + ny, 1, column, plusY, 2);
            send(pitch + 1, 1, column, minusY, 3);
        }
    }
    ~World() {
        for (auto& requests : exchanges)
            for (auto& request : requests) MPI_Request_free(&request);
        MPI_Type_free(&column);
    }
};

void runSimulation(World& world, int n_iters) {
    const int nx = world.nx, ny = world.ny;
    const size_t pitch = world.pitch;
    for (int iter = 0; iter < n_iters; ++iter) {
        auto& requests = world.exchanges[world.current];
        const int count = static_cast<int>(requests.size());
        if (count) MPI_Startall(count, requests.data());
        const val_t* read = world.energy[world.current].data();
        val_t* write = world.energy[1 - world.current].data();
        auto update = [&](int x, int beginY, int endY) {
            size_t i = static_cast<size_t>(x) * ny + beginY;
            size_t local = (static_cast<size_t>(x) + 1) * pitch + beginY + 1;
            for (int y = beginY; y < endY; ++y, ++i, ++local) {
                const auto& e = world.elements_static[i];
                const auto& mat = world.materials[e.material_idx];
                const val_t value = read[local];
                val_t flux = mat.external_flow;
                for (unsigned int j = 0; j < e.num_connections; ++j)
                    flux += (read[e.connected_idx[j]] - value) * mat.transfer_coeff *
                            e.connected_flux[j] * 0.25;
                write[local] = value + flux;
                world.total_flux[i] += std::abs(flux);
            }
        };
        // Interior work reads no receive buffers and writes the other energy buffer.
        for (int x = 1; x < nx - 1; ++x) update(x, 1, ny - 1);
        if (count) MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
        update(0, 0, ny);
        if (nx > 1) update(nx - 1, 0, ny);
        for (int x = 1; x < nx - 1; ++x) {
            update(x, 0, 1);
            if (ny > 1) update(x, ny - 1, ny);
        }
        world.current = 1 - world.current;
    }
}

uint64_t computeHash(const World& world) {
    uint64_t localHash = 0, hash = 0;
    const auto& energy = world.energy[world.current];
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            uint64_t e, f;
            const size_t i = static_cast<size_t>(x) * world.ny + y;
            const idx_t global = static_cast<idx_t>(world.x0 + x) * world.n + world.y0 + y;
            // memcpy avoids the original hash's strict-aliasing violation.
            std::memcpy(&e, &energy[(static_cast<size_t>(x) + 1) * world.pitch + y + 1], sizeof(e));
            std::memcpy(&f, &world.total_flux[i], sizeof(f));
            localHash ^= (e + global) * 0x9e3779b97f4a7c15ULL;
            localHash ^= (f + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return hash;
}

// Full global storage is needed only for the optional, order-dependent result
// printer. Receive rectangles directly into their global row-major positions.
void printGlobalResults(const World& world, int rank) {
    std::vector<double> global;
    if (rank == 0) global.resize(static_cast<size_t>(world.n) * world.n);
    MPI_Datatype owned;
    MPI_Type_vector(world.nx, world.ny, static_cast<int>(world.pitch), MPI_DOUBLE, &owned);
    MPI_Type_commit(&owned);
    const val_t* data = world.energy[world.current].data() + world.pitch + 1;
    if (rank != 0) {
        MPI_Send(data, 1, owned, 0, 4, world.comm);
    } else {
        for (int x = 0; x < world.nx; ++x)
            std::copy_n(data + static_cast<size_t>(x) * world.pitch, world.ny,
                        global.data() + static_cast<size_t>(world.x0 + x) * world.n + world.y0);
        int dims[2], periods[2], coords[2], size;
        MPI_Cart_get(world.comm, 2, dims, periods, coords);
        MPI_Comm_size(world.comm, &size);
        for (int peer = 1; peer < size; ++peer) {
            MPI_Cart_coords(world.comm, peer, 2, coords);
            int sizes[2] = {world.n, world.n};
            int subsizes[2] = {blockSize(world.n, dims[0], coords[0]),
                               blockSize(world.n, dims[1], coords[1])};
            int starts[2] = {blockStart(world.n, dims[0], coords[0]),
                             blockStart(world.n, dims[1], coords[1])};
            MPI_Datatype rectangle;
            MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &rectangle);
            MPI_Type_commit(&rectangle);
            MPI_Recv(global.data(), 1, rectangle, peer, 4, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&rectangle);
        }
        print_results(global, "ElementEnergy");
    }
    MPI_Type_free(&owned);
}

bool validateResults(const World& world, int rank) {
    val_t sums[2] = {0, 0}, globalSums[2];
    val_t low = std::numeric_limits<val_t>::max(), high = std::numeric_limits<val_t>::lowest();
    val_t globalLow, globalHigh;
    int finite = 1, globalFinite;
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const val_t e = world.energy[world.current][(static_cast<size_t>(x) + 1) * world.pitch + y + 1];
            const val_t f = world.total_flux[static_cast<size_t>(x) * world.ny + y];
            sums[0] += e;
            sums[1] += f;
            low = std::min(low, e);
            high = std::max(high, e);
            finite &= std::isfinite(e) && std::isfinite(f);
        }
    }
    MPI_Reduce(sums, globalSums, 2, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, world.comm);
    MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    MPI_Reduce(&finite, &globalFinite, 1, MPI_INT, MPI_MIN, 0, world.comm);
    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
               "  Energy range: [%.6f, %.6f]\n", globalSums[0], globalSums[1], globalLow, globalHigh);
        if (!globalFinite || !std::isfinite(globalSums[0]) || !std::isfinite(globalSums[1]) ||
            !std::isfinite(globalLow) || !std::isfinite(globalHigh)) {
            printf("  ERROR: Non-finite simulation results\n");
            valid = 0;
        } else {
            if (std::abs(globalSums[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, world.comm);
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int worldRank, int worldSize) {
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
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
    if (n <= 0 || n > std::numeric_limits<int>::max() - 2 || n_iters < 0) {
        if (worldRank == 0) fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        return 1;
    }

    // Select the most balanced rectangle using the largest possible rank count.
    // For tiny grids excess ranks remain in MPI but own no cells.
    int dims[2] = {1, 1}, active = 1;
    for (int x = 1; x <= std::min(n, worldSize); ++x) {
        int y = std::min(n, worldSize / x);
        int count = x * y;
        if (count > active || (count == active && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            active = count;
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    int result = 0;
    if (workers != MPI_COMM_NULL) {
        int periods[2] = {0, 0};
        MPI_Comm cart;
        MPI_Cart_create(workers, 2, dims, periods, 0, &cart);
        int rank;
        MPI_Comm_rank(cart, &rank);
        {
            const uint64_t n_elems = static_cast<uint64_t>(n) * n;
            if (rank == 0) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n");
                printf("============================================\n");
                printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, n_elems);
                printf("Iterations: %d\nValidation: %s\n", n_iters, validate ? "enabled" : "disabled");
                printf("MPI ranks: %d (%d x %d domains)\n\nBuilding unstructured mesh...\n", active, dims[0], dims[1]);
            }
            World world(cart, n);
            const uint64_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
            const uint64_t dynamic_mem = (world.energy[0].size() * 2 + world.total_flux.size()) * sizeof(val_t);
            uint64_t localMem[2] = {static_mem, dynamic_mem}, globalMem[2];
            MPI_Reduce(localMem, globalMem, 2, MPI_UINT64_T, MPI_SUM, 0, cart);
            if (rank == 0) {
                printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                       (globalMem[0] + globalMem[1]) / (1024.0 * 1024.0),
                       globalMem[0] / (1024.0 * 1024.0), globalMem[1] / (1024.0 * 1024.0));
                printf("Running simulation...\n");
            }
            MPI_Barrier(cart);
            const double start = MPI_Wtime();
            runSimulation(world, n_iters);
            const double elapsed = MPI_Wtime() - start;
            double duration;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            const uint64_t hash = computeHash(world);
            if (rank == 0) {
                const double duration_ms = duration * 1000.0;
                const int measured = std::max(n_iters - 1, 1);
                const double rate = static_cast<double>(measured) * n_elems / duration / 1e9;
                printf("Computation time: %.3f ms\nPerformance:\n", duration_ms);
                printf("  Time per iteration: %.4f ms\n", duration_ms / measured);
                printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22.0);
                printf("  Result hash: %016" PRIX64 "\n\n", hash);
            }
            if (printResults) printGlobalResults(world, rank);
            if (validate && !validateResults(world, rank)) result = 1;
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    // All launched ranks return the same exit status, including inactive ranks.
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 1;
    try {
        result = benchmark(argc, argv, rank, size);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
