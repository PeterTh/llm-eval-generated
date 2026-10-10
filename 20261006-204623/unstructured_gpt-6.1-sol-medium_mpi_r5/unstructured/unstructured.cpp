#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Connectivity uses local indices, including the one-element halo. The
// connection order is identical to the original unstructured mesh.
struct ElementStatic {
    unsigned char material_idx;
    unsigned char num_connections;
    size_t connected_idx[4];
};

struct World {
    std::vector<Material> materials{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> energy[2];
    std::vector<val_t> flux;
    int nx, ny, x0, y0, stride;
    int current = 0;
    MPI_Comm comm;
    MPI_Datatype column;
    std::vector<MPI_Request> requests[2];
};

// Balanced contiguous intervals also support grid sizes not divisible by the
// process grid dimensions.
int blockSize(int n, int parts, int coord) {
    return n / parts + (coord < n % parts);
}
int blockStart(int n, int parts, int coord) {
    return coord * (n / parts) + std::min(coord, n % parts);
}

void buildSquare2D(World& w, int n, MPI_Comm comm) {
    w.comm = comm;
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    w.nx = blockSize(n, dims[0], coords[0]);
    w.ny = blockSize(n, dims[1], coords[1]);
    w.x0 = blockStart(n, dims[0], coords[0]);
    w.y0 = blockStart(n, dims[1], coords[1]);
    w.stride = w.ny + 2;
    const size_t padded = (size_t(w.nx) + 2) * w.stride;
    w.energy[0].resize(padded, 0.0);
    w.energy[1].resize(padded, 0.0);
    w.flux.resize(padded, 0.0);
    w.elements_static.resize(size_t(w.nx) * w.ny);
    for (int x = 1; x <= w.nx; ++x) {
        for (int y = 1; y <= w.ny; ++y) {
            auto& elem = w.elements_static[size_t(x - 1) * w.ny + y - 1];
            const size_t i = size_t(x) * w.stride + y;
            const int gx = w.x0 + x - 1, gy = w.y0 + y - 1;
            elem.material_idx = 0;
            if (gx == 0 && gy == 0) elem.material_idx = 1;
            if (gx == 0 && gy == n - 1) elem.material_idx = 2;
            if (gx == n - 1 && gy == 0) elem.material_idx = 2;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = 1;
            elem.num_connections = 0;
            if (gx + 1 < n) elem.connected_idx[elem.num_connections++] = i + w.stride;
            if (gx > 0) elem.connected_idx[elem.num_connections++] = i - w.stride;
            if (gy + 1 < n) elem.connected_idx[elem.num_connections++] = i + 1;
            if (gy > 0) elem.connected_idx[elem.num_connections++] = i - 1;
        }
    }

    int xm, xp, ym, yp;
    MPI_Cart_shift(comm, 0, 1, &xm, &xp);
    MPI_Cart_shift(comm, 1, 1, &ym, &yp);
    MPI_Type_vector(w.nx, 1, w.stride, MPI_DOUBLE, &w.column);
    MPI_Type_commit(&w.column);
    // Persistent requests avoid recreating the communication plan each step.
    // Only energies cross rank boundaries; cumulative flux stays local.
    for (int b = 0; b < 2; ++b) {
        auto* e = w.energy[b].data();
        auto& requests = w.requests[b];
        auto receive = [&](int peer, size_t offset, int count, MPI_Datatype type, int tag) {
            if (peer == MPI_PROC_NULL) return;
            MPI_Request r;
            MPI_Recv_init(e + offset, count, type, peer, tag, comm, &r);
            requests.push_back(r);
        };
        auto send = [&](int peer, size_t offset, int count, MPI_Datatype type, int tag) {
            if (peer == MPI_PROC_NULL) return;
            MPI_Request r;
            MPI_Send_init(e + offset, count, type, peer, tag, comm, &r);
            requests.push_back(r);
        };
        receive(xm, 1, w.ny, MPI_DOUBLE, 0);
        receive(xp, size_t(w.nx + 1) * w.stride + 1, w.ny, MPI_DOUBLE, 1);
        receive(ym, w.stride, 1, w.column, 2);
        receive(yp, w.stride + w.ny + 1, 1, w.column, 3);
        send(xp, size_t(w.nx) * w.stride + 1, w.ny, MPI_DOUBLE, 0);
        send(xm, w.stride + 1, w.ny, MPI_DOUBLE, 1);
        send(yp, w.stride + w.ny, 1, w.column, 2);
        send(ym, w.stride + 1, 1, w.column, 3);
    }
}

void runSimulation(World& w, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        auto& requests = w.requests[w.current];
        if (!requests.empty()) MPI_Startall(int(requests.size()), requests.data());
        const auto* energy = w.energy[w.current].data();
        auto* next = w.energy[1 - w.current].data();
        auto update = [&](int x, int first, int last) {
            size_t i = size_t(x) * w.stride + first;
            size_t local = size_t(x - 1) * w.ny + first - 1;
            for (int y = first; y <= last; ++y, ++i, ++local) {
                const auto& elem = w.elements_static[local];
                const auto& mat = w.materials[elem.material_idx];
                val_t total = mat.external_flow;
                for (unsigned j = 0; j < elem.num_connections; ++j)
                    total += (energy[elem.connected_idx[j]] - energy[i]) * mat.transfer_coeff * 0.25;
                next[i] = energy[i] + total;
                w.flux[i] += std::abs(total);
            }
        };
        // No remote data is needed for the interior. Receives touch only halo
        // entries and sends only read the old buffer during these updates.
        for (int x = 2; x < w.nx; ++x) update(x, 2, w.ny - 1);
        if (!requests.empty()) MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        update(1, 1, w.ny);
        if (w.nx > 1) update(w.nx, 1, w.ny);
        for (int x = 2; x < w.nx; ++x) {
            update(x, 1, 1);
            if (w.ny > 1) update(x, w.ny, w.ny);
        }
        w.current = 1 - w.current;
    }
}

uint64_t computeHash(const World& w, int n) {
    uint64_t local_hash = 0, global_hash = 0;
    for (int x = 1; x <= w.nx; ++x) {
        for (int y = 1; y <= w.ny; ++y) {
            const size_t i = size_t(x) * w.stride + y;
            const uint64_t global = uint64_t(w.x0 + x - 1) * n + w.y0 + y - 1;
            uint64_t energy_bits, flux_bits;
            std::memcpy(&energy_bits, &w.energy[w.current][i], sizeof(energy_bits));
            std::memcpy(&flux_bits, &w.flux[i], sizeof(flux_bits));
            local_hash ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, w.comm);
    return global_hash;
}

// Assemble output in original global row order without replicating the mesh.
// Derived datatypes avoid packing buffers and MPI's total-count int limit.
std::vector<val_t> gatherField(const World& w, const std::vector<val_t>& field, int n) {
    int rank, size, dims[2], periods[2], coords[2];
    MPI_Comm_rank(w.comm, &rank);
    MPI_Comm_size(w.comm, &size);
    MPI_Cart_get(w.comm, 2, dims, periods, coords);
    std::vector<val_t> result;
    if (rank == 0) {
        result.resize(size_t(n) * n);
        for (int x = 0; x < w.nx; ++x)
            std::copy_n(field.data() + size_t(x + 1) * w.stride + 1, w.ny,
                        result.data() + size_t(w.x0 + x) * n + w.y0);
        for (int peer = 1; peer < size; ++peer) {
            MPI_Cart_coords(w.comm, peer, 2, coords);
            const int nx = blockSize(n, dims[0], coords[0]);
            const int ny = blockSize(n, dims[1], coords[1]);
            MPI_Datatype block;
            MPI_Type_vector(nx, ny, n, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            const size_t start = size_t(blockStart(n, dims[0], coords[0])) * n
                               + blockStart(n, dims[1], coords[1]);
            MPI_Recv(result.data() + start, 1, block, peer, 4, w.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    } else {
        MPI_Datatype block;
        MPI_Type_vector(w.nx, w.ny, w.stride, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(field.data() + w.stride + 1, 1, block, 0, 4, w.comm);
        MPI_Type_free(&block);
    }
    return result;
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }

    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    if (n <= 0 || n > std::numeric_limits<int>::max() - 2 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    // Choose a near-square process grid with no empty partitions. Excess ranks
    // participate in MPI startup/shutdown but own no mesh or simulation state.
    int active = int(std::min<uint64_t>(size, uint64_t(n) * n));
    int dims[2];
    for (;;) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(active, 2, dims);
        if (dims[0] <= n && dims[1] <= n) break;
        --active;
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (rank >= active) {
        MPI_Finalize();
        return 0;
    }
    int periods[2] = {0, 0};
    MPI_Comm comm;
    MPI_Cart_create(workers, 2, dims, periods, 0, &comm);
    MPI_Comm_free(&workers);
    const uint64_t n_elems = uint64_t(n) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d active of %d (%d x %d)\n\n", active, size, dims[0], dims[1]);
        printf("Building unstructured mesh...\n");
    }
    World w;
    buildSquare2D(w, n, comm);
    uint64_t local_mem[2] = {w.elements_static.size() * sizeof(ElementStatic),
                            (w.energy[0].size() * 2 + w.flux.size()) * sizeof(val_t)};
    uint64_t total_mem[2];
    MPI_Reduce(local_mem, total_mem, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (total_mem[0] + total_mem[1]) / (1024.0 * 1024.0),
               total_mem[0] / (1024.0 * 1024.0), total_mem[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(w, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    const uint64_t hash = computeHash(w, n);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        const double measured_iters = std::max(n_iters - 1, 1);
        const double giga_elems = measured_iters * double(n_elems) / duration / 1e9;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems);
        printf("  Performance: %.4f GFLOPS\n", giga_elems * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }
    int valid = 1;
    if (printResults || validate) {
        auto energy = gatherField(w, w.energy[w.current], n);
        std::vector<val_t> flux;
        if (validate) flux = gatherField(w, w.flux, n);
        if (rank == 0) {
            if (printResults) print_results(energy, "ElementEnergy");
            if (validate) valid = validateResults(energy, flux);
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    for (auto& requests : w.requests)
        for (auto& request : requests) MPI_Request_free(&request);
    MPI_Type_free(&w.column);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
