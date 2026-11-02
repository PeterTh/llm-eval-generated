#include <mpi.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 4;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};
struct ElementStatic {
    uint32_t position;
    uint32_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx;
    uint8_t num_connections;
};
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// A compact, balanced rectangular decomposition. If there are more ranks than
// useful rectangles, the remaining ranks still participate in world collectives.
struct Partition {
    int px, py, active, rx, ry, x0, y0, nx, ny;
};

static int blockSize(int n, int parts, int coord) {
    return n / parts + (coord < n % parts);
}
static int blockStart(int n, int parts, int coord) {
    return coord * (n / parts) + std::min(coord, n % parts);
}
static Partition partitionForGrid(int n, int px, int py, int rank) {
    const int active = px * py;
    if (rank >= active) return {px, py, active, 0, 0, 0, 0, 0, 0};
    int rx = rank / py, ry = rank % py;
    return {px, py, active, rx, ry, blockStart(n, px, rx), blockStart(n, py, ry),
            blockSize(n, px, rx), blockSize(n, py, ry)};
}
static Partition partition(int n, int size, int rank) {
    int px = 1, py = 1, active = 1;
    for (int x = 1; x <= std::min(n, size); ++x) {
        int y = std::min(n, size / x);
        int count = x * y;
        if (count > active || (count == active && std::abs(x - y) < std::abs(px - py))) {
            px = x;
            py = y;
            active = count;
        }
    }
    return partitionForGrid(n, px, py, rank);
}

struct World {
    Material materials[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    Partition part;
    int stride;
    std::vector<ElementStatic> elements_static;
    // One ghost cell around the local rectangle, in both time buffers.
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
    std::vector<uint32_t> interior, boundary;
    int peers[4]; // +x, -x, +y, -y
    std::vector<double> send[4], recv[4];
};

static void buildSquare2D(World& w, int n, int rank, int size) {
    w.part = partition(n, size, rank);
    const auto& p = w.part;
    w.stride = p.ny + 2;
    std::fill(std::begin(w.peers), std::end(w.peers), MPI_PROC_NULL);
    if (rank >= p.active) return;
    if (p.rx + 1 < p.px) w.peers[0] = rank + p.py;
    if (p.rx > 0) w.peers[1] = rank - p.py;
    if (p.ry + 1 < p.py) w.peers[2] = rank + 1;
    if (p.ry > 0) w.peers[3] = rank - 1;
    for (int d = 0; d < 4; ++d) {
        if (w.peers[d] != MPI_PROC_NULL) {
            const int count = d < 2 ? p.ny : p.nx;
            w.send[d].resize(count);
            w.recv[d].resize(count);
        }
    }
    const size_t local_count = static_cast<size_t>(p.nx) * p.ny;
    const size_t padded_count = static_cast<size_t>(p.nx + 2) * w.stride;
    w.elements_static.resize(local_count);
    w.elements_dynamic.resize(padded_count);
    w.elements_dynamic_swap.resize(padded_count);
    w.interior.reserve(local_count);
    for (int x = 0; x < p.nx; ++x) {
        for (int y = 0; y < p.ny; ++y) {
            const int gx = p.x0 + x, gy = p.y0 + y;
            const size_t i = static_cast<size_t>(x) * p.ny + y;
            const size_t pos = static_cast<size_t>(x + 1) * w.stride + y + 1;
            auto& e = w.elements_static[i];
            e.position = static_cast<uint32_t>(pos);
            e.material_idx = 0;
            // Preserve the order of the original four corner assignments.
            if (gx == 0 && gy == 0) e.material_idx = 1;
            if (gx == 0 && gy == n - 1) e.material_idx = 2;
            if (gx == n - 1 && gy == 0) e.material_idx = 2;
            if (gx == n - 1 && gy == n - 1) e.material_idx = 1;
            e.num_connections = 0;
            auto add = [&](size_t neighbor) {
                e.connected_idx[e.num_connections++] = static_cast<uint32_t>(neighbor);
            };
            if (gx + 1 < n) add(pos + w.stride);
            if (gx > 0) add(pos - w.stride);
            if (gy + 1 < n) add(pos + 1);
            if (gy > 0) add(pos - 1);
            const bool remote = (x == p.nx - 1 && w.peers[0] != MPI_PROC_NULL) ||
                                (x == 0 && w.peers[1] != MPI_PROC_NULL) ||
                                (y == p.ny - 1 && w.peers[2] != MPI_PROC_NULL) ||
                                (y == 0 && w.peers[3] != MPI_PROC_NULL);
            (remote ? w.boundary : w.interior).push_back(i);
        }
    }
}

static inline void updateElement(World& w, size_t i) {
    const auto& e = w.elements_static[i];
    const size_t pos = e.position;
    const auto& state = w.elements_dynamic[pos];
    const auto& mat = w.materials[e.material_idx];
    val_t flux = mat.external_flow;
    for (idx_t j = 0; j < e.num_connections; ++j) {
        flux += (w.elements_dynamic[e.connected_idx[j]].current_energy - state.current_energy) *
                mat.transfer_coeff * 0.25;
    }
    auto& out = w.elements_dynamic_swap[pos];
    out.current_energy = state.current_energy + flux;
    out.total_flux = state.total_flux + std::abs(flux);
}

static void runSimulation(World& w, int iterations) {
    const auto& p = w.part;
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[8];
        int nrequests = 0;
        for (int d = 0; d < 4; ++d) {
            if (w.peers[d] == MPI_PROC_NULL) continue;
            MPI_Irecv(w.recv[d].data(), static_cast<int>(w.recv[d].size()), MPI_DOUBLE,
                      w.peers[d], d ^ 1, MPI_COMM_WORLD, &requests[nrequests++]);
            for (size_t k = 0; k < w.send[d].size(); ++k) {
                size_t x = d == 0 ? p.nx : d == 1 ? 1 : k + 1;
                size_t y = d == 2 ? p.ny : d == 3 ? 1 : k + 1;
                w.send[d][k] = w.elements_dynamic[x * w.stride + y].current_energy;
            }
            MPI_Isend(w.send[d].data(), static_cast<int>(w.send[d].size()), MPI_DOUBLE,
                      w.peers[d], d, MPI_COMM_WORLD, &requests[nrequests++]);
        }
        for (size_t i : w.interior) updateElement(w, i);
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d) {
            for (size_t k = 0; k < w.recv[d].size(); ++k) {
                size_t x = d == 0 ? p.nx + 1 : d == 1 ? 0 : k + 1;
                size_t y = d == 2 ? p.ny + 1 : d == 3 ? 0 : k + 1;
                w.elements_dynamic[x * w.stride + y].current_energy = w.recv[d][k];
            }
        }
        for (size_t i : w.boundary) updateElement(w, i);
        std::swap(w.elements_dynamic, w.elements_dynamic_swap);
    }
}

static uint64_t computeHash(const World& w, int n) {
    uint64_t hash = 0;
    const auto& p = w.part;
    for (int x = 0; x < p.nx; ++x) {
        for (int y = 0; y < p.ny; ++y) {
            const auto& e = w.elements_dynamic[static_cast<size_t>(x + 1) * w.stride + y + 1];
            const uint64_t i = static_cast<uint64_t>(p.x0 + x) * n + p.y0 + y;
            hash ^= (std::bit_cast<uint64_t>(e.current_energy) + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(e.total_flux) + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Reassemble in the original global order for the existing reporting routines.
static std::vector<ElementDynamic> gatherResults(const World& w, int n, int rank, int size) {
    const auto& p = w.part;
    std::vector<ElementDynamic> local;
    local.reserve(static_cast<size_t>(p.nx) * p.ny);
    for (int x = 0; x < p.nx; ++x)
        for (int y = 0; y < p.ny; ++y)
            local.push_back(w.elements_dynamic[static_cast<size_t>(x + 1) * w.stride + y + 1]);
    std::vector<int> counts, displs;
    std::vector<ElementDynamic> packed, result;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            auto rp = partitionForGrid(n, p.px, p.py, r);
            counts[r] = rp.nx * rp.ny;
            displs[r] = offset;
            offset += counts[r];
        }
        packed.resize(offset);
        result.resize(offset);
    }
    MPI_Datatype element_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
    MPI_Type_commit(&element_type);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), element_type,
                packed.data(), counts.data(), displs.data(), element_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&element_type);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            auto rp = partitionForGrid(n, p.px, p.py, r);
            for (int x = 0; x < rp.nx; ++x) {
                auto src = packed.begin() + displs[r] + static_cast<size_t>(x) * rp.ny;
                auto dst = result.begin() + static_cast<size_t>(rp.x0 + x) * n + rp.y0;
                std::copy_n(src, rp.ny, dst);
            }
        }
    }
    return result;
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
           energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive with at most INT_MAX elements.\n");
        MPI_Finalize(); return 1;
    }
    const int n_elems = n * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n",
               n, n, n_elems, iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World w;
    buildSquare2D(w, n, rank, size);
    unsigned long long local_mem[2] = {w.elements_static.size() * sizeof(ElementStatic),
                                      2 * w.elements_dynamic.size() * sizeof(ElementDynamic)};
    unsigned long long global_mem[2] = {};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("MPI ranks: %d (active: %d, process grid: %d x %d)\n", size, w.part.active, w.part.px, w.part.py);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (global_mem[0] + global_mem[1]) / 1048576.0, global_mem[0] / 1048576.0,
               global_mem[1] / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    runSimulation(w, iterations);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(w, n), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int completed_iterations = std::max(iterations, 0);
        double rate = max_elapsed > 0 ? static_cast<double>(completed_iterations) * n_elems / max_elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", max_elapsed * 1000);
        printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n",
               completed_iterations ? max_elapsed * 1000 / completed_iterations : 0.0, rate, rate * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));
    }
    int valid = 1;
    if (printResults || validate) {
        auto elements = gatherResults(w, n, rank, size);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energy;
                energy.reserve(elements.size());
                for (const auto& e : elements) energy.push_back(e.current_energy);
                print_results(energy, "ElementEnergy");
            }
            if (validate) valid = validateResults(elements);
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
