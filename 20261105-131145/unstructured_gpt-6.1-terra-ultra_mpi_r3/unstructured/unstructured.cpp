#include <mpi.h>

#include <algorithm>
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
constexpr int MAX_CONNECTIONS = 8;
struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy, total_flux; };

// The rank grid has no empty tiles. Balance the largest tile's work against
// its halo size; this can leave a few ranks idle for awkward process counts.
struct Partition {
    int n, px, py, active, rank, x, y, x0, y0, nx, ny;
    Partition(int size, int rank_, int n_) : n(n_), px(1), py(1), active(1), rank(rank_),
        x(0), y(0), x0(0), y0(0), nx(0), ny(0) {
        double best_cost = std::numeric_limits<double>::max();
        for (int a = 1; a <= std::min(n, size); ++a) {
            int b = std::min(n, size / a);
            double rows = n / a + (n % a != 0);
            double cols = n / b + (n % b != 0);
            double cost = rows * cols + 2.0 * (rows + cols);
            if (cost < best_cost || (cost == best_cost && a * b > active)) {
                px = a; py = b; active = a * b; best_cost = cost;
            }
        }
        if (rank < active) {
            x = rank / py; y = rank % py;
            nx = n / px + (x < n % px); ny = n / py + (y < n % py);
            x0 = x * (n / px) + std::min(x, n % px);
            y0 = y * (n / py) + std::min(y, n % py);
        }
    }
    int neighbor(int dx, int dy) const {
        int xx = x + dx, yy = y + dy;
        return xx < 0 || xx >= px || yy < 0 || yy >= py ? MPI_PROC_NULL : xx * py + yy;
    }
};

struct World {
    Partition part;
    std::vector<Material> materials{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    std::vector<ElementStatic> elements_static;
    // Dynamic arrays include one halo cell on each side of the tile.
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
    explicit World(const Partition& p) : part(p) {
        if (p.rank >= p.active) return;
        const size_t stride = static_cast<size_t>(p.ny) + 2;
        elements_static.resize(static_cast<size_t>(p.nx) * p.ny);
        elements_dynamic.resize((static_cast<size_t>(p.nx) + 2) * stride);
        elements_dynamic_swap.resize(elements_dynamic.size());
        for (int i = 0; i < p.nx; ++i) for (int j = 0; j < p.ny; ++j) {
            int gx = p.x0 + i, gy = p.y0 + j;
            ElementStatic& e = elements_static[static_cast<size_t>(i) * p.ny + j];
            e.material_idx = 0; e.num_connections = 0;
            // Preserve the original corner assignments and neighbor order.
            if (gx == 0 && gy == 0) e.material_idx = 1;
            if (gx == 0 && gy == p.n - 1) e.material_idx = 2;
            if (gx == p.n - 1 && gy == 0) e.material_idx = 2;
            if (gx == p.n - 1 && gy == p.n - 1) e.material_idx = 1;
            const size_t at = static_cast<size_t>(i + 1) * stride + j + 1;
            auto add = [&](size_t idx) { e.connected_idx[e.num_connections] = idx; e.connected_flux[e.num_connections++] = 1.0; };
            if (gx + 1 < p.n) add(at + stride);
            if (gx > 0) add(at - stride);
            if (gy + 1 < p.n) add(at + 1);
            if (gy > 0) add(at - 1);
        }
    }
};

inline val_t computeFlux(const Material& mat, const ElementDynamic& self,
                         val_t connection_flux, const ElementDynamic& other) {
    return (other.current_energy - self.current_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& world, int n_iters) {
    const Partition& p = world.part;
    if (p.rank >= p.active) return;
    const int nx = p.nx, ny = p.ny;
    const size_t stride = static_cast<size_t>(ny) + 2;
    const int up = p.neighbor(-1, 0), down = p.neighbor(1, 0);
    const int left = p.neighbor(0, -1), right = p.neighbor(0, 1);
    std::vector<double> su(ny), sd(ny), sl(nx), sr(nx), ru(ny), rd(ny), rl(nx), rr(nx);
    for (int iter = 0; iter < n_iters; ++iter) {
        auto& current = world.elements_dynamic;
        auto& next = world.elements_dynamic_swap;
        for (int j = 0; j < ny; ++j) {
            su[j] = current[stride + j + 1].current_energy;
            sd[j] = current[static_cast<size_t>(nx) * stride + j + 1].current_energy;
        }
        for (int i = 0; i < nx; ++i) {
            sl[i] = current[static_cast<size_t>(i + 1) * stride + 1].current_energy;
            sr[i] = current[static_cast<size_t>(i + 1) * stride + ny].current_energy;
        }
        MPI_Request requests[8];
        MPI_Irecv(ru.data(), ny, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(rd.data(), ny, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Irecv(rl.data(), nx, MPI_DOUBLE, left, 2, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(rr.data(), nx, MPI_DOUBLE, right, 3, MPI_COMM_WORLD, &requests[3]);
        MPI_Isend(sd.data(), ny, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &requests[4]);
        MPI_Isend(su.data(), ny, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[5]);
        MPI_Isend(sr.data(), nx, MPI_DOUBLE, right, 2, MPI_COMM_WORLD, &requests[6]);
        MPI_Isend(sl.data(), nx, MPI_DOUBLE, left, 3, MPI_COMM_WORLD, &requests[7]);
        auto update = [&](int i, int j) {
            const ElementStatic& e = world.elements_static[static_cast<size_t>(i - 1) * ny + j - 1];
            const size_t at = static_cast<size_t>(i) * stride + j;
            const ElementDynamic& self = current[at];
            const Material& mat = world.materials[e.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t k = 0; k < e.num_connections; ++k)
                total_flux += computeFlux(mat, self, e.connected_flux[k], current[e.connected_idx[k]]);
            next[at].current_energy = self.current_energy + total_flux;
            next[at].total_flux = self.total_flux + std::abs(total_flux);
        };
        for (int i = 2; i < nx; ++i) for (int j = 2; j < ny; ++j) update(i, j);
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        if (up != MPI_PROC_NULL) for (int j = 0; j < ny; ++j) current[j + 1].current_energy = ru[j];
        if (down != MPI_PROC_NULL) for (int j = 0; j < ny; ++j) current[static_cast<size_t>(nx + 1) * stride + j + 1].current_energy = rd[j];
        if (left != MPI_PROC_NULL) for (int i = 0; i < nx; ++i) current[static_cast<size_t>(i + 1) * stride].current_energy = rl[i];
        if (right != MPI_PROC_NULL) for (int i = 0; i < nx; ++i) current[static_cast<size_t>(i + 1) * stride + ny + 1].current_energy = rr[i];
        for (int j = 1; j <= ny; ++j) update(1, j);
        if (nx > 1) for (int j = 1; j <= ny; ++j) update(nx, j);
        for (int i = 2; i < nx; ++i) {
            update(i, 1);
            if (ny > 1) update(i, ny);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

uint64_t localHash(const World& world) {
    const auto& p = world.part;
    uint64_t hash = 0;
    size_t stride = static_cast<size_t>(p.ny) + 2;
    for (int i = 0; i < p.nx; ++i) for (int j = 0; j < p.ny; ++j) {
        const auto& e = world.elements_dynamic[static_cast<size_t>(i + 1) * stride + j + 1];
        uint64_t ebits, fbits;
        std::memcpy(&ebits, &e.current_energy, sizeof(ebits));
        std::memcpy(&fbits, &e.total_flux, sizeof(fbits));
        uint64_t global = static_cast<uint64_t>(p.x0 + i) * p.n + p.y0 + j;
        hash ^= (ebits + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fbits + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

// Gather only when ordered output or validation is requested. Transfers are
// chunked so even a large tile does not exceed MPI's int count limit.
std::vector<ElementDynamic> gatherResults(const World& world) {
    const auto& p = world.part;
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));
    MPI_Datatype element_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
    MPI_Type_commit(&element_type);
    std::vector<ElementDynamic> tile(static_cast<size_t>(p.nx) * p.ny);
    size_t stride = static_cast<size_t>(p.ny) + 2;
    for (int i = 0; i < p.nx; ++i)
        std::copy_n(world.elements_dynamic.begin() + static_cast<size_t>(i + 1) * stride + 1, p.ny,
                    tile.begin() + static_cast<size_t>(i) * p.ny);
    std::vector<ElementDynamic> global;
    if (p.rank == 0) {
        global.resize(static_cast<size_t>(p.n) * p.n);
        for (int r = 0; r < p.active; ++r) {
            Partition q = p;
            q.rank = r; q.x = r / p.py; q.y = r % p.py;
            q.nx = p.n / p.px + (q.x < p.n % p.px);
            q.ny = p.n / p.py + (q.y < p.n % p.py);
            q.x0 = q.x * (p.n / p.px) + std::min(q.x, p.n % p.px);
            q.y0 = q.y * (p.n / p.py) + std::min(q.y, p.n % p.py);
            if (r != 0) {
                tile.resize(static_cast<size_t>(q.nx) * q.ny);
                size_t offset = 0;
                while (offset < tile.size()) {
                    int count = static_cast<int>(std::min(tile.size() - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(tile.data() + offset, count, element_type, r, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += count;
                }
            }
            for (int i = 0; i < q.nx; ++i)
                std::copy_n(tile.begin() + static_cast<size_t>(i) * q.ny, q.ny,
                            global.begin() + static_cast<size_t>(q.x0 + i) * p.n + q.y0);
        }
    } else if (p.rank < p.active) {
        size_t offset = 0;
        while (offset < tile.size()) {
            int count = static_cast<int>(std::min(tile.size() - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(tile.data() + offset, count, element_type, 0, 4, MPI_COMM_WORLD);
            offset += count;
        }
    }
    MPI_Type_free(&element_type);
    return global;
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy; flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) { printf("  ERROR: Energy extrema are not finite\n"); return false; }
    printf("  Validation: PASSED\n"); return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, n_iters = 10, exit_code = 0;
    bool validate = false, printResults = false, stop = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); stop = true; break; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } exit_code = 1; stop = true; break; }
    }
    if (!stop && n <= 0) { if (rank == 0) printf("Grid size must be positive\n"); exit_code = 1; stop = true; }
    if (stop) { MPI_Finalize(); return exit_code; }
    const size_t n_elems = static_cast<size_t>(n) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n", n, n, n_elems, n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    Partition part(size, rank, n);
    World world(part);
    if (rank == 0) {
        size_t static_mem = n_elems * sizeof(ElementStatic), dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n", (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0, dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    runSimulation(world, n_iters);
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = localHash(world), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        int measured = std::max(n_iters - 1, 1);
        double time_per_iter = static_cast<double>(duration_ms) / measured;
        double giga = (static_cast<double>(measured) * n_elems) / (duration_ms / 1000.0) / 1e9;
        printf("Computation time: %ld ms\nPerformance:\n", duration_ms);
        printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", time_per_iter, giga, giga * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    if (validate || printResults) {
        auto elements = gatherResults(world);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energies; energies.reserve(elements.size());
                for (const auto& e : elements) energies.push_back(e.current_energy);
                print_results(energies, "ElementEnergy");
            }
            if (validate && !validateResults(elements)) exit_code = 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
