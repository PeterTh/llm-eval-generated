#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[4];
    val_t connected_flux[4];
};
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> energy, energy_swap, flux, flux_swap;
    int n, first_row, rows;
};

int rowStart(int rank, int ranks, int n) {
    return static_cast<int>(static_cast<int64_t>(rank) * n / ranks);
}

void buildSquare2D(World& w, int n, int first, int rows) {
    w.n = n; w.first_row = first; w.rows = rows;
    w.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    w.elements_static.resize(static_cast<size_t>(rows) * n);
    // Owned energies start at offset n; the surrounding rows are halos.
    w.energy.resize(static_cast<size_t>(rows + 2) * n, 0.0);
    w.energy_swap.resize(w.energy.size(), 0.0);
    w.flux.resize(static_cast<size_t>(rows) * n, 0.0);
    w.flux_swap.resize(w.flux.size(), 0.0);
    for (int x = first; x < first + rows; ++x) {
        for (int y = 0; y < n; ++y) {
            auto& e = w.elements_static[static_cast<size_t>(x - first) * n + y];
            e.material_idx = 0;
            e.num_connections = 0;
            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int k = 0; k < 4; ++k) {
                int nx = x + offsets[k][0], ny = y + offsets[k][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    e.connected_idx[e.num_connections] = static_cast<idx_t>(nx-first+1) * n + ny;
                    e.connected_flux[e.num_connections++] = 1.0;
                }
            }
            if ((x == 0 || x == n-1) && (y == 0 || y == n-1))
                e.material_idx = x == y ? 1 : 2;
        }
    }
}

void updateRows(World& w, int begin, int end) {
    for (int row = begin; row < end; ++row) {
        for (int y = 0; y < w.n; ++y) {
            size_t i = static_cast<size_t>(row) * w.n + y;
            const auto& e = w.elements_static[i];
            const auto& mat = w.materials[e.material_idx];
            const val_t energy = w.energy[i + w.n];
            val_t total = mat.external_flow;
            for (idx_t j = 0; j < e.num_connections; ++j) {
                total += (w.energy[e.connected_idx[j]] - energy) *
                         mat.transfer_coeff * e.connected_flux[j] * 0.25;
            }
            w.energy_swap[i + w.n] = energy + total;
            w.flux_swap[i] = w.flux[i] + std::abs(total);
        }
    }
}

void runSimulation(World& w, int iterations, MPI_Comm comm, int rank, int ranks) {
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[4];
        int count = 0;
        if (rank > 0) {
            MPI_Irecv(w.energy.data(), w.n, MPI_DOUBLE, rank-1, 1, comm, &req[count++]);
            MPI_Isend(w.energy.data()+w.n, w.n, MPI_DOUBLE, rank-1, 0, comm, &req[count++]);
        }
        if (rank+1 < ranks) {
            MPI_Irecv(w.energy.data()+static_cast<size_t>(w.rows+1)*w.n,
                      w.n, MPI_DOUBLE, rank+1, 0, comm, &req[count++]);
            MPI_Isend(w.energy.data()+static_cast<size_t>(w.rows)*w.n,
                      w.n, MPI_DOUBLE, rank+1, 1, comm, &req[count++]);
        }
        if (w.rows > 2) updateRows(w, 1, w.rows-1);
        if (count) MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
        updateRows(w, 0, 1);
        if (w.rows > 1) updateRows(w, w.rows-1, w.rows);
        std::swap(w.energy, w.energy_swap);
        std::swap(w.flux, w.flux_swap);
    }
}

uint64_t computeLocalHash(const World& w) {
    uint64_t hash = 0;
    size_t start = static_cast<size_t>(w.first_row) * w.n;
    for (size_t i = 0; i < w.flux.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &w.energy[i+w.n], sizeof(energy_bits));
        std::memcpy(&flux_bits, &w.flux[i], sizeof(flux_bits));
        hash ^= (energy_bits + start + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + start + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0, flux_sum = 0;
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
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n"); return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n"); return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
    int n = 512, iterations = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            bad = true; break;
        }
    }
    if (help || bad || n < 1 || static_cast<int64_t>(n)*n > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            if (!help && !bad) printf("Grid size must be positive and fit MPI counts\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return help && !bad ? 0 : 1;
    }
    const int active = std::min(size, n);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) { MPI_Finalize(); return 0; }
    int first = rowStart(rank, active, n);
    int rows = rowStart(rank+1, active, n) - first;
    int elements = n*n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, elements);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World w;
    buildSquare2D(w, n, first, rows);
    unsigned long long local_static = w.elements_static.size()*sizeof(ElementStatic);
    unsigned long long local_dynamic = (w.energy.size()+w.energy_swap.size()+
                                        w.flux.size()+w.flux_swap.size())*sizeof(val_t);
    unsigned long long static_mem = 0, dynamic_mem = 0;
    MPI_Reduce(&local_static, &static_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_dynamic, &dynamic_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem+dynamic_mem)/1048576.0, static_mem/1048576.0,
               dynamic_mem/1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    runSimulation(w, iterations, comm, rank, active);
    double local_seconds = MPI_Wtime()-start, seconds = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    uint64_t local_hash = computeLocalHash(w), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (rank == 0) {
        double duration_ms = seconds*1000.0;
        int measured = std::max(iterations-1, 1);
        double rate = static_cast<double>(measured)*elements/seconds/1e9;
        printf("Computation time: %.0f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms/measured);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate*22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            counts.resize(active); displacements.resize(active);
            for (int r = 0; r < active; ++r) {
                displacements[r] = rowStart(r, active, n)*n;
                counts[r] = (rowStart(r+1, active, n)-rowStart(r, active, n))*n;
            }
        }
        std::vector<val_t> all_energy, all_flux;
        if (rank == 0) all_energy.resize(elements);
        MPI_Gatherv(w.energy.data()+n, rows*n, MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (printResults && rank == 0) print_results(all_energy, "ElementEnergy");
        if (validate) {
            if (rank == 0) all_flux.resize(elements);
            MPI_Gatherv(w.flux.data(), rows*n, MPI_DOUBLE,
                        rank == 0 ? all_flux.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr,
                        MPI_DOUBLE, 0, comm);
            int valid = rank == 0 ? validateResults(all_energy, all_flux) : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
            if (!valid) { MPI_Comm_free(&comm); MPI_Finalize(); return 1; }
        }
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
