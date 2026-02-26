#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (retained for size/semantics reporting)
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element (retained for size/semantics reporting)
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct RowPartition {
    int x_start;
    int x_count;
};

static inline RowPartition partition_rows(int n_rows, int rank, int size) {
    const int base = n_rows / size;
    const int rem = n_rows % size;
    const int x_count = base + (rank < rem ? 1 : 0);
    const int x_start = rank * base + (rank < rem ? rank : rem);
    return RowPartition{x_start, x_count};
}

static inline Material material_for_cell(int x, int y, int n) {
    // Matches buildSquare2D's corner material assignment.
    if ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) {
        return Material{0.8, 0.5};
    }
    if ((x == 0 && y == n - 1) || (x == n - 1 && y == 0)) {
        return Material{0.8, -0.5};
    }
    return Material{0.8, 0.0};
}

// Compute energy flux between two elements (preserve original operation ordering)
static inline val_t computeFlux(const Material& mat, val_t this_energy, val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

static inline void update_rows(const int n, const RowPartition& part, const int local_x_begin, const int local_x_end,
                               const val_t* halo_top, const val_t* halo_bottom,
                               const std::vector<val_t>& energy, const std::vector<val_t>& flux_total,
                               std::vector<val_t>& energy_swap, std::vector<val_t>& flux_total_swap) {
    for (int lx = local_x_begin; lx < local_x_end; ++lx) {
        const int gx = part.x_start + lx;
        const size_t row_off = static_cast<size_t>(lx) * static_cast<size_t>(n);
        for (int y = 0; y < n; ++y) {
            const size_t li = row_off + static_cast<size_t>(y);
            const val_t this_energy = energy[li];
            const Material mat = material_for_cell(gx, y, n);

            val_t total_flux = mat.external_flow;

            // Preserve neighbor order from buildSquare2D offsets: down, up, right, left.
            if (gx + 1 < n) {
                const val_t other = (lx + 1 < part.x_count) ? energy[li + static_cast<size_t>(n)] : halo_bottom[y];
                total_flux += computeFlux(mat, this_energy, 1.0, other);
            }
            if (gx - 1 >= 0) {
                const val_t other = (lx - 1 >= 0) ? energy[li - static_cast<size_t>(n)] : halo_top[y];
                total_flux += computeFlux(mat, this_energy, 1.0, other);
            }
            if (y + 1 < n) {
                total_flux += computeFlux(mat, this_energy, 1.0, energy[li + 1]);
            }
            if (y - 1 >= 0) {
                total_flux += computeFlux(mat, this_energy, 1.0, energy[li - 1]);
            }

            energy_swap[li] = this_energy + total_flux;
            flux_total_swap[li] = flux_total[li] + std::abs(total_flux);
        }
    }
}

static void runSimulationMPI(const int n, const int n_iters, const RowPartition& part, const int rank,
                            std::vector<val_t>& energy, std::vector<val_t>& flux_total,
                            std::vector<val_t>& energy_swap, std::vector<val_t>& flux_total_swap) {
    const int upRank = (part.x_start > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int downRank = (part.x_start + part.x_count < n) ? (rank + 1) : MPI_PROC_NULL;

    std::vector<val_t> halo_top(static_cast<size_t>(n), 0.0);
    std::vector<val_t> halo_bottom(static_cast<size_t>(n), 0.0);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request reqs[4];
        int req_count = 0;

        // Exchange boundary rows (overlap comm/compute).
        if (upRank != MPI_PROC_NULL && part.x_count > 0) {
            MPI_Irecv(halo_top.data(), n, MPI_DOUBLE, upRank, 2, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(energy.data(), n, MPI_DOUBLE, upRank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (downRank != MPI_PROC_NULL && part.x_count > 0) {
            MPI_Irecv(halo_bottom.data(), n, MPI_DOUBLE, downRank, 1, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(energy.data() + static_cast<size_t>(part.x_count - 1) * static_cast<size_t>(n),
                      n, MPI_DOUBLE, downRank, 2, MPI_COMM_WORLD, &reqs[req_count++]);
        }

        // Compute interior rows that do not depend on halo values.
        if (part.x_count > 2) {
            update_rows(n, part, 1, part.x_count - 1, halo_top.data(), halo_bottom.data(), energy, flux_total,
                        energy_swap, flux_total_swap);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
        }

        // Compute boundary rows (need halo values when neighbors exist).
        if (part.x_count == 1) {
            update_rows(n, part, 0, 1, halo_top.data(), halo_bottom.data(), energy, flux_total, energy_swap,
                        flux_total_swap);
        } else if (part.x_count > 1) {
            update_rows(n, part, 0, 1, halo_top.data(), halo_bottom.data(), energy, flux_total, energy_swap,
                        flux_total_swap);
            update_rows(n, part, part.x_count - 1, part.x_count, halo_top.data(), halo_bottom.data(), energy,
                        flux_total, energy_swap, flux_total_swap);
        }

        std::swap(energy, energy_swap);
        std::swap(flux_total, flux_total_swap);
    }
}

static uint64_t computeHashDistributed(const int n, const RowPartition& part,
                                      const std::vector<val_t>& energy, const std::vector<val_t>& flux_total) {
    uint64_t hash = 0;
    for (int lx = 0; lx < part.x_count; ++lx) {
        const int gx = part.x_start + lx;
        const size_t row_off = static_cast<size_t>(lx) * static_cast<size_t>(n);
        const uint64_t base_i = static_cast<uint64_t>(gx) * static_cast<uint64_t>(n);
        for (int y = 0; y < n; ++y) {
            const size_t li = row_off + static_cast<size_t>(y);
            const uint64_t gi = base_i + static_cast<uint64_t>(y);

            uint64_t e_bits = 0;
            uint64_t f_bits = 0;
            std::memcpy(&e_bits, &energy[li], sizeof(uint64_t));
            std::memcpy(&f_bits, &flux_total[li], sizeof(uint64_t));

            hash ^= (e_bits + gi) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f_bits + gi) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

static bool validateResultsMPI(const int n, const RowPartition& part,
                              const std::vector<val_t>& energy, const std::vector<val_t>& flux_total,
                              const int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (int lx = 0; lx < part.x_count; ++lx) {
        const size_t row_off = static_cast<size_t>(lx) * static_cast<size_t>(n);
        for (int y = 0; y < n; ++y) {
            const size_t li = row_off + static_cast<size_t>(y);
            const val_t e = energy[li];
            local_energy_sum += e;
            local_flux_sum += flux_total[li];
            local_energy_max = std::max(local_energy_max, e);
            local_energy_min = std::min(local_energy_min, e);
        }
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;

    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int ok = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            ok = 0;
        }

        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            ok = 0;
        }

        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            ok = 0;
        }

        if (ok) {
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

static void printUsage(const char* progName) {
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;
    int do_help = 0;
    int parse_ok = 1;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                do_help = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_ok = 0;
                break;
            }
        }

        if (do_help) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&do_help, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parse_ok || do_help) {
        MPI_Finalize();
        return parse_ok ? 0 : 1;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");

        // Preserve original memory usage reporting (global problem size).
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    const RowPartition part = partition_rows(n_elems_root, rank, size);
    const size_t local_count = static_cast<size_t>(part.x_count) * static_cast<size_t>(n_elems_root);

    std::vector<val_t> energy(local_count, 0.0);
    std::vector<val_t> energy_swap(local_count, 0.0);
    std::vector<val_t> flux_total(local_count, 0.0);
    std::vector<val_t> flux_total_swap(local_count, 0.0);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    runSimulationMPI(n_elems_root, n_iters, part, rank, energy, flux_total, energy_swap, flux_total_swap);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double local_s = t1 - t0;
    double max_s = 0.0;
    MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    uint64_t local_hash = computeHashDistributed(n_elems_root, part, energy, flux_total);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_s * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = (max_s * 1000.0) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / max_s / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printResults) {
        // Gather global energy array on rank 0 for external validation printing.
        std::vector<int> recvcounts;
        std::vector<int> displs;
        std::vector<val_t> all_energy;

        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            int disp = 0;
            for (int r = 0; r < size; ++r) {
                const RowPartition p = partition_rows(n_elems_root, r, size);
                const int cnt = p.x_count * n_elems_root;
                recvcounts[static_cast<size_t>(r)] = cnt;
                displs[static_cast<size_t>(r)] = disp;
                disp += cnt;
            }
            all_energy.resize(static_cast<size_t>(n_elems_root) * static_cast<size_t>(n_elems_root));
        }

        MPI_Gatherv(energy.empty() ? nullptr : energy.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    rank == 0 ? all_energy.data() : nullptr,
                    rank == 0 ? recvcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(all_energy, "ElementEnergy");
        }
    }

    if (validate) {
        const bool ok = validateResultsMPI(n_elems_root, part, energy, flux_total, rank);
        if (!ok) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
