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

// Types
using idx_t = uint64_t;
using val_t = double;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct Decomp {
    int rank = 0;
    int size = 1;
    int n_root = 0;

    int x_start = 0;   // global, inclusive
    int x_end = 0;     // global, exclusive
    int local_nx = 0;  // owned rows

    int up_rank = MPI_PROC_NULL;
    int down_rank = MPI_PROC_NULL;

    size_t local_elems = 0;
};

static inline size_t idx2(const int lx, const int y, const int n_root) {
    return static_cast<size_t>(lx) * static_cast<size_t>(n_root) + static_cast<size_t>(y);
}

static Decomp makeDecomp(const int n_root, MPI_Comm comm) {
    Decomp d;
    MPI_Comm_rank(comm, &d.rank);
    MPI_Comm_size(comm, &d.size);
    d.n_root = n_root;

    const int base = n_root / d.size;
    const int rem = n_root % d.size;

    const int my_nx = base + (d.rank < rem ? 1 : 0);
    const int my_start = d.rank * base + std::min(d.rank, rem);

    d.x_start = my_start;
    d.x_end = my_start + my_nx;
    d.local_nx = my_nx;

    d.up_rank = (d.x_start > 0) ? (d.rank - 1) : MPI_PROC_NULL;
    d.down_rank = (d.x_end < n_root) ? (d.rank + 1) : MPI_PROC_NULL;

    d.local_elems = static_cast<size_t>(d.local_nx) * static_cast<size_t>(n_root);
    return d;
}

static inline idx_t materialFor(const int gx, const int gy, const int n_root) {
    const int last = n_root - 1;
    if ((gx == 0 && gy == 0) || (gx == last && gy == last)) return INFLOW_MAT_ID;
    if ((gx == 0 && gy == last) || (gx == last && gy == 0)) return OUTFLOW_MAT_ID;
    return DEFAULT_MAT_ID;
}

static inline val_t computeFlux(const val_t transfer_coeff, const val_t this_energy,
                               const val_t connection_flux, const val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

struct World {
    std::vector<Material> materials;
    Decomp decomp;

    // Arrays are sized (local_nx + 2) * n_root to hold 1-row halos in X.
    // Owned rows are lx = [1, local_nx]. Halos are lx = 0 and lx = local_nx+1.
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux;
    std::vector<val_t> flux_swap;
};

// Build a 2D square grid decomposed in X across MPI ranks
static void buildSquare2D(World& world, const int n_root, MPI_Comm comm) {
    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.decomp = makeDecomp(n_root, comm);

    const size_t n = static_cast<size_t>(world.decomp.local_nx + 2) * static_cast<size_t>(n_root);
    world.energy.assign(n, 0.0);
    world.energy_swap.assign(n, 0.0);
    world.flux.assign(n, 0.0);
    world.flux_swap.assign(n, 0.0);
}

static inline void postHaloExchange(const World& world, MPI_Comm comm, MPI_Request reqs[4], int& nreq) {
    nreq = 0;
    const auto& d = world.decomp;
    if (d.size == 1 || d.local_nx == 0) return;

    // Tags chosen to avoid ambiguity:
    // tag 0: first owned row (sent to up rank, received from down rank)
    // tag 1: last owned row  (sent to down rank, received from up rank)
    const int n_root = d.n_root;
    val_t* e = const_cast<val_t*>(world.energy.data());

    if (d.up_rank != MPI_PROC_NULL) {
        MPI_Irecv(&e[idx2(0, 0, n_root)], n_root, MPI_DOUBLE, d.up_rank, 1, comm, &reqs[nreq++]);
        MPI_Isend(&e[idx2(1, 0, n_root)], n_root, MPI_DOUBLE, d.up_rank, 0, comm, &reqs[nreq++]);
    }
    if (d.down_rank != MPI_PROC_NULL) {
        MPI_Irecv(&e[idx2(d.local_nx + 1, 0, n_root)], n_root, MPI_DOUBLE, d.down_rank, 0, comm, &reqs[nreq++]);
        MPI_Isend(&e[idx2(d.local_nx, 0, n_root)], n_root, MPI_DOUBLE, d.down_rank, 1, comm, &reqs[nreq++]);
    }
}

// Run simulation for n_iters iterations (MPI decomposed with 1-row halos)
static void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const auto& d = world.decomp;
    const int n_root = d.n_root;

    MPI_Request reqs[4];
    int nreq = 0;

    for (int iter = 0; iter < n_iters; ++iter) {
        postHaloExchange(world, comm, reqs, nreq);

        val_t* e = world.energy.data();
        val_t* e2 = world.energy_swap.data();
        val_t* f = world.flux.data();
        val_t* f2 = world.flux_swap.data();

        const int lx_begin_owned = 1;
        const int lx_end_owned = d.local_nx;

        auto compute_rows = [&](int lx_begin, int lx_end) {
            if (lx_begin > lx_end) return;
            for (int lx = lx_begin; lx <= lx_end; ++lx) {
                const int gx = d.x_start + (lx - 1);
                const bool has_xp = (gx + 1) < n_root;
                const bool has_xm = (gx - 1) >= 0;

                for (int y = 0; y < n_root; ++y) {
                    const size_t p = idx2(lx, y, n_root);
                    const val_t this_e = e[p];

                    const idx_t mat_id = materialFor(gx, y, n_root);
                    const Material& mat = world.materials[mat_id];

                    val_t total_flux = mat.external_flow;

                    // Maintain original neighbor accumulation order: (x+1), (x-1), (y+1), (y-1)
                    if (has_xp) total_flux += computeFlux(mat.transfer_coeff, this_e, 1.0, e[idx2(lx + 1, y, n_root)]);
                    if (has_xm) total_flux += computeFlux(mat.transfer_coeff, this_e, 1.0, e[idx2(lx - 1, y, n_root)]);
                    if ((y + 1) < n_root) total_flux += computeFlux(mat.transfer_coeff, this_e, 1.0, e[idx2(lx, y + 1, n_root)]);
                    if ((y - 1) >= 0) total_flux += computeFlux(mat.transfer_coeff, this_e, 1.0, e[idx2(lx, y - 1, n_root)]);

                    e2[p] = this_e + total_flux;
                    f2[p] = f[p] + std::abs(total_flux);
                }
            }
        };

        // Overlap: compute interior rows while halos are in-flight, then compute boundary rows.
        if (nreq > 0 && d.local_nx > 2) {
            compute_rows(lx_begin_owned + 1, lx_end_owned - 1);
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            compute_rows(lx_begin_owned, lx_begin_owned);
            compute_rows(lx_end_owned, lx_end_owned);
        } else {
            if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            compute_rows(lx_begin_owned, lx_end_owned);
        }

        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

static bool validateResults(const World& world, MPI_Comm comm) {
    const auto& d = world.decomp;
    const int n_root = d.n_root;

    val_t energy_sum_local = 0.0;
    val_t flux_sum_local = 0.0;
    val_t energy_max_local = std::numeric_limits<val_t>::lowest();
    val_t energy_min_local = std::numeric_limits<val_t>::max();

    const val_t* e = world.energy.data();
    const val_t* f = world.flux.data();

    for (int lx = 1; lx <= d.local_nx; ++lx) {
        for (int y = 0; y < n_root; ++y) {
            const size_t p = idx2(lx, y, n_root);
            const val_t ev = e[p];
            const val_t fv = f[p];
            energy_sum_local += ev;
            flux_sum_local += fv;
            energy_max_local = std::max(ev, energy_max_local);
            energy_min_local = std::min(ev, energy_min_local);
        }
    }

    val_t energy_sum = 0.0, flux_sum = 0.0, energy_max = 0.0, energy_min = 0.0;
    MPI_Allreduce(&energy_sum_local, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&flux_sum_local, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&energy_max_local, &energy_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&energy_min_local, &energy_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
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
    }

    return true;
}

// Compute a simple hash of the global results for verification (bitwise-stable across ranks)
static uint64_t computeHashMPI(const World& world, MPI_Comm comm) {
    const auto& d = world.decomp;
    const int n_root = d.n_root;

    const val_t* e = world.energy.data();
    const val_t* f = world.flux.data();

    uint64_t hash_local = 0;
    for (int lx = 1; lx <= d.local_nx; ++lx) {
        const int gx = d.x_start + (lx - 1);
        const uint64_t row_base = static_cast<uint64_t>(gx) * static_cast<uint64_t>(n_root);
        for (int y = 0; y < n_root; ++y) {
            const uint64_t gi = row_base + static_cast<uint64_t>(y);
            const size_t p = idx2(lx, y, n_root);

            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&e[p]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&f[p]);
            hash_local ^= (*e_ptr + gi) * 0x9e3779b97f4a7c15ULL;
            hash_local ^= (*f_ptr + gi) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t hash_global = 0;
    MPI_Reduce(&hash_local, &hash_global, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    return hash_global;
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
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0 then broadcast
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(comm, 0);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(comm, 1);
            }
        }
    }

    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, comm);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, comm);
    MPI_Bcast(flags, 2, MPI_INT, 0, comm);
    validate = (flags[0] != 0);
    printResults = (flags[1] != 0);

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark (MPI)\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, comm);

    const size_t local_bytes =
        (world.energy.size() + world.energy_swap.size() + world.flux.size() + world.flux_swap.size()) * sizeof(val_t);
    size_t total_bytes = 0;
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

    if (rank == 0) {
        printf("Memory usage: %.2f MB (distributed arrays total)\n", total_bytes / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();

    const double duration_ms_local =
        static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    double duration_ms = 0.0;
    MPI_Reduce(&duration_ms_local, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    const uint64_t hash = computeHashMPI(world, comm);
    if (rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printResults) {
        const int local_count = static_cast<int>(world.decomp.local_elems);
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> energyGlobal;

        if (rank == 0) {
            counts.resize(size);
        }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm);

        if (rank == 0) {
            displs.resize(size);
            int disp = 0;
            for (int r = 0; r < size; ++r) {
                displs[r] = disp;
                disp += counts[r];
            }
            energyGlobal.resize(static_cast<size_t>(n_elems));
        }

        const val_t* sendbuf = (local_count > 0) ? &world.energy[idx2(1, 0, n_elems_root)] : nullptr;
        MPI_Gatherv(sendbuf, local_count, MPI_DOUBLE,
                    rank == 0 ? energyGlobal.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, comm);

        if (rank == 0) {
            print_results(energyGlobal, "ElementEnergy");
        }
    }

    bool valid = true;
    if (validate) {
        valid = validateResults(world, comm);
        int ok = valid ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        valid = (ok != 0);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
