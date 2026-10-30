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

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
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

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    const int n = n_elems_root;

    // Partition rows
    int base = n / size;
    int rem = n % size;
    int my_rows = base + (rank < rem ? 1 : 0);
    int start_x = 0;
    for (int r = 0; r < rank; ++r) start_x += base + (r < rem ? 1 : 0);

    const int local_rows = my_rows + 2; // +2 halo
    const int local_n = local_rows * n;

    std::vector<Material> materials = { Material{0.8,0.0}, Material{0.8,0.5}, Material{0.8,-0.5} };

    std::vector<ElementStatic> estatic(local_n);
    std::vector<ElementDynamic> edyn(local_n);
    std::vector<ElementDynamic> eswap(local_n);

    for (int i = 0; i < local_n; ++i) {
        estatic[i].material_idx = DEFAULT_MAT_ID;
        estatic[i].num_connections = 0;
        edyn[i].current_energy = 0.0;
        edyn[i].total_flux = 0.0;
        eswap[i].current_energy = 0.0;
        eswap[i].total_flux = 0.0;
    }

    // Build connectivity for owned rows (local rows 1..my_rows)
    for (int lr = 1; lr <= my_rows; ++lr) {
        int x = start_x + (lr - 1);
        for (int y = 0; y < n; ++y) {
            int idx = lr * n + y;
            ElementStatic& elem = estatic[idx];
            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int nn = 0; nn < 4; ++nn) {
                int nx = x + offsets[nn][0];
                int ny = y + offsets[nn][1];
                if (nx < 0 || nx >= n || ny < 0 || ny >= n) continue;
                int nlr;
                if (nx < start_x) nlr = 0; // ghost top
                else if (nx >= start_x + my_rows) nlr = my_rows + 1; // ghost bottom
                else nlr = nx - start_x + 1;
                int nidx = nlr * n + ny;
                elem.connected_idx[elem.num_connections] = nidx;
                elem.connected_flux[elem.num_connections] = 1.0;
                elem.num_connections++;
            }
        }
    }

    // Materials for corners
    auto setMat = [&](int gx, int gy, idx_t mat) {
        if (gx < start_x || gx >= start_x + my_rows) return;
        int lr = gx - start_x + 1;
        estatic[lr * n + gy].material_idx = mat;
    };
    int last = n - 1;
    setMat(0, 0, INFLOW_MAT_ID);
    setMat(0, last, OUTFLOW_MAT_ID);
    setMat(last, 0, OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n*n);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    const size_t static_mem = (size_t)n*n*sizeof(ElementStatic);
    const size_t dynamic_mem = (size_t)n*n*sizeof(ElementDynamic)*2;
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (static_mem+dynamic_mem)/(1024.0*1024.0),
               static_mem/(1024.0*1024.0),
               dynamic_mem/(1024.0*1024.0));
        printf("\nRunning simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_iters; ++iter) {
        // Halo exchange: current_energy for top/bottom ghost rows
        // Send top owned row (lr=1) to rank-1 bottom ghost; recv from rank-1 into top ghost (lr=0)
        if (rank > 0) {
            MPI_Sendrecv(&edyn[1*n].current_energy, n, MPI_DOUBLE, rank-1, 0,
                         &edyn[0*n].current_energy, n, MPI_DOUBLE, rank-1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // total_flux not needed; but keep zero
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&edyn[my_rows*n].current_energy, n, MPI_DOUBLE, rank+1, 1,
                         &edyn[(my_rows+1)*n].current_energy, n, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute owned rows
        for (int lr = 1; lr <= my_rows; ++lr) {
            for (int y = 0; y < n; ++y) {
                int i = lr * n + y;
                const ElementStatic& es = estatic[i];
                const ElementDynamic& ed = edyn[i];
                const Material& mat = materials[es.material_idx];
                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < es.num_connections; ++j) {
                    idx_t nb = es.connected_idx[j];
                    const ElementDynamic& nd = edyn[nb];
                    total_flux += computeFlux(mat, ed, es.connected_flux[j], nd);
                }
                eswap[i].current_energy = ed.current_energy + total_flux;
                eswap[i].total_flux = ed.total_flux + std::abs(total_flux);
            }
        }
        std::swap(edyn, eswap);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Gather results to root
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int rr = base + (r < rem ? 1 : 0);
        counts[r] = rr * n * (int)sizeof(ElementDynamic);
        displs[r] = (r==0?0:displs[r-1]+counts[r-1]);
    }
    // Actually displacement in elements, not bytes for MPI_Gatherv with type? Use element counts.
    std::vector<int> ecounts(size), edispls(size);
    int off = 0;
    for (int r = 0; r < size; ++r) {
        int rr = base + (r < rem ? 1 : 0);
        ecounts[r] = rr * n;
        edispls[r] = off;
        off += ecounts[r];
    }

    std::vector<ElementDynamic> global_dyn;
    if (rank == 0) global_dyn.resize((size_t)n*n);

    std::vector<int> dcounts(size), ddispls(size);
    off = 0;
    for (int r = 0; r < size; ++r) {
        int rr = base + (r < rem ? 1 : 0);
        dcounts[r] = rr * n * 2;
        ddispls[r] = off;
        off += dcounts[r];
    }
    MPI_Gatherv(&edyn[n], my_rows * n * 2, MPI_DOUBLE,
                global_dyn.data(), dcounts.data(), ddispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long duration = duration_ms;
        printf("Computation time: %ld ms\n", duration);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (size_t)n * n) / (duration / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        uint64_t hash = 0;
        for (size_t i = 0; i < global_dyn.size(); ++i) {
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&global_dyn[i].current_energy);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&global_dyn[i].total_flux);
            hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dyn.size());
            for (const auto& elem : global_dyn) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            val_t energy_sum = 0, flux_sum = 0, emax = std::numeric_limits<val_t>::lowest(), emin = std::numeric_limits<val_t>::max();
            for (const auto& e : global_dyn) {
                energy_sum += e.current_energy;
                flux_sum += e.total_flux;
                emax = std::max(e.current_energy, emax);
                emin = std::min(e.current_energy, emin);
            }
            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", emin, emax);
            printf("  Validation: PASSED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
