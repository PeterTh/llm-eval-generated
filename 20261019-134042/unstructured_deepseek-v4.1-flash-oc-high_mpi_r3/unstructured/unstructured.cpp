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

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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

// Select a near-square process grid whose extents do not exceed the mesh size.
// Returns false when no such factorization exists (small meshes / too many ranks).
bool chooseProcessGrid(const int n, const int size, int dims[2]) {
    int best_a = 0;
    int best_b = 0;
    int best_diff = std::numeric_limits<int>::max();

    for (int a = 1; a <= size; ++a) {
        if (size % a != 0) {
            continue;
        }
        const int b = size / a;
        if (a <= n && b <= n) {
            const int diff = std::abs(a - b);
            if (diff < best_diff) {
                best_diff = diff;
                best_a = a;
                best_b = b;
            }
        }
    }

    if (best_a == 0) {
        return false;
    }

    dims[0] = best_a;
    dims[1] = best_b;
    return true;
}

static inline void blockRange(const int n, const int parts, const int coord,
                              int& begin, int& end) {
    begin = static_cast<int>((static_cast<long long>(n) * coord) / parts);
    end = static_cast<int>((static_cast<long long>(n) * (coord + 1)) / parts);
}

// Convert a local block (owned rows of the global row-major grid) into the
// full global element array on the root rank.
std::vector<ElementDynamic> runSimulation2D(const int n, const int n_iters,
                                            MPI_Comm comm, const int rank,
                                            const int size, const int dims[2],
                                            MPI_Datatype elem_dyn_type) {
    int coords[2];
    MPI_Cart_coords(comm, rank, 2, coords);

    int x0, x1, y0, y1;
    blockRange(n, dims[0], coords[0], x0, x1);
    blockRange(n, dims[1], coords[1], y0, y1);

    const int lx = x1 - x0;
    const int ly = y1 - y0;
    const int stride = ly + 2;
    const size_t padded_size = static_cast<size_t>(lx + 2) * static_cast<size_t>(stride);
    const size_t owned_size = static_cast<size_t>(lx) * static_cast<size_t>(ly);

    // The unknown corners of the padded block are never accessed: only the
    // four face neighbours matter.  Ghost cells at physical boundaries are
    // mirrored, which turns the boundary check into a harmless zero flux.
    std::vector<val_t> energy(padded_size, 0.0);
    std::vector<val_t> energy_swap(padded_size, 0.0);
    std::vector<val_t> flux(padded_size, 0.0);
    std::vector<val_t> transfer(padded_size, 0.8);
    std::vector<val_t> external(padded_size, 0.0);

    const Material materials[3] = {
        Material{0.8, 0.0},
        Material{0.8, 0.5},
        Material{0.8, -0.5}
    };

    for (int i = 1; i <= lx; ++i) {
        const int gx = x0 + i - 1;
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(stride);
        for (int j = 1; j <= ly; ++j) {
            const int gy = y0 + j - 1;
            idx_t mat = DEFAULT_MAT_ID;
            if (gx == 0 && gy == 0) mat = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1) mat = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0) mat = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) mat = INFLOW_MAT_ID;

            const size_t p = base + static_cast<size_t>(j);
            transfer[p] = materials[mat].transfer_coeff;
            external[p] = materials[mat].external_flow;
        }
    }

    int down, up, left, right;
    MPI_Cart_shift(comm, 0, 1, &down, &up);
    MPI_Cart_shift(comm, 1, 1, &left, &right);

    MPI_Datatype column_type;
    MPI_Type_vector(lx, 1, stride, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    const auto exchange = [&]() {
        MPI_Request requests[8];
        int nreq = 0;

        val_t* e = energy.data();

        if (down == MPI_PROC_NULL) {
            for (int j = 1; j <= ly; ++j) {
                e[0 * stride + j] = e[1 * stride + j];
            }
        } else {
            MPI_Irecv(&e[0 * stride + 1], ly, MPI_DOUBLE, down, 101, comm, &requests[nreq++]);
            MPI_Isend(&e[1 * stride + 1], ly, MPI_DOUBLE, down, 100, comm, &requests[nreq++]);
        }

        if (up == MPI_PROC_NULL) {
            for (int j = 1; j <= ly; ++j) {
                e[(lx + 1) * stride + j] = e[lx * stride + j];
            }
        } else {
            MPI_Irecv(&e[(lx + 1) * stride + 1], ly, MPI_DOUBLE, up, 100, comm, &requests[nreq++]);
            MPI_Isend(&e[lx * stride + 1], ly, MPI_DOUBLE, up, 101, comm, &requests[nreq++]);
        }

        if (left == MPI_PROC_NULL) {
            for (int i = 1; i <= lx; ++i) {
                e[i * stride + 0] = e[i * stride + 1];
            }
        } else {
            MPI_Irecv(&e[1 * stride + 0], 1, column_type, left, 201, comm, &requests[nreq++]);
            MPI_Isend(&e[1 * stride + 1], 1, column_type, left, 200, comm, &requests[nreq++]);
        }

        if (right == MPI_PROC_NULL) {
            for (int i = 1; i <= lx; ++i) {
                e[i * stride + (ly + 1)] = e[i * stride + ly];
            }
        } else {
            MPI_Irecv(&e[1 * stride + (ly + 1)], 1, column_type, right, 200, comm, &requests[nreq++]);
            MPI_Isend(&e[1 * stride + ly], 1, column_type, right, 201, comm, &requests[nreq++]);
        }

        if (nreq > 0) {
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        }
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        exchange();

        const val_t* __restrict__ e = energy.data();
        val_t* __restrict__ en = energy_swap.data();
        val_t* __restrict__ f = flux.data();
        const val_t* __restrict__ tc = transfer.data();
        const val_t* __restrict__ ext = external.data();

        for (int i = 1; i <= lx; ++i) {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(stride);
            for (int j = 1; j <= ly; ++j) {
                const size_t p = base + static_cast<size_t>(j);
                const val_t cur = e[p];
                const val_t coeff = tc[p];

                // Same connection order as the original mesh: +x, -x, +y, -y.
                val_t total_flux = ext[p];
                total_flux += (e[p + stride] - cur) * coeff * 0.25;
                total_flux += (e[p - stride] - cur) * coeff * 0.25;
                total_flux += (e[p + 1] - cur) * coeff * 0.25;
                total_flux += (e[p - 1] - cur) * coeff * 0.25;

                en[p] = cur + total_flux;
                f[p] += std::abs(total_flux);
            }
        }

        energy.swap(energy_swap);
    }

    MPI_Type_free(&column_type);

    // Pack the owned interior in local row-major order.
    std::vector<ElementDynamic> sendbuf(owned_size);
    const val_t* e = energy.data();
    const val_t* f = flux.data();
    for (int i = 1; i <= lx; ++i) {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(stride);
        const size_t out_base = static_cast<size_t>(i - 1) * static_cast<size_t>(ly);
        for (int j = 1; j <= ly; ++j) {
            sendbuf[out_base + static_cast<size_t>(j - 1)] =
                ElementDynamic{e[base + j], f[base + j]};
        }
    }

    int local_count = static_cast<int>(owned_size);
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
    }

    MPI_Gather(&local_count, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm);

    int total_count = 0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            displs[r] = total_count;
            total_count += counts[r];
        }
    }

    std::vector<ElementDynamic> recvbuf;
    if (rank == 0) {
        recvbuf.resize(static_cast<size_t>(total_count));
    }

    MPI_Gatherv(sendbuf.data(), local_count, elem_dyn_type,
                rank == 0 ? recvbuf.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                elem_dyn_type, 0, comm);

    std::vector<ElementDynamic> global;
    if (rank == 0) {
        global.resize(static_cast<size_t>(n) * static_cast<size_t>(n));
        for (int r = 0; r < size; ++r) {
            int rc[2];
            MPI_Cart_coords(comm, r, 2, rc);
            int rx0, rx1, ry0, ry1;
            blockRange(n, dims[0], rc[0], rx0, rx1);
            blockRange(n, dims[1], rc[1], ry0, ry1);
            const int rlx = rx1 - rx0;
            const int rly = ry1 - ry0;
            const ElementDynamic* block = recvbuf.data() + displs[r];
            for (int i = 0; i < rlx; ++i) {
                const size_t gbase =
                    static_cast<size_t>(rx0 + i) * static_cast<size_t>(n) + ry0;
                const size_t bbase = static_cast<size_t>(i) * static_cast<size_t>(rly);
                for (int j = 0; j < rly; ++j) {
                    global[gbase + static_cast<size_t>(j)] = block[bbase + static_cast<size_t>(j)];
                }
            }
        }
    }

    return global;
}

// Generic fallback for meshes too small to form a process grid.  Elements are
// split into contiguous global ranges and the current energies are replicated
// once per iteration.  This path is only used for extreme rank/mesh ratios.
std::vector<ElementDynamic> runSimulationLinear(const int n, const int n_iters,
                                                MPI_Comm comm, const int rank,
                                                const int size,
                                                MPI_Datatype elem_dyn_type) {
    const long long total_ll = static_cast<long long>(n) * static_cast<long long>(n);
    const int total = static_cast<int>(total_ll);

    const int start = static_cast<int>((total_ll * rank) / size);
    const int end = static_cast<int>((total_ll * (rank + 1)) / size);
    const int count = end - start;

    std::vector<val_t> energy(count, 0.0);
    std::vector<val_t> energy_swap(count, 0.0);
    std::vector<val_t> flux(count, 0.0);
    std::vector<val_t> transfer(count, 0.8);
    std::vector<val_t> external(count, 0.0);
    std::vector<int> num_connections(count, 0);
    std::vector<idx_t> connections(static_cast<size_t>(count) * 4);

    const Material materials[3] = {
        Material{0.8, 0.0},
        Material{0.8, 0.5},
        Material{0.8, -0.5}
    };

    for (int l = 0; l < count; ++l) {
        const int g = start + l;
        const int x = g / n;
        const int y = g % n;

        idx_t mat = DEFAULT_MAT_ID;
        if (x == 0 && y == 0) mat = INFLOW_MAT_ID;
        if (x == 0 && y == n - 1) mat = OUTFLOW_MAT_ID;
        if (x == n - 1 && y == 0) mat = OUTFLOW_MAT_ID;
        if (x == n - 1 && y == n - 1) mat = INFLOW_MAT_ID;

        transfer[l] = materials[mat].transfer_coeff;
        external[l] = materials[mat].external_flow;

        int nc = 0;
        if (x + 1 < n) connections[static_cast<size_t>(l) * 4 + nc++] = g + n;
        if (x > 0) connections[static_cast<size_t>(l) * 4 + nc++] = g - n;
        if (y + 1 < n) connections[static_cast<size_t>(l) * 4 + nc++] = g + 1;
        if (y > 0) connections[static_cast<size_t>(l) * 4 + nc++] = g - 1;
        num_connections[l] = nc;
    }

    std::vector<int> recv_counts(size);
    std::vector<int> recv_displs(size);
    for (int r = 0; r < size; ++r) {
        recv_displs[r] = static_cast<int>((total_ll * r) / size);
        recv_counts[r] = static_cast<int>((total_ll * (r + 1)) / size) - recv_displs[r];
    }

    std::vector<val_t> global_energy(static_cast<size_t>(total), 0.0);

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Allgatherv(energy.data(), count, MPI_DOUBLE,
                       global_energy.data(), recv_counts.data(),
                       recv_displs.data(), MPI_DOUBLE, comm);

        const val_t* __restrict__ ge = global_energy.data();
        for (int l = 0; l < count; ++l) {
            const val_t cur = energy[l];
            const val_t coeff = transfer[l];
            val_t total_flux = external[l];
            const int nc = num_connections[l];
            const idx_t* conn = &connections[static_cast<size_t>(l) * 4];
            for (int j = 0; j < nc; ++j) {
                const val_t other = ge[conn[j]];
                total_flux += (other - cur) * coeff * 1.0 * 0.25;
            }
            energy_swap[l] = cur + total_flux;
            flux[l] += std::abs(total_flux);
        }

        energy.swap(energy_swap);
    }

    std::vector<ElementDynamic> sendbuf(static_cast<size_t>(count));
    for (int l = 0; l < count; ++l) {
        sendbuf[l] = ElementDynamic{energy[l], flux[l]};
    }

    std::vector<ElementDynamic> recvbuf;
    if (rank == 0) {
        recvbuf.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(sendbuf.data(), count, elem_dyn_type,
                rank == 0 ? recvbuf.data() : nullptr,
                recv_counts.data(), recv_displs.data(),
                elem_dyn_type, 0, comm);

    return recvbuf;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root < 0) {
        if (world_rank == 0) {
            printf("Grid size must be non-negative\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {1, 1};
    const bool use_2d = chooseProcessGrid(n_elems_root, world_size, dims);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = world_rank;
    int size = world_size;

    if (use_2d) {
        int periods[2] = {0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    MPI_Datatype elem_dyn_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &elem_dyn_type);
    MPI_Type_commit(&elem_dyn_type);

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");

        printf("Building unstructured mesh...\n");

        // Report the same problem size and storage estimate as the serial code.
        const size_t static_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem =
            static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");

        printf("Running simulation...\n");
    }

    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<ElementDynamic> global;
    if (use_2d) {
        global = runSimulation2D(n_elems_root, n_iters, comm, rank, size,
                                 dims, elem_dyn_type);
    } else {
        global = runSimulationLinear(n_elems_root, n_iters, comm, rank, size,
                                     elem_dyn_type);
    }

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();

    bool valid = true;

    if (rank == 0) {
        auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        const uint64_t hash = computeHash(global);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global.size());
            for (const auto& elem : global) {
                energyData.push_back(elem.current_energy);
            }
            print_results(energyData, "ElementEnergy");
        }

        // Validation
        if (validate) {
            valid = validateResults(global);
        }
    }

    if (validate) {
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, comm);
        valid = (valid_int != 0);
    }

    if (use_2d) {
        MPI_Comm_free(&comm);
    }
    MPI_Type_free(&elem_dyn_type);
    MPI_Finalize();

    return valid ? 0 : 1;
}
