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

constexpr val_t kTransferCoeff = 0.8;
constexpr val_t kFluxScale = kTransferCoeff * 0.25;

struct Decomp1D {
    int start;
    int count;
};

struct Domain {
    int n_global = 0;
    int local_nx = 0;
    int local_ny = 0;
    int offset_x = 0;
    int offset_y = 0;
    int stride = 0;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux;
    std::vector<val_t> flux_swap;
    std::vector<val_t> external_flow;
};

inline size_t idx(const Domain& domain, int x, int y) {
    return static_cast<size_t>(x) * static_cast<size_t>(domain.stride) + static_cast<size_t>(y);
}

Decomp1D split_range(const int n, const int coord, const int dims) {
    const int base = n / dims;
    const int rem = n % dims;
    const int count = base + (coord < rem ? 1 : 0);
    const int start = coord * base + std::min(coord, rem);
    return {start, count};
}

bool compute_cart_dims(const int world_size, const int n_global, int dims[2]) {
    int best_d0 = 0;
    int best_d1 = 0;
    int best_ratio = std::numeric_limits<int>::max();
    const int limit = std::min(n_global, world_size);
    for (int d = 1; d <= limit; ++d) {
        if (world_size % d != 0) {
            continue;
        }
        const int other = world_size / d;
        if (other > n_global) {
            continue;
        }
        const int ratio = std::abs(other - d);
        if (ratio < best_ratio) {
            best_ratio = ratio;
            best_d0 = d;
            best_d1 = other;
        }
    }
    if (best_d0 == 0) {
        return false;
    }
    dims[0] = best_d0;
    dims[1] = best_d1;
    return true;
}

void initializeDomain(Domain& domain, const int n_global, const int coords[2], const int dims[2]) {
    domain.n_global = n_global;
    const Decomp1D dx = split_range(n_global, coords[0], dims[0]);
    const Decomp1D dy = split_range(n_global, coords[1], dims[1]);
    domain.local_nx = dx.count;
    domain.local_ny = dy.count;
    domain.offset_x = dx.start;
    domain.offset_y = dy.start;
    domain.stride = domain.local_ny + 2;

    const size_t total = static_cast<size_t>(domain.local_nx + 2) * static_cast<size_t>(domain.local_ny + 2);
    domain.energy.assign(total, 0.0);
    domain.energy_swap.assign(total, 0.0);
    domain.flux.assign(total, 0.0);
    domain.flux_swap.assign(total, 0.0);
    domain.external_flow.assign(total, 0.0);

    for (int x = 1; x <= domain.local_nx; ++x) {
        const int gx = domain.offset_x + x - 1;
        for (int y = 1; y <= domain.local_ny; ++y) {
            const int gy = domain.offset_y + y - 1;
            val_t flow = 0.0;
            if ((gx == 0 && gy == 0) || (gx == n_global - 1 && gy == n_global - 1)) {
                flow = 0.5;
            } else if ((gx == 0 && gy == n_global - 1) || (gx == n_global - 1 && gy == 0)) {
                flow = -0.5;
            }
            domain.external_flow[idx(domain, x, y)] = flow;
        }
    }
}

void exchangeHalos(Domain& domain, MPI_Comm cart_comm, const int north, const int south, const int west,
                   const int east, const MPI_Datatype column_type) {
    const int local_nx = domain.local_nx;
    const int local_ny = domain.local_ny;
    const int stride = domain.stride;
    val_t* energy = domain.energy.data();

    const auto index = [stride](int x, int y) {
        return static_cast<size_t>(x) * static_cast<size_t>(stride) + static_cast<size_t>(y);
    };

    MPI_Sendrecv(&energy[index(1, 1)], local_ny, MPI_DOUBLE, north, 0,
                 &energy[index(local_nx + 1, 1)], local_ny, MPI_DOUBLE, south, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&energy[index(local_nx, 1)], local_ny, MPI_DOUBLE, south, 1,
                 &energy[index(0, 1)], local_ny, MPI_DOUBLE, north, 1,
                 cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&energy[index(1, 1)], local_nx, column_type, west, 2,
                 &energy[index(1, local_ny + 1)], local_nx, column_type, east, 2,
                 cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&energy[index(1, local_ny)], local_nx, column_type, east, 3,
                 &energy[index(1, 0)], local_nx, column_type, west, 3,
                 cart_comm, MPI_STATUS_IGNORE);
}

void runSimulation(Domain& domain, const int n_iters, MPI_Comm cart_comm, const int north, const int south,
                   const int west, const int east, const MPI_Datatype column_type) {
    const int n_global = domain.n_global;
    const int local_nx = domain.local_nx;
    const int local_ny = domain.local_ny;
    const int stride = domain.stride;

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(domain, cart_comm, north, south, west, east, column_type);

        for (int x = 1; x <= local_nx; ++x) {
            const int gx = domain.offset_x + x - 1;
            const bool has_north = gx > 0;
            const bool has_south = (gx + 1) < n_global;

            for (int y = 1; y <= local_ny; ++y) {
                const int gy = domain.offset_y + y - 1;
                const bool has_west = gy > 0;
                const bool has_east = (gy + 1) < n_global;
                const size_t id = static_cast<size_t>(x) * static_cast<size_t>(stride) + static_cast<size_t>(y);

                const val_t this_energy = domain.energy[id];
                val_t total_flux = domain.external_flow[id];

                if (has_north) {
                    total_flux += kFluxScale * (domain.energy[id - stride] - this_energy);
                }
                if (has_south) {
                    total_flux += kFluxScale * (domain.energy[id + stride] - this_energy);
                }
                if (has_west) {
                    total_flux += kFluxScale * (domain.energy[id - 1] - this_energy);
                }
                if (has_east) {
                    total_flux += kFluxScale * (domain.energy[id + 1] - this_energy);
                }

                domain.energy_swap[id] = this_energy + total_flux;
                domain.flux_swap[id] = domain.flux[id] + std::abs(total_flux);
            }
        }

        std::swap(domain.energy, domain.energy_swap);
        std::swap(domain.flux, domain.flux_swap);
    }
}

bool validateResults(const Domain& domain, MPI_Comm cart_comm) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (int x = 1; x <= domain.local_nx; ++x) {
        for (int y = 1; y <= domain.local_ny; ++y) {
            const size_t id = idx(domain, x, y);
            const val_t energy = domain.energy[id];
            energy_sum += energy;
            flux_sum += domain.flux[id];
            energy_max = std::max(energy, energy_max);
            energy_min = std::min(energy, energy_min);
        }
    }

    val_t global_energy_sum = 0.0;
    val_t global_flux_sum = 0.0;
    val_t global_energy_max = 0.0;
    val_t global_energy_min = 0.0;

    MPI_Allreduce(&energy_sum, &global_energy_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    MPI_Allreduce(&flux_sum, &global_flux_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
    MPI_Allreduce(&energy_max, &global_energy_max, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    MPI_Allreduce(&energy_min, &global_energy_min, 1, MPI_DOUBLE, MPI_MIN, cart_comm);

    int rank = 0;
    MPI_Comm_rank(cart_comm, &rank);

    bool valid = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_energy_sum);
        printf("  Flux sum: %.2f\n", global_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", global_energy_min, global_energy_max);

        constexpr val_t energy_epsilon = 1e-8;
        if (!std::isfinite(global_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }

        if (std::abs(global_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(global_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }

        if (!std::isfinite(global_energy_max) || !std::isfinite(global_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }

        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    int valid_int = valid ? 1 : 0;
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, cart_comm);
    return valid_int == 1;
}

uint64_t computeHash(const Domain& domain, MPI_Comm cart_comm) {
    uint64_t local_hash = 0;

    for (int x = 1; x <= domain.local_nx; ++x) {
        const int gx = domain.offset_x + x - 1;
        for (int y = 1; y <= domain.local_ny; ++y) {
            const int gy = domain.offset_y + y - 1;
            const size_t id = idx(domain, x, y);
            const uint64_t i = static_cast<uint64_t>(gx) * static_cast<uint64_t>(domain.n_global) +
                               static_cast<uint64_t>(gy);
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&domain.energy[id]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&domain.flux[id]);
            local_hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart_comm);
    MPI_Bcast(&global_hash, 1, MPI_UINT64_T, 0, cart_comm);
    return global_hash;
}

void gatherEnergyData(const Domain& domain, MPI_Comm cart_comm, const int dims[2],
                      std::vector<double>& energyData) {
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &size);

    const int local_nx = domain.local_nx;
    const int local_ny = domain.local_ny;
    std::vector<val_t> sendbuf(static_cast<size_t>(local_nx) * static_cast<size_t>(local_ny));
    for (int x = 1; x <= local_nx; ++x) {
        const size_t src = idx(domain, x, 1);
        const size_t dst = static_cast<size_t>(x - 1) * static_cast<size_t>(local_ny);
        std::memcpy(&sendbuf[dst], &domain.energy[src], static_cast<size_t>(local_ny) * sizeof(val_t));
    }

    if (rank == 0) {
        energyData.resize(static_cast<size_t>(domain.n_global) * static_cast<size_t>(domain.n_global));

        auto placeBlock = [&](const std::vector<val_t>& buffer, int offset_x, int offset_y, int nx, int ny) {
            for (int x = 0; x < nx; ++x) {
                const size_t src = static_cast<size_t>(x) * static_cast<size_t>(ny);
                const size_t dst = (static_cast<size_t>(offset_x + x) * static_cast<size_t>(domain.n_global)) +
                                   static_cast<size_t>(offset_y);
                std::memcpy(&energyData[dst], &buffer[src], static_cast<size_t>(ny) * sizeof(val_t));
            }
        };

        placeBlock(sendbuf, domain.offset_x, domain.offset_y, local_nx, local_ny);

        for (int r = 1; r < size; ++r) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(cart_comm, r, 2, coords);
            const Decomp1D dx = split_range(domain.n_global, coords[0], dims[0]);
            const Decomp1D dy = split_range(domain.n_global, coords[1], dims[1]);
            std::vector<val_t> recvbuf(static_cast<size_t>(dx.count) * static_cast<size_t>(dy.count));
            MPI_Recv(recvbuf.data(), static_cast<int>(recvbuf.size()), MPI_DOUBLE, r, 42, cart_comm,
                     MPI_STATUS_IGNORE);
            placeBlock(recvbuf, dx.start, dy.start, dx.count, dy.count);
        }
    } else {
        MPI_Send(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE, 0, 42, cart_comm);
    }
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

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0;

    if (world_rank == 0) {
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
                parse_status = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_status = 1;
                break;
            }
        }
    }

    MPI_Bcast(&parse_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parse_status != 0) {
        MPI_Finalize();
        return parse_status == 1 ? 1 : 0;
    }

    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validate_int = validate ? 1 : 0;
    int printResults_int = printResults ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validate_int == 1;
    printResults = printResults_int == 1;

    int dims[2] = {0, 0};
    const bool dims_ok = compute_cart_dims(world_size, n_elems_root, dims);
    if (!dims_ok) {
        if (world_rank == 0) {
            printf("ERROR: MPI size %d cannot be decomposed for grid size %d. Use <= %d ranks.\n",
                   world_size, n_elems_root, n_elems_root * n_elems_root);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const int periods[2] = {0, 0};
    MPI_Comm cart_comm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int cart_rank = 0;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2] = {0, 0};
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    Domain domain;
    initializeDomain(domain, n_elems_root, coords, dims);

    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    MPI_Type_vector(domain.local_nx, 1, domain.stride, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    const int n_elems = n_elems_root * n_elems_root;

    if (cart_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (decomposition %d x %d)\n", world_size, dims[0], dims[1]);
        printf("\n");
        printf("Building distributed mesh...\n");
    }

    const uint64_t local_static = static_cast<uint64_t>(domain.external_flow.size()) * sizeof(val_t);
    const uint64_t local_dynamic = static_cast<uint64_t>(domain.energy.size() + domain.energy_swap.size() +
                                                        domain.flux.size() + domain.flux_swap.size()) *
                                   sizeof(val_t);
    uint64_t global_static = 0;
    uint64_t global_dynamic = 0;
    MPI_Reduce(&local_static, &global_static, 1, MPI_UINT64_T, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_dynamic, &global_dynamic, 1, MPI_UINT64_T, MPI_SUM, 0, cart_comm);

    if (cart_rank == 0) {
        const uint64_t global_total = global_static + global_dynamic;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               global_total / (1024.0 * 1024.0),
               global_static / (1024.0 * 1024.0),
               global_dynamic / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(cart_comm);
    const double start = MPI_Wtime();

    runSimulation(domain, n_iters, cart_comm, north, south, west, east, column_type);

    MPI_Barrier(cart_comm);
    const double elapsed = MPI_Wtime() - start;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (cart_rank == 0) {
        const double duration_ms = max_elapsed * 1000.0;
        printf("Computation time: %.0f ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * static_cast<double>(n_elems)) /
            (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    const uint64_t hash = computeHash(domain, cart_comm);
    if (cart_rank == 0) {
        printf("  Result hash: %016llX\n", static_cast<unsigned long long>(hash));
        printf("\n");
    }

    if (printResults) {
        std::vector<double> energyData;
        gatherEnergyData(domain, cart_comm, dims, energyData);
        if (cart_rank == 0) {
            print_results(energyData, "ElementEnergy");
        }
    }

    if (validate) {
        const bool valid = validateResults(domain, cart_comm);
        if (!valid) {
            MPI_Type_free(&column_type);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Type_free(&column_type);
    MPI_Finalize();
    return 0;
}
