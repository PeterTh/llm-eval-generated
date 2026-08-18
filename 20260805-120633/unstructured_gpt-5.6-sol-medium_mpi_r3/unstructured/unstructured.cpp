#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

inline val_t computeFlux(val_t current, val_t neighbor) {
    // Match the reference operation ordering (including its intermediate
    // rounding) rather than pre-combining the two constants.
    return (neighbor - current) * 0.8 * 1.0 * 0.25;
}

struct Domain {
    int grid_size = 0;
    int first_row = 0;
    int row_count = 0;
    int rank = 0;
    int ranks = 1;
    std::vector<val_t> energy;
    std::vector<val_t> total_flux;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux_swap;
    std::vector<val_t> upper_halo;
    std::vector<val_t> lower_halo;
};

// Use a balanced, contiguous row decomposition.  Contiguous rows make both
// halo messages contiguous and preserve global element ordering for output.
Domain buildSquare2D(int grid_size, int rank, int ranks) {
    Domain domain;
    domain.grid_size = grid_size;
    domain.rank = rank;
    domain.ranks = ranks;

    const int base = grid_size / ranks;
    const int remainder = grid_size % ranks;
    domain.row_count = base + (rank < remainder ? 1 : 0);
    domain.first_row = rank * base + std::min(rank, remainder);

    const size_t local_elements =
        static_cast<size_t>(domain.row_count) * static_cast<size_t>(grid_size);
    domain.energy.resize(local_elements);
    domain.total_flux.resize(local_elements);
    domain.energy_swap.resize(local_elements);
    domain.total_flux_swap.resize(local_elements);
    if (domain.row_count != 0) {
        domain.upper_halo.resize(static_cast<size_t>(grid_size));
        domain.lower_halo.resize(static_cast<size_t>(grid_size));
    }
    return domain;
}

inline void updateElement(const Domain& domain, int local_x, int y,
                          const val_t* upper, const val_t* lower,
                          val_t* energy_output, val_t* flux_output) {
    const int n = domain.grid_size;
    const int global_x = domain.first_row + local_x;
    const size_t local_idx = static_cast<size_t>(local_x) * n + y;
    const val_t current = domain.energy[local_idx];

    val_t total_flux = 0.0;
    const bool corner = (global_x == 0 || global_x == n - 1) &&
                        (y == 0 || y == n - 1);
    if (corner) {
        // The two diagonal corners are inflows and the other two outflows.
        total_flux = (global_x == y) ? 0.5 : -0.5;
    }

    // Keep the original connection order: +x, -x, +y, -y.  This preserves
    // the floating-point result bit for bit, independent of the partition.
    if (global_x + 1 < n) {
        const val_t neighbor = (local_x + 1 < domain.row_count)
            ? domain.energy[local_idx + n]
            : lower[y];
        total_flux += computeFlux(current, neighbor);
    }
    if (global_x > 0) {
        const val_t neighbor = (local_x > 0)
            ? domain.energy[local_idx - n]
            : upper[y];
        total_flux += computeFlux(current, neighbor);
    }
    if (y + 1 < n) {
        total_flux += computeFlux(current, domain.energy[local_idx + 1]);
    }
    if (y > 0) {
        total_flux += computeFlux(current, domain.energy[local_idx - 1]);
    }

    energy_output[local_idx] = current + total_flux;
    flux_output[local_idx] = domain.total_flux[local_idx] + std::abs(total_flux);
}

void updateRows(const Domain& domain, int begin, int end,
                const val_t* upper, const val_t* lower,
                val_t* energy_output, val_t* flux_output) {
    for (int local_x = begin; local_x < end; ++local_x) {
        for (int y = 0; y < domain.grid_size; ++y) {
            updateElement(domain, local_x, y, upper, lower, energy_output,
                          flux_output);
        }
    }
}

void runSimulation(Domain& domain, int iterations) {
    if (domain.row_count == 0) {
        return;
    }

    const int previous = domain.first_row == 0 ? MPI_PROC_NULL : domain.rank - 1;
    const int next = domain.first_row + domain.row_count == domain.grid_size
        ? MPI_PROC_NULL : domain.rank + 1;
    const int n = domain.grid_size;

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;

        if (previous != MPI_PROC_NULL) {
            MPI_Irecv(domain.upper_halo.data(), n, MPI_DOUBLE, previous, 101,
                      MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Isend(domain.energy.data(), n, MPI_DOUBLE,
                      previous, 102, MPI_COMM_WORLD,
                      &requests[request_count++]);
        }
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(domain.lower_halo.data(), n, MPI_DOUBLE, next, 102,
                      MPI_COMM_WORLD, &requests[request_count++]);
            const size_t last_row = static_cast<size_t>(domain.row_count - 1) * n;
            MPI_Isend(domain.energy.data() + last_row, n, MPI_DOUBLE,
                      next, 101, MPI_COMM_WORLD,
                      &requests[request_count++]);
        }

        // Rows not touching a rank boundary do not depend on incoming halos.
        if (domain.row_count > 2) {
            updateRows(domain, 1, domain.row_count - 1,
                       domain.upper_halo.data(), domain.lower_halo.data(),
                       domain.energy_swap.data(), domain.total_flux_swap.data());
        }
        if (request_count != 0) {
            MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
        }

        updateRows(domain, 0, 1, domain.upper_halo.data(),
                   domain.lower_halo.data(), domain.energy_swap.data(),
                   domain.total_flux_swap.data());
        if (domain.row_count > 1) {
            updateRows(domain, domain.row_count - 1, domain.row_count,
                       domain.upper_halo.data(), domain.lower_halo.data(),
                       domain.energy_swap.data(), domain.total_flux_swap.data());
        }
        domain.energy.swap(domain.energy_swap);
        domain.total_flux.swap(domain.total_flux_swap);
    }
}

uint64_t computeLocalHash(const Domain& domain) {
    uint64_t result = 0;
    const uint64_t global_begin =
        static_cast<uint64_t>(domain.first_row) * domain.grid_size;
    for (size_t i = 0; i < domain.energy.size(); ++i) {
        uint64_t energy_bits;
        uint64_t flux_bits;
        std::memcpy(&energy_bits, &domain.energy[i],
                    sizeof(energy_bits));
        std::memcpy(&flux_bits, &domain.total_flux[i],
                    sizeof(flux_bits));
        const uint64_t global_i = global_begin + i;
        result ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

bool validateResults(const Domain& domain) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < domain.energy.size(); ++i) {
        local_energy_sum += domain.energy[i];
        local_flux_sum += domain.total_flux[i];
        local_max = std::max(local_max, domain.energy[i]);
        local_min = std::min(local_min, domain.energy[i]);
    }

    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = 0.0;
    val_t energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);

    int valid = 1;
    if (domain.rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(energy_sum) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) {
            std::printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

std::vector<double> gatherEnergy(const Domain& domain) {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> global;
    if (domain.rank == 0) {
        counts.resize(static_cast<size_t>(domain.ranks));
        displacements.resize(static_cast<size_t>(domain.ranks));
        const int base = domain.grid_size / domain.ranks;
        const int remainder = domain.grid_size % domain.ranks;
        int offset = 0;
        for (int rank = 0; rank < domain.ranks; ++rank) {
            const int rows = base + (rank < remainder ? 1 : 0);
            counts[rank] = rows * domain.grid_size;
            displacements[rank] = offset;
            offset += counts[rank];
        }
        global.resize(static_cast<size_t>(domain.grid_size) * domain.grid_size);
    }

    MPI_Gatherv(domain.energy.data(), static_cast<int>(domain.energy.size()),
                MPI_DOUBLE,
                global.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return global;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int grid_size = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            grid_size = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parse_status = 2;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parse_status = 1;
        }
    }
    if (grid_size <= 0 || iterations < 0) {
        if (rank == 0) {
            std::printf("Grid size must be positive and iterations non-negative.\n");
        }
        parse_status = 1;
    }
    if (parse_status != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    const uint64_t element_count =
        static_cast<uint64_t>(grid_size) * static_cast<uint64_t>(grid_size);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %llu elements\n", grid_size,
                    grid_size, static_cast<unsigned long long>(element_count));
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
    }

    Domain domain = buildSquare2D(grid_size, rank, ranks);
    const uint64_t local_memory =
        static_cast<uint64_t>(domain.energy.capacity() +
                              domain.total_flux.capacity() +
                              domain.energy_swap.capacity() +
                              domain.total_flux_swap.capacity()) * sizeof(val_t) +
        static_cast<uint64_t>(domain.upper_halo.capacity() +
                              domain.lower_halo.capacity()) * sizeof(val_t);
    uint64_t total_memory = 0;
    uint64_t max_rank_memory = 0;
    MPI_Reduce(&local_memory, &total_memory, 1, MPI_UINT64_T, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_memory, &max_rank_memory, 1, MPI_UINT64_T, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Distributed memory usage: %.2f MB total, %.2f MB max/rank\n\n",
                    total_memory / (1024.0 * 1024.0),
                    max_rank_memory / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(domain, iterations);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    const uint64_t local_hash = computeLocalHash(domain);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const int measured_iterations = std::max(iterations - 1, 1);
        const double milliseconds = seconds * 1000.0;
        const double time_per_iteration = milliseconds / measured_iterations;
        const double giga_elements_per_second = seconds > 0.0
            ? (static_cast<double>(measured_iterations) * element_count) /
                  seconds / 1e9
            : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n",
                    giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n",
                    giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(result_hash));
    }

    if (printResults) {
        std::vector<double> energy = gatherEnergy(domain);
        if (rank == 0) {
            print_results(energy, "ElementEnergy");
        }
    }

    const bool valid = !validate || validateResults(domain);
    MPI_Finalize();
    return valid ? 0 : 1;
}
