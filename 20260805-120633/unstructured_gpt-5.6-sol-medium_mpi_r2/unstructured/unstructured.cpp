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

// A rank owns a contiguous set of complete grid rows.  Rows 0 and
// local_rows + 1 in energy are receive-only halo rows.
struct Domain {
    int n = 0;
    int first_row = 0;
    int local_rows = 0;
    int rank = 0;
    int ranks = 1;
    val_t transfer_coeff = 0.8;
    std::vector<val_t> energy;
    std::vector<val_t> energy_next;
    std::vector<val_t> total_flux;
    std::vector<val_t> total_flux_next;
};

static void mpiCheck(int error, MPI_Comm comm, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, operation, length, message);
    MPI_Abort(comm, error);
}

static Domain buildDomain(int n, int rank, int ranks) {
    Domain domain;
    domain.n = n;
    domain.rank = rank;
    domain.ranks = ranks;

    const int base = n / ranks;
    const int remainder = n % ranks;
    domain.local_rows = base + (rank < remainder ? 1 : 0);
    domain.first_row = rank * base + std::min(rank, remainder);

    const size_t allocated_rows = static_cast<size_t>(domain.local_rows) + 2;
    const size_t allocated_elements = allocated_rows * static_cast<size_t>(n);
    domain.energy.assign(allocated_elements, 0.0);
    domain.energy_next.assign(allocated_elements, 0.0);
    domain.total_flux.assign(allocated_elements, 0.0);
    domain.total_flux_next.assign(allocated_elements, 0.0);
    return domain;
}

inline val_t externalFlow(int x, int y, int n) {
    if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1)) {
        return x == y ? 0.5 : -0.5;
    }
    return 0.0;
}

// Preserve the reference program's neighbor order: +x, -x, +y, -y.
static inline void updateRow(Domain& domain, int local_x) {
    const int n = domain.n;
    const int global_x = domain.first_row + local_x - 1;
    const size_t row = static_cast<size_t>(local_x) * n;
    const size_t next_row = row + n;
    const size_t previous_row = row - n;

    for (int y = 0; y < n; ++y) {
        const size_t i = row + static_cast<size_t>(y);
        const val_t current = domain.energy[i];
        val_t flux = externalFlow(global_x, y, n);
        // Keep the reference computation's left-associated arithmetic.  Folding
        // the two constants changes the low bits and therefore its result hash.
        if (global_x + 1 < n)
            flux += (domain.energy[next_row + y] - current) * domain.transfer_coeff * 0.25;
        if (global_x > 0)
            flux += (domain.energy[previous_row + y] - current) * domain.transfer_coeff * 0.25;
        if (y + 1 < n)
            flux += (domain.energy[i + 1] - current) * domain.transfer_coeff * 0.25;
        if (y > 0)
            flux += (domain.energy[i - 1] - current) * domain.transfer_coeff * 0.25;
        domain.energy_next[i] = current + flux;
        domain.total_flux_next[i] = domain.total_flux[i] + std::abs(flux);
    }
}

static void runSimulation(Domain& domain, int iterations, MPI_Comm comm) {
    if (domain.local_rows == 0) return;

    const int top = domain.first_row == 0 ? MPI_PROC_NULL : domain.rank - 1;
    const int bottom = domain.first_row + domain.local_rows == domain.n
                           ? MPI_PROC_NULL
                           : domain.rank + 1;
    const size_t first = static_cast<size_t>(domain.n);
    const size_t last = static_cast<size_t>(domain.local_rows) * domain.n;
    const size_t bottom_halo = last + domain.n;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        MPI_Request requests[4];
        mpiCheck(MPI_Irecv(domain.energy.data(), domain.n, MPI_DOUBLE, top, 1,
                           comm, &requests[0]), comm, "MPI_Irecv(top)");
        mpiCheck(MPI_Irecv(domain.energy.data() + bottom_halo, domain.n, MPI_DOUBLE,
                           bottom, 0, comm, &requests[1]), comm, "MPI_Irecv(bottom)");
        mpiCheck(MPI_Isend(domain.energy.data() + first, domain.n, MPI_DOUBLE, top, 0,
                           comm, &requests[2]), comm, "MPI_Isend(top)");
        mpiCheck(MPI_Isend(domain.energy.data() + last, domain.n, MPI_DOUBLE, bottom, 1,
                           comm, &requests[3]), comm, "MPI_Isend(bottom)");

        // These rows do not consume halo data, so their work hides communication.
        for (int local_x = 2; local_x < domain.local_rows; ++local_x) {
            updateRow(domain, local_x);
        }
        mpiCheck(MPI_Waitall(4, requests, MPI_STATUSES_IGNORE), comm, "MPI_Waitall");

        updateRow(domain, 1);
        if (domain.local_rows > 1) updateRow(domain, domain.local_rows);
        std::swap(domain.energy, domain.energy_next);
        std::swap(domain.total_flux, domain.total_flux_next);
    }
}

struct ValidationData {
    val_t energy_sum;
    val_t flux_sum;
    val_t energy_min;
    val_t energy_max;
};

static ValidationData localValidation(const Domain& domain) {
    ValidationData result{0.0, 0.0, std::numeric_limits<val_t>::max(),
                          std::numeric_limits<val_t>::lowest()};
    const size_t begin = static_cast<size_t>(domain.n);
    const size_t end = begin + static_cast<size_t>(domain.local_rows) * domain.n;
    for (size_t i = begin; i < end; ++i) {
        result.energy_sum += domain.energy[i];
        result.flux_sum += domain.total_flux[i];
        result.energy_min = std::min(result.energy_min, domain.energy[i]);
        result.energy_max = std::max(result.energy_max, domain.energy[i]);
    }
    return result;
}

static bool validateResults(const Domain& domain, MPI_Comm comm) {
    const ValidationData local = localValidation(domain);
    ValidationData global{};
    mpiCheck(MPI_Reduce(&local.energy_sum, &global.energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                        comm), comm, "MPI_Reduce(energy sum)");
    mpiCheck(MPI_Reduce(&local.flux_sum, &global.flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
                        comm), comm, "MPI_Reduce(flux sum)");
    mpiCheck(MPI_Reduce(&local.energy_min, &global.energy_min, 1, MPI_DOUBLE, MPI_MIN, 0,
                        comm), comm, "MPI_Reduce(energy min)");
    mpiCheck(MPI_Reduce(&local.energy_max, &global.energy_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                        comm), comm, "MPI_Reduce(energy max)");

    bool valid = true;
    if (domain.rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", global.energy_sum);
        std::printf("  Flux sum: %.2f\n", global.flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", global.energy_min, global.energy_max);
        if (!std::isfinite(global.energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = false;
        }
        if (std::abs(global.energy_sum) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(global.flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = false;
        }
        if (!std::isfinite(global.energy_min) || !std::isfinite(global.energy_max)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = false;
        }
        if (valid) std::printf("  Validation: PASSED\n");
    }
    int valid_int = valid ? 1 : 0;
    mpiCheck(MPI_Bcast(&valid_int, 1, MPI_INT, 0, comm), comm, "MPI_Bcast(validation)");
    return valid_int != 0;
}

static uint64_t localHash(const Domain& domain) {
    uint64_t result = 0;
    const size_t begin = static_cast<size_t>(domain.n);
    const size_t count = static_cast<size_t>(domain.local_rows) * domain.n;
    const uint64_t global_begin = static_cast<uint64_t>(domain.first_row) * domain.n;
    for (size_t j = 0; j < count; ++j) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &domain.energy[begin + j], sizeof(energy_bits));
        std::memcpy(&flux_bits, &domain.total_flux[begin + j], sizeof(flux_bits));
        const uint64_t i = global_begin + j;
        result ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
}

static std::vector<double> gatherEnergy(const Domain& domain, MPI_Comm comm) {
    const int local_count = domain.local_rows * domain.n;
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> global;
    if (domain.rank == 0) {
        counts.resize(domain.ranks);
        displacements.resize(domain.ranks);
    }
    mpiCheck(MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm),
             comm, "MPI_Gather(counts)");
    if (domain.rank == 0) {
        int offset = 0;
        for (int r = 0; r < domain.ranks; ++r) {
            displacements[r] = offset;
            offset += counts[r];
        }
        global.resize(static_cast<size_t>(domain.n) * domain.n);
    }
    mpiCheck(MPI_Gatherv(domain.energy.data() + domain.n, local_count, MPI_DOUBLE,
                         global.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0,
                         comm), comm, "MPI_Gatherv(results)");
    return global;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    mpiCheck(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN), comm, "MPI_Comm_set_errhandler");
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(comm, &rank), comm, "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(comm, &ranks), comm, "MPI_Comm_size");

    int n = 512;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool arguments_valid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            help = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            arguments_valid = false;
        }
    }
    // MPI message counts and result collection use int counts.
    if (n <= 0 || iterations < 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) std::fprintf(stderr, "Grid size must be positive, iterations nonnegative, and N*N must fit in an MPI int count.\n");
        arguments_valid = false;
    }
    if (help || !arguments_valid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return arguments_valid ? 0 : 1;
    }

    Domain domain = buildDomain(n, rank, ranks);
    const uint64_t elements = static_cast<uint64_t>(n) * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %llu elements\n", n, n,
                    static_cast<unsigned long long>(elements));
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building distributed unstructured mesh...\n");
    }

    const uint64_t local_bytes = 4ULL * (static_cast<uint64_t>(domain.local_rows) + 2) * n * sizeof(val_t);
    uint64_t aggregate_bytes = 0, maximum_bytes = 0;
    mpiCheck(MPI_Reduce(&local_bytes, &aggregate_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, comm),
             comm, "MPI_Reduce(memory sum)");
    mpiCheck(MPI_Reduce(&local_bytes, &maximum_bytes, 1, MPI_UINT64_T, MPI_MAX, 0, comm),
             comm, "MPI_Reduce(memory max)");
    if (rank == 0) {
        std::printf("Distributed memory usage: %.2f MB aggregate, %.2f MB maximum/rank\n\n",
                    aggregate_bytes / (1024.0 * 1024.0), maximum_bytes / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    mpiCheck(MPI_Barrier(comm), comm, "MPI_Barrier(start)");
    const double start = MPI_Wtime();
    runSimulation(domain, iterations, comm);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpiCheck(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm),
             comm, "MPI_Reduce(time)");

    const uint64_t local_hash = localHash(domain);
    uint64_t global_hash = 0;
    mpiCheck(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm),
             comm, "MPI_Reduce(hash)");

    if (rank == 0) {
        const double duration_ms = elapsed * 1000.0;
        const int measured_iterations = std::max(iterations - 1, 1);
        const double time_per_iteration = duration_ms / measured_iterations;
        const double giga_elements_per_second = elapsed > 0.0
            ? (static_cast<double>(measured_iterations) * elements) / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iteration);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(global_hash));
    }

    if (printResults) {
        std::vector<double> energy = gatherEnergy(domain, comm);
        if (rank == 0) print_results(energy, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(domain, comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
