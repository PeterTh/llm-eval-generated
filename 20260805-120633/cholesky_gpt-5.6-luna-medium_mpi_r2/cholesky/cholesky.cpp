#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static size_t first_row(const int rank, const int size, const size_t n) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(size);
}

static size_t row_count(const int rank, const int size, const size_t n) {
    return first_row(rank + 1, size, n) - first_row(rank, size, n);
}

static int owner_of_row(const size_t row, const int size, const size_t n) {
    int low = 0;
    int high = size;
    while (low + 1 < high) {
        const int middle = low + (high - low) / 2;
        if (first_row(middle, size, n) <= row)
            low = middle;
        else
            high = middle;
    }
    return low;
}

// Generate A = B B^T + nI without replicating B.  The seed is passed in rank
// order so the generated values are identical to the serial implementation.
void generatePositiveDefiniteMatrix(std::vector<double>& a, const size_t n,
                                    const int rank, const int size) {
    const size_t rows = row_count(rank, size, n);
    std::vector<double> b(rows * n);
    unsigned int seed = 42;

    for (int r = 0; r < size; ++r) {
        if (rank == r) {
            for (double& value : b)
                value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (r + 1 < size)
                MPI_Send(&seed, 1, MPI_UNSIGNED, r + 1, 7, MPI_COMM_WORLD);
        } else if (rank == r + 1) {
            MPI_Recv(&seed, 1, MPI_UNSIGNED, r, 7, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    }

    std::vector<double> panel = b;
    int panel_owner = rank;
    for (int step = 0; step < size; ++step) {
        const size_t panel_rows = row_count(panel_owner, size, n);
        const size_t global_panel_row = first_row(panel_owner, size, n);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < panel_rows; ++j) {
                double sum = 0.0;
                const double* bi = b.data() + i * n;
                const double* bj = panel.data() + j * n;
                for (size_t k = 0; k < n; ++k)
                    sum += bi[k] * bj[k];
                a[i * n + global_panel_row + j] = sum;
            }
        }

        if (step + 1 < size) {
            const int receive_owner = (panel_owner + size - 1) % size;
            const size_t send_count = row_count(panel_owner, size, n) * n;
            const size_t receive_count = row_count(receive_owner, size, n) * n;
            std::vector<double> received(receive_count);
            MPI_Sendrecv(panel.data(), static_cast<int>(send_count), MPI_DOUBLE,
                         (rank + 1) % size, 8, received.data(),
                         static_cast<int>(receive_count), MPI_DOUBLE,
                         (rank + size - 1) % size, 8, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            panel.swap(received);
            panel_owner = receive_owner;
        }
    }
    for (size_t i = 0; i < rows; ++i)
        a[i * n + first_row(rank, size, n) + i] += static_cast<double>(n);
}

// Distributed row-block Cholesky.  Each panel row is broadcast once, and all
// trailing updates are performed by the rank owning the corresponding row.
bool choleskyDecomposition(std::vector<double>& a, const size_t n,
                           const int rank, const int size) {
    const size_t begin = first_row(rank, size, n);
    const size_t rows = row_count(rank, size, n);
    std::vector<double> panel(n);
    bool success = true;

    for (size_t j = 0; j < n; ++j) {
        const int owner = owner_of_row(j, size, n);
        if (rank == owner) {
            const size_t local = j - begin;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += a[local * n + k] * a[local * n + k];
            const double value = a[local * n + j] - sum;
            if (value <= 0.0) {
                std::fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
                success = false;
                std::fill(panel.begin(), panel.end(), 0.0);
            } else {
                a[local * n + j] = std::sqrt(value);
                std::copy(a.begin() + local * n, a.begin() + (local + 1) * n,
                          panel.begin());
                std::fill(panel.begin() + j + 1, panel.end(), 0.0);
            }
        }
        MPI_Bcast(panel.data(), static_cast<int>(n), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);
        int global_success = 0;
        const int local_success = success ? 1 : 0;
        MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if (!global_success)
            return false;

        const size_t local_begin = std::max(begin, j + 1);
        for (size_t global_i = local_begin; global_i < begin + rows; ++global_i) {
            const size_t local_i = global_i - begin;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k)
                sum += a[local_i * n + k] * panel[k];
            a[local_i * n + j] = (a[local_i * n + j] - sum) / panel[j];
        }
        if (j >= begin && j < begin + rows) {
            const size_t local = j - begin;
            std::fill(a.begin() + local * n + j + 1,
                      a.begin() + (local + 1) * n, 0.0);
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n == 0 || size > static_cast<int>(n)) {
        if (rank == 0) std::fprintf(stderr, "Matrix size must be at least the MPI process count\n");
        MPI_Finalize();
        return 1;
    }

    const size_t rows = row_count(rank, size, n);
    std::vector<double> a(rows * n), original;
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\nGenerating positive definite matrix...\n", size);
    }
    generatePositiveDefiniteMatrix(a, n, rank, size);
    if (validate) original = a;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    const double start = MPI_Wtime();
    const bool local_success = choleskyDecomposition(a, n, rank, size);
    int global_success = 0;
    const int local_flag = local_success ? 1 : 0;
    MPI_Allreduce(&local_flag, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    if (!global_success) { MPI_Finalize(); return 1; }

    std::vector<int> counts(size), displacements(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(row_count(r, size, n) * n);
        displacements[r] = static_cast<int>(first_row(r, size, n) * n);
    }
    std::vector<double> full_a, full_original;
    if (rank == 0) { full_a.resize(n * n); if (validate) full_original.resize(n * n); }
    MPI_Gatherv(a.data(), counts[rank], MPI_DOUBLE, full_a.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate)
        MPI_Gatherv(original.data(), counts[rank], MPI_DOUBLE, full_original.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f GFLOPS\n", (n * n * n / 3.0) / elapsed / 1e9);
        if (printResults) print_results(full_a, "CholeskyL");
        if (validate) {
            double max_error = 0.0, max_relative = 0.0;
            for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += full_a[i*n+k] * full_a[j*n+k];
                const double error = std::fabs(sum - full_original[i*n+j]);
                max_error = std::max(max_error, error);
                max_relative = std::max(max_relative, error / (std::fabs(full_original[i*n+j]) + 1e-10));
            }
            std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, max_relative);
            std::printf("Validation: %s\n", max_relative <= 1e-6 ? "PASSED" : "FAILED");
            MPI_Finalize();
            return max_relative <= 1e-6 ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
