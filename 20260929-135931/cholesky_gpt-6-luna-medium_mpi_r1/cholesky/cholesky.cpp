#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t limit = std::min(i, j);
            for (size_t k = 0; k <= limit; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

static int rowOwner(size_t row, size_t n, int processes) {
    const size_t base = n / static_cast<size_t>(processes);
    const size_t extra = n % static_cast<size_t>(processes);
    const size_t large = (base + 1) * extra;
    return row < large ? static_cast<int>(row / (base + 1))
                       : static_cast<int>(extra + (row - large) / base);
}

static size_t localRow(size_t row, size_t n, int processes) {
    const size_t base = n / static_cast<size_t>(processes);
    const size_t extra = n % static_cast<size_t>(processes);
    const size_t large = (base + 1) * extra;
    return row < large ? row % (base + 1) : (row - large) % base;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        n * n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Error: matrix size is unsupported\n");
        MPI_Finalize();
        return 1;
    }
    // Ensure all ranks use identical arguments even if launch environments differ.
    unsigned long long n_wire = static_cast<unsigned long long>(n);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&n_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_wire); validate = flags[0]; printResults = flags[1];

    const size_t base = n / static_cast<size_t>(processes), extra = n % static_cast<size_t>(processes);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<double> A(localRows * n);
    std::vector<double> full;
    std::vector<double> original;
    std::vector<int> counts(processes), displacements(processes);
    for (int p = 0; p < processes; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        counts[p] = static_cast<int>(rows * n);
        displacements[p] = static_cast<int>((base * static_cast<size_t>(p) + std::min(static_cast<size_t>(p), extra)) * n);
    }
    if (rank == 0) {
        full.resize(n * n);
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(full, n);
        if (validate) original = full;
    }
    MPI_Scatterv(rank == 0 ? full.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE,
                 A.data(), static_cast<int>(localRows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    std::vector<double> pivot(n);
    int success = 1;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const int owner = rowOwner(k, n, processes);
        const size_t owner_local = localRow(k, n, processes);
        if (rank == owner) {
            double* row = A.data() + owner_local * n;
            const double diagonal = row[k];
            if (!(diagonal > 0.0)) success = 0;
            else row[k] = sqrt(diagonal);
            pivot[k] = success ? row[k] : 0.0;
            for (size_t j = k + 1; j < n; ++j) pivot[j] = success ? row[j] / pivot[k] : 0.0;
        }
        MPI_Bcast(pivot.data() + k, static_cast<int>(n - k), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        success = pivot[k] > 0.0 ? 1 : 0;
        if (!success) break;
        const double invDiag = 1.0 / pivot[k];
        for (size_t li = 0; li < localRows; ++li) {
            const size_t i = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra) + li;
            if (i <= k) continue;
            double* row = A.data() + li * n;
            const double lik = (row[k] *= invDiag);
            for (size_t j = k + 1; j < n; ++j)
                row[j] -= lik * pivot[j];
        }
    }
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    // The trailing matrix is overwritten in place; clear upper triangle for output compatibility.
    for (size_t li = 0; li < localRows; ++li) {
        const size_t i = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra) + li;
        for (size_t j = i + 1; j < n; ++j) A[li * n + j] = 0.0;
    }
    if (rank == 0) full.resize(n * n);
    MPI_Gatherv(A.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
                rank == 0 ? full.data() : nullptr, counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const double gflops = (static_cast<double>(n) * n * n / 3.0) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(full, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(full, original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
