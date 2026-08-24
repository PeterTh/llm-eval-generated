#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

static void rows_for_rank(size_t n, int rank, int size, int& first, int& count) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t extra = n % static_cast<size_t>(size);
    first = static_cast<int>(rank * base + std::min<size_t>(rank, extra));
    count = static_cast<int>(base + (static_cast<size_t>(rank) < extra));
}

static int owner_of(size_t row, size_t n, int size) {
    for (int r = 0; r < size; ++r) {
        int first, count;
        rows_for_rank(n, r, size, first, count);
        if (row >= static_cast<size_t>(first) && row < static_cast<size_t>(first + count)) return r;
    }
    return size - 1;
}

// Block-row distributed right-looking Cholesky. The pivot row is broadcast,
// while each rank performs all updates for its locally owned rows.
static bool choleskyDecomposition(std::vector<double>& local, size_t n,
                                  int first, int local_rows, int rank, int size) {
    std::vector<double> pivot(n);
    for (size_t j = 0; j < n; ++j) {
        const int owner = owner_of(j, n, size);
        if (rank == owner) {
            const size_t li = j - static_cast<size_t>(first);
            std::copy_n(local.data() + li * n, n, pivot.data());
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += pivot[k] * pivot[k];
            const double value = pivot[j] - sum;
            if (value <= 0.0) {
                if (rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            pivot[j] = std::sqrt(value);
            local[li * n + j] = pivot[j];
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        for (int li = 0; li < local_rows; ++li) {
            const size_t i = static_cast<size_t>(first + li);
            if (i <= j) continue;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += local[static_cast<size_t>(li) * n + k] * pivot[k];
            local[static_cast<size_t>(li) * n + j] =
                (local[static_cast<size_t>(li) * n + j] - sum) / pivot[j];
        }
    }
    for (int li = 0; li < local_rows; ++li)
        for (size_t j = static_cast<size_t>(first + li) + 1; j < n; ++j)
            local[static_cast<size_t>(li) * n + j] = 0.0;
    return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B) x = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A, size_t n) {
    double max_error = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += L[i*n+k] * L[j*n+k];
        const double error = std::fabs(sum - A[i*n+j]);
        max_error = std::max(max_error, error);
        max_rel = std::max(max_rel, error / (std::fabs(A[i*n+j]) + 1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, max_rel);
    if (max_rel > 1e-6) std::printf("Validation failed: relative error too large\n");
    return max_rel <= 1e-6;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    int first, local_rows;
    rows_for_rank(n, rank, size, first, local_rows);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) { int f, c; rows_for_rank(n, r, size, f, c); counts[r] = c * static_cast<int>(n); displs[r] = f * static_cast<int>(n); }
    std::vector<double> global, original;
    if (rank == 0) { global.resize(n*n); std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled"); generatePositiveDefiniteMatrix(global, n); if (validate) original = global; }
    std::vector<double> local(static_cast<size_t>(local_rows) * n);
    MPI_Scatterv(rank == 0 ? global.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
    const double start = MPI_Wtime();
    const bool success = choleskyDecomposition(local, n, first, local_rows, rank, size);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed;
    MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int ok = success ? 1 : 0; MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!ok) { if (rank == 0) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    std::vector<double> result;
    if (rank == 0) result.resize(n*n);
    MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE, rank == 0 ? result.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) { const double time = max_elapsed; std::printf("Computation time: %.0f ms\nPerformance: %.3f GFLOPS\n", time*1000.0, (n*n*n/3.0)/time/1e9); if (printResults) print_results(result, "CholeskyL"); if (validate) { std::printf("Validating result...\nValidation: %s\n", validateCholesky(result, original, n) ? "PASSED" : "FAILED"); } }
    MPI_Finalize(); return 0;
}
