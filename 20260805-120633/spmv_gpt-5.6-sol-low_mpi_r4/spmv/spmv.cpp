#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* a, index_t n, double max_val) {
    for (index_t i = 0; i < n; ++i)
        a[i] = max_val * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double probability = static_cast<double>(n) /
                               (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fill_remaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t entries_left = static_cast<uint64_t>(dim) * dim -
                                          (static_cast<uint64_t>(i) * dim + j);
            const index_t needed = n - assigned;
            if (entries_left <= needed) fill_remaining = true;
            const double random_value = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random_value <= probability) || fill_remaining)
                cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rows,
             const double* vec, index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (index_t j = rows[i]; j < rows[i + 1]; ++j)
            sum += val[j] * vec[cols[j]];
        out[i] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res)
                                                   : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -n <num>  Matrix rows/columns (default: 1024)\n");
    printf("  -s <num>  One out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>  Iterations (default: 10)\n");
    printf("  -m <val>  Maximum element value (default: 1.0)\n");
    printf("  -v        Enable validation\n  -r        Print results\n  -h        Help\n");
}

static void mpiCheck(int error, MPI_Comm comm) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(comm, error);
    }
}

int main(int argc, char** argv) {
    mpiCheck(MPI_Init(&argc, &argv), MPI_COMM_WORLD);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t num_rows = 1024, sparsity = 10, iterations = 10;
    double max_val = 1.0;
    bool validate = false, print_results_requested = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) num_rows = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = static_cast<index_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) max_val = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!strcmp(argv[i], "-h")) parse_status = 2;
        else parse_status = 1;
    }
    if (parse_status || !num_rows || !sparsity || !iterations) {
        if (rank == 0) {
            if (parse_status == 1) printf("Invalid command line\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_status == 1 ? 1 : 0;
    }

    const uint64_t item_count64 = static_cast<uint64_t>(num_rows) * num_rows / sparsity;
    if (item_count64 > UINT32_MAX || item_count64 > INT_MAX || num_rows > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Problem is too large for this CSR/MPI implementation\n");
        MPI_Finalize(); return 1;
    }
    const index_t item_count = static_cast<index_t>(item_count64);
    std::vector<double> full_val, vec(num_rows), full_out, reference;
    std::vector<index_t> full_cols, full_rows;
    std::vector<int> row_counts(ranks), row_displs(ranks), nnz_counts(ranks), nnz_displs(ranks);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", num_rows, num_rows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\nValidation: %s\nMPI ranks: %d\n",
               item_count, 100.0 * (1.0 - static_cast<double>(item_count) /
               (static_cast<double>(num_rows) * num_rows)), iterations, max_val,
               validate ? "enabled" : "disabled", ranks);
        full_val.resize(item_count); full_cols.resize(item_count); full_rows.resize(num_rows + 1);
        printf("Initializing data structures...\n");
        fill(vec.data(), num_rows, max_val); fill(full_val.data(), item_count, max_val);
        initRandomMatrix(full_cols.data(), full_rows.data(), item_count, num_rows);

        // Choose contiguous boundaries close to equal nonzero counts.
        row_displs[0] = 0;
        for (int p = 1; p < ranks; ++p) {
            const index_t target = static_cast<index_t>((item_count64 * p) / ranks);
            row_displs[p] = static_cast<int>(std::lower_bound(full_rows.begin(), full_rows.end(), target) - full_rows.begin());
            row_displs[p] = std::min(row_displs[p], static_cast<int>(num_rows));
        }
        for (int p = 0; p < ranks; ++p) {
            const int end = p + 1 < ranks ? row_displs[p + 1] : static_cast<int>(num_rows);
            row_counts[p] = end - row_displs[p];
            nnz_displs[p] = static_cast<int>(full_rows[row_displs[p]]);
            nnz_counts[p] = static_cast<int>(full_rows[end] - full_rows[row_displs[p]]);
        }
        if (validate) {
            reference.resize(num_rows);
            spmvCpu(full_val.data(), full_cols.data(), full_rows.data(), vec.data(), num_rows, reference.data());
        }
    }

    MPI_Bcast(vec.data(), static_cast<int>(num_rows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_counts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_displs.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_counts.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_displs.data(), ranks, MPI_INT, 0, MPI_COMM_WORLD);
    const int local_rows_count = row_counts[rank], local_nnz = nnz_counts[rank];
    std::vector<double> local_val(local_nnz), local_out(local_rows_count);
    std::vector<index_t> local_cols(local_nnz), local_rows(local_rows_count + 1);
    MPI_Scatterv(rank == 0 ? full_val.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
                 local_val.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_cols.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_UINT32_T,
                 local_cols.data(), local_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // Row delimiters overlap by one entry, which MPI_Scatterv permits.
    std::vector<int> delimiter_counts(ranks);
    for (int p = 0; p < ranks; ++p) delimiter_counts[p] = row_counts[p] + 1;
    MPI_Scatterv(rank == 0 ? full_rows.data() : nullptr, delimiter_counts.data(), row_displs.data(), MPI_UINT32_T,
                 local_rows.data(), local_rows_count + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t offset = local_rows[0];
    for (index_t& x : local_rows) x -= offset;

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter)
        spmvCpu(local_val.data(), local_cols.data(), local_rows.data(), vec.data(),
                static_cast<index_t>(local_rows_count), local_out.data());
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || print_results_requested) {
        if (rank == 0) full_out.resize(num_rows);
        MPI_Gatherv(local_out.data(), local_rows_count, MPI_DOUBLE,
                    rank == 0 ? full_out.data() : nullptr, row_counts.data(), row_displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exit_code = 0;
    if (rank == 0) {
        const double milliseconds = max_elapsed * 1000.0;
        const double gflops = max_elapsed > 0 ? 2.0 * item_count * iterations / max_elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
               milliseconds, milliseconds / iterations, gflops);
        if (print_results_requested) print_results(full_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), full_out.data(), num_rows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
