#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                             \
    }                                                                            \
} while (0)

static void fill(double* a, index_t n, double max_val) {
    for (index_t i = 0; i < n; ++i)
        a[i] = max_val * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

// Kept deliberately identical to the original generator so a one-rank run has
// exactly the same matrix and multi-rank runs have the same global problem.
static void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * dim);
    srand(8675309);
    bool fill_remaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
            const index_t needed = n - assigned;
            if (left <= needed) fill_remaining = true;
            const double r = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && r <= prob) || fill_remaining)
                cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

static void spmvCpu(const double* val, const index_t* col, const index_t* row,
                    const double* x, index_t nrows, double* y) {
#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < static_cast<int64_t>(nrows); ++i) {
        double sum = 0.0;
        for (index_t j = row[i]; j < row[i + 1]; ++j) sum += val[j] * x[col[j]];
        y[i] = sum;
    }
}

// A warp owns a row.  This is substantially better than one thread per row for
// the benchmark's long rows and remains efficient for short, irregular rows.
__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ col,
                               const index_t* __restrict__ row,
                               const double* __restrict__ x,
                               index_t nrows, double* __restrict__ y) {
    const unsigned lane = threadIdx.x & 31u;
    const index_t r = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (r >= nrows) return;
    double sum = 0.0;
    for (index_t j = row[r] + lane; j < row[r + 1]; j += 32)
        sum += val[j] * __ldg(x + col[j]);
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[r] = sum;
}

static bool verifyResults(const double* ref, const double* got, index_t n) {
    for (index_t i = 0; i < n; ++i) {
        const double err = std::abs(got[i] - ref[i]);
        if ((std::abs(ref[i]) < 1e-10 && err > MAX_RELATIVE_ERROR) ||
            (std::abs(ref[i]) >= 1e-10 && err / std::abs(ref[i]) > MAX_RELATIVE_ERROR)) {
            std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                        i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num> Matrix dimension (default: 1024)\n"
                "  -s <num> Sparsity denominator (default: 10)\n"
                "  -i <num> Iterations (default: 10)\n"
                "  -m <val> Maximum element value (default: 1.0)\n"
                "  -v Validate\n  -r Print results\n  -h Show help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    index_t nrows = 1024, sparsity = 10, iterations = 10;
    double max_val = 1.0;
    bool validate = false, print_results_flag = false, help = false;
    int parse_ok = 1;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-n") && i + 1 < argc) nrows = std::strtoul(argv[++i], nullptr, 10);
            else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) sparsity = std::strtoul(argv[++i], nullptr, 10);
            else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::strtoul(argv[++i], nullptr, 10);
            else if (!std::strcmp(argv[i], "-m") && i + 1 < argc) max_val = std::strtod(argv[++i], nullptr);
            else if (!std::strcmp(argv[i], "-v")) validate = true;
            else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
            else if (!std::strcmp(argv[i], "-h")) help = true;
            else { std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]); parse_ok = 0; }
        }
        if (!nrows || !sparsity || !iterations || uint64_t(nrows) * nrows / sparsity > std::numeric_limits<index_t>::max()) {
            std::fprintf(stderr, "Invalid problem size, sparsity, or iteration count\n"); parse_ok = 0;
        }
        if (help || !parse_ok) usage(argv[0]);
    }
    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&help, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!parse_ok || help) { MPI_Finalize(); return parse_ok ? 0 : 1; }
    MPI_Bcast(&nrows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_results_flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    const index_t nnz = static_cast<index_t>(uint64_t(nrows) * nrows / sparsity);
    std::vector<double> val, x(nrows), output, reference;
    std::vector<index_t> col, row;
    if (rank == 0) {
        val.resize(nnz); col.resize(nnz); row.resize(nrows + 1); output.resize(nrows);
        fill(x.data(), nrows, max_val); fill(val.data(), nnz, max_val);
        initRandomMatrix(col.data(), row.data(), nnz, nrows);
        if (validate) { reference.resize(nrows); spmvCpu(val.data(), col.data(), row.data(), x.data(), nrows, reference.data()); }
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n"
                    "Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n"
                    "Non-zero elements: %u (%.2f%% sparse)\nIterations: %u\nMax value: %.2f\n"
                    "Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\nInitializing data structures...\n",
                    nrows, nrows, sparsity, nnz,
                    100.0 * (1.0 - double(nnz) / (double(nrows) * nrows)), iterations,
                    max_val, validate ? "enabled" : "disabled", nranks, omp_get_max_threads());
    }
    MPI_Bcast(x.data(), static_cast<int>(nrows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Contiguous row ownership makes collection cheap; CSR payloads are scattered
    // without replication.  MPI counts impose the same practical <2^31 limit.
    std::vector<int> row_counts(nranks), row_displs(nranks), nz_counts(nranks), nz_displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const uint64_t begin = uint64_t(nrows) * r / nranks;
        const uint64_t end = uint64_t(nrows) * (r + 1) / nranks;
        row_displs[r] = static_cast<int>(begin); row_counts[r] = static_cast<int>(end - begin);
        if (rank == 0) {
            nz_displs[r] = static_cast<int>(row[begin]);
            nz_counts[r] = static_cast<int>(row[end] - row[begin]);
        }
    }
    MPI_Bcast(nz_counts.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nz_displs.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);
    const int local_rows = row_counts[rank], local_nnz = nz_counts[rank];
    std::vector<double> lval(local_nnz), lout(local_rows);
    std::vector<index_t> lcol(local_nnz), lrow(local_rows + 1);
    MPI_Scatterv(rank == 0 ? val.data() : nullptr, nz_counts.data(), nz_displs.data(), MPI_DOUBLE,
                 lval.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? col.data() : nullptr, nz_counts.data(), nz_displs.data(), MPI_UINT32_T,
                 lcol.data(), local_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<int> delim_counts(nranks), delim_displs(nranks);
    for (int r = 0; r < nranks; ++r) { delim_counts[r] = row_counts[r] + 1; delim_displs[r] = row_displs[r]; }
    MPI_Scatterv(rank == 0 ? row.data() : nullptr, delim_counts.data(), delim_displs.data(), MPI_UINT32_T,
                 lrow.data(), local_rows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t base = lrow.empty() ? 0 : lrow[0];
#pragma omp parallel for
    for (int i = 0; i <= local_rows; ++i) lrow[i] -= base;

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    double *dval = nullptr, *dx = nullptr, *dy = nullptr;
    index_t *dcol = nullptr, *drow = nullptr;
    CUDA_CHECK(cudaMalloc(&dval, std::max<size_t>(1, local_nnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dcol, std::max<size_t>(1, local_nnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&drow, (size_t(local_rows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dx, size_t(nrows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dy, std::max<size_t>(1, local_rows) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dval, lval.data(), size_t(local_nnz) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcol, lcol.data(), size_t(local_nnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(drow, lrow.data(), (size_t(local_rows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), size_t(nrows) * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int block = 256;
    const int grid = (local_rows * 32 + block - 1) / block;
    if (local_rows) { spmvWarpKernel<<<grid, block>>>(dval, dcol, drow, dx, local_rows, dy); CUDA_CHECK(cudaDeviceSynchronize()); }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing SpMV...\n");
    const double start = MPI_Wtime();
    for (index_t it = 0; it < iterations; ++it)
        if (local_rows) spmvWarpKernel<<<grid, block>>>(dval, dcol, drow, dx, local_rows, dy);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double global_elapsed = 0.0;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (local_rows) CUDA_CHECK(cudaMemcpy(lout.data(), dy, size_t(local_rows) * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Gatherv(lout.data(), local_rows, MPI_DOUBLE, rank == 0 ? output.data() : nullptr,
                row_counts.data(), row_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        const double ms = global_elapsed * 1000.0;
        std::printf("Computation time: %.3f ms\nAverage time per iteration: %.3f ms\nPerformance: %.3f GFLOPS\n",
                    ms, ms / iterations, 2.0 * nnz * iterations / global_elapsed / 1e9);
        if (print_results_flag) print_results(output, "OutputVector");
        if (validate) {
            std::printf("Validating result...\n");
            status = verifyResults(reference.data(), output.data(), nrows) ? 0 : 1;
            std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(dval); cudaFree(dcol); cudaFree(drow); cudaFree(dx); cudaFree(dy);
    MPI_Finalize();
    return status;
}
