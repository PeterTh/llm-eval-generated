#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
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

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        int rank_ = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank_); \
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_, __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Preserve the original random-number sequence and CSR construction.
void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* row, index_t n, index_t dim) {
    index_t assigned = 0;
    const uint64_t total = static_cast<uint64_t>(dim) * dim;
    const double prob = total ? static_cast<double>(n) / static_cast<double>(total) : 0.0;
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        row[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = total - (static_cast<uint64_t>(i) * dim + j);
            if (left <= static_cast<uint64_t>(n - assigned)) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= prob) || fillRemaining) cols[assigned++] = j;
        }
    }
    row[dim] = n;
}

// One thread per row avoids warp reduction overhead for very short rows.
__global__ void spmvScalar(const double* values, const index_t* cols, const index_t* row,
                           const double* vec, index_t rows, double* out) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = 0.0;
        for (index_t j = row[i]; j < row[i + 1]; ++j) sum += values[j] * vec[cols[j]];
        out[i] = sum;
    }
}

// A warp cooperatively reads each row, coalescing the matrix accesses.
__global__ void spmvWarp(const double* values, const index_t* cols, const index_t* row,
                         const double* vec, index_t rows, double* out) {
    const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t i = thread / 32;
    const unsigned lane = thread % 32;
    if (i < rows) {
        double sum = 0.0;
        const index_t end = row[i + 1];
        for (uint64_t j = static_cast<uint64_t>(row[i]) + lane; j < end; j += 32)
            sum += values[j] * vec[cols[j]];
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[i] = sum;
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// MPI counts are int, while CSR offsets can span the full uint32_t range.
template<class T>
void sendChunks(const T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        data += chunk; count -= chunk;
    }
}
template<class T>
void recvChunks(T* data, uint64_t count, MPI_Datatype type, int peer, int tag) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk; count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-s") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value > UINT32_MAX) bad = true;
            else if (option == 'n') numRows = static_cast<index_t>(value);
            else if (option == 's') sparsity = static_cast<index_t>(value);
            else iterations = static_cast<index_t>(value);
        } else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr; const char* arg = argv[++i]; maxVal = strtod(arg, &end);
            if (*arg == '\0' || *end != '\0' || !std::isfinite(maxVal)) bad = true;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) fprintf(stderr, "Unknown option: %s\n", argv[i]); bad = true; }
    }
    const uint64_t total = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t count64 = sparsity ? total / sparsity : 0;
    if (!sparsity || count64 > UINT32_MAX || numRows > INT_MAX) bad = true;
    if (help || bad) {
        if (rank == 0) {
            if (bad) fprintf(stderr, "Invalid arguments or matrix exceeds supported CSR/MPI limits\n");
            printUsage(argv[0]);
        }
        MPI_Finalize(); return bad ? 1 : 0;
    }
    const index_t nItems = static_cast<index_t>(count64);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, total ? 100.0 * (1.0 - static_cast<double>(nItems) / total) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n"); fflush(stdout);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * r / ranks;
        const uint64_t end = static_cast<uint64_t>(numRows) * (r + 1) / ranks;
        counts[r] = static_cast<int>(end - begin); displs[r] = static_cast<int>(begin);
    }
    const index_t localRows = static_cast<index_t>(counts[rank]);
    std::vector<double> vec(numRows), allValues;
    std::vector<index_t> allCols, allRow;
    if (rank == 0) {
        allValues.resize(nItems); allCols.resize(nItems); allRow.resize(static_cast<size_t>(numRows) + 1);
        srand(1); fill(vec.data(), numRows, maxVal); fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRow.data(), nItems, numRows);
    }
    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<index_t> row(static_cast<size_t>(localRows) + 1);
    // The row boundaries are sent separately because adjacent rank slices overlap by one entry.
    if (rank == 0) {
        for (int r = 1; r < ranks; ++r)
            sendChunks(allRow.data() + displs[r], static_cast<uint64_t>(counts[r]) + 1, MPI_UINT32_T, r, 0);
        std::copy_n(allRow.data(), row.size(), row.data());
    } else recvChunks(row.data(), row.size(), MPI_UINT32_T, 0, 0);
    const index_t base = row[0], localNnz = row[localRows] - base;
    std::vector<double> values(localNnz);
    std::vector<index_t> cols(localNnz);
    if (rank == 0) {
        for (int r = 1; r < ranks; ++r) {
            const index_t begin = allRow[displs[r]], end = allRow[displs[r] + counts[r]];
            if (end != begin) {
                sendChunks(allValues.data() + begin, static_cast<uint64_t>(end) - begin, MPI_DOUBLE, r, 1);
                sendChunks(allCols.data() + begin, static_cast<uint64_t>(end) - begin, MPI_UINT32_T, r, 2);
            }
        }
        if (localNnz) {
            std::copy_n(allValues.data() + base, localNnz, values.data());
            std::copy_n(allCols.data() + base, localNnz, cols.data());
        }
        std::vector<double>().swap(allValues); std::vector<index_t>().swap(allCols); std::vector<index_t>().swap(allRow);
    } else {
        recvChunks(values.data(), localNnz, MPI_DOUBLE, 0, 1);
        recvChunks(cols.data(), localNnz, MPI_UINT32_T, 0, 2);
    }
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRows); ++i) row[i] -= base;

    double *dValues = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRow = nullptr;
    CUDA_CHECK(cudaMalloc(&dValues, std::max<size_t>(1, values.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCols, std::max<size_t>(1, cols.size()) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dRow, row.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&dVec, std::max<size_t>(1, vec.size()) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dOut, std::max<size_t>(1, localRows) * sizeof(double)));
    if (localNnz) {
        CUDA_CHECK(cudaMemcpy(dValues, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols, cols.data(), cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRow, row.data(), row.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    if (numRows) CUDA_CHECK(cudaMemcpy(dVec, vec.data(), vec.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (localRows) CUDA_CHECK(cudaMemset(dOut, 0, localRows * sizeof(double)));
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(localRows); ++i) {
            double sum = 0.0;
            for (index_t j = row[i]; j < row[i + 1]; ++j) sum += values[j] * vec[cols[j]];
            reference[i] = sum;
        }
    }
    constexpr unsigned block = 256;
    const bool useWarp = localRows && static_cast<uint64_t>(localNnz) >= static_cast<uint64_t>(localRows) * 8;
    const unsigned blocks = useWarp ? static_cast<unsigned>((static_cast<uint64_t>(localRows) + 7) / 8)
                                    : static_cast<unsigned>((static_cast<uint64_t>(localRows) + block - 1) / block);
    auto launch = [&]() {
        if (localRows) {
            if (useWarp) spmvWarp<<<blocks, block>>>(dValues, dCols, dRow, dVec, localRows, dOut);
            else spmvScalar<<<blocks, block>>>(dValues, dCols, dRow, dVec, localRows, dOut);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    // Warm up outside the measured region, while preserving the zero-iteration result.
    if (iterations) { launch(); CUDA_CHECK(cudaDeviceSynchronize()); }
    if (rank == 0) { printf("Computing SpMV...\n"); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> out(localRows);
    if (localRows) CUDA_CHECK(cudaMemcpy(out.data(), dOut, localRows * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? 2.0 * nItems * iterations / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        std::vector<double> result(rank == 0 ? numRows : 0);
        MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE, result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(result, "OutputVector");
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        #pragma omp parallel for reduction(&:valid) schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(localRows); ++i) {
            const double ref = reference[i], res = out[i];
            const bool ok = std::abs(ref) < 1e-10 ? std::abs(res) <= MAX_RELATIVE_ERROR
                                                     : std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
            if (!ok) {
                #pragma omp critical
                fprintf(stderr, "Validation failed at index %u: reference %.10e, got %.10e\n", static_cast<unsigned>(displs[rank] + i), ref, res);
                valid = 0;
            }
        }
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        valid = allValid;
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(dValues)); CUDA_CHECK(cudaFree(dCols)); CUDA_CHECK(cudaFree(dRow));
    CUDA_CHECK(cudaFree(dVec)); CUDA_CHECK(cudaFree(dOut));
    MPI_Finalize();
    return valid ? 0 : 1;
}
