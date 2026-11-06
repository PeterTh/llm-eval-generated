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
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Preserve the original random number stream and its order.
void fill(double* a, index_t n, double maxVal) {
    for (index_t i = 0; i < n; ++i)
        a[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t n, index_t dim) {
    index_t assigned = 0;
    const uint64_t total = static_cast<uint64_t>(dim) * dim;
    const double prob = total ? static_cast<double>(n) / static_cast<double>(total) : 0.0;
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rows[i] = assigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t left = total - (static_cast<uint64_t>(i) * dim + j);
            if (left <= static_cast<uint64_t>(n - assigned)) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < n && random <= prob) || fillRemaining) cols[assigned++] = j;
        }
    }
    rows[dim] = n;
}

// Cooperating lanes read consecutive CSR entries. A grid-stride loop also
// supports matrices with more rows than can be covered by one CUDA grid.
template<int LANES>
__global__ void spmvKernel(const double* __restrict__ values, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows, const double* __restrict__ vec,
                           index_t nrows, double* __restrict__ out) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & (LANES - 1);
    const unsigned group = tid / LANES;
    const unsigned stride = gridDim.x * blockDim.x / LANES;
    for (uint64_t row = group; row < nrows; row += stride) {
        double sum = 0.0;
        const index_t end = rows[row + 1];
        for (uint64_t j = static_cast<uint64_t>(rows[row]) + lane; j < end; j += LANES)
            sum += values[j] * vec[cols[j]];
        for (int offset = LANES / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(__activemask(), sum, offset, LANES);
        if (lane == 0) out[row] = sum;
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

// MPI's count parameter is int, while CSR arrays may contain more elements.
void transfer(void* data, uint64_t count, MPI_Datatype type, size_t elementSize,
              int peer, int tag, bool send) {
    auto* ptr = static_cast<char*>(data);
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        if (send) MPI_Send(ptr, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(ptr, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ptr += static_cast<size_t>(chunk) * elementSize;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    int provided = 0, rank = 0, ranks = 1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-s") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || !*arg || *end || value > UINT32_MAX) bad = true;
            else if (option == 'n') numRows = static_cast<index_t>(value);
            else if (option == 's') sparsity = static_cast<index_t>(value);
            else iterations = static_cast<index_t>(value);
        } else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            maxVal = strtod(arg, &end);
            if (!*arg || *end) bad = true;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) fprintf(stderr, "Unknown option: %s\n", argv[i]); bad = true; }
    }
    const uint64_t total = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t count = sparsity ? total / sparsity : 0;
    if (!sparsity || numRows > INT_MAX || count > UINT32_MAX) bad = true;
    if (help || bad) {
        if (rank == 0) {
            if (bad) fprintf(stderr, "Invalid arguments or matrix exceeds CSR/MPI limits\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    const index_t nItems = static_cast<index_t>(count);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, deviceCount = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<double> vec(numRows), allValues;
    std::vector<index_t> allCols, allRows, boundaries(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               total ? 100.0 * (1.0 - static_cast<double>(nItems) / total) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        allValues.resize(nItems); allCols.resize(nItems); allRows.resize(static_cast<size_t>(numRows) + 1);
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        boundaries[0] = 0;
        for (int p = 1; p < ranks; ++p) {
            if (nItems) {
                const index_t target = static_cast<index_t>(static_cast<uint64_t>(nItems) * p / ranks);
                boundaries[p] = static_cast<index_t>(std::lower_bound(allRows.begin(), allRows.end(), target) - allRows.begin());
            } else boundaries[p] = static_cast<index_t>(static_cast<uint64_t>(numRows) * p / ranks);
        }
        boundaries[ranks] = numRows;
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    if (numRows) MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    index_t localNnz = 0;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t begin = boundaries[p], end = boundaries[p + 1];
            index_t nnz = allRows[end] - allRows[begin];
            MPI_Send(&nnz, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            transfer(allRows.data() + begin, static_cast<uint64_t>(end - begin) + 1, MPI_UINT32_T, sizeof(index_t), p, 1, true);
            if (nnz) {
                transfer(allCols.data() + allRows[begin], nnz, MPI_UINT32_T, sizeof(index_t), p, 2, true);
                transfer(allValues.data() + allRows[begin], nnz, MPI_DOUBLE, sizeof(double), p, 3, true);
            }
        }
        localNnz = allRows[boundaries[1]];
    } else MPI_Recv(&localNnz, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<index_t> rows(static_cast<size_t>(localRows) + 1), cols(localNnz);
    std::vector<double> values(localNnz), out(localRows, 0.0);
    if (rank == 0) {
        std::copy_n(allRows.data(), static_cast<size_t>(localRows) + 1, rows.data());
        std::copy_n(allCols.data(), localNnz, cols.data());
        std::copy_n(allValues.data(), localNnz, values.data());
        std::vector<index_t>().swap(allRows); std::vector<index_t>().swap(allCols); std::vector<double>().swap(allValues);
    } else {
        transfer(rows.data(), static_cast<uint64_t>(localRows) + 1, MPI_UINT32_T, sizeof(index_t), 0, 1, false);
        if (localNnz) {
            transfer(cols.data(), localNnz, MPI_UINT32_T, sizeof(index_t), 0, 2, false);
            transfer(values.data(), localNnz, MPI_DOUBLE, sizeof(double), 0, 3, false);
        }
    }
    const index_t base = rows[0];
    #pragma omp parallel for schedule(static) if(localRows > 4096)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= base;

    double *dValues = nullptr, *dVec = nullptr, *dOut = nullptr;
    index_t *dCols = nullptr, *dRows = nullptr;
    CUDA_CHECK(cudaMalloc(&dRows, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMemcpy(dRows, rows.data(), (static_cast<size_t>(localRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    if (localNnz) {
        CUDA_CHECK(cudaMalloc(&dValues, static_cast<size_t>(localNnz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dCols, static_cast<size_t>(localNnz) * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(dValues, values.data(), static_cast<size_t>(localNnz) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols, cols.data(), static_cast<size_t>(localNnz) * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    if (numRows) {
        CUDA_CHECK(cudaMalloc(&dVec, static_cast<size_t>(numRows) * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dVec, vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (localRows) {
        CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(localRows) * sizeof(double)));
        CUDA_CHECK(cudaMemset(dOut, 0, static_cast<size_t>(localRows) * sizeof(double)));
    }
    if (rank == 0 && validate) printf("Computing reference solution...\n");
    std::vector<double> reference(validate ? localRows : 0);
    if (validate) {
        #pragma omp parallel for schedule(dynamic, 64) if(localRows > 256)
        for (uint64_t i = 0; i < localRows; ++i) {
            double sum = 0.0;
            for (uint64_t j = rows[i]; j < rows[i + 1]; ++j) sum += values[j] * vec[cols[j]];
            reference[i] = sum;
        }
    }
    if (rank == 0) printf("Computing SpMV...\n");
    const double average = localRows ? static_cast<double>(localNnz) / localRows : 0;
    int lanes = average <= 2 ? 1 : average <= 4 ? 2 : average <= 8 ? 4 : average <= 16 ? 8 : average <= 32 ? 16 : 32;
    constexpr int threads = 256;
    const int blocks = localRows ? static_cast<int>(std::min<uint64_t>((static_cast<uint64_t>(localRows) * lanes + threads - 1) / threads, 65535)) : 0;
    auto launch = [&]() {
        switch (lanes) {
            case 1: spmvKernel<1><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 2: spmvKernel<2><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 4: spmvKernel<4><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 8: spmvKernel<8><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 16: spmvKernel<16><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            default: spmvKernel<32><<<blocks, threads>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
        }
    };
    // Exclude CUDA context/JIT and first-launch costs from the measured iterations.
    if (localRows && iterations) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t i = 0; i < iterations && localRows; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows) CUDA_CHECK(cudaMemcpy(out.data(), dOut, static_cast<size_t>(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? maxElapsed * 1000 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", maxElapsed > 0 ? (2.0 * nItems * iterations) / maxElapsed / 1e9 : 0.0);
    }
    if (printResults) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = static_cast<int>(boundaries[p + 1] - boundaries[p]);
            offsets[p] = static_cast<int>(boundaries[p]);
        }
        std::vector<double> result(rank == 0 ? numRows : 0);
        MPI_Gatherv(out.data(), static_cast<int>(localRows), MPI_DOUBLE, result.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(result, "OutputVector");
    }
    int valid = 1;
    if (validate) {
        unsigned long long first = localRows;
        #pragma omp parallel for reduction(min:first) schedule(static) if(localRows > 256)
        for (uint64_t i = 0; i < localRows; ++i) {
            const double ref = reference[i], res = out[i];
            if ((std::abs(ref) < 1e-10 && std::abs(res) > MAX_RELATIVE_ERROR) ||
                (std::abs(ref) >= 1e-10 && std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR))
                first = std::min(first, static_cast<unsigned long long>(i));
        }
        if (first < localRows) {
            fprintf(stderr, "Validation failed at index %u: reference %.10e, got %.10e\n", boundaries[rank] + static_cast<index_t>(first), reference[first], out[first]);
            valid = 0;
        }
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        valid = allValid;
        if (rank == 0) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    cudaFree(dValues); cudaFree(dCols); cudaFree(dRows); cudaFree(dVec); cudaFree(dOut);
    MPI_Finalize();
    return valid ? 0 : 1;
}
