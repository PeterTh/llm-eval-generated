#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <exception>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (!std::isfinite(ref) || !std::isfinite(res)) {
            printf("Validation failed at index %u: non-finite result\n", i);
            return false;
        }
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}


// MPI calls are confined to the main thread; OpenMP workers own CUDA devices.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template <int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           index_t count, double* __restrict__ output) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        // Widen the loop index to prevent overflow near the CSR index limit.
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        const unsigned mask = __activemask();
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(mask, sum, offset, Width);
        if (lane == 0) output[row] = sum;
    }
}

// Contiguous partitions preserve output order and balance actual CSR work.
static std::vector<index_t> partition(const index_t* rows, index_t count, int parts) {
    std::vector<index_t> boundaries(parts + 1, 0);
    const uint64_t total = rows[count] - rows[0];
    for (int p = 1; p < parts; ++p) {
        if (total == 0) boundaries[p] = uint64_t(count) * p / parts;
        else {
            const index_t target = rows[0] + total * p / parts;
            boundaries[p] = std::lower_bound(rows, rows + count, target) - rows;
        }
    }
    boundaries[parts] = count;
    return boundaries;
}

// Use native collectives normally, and bounded messages for CSR arrays exceeding
// MPI's int count/displacement limit. No CUDA-aware MPI extension is needed.
template <typename T>
static void scatter(const std::vector<T>& source, std::vector<T>& local,
                    const std::vector<uint64_t>& counts,
                    const std::vector<uint64_t>& offsets, MPI_Datatype type,
                    int rank, int ranks) {
    bool small = true;
    for (int p = 0; p < ranks; ++p)
        small = small && counts[p] <= INT_MAX && offsets[p] <= INT_MAX;
    if (small) {
        std::vector<int> c(counts.begin(), counts.end()), d(offsets.begin(), offsets.end());
        MPI_Scatterv(source.data(), c.data(), d.data(), type, local.data(),
                     static_cast<int>(local.size()), type, 0, MPI_COMM_WORLD);
    } else {
        constexpr uint64_t chunk = 1u << 26;
        if (rank == 0) {
            std::copy_n(source.data() + offsets[0], local.size(), local.data());
            for (int p = 1; p < ranks; ++p)
                for (uint64_t j = 0; j < counts[p]; j += chunk)
                    MPI_Send(source.data() + offsets[p] + j,
                             static_cast<int>(std::min(chunk, counts[p] - j)), type,
                             p, 0, MPI_COMM_WORLD);
        } else {
            for (uint64_t j = 0; j < local.size(); j += chunk)
                MPI_Recv(local.data() + j,
                         static_cast<int>(std::min(chunk, local.size() - j)), type,
                         0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

struct DeviceWork {
    int device = 0, width = 32, blocks = 1;
    index_t begin = 0, count = 0;
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
};

static void launch(DeviceWork& w) {
    if (!w.count) return;
    if (w.width == 4)
        spmvKernel<4><<<w.blocks, 256, 0, w.stream>>>(w.values, w.columns, w.rows, w.vector, w.count, w.output);
    else if (w.width == 8)
        spmvKernel<8><<<w.blocks, 256, 0, w.stream>>>(w.values, w.columns, w.rows, w.vector, w.count, w.output);
    else if (w.width == 16)
        spmvKernel<16><<<w.blocks, 256, 0, w.stream>>>(w.values, w.columns, w.rows, w.vector, w.count, w.output);
    else
        spmvKernel<32><<<w.blocks, 256, 0, w.stream>>>(w.values, w.columns, w.rows, w.vector, w.count, w.output);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void upload(T*& destination, const T* source, size_t count, cudaStream_t stream) {
    // CUDA rejects zero-byte allocations; empty matrices still run on CUDA.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&destination), std::max(size_t(1), count) * sizeof(T)));
    if (count) CUDA_CHECK(cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

static int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0 ||
             strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* argument = argv[++i];
            const unsigned long long value = strtoull(argument, &end, 10);
            if (*argument == '-' || !*argument || *end || value > UINT32_MAX) {
                if (!rank) fprintf(stderr, "Invalid integer: %s\n", argument);
                return 1;
            }
            if (option == 'n') numRows = value;
            else if (option == 's') sparsity = value;
            else iterations = value;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    const uint64_t cells = uint64_t(numRows) * numRows;
    if (!sparsity || numRows > INT_MAX || !std::isfinite(maxVal) ||
        (sparsity && cells / sparsity > UINT32_MAX)) {
        if (!rank) fprintf(stderr, "Invalid dimensions, sparsity, or maximum value (CSR uses 32-bit indices).\n");
        return 1;
    }
    const index_t nItems = cells / sparsity;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: CUDA device required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<DeviceWork> work;
    // Respect CUDA_VISIBLE_DEVICES, including launchers exposing one GPU per rank.
    for (int d = localRank; d < devices; d += localRanks) {
        DeviceWork w; w.device = d; work.push_back(w);
    }
    if (work.empty()) { DeviceWork w; w.device = localRank % devices; work.push_back(w); }
    MPI_Comm_free(&node);
    const int workers = static_cast<int>(work.size());
    int allDevices = 0;
    MPI_Reduce(&workers, &allDevices, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    std::vector<double> values, vector(numRows), output, reference;
    std::vector<index_t> columns, rows, boundaries(ranks + 1);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               cells ? 100.0 * (1.0 - double(nItems) / cells) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; CUDA workers: %d; OpenMP host threads: %d\n", ranks, allDevices, omp_get_max_threads());
        printf("Initializing data structures...\n");
        values.resize(nItems); columns.resize(nItems); rows.resize(size_t(numRows) + 1);
        // Match the original default C RNG state, independent of MPI startup.
        srand(1);
        fill(vector.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rows.data(), nItems, numRows);
        boundaries = partition(rows.data(), numRows, ranks);
    }
    MPI_Bcast(vector.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<uint64_t> nnzCounts(ranks), nnzOffsets(ranks), rowCounts(ranks), rowOffsets(ranks);
    for (int p = 0; p < ranks; ++p) {
        rowCounts[p] = uint64_t(boundaries[p + 1]) - boundaries[p] + 1;
        rowOffsets[p] = boundaries[p];
        if (!rank) {
            nnzOffsets[p] = rows[boundaries[p]];
            nnzCounts[p] = rows[boundaries[p + 1]] - rows[boundaries[p]];
        }
    }
    MPI_Bcast(nnzCounts.data(), ranks, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzOffsets.data(), ranks, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<double> localValues(nnzCounts[rank]), localOutput(localRows, 0.0);
    std::vector<index_t> localColumns(nnzCounts[rank]), localDelimiters(size_t(localRows) + 1);
    scatter(values, localValues, nnzCounts, nnzOffsets, MPI_DOUBLE, rank, ranks);
    scatter(columns, localColumns, nnzCounts, nnzOffsets, MPI_UINT32_T, rank, ranks);
    scatter(rows, localDelimiters, rowCounts, rowOffsets, MPI_UINT32_T, rank, ranks);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i <= localRows; ++i) localDelimiters[i] -= nnzOffsets[rank];
    // Drop the root's global CSR after distribution; validate locally in parallel.
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(columns);
    std::vector<index_t>().swap(rows);
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(localValues.data(), localColumns.data(), localDelimiters.data(), vector.data(), localRows, reference.data());
    }
    const auto gpuBoundaries = partition(localDelimiters.data(), localRows, workers);
    const index_t batch = std::min(iterations, index_t(64));
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int g = 0; g < workers; ++g) {
        auto& w = work[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        w.begin = gpuBoundaries[g]; w.count = gpuBoundaries[g + 1] - w.begin;
        if (!w.count) continue;
        CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
        const index_t offset = localDelimiters[w.begin];
        const size_t nonzeros = localDelimiters[w.begin + w.count] - offset;
        std::vector<index_t> deviceRows(size_t(w.count) + 1);
        for (size_t i = 0; i < deviceRows.size(); ++i) deviceRows[i] = localDelimiters[w.begin + i] - offset;
        upload(w.values, nonzeros ? localValues.data() + offset : nullptr, nonzeros, w.stream);
        upload(w.columns, nonzeros ? localColumns.data() + offset : nullptr, nonzeros, w.stream);
        upload(w.rows, deviceRows.data(), deviceRows.size(), w.stream);
        upload(w.vector, vector.data(), vector.size(), w.stream);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&w.output), size_t(w.count) * sizeof(double)));
        const double average = double(nonzeros) / w.count;
        w.width = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
        cudaDeviceProp properties;
        CUDA_CHECK(cudaGetDeviceProperties(&properties, w.device));
        w.blocks = std::min(uint64_t(properties.multiProcessorCount) * 32,
                            (uint64_t(w.count) * w.width + 255) / 256);
        // Warm up before timing. Capture a bounded batch to amortize launch costs
        // without growing graph storage with the user-selected iteration count.
        launch(w);
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
        if (batch) {
            CUDA_CHECK(cudaStreamBeginCapture(w.stream, cudaStreamCaptureModeThreadLocal));
            for (index_t i = 0; i < batch; ++i) launch(w);
            CUDA_CHECK(cudaStreamEndCapture(w.stream, &w.graph));
            CUDA_CHECK(cudaGraphInstantiate(&w.executable, w.graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(w.executable, w.stream));
            CUDA_CHECK(cudaStreamSynchronize(w.stream));
        }
    }
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int g = 0; g < workers; ++g) {
        auto& w = work[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        if (!w.count || !iterations) continue;
        for (uint64_t i = 0; i < iterations / batch; ++i)
            CUDA_CHECK(cudaGraphLaunch(w.executable, w.stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch(w);
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int g = 0; g < workers; ++g) {
        auto& w = work[g];
        CUDA_CHECK(cudaSetDevice(w.device));
        if (!w.count) continue;
        if (iterations) CUDA_CHECK(cudaMemcpyAsync(localOutput.data() + w.begin, w.output,
                                    size_t(w.count) * sizeof(double), cudaMemcpyDeviceToHost, w.stream));
        CUDA_CHECK(cudaStreamSynchronize(w.stream));
        if (w.executable) CUDA_CHECK(cudaGraphExecDestroy(w.executable));
        if (w.graph) CUDA_CHECK(cudaGraphDestroy(w.graph));
        CUDA_CHECK(cudaFree(w.values)); CUDA_CHECK(cudaFree(w.columns)); CUDA_CHECK(cudaFree(w.rows));
        CUDA_CHECK(cudaFree(w.vector)); CUDA_CHECK(cudaFree(w.output));
        CUDA_CHECK(cudaStreamDestroy(w.stream));
    }
    if (!rank) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
    }
    if (printResults) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = boundaries[p + 1] - boundaries[p]; offsets[p] = boundaries[p];
        }
        if (!rank) output.resize(numRows);
        MPI_Gatherv(localOutput.data(), localRows, MPI_DOUBLE, output.data(), counts.data(),
                    offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(output, "OutputVector");
    }
    int valid = 1;
    if (validate) {
        const int localValid = verifyResults(reference.data(), localOutput.data(), localRows);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try { status = run(argc, argv, rank, ranks); }
    catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
