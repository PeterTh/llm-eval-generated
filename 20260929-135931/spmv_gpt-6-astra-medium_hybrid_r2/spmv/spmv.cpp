#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
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
    #pragma omp parallel for schedule(dynamic, 64)
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
        if (!std::isfinite(res)) return false;
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


void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers so CSR arrays are not limited by MPI's signed int count.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool sending) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(1) << 26));
        if (sending) MPI_Send(data, chunk, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template<class T>
T* deviceCopy(const std::vector<T>& data) {
    T* ptr = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(size_t(1), data.size()) * sizeof(T)));
    if (!data.empty())
        cudaCheck(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

// Subwarps cooperate on a row; narrow groups avoid wasting lanes for sparse rows.
template<int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           double* __restrict__ output, index_t count) {
    const unsigned lane = threadIdx.x % Width;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        const unsigned mask = __activemask();
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(mask, sum, offset, Width);
        if (lane == 0) output[row] = sum;
    }
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    auto parseCount = [](const char* text) {
        char* end = nullptr;
        unsigned long long value = strtoull(text, &end, 10);
        if (*text == '-' || end == text || *end || value == 0 ||
            value > std::numeric_limits<index_t>::max())
            throw std::runtime_error("Expected a positive 32-bit integer");
        return static_cast<index_t>(value);
    };
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = parseCount(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = parseCount(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = parseCount(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }
    const uint64_t total = uint64_t(numRows) * numRows;
    if (total / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Nonzero count exceeds 32-bit CSR capacity");
    const index_t nItems = static_cast<index_t>(total / sparsity);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("Every MPI rank requires a CUDA device");
    cudaCheck(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, localRank % devices));

    std::vector<double> values, vector(numRows);
    std::vector<index_t> columns, rows, boundaries(size_t(ranks) + 1);
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / total));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n",
               iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
        values.resize(nItems);
        columns.resize(nItems);
        rows.resize(size_t(numRows) + 1);
        fill(vector.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rows.data(), nItems, numRows);
        // Include a small per-row cost to balance empty rows as well as nonzeros.
        const uint64_t work = uint64_t(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = work * p / ranks;
            index_t lo = boundaries[p - 1], hi = numRows;
            while (lo < hi) {
                index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(rows[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
        boundaries[ranks] = numRows;
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t offset = 0; offset < numRows;) {
        int count = static_cast<int>(std::min(size_t(numRows) - offset, size_t(1) << 26));
        MPI_Bcast(vector.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> localOffsets(size_t(localRows) + 1), localColumns;
    std::vector<double> localValues, output(localRows);
    if (!rank) {
        for (int p = 1; p < ranks; ++p) {
            index_t begin = rows[boundaries[p]], end = rows[boundaries[p + 1]];
            transfer(rows.data() + boundaries[p], size_t(boundaries[p + 1]) - boundaries[p] + 1,
                     MPI_UINT32_T, p, true);
            if (end > begin) {
                transfer(columns.data() + begin, end - begin, MPI_UINT32_T, p, true);
                transfer(values.data() + begin, end - begin, MPI_DOUBLE, p, true);
            }
        }
        std::copy_n(rows.data(), localOffsets.size(), localOffsets.data());
        localColumns.assign(columns.begin(), columns.begin() + rows[localRows]);
        localValues.assign(values.begin(), values.begin() + rows[localRows]);
        std::vector<double>().swap(values);
        std::vector<index_t>().swap(columns);
        std::vector<index_t>().swap(rows);
    } else {
        transfer(localOffsets.data(), localOffsets.size(), MPI_UINT32_T, 0, false);
        size_t count = localOffsets.back() - localOffsets.front();
        localColumns.resize(count);
        localValues.resize(count);
        transfer(localColumns.data(), count, MPI_UINT32_T, 0, false);
        transfer(localValues.data(), count, MPI_DOUBLE, 0, false);
    }
    const index_t base = localOffsets.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localOffsets.size(); ++i) localOffsets[i] -= base;

    double* dValues = deviceCopy(localValues);
    double* dVector = deviceCopy(vector);
    index_t* dColumns = deviceCopy(localColumns);
    index_t* dRows = deviceCopy(localOffsets);
    double* dOutput = deviceCopy(output);
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const size_t average = localRows ? localValues.size() / localRows : 0;
    const int width = average <= 4 ? 4 : average <= 8 ? 8 : 32;
    const unsigned blocks = static_cast<unsigned>(std::min(
        (uint64_t(localRows) * width + 255) / 256, uint64_t(prop.multiProcessorCount) * 32));
    auto launch = [&]() {
        if (!localRows) return;
        if (width == 4) spmvKernel<4><<<blocks, 256, 0, stream>>>(dValues, dColumns, dRows, dVector, dOutput, localRows);
        else if (width == 8) spmvKernel<8><<<blocks, 256, 0, stream>>>(dValues, dColumns, dRows, dVector, dOutput, localRows);
        else spmvKernel<32><<<blocks, 256, 0, stream>>>(dValues, dColumns, dRows, dVector, dOutput, localRows);
    };
    launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    // Batch launches in a graph to amortize host overhead for small matrices.
    const index_t batch = std::min(iterations, index_t(32));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (index_t i = 0; i < batch; ++i) launch();
    cudaCheck(cudaStreamEndCapture(stream, &graph));
    cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (uint64_t i = 0; i < iterations / batch; ++i)
        cudaCheck(cudaGraphLaunch(executable, stream));
    for (index_t i = 0; i < iterations % batch; ++i) launch();
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }
    if (validate || printResults)
        cudaCheck(cudaMemcpy(output.data(), dOutput, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (!rank) {
            std::vector<double> result(numRows);
            std::copy(output.begin(), output.end(), result.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(result.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(result, "OutputVector");
        } else transfer(output.data(), output.size(), MPI_DOUBLE, 0, true);
    }
    int valid = 1, allValid = 1;
    if (validate) {
        std::vector<double> reference(localRows);
        spmvCpu(localValues.data(), localColumns.data(), localOffsets.data(), vector.data(), localRows, reference.data());
        valid = verifyResults(reference.data(), output.data(), localRows);
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
    }
    cudaCheck(cudaGraphExecDestroy(executable));
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(dValues));
    cudaCheck(cudaFree(dVector));
    cudaCheck(cudaFree(dColumns));
    cudaCheck(cudaFree(dRows));
    cudaCheck(cudaFree(dOutput));
    return allValid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
