#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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
        if (!std::isfinite(res)) {
            printf("Validation failed at index %u: non-finite result\n", i);
            return false;
        } else if (std::abs(ref) < 1e-10) {
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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

// Chunked transfers avoid MPI's signed-int count limit for large CSR blocks.
template <typename T>
static void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send) {
    constexpr size_t chunk = 1u << 26;
    for (size_t offset = 0; offset < count; offset += chunk) {
        int length = static_cast<int>(std::min(chunk, count - offset));
        if (send) MPI_Send(data + offset, length, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data + offset, length, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

template <typename T>
static void broadcast(T* data, size_t count, MPI_Datatype type) {
    constexpr size_t chunk = 1u << 26;
    for (size_t offset = 0; offset < count; offset += chunk)
        MPI_Bcast(data + offset, static_cast<int>(std::min(chunk, count - offset)),
                  type, 0, MPI_COMM_WORLD);
}

// A power-of-two subgroup cooperates on each row. Short rows use fewer lanes;
// long rows use a whole warp for coalesced matrix reads and vector gathers.
template <int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           double* __restrict__ output, index_t count) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        const unsigned mask = __activemask();
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(mask, sum, offset, Width);
        if (lane == 0) output[row] = sum;
    }
}

static index_t parseIndex(const char* text) {
    char* end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || text[0] == '-' ||
        value == 0 || value >= std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Expected a positive 32-bit integer, got: %s\n", text);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<index_t>(value);
}

static int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const uint64_t items = uint64_t(numRows) * numRows / sparsity;
    if (items > std::numeric_limits<index_t>::max() || !std::isfinite(maxVal)) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds 32-bit CSR capacity or maximum value is non-finite\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(items);

    // Respect CUDA_VISIBLE_DEVICES and assign ranks on each node to local GPUs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (uint64_t(numRows) * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(size_t(numRows) + 1);
        printf("Initializing data structures...\n");
        // rand's original default seed, independent of library initialization.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    broadcast(h_vec.data(), h_vec.size(), MPI_DOUBLE);

    // Balance matrix traffic and row overhead, including matrices with empty rows.
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (rank == 0) {
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        const uint64_t work = uint64_t(nItems) + uint64_t(numRows) * 4;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (work / ranks) * p + (work % ranks) * p / ranks;
            index_t low = boundaries[p - 1], high = numRows;
            while (low < high) {
                index_t mid = low + (high - low) / 2;
                if (uint64_t(h_rowDelimiters[mid]) + uint64_t(mid) * 4 < target) low = mid + 1;
                else high = mid;
            }
            boundaries[p] = low;
        }
    }
    broadcast(boundaries.data(), boundaries.size(), MPI_UINT32_T);
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    index_t localItems = 0;
    std::vector<double> values;
    std::vector<index_t> columns, rows(size_t(localRows) + 1);
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            index_t begin = h_rowDelimiters[boundaries[p]];
            index_t count = h_rowDelimiters[boundaries[p + 1]] - begin;
            MPI_Send(&count, 1, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            transfer(h_rowDelimiters.data() + boundaries[p], size_t(boundaries[p + 1]) - boundaries[p] + 1,
                     MPI_UINT32_T, p, true);
            if (count) {
                transfer(h_val.data() + begin, count, MPI_DOUBLE, p, true);
                transfer(h_cols.data() + begin, count, MPI_UINT32_T, p, true);
            }
        }
        localItems = h_rowDelimiters[localRows];
        rows.assign(h_rowDelimiters.begin(), h_rowDelimiters.begin() + size_t(localRows) + 1);
        // Reuse the root allocations instead of copying its matrix partition.
        values.swap(h_val);
        columns.swap(h_cols);
        values.resize(localItems);
        columns.resize(localItems);
        std::vector<index_t>().swap(h_rowDelimiters);
    } else {
        MPI_Recv(&localItems, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values.resize(localItems);
        columns.resize(localItems);
        transfer(rows.data(), rows.size(), MPI_UINT32_T, 0, false);
        transfer(values.data(), localItems, MPI_DOUBLE, 0, false);
        transfer(columns.data(), localItems, MPI_UINT32_T, 0, false);
    }
    const index_t base = rows[0];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows.size(); ++i) rows[i] -= base;

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    double *d_values = nullptr, *d_vector = nullptr, *d_output = nullptr;
    index_t *d_columns = nullptr, *d_rows = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (localRows) {
        CUDA_CHECK(cudaMalloc(&d_values, std::max(size_t(1), size_t(localItems)) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_columns, std::max(size_t(1), size_t(localItems)) * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_rows, rows.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMalloc(&d_vector, h_vec.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_output, size_t(localRows) * sizeof(double)));
        if (localItems) {
            CUDA_CHECK(cudaMemcpyAsync(d_values, values.data(), size_t(localItems) * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_columns, columns.data(), size_t(localItems) * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(d_rows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vector, h_vec.data(), h_vec.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    }
    const uint64_t average = localRows ? uint64_t(localItems) / localRows : 0;
    const int width = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const int blocks = static_cast<int>(std::min<uint64_t>(
        (uint64_t(localRows) * width + 255) / 256, uint64_t(properties.multiProcessorCount) * 32));
    auto launch = [&]() {
        if (!localRows) return;
        if (width == 4) spmvKernel<4><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, d_output, localRows);
        else if (width == 8) spmvKernel<8><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, d_output, localRows);
        else if (width == 16) spmvKernel<16><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, d_output, localRows);
        else spmvKernel<32><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, d_output, localRows);
    };
    // Warm up the context and kernel before timing. All repeated multiplies use
    // resident data; neither PCIe transfers nor MPI collectives occur per iteration.
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(columns);
    std::vector<index_t>().swap(rows);
    std::vector<double>().swap(h_vec);

    // Bounded-size graphs amortize CPU launch overhead without growing storage
    // with the iteration count. Each graph node is still one complete SpMV.
    const index_t batch = std::min<index_t>(iterations, 64);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (localRows) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphUpload(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localRows) {
        for (index_t i = 0; i < iterations / batch; ++i)
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);

        // Report the slowest rank's elapsed time for global throughput.
        const double gflops = seconds > 0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0;
        const double avgTime = seconds * 1000.0 / iterations;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> localOutput(localRows);
    if (localRows && (validate || printResults)) {
        CUDA_CHECK(cudaMemcpyAsync(localOutput.data(), d_output, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_columns));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_output));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            h_out.resize(numRows);
            std::copy(localOutput.begin(), localOutput.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(h_out.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(h_out, "OutputVector");
        } else {
            transfer(localOutput.data(), localOutput.size(), MPI_DOUBLE, 0, true);
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = verifyResults(h_reference.data(), localOutput.data(), localRows);
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
