#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <exception>
#include <limits>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Chunk transfers so the 32-bit CSR format is not limited by MPI's int counts.
template <typename T>
static void transfer(T* data, size_t count, MPI_Datatype type, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// Each power-of-two subgroup cooperates on one CSR row. Short rows use fewer
// lanes, while long rows use a warp and coalesced loads. No atomics are needed.
template <int Width>
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vec,
                           index_t count, double* __restrict__ out) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += val[j] * vec[cols[j]];
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(__activemask(), sum, offset, Width);
        if (lane == 0) out[row] = sum;
    }
}

template <typename T>
static void deviceAllocate(T** ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), std::max(size_t(1), count) * sizeof(T)));
}

static index_t parseCount(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || text[0] == '-' || end == text || *end || !value ||
        value > std::numeric_limits<index_t>::max()) return 0;
    return static_cast<index_t>(value);
}

static int benchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseCount(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseCount(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseCount(argv[++i]);
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
    if (!numRows || !sparsity || !iterations || !std::isfinite(maxVal)) {
        if (rank == 0) fprintf(stderr, "Dimensions, sparsity and iterations must be positive; max value must be finite.\n");
        return 1;
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix exceeds the 32-bit CSR nonzero limit.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, localRank % devices));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA device: %s\n",
               ranks, omp_get_max_threads(), properties.name);
    }

    std::vector<double> globalVal, h_vec(numRows);
    std::vector<index_t> globalCols, globalRows;
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRows.resize(size_t(numRows) + 1);
        // rand() ordering is deliberately identical to the serial benchmark.
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        for (int p = 1; p < ranks; ++p) {
            if (nItems) {
                const index_t target = uint64_t(nItems) * p / ranks;
                boundaries[p] = static_cast<index_t>(std::lower_bound(
                    globalRows.begin(), globalRows.end(), target) - globalRows.begin());
            } else boundaries[p] = uint64_t(numRows) * p / ranks;
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t offset = 0; offset < numRows;) {
        const int chunk = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        MPI_Bcast(h_vec.data() + offset, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += chunk;
    }
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> h_rows(size_t(localRows) + 1);
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t first = globalRows[boundaries[p]];
            const index_t count = globalRows[boundaries[p + 1]] - first;
            transfer(globalRows.data() + boundaries[p], size_t(boundaries[p + 1]) - boundaries[p] + 1,
                     MPI_UINT32_T, p, true);
            if (count) {
                transfer(globalVal.data() + first, count, MPI_DOUBLE, p, true);
                transfer(globalCols.data() + first, count, MPI_UINT32_T, p, true);
            }
        }
        std::copy_n(globalRows.data(), h_rows.size(), h_rows.data());
        if (h_rows.back()) {
            h_val.assign(globalVal.begin(), globalVal.begin() + h_rows.back());
            h_cols.assign(globalCols.begin(), globalCols.begin() + h_rows.back());
        }
        std::vector<double>().swap(globalVal);
        std::vector<index_t>().swap(globalCols);
        std::vector<index_t>().swap(globalRows);
    } else {
        transfer(h_rows.data(), h_rows.size(), MPI_UINT32_T, 0, false);
        const index_t count = h_rows.back() - h_rows.front();
        h_val.resize(count);
        h_cols.resize(count);
        transfer(h_val.data(), count, MPI_DOUBLE, 0, false);
        transfer(h_cols.data(), count, MPI_UINT32_T, 0, false);
    }
    const index_t base = h_rows.front();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < h_rows.size(); ++i) h_rows[i] -= base;

    std::vector<double> h_out(localRows), h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rows.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    double *d_val, *d_vec, *d_out;
    index_t *d_cols, *d_rows;
    deviceAllocate(&d_val, h_val.size());
    deviceAllocate(&d_cols, h_cols.size());
    deviceAllocate(&d_rows, h_rows.size());
    deviceAllocate(&d_vec, h_vec.size());
    deviceAllocate(&d_out, localRows);
    if (!h_val.empty()) {
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), h_val.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), h_cols.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rows, h_rows.data(), h_rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), h_vec.size() * sizeof(double), cudaMemcpyHostToDevice));

    const size_t average = localRows ? h_val.size() / localRows : 0;
    const int width = average <= 2 ? 2 : average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const unsigned blocks = static_cast<unsigned>(std::max(uint64_t(1), std::min(
        (uint64_t(localRows) * width + 255) / 256, uint64_t(properties.multiProcessorCount) * 32)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&]() {
        if (!localRows) return;
        switch (width) {
            case 2: spmvKernel<2><<<blocks, 256, 0, stream>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out); break;
            case 4: spmvKernel<4><<<blocks, 256, 0, stream>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out); break;
            case 8: spmvKernel<8><<<blocks, 256, 0, stream>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out); break;
            case 16: spmvKernel<16><<<blocks, 256, 0, stream>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out); break;
            default: spmvKernel<32><<<blocks, 256, 0, stream>>>(d_val, d_cols, d_rows, d_vec, localRows, d_out);
        }
    };
    launch(); // Warm up outside the timed region.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Batch repeated kernels into a graph to amortize launch overhead for small
    // matrices, without inserting any per-iteration MPI synchronization.
    const index_t batch = std::min(iterations, index_t(32));
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
    if (localRows)
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    if (executable) CUDA_CHECK(cudaGraphExecDestroy(executable));
    if (graph) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = seconds * 1000.0 / iterations;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather only for external output; validation uses independent local CPU
    // references and a global pass/fail reduction.
    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(numRows);
            std::copy(h_out.begin(), h_out.end(), result.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(result.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(result, "OutputVector");
        } else transfer(h_out.data(), localRows, MPI_DOUBLE, 0, true);
    }

    if (validate) {
        const int valid = verifyResults(h_reference.data(), h_out.data(), localRows) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", allValid ? "PASSED" : "FAILED");
        return allValid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
