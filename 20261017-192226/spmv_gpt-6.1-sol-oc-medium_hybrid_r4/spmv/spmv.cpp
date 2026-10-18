#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <exception>

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
        if (!std::isfinite(res) || !std::isfinite(ref)) {
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

// Chunked transfers also support CSR arrays larger than MPI's int count limit.
template<class T>
static void transfer(T* data, uint64_t count, MPI_Datatype type, int peer,
                     bool sending) {
    while (count) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        if (sending) MPI_Send(data, chunk, type, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// A power-of-two subgroup cooperates on a row. Short rows use smaller groups
// so that sparse matrices do not waste an entire warp on each row.
template<int Width>
__global__ void spmvGpu(const double* __restrict__ values,
                        const index_t* __restrict__ columns,
                        const index_t* __restrict__ rows,
                        const double* __restrict__ vector,
                        index_t count, double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    const unsigned mask = (0xffffffffu >> (32 - Width)) << (threadIdx.x % 32 - lane);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t row = first; row < count; row += stride) {
        double sum = 0.0;
        for (uint64_t j = uint64_t(rows[row]) + lane; j < rows[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(mask, sum, offset, Width);
        if (lane == 0) output[row] = sum;
    }
}

static int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
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
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (!numRows || !sparsity || !iterations || !std::isfinite(maxVal) ||
        entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid dimensions, sparsity, iterations, or maximum value (CSR uses 32-bit indices).\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: CUDA device required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
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
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA devices visible to rank 0: %d\n",
               ranks, omp_get_max_threads(), devices);
    }

    // Allocate and initialize data structures
    std::vector<double> h_val(rank == 0 ? nItems : 0);
    std::vector<index_t> h_cols(rank == 0 ? nItems : 0);
    std::vector<index_t> h_rowDelimiters(rank == 0 ? size_t(numRows) + 1 : 0);
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(rank == 0 ? numRows : 0);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        // libc's original default seed, independent of MPI/CUDA initialization.
        srand(1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // The input vector is constant across benchmark iterations: communicate it
    // once, and keep all GPU inputs resident until the benchmark finishes.
    for (uint64_t offset = 0; offset < numRows;) {
        const int count = static_cast<int>(std::min<uint64_t>(numRows - offset, INT_MAX));
        MPI_Bcast(h_vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }
    std::vector<index_t> boundaries(size_t(ranks) + 1);
    if (rank == 0) {
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        #pragma omp parallel for schedule(static)
        for (int p = 1; p < ranks; ++p) {
            if (nItems == 0) boundaries[p] = uint64_t(numRows) * p / ranks;
            else boundaries[p] = static_cast<index_t>(std::lower_bound(
                h_rowDelimiters.begin(), h_rowDelimiters.end(),
                uint64_t(nItems) * p / ranks) - h_rowDelimiters.begin());
        }
    }
    MPI_Bcast(boundaries.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = boundaries[rank + 1] - boundaries[rank];
    std::vector<index_t> rows(size_t(localRows) + 1), columns;
    std::vector<double> values, output(localRows);
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t begin = h_rowDelimiters[boundaries[p]];
            const index_t count = h_rowDelimiters[boundaries[p + 1]] - begin;
            transfer(h_rowDelimiters.data() + boundaries[p],
                     uint64_t(boundaries[p + 1]) - boundaries[p] + 1, MPI_UINT32_T, p, true);
            if (count) {
                transfer(h_cols.data() + begin, count, MPI_UINT32_T, p, true);
                transfer(h_val.data() + begin, count, MPI_DOUBLE, p, true);
            }
        }
        std::copy_n(h_rowDelimiters.begin(), size_t(localRows) + 1, rows.begin());
        columns.assign(h_cols.begin(), h_cols.begin() + rows.back());
        values.assign(h_val.begin(), h_val.begin() + rows.back());
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
        std::vector<double>().swap(h_val);
    } else {
        transfer(rows.data(), uint64_t(localRows) + 1, MPI_UINT32_T, 0, false);
        const index_t count = rows.back() - rows.front();
        columns.resize(count);
        values.resize(count);
        transfer(columns.data(), count, MPI_UINT32_T, 0, false);
        transfer(values.data(), count, MPI_DOUBLE, 0, false);
    }
    const index_t base = rows.front();
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i <= localRows; ++i) rows[i] -= base;

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    double *d_values, *d_vector, *d_output;
    index_t *d_columns, *d_rows;
    CUDA_CHECK(cudaMalloc(&d_values, std::max<size_t>(values.size(), 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_columns, std::max<size_t>(columns.size(), 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rows, rows.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vector, size_t(numRows) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, std::max<size_t>(localRows, 1) * sizeof(double)));
    if (!values.empty()) {
        CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_columns, columns.data(), columns.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector, h_vec.data(), size_t(numRows) * sizeof(double), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const double average = localRows ? double(values.size()) / localRows : 0.0;
    const int width = average <= 2 ? 2 : average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const unsigned blocks = static_cast<unsigned>(std::max<uint64_t>(1,
        std::min<uint64_t>((uint64_t(localRows) * width + 255) / 256,
                          properties.multiProcessorCount * 32)));
    auto launch = [&]() {
        if (!localRows) return;
        switch (width) {
            case 2: spmvGpu<2><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, localRows, d_output); break;
            case 4: spmvGpu<4><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, localRows, d_output); break;
            case 8: spmvGpu<8><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, localRows, d_output); break;
            case 16: spmvGpu<16><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, localRows, d_output); break;
            default: spmvGpu<32><<<blocks, 256, 0, stream>>>(d_values, d_columns, d_rows, d_vector, localRows, d_output);
        }
    };
    launch(); // Warm up the GPU before timing.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Batch launches in a CUDA graph to amortize CPU submission overhead.
    const index_t batch = std::min<index_t>(iterations, 32);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (index_t i = 0; i < batch; ++i) launch();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t i = 0; i < iterations / batch; ++i)
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
    for (index_t i = 0; i < iterations % batch; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (localRows)
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_columns));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_output));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double gflops = (2.0 * nItems * iterations) / seconds / 1e9;
        const double avgTime = seconds * 1000.0 / iterations;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            std::copy(output.begin(), output.end(), h_out.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(h_out.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, false);
            print_results(h_out, "OutputVector");
        } else transfer(output.data(), localRows, MPI_DOUBLE, 0, true);
    }

    // Validation
    if (validate) {
        const int localValid = verifyResults(h_reference.data(), output.data(), localRows);
        int valid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
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
