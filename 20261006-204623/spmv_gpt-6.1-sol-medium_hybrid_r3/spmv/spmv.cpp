#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
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


// MPI calls stay on the main thread; OpenMP is used only for independent host work.
void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers so CSR arrays are not limited by MPI's signed int counts.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send) mpiCheck(MPI_Send(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

template<class T>
T* deviceCopy(const std::vector<T>& source) {
    T* device = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), std::max(size_t(1), source.size()) * sizeof(T)));
    if (!source.empty())
        cudaCheck(cudaMemcpy(device, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

// Small subwarps handle sparse rows; full warps coalesce accesses in longer rows.
// Every lane participates in shuffles, including lanes beyond the final row.
template<int Width>
__global__ void spmvGpu(const double* __restrict__ values, const index_t* __restrict__ cols,
                        const index_t* __restrict__ rows, const double* __restrict__ vec,
                        index_t count, double* __restrict__ out) {
    const unsigned lane = threadIdx.x % Width;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    const uint64_t warpFirst = first - (threadIdx.x % 32) / Width;
    for (uint64_t step = 0; warpFirst + step < count; step += stride) {
        const uint64_t base = first + step;
        double sum = 0.0;
        if (base < count) {
            const uint64_t end = rows[base + 1];
            for (uint64_t j = uint64_t(rows[base]) + lane; j < end; j += Width)
                sum += values[j] * vec[cols[j]];
        }
        for (int offset = Width / 2; offset; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset, Width);
        if (lane == 0 && base < count) out[base] = sum;
    }
}

bool parseIndex(const char* text, index_t& value) {
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *text == '-' || end == text || *end || parsed > std::numeric_limits<index_t>::max()) return false;
    value = static_cast<index_t>(parsed);
    return true;
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        bool ok = true;
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) ok = parseIndex(argv[++i], numRows);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) ok = parseIndex(argv[++i], sparsity);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) ok = parseIndex(argv[++i], iterations);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* text = argv[++i];
            errno = 0;
            maxVal = strtod(text, &end);
            ok = !errno && end != text && !*end && std::isfinite(maxVal);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else ok = false;
        if (!ok) {
            if (rank == 0) { fprintf(stderr, "Invalid option or value: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (!numRows || numRows == std::numeric_limits<index_t>::max() || !sparsity || !iterations ||
        entries / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Rows, sparsity and iterations must be positive; CSR must fit 32-bit indices.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    mpiCheck(MPI_Comm_size(node, &localRanks));
    // Avoid CPU oversubscription unless the user explicitly sets the thread count.
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, omp_get_num_procs() / localRanks));
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Honors CUDA_VISIBLE_DEVICES, including one visible GPU per scheduler task.
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&node));

    std::vector<double> globalValues, vec(numRows);
    std::vector<index_t> globalCols, globalRows, borders(size_t(ranks) + 1), nzBorders(size_t(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\nSparsity: 1 out of %u entries is non-zero\n", numRows, numRows, sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, 100.0 * (1.0 - double(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Parallel execution: %d MPI ranks, OpenMP host processing, CUDA SpMV\n", ranks);
        printf("Initializing data structures...\n");
        globalValues.resize(nItems); globalCols.resize(nItems); globalRows.resize(size_t(numRows) + 1);
        // Preserve the original rand() sequence and matrix exactly.
        fill(vec.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
        borders[0] = nzBorders[0] = 0;
        borders[ranks] = numRows; nzBorders[ranks] = nItems;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (uint64_t(nItems) + numRows) * p / ranks;
            index_t lo = 0, hi = numRows;
            while (lo < hi) {
                index_t mid = lo + (hi - lo) / 2;
                if (uint64_t(globalRows[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            borders[p] = lo; nzBorders[p] = globalRows[lo];
        }
    }
    mpiCheck(MPI_Bcast(borders.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Bcast(nzBorders.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    for (size_t offset = 0; offset < numRows;) {
        const int chunk = static_cast<int>(std::min(size_t(numRows) - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(vec.data() + offset, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        offset += chunk;
    }
    const index_t localRows = borders[rank + 1] - borders[rank];
    const index_t localNnz = nzBorders[rank + 1] - nzBorders[rank];
    std::vector<double> values(localNnz), out(localRows);
    std::vector<index_t> cols(localNnz), rows(size_t(localRows) + 1);
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const size_t count = size_t(nzBorders[p + 1]) - nzBorders[p];
            if (count) {
                transfer(globalValues.data() + nzBorders[p], count, MPI_DOUBLE, p, 0, true);
                transfer(globalCols.data() + nzBorders[p], count, MPI_UINT32_T, p, 1, true);
            }
            transfer(globalRows.data() + borders[p], size_t(borders[p + 1]) - borders[p] + 1, MPI_UINT32_T, p, 2, true);
        }
        std::copy_n(globalValues.begin(), localNnz, values.begin());
        std::copy_n(globalCols.begin(), localNnz, cols.begin());
        std::copy_n(globalRows.begin(), size_t(localRows) + 1, rows.begin());
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalCols); std::vector<index_t>().swap(globalRows);
    } else {
        transfer(values.data(), localNnz, MPI_DOUBLE, 0, 0, false);
        transfer(cols.data(), localNnz, MPI_UINT32_T, 0, 1, false);
        transfer(rows.data(), size_t(localRows) + 1, MPI_UINT32_T, 0, 2, false);
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows.size(); ++i) rows[i] -= nzBorders[rank];
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), localRows, reference.data());
    }
    double* dValues = deviceCopy(values);
    index_t* dCols = deviceCopy(cols);
    index_t* dRows = deviceCopy(rows);
    double* dVec = deviceCopy(vec);
    double* dOut = deviceCopy(out);
    // No matrix/vector communication is necessary between repeated multiplies.
    std::vector<double>().swap(values); std::vector<index_t>().swap(cols);
    std::vector<index_t>().swap(rows); std::vector<double>().swap(vec);
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const double average = localRows ? double(localNnz) / localRows : 0.0;
    const int width = average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const unsigned blocks = static_cast<unsigned>(std::min(uint64_t(65535),
        (uint64_t(localRows) * width + 255) / 256));
    auto launch = [&]() {
        if (!localRows) return;
        switch (width) {
            case 4: spmvGpu<4><<<blocks, 256, 0, stream>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 8: spmvGpu<8><<<blocks, 256, 0, stream>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            case 16: spmvGpu<16><<<blocks, 256, 0, stream>>>(dValues, dCols, dRows, dVec, localRows, dOut); break;
            default: spmvGpu<32><<<blocks, 256, 0, stream>>>(dValues, dCols, dRows, dVec, localRows, dOut);
        }
        cudaCheck(cudaGetLastError());
    };
    // Warm up outside timing. A bounded graph amortizes launch overhead without
    // storing a node for every iteration of arbitrarily long benchmark runs.
    launch();
    cudaCheck(cudaStreamSynchronize(stream));
    const index_t batch = std::min(iterations, index_t(64));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (localRows) {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) launch();
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(executable, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (localRows) {
        for (uint64_t i = 0; i < iterations / batch; ++i) cudaCheck(cudaGraphLaunch(executable, stream));
        for (index_t i = 0; i < iterations % batch; ++i) launch();
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0);
    }
    if (validate || printResults) {
        if (localRows) cudaCheck(cudaMemcpy(out.data(), dOut, size_t(localRows) * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(numRows);
            std::copy(out.begin(), out.end(), result.begin());
            for (int p = 1; p < ranks; ++p)
                transfer(result.data() + borders[p], size_t(borders[p + 1]) - borders[p], MPI_DOUBLE, p, 3, false);
            print_results(result, "OutputVector");
        } else transfer(out.data(), out.size(), MPI_DOUBLE, 0, 3, true);
    }
    int valid = 1;
    if (validate) {
        const int localValid = verifyResults(reference.data(), out.data(), localRows) ? 1 : 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (executable) cudaCheck(cudaGraphExecDestroy(executable));
    if (graph) cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(dValues)); cudaCheck(cudaFree(dCols)); cudaCheck(cudaFree(dRows));
    cudaCheck(cudaFree(dVec)); cudaCheck(cudaFree(dOut));
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 0;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
