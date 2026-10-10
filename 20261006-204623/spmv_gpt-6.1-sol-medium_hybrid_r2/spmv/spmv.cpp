#include <algorithm>
#include <climits>
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

// Avoid creating large host teams for tiny partitions; respect OMP_NUM_THREADS.
int hostThreads(uint64_t work, uint64_t grain) {
    return static_cast<int>(std::min(uint64_t(omp_get_max_threads()),
                                    std::max(uint64_t(1), (work + grain - 1) / grain)));
}

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
    #pragma omp parallel for schedule(static) num_threads(hostThreads(dim, 128))
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


// MPI calls stay on the main thread (MPI_THREAD_FUNNELED). All errors abort
// the communicator so another rank cannot hang inside a collective.
void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunked transfers avoid MPI's signed-int count/displacement limit for large
// CSR matrices. Root posts all transfers before waiting, allowing overlap.
template <typename T>
void scatterSlices(const std::vector<T>& source, std::vector<T>& local,
                   const std::vector<index_t>& offsets,
                   const std::vector<uint64_t>& counts, MPI_Datatype type,
                   int rank, int ranks, int tag) {
    constexpr uint64_t chunk = 1u << 26;
    local.resize(static_cast<size_t>(counts[rank]));
    if (rank == 0) {
        #pragma omp parallel for schedule(static) num_threads(hostThreads(counts[0], 65536))
        for (uint64_t i = 0; i < counts[0]; ++i)
            local[i] = source[offsets[0] + i];
        std::vector<MPI_Request> requests;
        for (int p = 1; p < ranks; ++p) {
            for (uint64_t i = 0; i < counts[p]; i += chunk) {
                MPI_Request request;
                mpiCheck(MPI_Isend(source.data() + offsets[p] + i,
                                  static_cast<int>(std::min(chunk, counts[p] - i)),
                                  type, p, tag, MPI_COMM_WORLD, &request));
                requests.push_back(request);
            }
        }
        if (!requests.empty())
            mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    } else {
        for (uint64_t i = 0; i < counts[rank]; i += chunk)
            mpiCheck(MPI_Recv(local.data() + i,
                             static_cast<int>(std::min(chunk, counts[rank] - i)),
                             type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
}

// Each subgroup cooperatively evaluates a row, with coalesced CSR loads.
// Every lane participates in the shuffle, including lanes beyond the last row.
template <int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ rows,
                           const double* __restrict__ vector,
                           index_t count, double* __restrict__ output) {
    const unsigned lane = threadIdx.x % Width;
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    // Uniform warp loop bounds keep the full shuffle mask valid.
    for (uint64_t base = first - (threadIdx.x % 32) / Width;
         base < count; base += stride) {
        const uint64_t row = base + (threadIdx.x % 32) / Width;
        double sum = 0.0;
        if (row < count) {
            const uint64_t end = rows[row + 1];
            for (uint64_t j = uint64_t(rows[row]) + lane; j < end; j += Width)
                sum += values[j] * vector[columns[j]];
        }
        #pragma unroll
        for (int offset = Width / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, offset, Width);
        if (lane == 0 && row < count)
            output[row] = sum;
    }
}

struct DeviceMatrix {
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    template <typename T>
    void upload(T*& device, const std::vector<T>& host) {
        // A valid allocation also handles zero-nonzero and zero-row partitions.
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device),
                             std::max(size_t(1), host.size()) * sizeof(T)));
        if (!host.empty())
            cudaCheck(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(T),
                                      cudaMemcpyHostToDevice, stream));
    }

    ~DeviceMatrix() {
        if (executable) cudaCheck(cudaGraphExecDestroy(executable));
        if (graph) cudaCheck(cudaGraphDestroy(graph));
        cudaCheck(cudaFree(values));
        cudaCheck(cudaFree(columns));
        cudaCheck(cudaFree(rows));
        cudaCheck(cudaFree(vector));
        cudaCheck(cudaFree(output));
        if (stream) cudaCheck(cudaStreamDestroy(stream));
    }

    void launch(index_t count, int width, int maxBlocks) {
        if (!count) return;
        const int blocks = static_cast<int>(std::min(uint64_t(maxBlocks),
                                  (uint64_t(count) * width + 255) / 256));
        #define LAUNCH(W) spmvKernel<W><<<blocks, 256, 0, stream>>>(values, columns, rows, vector, count, output)
        switch (width) {
            case 2: LAUNCH(2); break;
            case 4: LAUNCH(4); break;
            case 8: LAUNCH(8); break;
            case 16: LAUNCH(16); break;
            default: LAUNCH(32); break;
        }
        #undef LAUNCH
        cudaCheck(cudaGetLastError());
    }
};

index_t parseIndex(const char* value) {
    char* end = nullptr;
    const unsigned long long number = strtoull(value, &end, 10);
    if (value[0] == '-' || end == value || *end || number > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Invalid integer option");
    return static_cast<index_t>(number);
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = parseIndex(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            char* end = nullptr;
            maxVal = strtod(argv[++i], &end);
            if (end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
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
    if (!sparsity)
        throw std::runtime_error("Sparsity must be positive");
    const uint64_t entries = uint64_t(numRows) * numRows;
    if (entries / sparsity > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Nonzero count exceeds the 32-bit CSR index range");
    const index_t nItems = static_cast<index_t>(entries / sparsity);

    // Select GPUs by node-local MPI rank, including schedulers that expose only
    // one assigned GPU through CUDA_VISIBLE_DEVICES.
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA device is required on every rank");
    const int device = localRank % devices;
    cudaCheck(cudaSetDevice(device));
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device));
    mpiCheck(MPI_Comm_free(&node));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n",
               iterations, maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA enabled\n", ranks, omp_get_max_threads());
        printf("Initializing data structures...\n");
    }

    std::vector<double> globalValues, vector(numRows);
    std::vector<index_t> globalColumns, globalRows;
    std::vector<index_t> cuts(static_cast<size_t>(ranks) + 1), nonzeroCuts(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRows.resize(size_t(numRows) + 1);
        // Keep exactly the original libc random stream and generation order.
        srand(1);
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRows.data(), nItems, numRows);
        cuts[ranks] = numRows;
        // Balance both nonzero work and row overhead, preserving row ownership.
        const uint64_t totalWork = uint64_t(nItems) + uint64_t(numRows) * 4;
        #pragma omp parallel for schedule(static) num_threads(hostThreads(ranks, 32))
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = totalWork * p / ranks;
            index_t low = 0, high = numRows;
            while (low < high) {
                const index_t middle = low + (high - low) / 2;
                if (uint64_t(globalRows[middle]) + uint64_t(middle) * 4 < target)
                    low = middle + 1;
                else high = middle;
            }
            cuts[p] = low;
        }
        #pragma omp parallel for schedule(static) num_threads(hostThreads(ranks, 1024))
        for (int p = 0; p <= ranks; ++p)
            nonzeroCuts[p] = globalRows[cuts[p]];
    }
    mpiCheck(MPI_Bcast(cuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Bcast(nonzeroCuts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    for (uint64_t i = 0; i < numRows; i += uint64_t(INT_MAX))
        mpiCheck(MPI_Bcast(vector.data() + i, static_cast<int>(std::min(uint64_t(INT_MAX), uint64_t(numRows) - i)),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));

    std::vector<uint64_t> rowCounts(ranks), nonzeroCounts(ranks);
    #pragma omp parallel for schedule(static) num_threads(hostThreads(ranks, 1024))
    for (int p = 0; p < ranks; ++p) {
        rowCounts[p] = uint64_t(cuts[p + 1]) - cuts[p] + 1;
        nonzeroCounts[p] = uint64_t(nonzeroCuts[p + 1]) - nonzeroCuts[p];
    }
    std::vector<index_t> rows, columns;
    std::vector<double> values;
    scatterSlices(globalRows, rows, cuts, rowCounts, MPI_UINT32_T, rank, ranks, 0);
    scatterSlices(globalColumns, columns, nonzeroCuts, nonzeroCounts, MPI_UINT32_T, rank, ranks, 1);
    scatterSlices(globalValues, values, nonzeroCuts, nonzeroCounts, MPI_DOUBLE, rank, ranks, 2);
    const index_t localRows = cuts[rank + 1] - cuts[rank];
    #pragma omp parallel for schedule(static) num_threads(hostThreads(localRows, 4096))
    for (uint64_t i = 0; i <= localRows; ++i)
        rows[i] -= nonzeroCuts[rank];
    // Root no longer needs the full matrix after distribution.
    std::vector<double>().swap(globalValues);
    std::vector<index_t>().swap(globalColumns);
    std::vector<index_t>().swap(globalRows);

    std::vector<double> output(localRows), reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), rows.data(), vector.data(), localRows, reference.data());
    }

    DeviceMatrix gpu;
    cudaCheck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    gpu.upload(gpu.values, values);
    gpu.upload(gpu.columns, columns);
    gpu.upload(gpu.rows, rows);
    gpu.upload(gpu.vector, vector);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.output), std::max(size_t(1), output.size()) * sizeof(double)));
    const double average = localRows ? double(values.size()) / localRows : 0;
    const int width = average <= 2 ? 2 : average <= 4 ? 4 : average <= 8 ? 8 : average <= 16 ? 16 : 32;
    const int maxBlocks = properties.multiProcessorCount * 32;
    if (iterations) gpu.launch(localRows, width, maxBlocks); // warm-up, excluded from timing
    cudaCheck(cudaStreamSynchronize(gpu.stream));

    // Uploaded inputs and the optional reference are sufficient from here on.
    std::vector<double>().swap(values);
    std::vector<double>().swap(vector);
    std::vector<index_t>().swap(columns);
    std::vector<index_t>().swap(rows);

    // The inputs never change across iterations. Replaying batches of launches
    // reduces CPU submission overhead without dropping any benchmark iterations.
    const index_t batch = std::min(iterations, index_t(128));
    if (localRows && iterations) {
        cudaCheck(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) gpu.launch(localRows, width, maxBlocks);
        cudaCheck(cudaStreamEndCapture(gpu.stream, &gpu.graph));
        cudaCheck(cudaGraphInstantiate(&gpu.executable, gpu.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(gpu.executable, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    if (localRows && iterations) {
        for (uint64_t i = 0; i < iterations / batch; ++i)
            cudaCheck(cudaGraphLaunch(gpu.executable, gpu.stream));
        for (index_t i = 0; i < iterations % batch; ++i)
            gpu.launch(localRows, width, maxBlocks);
    }
    cudaCheck(cudaStreamSynchronize(gpu.stream));
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (localRows && iterations)
        cudaCheck(cudaMemcpyAsync(output.data(), gpu.output, output.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
    cudaCheck(cudaStreamSynchronize(gpu.stream));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", seconds * 1000 / iterations);
        printf("Performance: %.3f GFLOPS\n", 2.0 * nItems * iterations / seconds / 1e9);
    }

    if (printResults) {
        constexpr uint64_t chunk = 1u << 26;
        if (rank == 0) {
            std::vector<double> gathered(numRows);
            std::copy(output.begin(), output.end(), gathered.begin());
            std::vector<MPI_Request> requests;
            for (int p = 1; p < ranks; ++p) {
                const uint64_t count = uint64_t(cuts[p + 1]) - cuts[p];
                for (uint64_t i = 0; i < count; i += chunk) {
                    MPI_Request request;
                    mpiCheck(MPI_Irecv(gathered.data() + cuts[p] + i,
                                      static_cast<int>(std::min(chunk, count - i)), MPI_DOUBLE,
                                      p, 3, MPI_COMM_WORLD, &request));
                    requests.push_back(request);
                }
            }
            if (!requests.empty())
                mpiCheck(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            print_results(gathered, "OutputVector");
        } else {
            for (uint64_t i = 0; i < localRows; i += chunk)
                mpiCheck(MPI_Send(output.data() + i, static_cast<int>(std::min(chunk, uint64_t(localRows) - i)),
                                  MPI_DOUBLE, 0, 3, MPI_COMM_WORLD));
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = verifyResults(reference.data(), output.data(), localRows);
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
        return 1;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return result;
}
