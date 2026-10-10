#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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


// Consecutive rows remain together. Include row overhead so empty matrices and
// very sparse matrices are balanced too, without splitting any row's sum.
std::vector<index_t> partitionRows(const index_t* offsets, index_t rows, int parts) {
    std::vector<index_t> cuts(static_cast<size_t>(parts) + 1);
    const uint64_t total = uint64_t(offsets[rows] - offsets[0]) + rows;
    for (int p = 1; p < parts; ++p) {
        const uint64_t target = total * p / parts;
        index_t lo = cuts[p - 1], hi = rows;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (uint64_t(offsets[mid] - offsets[0]) + mid < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        cuts[p] = lo;
    }
    cuts[parts] = rows;
    return cuts;
}

void mpiCheck(int code) {
    if (code != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(code, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, code);
    }
}

// Chunk transfers to support CSR arrays larger than MPI's int count limit.
template<class T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD));
        else
            mpiCheck(MPI_Recv(data + offset, chunk, type, peer, tag, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

void cudaCheck(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

template<class T>
void upload(T*& device, const T* host, size_t count, cudaStream_t stream) {
    if (!count) return;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T)));
    cudaCheck(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

// A power-of-two subgroup cooperates on each row. Small rows use fewer lanes;
// dense rows use a full warp. All lanes execute the shuffle, including padding
// lanes at the end of a partition. The vector is read-only and shared in cache.
template<int Width>
__global__ void spmvGpu(const double* __restrict__ values,
                        const index_t* __restrict__ columns,
                        const index_t* __restrict__ offsets,
                        const double* __restrict__ vector,
                        index_t rows, index_t base, double* __restrict__ output) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x / Width;
    for (uint64_t group = first; group < (uint64_t(rows) + 32 / Width - 1) / (32 / Width)
                                               * (32 / Width); group += stride) {
        double sum = 0.0;
        if (group < rows) {
            const index_t end = offsets[group + 1] - base;
            for (uint64_t j = uint64_t(offsets[group] - base) + lane; j < end; j += Width)
                sum += values[j] * __ldg(vector + columns[j]);
        }
        for (int delta = Width / 2; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffffu, sum, delta, Width);
        if (group < rows && lane == 0) output[group] = sum;
    }
}

struct GpuPartition {
    int device = 0, blocks = 0, width = 1;
    index_t first = 0, rows = 0, base = 0;
    double *values = nullptr, *vector = nullptr, *output = nullptr;
    index_t *columns = nullptr, *offsets = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    void launch() const {
        if (!rows) return;
        switch (width) {
#define LAUNCH(W) case W: spmvGpu<W><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, base, output); break
            LAUNCH(1); LAUNCH(2); LAUNCH(4); LAUNCH(8); LAUNCH(16); LAUNCH(32);
#undef LAUNCH
        }
        cudaCheck(cudaGetLastError());
    }

    void release() {
        cudaCheck(cudaSetDevice(device));
        if (executable) cudaCheck(cudaGraphExecDestroy(executable));
        if (graph) cudaCheck(cudaGraphDestroy(graph));
        if (values) cudaCheck(cudaFree(values));
        if (columns) cudaCheck(cudaFree(columns));
        if (offsets) cudaCheck(cudaFree(offsets));
        if (vector) cudaCheck(cudaFree(vector));
        if (output) cudaCheck(cudaFree(output));
        if (stream) cudaCheck(cudaStreamDestroy(stream));
    }
};

// Each OpenMP worker owns a GPU. MPI calls remain on the main thread, including
// failures: exceptions from CUDA workers are reported after the team joins.
template<class F>
void onGpus(std::vector<GpuPartition>& gpus, F work) {
    std::string error;
    #pragma omp parallel for num_threads(gpus.size()) schedule(static, 1)
    for (size_t i = 0; i < gpus.size(); ++i) {
        try {
            cudaCheck(cudaSetDevice(gpus[i].device));
            work(gpus[i]);
        } catch (const std::exception& e) {
            #pragma omp critical
            error = e.what();
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
}

index_t parseIndex(const char* argument) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(argument, &end, 10);
    if (errno || argument[0] == '-' || end == argument || *end ||
        value > std::numeric_limits<index_t>::max())
        throw std::runtime_error("Invalid integer argument");
    return static_cast<index_t>(value);
}

int benchmark(int argc, char** argv, int rank, int ranks) {
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

    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &localComm));
    int localRank = 0, localRanks = 1;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    mpiCheck(MPI_Comm_size(localComm, &localRanks));
    mpiCheck(MPI_Comm_free(&localComm));
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) throw std::runtime_error("CUDA GPU required on every MPI rank");
    std::vector<GpuPartition> gpus;
    for (int device = localRank; device < deviceCount; device += localRanks) {
        gpus.emplace_back();
        gpus.back().device = device;
    }
    // Also supports schedulers that expose one GPU per rank, and oversubscription.
    if (gpus.empty()) {
        gpus.emplace_back();
        gpus.back().device = localRank % deviceCount;
    }
    const int gpuCount = static_cast<int>(gpus.size());
    std::vector<int> gpuCounts(ranks);
    mpiCheck(MPI_Allgather(&gpuCount, 1, MPI_INT, gpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    const int totalGpus = [&] { int n = 0; for (int c : gpuCounts) n += c; return n; }();

    std::vector<double> fullValues, vector(numRows), output, reference;
    std::vector<index_t> fullColumns, fullOffsets, cuts(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> metadata(static_cast<size_t>(ranks) * 2);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations,
               maxVal, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPU workers: %d\n", ranks, totalGpus);
        printf("Initializing data structures...\n");
        fullValues.resize(nItems);
        fullColumns.resize(nItems);
        fullOffsets.resize(size_t(numRows) + 1);
        // Keep the original rand() call order: changing this changes the matrix.
        fill(vector.data(), numRows, maxVal);
        fill(fullValues.data(), nItems, maxVal);
        initRandomMatrix(fullColumns.data(), fullOffsets.data(), nItems, numRows);
        const auto gpuCuts = partitionRows(fullOffsets.data(), numRows, totalGpus);
        int worker = 0;
        for (int p = 0; p < ranks; ++p) {
            cuts[p] = gpuCuts[worker];
            worker += gpuCounts[p];
            metadata[2 * p] = gpuCuts[worker] - cuts[p];
            metadata[2 * p + 1] = fullOffsets[gpuCuts[worker]] - fullOffsets[cuts[p]];
        }
        cuts[ranks] = numRows;
    }
    index_t localMetadata[2];
    mpiCheck(MPI_Scatter(metadata.data(), 2, MPI_UINT32_T, localMetadata, 2,
                         MPI_UINT32_T, 0, MPI_COMM_WORLD));
    const index_t localRows = localMetadata[0], localNnz = localMetadata[1];
    std::vector<double> values(localNnz), localOutput(localRows), localReference;
    std::vector<index_t> columns(localNnz), offsets(size_t(localRows) + 1);
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t first = cuts[p], begin = fullOffsets[first];
            transfer(fullOffsets.data() + first, size_t(metadata[2*p]) + 1,
                     MPI_UINT32_T, p, 0, true);
            transfer(fullValues.empty() ? nullptr : fullValues.data() + begin,
                     metadata[2*p+1], MPI_DOUBLE, p, 1, true);
            transfer(fullColumns.empty() ? nullptr : fullColumns.data() + begin,
                     metadata[2*p+1], MPI_UINT32_T, p, 2, true);
        }
        std::copy_n(fullOffsets.data(), size_t(localRows) + 1, offsets.data());
        std::copy_n(fullValues.data(), localNnz, values.data());
        std::copy_n(fullColumns.data(), localNnz, columns.data());
        std::vector<double>().swap(fullValues);
        std::vector<index_t>().swap(fullColumns);
        std::vector<index_t>().swap(fullOffsets);
    } else {
        transfer(offsets.data(), offsets.size(), MPI_UINT32_T, 0, 0, false);
        transfer(values.data(), values.size(), MPI_DOUBLE, 0, 1, false);
        transfer(columns.data(), columns.size(), MPI_UINT32_T, 0, 2, false);
    }
    for (size_t offset = 0; offset < vector.size();) {
        const int chunk = static_cast<int>(std::min(vector.size() - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(vector.data() + offset, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD));
        offset += chunk;
    }
    const index_t base = offsets[0];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < offsets.size(); ++i) offsets[i] -= base;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        localReference.resize(localRows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vector.data(),
                localRows, localReference.data());
    }

    const auto gpuCuts = partitionRows(offsets.data(), localRows, gpuCount);
    for (int i = 0; i < gpuCount; ++i) {
        gpus[i].first = gpuCuts[i];
        gpus[i].rows = gpuCuts[i + 1] - gpuCuts[i];
        gpus[i].base = offsets[gpuCuts[i]];
    }
    // Capture a bounded batch: launches stay asynchronous and graph replay
    // avoids repeated per-kernel host overhead for small matrices.
    const index_t batch = std::min(iterations, index_t(64));
    onGpus(gpus, [&](GpuPartition& gpu) {
        cudaCheck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
        if (!gpu.rows || !iterations) return;
        const size_t nnz = offsets[gpu.first + gpu.rows] - gpu.base;
        upload(gpu.values, nnz ? values.data() + gpu.base : nullptr, nnz, gpu.stream);
        upload(gpu.columns, nnz ? columns.data() + gpu.base : nullptr, nnz, gpu.stream);
        upload(gpu.offsets, offsets.data() + gpu.first, size_t(gpu.rows) + 1, gpu.stream);
        upload(gpu.vector, vector.data(), vector.size(), gpu.stream);
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.output), size_t(gpu.rows) * sizeof(double)));
        const double average = static_cast<double>(nnz) / gpu.rows;
        gpu.width = average < 2 ? 1 : average < 4 ? 2 : average < 8 ? 4 :
                    average < 16 ? 8 : average < 32 ? 16 : 32;
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, gpu.device));
        gpu.blocks = static_cast<int>(std::min((uint64_t(gpu.rows) * gpu.width + 255) / 256,
                                               uint64_t(properties.multiProcessorCount) * 32));
        gpu.launch(); // Warm caches and finish uploads before timing/capture.
        cudaCheck(cudaStreamSynchronize(gpu.stream));
        cudaCheck(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < batch; ++i) gpu.launch();
        cudaCheck(cudaStreamEndCapture(gpu.stream, &gpu.graph));
        cudaCheck(cudaGraphInstantiate(&gpu.executable, gpu.graph, nullptr, nullptr, 0));
        cudaCheck(cudaGraphUpload(gpu.executable, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    });
    std::vector<double>().swap(values);
    std::vector<index_t>().swap(columns);
    std::vector<index_t>().swap(offsets);
    std::vector<double>().swap(vector);

    if (rank == 0) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    onGpus(gpus, [&](GpuPartition& gpu) {
        if (!gpu.rows || !iterations) return;
        for (uint64_t i = 0; i < iterations / batch; ++i)
            cudaCheck(cudaGraphLaunch(gpu.executable, gpu.stream));
        for (index_t i = 0; i < iterations % batch; ++i) gpu.launch();
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    });
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    onGpus(gpus, [&](GpuPartition& gpu) {
        if (gpu.rows && iterations) {
            cudaCheck(cudaMemcpyAsync(localOutput.data() + gpu.first, gpu.output,
                                     size_t(gpu.rows) * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
            cudaCheck(cudaStreamSynchronize(gpu.stream));
        }
        gpu.release();
    });

    if (printResults || validate) {
        if (rank == 0) {
            output.resize(numRows);
            std::copy(localOutput.begin(), localOutput.end(), output.begin());
            if (validate) {
                reference.resize(numRows);
                std::copy(localReference.begin(), localReference.end(), reference.begin());
            }
            for (int p = 1; p < ranks; ++p) {
                const size_t count = cuts[p + 1] - cuts[p];
                transfer(output.data() + cuts[p], count, MPI_DOUBLE, p, 3, false);
                if (validate) transfer(reference.data() + cuts[p], count, MPI_DOUBLE, p, 4, false);
            }
        } else {
            transfer(localOutput.data(), localRows, MPI_DOUBLE, 0, 3, true);
            if (validate) transfer(localReference.data(), localRows, MPI_DOUBLE, 0, 4, true);
        }
    }
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), output.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED support required");
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
