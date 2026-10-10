#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(message, length));
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

// Keep rand() calls on rank zero, in the original order: the vector and values
// use the default seed, and the CSR pattern uses seed 8675309.
void fill(double* array, index_t count, double maxValue) {
    for (index_t i = 0; i < count; ++i)
        array[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
}

void initRandomMatrix(index_t* columns, index_t* offsets, index_t count, index_t dim) {
    const uint64_t entries = static_cast<uint64_t>(dim) * dim;
    const double probability = entries ? static_cast<double>(count) / entries : 0.0;
    srand(8675309);
    // These two cases have a unique pattern, independent of the random draws.
    if (count == 0) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row <= dim; ++row) offsets[row] = 0;
        return;
    }
    if (count == entries) {
        #pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < dim; ++row) {
            offsets[row] = static_cast<index_t>(static_cast<uint64_t>(row) * dim);
            for (index_t col = 0; col < dim; ++col)
                columns[static_cast<uint64_t>(row) * dim + col] = col;
        }
        offsets[dim] = count;
        return;
    }
    index_t assigned = 0;
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        offsets[row] = assigned;
        if (assigned == count) continue;
        for (index_t col = 0; col < dim; ++col) {
            const uint64_t left = entries - (static_cast<uint64_t>(row) * dim + col);
            if (left <= count - assigned) fillRemaining = true;
            const double random = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < count && random <= probability) || fillRemaining)
                columns[assigned++] = col;
        }
    }
    offsets[dim] = count;
}

// A serial accumulation within each row provides an independent reference;
// OpenMP distributes rows without changing their summation order.
void spmvCpu(const double* values, const index_t* columns, const index_t* offsets,
             const double* vector, index_t rows, double* output) {
    #pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (uint64_t j = offsets[row]; j < offsets[row + 1]; ++j)
            sum += values[j] * vector[columns[j]];
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t size,
                   index_t firstRow) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i], res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (!std::isfinite(res) || error > MAX_RELATIVE_ERROR) {
            fprintf(stderr, "Validation failed at index %u: reference %.10e, got %.10e\n",
                    firstRow + i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

index_t parseIndex(const char* text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || *text == '-' || value > UINT32_MAX)
        throw std::runtime_error("Invalid integer argument");
    return static_cast<index_t>(value);
}

// Balance the number of nonzeros plus a small per-row cost. The latter also
// distributes empty rows, and every row belongs to exactly one partition.
index_t rowBoundary(const index_t* offsets, index_t rows, uint64_t part, uint64_t parts) {
    if (!part) return 0;
    if (part == parts) return rows;
    const uint64_t total = static_cast<uint64_t>(offsets[rows]) + rows;
    const uint64_t target = total / parts * part + (total % parts) * part / parts;
    index_t low = 0, high = rows;
    while (low < high) {
        const index_t middle = low + (high - low) / 2;
        if (static_cast<uint64_t>(offsets[middle]) + middle < target) low = middle + 1;
        else high = middle;
    }
    return low;
}

// MPI-3 collectives use signed int counts. Chunk only unusually large buffers,
// retaining a collective scatter in the common case.
template <typename T>
void scatterNonzeros(const std::vector<T>& source, std::vector<T>& local,
                     const std::vector<index_t>& boundaries, MPI_Datatype type,
                     int rank, int ranks, int tag) {
    if (boundaries.back() <= static_cast<index_t>(INT_MAX)) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int p = 0; p < ranks; ++p) {
            counts[p] = static_cast<int>(boundaries[p + 1] - boundaries[p]);
            displacements[p] = static_cast<int>(boundaries[p]);
        }
        mpiCheck(MPI_Scatterv(source.data(), counts.data(), displacements.data(), type,
                              local.data(), static_cast<int>(local.size()), type, 0, MPI_COMM_WORLD));
    } else if (rank == 0) {
        std::copy_n(source.data(), local.size(), local.data());
        for (int p = 1; p < ranks; ++p) {
            for (uint64_t pos = boundaries[p]; pos < boundaries[p + 1];) {
                const int count = static_cast<int>(std::min<uint64_t>(INT_MAX, boundaries[p + 1] - pos));
                mpiCheck(MPI_Send(source.data() + pos, count, type, p, tag, MPI_COMM_WORLD));
                pos += count;
            }
        }
    } else {
        for (size_t pos = 0; pos < local.size();) {
            const int count = static_cast<int>(std::min<size_t>(INT_MAX, local.size() - pos));
            mpiCheck(MPI_Recv(local.data() + pos, count, type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            pos += count;
        }
    }
}

// A power-of-two group of lanes handles one CSR row. Small rows use subwarps;
// longer rows use a full warp, with coalesced accesses and no output atomics.
template <int Width>
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const index_t* __restrict__ offsets,
                           const double* __restrict__ vector,
                           index_t rows, double* __restrict__ output) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x & 31) & ~(Width - 1));
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x / Width;
    for (uint64_t row = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / Width;
         row < rows; row += stride) {
        double sum = 0.0;
        for (uint64_t j = static_cast<uint64_t>(offsets[row]) + lane; j < offsets[row + 1]; j += Width)
            sum += values[j] * vector[columns[j]];
        #pragma unroll
        for (int shift = Width / 2; shift; shift /= 2)
            sum += __shfl_down_sync(mask, sum, shift, Width);
        if (!lane) output[row] = sum;
    }
}

struct GpuPartition {
    int device = 0;
    index_t first = 0, rows = 0;
    unsigned width = 1;
    index_t batch = 0;
    double* values = nullptr;
    double* vector = nullptr;
    double* output = nullptr;
    index_t* columns = nullptr;
    index_t* offsets = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t repeated = nullptr, tail = nullptr;

    GpuPartition(int id, index_t begin, index_t end) : device(id), first(begin), rows(end - begin) {}
    GpuPartition(const GpuPartition&) = delete;
    GpuPartition& operator=(const GpuPartition&) = delete;
    ~GpuPartition() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (repeated) cudaGraphExecDestroy(repeated);
        if (tail) cudaGraphExecDestroy(tail);
        if (values) cudaFree(values);
        if (columns) cudaFree(columns);
        if (offsets) cudaFree(offsets);
        if (vector) cudaFree(vector);
        if (output) cudaFree(output);
        if (stream) cudaStreamDestroy(stream);
    }

    void launch() {
        const unsigned blocks = static_cast<unsigned>(std::min<uint64_t>(65535,
            (static_cast<uint64_t>(rows) * width + 255) / 256));
        switch (width) {
            case 1: spmvKernel<1><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
            case 2: spmvKernel<2><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
            case 4: spmvKernel<4><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
            case 8: spmvKernel<8><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
            case 16: spmvKernel<16><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
            default: spmvKernel<32><<<blocks, 256, 0, stream>>>(values, columns, offsets, vector, rows, output); break;
        }
        cudaCheck(cudaGetLastError());
    }

    cudaGraphExec_t makeGraph(index_t count) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (index_t i = 0; i < count; ++i) launch();
        cudaCheck(cudaStreamEndCapture(stream, &graph));
        const cudaError_t status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        cudaCheck(status);
        return executable;
    }

    void prepare(const std::vector<double>& hostValues, const std::vector<index_t>& hostColumns,
                 const std::vector<index_t>& hostOffsets, const std::vector<double>& hostVector,
                 index_t iterations) {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (!rows) return;
        const index_t base = hostOffsets[first];
        const size_t nnz = static_cast<size_t>(hostOffsets[first + rows]) - base;
        std::vector<index_t> rebased(static_cast<size_t>(rows) + 1);
        for (size_t i = 0; i < rebased.size(); ++i) rebased[i] = hostOffsets[first + i] - base;
        if (nnz) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&values), nnz * sizeof(double)));
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&columns), nnz * sizeof(index_t)));
            cudaCheck(cudaMemcpyAsync(values, hostValues.data() + base, nnz * sizeof(double), cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaMemcpyAsync(columns, hostColumns.data() + base, nnz * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        }
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&offsets), rebased.size() * sizeof(index_t)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&vector), hostVector.size() * sizeof(double)));
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&output), static_cast<size_t>(rows) * sizeof(double)));
        cudaCheck(cudaMemcpyAsync(offsets, rebased.data(), rebased.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(vector, hostVector.data(), hostVector.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemsetAsync(output, 0, static_cast<size_t>(rows) * sizeof(double), stream));
        const uint64_t average = (nnz + rows - 1) / rows;
        while (width < 32 && width < average) width *= 2;
        if (iterations) {
            launch(); // Warm up the kernel and caches outside the timed region.
            cudaCheck(cudaStreamSynchronize(stream));
            batch = std::min<index_t>(iterations, 32);
            repeated = makeGraph(batch);
            if (iterations % batch) tail = makeGraph(iterations % batch);
        }
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void run(index_t iterations) {
        cudaCheck(cudaSetDevice(device));
        if (!rows || !iterations) return;
        for (index_t i = 0; i < iterations / batch; ++i)
            cudaCheck(cudaGraphLaunch(repeated, stream));
        if (tail) cudaCheck(cudaGraphLaunch(tail, stream));
    }

    void synchronize() {
        cudaCheck(cudaSetDevice(device));
        cudaCheck(cudaStreamSynchronize(stream));
    }

    void download(std::vector<double>& hostOutput) {
        cudaCheck(cudaSetDevice(device));
        if (!rows) return;
        cudaCheck(cudaMemcpyAsync(hostOutput.data() + first, output,
                  static_cast<size_t>(rows) * sizeof(double), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
    }
};

// MPI remains on the main thread (MPI_THREAD_FUNNELED). CUDA contexts, streams,
// and graphs are owned per GPU; OpenMP workers independently drive the devices.
template <typename Function>
void forEachGpu(std::vector<std::unique_ptr<GpuPartition>>& partitions, Function function) {
    const int count = static_cast<int>(partitions.size());
    const int threads = std::min(count, omp_get_max_threads());
    std::vector<std::exception_ptr> errors(count);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < count; ++i) {
        try { function(*partitions[i]); }
        catch (...) { errors[i] = std::current_exception(); }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numRows = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) sparsity = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = parseIndex(argv[++i]);
        else if (!strcmp(argv[i], "-m") && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            maxVal = strtod(argv[++i], &end);
            if (errno || end == argv[i] || *end || !std::isfinite(maxVal))
                throw std::runtime_error("Invalid maximum value");
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); return 0; }
        else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!sparsity) throw std::runtime_error("Sparsity must be positive");
    if (numRows >= static_cast<index_t>(INT_MAX))
        throw std::runtime_error("The row count exceeds MPI's supported count range");
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nnz64 = entries / sparsity;
    if (nnz64 > UINT32_MAX) throw std::runtime_error("The nonzero count exceeds the CSR index range");
    const index_t nItems = static_cast<index_t>(nnz64);

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int nodeRank = 0, nodeRanks = 0, deviceCount = 0;
    mpiCheck(MPI_Comm_rank(node, &nodeRank));
    mpiCheck(MPI_Comm_size(node, &nodeRanks));
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) throw std::runtime_error("Every MPI rank requires a visible CUDA GPU");
    std::vector<int> devices;
    for (int device = nodeRank; device < deviceCount; device += nodeRanks) devices.push_back(device);
    // With one GPU exposed per process, ordinal zero already identifies its GPU.
    // Oversubscribed ranks share devices instead of silently falling back to CPU.
    if (devices.empty()) devices.push_back(nodeRank % deviceCount);
    mpiCheck(MPI_Comm_free(&node));
    const int localDevices = static_cast<int>(devices.size());
    std::vector<int> deviceCounts(ranks);
    mpiCheck(MPI_Allgather(&localDevices, 1, MPI_INT, deviceCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    uint64_t totalDevices = 0;
    for (int count : deviceCounts) totalDevices += count;

    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPU partitions: %llu, OpenMP threads per rank: up to %d\n",
               ranks, static_cast<unsigned long long>(totalDevices), omp_get_max_threads());
        printf("Initializing data structures...\n");
    }

    std::vector<double> fullValues, vector(numRows);
    std::vector<index_t> fullColumns, fullOffsets;
    std::vector<index_t> rowStarts(static_cast<size_t>(ranks) + 1), nnzStarts(rowStarts.size());
    if (!rank) {
        fullValues.resize(nItems);
        fullColumns.resize(nItems);
        fullOffsets.resize(static_cast<size_t>(numRows) + 1);
        srand(1); // Restore the original default seed after MPI/CUDA initialization.
        fill(vector.data(), numRows, maxVal);
        fill(fullValues.data(), nItems, maxVal);
        initRandomMatrix(fullColumns.data(), fullOffsets.data(), nItems, numRows);
        uint64_t part = 0;
        for (int p = 0; p <= ranks; ++p) {
            rowStarts[p] = rowBoundary(fullOffsets.data(), numRows, part, totalDevices);
            nnzStarts[p] = fullOffsets[rowStarts[p]];
            if (p < ranks) part += deviceCounts[p];
        }
    }
    mpiCheck(MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Bcast(rowStarts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Bcast(nnzStarts.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    const index_t localRows = rowStarts[rank + 1] - rowStarts[rank];
    const index_t localNnz = nnzStarts[rank + 1] - nnzStarts[rank];
    std::vector<int> rowCounts(ranks), rowDisplacements(ranks);
    for (int p = 0; p < ranks; ++p) {
        rowCounts[p] = static_cast<int>(rowStarts[p + 1] - rowStarts[p]);
        rowDisplacements[p] = static_cast<int>(rowStarts[p]);
    }
    std::vector<double> values, output(localRows), reference;
    std::vector<index_t> columns, offsets;
    if (ranks == 1) {
        values = std::move(fullValues);
        columns = std::move(fullColumns);
        offsets = std::move(fullOffsets);
    } else {
        values.resize(localNnz);
        columns.resize(localNnz);
        offsets.resize(static_cast<size_t>(localRows) + 1);
        scatterNonzeros(fullValues, values, nnzStarts, MPI_DOUBLE, rank, ranks, 1);
        scatterNonzeros(fullColumns, columns, nnzStarts, MPI_UINT32_T, rank, ranks, 2);
        mpiCheck(MPI_Scatterv(fullOffsets.data(), rowCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                             offsets.data(), static_cast<int>(localRows), MPI_UINT32_T, 0, MPI_COMM_WORLD));
        offsets[localRows] = nnzStarts[rank + 1];
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i <= localRows; ++i) offsets[i] -= nnzStarts[rank];
        std::vector<double>().swap(fullValues);
        std::vector<index_t>().swap(fullColumns);
        std::vector<index_t>().swap(fullOffsets);
    }
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        reference.resize(localRows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vector.data(), localRows, reference.data());
    }

    std::vector<std::unique_ptr<GpuPartition>> partitions;
    for (int i = 0; i < localDevices; ++i)
        partitions.emplace_back(new GpuPartition(devices[i],
            rowBoundary(offsets.data(), localRows, i, localDevices),
            rowBoundary(offsets.data(), localRows, i + 1, localDevices)));
    forEachGpu(partitions, [&](GpuPartition& gpu) { gpu.prepare(values, columns, offsets, vector, iterations); });
    // All immutable inputs now reside on the devices; free the host CSR copy.
    std::vector<double>().swap(values);
    std::vector<double>().swap(vector);
    std::vector<index_t>().swap(columns);
    std::vector<index_t>().swap(offsets);

    if (!rank) printf("Computing SpMV...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    forEachGpu(partitions, [&](GpuPartition& gpu) { gpu.run(iterations); });
    // Submit all devices before waiting, even when OpenMP is limited to one
    // worker by the launcher or by OMP_NUM_THREADS.
    forEachGpu(partitions, [&](GpuPartition& gpu) { gpu.synchronize(); });
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    mpiCheck(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0.0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
    }
    if (validate || printResults)
        forEachGpu(partitions, [&](GpuPartition& gpu) { gpu.download(output); });
    if (printResults) {
        std::vector<double> fullOutput;
        if (!rank) fullOutput.resize(numRows);
        mpiCheck(MPI_Gatherv(output.data(), static_cast<int>(localRows), MPI_DOUBLE, fullOutput.data(),
                            rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (!rank) print_results(fullOutput, "OutputVector");
    }
    int valid = 1;
    if (validate) {
        if (!rank) printf("Validating result...\n");
        const int localValid = verifyResults(reference.data(), output.data(), localRows, rowStarts[rank]) ? 1 : 0;
        mpiCheck(MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI does not provide the required thread support");
        const int status = benchmark(argc, argv, rank, ranks);
        mpiCheck(MPI_Finalize());
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
