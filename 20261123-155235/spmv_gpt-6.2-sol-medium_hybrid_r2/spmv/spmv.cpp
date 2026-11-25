#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int ROOT = 0;
constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

#if defined(__GLIBC__)
// glibc rand() returns the upper 31 bits of a lagged sum modulo 2^32.
// Jump ahead with x^31 = x^28 + 1 so each worker can start at its own
// position in the original stream without generating the preceding draws.
class RandomStream {
    using State = std::array<uint32_t, 31>;
    using Product = std::array<uint32_t, 61>;

    static State multiply(const State& a, const State& b) {
        Product product{};
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                product[i + j] += a[i] * b[j];
            }
        }
        for (size_t i = product.size() - 1; i >= a.size(); --i) {
            product[i - 3] += product[i];
            product[i - 31] += product[i];
        }
        State result;
        std::copy_n(product.begin(), result.size(), result.begin());
        return result;
    }

    static const std::array<State, 64>& powers() {
        static const auto result = [] {
            std::array<State, 64> table{};
            table[0][1] = 1;
            for (size_t i = 1; i < table.size(); ++i) {
                table[i] = multiply(table[i - 1], table[i - 1]);
            }
            return table;
        }();
        return result;
    }

    State state;
    size_t head = 0;
    size_t lag = 28;

public:
    RandomStream(unsigned seed, offset_t position) {
        State seedState;
        seedState[0] = seed;
        for (size_t i = 1; i < seedState.size(); ++i) {
            seedState[i] = (16807ull * seedState[i - 1]) % 2147483647;
        }

        size_t front = 3;
        size_t rear = 0;
        for (int i = 0; i < 341; ++i) {
            seedState[front] += seedState[rear];
            if (i >= 310) {
                state[i - 310] = seedState[front];
            }
            if (++front == seedState.size()) front = 0;
            if (++rear == seedState.size()) rear = 0;
        }

        if (position == 0) return;
        State coefficient{};
        coefficient[0] = 1;
        for (size_t bit = 0; position != 0; ++bit, position >>= 1) {
            if (position & 1) coefficient = multiply(coefficient, powers()[bit]);
        }

        const State initial = state;
        for (auto& word : state) {
            word = 0;
            for (size_t i = 0; i < initial.size(); ++i) {
                word += coefficient[i] * initial[i];
            }
            const auto high = coefficient.back();
            for (size_t i = coefficient.size() - 1; i > 0; --i) {
                coefficient[i] = coefficient[i - 1];
            }
            coefficient[28] += high;
            coefficient[0] = high;
        }
    }

    unsigned next() {
        const auto word = state[head];
        state[head] += state[lag];
        if (++head == state.size()) head = 0;
        if (++lag == state.size()) lag = 0;
        return word >> 1;
    }
};
#else
// Other C libraries can use a different rand() sequence.
class RandomStream {
    static omp_lock_t& lock() {
        struct Lock {
            omp_lock_t value;
            Lock() { omp_init_lock(&value); }
            ~Lock() { omp_destroy_lock(&value); }
        };
        static Lock mutex;
        return mutex.value;
    }

public:
    RandomStream(unsigned seed, offset_t position) {
        omp_set_lock(&lock());
        srand(seed);
        while (position-- > 0) rand();
    }
    ~RandomStream() { omp_unset_lock(&lock()); }
    unsigned next() { return rand(); }
};
#endif

void fill(double* values, offset_t count, offset_t start, double maxValue) {
    if (count == 0) return;
    RandomStream random(1, start);
    for (offset_t i = 0; i < count; ++i) {
        values[i] = maxValue * (random.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

offset_t partition(offset_t size, int part, int parts) {
    return size * part / parts;
}

int workerCount(index_t rows, offset_t work) {
    return static_cast<int>(std::min<offset_t>(
        omp_get_max_threads(), std::min<offset_t>(std::max(1u, rows), 1 + work / 16384)));
}

struct LocalMatrix {
    std::vector<offset_t> offsets;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
    std::unique_ptr<double[]> input;
};

struct RowChunk {
    std::vector<index_t> candidates;
    offset_t precedingCandidates = 0;
};

offset_t accepted(offset_t candidates, offset_t cellsLeft, offset_t nnz) {
    // The original scan stops at nnz successes, or fills every remaining
    // cell when the remaining capacity equals the number still needed.
    return std::max(std::min(candidates, nnz), nnz - std::min(nnz, cellsLeft));
}

LocalMatrix makeMatrix(index_t dim, offset_t nnz, index_t firstRow, index_t rows,
                       double maxValue, int rank) {
    const offset_t cells = static_cast<offset_t>(dim) * dim;
    const int workers = workerCount(rows, offset_t{rows} * dim);
    std::vector<RowChunk> chunks(workers);
    std::vector<offset_t> candidateEnds(rows);
    offset_t rankCandidates = 0;

    if (nnz > 0 && nnz < cells) {
        const double probability = static_cast<double>(nnz) / static_cast<double>(cells);
        unsigned cutoff = static_cast<unsigned>(probability * RAND_MAX);
        // Match the original floating-point comparison at a rounding boundary.
        if (cutoff < RAND_MAX && static_cast<double>(cutoff + 1) / RAND_MAX <= probability) ++cutoff;
        if (static_cast<double>(cutoff) / RAND_MAX > probability) --cutoff;

        #pragma omp parallel for num_threads(workers) schedule(static) reduction(+:rankCandidates)
        for (int worker = 0; worker < workers; ++worker) {
            const auto begin = partition(rows, worker, workers);
            const auto end = partition(rows, worker + 1, workers);
            if (begin == end) continue;
            auto& candidates = chunks[worker].candidates;
            const auto chunkCells = (end - begin) * dim;
            const double expected = chunkCells * probability;
            candidates.reserve(std::min(chunkCells, static_cast<offset_t>(expected + 8 * std::sqrt(expected) + 16)));
            RandomStream random(8675309, (firstRow + begin) * dim);
            for (auto row = begin; row < end; ++row) {
                for (index_t col = 0; col < dim; ++col) {
                    if (random.next() <= cutoff) candidates.push_back(col);
                }
                candidateEnds[row] = candidates.size();
            }
            rankCandidates += candidates.size();
        }
    }

    offset_t prefix = 0;
    if (nnz > 0 && nnz < cells) {
        MPI_Exscan(&rankCandidates, &prefix, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (rank == ROOT) prefix = 0;
    }
    const auto globalBegin = accepted(prefix, cells - offset_t{firstRow} * dim, nnz);
    const auto globalEnd = accepted(prefix + rankCandidates, cells - (offset_t{firstRow} + rows) * dim, nnz);
    const auto localNnz = globalEnd - globalBegin;
    for (auto& chunk : chunks) {
        chunk.precedingCandidates = prefix;
        prefix += chunk.candidates.size();
    }

    LocalMatrix matrix;
    matrix.offsets.resize(static_cast<size_t>(rows) + 1);
    matrix.offsets[rows] = localNnz;
    if (localNnz != 0) {
        matrix.columns.reset(new index_t[localNnz]);
        matrix.values.reset(new double[localNnz]);
    }
    if (rows != 0) matrix.input.reset(new double[dim]);

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const auto begin = partition(rows, worker, workers);
        const auto end = partition(rows, worker + 1, workers);
        if (begin == end) continue;
        auto& chunk = chunks[worker];
        const auto vectorBegin = partition(dim, worker, workers);
        const auto vectorEnd = partition(dim, worker + 1, workers);
        fill(matrix.input.get() + vectorBegin, vectorEnd - vectorBegin, vectorBegin, maxValue);

        auto cell = (firstRow + begin) * dim;
        auto current = accepted(chunk.precedingCandidates, cells - cell, nnz);
        const auto valueBegin = current;
        offset_t rawBegin = 0;
        for (auto row = begin; row < end; ++row, cell += dim) {
            const auto rawEnd = candidateEnds[row];
            const auto rawGlobalEnd = chunk.precedingCandidates + rawEnd;
            const auto next = accepted(rawGlobalEnd, cells - cell - dim, nnz);
            matrix.offsets[row] = current - globalBegin;
            if (next > current) {
                auto* destination = matrix.columns.get() + matrix.offsets[row];
                if (next == std::min(rawGlobalEnd, nnz)) {
                    std::copy_n(chunk.candidates.begin() + rawBegin, next - current, destination);
                } else {
                    const auto failures = cell - (chunk.precedingCandidates + rawBegin);
                    const auto budget = cells - nnz;
                    const auto failuresLeft = budget - std::min(budget, failures);
                    offset_t kept = 0;
                    while (rawBegin + kept < rawEnd &&
                           chunk.candidates[rawBegin + kept] < failuresLeft + kept) {
                        destination[kept] = chunk.candidates[rawBegin + kept];
                        ++kept;
                    }
                    // Once the failure budget runs out, the row ends in consecutive columns.
                    for (auto col = failuresLeft + kept; col < dim; ++col) {
                        destination[kept++] = static_cast<index_t>(col);
                    }
                }
            }
            rawBegin = rawEnd;
            current = next;
        }
        if (current > valueBegin) {
            fill(matrix.values.get() + (valueBegin - globalBegin), current - valueBegin,
                 dim + valueBegin, maxValue);
        }
        std::vector<index_t>().swap(chunk.candidates);
    }
    return matrix;
}

struct PackedRows {
    std::vector<offset_t> starts;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
};

PackedRows packOrderedRows(const LocalMatrix& matrix, index_t rows) {
    const auto nnz = matrix.offsets.back();
    if (rows < WARP_SIZE || nnz < 4 * offset_t{rows}) return {};

    // Interleave each warp's rows for coalesced loads without changing the
    // order of additions within any row.
    const auto slices = (rows + WARP_SIZE - 1) / WARP_SIZE;
    PackedRows packed;
    packed.starts.resize(slices + 1);
    for (index_t slice = 0; slice < slices; ++slice) {
        const auto first = slice * WARP_SIZE;
        const auto end = std::min(rows, first + WARP_SIZE);
        offset_t width = 0;
        for (auto row = first; row < end; ++row) {
            width = std::max(width, matrix.offsets[row + 1] - matrix.offsets[row]);
        }
        packed.starts[slice + 1] = packed.starts[slice] + width * (end - first);
    }
    const auto slots = packed.starts.back();
    if (slots - nnz > nnz / 4) return {};
    packed.columns.reset(new index_t[slots]);
    packed.values.reset(new double[slots]);

    const int workers = workerCount(slices, nnz);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (index_t slice = 0; slice < slices; ++slice) {
        const auto first = slice * WARP_SIZE;
        const auto stride = std::min<index_t>(WARP_SIZE, rows - first);
        const auto width = (packed.starts[slice + 1] - packed.starts[slice]) / stride;
        auto destination = packed.starts[slice];
        for (offset_t slot = 0; slot < width; ++slot) {
            for (index_t lane = 0; lane < stride; ++lane, ++destination) {
                const auto row = first + lane;
                const auto source = matrix.offsets[row] + slot;
                if (source < matrix.offsets[row + 1]) {
                    packed.columns[destination] = matrix.columns[source];
                    packed.values[destination] = matrix.values[source];
                } else {
                    packed.columns[destination] = 0;
                    packed.values[destination] = 0.0;
                }
            }
        }
    }
    return packed;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
    T* pointer = nullptr;

public:
    explicit DeviceBuffer(offset_t count) {
        if (count != 0) checkCuda(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return pointer; }

    void upload(const T* source, offset_t count, cudaStream_t stream) const {
        if (count != 0) {
            checkCuda(cudaMemcpyAsync(pointer, source, count * sizeof(T), cudaMemcpyHostToDevice, stream),
                      "uploading matrix data");
        }
    }
};

class Stream {
    cudaStream_t stream{};

public:
    Stream() { checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating CUDA stream"); }
    ~Stream() { cudaStreamDestroy(stream); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    operator cudaStream_t() const { return stream; }
    void synchronize() const { checkCuda(cudaStreamSynchronize(stream), "synchronizing CUDA stream"); }
};

template <int Width, bool Ordered>
__global__ void spmvKernel(const offset_t* __restrict__ offsets, const index_t* __restrict__ columns,
                           const double* __restrict__ values, const double* __restrict__ input,
                           index_t rows, double* __restrict__ output) {
    const auto row = blockIdx.x * (blockDim.x / Width) + threadIdx.x / Width;
    const auto lane = threadIdx.x % Width;
    double sum = 0.0;
    if (row < rows) {
        const auto end = offsets[row + 1];
        for (auto j = offsets[row] + lane; j < end; j += Width) {
            if constexpr (Ordered) {
                sum = __dadd_rn(sum, __dmul_rn(values[j], input[columns[j]]));
            } else {
                sum = __fma_rn(values[j], input[columns[j]], sum);
            }
        }
    }
    for (int step = Width / 2; step > 0; step /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, step, Width);
    }
    if (row < rows && lane == 0) output[row] = sum;
}

__global__ void orderedSpmvKernel(const offset_t* __restrict__ offsets, const offset_t* __restrict__ starts,
                                  const index_t* __restrict__ columns, const double* __restrict__ values,
                                  const double* __restrict__ input, index_t rows, double* __restrict__ output) {
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const auto slice = row / WARP_SIZE;
    const auto lane = row % WARP_SIZE;
    const auto stride = min(static_cast<index_t>(WARP_SIZE), rows - slice * WARP_SIZE);
    const auto length = offsets[row + 1] - offsets[row];
    auto position = starts[slice] + lane;
    double sum = 0.0;
    for (offset_t j = 0; j < length; ++j, position += stride) {
        sum = __dadd_rn(sum, __dmul_rn(values[position], input[columns[position]]));
    }
    output[row] = sum;
}

class DeviceMatrix {
    Stream stream;
    DeviceBuffer<offset_t> offsets;
    DeviceBuffer<offset_t> sliceStarts;
    DeviceBuffer<index_t> columns;
    DeviceBuffer<double> values;
    DeviceBuffer<double> input;
    DeviceBuffer<double> output;
    index_t rows;
    int width;
    bool ordered;
    cudaGraphExec_t batch = nullptr;
    index_t batchIterations = 0;

    template <int Width, bool Ordered = false>
    void launch() const {
        constexpr int threads = Ordered ? WARP_SIZE : BLOCK_SIZE;
        constexpr int rowsPerBlock = threads / Width;
        const auto blocks = (offset_t{rows} + rowsPerBlock - 1) / rowsPerBlock;
        spmvKernel<Width, Ordered><<<blocks, threads, 0, stream>>>(
            offsets.get(), columns.get(), values.get(), input.get(), rows, output.get());
    }

    void multiply() const {
        // The result hash requires the original per-row floating-point order.
        if (ordered) {
            if (sliceStarts.get()) {
                const auto blocks = (rows + WARP_SIZE - 1) / WARP_SIZE;
                orderedSpmvKernel<<<blocks, WARP_SIZE, 0, stream>>>(
                    offsets.get(), sliceStarts.get(), columns.get(), values.get(), input.get(), rows, output.get());
            } else {
                launch<1, true>();
            }
            return;
        }
        switch (width) {
            case 1: launch<1>(); break;
            case 4: launch<4>(); break;
            case 8: launch<8>(); break;
            case 16: launch<16>(); break;
            case WARP_SIZE: launch<WARP_SIZE>(); break;
        }
    }

    void synchronize() const {
        checkCuda(cudaGetLastError(), "launching SpMV kernel");
        stream.synchronize();
    }

public:
    DeviceMatrix(const LocalMatrix& matrix, const PackedRows& packed, index_t dim, index_t localRows,
                 bool printResults)
        : offsets(matrix.offsets.size()), sliceStarts(packed.starts.size()),
          columns(packed.starts.empty() ? matrix.offsets.back() : packed.starts.back()),
          values(packed.starts.empty() ? matrix.offsets.back() : packed.starts.back()),
          input(dim), output(localRows), rows(localRows), ordered(printResults) {
        const double mean = static_cast<double>(matrix.offsets.back()) / rows;
        width = mean < 4 ? 1 : mean < 16 ? 4 : mean < 48 ? 8 : mean < 128 ? 16 : WARP_SIZE;
        offsets.upload(matrix.offsets.data(), matrix.offsets.size(), stream);
        sliceStarts.upload(packed.starts.data(), packed.starts.size(), stream);
        if (packed.starts.empty()) {
            columns.upload(matrix.columns.get(), matrix.offsets.back(), stream);
            values.upload(matrix.values.get(), matrix.offsets.back(), stream);
        } else {
            columns.upload(packed.columns.get(), packed.starts.back(), stream);
            values.upload(packed.values.get(), packed.starts.back(), stream);
        }
        input.upload(matrix.input.get(), dim, stream);
        stream.synchronize();
    }

    ~DeviceMatrix() { if (batch) cudaGraphExecDestroy(batch); }

    void prepare(index_t iterations, offset_t nnz) {
        if (iterations == 0) return;
        multiply();
        synchronize();
        // Batch short kernels to reduce host launch overhead.
        if (iterations < 8 || nnz > 524288) return;

        batchIterations = std::min<index_t>(iterations, 32);
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capturing SpMV batch");
        for (index_t i = 0; i < batchIterations; ++i) multiply();
        cudaGraph_t graph;
        checkCuda(cudaStreamEndCapture(stream, &graph), "ending SpMV batch capture");
        const auto status = cudaGraphInstantiate(&batch, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);
        checkCuda(status, "instantiating SpMV batch");
        checkCuda(cudaGraphUpload(batch, stream), "uploading SpMV batch");
        synchronize();
    }

    void execute(index_t iterations) const {
        if (iterations == 0) return;
        if (batch) {
            const auto batches = iterations / batchIterations;
            for (index_t i = 0; i < batches; ++i) {
                checkCuda(cudaGraphLaunch(batch, stream), "launching SpMV batch");
            }
            iterations %= batchIterations;
        }
        for (index_t i = 0; i < iterations; ++i) multiply();
        synchronize();
    }

    void download(double* destination) const {
        checkCuda(cudaMemcpyAsync(destination, output.get(), rows * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "downloading the result");
        stream.synchronize();
    }
};

std::vector<double> referenceSpmv(const LocalMatrix& matrix, index_t rows) {
    std::vector<double> reference(rows);
    const int workers = workerCount(rows, matrix.offsets.back());
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (auto j = matrix.offsets[row]; j < matrix.offsets[row + 1]; ++j) {
            sum += matrix.values[j] * matrix.input[matrix.columns[j]];
        }
        reference[row] = sum;
    }
    return reference;
}

bool validateOutput(const std::vector<double>& reference, const std::vector<double>& result,
                    index_t firstRow, index_t dim, int rank, int ranks) {
    index_t failure = dim;
    #pragma omp parallel for schedule(static) reduction(min:failure) if(result.size() >= 16384)
    for (size_t row = 0; row < result.size(); ++row) {
        const double ref = reference[row];
        const double res = result[row];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) failure = std::min(failure, firstRow + static_cast<index_t>(row));
    }

    index_t firstFailure;
    MPI_Allreduce(&failure, &firstFailure, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (firstFailure == dim) return true;

    const int owner = static_cast<int>(((offset_t{firstFailure} + 1) * ranks - 1) / dim);
    double pair[2];
    if (rank == owner) {
        pair[0] = reference[firstFailure - firstRow];
        pair[1] = result[firstFailure - firstRow];
    }
    MPI_Bcast(pair, 2, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    if (rank == ROOT) {
        printf("Validation failed at index %u: reference %.10e, got %.10e", firstFailure, pair[0], pair[1]);
        if (std::abs(pair[0]) >= 1e-10) {
            printf(" (rel error: %.10e)", std::abs((pair[1] - pair[0]) / pair[0]));
        }
        printf("\n");
    }
    return false;
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

bool parseUnsigned(const char* text, index_t& result) {
    char* end;
    errno = 0;
    const auto value = strtoull(text, &end, 10);
    if (text == end || *end != '\0' || errno == ERANGE || value > std::numeric_limits<index_t>::max()) {
        return false;
    }
    result = static_cast<index_t>(value);
    return true;
}

int run(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        index_t* number = nullptr;
        if (strcmp(option, "-n") == 0) {
            number = &numRows;
        } else if (strcmp(option, "-s") == 0) {
            number = &sparsity;
        } else if (strcmp(option, "-i") == 0) {
            number = &iterations;
        } else if (strcmp(option, "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else if (strcmp(option, "-v") == 0) {
            validate = true;
        } else if (strcmp(option, "-r") == 0) {
            printResults = true;
        } else if (strcmp(option, "-h") == 0) {
            if (rank == ROOT) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == ROOT) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
        if (number && (i + 1 == argc || !parseUnsigned(argv[++i], *number))) {
            if (rank == ROOT) fprintf(stderr, "Invalid integer argument for %s\n", option);
            return 1;
        }
    }

    if (sparsity == 0 || numRows > std::numeric_limits<int>::max()) {
        if (rank == ROOT) fprintf(stderr, "Sparsity must be positive and the row count must fit in an MPI int.\n");
        return 1;
    }

    const offset_t cells = offset_t{numRows} * numRows;
    const offset_t nnz = cells / sparsity;
    const auto firstRow = static_cast<index_t>(partition(numRows, rank, ranks));
    const auto rows = static_cast<index_t>(partition(numRows, rank + 1, ranks) - firstRow);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    if (rows != 0) {
        int devices;
        checkCuda(cudaGetDeviceCount(&devices), "querying CUDA devices");
        if (devices == 0) throw std::runtime_error("no CUDA device is visible");
        checkCuda(cudaSetDevice(localRank % devices), "selecting CUDA device");
    }

    if (rank == ROOT) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nnz,
               cells == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nnz) / cells));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    auto matrix = makeMatrix(numRows, nnz, firstRow, rows, maxVal, rank);
    std::vector<double> reference;
    if (validate) {
        if (rank == ROOT) printf("Computing reference solution...\n");
        reference = referenceSpmv(matrix, rows);
    }

    std::unique_ptr<DeviceMatrix> gpu;
    if (rows != 0) {
        const auto packed = printResults && iterations != 0 ? packOrderedRows(matrix, rows) : PackedRows{};
        gpu = std::make_unique<DeviceMatrix>(matrix, packed, numRows, rows, printResults);
        gpu->prepare(iterations, matrix.offsets.back());
    }
    matrix = {};

    if (rank == ROOT) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (gpu) gpu->execute(iterations);
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        const double average = iterations == 0 ? 0.0 : 1000.0 * duration / iterations;
        const double gflops = duration == 0.0 ? 0.0 : (2.0 * nnz * iterations) / duration / 1e9;
        printf("Computation time: %.3f ms\n", 1000.0 * duration);
        printf("Average time per iteration: %.3f ms\n", average);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> localResult;
    if (printResults || validate) {
        localResult.resize(rows);
        if (gpu && iterations != 0) gpu->download(localResult.data());
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> result;
        if (rank == ROOT) {
            result.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int i = 0; i < ranks; ++i) {
                displacements[i] = partition(numRows, i, ranks);
                counts[i] = partition(numRows, i + 1, ranks) - displacements[i];
            }
        }
        MPI_Gatherv(localResult.data(), rows, MPI_DOUBLE, result.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if (rank == ROOT) print_results(result, "OutputVector");
    }

    if (validate) {
        if (rank == ROOT) printf("Validating result...\n");
        const bool valid = validateOutput(reference, localResult, firstRow, numRows, rank, ranks);
        if (rank == ROOT) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == ROOT) fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int status;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
