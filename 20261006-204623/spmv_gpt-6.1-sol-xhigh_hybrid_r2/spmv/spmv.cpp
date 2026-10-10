#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

void mpiCheck(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

int hostThreads(uint64_t work, uint64_t chunk = 65536) {
    return static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(
        omp_get_max_threads(), (work + chunk - 1) / chunk)));
}

// The original libc random stream must be preserved even when its generation
// is distributed. On glibc, rand() is the degree-31 additive generator used by
// random_r(). Its full state obeys s[k] = s[k-31] + s[k-3] modulo 2^32.
// Polynomial jump-ahead gives each OpenMP thread its exact stream position.
// Other C libraries retain the original serial initialization on rank zero.
class RandomSequence {
    using State = std::array<uint32_t, 31>;
    std::array<State, 64> powers{};
    State values{}, positions{};

    static State multiply(const State& a, const State& b) {
        std::array<uint32_t, 61> product{};
        for (int i = 0; i < 31; ++i) {
            for (int j = 0; j < 31; ++j) {
                product[i + j] += a[i] * b[j];
            }
        }
        // x^31 = x^28 + 1.
        for (int i = 60; i >= 31; --i) {
            product[i - 3] += product[i];
            product[i - 31] += product[i];
        }
        State result{};
        std::copy_n(product.begin(), 31, result.begin());
        return result;
    }

    static bool seed(unsigned seedValue, State& state) {
#if defined(__GLIBC__)
        std::array<int32_t, 32> storage{};
        struct random_data data{};
        if (initstate_r(seedValue, reinterpret_cast<char*>(storage.data()),
                        sizeof(storage), &data) != 0 ||
            data.rand_deg != 31 || data.rand_sep != 3 || RAND_MAX != 2147483647) {
            return false;
        }
        const int first = static_cast<int>(data.fptr - data.state);
        for (int i = 0; i < 31; ++i) {
            state[i] = static_cast<uint32_t>(data.state[(first + i) % 31]);
        }
        return true;
#else
        (void)seedValue;
        (void)state;
        return false;
#endif
    }

public:
    class Stream {
        State state;
        int first = 0;
    public:
        explicit Stream(const State& initial) : state(initial) {}
        uint32_t next() {
            int other = first + 28;
            if (other >= 31) other -= 31;
            const uint32_t value = state[first] + state[other];
            state[first] = value;
            if (++first == 31) first = 0;
            return value >> 1;
        }
    };

    bool compatible = false;

    RandomSequence() {
        if (!seed(1, values) || !seed(8675309, positions)) return;
        // Check the actual rand implementation, including interposed libraries.
        for (int which = 0; which < 2; ++which) {
            Stream stream(which == 0 ? values : positions);
            srand(which == 0 ? 1 : 8675309);
            for (int i = 0; i < 64; ++i) {
                if (stream.next() != static_cast<uint32_t>(rand())) return;
            }
        }
        powers[0][1] = 1;
        for (int i = 1; i < 64; ++i) powers[i] = multiply(powers[i - 1], powers[i - 1]);
        compatible = true;
    }

    Stream stream(bool matrixPositions, uint64_t offset) const {
        const State& initial = matrixPositions ? positions : values;
        if (offset == 0) return Stream(initial);
        State coefficients{};
        coefficients[0] = 1;
        for (int bit = 0; offset != 0; ++bit, offset >>= 1) {
            if (offset & 1) coefficients = multiply(coefficients, powers[bit]);
        }
        State advanced{};
        for (int i = 0; i < 31; ++i) {
            for (int j = 0; j < 31; ++j) advanced[i] += coefficients[j] * initial[j];
            const uint32_t carry = coefficients[30];
            for (int j = 30; j > 0; --j) coefficients[j] = coefficients[j - 1];
            coefficients[0] = carry;
            coefficients[28] += carry;
        }
        return Stream(advanced);
    }
};

void fill(double* data, index_t count, double maxValue) {
    for (index_t i = 0; i < count; ++i) {
        data[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void parallelFill(double* data, index_t count, double maxValue,
                  uint64_t offset, const RandomSequence& random) {
    #pragma omp parallel num_threads(hostThreads(count, 16384))
    {
        const uint64_t first = uint64_t(count) * omp_get_thread_num() / omp_get_num_threads();
        const uint64_t last = uint64_t(count) * (omp_get_thread_num() + 1) / omp_get_num_threads();
        auto stream = random.stream(false, offset + first);
        for (uint64_t i = first; i < last; ++i) {
            data[i] = maxValue * (stream.next() / (static_cast<double>(RAND_MAX) + 1.0));
        }
    }
}

// Replace the per-cell floating-point division with an integer comparison.
// Adjust the boundary using the original expression to preserve its rounding.
index_t randomThreshold(double probability) {
    index_t threshold = static_cast<index_t>(probability * RAND_MAX);
    while (static_cast<double>(threshold) / RAND_MAX > probability) --threshold;
    while (threshold < RAND_MAX && static_cast<double>(threshold + 1) / RAND_MAX <= probability) ++threshold;
    return threshold;
}

// Count Bernoulli successes independently. The prefix on rank zero then applies
// the original cap and forced final entries; a second, distributed pass writes
// the column indices. Both passes use precisely the original random sequence.
void countRows(index_t* counts, index_t firstRow, index_t rowCount,
               index_t dim, index_t nnz, const RandomSequence& random) {
    const uint64_t cells = uint64_t(dim) * dim;
    if (nnz == 0 || nnz == cells) {
        std::fill_n(counts, rowCount, nnz == 0 ? 0 : dim);
        return;
    }
    const index_t threshold = randomThreshold(static_cast<double>(nnz) / static_cast<double>(cells));
    #pragma omp parallel num_threads(hostThreads(uint64_t(rowCount) * dim))
    {
        const uint64_t first = uint64_t(rowCount) * omp_get_thread_num() / omp_get_num_threads();
        const uint64_t last = uint64_t(rowCount) * (omp_get_thread_num() + 1) / omp_get_num_threads();
        auto stream = random.stream(true, (firstRow + first) * dim);
        for (uint64_t row = first; row < last; ++row) {
            index_t count = 0;
            for (index_t col = 0; col < dim; ++col) {
                count += stream.next() <= threshold;
            }
            counts[row] = count;
        }
    }
}

void fillColumns(index_t* cols, const index_t* rows, index_t firstRow,
                 index_t rowCount, index_t dim, index_t nnz, index_t nnzBase,
                 const RandomSequence& random) {
    const uint64_t cells = uint64_t(dim) * dim;
    if (nnz == 0) return;
    const index_t threshold = randomThreshold(static_cast<double>(nnz) / static_cast<double>(cells));
    #pragma omp parallel num_threads(hostThreads(uint64_t(rowCount) * dim))
    {
        const uint64_t first = uint64_t(rowCount) * omp_get_thread_num() / omp_get_num_threads();
        const uint64_t last = uint64_t(rowCount) * (omp_get_thread_num() + 1) / omp_get_num_threads();
        auto stream = random.stream(true, (firstRow + first) * dim);
        for (uint64_t row = first; row < last; ++row) {
            index_t assigned = rows[row];
            if (assigned == nnz) break;
            for (index_t col = 0; col < dim; ++col) {
                const uint64_t remaining = cells - ((firstRow + row) * dim + col);
                const bool forced = remaining <= nnz - assigned;
                // Once forced, all subsequent entries are forced as well.
                // The random stream is no longer needed after that point.
                if (forced || stream.next() <= threshold) {
                    cols[assigned++ - nnzBase] = col;
                    if (assigned == nnz) break;
                }
            }
        }
    }
}

void initRandomMatrix(index_t* cols, index_t* rows, index_t nnz, index_t dim) {
    index_t assigned = 0;
    const uint64_t cells = uint64_t(dim) * dim;
    const double probability = cells == 0 ? 0.0 : static_cast<double>(nnz) / cells;
    srand(8675309);
    for (index_t row = 0; row < dim; ++row) {
        rows[row] = assigned;
        for (index_t col = 0; col < dim; ++col) {
            const uint64_t remaining = cells - (uint64_t(row) * dim + col);
            const double value = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < nnz && value <= probability) || remaining <= nnz - assigned) {
                cols[assigned++] = col;
            }
        }
    }
    rows[dim] = nnz;
}

void spmvCpu(const double* values, const index_t* cols, const index_t* rows,
             const double* vec, index_t rowCount, double* out) {
    [[maybe_unused]] const uint64_t work = rowCount == 0 ? 0 : uint64_t(rows[rowCount]) + rowCount;
    #pragma omp parallel for schedule(static) num_threads(hostThreads(work, 32768))
    for (index_t row = 0; row < rowCount; ++row) {
        double sum = 0.0;
        for (index_t item = rows[row]; item < rows[row + 1]; ++item) {
            sum += values[item] * vec[cols[item]];
        }
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result,
                   index_t count, index_t firstRow) {
    for (index_t i = 0; i < count; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (!std::isfinite(res) || error > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", firstRow + i, ref, res);
            return false;
        }
    }
    return true;
}

// Balance nonzero work plus one unit per row, so empty rows are distributed too.
std::vector<index_t> partitionRows(const index_t* rows, index_t count, int parts,
                                 const std::vector<int>* weights = nullptr) {
    std::vector<index_t> boundaries(static_cast<size_t>(parts) + 1, 0);
    boundaries.back() = count;
    const uint64_t total = uint64_t(rows[count]) + count;
    uint64_t totalWeight = parts, prefixWeight = 0;
    if (weights) {
        totalWeight = 0;
        for (int weight : *weights) totalWeight += weight;
    }
    for (int part = 1; part < parts; ++part) {
        prefixWeight += weights ? (*weights)[part - 1] : 1;
        const uint64_t target = total * prefixWeight / totalWeight;
        index_t low = 0, high = count;
        while (low < high) {
            const index_t mid = low + (high - low) / 2;
            if (uint64_t(rows[mid]) + mid < target) low = mid + 1;
            else high = mid;
        }
        if (low > 0 && target - (uint64_t(rows[low - 1]) + low - 1) <
                       uint64_t(rows[low]) + low - target) --low;
        boundaries[part] = low;
    }
    return boundaries;
}

// MPI-3 collective counts are int. Chunk the rare >INT_MAX nonzero partition
// rather than truncating its counts or requiring CUDA-aware MPI.
template<typename T>
void scatterNonzeros(const std::vector<T>& source, std::vector<T>& destination,
                     const std::vector<index_t>& offsets, MPI_Datatype type,
                     int rank, int ranks, int tag) {
    if (offsets.back() <= static_cast<index_t>(std::numeric_limits<int>::max())) {
        std::vector<int> counts(ranks), displacements(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(offsets[r + 1] - offsets[r]);
            displacements[r] = static_cast<int>(offsets[r]);
        }
        mpiCheck(MPI_Scatterv(source.data(), counts.data(), displacements.data(), type,
                             destination.data(), counts[rank], type, 0, MPI_COMM_WORLD));
    } else {
        for (int r = 0; r < ranks; ++r) {
            const uint64_t count = uint64_t(offsets[r + 1]) - offsets[r];
            for (uint64_t first = 0; first < count;) {
                const int chunk = static_cast<int>(std::min<uint64_t>(count - first, 1u << 26));
                if (rank == 0) {
                    if (r == 0) std::copy_n(source.data() + offsets[r] + first, chunk, destination.data() + first);
                    else mpiCheck(MPI_Send(source.data() + offsets[r] + first, chunk, type, r, tag, MPI_COMM_WORLD));
                } else if (rank == r) {
                    mpiCheck(MPI_Recv(destination.data() + first, chunk, type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                }
                first += chunk;
            }
        }
    }
}

template<int Width>
__global__ void csrVector(const double* __restrict__ values,
                          const index_t* __restrict__ cols,
                          const index_t* __restrict__ rows,
                          const double* __restrict__ vec,
                          double* __restrict__ out, index_t count, index_t base) {
    const unsigned lane = threadIdx.x & (Width - 1);
    const uint64_t stride = uint64_t(gridDim.x) * (blockDim.x / Width);
    for (uint64_t row = uint64_t(blockIdx.x) * (blockDim.x / Width) + threadIdx.x / Width;
         row < count; row += stride) {
        double sum = 0.0;
        const index_t end = rows[row + 1] - base;
        for (uint64_t item = uint64_t(rows[row] - base) + lane; item < end; item += Width) {
            sum += values[item] * __ldg(vec + cols[item]);
        }
        const unsigned mask = __activemask();
        #pragma unroll
        for (int offset = Width / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset, Width);
        }
        if (lane == 0) out[row] = sum;
    }
}

__global__ void csrLongRows(const double* __restrict__ values,
                            const index_t* __restrict__ cols,
                            const index_t* __restrict__ rows,
                            const double* __restrict__ vec,
                            double* __restrict__ out, index_t count, index_t base) {
    __shared__ double warpSums[8];
    for (uint64_t row = blockIdx.x; row < count; row += gridDim.x) {
        double sum = 0.0;
        const index_t end = rows[row + 1] - base;
        for (uint64_t item = uint64_t(rows[row] - base) + threadIdx.x; item < end; item += 256) {
            sum += values[item] * __ldg(vec + cols[item]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x < 32) {
            sum = threadIdx.x < 8 ? warpSums[threadIdx.x] : 0.0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffffu, sum, offset);
            if (threadIdx.x == 0) out[row] = sum;
        }
        __syncthreads();
    }
}

struct GpuPartition {
    int device = 0, width = 1, blocks = 1;
    index_t firstRow = 0, rowCount = 0, nnzBase = 0, nnz = 0;
    index_t batch = 0, repetitions = 0, remainder = 0;
    double *values = nullptr, *vec = nullptr, *out = nullptr;
    index_t *cols = nullptr, *rows = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph = nullptr, tail = nullptr;
};

void launchSpmv(const GpuPartition& gpu) {
    #define LAUNCH_VECTOR(W) csrVector<W><<<gpu.blocks, 256, 0, gpu.stream>>>(gpu.values, gpu.cols, gpu.rows, gpu.vec, gpu.out, gpu.rowCount, gpu.nnzBase)
    switch (gpu.width) {
        case 1: LAUNCH_VECTOR(1); break;
        case 2: LAUNCH_VECTOR(2); break;
        case 4: LAUNCH_VECTOR(4); break;
        case 8: LAUNCH_VECTOR(8); break;
        case 16: LAUNCH_VECTOR(16); break;
        case 32: LAUNCH_VECTOR(32); break;
        default: csrLongRows<<<gpu.blocks, 256, 0, gpu.stream>>>(gpu.values, gpu.cols, gpu.rows, gpu.vec, gpu.out, gpu.rowCount, gpu.nnzBase);
    }
    #undef LAUNCH_VECTOR
}

cudaGraphExec_t captureSpmv(const GpuPartition& gpu, index_t iterations) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaCheck(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeThreadLocal));
    for (index_t i = 0; i < iterations; ++i) launchSpmv(gpu);
    cudaCheck(cudaStreamEndCapture(gpu.stream, &graph));
    cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaGraphUpload(executable, gpu.stream));
    return executable;
}

template<typename T>
void deviceCopy(T*& device, const T* host, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(T)));
    cudaCheck(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
}

void prepareGpu(GpuPartition& gpu, const std::vector<double>& values,
                const std::vector<index_t>& cols, const std::vector<index_t>& rows,
                const std::vector<double>& vec, index_t iterations) {
    cudaCheck(cudaSetDevice(gpu.device));
    cudaCheck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    if (gpu.rowCount == 0) return;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, gpu.device));
    const double average = static_cast<double>(gpu.nnz) / gpu.rowCount;
    gpu.width = average >= 2048 ? 256 : average >= 32 ? 32 : average >= 16 ? 16 :
                average >= 8 ? 8 : average >= 4 ? 4 : average >= 2 ? 2 : 1;
    const unsigned rowsPerBlock = gpu.width == 256 ? 1 : 256 / gpu.width;
    gpu.blocks = static_cast<int>(std::min<uint64_t>(
        (uint64_t(gpu.rowCount) + rowsPerBlock - 1) / rowsPerBlock,
        uint64_t(properties.multiProcessorCount) * 8));
    if (gpu.nnz) {
        deviceCopy(gpu.values, values.data() + gpu.nnzBase, gpu.nnz, gpu.stream);
        deviceCopy(gpu.cols, cols.data() + gpu.nnzBase, gpu.nnz, gpu.stream);
    }
    deviceCopy(gpu.rows, rows.data() + gpu.firstRow, size_t(gpu.rowCount) + 1, gpu.stream);
    deviceCopy(gpu.vec, vec.data(), vec.size(), gpu.stream);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&gpu.out), size_t(gpu.rowCount) * sizeof(double)));
    if (iterations == 0) {
        cudaCheck(cudaMemsetAsync(gpu.out, 0, size_t(gpu.rowCount) * sizeof(double), gpu.stream));
    } else {
        // Warm up outside the measured interval. Graph batches amortize host
        // launch overhead without eliding any of the requested multiplications.
        launchSpmv(gpu);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaStreamSynchronize(gpu.stream));
        gpu.batch = std::min<index_t>(iterations, 64);
        gpu.repetitions = iterations / gpu.batch;
        gpu.remainder = iterations % gpu.batch;
        gpu.graph = captureSpmv(gpu, gpu.batch);
        if (gpu.remainder) gpu.tail = captureSpmv(gpu, gpu.remainder);
    }
    cudaCheck(cudaStreamSynchronize(gpu.stream));
}

void releaseGpu(GpuPartition& gpu, std::vector<double>& output) {
    if (gpu.rowCount) {
        cudaCheck(cudaMemcpyAsync(output.data() + gpu.firstRow, gpu.out,
                                  size_t(gpu.rowCount) * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        cudaCheck(cudaStreamSynchronize(gpu.stream));
    }
    if (gpu.graph) cudaCheck(cudaGraphExecDestroy(gpu.graph));
    if (gpu.tail) cudaCheck(cudaGraphExecDestroy(gpu.tail));
    cudaCheck(cudaFree(gpu.values));
    cudaCheck(cudaFree(gpu.cols));
    cudaCheck(cudaFree(gpu.rows));
    cudaCheck(cudaFree(gpu.vec));
    cudaCheck(cudaFree(gpu.out));
    cudaCheck(cudaStreamDestroy(gpu.stream));
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

index_t parseIndex(const char* value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || value[0] == '-' || end == value || *end || parsed > std::numeric_limits<index_t>::max()) {
        throw std::runtime_error("Invalid integer argument");
    }
    return static_cast<index_t>(parsed);
}

int benchmark(int argc, char** argv, int rank, int ranks) {
    index_t dim = 1024, sparsity = 10, iterations = 10;
    double maxValue = 1.0;
    bool validate = false, printResults = false;
    for (int arg = 1; arg < argc; ++arg) {
        if (!strcmp(argv[arg], "-n") && arg + 1 < argc) dim = parseIndex(argv[++arg]);
        else if (!strcmp(argv[arg], "-s") && arg + 1 < argc) sparsity = parseIndex(argv[++arg]);
        else if (!strcmp(argv[arg], "-i") && arg + 1 < argc) iterations = parseIndex(argv[++arg]);
        else if (!strcmp(argv[arg], "-m") && arg + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++arg];
            maxValue = strtod(value, &end);
            if (end == value || *end || !std::isfinite(maxValue)) throw std::runtime_error("Invalid maximum value");
        } else if (!strcmp(argv[arg], "-v")) validate = true;
        else if (!strcmp(argv[arg], "-r")) printResults = true;
        else if (!strcmp(argv[arg], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[arg]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (sparsity == 0) throw std::runtime_error("Sparsity must be positive");
    const uint64_t cells = uint64_t(dim) * dim;
    if (dim >= static_cast<index_t>(std::numeric_limits<int>::max()) ||
        cells / sparsity > std::numeric_limits<index_t>::max()) {
        throw std::runtime_error("Matrix exceeds the 32-bit CSR index range");
    }
    const index_t nnz = static_cast<index_t>(cells / sparsity);

    // A rank owns all visible GPUs congruent to its node-local rank. This also
    // works with launchers that expose a single assigned GPU per rank.
    MPI_Comm node = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, localRanks = 1, deviceCount = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    mpiCheck(MPI_Comm_size(node, &localRanks));
    mpiCheck(MPI_Comm_free(&node));
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) throw std::runtime_error("MPI/OpenMP/CUDA execution requires a visible CUDA GPU on every rank");
    std::vector<GpuPartition> gpus;
    if (localRanks <= deviceCount) {
        for (int device = localRank; device < deviceCount; device += localRanks) {
            GpuPartition gpu;
            gpu.device = device;
            gpus.push_back(gpu);
        }
    } else {
        GpuPartition gpu;
        gpu.device = localRank % deviceCount;
        gpus.push_back(gpu);
    }
    const int gpuCount = static_cast<int>(gpus.size());
    int allGpuWorkers = 0;
    std::vector<int> rankGpuCounts(ranks);
    mpiCheck(MPI_Allgather(&gpuCount, 1, MPI_INT, rankGpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int count : rankGpuCounts) allGpuWorkers += count;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", dim, dim);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nnz,
               cells == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nnz) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxValue, validate ? "enabled" : "disabled");
        printf("Parallel execution: %d MPI ranks, %d GPU workers, OpenMP host threads\n", ranks, allGpuWorkers);
        printf("Initializing data structures...\n");
    }

    RandomSequence random;
    int fastInitialization = random.compatible ? 1 : 0;
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &fastInitialization, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    std::vector<index_t> globalRows(rank == 0 ? size_t(dim) + 1 : 0);
    std::vector<double> globalValues, vec(dim);
    std::vector<index_t> globalCols;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        displacements[r] = static_cast<int>(uint64_t(dim) * r / ranks);
        counts[r] = static_cast<int>(uint64_t(dim) * (r + 1) / ranks) - displacements[r];
    }
    if (fastInitialization) {
        std::vector<index_t> rowCounts(counts[rank]);
        countRows(rowCounts.data(), displacements[rank], counts[rank], dim, nnz, random);
        mpiCheck(MPI_Gatherv(rowCounts.data(), counts[rank], MPI_UINT32_T,
                             globalRows.data(), counts.data(), displacements.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            uint64_t assigned = 0;
            for (index_t row = 0; row < dim; ++row) {
                const index_t successes = globalRows[row];
                globalRows[row] = static_cast<index_t>(assigned);
                const uint64_t remaining = cells - uint64_t(row + 1) * dim;
                const uint64_t forced = nnz > remaining ? nnz - remaining : 0;
                assigned = std::min<uint64_t>(nnz, std::max(assigned + successes, forced));
            }
            globalRows[dim] = nnz;
            parallelFill(vec.data(), dim, maxValue, 0, random);
        }
    } else if (rank == 0) {
        globalValues.resize(nnz);
        globalCols.resize(nnz);
        srand(1);
        fill(vec.data(), dim, maxValue);
        fill(globalValues.data(), nnz, maxValue);
        initRandomMatrix(globalCols.data(), globalRows.data(), nnz, dim);
    }
    mpiCheck(MPI_Bcast(vec.data(), static_cast<int>(dim), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    std::vector<index_t> rankRows(static_cast<size_t>(ranks) + 1);
    std::vector<index_t> rankItems(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        rankRows = partitionRows(globalRows.data(), dim, ranks, &rankGpuCounts);
        for (int r = 0; r <= ranks; ++r) rankItems[r] = globalRows[rankRows[r]];
    }
    mpiCheck(MPI_Bcast(rankRows.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Bcast(rankItems.data(), ranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    const index_t rowCount = rankRows[rank + 1] - rankRows[rank];
    const index_t itemCount = rankItems[rank + 1] - rankItems[rank];
    std::vector<index_t> rows(size_t(rowCount) + 1), cols(itemCount);
    std::vector<double> values(itemCount), output(rowCount, 0.0), reference;
    for (int r = 0; r < ranks; ++r) {
        counts[r] = static_cast<int>(rankRows[r + 1] - rankRows[r]);
        displacements[r] = static_cast<int>(rankRows[r]);
    }
    // Send each delimiter once; the final delimiter is already in rankItems.
    mpiCheck(MPI_Scatterv(globalRows.data(), counts.data(), displacements.data(), MPI_UINT32_T,
                         rows.data(), static_cast<int>(rowCount), MPI_UINT32_T, 0, MPI_COMM_WORLD));
    rows[rowCount] = rankItems[rank + 1];
    if (fastInitialization) {
        parallelFill(values.data(), itemCount, maxValue, uint64_t(dim) + rankItems[rank], random);
        fillColumns(cols.data(), rows.data(), rankRows[rank], rowCount, dim, nnz, rankItems[rank], random);
    } else {
        scatterNonzeros(globalValues, values, rankItems, MPI_DOUBLE, rank, ranks, 100);
        scatterNonzeros(globalCols, cols, rankItems, MPI_UINT32_T, rank, ranks, 101);
    }
    #pragma omp parallel for schedule(static) num_threads(hostThreads(uint64_t(rowCount) + 1, 16384))
    for (uint64_t i = 0; i <= rowCount; ++i) rows[i] -= rankItems[rank];
    std::vector<index_t>().swap(globalRows);
    std::vector<index_t>().swap(globalCols);
    std::vector<double>().swap(globalValues);
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(rowCount);
        spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), rowCount, reference.data());
    }
    const auto gpuRows = partitionRows(rows.data(), rowCount, gpuCount);
    for (int g = 0; g < gpuCount; ++g) {
        gpus[g].firstRow = gpuRows[g];
        gpus[g].rowCount = gpuRows[g + 1] - gpuRows[g];
        gpus[g].nnzBase = rows[gpuRows[g]];
        gpus[g].nnz = rows[gpuRows[g + 1]] - gpus[g].nnzBase;
    }
    std::vector<std::string> errors(gpuCount);
    double start = 0.0, elapsed = 0.0;
    auto checkWorkers = [&]() {
        for (int g = 0; g < gpuCount; ++g) {
            if (!errors[g].empty()) {
                fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, gpus[g].device, errors[g].c_str());
                MPI_Abort(MPI_COMM_WORLD, 1);
                std::abort();
            }
        }
    };
    // One persistent host team drives all local GPUs. MPI stays on the main
    // thread (MPI_THREAD_FUNNELED); buffers and graphs remain resident throughout.
    #pragma omp parallel num_threads(gpuCount)
    {
        #pragma omp for schedule(static, 1)
        for (int g = 0; g < gpuCount; ++g) {
            try { prepareGpu(gpus[g], values, cols, rows, vec, iterations); }
            catch (const std::exception& e) { errors[g] = e.what(); }
        }
        #pragma omp master
        {
            checkWorkers();
            if (rank == 0) printf("Computing SpMV...\n");
            mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
            start = MPI_Wtime();
        }
        #pragma omp barrier
        #pragma omp for schedule(static, 1)
        for (int g = 0; g < gpuCount; ++g) {
            try {
                auto& gpu = gpus[g];
                cudaCheck(cudaSetDevice(gpu.device));
                for (index_t batch = 0; batch < gpu.repetitions; ++batch) cudaCheck(cudaGraphLaunch(gpu.graph, gpu.stream));
                if (gpu.remainder) cudaCheck(cudaGraphLaunch(gpu.tail, gpu.stream));
                cudaCheck(cudaStreamSynchronize(gpu.stream));
            } catch (const std::exception& e) { errors[g] = e.what(); }
        }
        #pragma omp master
        {
            const double localElapsed = MPI_Wtime() - start;
            checkWorkers();
            mpiCheck(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        }
        #pragma omp barrier
        #pragma omp for schedule(static, 1)
        for (int g = 0; g < gpuCount; ++g) {
            try {
                cudaCheck(cudaSetDevice(gpus[g].device));
                releaseGpu(gpus[g], output);
            } catch (const std::exception& e) { errors[g] = e.what(); }
        }
    }
    checkWorkers();
    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double gflops = elapsed > 0.0 ? (2.0 * nnz * iterations) / elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", iterations ? milliseconds / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        std::vector<double> allOutput(rank == 0 ? dim : 0);
        for (int r = 0; r < ranks; ++r) counts[r] = static_cast<int>(rankRows[r + 1] - rankRows[r]);
        mpiCheck(MPI_Gatherv(output.data(), static_cast<int>(rowCount), MPI_DOUBLE,
                             allOutput.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) print_results(allOutput, "OutputVector");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = verifyResults(reference.data(), output.data(), rowCount, rankRows[rank]) ? 1 : 0;
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int status = 1;
    try {
        mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
