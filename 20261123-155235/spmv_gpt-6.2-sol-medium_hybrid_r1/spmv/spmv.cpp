#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr uint32_t RANDOM_MAX = 0x7fffffff;
constexpr int BLOCK_THREADS = 256;

// The original benchmark uses the 31-word glibc rand() sequence. Jumping
// directly to a stream position lets every rank and thread reproduce its
// part of that sequence without generating the preceding draws.
class BenchmarkRandom {
public:
    explicit BenchmarkRandom(uint32_t seed, offset_t skip = 0) {
        State seedState;
        seedState[0] = seed;
        for (int i = 1; i < degree; ++i) {
            seedState[i] = (16807ull * seedState[i - 1]) % RANDOM_MAX;
        }
        std::rotate_copy(seedState.begin(), seedState.begin() + 3, seedState.end(), state.begin());
        for (int i = 0; i < 10 * degree; ++i) {
            next();
        }

        if (skip != 0) {
            // Exponentiation modulo x^31 - x^28 - 1 in arithmetic modulo 2^32.
            offset_t highestBit = 1;
            while (highestBit <= skip / 2) {
                highestBit <<= 1;
            }
            State coefficients{};
            coefficients[0] = 1;
            for (offset_t bit = highestBit; bit != 0; bit >>= 1) {
                coefficients = square(coefficients);
                if (skip & bit) {
                    multiplyByX(coefficients);
                }
            }

            State advanced;
            for (auto& word : advanced) {
                word = 0;
                for (int i = 0; i < degree; ++i) {
                    word += coefficients[i] * state[i];
                }
                multiplyByX(coefficients);
            }
            state = advanced;
        }
    }

    uint32_t next() {
        const uint32_t value = state[oldest] + state[lagged];
        state[oldest] = value;
        if (++oldest == degree) {
            oldest = 0;
        }
        if (++lagged == degree) {
            lagged = 0;
        }
        return value >> 1;
    }

    double scaled(double maxValue) {
        return maxValue * (next() / (static_cast<double>(RANDOM_MAX) + 1.0));
    }

private:
    static constexpr int degree = 31;
    static constexpr int lag = 3;
    using State = std::array<uint32_t, degree>;

    static void multiplyByX(State& coefficients) {
        const uint32_t leading = coefficients.back();
        std::move_backward(coefficients.begin(), coefficients.end() - 1, coefficients.end());
        coefficients[degree - lag] += leading;
        coefficients[0] = leading;
    }

    static State square(const State& coefficients) {
        std::array<uint32_t, 2 * degree - 1> product{};
        for (int i = 0; i < degree; ++i) {
            for (int j = 0; j < degree; ++j) {
                product[i + j] += coefficients[i] * coefficients[j];
            }
        }
        for (int i = 2 * degree - 2; i >= degree; --i) {
            product[i - lag] += product[i];
            product[i - degree] += product[i];
        }
        State reduced;
        std::copy_n(product.begin(), degree, reduced.begin());
        return reduced;
    }

    State state;
    int oldest = 0;
    int lagged = degree - lag;
};

[[noreturn]] void abortBenchmark(int rank, const char* message) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }

    void allocate(size_t count, int rank) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc", rank);
        }
    }

    void upload(const T* source, size_t count, int rank) {
        allocate(count, rank);
        if (count != 0) {
            checkCuda(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice),
                      "cudaMemcpy to device", rank);
        }
    }

    T* data = nullptr;
};

struct RowChunk {
    index_t begin;
    index_t end;
    offset_t hits = 0;
    offset_t assigned = 0;
};

index_t rowBoundary(index_t rows, int partition, int partitions) {
    return static_cast<offset_t>(rows) * partition / partitions;
}

offset_t assignedAt(offset_t hits, offset_t position, offset_t entries, offset_t nonzeros) {
    // Once the remaining slots are all needed, the original fills every
    // remaining slot. Before that point it takes the first nonzeros random hits.
    const offset_t remaining = entries - position;
    const offset_t forced = nonzeros > remaining ? nonzeros - remaining : 0;
    return std::min(nonzeros, std::max(hits, forced));
}

uint32_t randomThreshold(double probability) {
    uint32_t threshold = static_cast<uint32_t>(probability * RANDOM_MAX);
    // Match the original floating-point comparison at the integer boundary.
    if (threshold < RANDOM_MAX &&
        static_cast<double>(threshold + 1) / RANDOM_MAX <= probability) {
        ++threshold;
    }
    if (static_cast<double>(threshold) / RANDOM_MAX > probability) {
        --threshold;
    }
    return threshold;
}

struct LocalMatrix {
    std::unique_ptr<offset_t[]> offsets;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
    offset_t nonzeros;
};

LocalMatrix generateMatrix(index_t dimension, index_t firstRow, index_t rows, offset_t nonzeros,
                           double maxValue, int rank, int ranks) {
    const offset_t entries = static_cast<offset_t>(dimension) * dimension;
    const offset_t localEntries = static_cast<offset_t>(rows) * dimension;
    const int workers = static_cast<int>(std::min<offset_t>(
        {static_cast<offset_t>(omp_get_max_threads()), std::max<offset_t>(rows, 1),
         1 + localEntries / 131072}));
    std::vector<RowChunk> chunks(workers);
    for (int i = 0; i < workers; ++i) {
        chunks[i] = {firstRow + rowBoundary(rows, i, workers),
                     firstRow + rowBoundary(rows, i + 1, workers)};
    }

    const bool randomPattern = nonzeros != 0 && nonzeros < entries;
    const uint32_t threshold = randomPattern
        ? randomThreshold(static_cast<double>(nonzeros) / static_cast<double>(entries))
        : 0;
    offset_t localHits = 0;
    // The final chunk can end at nonzeros without counting its hits.
    const int countChunks = rank == ranks - 1 ? workers - 1 : workers;
    if (randomPattern && countChunks != 0) {
        #pragma omp parallel for num_threads(countChunks) schedule(static) reduction(+:localHits)
        for (int i = 0; i < countChunks; ++i) {
            auto& chunk = chunks[i];
            BenchmarkRandom positions(8675309, static_cast<offset_t>(chunk.begin) * dimension);
            const offset_t count = static_cast<offset_t>(chunk.end - chunk.begin) * dimension;
            offset_t hits = 0;
            for (offset_t j = 0; j < count; ++j) {
                hits += positions.next() <= threshold;
            }
            chunk.hits = hits;
            localHits += hits;
        }
    }

    offset_t globalHits = 0;
    MPI_Exscan(&localHits, &globalHits, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        globalHits = 0;
    }
    for (auto& chunk : chunks) {
        chunk.assigned = assignedAt(globalHits, static_cast<offset_t>(chunk.begin) * dimension,
                                    entries, nonzeros);
        globalHits += chunk.hits;
    }

    const offset_t first = chunks.front().assigned;
    const offset_t last = assignedAt(globalHits, static_cast<offset_t>(firstRow + rows) * dimension,
                                     entries, nonzeros);
    LocalMatrix matrix{
        std::unique_ptr<offset_t[]>(new offset_t[static_cast<size_t>(rows) + 1]),
        std::unique_ptr<index_t[]>(new index_t[last - first]),
        std::unique_ptr<double[]>(new double[last - first]),
        last - first
    };
    matrix.offsets[rows] = matrix.nonzeros;

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int i = 0; i < workers; ++i) {
        const auto& chunk = chunks[i];
        offset_t assigned = chunk.assigned;
        const offset_t end = i + 1 == workers ? last : chunks[i + 1].assigned;
        if (assigned == end) {
            std::fill(matrix.offsets.get() + (chunk.begin - firstRow),
                      matrix.offsets.get() + (chunk.end - firstRow), assigned - first);
            continue;
        }
        BenchmarkRandom positions(8675309, static_cast<offset_t>(chunk.begin) * dimension);
        BenchmarkRandom values(1, dimension + assigned);
        for (index_t row = chunk.begin; row < chunk.end; ++row) {
            matrix.offsets[row - firstRow] = assigned - first;
            const offset_t rowPosition = static_cast<offset_t>(row) * dimension;
            for (index_t col = 0; col < dimension && assigned < end; ++col) {
                if (entries - (rowPosition + col) <= nonzeros - assigned ||
                    positions.next() <= threshold) {
                    matrix.columns[assigned - first] = col;
                    matrix.values[assigned - first] = values.scaled(maxValue);
                    ++assigned;
                }
            }
        }
    }
    return matrix;
}

template <int Width>
__global__ void spmvCuda(const double* values, const index_t* columns, const offset_t* offsets,
                         const double* input, index_t rows, double* output) {
    const unsigned int lane = threadIdx.x % Width;
    const unsigned int groupStart = (threadIdx.x % 32) - lane;
    const unsigned int mask = (0xffffffffu >> (32 - Width)) << groupStart;
    const offset_t firstRow = static_cast<offset_t>(blockIdx.x) * (blockDim.x / Width)
                              + threadIdx.x / Width;
    const offset_t stride = static_cast<offset_t>(gridDim.x) * (blockDim.x / Width);
    for (offset_t row = firstRow; row < rows; row += stride) {
        const offset_t end = offsets[row + 1];
        double sum = 0.0;
        for (offset_t j = offsets[row] + lane; j < end; j += Width) {
            sum = fma(values[j], input[columns[j]], sum);
        }
        for (int delta = Width / 2; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(mask, sum, delta, Width);
        }
        if (lane == 0) {
            output[row] = sum;
        }
    }
}

template <int Width>
void launchSpmv(const DeviceBuffer<double>& values, const DeviceBuffer<index_t>& columns,
                const DeviceBuffer<offset_t>& offsets, const DeviceBuffer<double>& input,
                index_t rows, const DeviceBuffer<double>& output) {
    constexpr int rowsPerBlock = BLOCK_THREADS / Width;
    const auto blocks = static_cast<int>(std::min<offset_t>(
        (static_cast<offset_t>(rows) + rowsPerBlock - 1) / rowsPerBlock, 65535));
    spmvCuda<Width><<<blocks, BLOCK_THREADS>>>(values.data, columns.data, offsets.data,
                                               input.data, rows, output.data);
}

using SpmvLauncher = decltype(&launchSpmv<1>);

SpmvLauncher selectLauncher(offset_t nonzeros, index_t rows) {
    const offset_t average = rows ? nonzeros / rows : 0;
    if (average <= 3) {
        return launchSpmv<1>;
    }
    if (average <= 8) {
        return launchSpmv<4>;
    }
    if (average <= 16) {
        return launchSpmv<8>;
    }
    if (average <= 32) {
        return launchSpmv<16>;
    }
    return launchSpmv<32>;
}

void spmvCpu(const LocalMatrix& matrix, const double* input, index_t rows, double* output) {
    const int workers = static_cast<int>(std::min<offset_t>(
        {static_cast<offset_t>(omp_get_max_threads()), std::max<offset_t>(rows, 1),
         1 + matrix.nonzeros / 65536}));
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (offset_t j = matrix.offsets[row]; j < matrix.offsets[row + 1]; ++j) {
            sum += matrix.values[j] * input[matrix.columns[j]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, index_t firstRow, index_t rows) {
    for (index_t i = 0; i < rows; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            printf("Validation failed at index %u: reference %.10e, got %.10e (error: %.10e)\n",
                   firstRow + i, ref, res, error);
            return false;
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        }
        if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
        } else {
            index_t* option = nullptr;
            if (strcmp(argv[i], "-n") == 0) {
                option = &numRows;
            } else if (strcmp(argv[i], "-s") == 0) {
                option = &sparsity;
            } else if (strcmp(argv[i], "-i") == 0) {
                option = &iterations;
            }
            if (!option || i + 1 == argc) {
                if (rank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
            const char* value = argv[++i];
            const char* end = value + strlen(value);
            const auto parsed = std::from_chars(value, end, *option);
            if (parsed.ec != std::errc{} || parsed.ptr != end) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid integer for %s: %s\n", argv[i - 1], value);
                }
                return 1;
            }
        }
    }
    if (sparsity == 0 || numRows > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be positive and the matrix dimension must fit an MPI count.\n");
        }
        return 1;
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
    if (devices == 0) {
        abortBenchmark(rank, "No visible CUDA devices");
    }
    checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice", rank);

    const offset_t entries = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = entries / sparsity;
    const index_t firstRow = rowBoundary(numRows, rank, ranks);
    const index_t rows = rowBoundary(numRows, rank + 1, ranks) - firstRow;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nItems,
               entries ? 100.0 * (1.0 - static_cast<double>(nItems) / entries) : 0.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    DeviceBuffer<double> dValues, dInput, dOutput;
    DeviceBuffer<index_t> dColumns;
    DeviceBuffer<offset_t> dOffsets;
    std::vector<double> reference;
    // Use the same reduction order for a row regardless of its assigned rank.
    const auto launch = selectLauncher(nItems, numRows);
    {
        auto matrix = generateMatrix(numRows, firstRow, rows, nItems, maxVal, rank, ranks);
        std::vector<double> input(rows ? numRows : 0);
        BenchmarkRandom values(1);
        for (double& value : input) {
            value = values.scaled(maxVal);
        }
        if (validate) {
            if (rank == 0) {
                printf("Computing reference solution...\n");
            }
            reference.resize(rows);
            spmvCpu(matrix, input.data(), rows, reference.data());
        }
        if (rows && iterations) {
            dValues.upload(matrix.values.get(), matrix.nonzeros, rank);
            dColumns.upload(matrix.columns.get(), matrix.nonzeros, rank);
            dOffsets.upload(matrix.offsets.get(), static_cast<size_t>(rows) + 1, rank);
            dInput.upload(input.data(), input.size(), rank);
            dOutput.allocate(rows, rank);
        }
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    cudaEvent_t start, stop;
    if (rows && iterations) {
        checkCuda(cudaEventCreate(&start), "cudaEventCreate", rank);
        checkCuda(cudaEventCreate(&stop), "cudaEventCreate", rank);
        launch(dValues, dColumns, dOffsets, dInput, rows, dOutput);
        checkCuda(cudaGetLastError(), "SpMV warm-up launch", rank);
        checkCuda(cudaDeviceSynchronize(), "SpMV warm-up", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    float localMilliseconds = 0.0f;
    if (rows && iterations) {
        checkCuda(cudaEventRecord(start), "cudaEventRecord", rank);
        for (index_t iter = 0; iter < iterations; ++iter) {
            launch(dValues, dColumns, dOffsets, dInput, rows, dOutput);
        }
        checkCuda(cudaGetLastError(), "SpMV kernel launch", rank);
        checkCuda(cudaEventRecord(stop), "cudaEventRecord", rank);
        checkCuda(cudaEventSynchronize(stop), "SpMV execution", rank);
        checkCuda(cudaEventElapsedTime(&localMilliseconds, start, stop), "cudaEventElapsedTime", rank);
        checkCuda(cudaEventDestroy(start), "cudaEventDestroy", rank);
        checkCuda(cudaEventDestroy(stop), "cudaEventDestroy", rank);
    }
    float milliseconds = 0.0f;
    MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double gflops = milliseconds > 0 ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0;
        const double avgTime = iterations ? milliseconds / static_cast<double>(iterations) : 0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> output(validate || printResults ? rows : 0);
    if (!output.empty() && iterations) {
        checkCuda(cudaMemcpy(output.data(), dOutput.data, rows * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy from device", rank);
    }
    if (printResults) {
        std::vector<double> gathered;
        std::vector<int> counts, displacements;
        if (rank == 0) {
            gathered.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int i = 0; i < ranks; ++i) {
                displacements[i] = rowBoundary(numRows, i, ranks);
                counts[i] = rowBoundary(numRows, i + 1, ranks) - displacements[i];
            }
        }
        MPI_Gatherv(output.data(), rows, MPI_DOUBLE, gathered.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(gathered, "OutputVector");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = verifyResults(reference.data(), output.data(), firstRow, rows);
        int valid;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortBenchmark(rank, "MPI_THREAD_FUNNELED support is required");
    }

    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        abortBenchmark(rank, error.what());
    }
    MPI_Finalize();
    return result;
}
