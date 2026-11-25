#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int WARP_SIZE = 32;
constexpr int SORT_WINDOW = 512;
constexpr int COOPERATIVE_TILE = 4 * WARP_SIZE;

[[noreturn]] void abortRun(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        char message[512];
        snprintf(message, sizeof(message), "%s: %s", operation, cudaGetErrorString(error));
        abortRun(message);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Jump ahead in the original Linux rand() sequence. The additive generator's
// state evolves modulo 2^32 with characteristic polynomial x^31 - x^28 - 1.
class RandomSequence {
    static constexpr int degree = 31;
    using State = std::array<uint32_t, degree>;
    State state{};
    int oldest = 0;
    int lagged = degree - 3;

    static void advancePolynomial(State& polynomial) {
        const uint32_t high = polynomial.back();
        for (int i = degree - 1; i > 0; --i) {
            polynomial[i] = polynomial[i - 1];
        }
        polynomial[0] = high;
        polynomial[degree - 3] += high;
    }

    static State multiply(const State& a, const State& b) {
        std::array<uint32_t, 2 * degree - 1> product{};
        for (int i = 0; i < degree; ++i) {
            for (int j = 0; j < degree; ++j) {
                product[i + j] += a[i] * b[j];
            }
        }
        for (int i = 2 * degree - 2; i >= degree; --i) {
            product[i - 3] += product[i];
            product[i - degree] += product[i];
        }
        State result;
        std::copy_n(product.begin(), degree, result.begin());
        return result;
    }

    static const std::array<State, 64>& jumpPowers() {
        static const auto powers = [] {
            std::array<State, 64> table{};
            table[0][1] = 1;
            for (size_t i = 1; i < table.size(); ++i) {
                table[i] = multiply(table[i - 1], table[i - 1]);
            }
            return table;
        }();
        return powers;
    }

public:
    static constexpr uint32_t maxOutput = 0x7fffffff;

    RandomSequence(uint32_t seed, offset_t skip) {
        State seeds;
        seeds[0] = seed;
        for (int i = 1; i < degree; ++i) {
            seeds[i] = (16807ULL * seeds[i - 1]) % maxOutput;
        }
        for (int i = 0; i < degree; ++i) {
            state[i] = seeds[(i + 3) % degree];
        }
        for (int i = 0; i < 10 * degree; ++i) {
            next();
        }
        if (skip == 0) {
            return;
        }

        const auto& powers = jumpPowers();
        State polynomial{};
        polynomial[0] = 1;
        for (int bit = 0; skip != 0; ++bit, skip >>= 1) {
            if (skip & 1) {
                polynomial = multiply(polynomial, powers[bit]);
            }
        }

        const State initial = state;
        for (int i = 0; i < degree; ++i) {
            uint32_t value = 0;
            for (int j = 0; j < degree; ++j) {
                value += polynomial[j] * initial[j];
            }
            state[i] = value;
            advancePolynomial(polynomial);
        }
    }

    uint32_t next() {
        const uint32_t value = (state[oldest] += state[lagged]);
        if (++oldest == degree) oldest = 0;
        if (++lagged == degree) lagged = 0;
        return value >> 1;
    }

    double value(double maximum) {
        return maximum * (next() / (static_cast<double>(maxOutput) + 1.0));
    }
};

struct CandidateRows {
    index_t begin = 0;
    index_t end = 0;
    offset_t preceding = 0;
    std::vector<offset_t> offsets;
    std::vector<index_t> columns;
};

struct LocalCsr {
    explicit LocalCsr(index_t rows) : offsets(static_cast<size_t>(rows) + 1) {}

    std::vector<offset_t> offsets;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
};

// The serial generator accepts each kind of draw up to its quota. In particular,
// after the last allowed rejection it fills the remaining positions in order.
offset_t acceptedBefore(offset_t successes, offset_t cells, offset_t nonzeros, offset_t rejects) {
    const offset_t failures = cells - successes;
    return std::min(successes, nonzeros) + failures - std::min(failures, rejects);
}

LocalCsr generateMatrix(index_t firstRow, index_t rows, index_t dim, offset_t nonzeros,
                        double maximum, int threads) {
    LocalCsr matrix(rows);
    if (nonzeros == 0) {
        return matrix;
    }

    const offset_t cells = static_cast<offset_t>(dim) * dim;
    const offset_t rejects = cells - nonzeros;
    if (rejects == 0) {
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (offset_t row = 0; row <= rows; ++row) {
            matrix.offsets[row] = row * dim;
        }
        matrix.columns.reset(new index_t[matrix.offsets.back()]);
        matrix.values.reset(new double[matrix.offsets.back()]);
        #pragma omp parallel num_threads(threads)
        {
            const offset_t begin = static_cast<offset_t>(rows) * omp_get_thread_num() / omp_get_num_threads();
            const offset_t end = static_cast<offset_t>(rows) * (omp_get_thread_num() + 1) / omp_get_num_threads();
            RandomSequence values(1, dim + (firstRow + begin) * dim);
            for (offset_t row = begin; row < end; ++row) {
                for (index_t col = 0; col < dim; ++col) {
                    matrix.columns[row * dim + col] = col;
                    matrix.values[row * dim + col] = values.value(maximum);
                }
            }
        }
        return matrix;
    }

    const double probability = static_cast<double>(nonzeros) / static_cast<double>(cells);
    uint32_t threshold = static_cast<uint32_t>(probability * RandomSequence::maxOutput);
    // Match the division and comparison used by the original generator.
    if (threshold < RandomSequence::maxOutput &&
        (threshold + 1) / static_cast<double>(RandomSequence::maxOutput) <= probability) {
        ++threshold;
    } else if (threshold / static_cast<double>(RandomSequence::maxOutput) > probability) {
        --threshold;
    }

    std::vector<CandidateRows> chunks(threads);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int part = 0; part < threads; ++part) {
        auto& chunk = chunks[part];
        chunk.begin = static_cast<offset_t>(rows) * part / threads;
        chunk.end = static_cast<offset_t>(rows) * (part + 1) / threads;
        chunk.offsets.resize(static_cast<size_t>(chunk.end - chunk.begin) + 1);
        chunk.columns.reserve(static_cast<offset_t>((chunk.end - chunk.begin) * static_cast<double>(dim) * probability));
        RandomSequence positions(8675309, static_cast<offset_t>(firstRow + chunk.begin) * dim);
        for (index_t row = chunk.begin; row < chunk.end; ++row) {
            for (index_t col = 0; col < dim; ++col) {
                if (positions.next() <= threshold) {
                    chunk.columns.push_back(col);
                }
            }
            chunk.offsets[row - chunk.begin + 1] = chunk.columns.size();
        }
    }

    offset_t localSuccesses = 0;
    for (const auto& chunk : chunks) {
        localSuccesses += chunk.columns.size();
    }
    offset_t rankPrefix = 0;
    MPI_Exscan(&localSuccesses, &rankPrefix, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) rankPrefix = 0;

    const offset_t firstEntry = acceptedBefore(rankPrefix, static_cast<offset_t>(firstRow) * dim,
                                               nonzeros, rejects);
    offset_t prefix = rankPrefix;
    for (auto& chunk : chunks) {
        chunk.preceding = prefix;
        prefix += chunk.columns.size();
    }
    matrix.offsets.back() = acceptedBefore(prefix, static_cast<offset_t>(firstRow + rows) * dim,
                                           nonzeros, rejects) - firstEntry;
    matrix.columns.reset(new index_t[matrix.offsets.back()]);
    matrix.values.reset(new double[matrix.offsets.back()]);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int part = 0; part < threads; ++part) {
        const auto& chunk = chunks[part];
        const offset_t start = acceptedBefore(chunk.preceding,
            static_cast<offset_t>(firstRow + chunk.begin) * dim, nonzeros, rejects);
        RandomSequence values(1, dim + start);
        offset_t written = start - firstEntry;
        for (index_t row = chunk.begin; row < chunk.end; ++row) {
            matrix.offsets[row] = written;
            const offset_t begin = chunk.offsets[row - chunk.begin];
            const offset_t end = chunk.offsets[row - chunk.begin + 1];
            const offset_t successes = chunk.preceding + begin;
            const offset_t rowCell = static_cast<offset_t>(firstRow + row) * dim;
            const offset_t rowEndRejects = rowCell + dim - (chunk.preceding + end);

            if (rowEndRejects > rejects) {
                const offset_t allowedRejects = rejects - std::min(rejects, rowCell - successes);
                // Locate the start of the forced suffix among the ordered candidates.
                offset_t low = begin;
                offset_t high = end;
                while (low < high) {
                    const offset_t mid = low + (high - low) / 2;
                    if (chunk.columns[mid] - (mid - begin) < allowedRejects) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                for (offset_t j = begin; j < low; ++j) {
                    matrix.columns[written] = chunk.columns[j];
                    matrix.values[written++] = values.value(maximum);
                }
                for (offset_t col = allowedRejects + low - begin; col < dim; ++col) {
                    matrix.columns[written] = static_cast<index_t>(col);
                    matrix.values[written++] = values.value(maximum);
                }
            } else {
                const offset_t count = std::min(end - begin, nonzeros - std::min(nonzeros, successes));
                for (offset_t j = begin; j < begin + count; ++j) {
                    matrix.columns[written] = chunk.columns[j];
                    matrix.values[written++] = values.value(maximum);
                }
            }
        }
    }
    return matrix;
}

struct PackedRow {
    offset_t offset;
    index_t length;
    index_t row;
};

struct Slice {
    offset_t offset;
    offset_t size;
    int stride;
};

struct WarpJob {
    index_t firstRow;
    int stride; // Zero assigns all lanes to one CSR row.
};

struct PackedMatrix {
    std::vector<PackedRow> rows;
    std::vector<WarpJob> jobs;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
    offset_t entries = 0;
};

PackedMatrix packMatrix(LocalCsr& matrix, index_t rows, int threads, bool cooperative) {
    PackedMatrix packed;
    packed.rows.resize(rows);
    auto rowLength = [&](index_t row) {
        return static_cast<index_t>(matrix.offsets[row + 1] - matrix.offsets[row]);
    };
    if (cooperative) {
        packed.jobs.resize(rows);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (index_t row = 0; row < rows; ++row) {
            packed.rows[row] = {matrix.offsets[row], rowLength(row), row};
            packed.jobs[row] = {row, 0};
        }
        packed.entries = matrix.offsets.back();
        packed.columns = std::move(matrix.columns);
        packed.values = std::move(matrix.values);
        return packed;
    }

    std::vector<Slice> slices((rows + WARP_SIZE - 1) / WARP_SIZE);
    // Group similar row lengths into warps. Interleaving keeps warp loads
    // coalesced while preserving each row's original summation order.
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (offset_t begin = 0; begin < rows; begin += SORT_WINDOW) {
        const index_t count = std::min<offset_t>(SORT_WINDOW, rows - begin);
        std::array<index_t, SORT_WINDOW> order;
        std::iota(order.begin(), order.begin() + count, static_cast<index_t>(begin));
        std::sort(order.begin(), order.begin() + count, [&](index_t a, index_t b) {
            return rowLength(a) < rowLength(b);
        });
        for (index_t start = 0; start < count; start += WARP_SIZE) {
            const index_t lanes = std::min<index_t>(WARP_SIZE, count - start);
            const offset_t paddedSize = static_cast<offset_t>(WARP_SIZE) * rowLength(order[start + lanes - 1]);
            offset_t actualSize = 0;
            for (index_t lane = 0; lane < lanes; ++lane) {
                const index_t row = order[start + lane];
                const index_t length = rowLength(row);
                packed.rows[begin + start + lane] = {actualSize, length, row};
                actualSize += length;
            }

            // Keep highly irregular slices in CSR order to bound padding overhead.
            const bool interleave = paddedSize <= actualSize + actualSize / 4;
            slices[(begin + start) / WARP_SIZE] = {
                0, interleave ? paddedSize : actualSize, interleave ? WARP_SIZE : 1
            };
        }
    }

    for (size_t i = 0; i < slices.size(); ++i) {
        auto& slice = slices[i];
        slice.offset = packed.entries;
        packed.entries += slice.size;
        const index_t first = i * WARP_SIZE;
        const index_t end = std::min<index_t>(first + WARP_SIZE, rows);
        if (slice.stride == 1 && packed.rows[end - 1].length >= 2 * WARP_SIZE) {
            // Long, irregular rows get a warp each to avoid leaving one lane
            // to load the dense tail that the generator sometimes produces.
            for (index_t row = end; row > first; --row) {
                packed.jobs.push_back({row - 1, 0});
            }
        } else {
            packed.jobs.push_back({first, slice.stride});
        }
    }
    packed.columns.reset(new index_t[packed.entries]);
    packed.values.reset(new double[packed.entries]);

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t sliceId = 0; sliceId < slices.size(); ++sliceId) {
        const auto& slice = slices[sliceId];
        const index_t begin = sliceId * WARP_SIZE;
        const index_t lanes = std::min<index_t>(WARP_SIZE, rows - begin);
        if (slice.stride == WARP_SIZE) {
            const offset_t width = slice.size / WARP_SIZE;
            std::array<offset_t, WARP_SIZE> sources{};
            std::array<index_t, WARP_SIZE> lengths{};
            for (index_t lane = 0; lane < lanes; ++lane) {
                auto& row = packed.rows[begin + lane];
                row.offset = slice.offset + lane;
                sources[lane] = matrix.offsets[row.row];
                lengths[lane] = row.length;
            }
            // Keep a tile of the transposed data in cache during packing.
            constexpr offset_t tileWidth = 64;
            for (offset_t tile = 0; tile < width; tile += tileWidth) {
                const offset_t end = std::min(tile + tileWidth, width);
                for (index_t lane = 0; lane < WARP_SIZE; ++lane) {
                    const offset_t source = sources[lane];
                    const offset_t length = lengths[lane];
                    const offset_t base = slice.offset + lane;
                    for (offset_t j = tile; j < std::min(length, end); ++j) {
                        const offset_t target = base + j * WARP_SIZE;
                        packed.columns[target] = matrix.columns[source + j];
                        packed.values[target] = matrix.values[source + j];
                    }
                    for (offset_t j = std::max(tile, length); j < end; ++j) {
                        const offset_t target = base + j * WARP_SIZE;
                        packed.columns[target] = 0;
                        packed.values[target] = 0.0;
                    }
                }
            }
        } else {
            for (index_t lane = 0; lane < lanes; ++lane) {
                auto& row = packed.rows[begin + lane];
                row.offset += slice.offset;
                if (row.length != 0) {
                    const offset_t source = matrix.offsets[row.row];
                    std::copy_n(matrix.columns.get() + source, row.length, packed.columns.get() + row.offset);
                    std::copy_n(matrix.values.get() + source, row.length, packed.values.get() + row.offset);
                }
            }
        }
    }
    return packed;
}

__global__ void spmvGpu(const PackedRow* __restrict__ rows, const WarpJob* __restrict__ jobs,
                        const index_t* __restrict__ columns, const double* __restrict__ values,
                        const double* __restrict__ input, double* __restrict__ output,
                        index_t rowCount, index_t jobCount) {
    extern __shared__ double products[];
    const index_t warp = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (warp >= jobCount) return;
    const WarpJob job = jobs[warp];
    const int lane = threadIdx.x % WARP_SIZE;
    if (job.stride != 0) {
        const index_t slot = job.firstRow + lane;
        if (slot < rowCount) {
            const PackedRow row = rows[slot];
            double sum = 0.0;
            // Expose independent loads while keeping the additions in order.
            constexpr int batchSize = 8;
            index_t j = 0;
            for (; j + batchSize <= row.length; j += batchSize) {
                double batch[batchSize];
                #pragma unroll
                for (int k = 0; k < batchSize; ++k) {
                    const offset_t entry = row.offset + static_cast<offset_t>(j + k) * job.stride;
                    batch[k] = values[entry] * input[columns[entry]];
                }
                #pragma unroll
                for (int k = 0; k < batchSize; ++k) {
                    sum += batch[k];
                }
            }
            for (; j < row.length; ++j) {
                const offset_t entry = row.offset + static_cast<offset_t>(j) * job.stride;
                sum += values[entry] * input[columns[entry]];
            }
            output[row.row] = sum;
        }
        return;
    }

    // Lanes load and multiply consecutive entries together. The leader adds
    // the products in the same order as the serial CSR implementation.
    const PackedRow row = rows[job.firstRow];
    const offset_t end = row.offset + row.length;
    const int sharedBase = threadIdx.x / WARP_SIZE * COOPERATIVE_TILE;
    double sum = 0.0;
    for (offset_t start = row.offset; start < end; start += COOPERATIVE_TILE) {
        const int count = min(static_cast<offset_t>(COOPERATIVE_TILE), end - start);
        #pragma unroll
        for (int step = 0; step < COOPERATIVE_TILE; step += WARP_SIZE) {
            if (step + lane < count) {
                const offset_t entry = start + step + lane;
                products[sharedBase + step + lane] = values[entry] * input[columns[entry]];
            }
        }
        __syncwarp();
        if (lane == 0) {
            int step = 0;
            for (; step + WARP_SIZE <= count; step += WARP_SIZE) {
                #pragma unroll
                for (int i = 0; i < WARP_SIZE; ++i) {
                    sum += products[sharedBase + step + i];
                }
            }
            for (; step < count; ++step) {
                sum += products[sharedBase + step];
            }
        }
        __syncwarp();
    }
    if (lane == 0) output[row.row] = sum;
}

template<class T>
class DeviceBuffer {
    T* data = nullptr;

public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
        }
    }
    ~DeviceBuffer() {
        if (data != nullptr) CUDA_CHECK(cudaFree(data));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data; }
    void upload(const T* source, size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
};

std::vector<double> cpuReference(const LocalCsr& matrix, const std::vector<double>& input,
                                 index_t rows, int threads) {
    std::vector<double> reference(rows);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (offset_t j = matrix.offsets[row]; j < matrix.offsets[row + 1]; ++j) {
            sum += matrix.values[j] * input[matrix.columns[j]];
        }
        reference[row] = sum;
    }
    return reference;
}

bool verifyResults(const std::vector<double>& reference, const std::vector<double>& output,
                   index_t firstRow, int rank, int threads) {
    index_t firstError = UINT32_MAX;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:firstError)
    for (index_t row = 0; row < output.size(); ++row) {
        const double ref = reference[row];
        const double result = output[row];
        const double error = std::abs(ref) < 1e-10 ? std::abs(result) : std::abs((result - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            firstError = std::min(firstError, firstRow + row);
        }
    }

    index_t globalError = UINT32_MAX;
    MPI_Allreduce(&firstError, &globalError, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (globalError == UINT32_MAX) return true;

    const int owner = firstError == globalError ? rank : INT_MAX;
    int errorRank = 0;
    MPI_Allreduce(&owner, &errorRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double values[2] = {};
    if (rank == errorRank) {
        values[0] = reference[globalError - firstRow];
        values[1] = output[globalError - firstRow];
    }
    MPI_Bcast(values, 2, MPI_DOUBLE, errorRank, MPI_COMM_WORLD);
    if (rank == 0) {
        if (std::abs(values[0]) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   globalError, values[0], values[1]);
        } else {
            const double relativeError = std::abs((values[1] - values[0]) / values[0]);
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   globalError, values[0], values[1], relativeError);
        }
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

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
    if (sparsity == 0 || numRows > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Matrix size must be between 0 and %d and sparsity must be nonzero.\n", INT_MAX);
        return 1;
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) abortRun("No CUDA devices available");
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    const offset_t cells = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = cells / sparsity;
    auto firstRowFor = [&](int r) {
        return static_cast<index_t>(static_cast<offset_t>(numRows) * r / ranks);
    };
    const index_t firstRow = firstRowFor(rank);
    const index_t localRows = firstRowFor(rank + 1) - firstRow;
    const offset_t localCells = static_cast<offset_t>(localRows) * numRows;
    const int threads = static_cast<int>(std::min<offset_t>(omp_get_max_threads(),
        std::min<offset_t>(std::max<index_t>(1, localRows), 1 + localCells / 262144)));

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nItems,
               cells == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / cells));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // The input vector is fixed across all iterations and can be regenerated
    // on each rank without a broadcast or any iteration-time communication.
    std::vector<double> input(numRows);
    RandomSequence vectorValues(1, 0);
    for (auto& value : input) value = vectorValues.value(maxVal);
    auto matrix = generateMatrix(firstRow, localRows, numRows, nItems, maxVal, threads);
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference = cpuReference(matrix, input, localRows, threads);
    }

    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, localRank % devices));
    const bool hasWork = localRows != 0 && iterations != 0;
    const bool cooperative = hasWork && localRows < 16u * multiprocessors &&
        matrix.offsets.back() >= 24u * static_cast<offset_t>(localRows);
    PackedMatrix packed;
    if (hasWork) {
        packed = packMatrix(matrix, localRows, threads, cooperative);
    }
    matrix = LocalCsr(0);

    DeviceBuffer<PackedRow> deviceRows(packed.rows.size());
    DeviceBuffer<WarpJob> deviceJobs(packed.jobs.size());
    DeviceBuffer<index_t> deviceCols(packed.entries);
    DeviceBuffer<double> deviceValues(packed.entries);
    DeviceBuffer<double> deviceInput(hasWork ? numRows : 0);
    DeviceBuffer<double> deviceOutput(hasWork ? localRows : 0);
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph = nullptr;
    constexpr index_t GRAPH_BATCH = 64;
    const index_t batch = std::min(GRAPH_BATCH, iterations);
    const index_t jobs = packed.jobs.size();
    const index_t warpsPerBlock = jobs < 8u * multiprocessors ? 1 : 4;
    const index_t blockSize = warpsPerBlock * WARP_SIZE;
    const index_t blocks = (jobs + warpsPerBlock - 1) / warpsPerBlock;
    auto launch = [&] {
        const size_t sharedBytes = warpsPerBlock * COOPERATIVE_TILE * sizeof(double);
        spmvGpu<<<blocks, blockSize, sharedBytes, stream>>>(deviceRows.get(),
            deviceJobs.get(), deviceCols.get(), deviceValues.get(), deviceInput.get(),
            deviceOutput.get(), localRows, jobs);
    };

    if (hasWork) {
        deviceRows.upload(packed.rows.data(), packed.rows.size());
        deviceJobs.upload(packed.jobs.data(), packed.jobs.size());
        deviceCols.upload(packed.columns.get(), packed.entries);
        deviceValues.upload(packed.values.get(), packed.entries);
        deviceInput.upload(input.data(), numRows);
        packed = PackedMatrix{};

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (index_t i = 0; i < batch; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        cudaGraph_t captured = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&graph, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaGraphUpload(graph, stream));
        // Bring up the CUDA context and kernel before starting the benchmark.
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (hasWork) {
        for (index_t i = 0; i < iterations / batch; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph, stream));
        }
        for (index_t i = 0; i < iterations % batch; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double gflops = iterations != 0 && duration > 0.0
            ? (2.0 * nItems * iterations) / (duration * 1e9) : 0.0;
        const double avgTime = iterations != 0 ? duration * 1000.0 / iterations : 0.0;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> output;
    if (validate || printResults) {
        output.resize(localRows);
        if (hasWork) {
            CUDA_CHECK(cudaMemcpy(output.data(), deviceOutput.get(),
                                  localRows * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
    if (hasWork) {
        CUDA_CHECK(cudaGraphExecDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> globalOutput;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                displacements[r] = static_cast<int>(firstRowFor(r));
                counts[r] = static_cast<int>(firstRowFor(r + 1) - firstRowFor(r));
            }
            globalOutput.resize(numRows);
        }
        MPI_Gatherv(output.data(), static_cast<int>(localRows), MPI_DOUBLE, globalOutput.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(globalOutput, "OutputVector");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = verifyResults(reference, output, firstRow, rank, threads);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int supplied = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &supplied) != MPI_SUCCESS) {
        return 1;
    }
    if (supplied < MPI_THREAD_FUNNELED) abortRun("MPI_THREAD_FUNNELED is required");

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        abortRun(error.what());
    }
    MPI_Finalize();
    return result;
}
