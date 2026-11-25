#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cinttypes>
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

namespace {

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr int warpThreads = 32;
constexpr int narrowSliceRows = 4;
constexpr int wideSliceRows = warpThreads;
constexpr int blockThreads = 128;
constexpr int tiledBlockRows = blockThreads / warpThreads * narrowSliceRows;
constexpr double maxRelativeError = 0.02;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : bytes(count * sizeof(T)) {
        if (bytes != 0) CUDA_CHECK(cudaMalloc(&data, bytes));
    }
    ~DeviceBuffer() {
        if (data) CUDA_CHECK(cudaFree(data));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void upload(const T* host) {
        if (bytes != 0) CUDA_CHECK(cudaMemcpy(data, host, bytes, cudaMemcpyHostToDevice));
    }
    void download(T* host) const {
        if (bytes != 0) CUDA_CHECK(cudaMemcpy(host, data, bytes, cudaMemcpyDeviceToHost));
    }
    T* get() const { return data; }

private:
    T* data = nullptr;
    size_t bytes;
};

void selectDevice(int rank) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

// The benchmark uses glibc's rand() stream. Its 31-word additive
// recurrence allows any worker to start at the serial stream's exact offset.
class RandomSequence {
    static constexpr int order = 31;
    using State = std::array<uint32_t, order>;

    static State multiply(const State& left, const State& right) {
        std::array<uint32_t, 2 * order - 1> product{};
        for (int i = 0; i < order; ++i) {
            for (int j = 0; j < order; ++j) {
                product[i + j] += left[i] * right[j];
            }
        }
        // x^31 = x^28 + 1, with coefficients reduced modulo 2^32.
        for (int i = 2 * order - 2; i >= order; --i) {
            product[i - 3] += product[i];
            product[i - order] += product[i];
        }

        State result;
        std::copy_n(product.begin(), order, result.begin());
        return result;
    }

    static const std::array<State, 64>& jumpPowers() {
        static const auto powers = [] {
            std::array<State, 64> result{};
            result[0][1] = 1;
            for (size_t i = 1; i < result.size(); ++i) {
                result[i] = multiply(result[i - 1], result[i - 1]);
            }
            return result;
        }();
        return powers;
    }

public:
    RandomSequence(uint32_t seed, offset_t skip) {
        // Park-Miller seeding, followed by the libc generator's warm-up.
        std::array<uint32_t, 344 + order> initial;
        initial[0] = seed;
        for (int i = 1; i < order; ++i) {
            initial[i] = (16807ULL * initial[i - 1]) % 2147483647;
        }
        for (int i = order; i < order + 3; ++i) initial[i] = initial[i - order];
        for (size_t i = order + 3; i < initial.size(); ++i) {
            initial[i] = initial[i - 3] + initial[i - order];
        }

        if (skip == 0) {
            std::copy_n(initial.begin() + 344, order, state.begin());
            return;
        }

        State cursor{};
        cursor[0] = 1;
        const auto& powers = jumpPowers();
        for (size_t bit = 0; skip != 0; ++bit, skip >>= 1) {
            if (skip & 1) cursor = multiply(cursor, powers[bit]);
        }
        for (int i = 0; i < order; ++i) {
            uint32_t word = 0;
            for (int j = 0; j < order; ++j) word += cursor[j] * initial[344 + j];
            state[i] = word;
            const uint32_t high = cursor.back();
            for (int j = order - 1; j > 0; --j) cursor[j] = cursor[j - 1];
            cursor[0] = high;
            cursor[order - 3] += high;
        }
    }

    uint32_t next() {
        const uint32_t word = state[head];
        state[head] += state[lag];
        if (++head == order) head = 0;
        if (++lag == order) lag = 0;
        return word >> 1;
    }

    double value(double maxVal) {
        return maxVal * (next() * (1.0 / 2147483648.0));
    }

private:
    State state;
    int head = 0;
    int lag = order - 3;
};

// Convert the original floating-point acceptance test into an exact integer test.
uint32_t acceptanceThreshold(double probability) {
    constexpr uint32_t randomMax = 2147483647;
    uint32_t low = 0;
    uint32_t high = randomMax;
    while (low < high) {
        const uint32_t mid = low + (high - low + 1) / 2;
        if (static_cast<double>(mid) / randomMax <= probability) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

struct RowBlock {
    index_t begin;
    index_t end;
    std::vector<index_t> candidates;
    offset_t prefix = 0;
};

struct LocalMatrix {
    std::vector<offset_t> slices;
    std::vector<index_t> lengths;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
};

// Given the number of natural hits in a prefix, reproduce the serial cap
// at nnz and its rule that fills every remaining position when necessary.
offset_t assigned(offset_t hits, offset_t cellsSeen, offset_t cells, offset_t nnz) {
    const offset_t remaining = cells - cellsSeen;
    const offset_t forced = nnz > remaining ? nnz - remaining : 0;
    return std::min(nnz, std::max(hits, forced));
}

LocalMatrix makeMatrix(index_t dim, index_t firstRow, index_t rows, offset_t nnz,
                       double maxVal, index_t sliceRows) {
    const offset_t cells = offset_t{dim} * dim;
    const index_t slices = (rows + sliceRows - 1) / sliceRows;
    const offset_t localCells = offset_t{rows} * dim;
    constexpr offset_t cellsPerWorker = 65536;
    const offset_t workChunks = (localCells + cellsPerWorker - 1) / cellsPerWorker;
    const int workers = std::max<offset_t>(1, std::min<offset_t>(
        omp_get_max_threads(), std::min<offset_t>(slices, workChunks)));

    std::vector<RowBlock> blocks(workers);
    std::vector<size_t> candidateEnds(rows);
    const bool sample = nnz != 0 && nnz != cells;
    const double probability = sample ? static_cast<double>(nnz) / cells : 0.0;
    const uint32_t threshold = acceptanceThreshold(probability);

    for (int worker = 0; worker < workers; ++worker) {
        auto& block = blocks[worker];
        block.begin = (offset_t{slices} * worker / workers) * sliceRows;
        block.end = std::min<offset_t>(rows, (offset_t{slices} * (worker + 1) / workers) * sliceRows);
    }

    if (sample && rows != 0) {
        #pragma omp parallel for num_threads(workers) schedule(static)
        for (int worker = 0; worker < workers; ++worker) {
            auto& block = blocks[worker];
            const double expected = static_cast<double>(block.end - block.begin) * dim * probability;
            block.candidates.reserve(static_cast<size_t>(expected + 8 * std::sqrt(expected) + 64));
            RandomSequence random(8675309, offset_t{firstRow + block.begin} * dim);
            for (index_t row = block.begin; row < block.end; ++row) {
                for (index_t col = 0; col < dim; ++col) {
                    if (random.next() <= threshold) block.candidates.push_back(col);
                }
                candidateEnds[row] = block.candidates.size();
            }
        }
    }

    offset_t localHits = 0;
    for (auto& block : blocks) {
        block.prefix = localHits;
        localHits += block.candidates.size();
    }
    offset_t precedingHits = 0;
    if (sample) {
        MPI_Exscan(&localHits, &precedingHits, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        // Rank zero's receive value is undefined. Empty leading ranks
        // contribute zero as well.
        if (firstRow == 0) precedingHits = 0;
    }

    LocalMatrix matrix;
    matrix.lengths.resize(rows);
    matrix.slices.resize(slices + 1);

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const auto& block = blocks[worker];
        const offset_t prefix = precedingHits + block.prefix;
        offset_t cell = offset_t{firstRow + block.begin} * dim;
        offset_t start = assigned(prefix, cell, cells, nnz);
        for (index_t row = block.begin; row < block.end; ++row) {
            cell += dim;
            const offset_t end = assigned(prefix + candidateEnds[row], cell, cells, nnz);
            matrix.lengths[row] = end - start;
            start = end;
        }
        for (index_t row = block.begin; row < block.end; row += sliceRows) {
            const index_t end = std::min<index_t>(row + sliceRows, rows);
            const index_t width = *std::max_element(matrix.lengths.begin() + row,
                                                    matrix.lengths.begin() + end);
            matrix.slices[row / sliceRows + 1] = offset_t{width} * sliceRows;
        }
    }

    for (index_t slice = 0; slice < slices; ++slice) {
        matrix.slices[slice + 1] += matrix.slices[slice];
    }

    const offset_t entries = matrix.slices.back();
    if (entries == 0) return matrix;
    matrix.columns.reset(new index_t[entries]);
    matrix.values.reset(new double[entries]);

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const auto& block = blocks[worker];
        const offset_t prefix = precedingHits + block.prefix;
        offset_t cell = offset_t{firstRow + block.begin} * dim;
        RandomSequence random(1, dim + assigned(prefix, cell, cells, nnz));
        size_t naturalBegin = 0;
        for (index_t row = block.begin; row < block.end; ++row, cell += dim) {
            const size_t naturalEnd = candidateEnds[row];
            const index_t length = matrix.lengths[row];
            const index_t naturalLength = naturalEnd - naturalBegin;
            index_t keep = std::min(length, naturalLength);
            index_t tailColumn = 0;
            if (length > naturalLength) {
                const offset_t missesBeforeRow = cell - prefix - naturalBegin;
                const offset_t missBudget = cells - nnz;
                keep = 0;
                if (missBudget > missesBeforeRow) {
                    // Keep hits before the forced suffix. For candidate k at
                    // column j, exactly j - k misses precede it in the row.
                    const offset_t misses = missBudget - missesBeforeRow;
                    index_t high = naturalLength;
                    while (keep < high) {
                        const index_t mid = keep + (high - keep) / 2;
                        if (block.candidates[naturalBegin + mid] - mid < misses) {
                            keep = mid + 1;
                        } else {
                            high = mid;
                        }
                    }
                    tailColumn = misses + keep;
                }
            }

            const index_t slice = row / sliceRows;
            const index_t lane = row % sliceRows;
            offset_t slot = matrix.slices[slice] + lane;
            for (index_t entry = 0; entry < keep; ++entry, slot += sliceRows) {
                matrix.columns[slot] = block.candidates[naturalBegin + entry];
                matrix.values[slot] = random.value(maxVal);
            }
            for (index_t entry = keep; entry < length; ++entry, slot += sliceRows) {
                matrix.columns[slot] = tailColumn++;
                matrix.values[slot] = random.value(maxVal);
            }
            // Padding is never used by the multiply, including lanes after
            // the last row in a partial slice.
            for (; slot < matrix.slices[slice + 1]; slot += sliceRows) {
                matrix.columns[slot] = 0;
                matrix.values[slot] = 0.0;
            }
            naturalBegin = naturalEnd;
        }
        if (block.end == rows && rows % sliceRows != 0) {
            for (offset_t start = matrix.slices[slices - 1]; start < entries; start += sliceRows) {
                std::fill(matrix.columns.get() + start + rows % sliceRows,
                          matrix.columns.get() + start + sliceRows, 0);
                std::fill(matrix.values.get() + start + rows % sliceRows,
                          matrix.values.get() + start + sliceRows, 0.0);
            }
        }
    }
    return matrix;
}

// With many short rows, give each lane a row and coalesce each warp's
// matrix reads across a full slice.
__global__ void spmvRows(const double* __restrict__ values, const index_t* __restrict__ columns,
                         const offset_t* __restrict__ slices, const index_t* __restrict__ lengths,
                         const double* __restrict__ input, index_t rows, double* __restrict__ output) {
    const index_t row = blockIdx.x * blockThreads + threadIdx.x;
    if (row >= rows) return;

    const offset_t start = slices[row / wideSliceRows] + row % wideSliceRows;
    const index_t length = lengths[row];
    double sum = 0.0;
    #pragma unroll 4
    for (index_t entry = 0; entry < length; ++entry) {
        const offset_t slot = start + offset_t{entry} * wideSliceRows;
        sum += values[slot] * input[columns[slot]];
    }
    output[row] = sum;
}

// A warp loads a tile of consecutive entries for four rows. Each row's
// products are then added in column order to preserve the serial result.
__global__ void spmvTiled(const double* __restrict__ values, const index_t* __restrict__ columns,
                          const offset_t* __restrict__ slices, const index_t* __restrict__ lengths,
                          const double* __restrict__ input, index_t rows, double* __restrict__ output) {
    constexpr int tileEntries = warpThreads / narrowSliceRows;
    const unsigned lane = threadIdx.x % warpThreads;
    const unsigned warp = threadIdx.x / warpThreads;
    const index_t slice = blockIdx.x * (blockThreads / warpThreads) + warp;
    const index_t first = slice * narrowSliceRows;
    if (first >= rows) return;

    const unsigned rowInSlice = lane % narrowSliceRows;
    const index_t row = first + rowInSlice;
    const index_t length = row < rows ? lengths[row] : 0;
    const offset_t start = slices[slice];
    const index_t width = (slices[slice + 1] - start) / narrowSliceRows;
    double sum = 0.0;
    for (index_t base = 0; base < width; base += tileEntries) {
        const index_t entry = base + lane / narrowSliceRows;
        double product = 0.0;
        if (entry < length) {
            const offset_t slot = start + offset_t{base} * narrowSliceRows + lane;
            product = values[slot] * input[columns[slot]];
        }
        #pragma unroll
        for (int k = 0; k < tileEntries; ++k) {
            const unsigned source = k * narrowSliceRows + rowInSlice;
            const double orderedProduct = __shfl_sync(0xffffffffu, product, source);
            if (base + k < length) sum += orderedProduct;
        }
    }
    if (lane < narrowSliceRows && row < rows) output[row] = sum;
}

std::vector<double> referenceSpmv(const LocalMatrix& matrix, const std::vector<double>& input,
                                  index_t sliceRows) {
    std::vector<double> reference(matrix.lengths.size());
    #pragma omp parallel for schedule(static) if(reference.size() >= 256)
    for (size_t row = 0; row < reference.size(); ++row) {
        const offset_t start = matrix.slices[row / sliceRows] + row % sliceRows;
        double sum = 0.0;
        for (index_t entry = 0; entry < matrix.lengths[row]; ++entry) {
            const offset_t slot = start + offset_t{entry} * sliceRows;
            sum += matrix.values[slot] * input[matrix.columns[slot]];
        }
        reference[row] = sum;
    }
    return reference;
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
    if (sparsity == 0 || numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Matrix dimension or sparsity is out of range\n");
        return 1;
    }

    const offset_t cells = offset_t{numRows} * numRows;
    const offset_t nItems = cells / sparsity;
    const auto rowBoundary = [=](int part) { return offset_t{numRows} * part / ranks; };
    const index_t firstRow = rowBoundary(rank);
    const index_t localRows = rowBoundary(rank + 1) - firstRow;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / cells));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    selectDevice(rank);

    // The input vector is fixed and can be generated identically on every
    // rank. Each rank only constructs and uploads its own matrix rows.
    std::vector<double> input(numRows);
    RandomSequence inputRandom(1, 0);
    for (double& value : input) value = inputRandom.value(maxVal);
    // Use more lanes per row when rows are long and the rank has too few
    // rows to expose enough parallelism with a lane per row.
    const bool tiled = localRows < 8192 && nItems >= offset_t{numRows} * 16;
    const index_t sliceRows = tiled ? narrowSliceRows : wideSliceRows;
    LocalMatrix matrix = makeMatrix(numRows, firstRow, localRows, nItems, maxVal, sliceRows);

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference = referenceSpmv(matrix, input, sliceRows);
    }

    DeviceBuffer<double> deviceInput(localRows != 0 ? numRows : 0);
    DeviceBuffer<double> deviceOutput(localRows);
    DeviceBuffer<offset_t> deviceSlices(localRows != 0 ? matrix.slices.size() : 0);
    DeviceBuffer<index_t> deviceLengths(localRows);
    DeviceBuffer<index_t> deviceColumns(matrix.slices.back());
    DeviceBuffer<double> deviceValues(matrix.slices.back());
    deviceInput.upload(input.data());
    deviceSlices.upload(matrix.slices.data());
    deviceLengths.upload(matrix.lengths.data());
    deviceColumns.upload(matrix.columns.get());
    deviceValues.upload(matrix.values.get());
    matrix = LocalMatrix{};

    const index_t rowsPerBlock = tiled ? tiledBlockRows : blockThreads;
    const index_t blocks = (localRows + rowsPerBlock - 1) / rowsPerBlock;
    const auto launch = [&] {
        if (tiled) {
            spmvTiled<<<blocks, blockThreads>>>(deviceValues.get(), deviceColumns.get(),
                deviceSlices.get(), deviceLengths.get(), deviceInput.get(),
                localRows, deviceOutput.get());
        } else {
            spmvRows<<<blocks, blockThreads>>>(deviceValues.get(), deviceColumns.get(),
                deviceSlices.get(), deviceLengths.get(), deviceInput.get(),
                localRows, deviceOutput.get());
        }
    };

    const bool runKernel = localRows != 0 && iterations != 0;
    if (runKernel) {
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (runKernel) {
        for (index_t iter = 0; iter < iterations; ++iter) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations != 0 ? 1000.0 * duration / iterations : 0.0;
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> localOutput;
    if (printResults || validate) {
        localOutput.resize(localRows);
        if (runKernel) deviceOutput.download(localOutput.data());
    }

    if (printResults) {
        std::vector<double> output;
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            output.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int part = 0; part < ranks; ++part) {
                displacements[part] = rowBoundary(part);
                counts[part] = rowBoundary(part + 1) - rowBoundary(part);
            }
        }
        MPI_Gatherv(localOutput.data(), localRows, MPI_DOUBLE, output.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(output, "OutputVector");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        index_t firstFailure = numRows;
        #pragma omp parallel for reduction(min:firstFailure) schedule(static) if(localRows >= 256)
        for (index_t row = 0; row < localRows; ++row) {
            const double ref = reference[row];
            const double res = localOutput[row];
            const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
            if (error > maxRelativeError) firstFailure = std::min(firstFailure, firstRow + row);
        }

        index_t failure;
        MPI_Allreduce(&firstFailure, &failure, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
        if (failure != numRows) {
            // Invert the floor-based row partition, including empty ranks.
            const int owner = ((offset_t{failure} + 1) * ranks - 1) / numRows;
            double result[2];
            if (rank == owner) {
                result[0] = reference[failure - firstRow];
                result[1] = localOutput[failure - firstRow];
            }
            MPI_Bcast(result, 2, MPI_DOUBLE, owner, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Validation failed at index %u: reference %.10e, got %.10e",
                       failure, result[0], result[1]);
                if (std::abs(result[0]) >= 1e-10) {
                    printf(" (rel error: %.10e)", std::abs((result[1] - result[0]) / result[0]));
                }
                printf("\nValidation: FAILED\n");
            }
            return 1;
        }
        if (rank == 0) printf("Validation: PASSED\n");
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
