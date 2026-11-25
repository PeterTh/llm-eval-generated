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
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

namespace {

// The original Linux rand() stream uses a degree-31 additive recurrence.
// Jump ahead in that same stream so each rank and thread can generate its
// own contiguous part of the input without replaying preceding draws.
class BenchmarkRandom {
public:
    BenchmarkRandom(uint32_t seed, offset_t skip) {
        std::array<uint32_t, 344> sequence{};
        sequence[0] = seed;
        for (int i = 1; i < degree; ++i) {
            sequence[i] = (16807ull * sequence[i - 1]) % 2147483647;
        }
        for (int i = degree; i < degree + 3; ++i) {
            sequence[i] = sequence[i - degree];
        }
        for (int i = degree + 3; i < static_cast<int>(sequence.size()); ++i) {
            sequence[i] = sequence[i - degree] + sequence[i - 3];
        }

        Polynomial power{};
        power[0] = 1;
        for (size_t bit = 0; skip != 0; ++bit, skip >>= 1) {
            if (skip & 1) {
                power = multiply(power, jumpPowers()[bit]);
            }
        }
        for (auto& value : state_) {
            value = 0;
            for (int i = 0; i < degree; ++i) {
                value += power[i] * sequence[sequence.size() - degree + i];
            }
            // Multiply by x modulo x^31 - x^28 - 1.
            const auto high = power.back();
            for (int i = degree - 1; i > 0; --i) {
                power[i] = power[i - 1];
            }
            power[28] += high;
            power[0] = high;
        }
    }

    uint32_t next() {
        const auto value = state_[oldest_] + state_[lagged_];
        state_[oldest_] = value;
        oldest_ = oldest_ == degree - 1 ? 0 : oldest_ + 1;
        lagged_ = lagged_ == degree - 1 ? 0 : lagged_ + 1;
        return value >> 1;
    }

    static constexpr uint32_t max = 2147483647;

private:
    static constexpr int degree = 31;
    using Polynomial = std::array<uint32_t, degree>;

    static Polynomial multiply(const Polynomial& a, const Polynomial& b) {
        std::array<uint32_t, 2 * degree - 1> terms{};
        for (int i = 0; i < degree; ++i) {
            for (int j = 0; j < degree; ++j) {
                terms[i + j] += a[i] * b[j];
            }
        }
        for (int i = static_cast<int>(terms.size()) - 1; i >= degree; --i) {
            terms[i - 3] += terms[i];
            terms[i - degree] += terms[i];
        }
        Polynomial result;
        std::copy_n(terms.begin(), degree, result.begin());
        return result;
    }

    static const std::array<Polynomial, 64>& jumpPowers() {
        static const auto powers = [] {
            std::array<Polynomial, 64> result{};
            result[0][1] = 1;
            for (size_t i = 1; i < result.size(); ++i) {
                result[i] = multiply(result[i - 1], result[i - 1]);
            }
            return result;
        }();
        return powers;
    }

    Polynomial state_;
    int oldest_ = 0;
    int lagged_ = degree - 3;
};

int hostThreads(offset_t work, offset_t grain) {
    return static_cast<int>(std::min<offset_t>(omp_get_max_threads(), 1 + work / grain));
}

void fill(double* values, offset_t size, double maxVal, offset_t skip = 0) {
    if (size == 0) {
        return;
    }
    const int chunks = hostThreads(size, 32768);
    #pragma omp parallel for num_threads(chunks) schedule(static)
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const offset_t begin = size * chunk / chunks;
        const offset_t end = size * (chunk + 1) / chunks;
        BenchmarkRandom random(1, skip + begin);
        for (offset_t i = begin; i < end; ++i) {
            values[i] = maxVal * (random.next() / (BenchmarkRandom::max + 1.0));
        }
    }
}

index_t rowBoundary(index_t rows, int part, int parts) {
    return static_cast<index_t>(static_cast<offset_t>(rows) * part / parts);
}

struct LocalMatrix {
    std::vector<offset_t> offsets;
    std::unique_ptr<index_t[]> columns;
    std::unique_ptr<double[]> values;
    offset_t size = 0;
    offset_t firstValue = 0;
};

struct CandidateRows {
    std::vector<index_t> columns;
    offset_t prefix = 0;
};

offset_t selectedBefore(offset_t hits, offset_t position, offset_t entries, offset_t nnz) {
    // Cap random hits at the requested count, and account for any entries
    // already forced by the number of positions left in the matrix.
    const offset_t remaining = entries - position;
    const offset_t mustHaveSelected = remaining < nnz ? nnz - remaining : 0;
    return std::min(nnz, std::max(hits, mustHaveSelected));
}

LocalMatrix initRandomMatrix(index_t firstRow, index_t rows, index_t dim, offset_t nnz, int rank) {
    LocalMatrix matrix;
    matrix.offsets.resize(static_cast<size_t>(rows) + 1);
    if (nnz == 0) {
        return matrix;
    }

    const offset_t entries = static_cast<offset_t>(dim) * dim;
    const offset_t localEntries = static_cast<offset_t>(rows) * dim;
    const int chunks = std::min<int>(std::max<index_t>(rows, 1), hostThreads(localEntries, 131072));
    const bool dense = nnz == entries;
    const double probability = static_cast<double>(nnz) / static_cast<double>(entries);
    auto threshold = static_cast<uint32_t>(probability * BenchmarkRandom::max);
    // Match the original inclusive floating-point comparison at the boundary.
    if (threshold < BenchmarkRandom::max &&
        static_cast<double>(threshold + 1) / BenchmarkRandom::max <= probability) {
        ++threshold;
    } else if (static_cast<double>(threshold) / BenchmarkRandom::max > probability) {
        --threshold;
    }

    std::vector<CandidateRows> candidates(chunks);
    offset_t localHits = localEntries;
    if (!dense) {
        #pragma omp parallel for num_threads(chunks) schedule(static)
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const index_t begin = rowBoundary(rows, chunk, chunks);
            const index_t end = rowBoundary(rows, chunk + 1, chunks);
            const auto positions = static_cast<offset_t>(end - begin) * dim;
            const double expectedHits = probability * positions;
            auto& columns = candidates[chunk].columns;
            columns.reserve(std::min(positions, static_cast<offset_t>(
                expectedHits + 4 * std::sqrt(expectedHits) + 8)));

            BenchmarkRandom random(8675309, static_cast<offset_t>(firstRow + begin) * dim);
            for (index_t row = begin; row < end; ++row) {
                matrix.offsets[row] = columns.size();
                for (index_t col = 0; col < dim; ++col) {
                    if (random.next() <= threshold) {
                        columns.push_back(col);
                    }
                }
            }
        }
        localHits = 0;
        for (auto& chunk : candidates) {
            chunk.prefix = localHits;
            localHits += chunk.columns.size();
        }
    }

    offset_t rankHits = 0;
    if (dense) {
        rankHits = static_cast<offset_t>(firstRow) * dim;
    } else {
        MPI_Exscan(&localHits, &rankHits, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            rankHits = 0;
        }
    }
    const offset_t firstEntry = static_cast<offset_t>(firstRow) * dim;
    const offset_t firstValue = selectedBefore(rankHits, firstEntry, entries, nnz);
    matrix.firstValue = firstValue;
    matrix.size = selectedBefore(rankHits + localHits, firstEntry + localEntries, entries, nnz) -
                  firstValue;
    matrix.offsets[rows] = matrix.size;
    matrix.columns.reset(new index_t[matrix.size]);
    matrix.values.reset(new double[matrix.size]);

    #pragma omp parallel for num_threads(chunks) schedule(static)
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const index_t begin = rowBoundary(rows, chunk, chunks);
        const index_t end = rowBoundary(rows, chunk + 1, chunks);
        auto& candidate = candidates[chunk];
        for (index_t row = begin; row < end; ++row) {
            const auto position = firstEntry + static_cast<offset_t>(row) * dim;
            if (dense) {
                matrix.offsets[row] = static_cast<offset_t>(row) * dim;
                std::iota(matrix.columns.get() + matrix.offsets[row],
                          matrix.columns.get() + matrix.offsets[row] + dim, index_t{0});
                continue;
            }
            const auto beginHit = matrix.offsets[row];
            const auto endHit = row + 1 < end ? matrix.offsets[row + 1] : candidate.columns.size();
            const auto hits = rankHits + candidate.prefix + beginHit;
            const auto before = selectedBefore(hits, position, entries, nnz);
            const auto after = selectedBefore(hits + endHit - beginHit, position + dim, entries, nnz);
            matrix.offsets[row] = before - firstValue;
            if (before == after) {
                continue;
            }

            auto* output = matrix.columns.get() + matrix.offsets[row];
            if (before > hits) {
                std::iota(output, output + dim, index_t{0});
                continue;
            }

            // Once remaining positions equal the number still needed, the
            // original generator assigns every remaining position. Overlap
            // that dense suffix with the random hits counted for this row.
            const auto rawAfter = hits + endHit - beginHit;
            const auto trailing = after > rawAfter ? after - rawAfter : 0;
            if (trailing == 0) {
                std::copy_n(candidate.columns.data() + beginHit, after - before, output);
                continue;
            }

            // In a forced row, walk backwards to find where the required
            // additional entries (misses in the random stream) begin.
            offset_t hit = endHit;
            index_t col = dim;
            for (offset_t misses = trailing; misses != 0;) {
                --col;
                if (hit > beginHit && candidate.columns[hit - 1] == col) {
                    --hit;
                } else {
                    --misses;
                }
            }
            if (hit != beginHit) {
                output = std::copy_n(candidate.columns.data() + beginHit, hit - beginHit, output);
            }
            std::iota(output, output + (dim - col), col);
        }
        std::vector<index_t>().swap(candidate.columns);
    }

    return matrix;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

void selectDevice() {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        throw std::runtime_error("no visible CUDA devices");
    }
    checkCuda(cudaSetDevice(localRank % devices));
}

template<typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(offset_t size) : size_(size) {
        if (size != 0) {
            checkCuda(cudaMalloc(&data_, size * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }

    void upload(const T* source) const {
        if (size_ != 0) {
            checkCuda(cudaMemcpy(data_, source, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void download(T* destination) const {
        if (size_ != 0) {
            checkCuda(cudaMemcpy(destination, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

private:
    offset_t size_;
    T* data_ = nullptr;
};

constexpr int BLOCK_THREADS = 128;
constexpr int WARP_THREADS = 32;

template<int lanes>
__device__ double warpSum(double sum, unsigned int mask) {
    for (int delta = lanes / 2; delta > 0; delta /= 2) {
        sum += __shfl_down_sync(mask, sum, delta, lanes);
    }
    return sum;
}

template<int lanes>
__global__ void spmvGpu(const double* __restrict__ values, const index_t* __restrict__ columns,
                        const offset_t* __restrict__ offsets, const double* __restrict__ vec,
                        index_t rows, double* __restrict__ out) {
    constexpr bool blockPerRow = lanes > WARP_THREADS;
    const unsigned int lane = threadIdx.x % lanes;
    const index_t row = blockPerRow ? blockIdx.x :
                        blockIdx.x * (BLOCK_THREADS / lanes) + threadIdx.x / lanes;
    unsigned int mask = 0xffffffffu;
    if constexpr (!blockPerRow) {
        if constexpr (lanes > 1) {
            mask = __ballot_sync(mask, row < rows);
        }
        if (row >= rows) {
            return;
        }
    }

    double sum = 0.0;
    const auto end = offsets[row + 1];
    for (auto i = offsets[row] + lane; i < end; i += lanes) {
        sum = fma(values[i], vec[columns[i]], sum);
    }

    if constexpr (blockPerRow) {
        __shared__ double warpSums[BLOCK_THREADS / WARP_THREADS];
        sum = warpSum<WARP_THREADS>(sum, mask);
        if (lane % WARP_THREADS == 0) {
            warpSums[lane / WARP_THREADS] = sum;
        }
        __syncthreads();
        if (lane < BLOCK_THREADS / WARP_THREADS) {
            const auto total = warpSum<BLOCK_THREADS / WARP_THREADS>(
                warpSums[lane], (1u << (BLOCK_THREADS / WARP_THREADS)) - 1);
            if (lane == 0) {
                out[row] = total;
            }
        }
    } else {
        sum = warpSum<lanes>(sum, mask);
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

template<int lanes>
void launchSpmv(const double* values, const index_t* columns, const offset_t* offsets,
                const double* vec, index_t rows, double* out) {
    constexpr int rowsPerBlock = BLOCK_THREADS / lanes;
    spmvGpu<lanes><<<(rows + rowsPerBlock - 1) / rowsPerBlock, BLOCK_THREADS>>>(
        values, columns, offsets, vec, rows, out);
}

auto spmvLauncher(offset_t averageRowLength) {
    if (averageRowLength >= 512) return launchSpmv<BLOCK_THREADS>;
    if (averageRowLength >= 48) return launchSpmv<WARP_THREADS>;
    if (averageRowLength >= 12) return launchSpmv<8>;
    if (averageRowLength >= 4) return launchSpmv<4>;
    return launchSpmv<1>;
}

void spmvCpu(const LocalMatrix& matrix, const double* vec, index_t rows, double* out) {
    const int threads = hostThreads(matrix.size + rows, 32768);
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (auto i = matrix.offsets[row]; i < matrix.offsets[row + 1]; ++i) {
            sum += matrix.values[i] * vec[matrix.columns[i]];
        }
        out[row] = sum;
    }
}

bool verifyResults(const std::vector<double>& reference, const std::vector<double>& result,
                   index_t firstRow, index_t dim, int rank) {
    const index_t rows = static_cast<index_t>(reference.size());
    index_t firstError = dim;
    const int threads = hostThreads(rows, 32768);
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:firstError)
    for (index_t row = 0; row < rows; ++row) {
        const double ref = reference[row];
        const double res = result[row];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            firstError = std::min(firstError, firstRow + row);
        }
    }
    index_t globalError;
    MPI_Allreduce(&firstError, &globalError, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (globalError == dim) {
        return true;
    }

    double detail[2]{};
    if (firstError == globalError) {
        detail[0] = reference[firstError - firstRow];
        detail[1] = result[firstError - firstRow];
    }
    double report[2];
    MPI_Reduce(detail, report, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (std::abs(report[0]) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   globalError, report[0], report[1]);
        } else {
            const double relError = std::abs((report[1] - report[0]) / report[0]);
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   globalError, report[0], report[1], relError);
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

int run(int argc, char** argv, int rank, int ranks) {
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

    if (sparsity == 0 || numRows > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity\n");
        return 1;
    }
    selectDevice();

    const auto entries = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = entries / sparsity;
    const index_t firstRow = rowBoundary(numRows, rank, ranks);
    const index_t rows = rowBoundary(numRows, rank + 1, ranks) - firstRow;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nItems,
               entries == 0 ? 100.0 : 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> reference;
    const bool active = rows != 0 && iterations != 0;
    auto matrix = initRandomMatrix(firstRow, rows, numRows, nItems, rank);
    std::vector<double> vec(rows != 0 ? numRows : 0);
    fill(vec.data(), vec.size(), maxVal);
    fill(matrix.values.get(), matrix.size, maxVal, numRows + matrix.firstValue);

    DeviceBuffer<double> dValues(active ? matrix.size : 0);
    DeviceBuffer<index_t> dColumns(active ? matrix.size : 0);
    DeviceBuffer<offset_t> dOffsets(active ? matrix.offsets.size() : 0);
    DeviceBuffer<double> dVec(active ? vec.size() : 0);
    DeviceBuffer<double> dOut(active ? rows : 0);
    dValues.upload(matrix.values.get());
    dColumns.upload(matrix.columns.get());
    dOffsets.upload(matrix.offsets.data());
    dVec.upload(vec.data());

    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(rows);
        spmvCpu(matrix, vec.data(), rows, reference.data());
    }
    matrix = LocalMatrix{};
    std::vector<double>().swap(vec);

    const auto launch = spmvLauncher(numRows == 0 ? 0 : nItems / numRows);
    if (active) {
        // Finish first-launch setup before measuring the steady-state loop.
        launch(dValues.data(), dColumns.data(), dOffsets.data(), dVec.data(), rows, dOut.data());
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            launch(dValues.data(), dColumns.data(), dOffsets.data(), dVec.data(), rows, dOut.data());
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double avgTime = iterations == 0 ? 0.0 : duration * 1000.0 / iterations;
        const double gflops = duration == 0.0 ? 0.0 : 2.0 * nItems * iterations / duration / 1e9;
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> out;
    if (printResults || validate) {
        out.resize(rows);
        dOut.download(out.data());
    }
    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> output;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            output.resize(numRows);
            for (int r = 0; r < ranks; ++r) {
                displacements[r] = rowBoundary(numRows, r, ranks);
                counts[r] = rowBoundary(numRows, r + 1, ranks) - displacements[r];
            }
        }
        MPI_Gatherv(out.data(), rows, MPI_DOUBLE, output.data(), counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(output, "OutputVector");
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = verifyResults(reference, out, firstRow, numRows, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
        }
        const int status = run(argc, argv, rank, ranks);
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
