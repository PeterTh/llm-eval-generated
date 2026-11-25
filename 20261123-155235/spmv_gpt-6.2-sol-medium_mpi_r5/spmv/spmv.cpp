#include <mpi.h>

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

using index_t = uint32_t;
using count_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

// Keep the original rand() stream. On glibc, jump directly to the start of
// each rank's segment using the recurrence for the default 128-byte state.
class RandomSequence {
public:
    RandomSequence(unsigned seed, count_t skip) {
#if defined(__GLIBC__)
        random_data generator{};
        alignas(uint32_t) char buffer[128];
        initstate_r(seed, buffer, sizeof(buffer), &generator);
        for (auto& word : state) {
            const auto* updated = generator.fptr;
            int32_t output;
            random_r(&generator, &output);
            word = static_cast<uint32_t>(*updated);
        }

        if (skip != 0) {
            State step{};
            step[1] = 1;
            State coefficients{};
            coefficients[0] = 1;
            for (; skip != 0; skip >>= 1) {
                if (skip & 1) {
                    coefficients = multiply(coefficients, step);
                }
                step = multiply(step, step);
            }

            State shifted{};
            for (auto& word : shifted) {
                for (size_t i = 0; i < state.size(); ++i) {
                    word += coefficients[i] * state[i];
                }
                // Multiply by x modulo x^31 - x^28 - 1 for the next word.
                const auto high = coefficients.back();
                std::move_backward(coefficients.begin(), coefficients.end() - 1, coefficients.end());
                coefficients[0] = high;
                coefficients[28] += high;
            }
            state = shifted;
        }
#else
        // The C standard does not specify the recurrence of rand().
        srand(seed);
        for (count_t i = 0; i < skip; ++i) {
            rand();
        }
#endif
    }

    unsigned next() {
#if defined(__GLIBC__)
        const uint32_t word = state[head];
        state[head] += state[head < 3 ? head + 28 : head - 3];
        if (++head == state.size()) {
            head = 0;
        }
        return word >> 1;
#else
        return rand();
#endif
    }

private:
#if defined(__GLIBC__)
    using State = std::array<uint32_t, 31>;

    // Polynomial multiplication modulo x^31 - x^28 - 1, with coefficients
    // modulo 2^32, advances the additive random state by composed offsets.
    static State multiply(const State& left, const State& right) {
        std::array<uint32_t, 2 * State{}.size() - 1> product{};
        for (size_t i = 0; i < left.size(); ++i) {
            for (size_t j = 0; j < right.size(); ++j) {
                product[i + j] += left[i] * right[j];
            }
        }
        constexpr size_t degree = State{}.size();
        for (size_t i = product.size(); i-- > degree;) {
            product[i - 3] += product[i];
            product[i - degree] += product[i];
        }
        State result;
        std::copy_n(product.begin(), result.size(), result.begin());
        return result;
    }

    State state;
    size_t head = 0;
#endif
};

struct RowRange {
    index_t first;
    index_t count;
};

RowRange partitionRows(index_t dim, int rank, int ranks) {
    const index_t block = dim / ranks;
    const index_t remainder = dim % ranks;
    const auto part = static_cast<index_t>(rank);
    return {part * block + std::min(part, remainder), block + (part < remainder)};
}

struct LocalMatrix {
    std::vector<index_t> columns;
    std::vector<size_t> rowOffsets;
    std::unique_ptr<double[]> values;
};

void fill(double* data, size_t count, double maxVal, count_t skip) {
    if (count == 0) {
        return;
    }
    RandomSequence random(1, skip);
    for (size_t i = 0; i < count; ++i) {
        data[i] = maxVal * (random.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void appendDenseSuffix(LocalMatrix& matrix, index_t dim, index_t row, index_t col) {
    const size_t rows = matrix.rowOffsets.size() - 1;
    if (row == rows) {
        return;
    }
    matrix.columns.reserve(matrix.columns.size() + (rows - row) * dim - col);
    for (; row < rows; ++row) {
        for (; col < dim; ++col) {
            matrix.columns.push_back(col);
        }
        matrix.rowOffsets[row + 1] = matrix.columns.size();
        col = 0;
    }
}

// Retain only the allowed number of misses. Starting at the next miss, the
// original generator forces every remaining entry to be present.
void completeRows(LocalMatrix& matrix, index_t dim, count_t allowedMisses) {
    const auto& offsets = matrix.rowOffsets;
    index_t low = 0;
    auto high = static_cast<index_t>(offsets.size() - 1);
    while (low < high) {
        const index_t mid = low + (high - low) / 2;
        const count_t misses = static_cast<count_t>(mid + 1) * dim - offsets[mid + 1];
        if (misses > allowedMisses) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    const index_t row = low;
    const count_t rowMisses = allowedMisses - (static_cast<count_t>(row) * dim - offsets[row]);
    // For each hit, its column minus the preceding hits counts earlier misses.
    size_t first = offsets[row], last = offsets[row + 1];
    while (first < last) {
        const size_t mid = first + (last - first) / 2;
        if (matrix.columns[mid] - (mid - offsets[row]) > rowMisses) {
            last = mid;
        } else {
            first = mid + 1;
        }
    }
    matrix.columns.resize(first);
    appendDenseSuffix(matrix, dim, row, static_cast<index_t>(rowMisses + first - offsets[row]));
}

LocalMatrix initializeMatrix(index_t dim, count_t nItems, RowRange rows, double maxVal) {
    LocalMatrix matrix;
    matrix.rowOffsets.resize(static_cast<size_t>(rows.count) + 1);
    if (nItems == 0) {
        return matrix;
    }

    const count_t totalEntries = static_cast<count_t>(dim) * dim;
    const count_t firstEntry = static_cast<count_t>(rows.first) * dim;
    const count_t localEntries = static_cast<count_t>(rows.count) * dim;
    count_t firstValue = firstEntry;
    if (nItems == totalEntries) {
        appendDenseSuffix(matrix, dim, 0, 0);
    } else {
        if (localEntries != 0) {
            const double probability = static_cast<double>(nItems) / totalEntries;
            // Compare integers in the scan, keeping the original inclusive
            // floating-point cutoff even when the division rounds at the boundary.
            auto threshold = static_cast<unsigned>(probability * RAND_MAX);
            if (static_cast<double>(threshold) / RAND_MAX > probability) {
                --threshold;
            } else if (threshold < RAND_MAX &&
                       static_cast<double>(threshold + 1) / RAND_MAX <= probability) {
                ++threshold;
            }

            // Allow for ordinary variation in the hit count to avoid growing
            // and copying a nearly full column array near the end of the scan.
            const double expected = probability * localEntries;
            const auto capacity = static_cast<count_t>(expected + 6.0 * std::sqrt(expected) + 1.0);
            matrix.columns.reserve(std::min(capacity, localEntries));
            RandomSequence random(8675309, firstEntry);
            for (index_t row = 0; row < rows.count; ++row) {
                for (index_t col = 0; col < dim; ++col) {
                    if (random.next() <= threshold) {
                        matrix.columns.push_back(col);
                    }
                }
                matrix.rowOffsets[row + 1] = matrix.columns.size();
            }
        }

        const count_t localHits = matrix.columns.size();
        count_t hits;
        MPI_Scan(&localHits, &hits, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        const count_t priorHits = hits - localHits;

        // The serial scan either exhausts the hit quota or exhausts the miss
        // quota. The prefix counts locate both boundaries without a second scan.
        const count_t maxMisses = totalEntries - nItems;
        const count_t priorMisses = firstEntry - priorHits;
        firstValue = std::min(priorHits, nItems);
        if (priorMisses > maxMisses) {
            firstValue += priorMisses - maxMisses;
        }
        if (hits > nItems) {
            const auto keep = static_cast<size_t>(nItems - std::min(priorHits, nItems));
            matrix.columns.resize(keep);
            for (auto offset = matrix.rowOffsets.rbegin(); offset != matrix.rowOffsets.rend() &&
                 *offset > keep; ++offset) {
                *offset = keep;
            }
        } else if (localEntries != 0 && priorMisses + localEntries - localHits > maxMisses) {
            completeRows(matrix, dim, maxMisses - std::min(priorMisses, maxMisses));
        }
    }

    if (!matrix.columns.empty()) {
        matrix.values = std::make_unique_for_overwrite<double[]>(matrix.columns.size());
        fill(matrix.values.get(), matrix.columns.size(), maxVal, dim + firstValue);
    }
    return matrix;
}

void spmvCpu(const double* values, const index_t* columns, const size_t* rowOffsets,
             const double* input, index_t rows, double* output) {
    for (index_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (size_t i = rowOffsets[row]; i < rowOffsets[row + 1]; ++i) {
            sum += values[i] * input[columns[i]];
        }
        output[row] = sum;
    }
}

bool verifyResults(const std::vector<double>& reference, const std::vector<double>& result,
                   RowRange rows, index_t dim, int rank) {
    index_t firstFailure = dim;
    for (index_t i = 0; i < rows.count; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            firstFailure = rows.first + i;
            break;
        }
    }

    index_t globalFailure;
    MPI_Allreduce(&firstFailure, &globalFailure, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (globalFailure == dim) {
        return true;
    }

    double details[2];
    if (firstFailure == globalFailure) {
        const index_t i = firstFailure - rows.first;
        details[0] = reference[i];
        details[1] = result[i];
        if (rank != 0) {
            MPI_Send(details, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    } else if (rank == 0) {
        MPI_Recv(details, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank == 0) {
        const double ref = details[0], res = details[1];
        if (std::abs(ref) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", globalFailure, ref, res);
        } else {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   globalFailure, ref, res, std::abs((res - ref) / ref));
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be positive and the matrix dimension must fit an MPI count.\n");
        }
        return 1;
    }

    const count_t totalEntries = static_cast<count_t>(numRows) * numRows;
    const count_t nItems = totalEntries / sparsity;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / totalEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    const RowRange rows = partitionRows(numRows, rank, ranks);
    auto input = std::make_unique_for_overwrite<double[]>(numRows);
    if (rank == 0) {
        fill(input.get(), numRows, maxVal, 0);
    }
    MPI_Bcast(input.get(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const auto matrix = initializeMatrix(numRows, nItems, rows, maxVal);
    std::vector<double> output(rows.count);
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        reference.resize(rows.count);
        spmvCpu(matrix.values.get(), matrix.columns.data(), matrix.rowOffsets.data(),
                input.get(), rows.count, reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    // The input is fixed across iterations. Time only local multiplication
    // and report the slowest rank's elapsed time.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(matrix.values.get(), matrix.columns.data(), matrix.rowOffsets.data(),
                input.get(), rows.count, output.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double avgTime = iterations == 0 ? 0.0 : duration * 1000.0 / iterations;
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> globalOutput;
        if (rank == 0) {
            globalOutput.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const auto range = partitionRows(numRows, r, ranks);
                counts[r] = range.count;
                displacements[r] = range.first;
            }
        }
        MPI_Gatherv(output.data(), rows.count, MPI_DOUBLE, globalOutput.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalOutput, "OutputVector");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = verifyResults(reference, output, rows, numRows, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    MPI_Finalize();
    return status;
}
