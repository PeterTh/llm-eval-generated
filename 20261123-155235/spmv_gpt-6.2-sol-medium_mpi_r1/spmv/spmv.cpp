#include <mpi.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

// Preserve the original libc rand() streams at arbitrary offsets. On glibc,
// the underlying 32-bit sequence satisfies x[i + 31] = x[i + 28] + x[i].
// Exponentiation modulo x^31 - x^28 - 1 jumps to a rank's first element.
class BenchmarkRandom {
public:
    BenchmarkRandom(unsigned seed, offset_t offset) {
#ifdef __GLIBC__
        Window initial{};
        initial[28] = seed;
        uint32_t word = seed;
        for (size_t i = 1; i < initial.size(); ++i) {
            word = (16807ull * word) % 2147483647;
            initial[(i + 28) % initial.size()] = word;
        }

        // The seeded state starts 31 steps before the first update, and
        // glibc discards the first 310 updates.
        offset += 341;
        Window position{};
        Window power{};
        position[0] = 1;
        power[1] = 1;
        while (offset != 0) {
            if (offset & 1) {
                position = multiply(position, power);
            }
            offset >>= 1;
            if (offset != 0) {
                power = multiply(power, power);
            }
        }

        for (size_t i = 0; i < initial.size(); ++i) {
            for (size_t j = 0; j < initial.size(); ++j) {
                state_[i] += position[j] * initial[j];
            }
            const uint32_t high = position.back();
            for (size_t j = position.size() - 1; j != 0; --j) {
                position[j] = position[j - 1];
            }
            position[0] = high;
            position[28] += high;
        }
#else
        std::srand(seed);
        for (offset_t i = 0; i < offset; ++i) {
            std::rand();
        }
#endif
    }

    uint32_t next() {
#ifdef __GLIBC__
        const uint32_t value = state_[cursor_];
        state_[(cursor_ + 31) & 31] = value + state_[(cursor_ + 28) & 31];
        cursor_ = (cursor_ + 1) & 31;
        return value >> 1;
#else
        return std::rand();
#endif
    }

private:
#ifdef __GLIBC__
    using Window = std::array<uint32_t, 31>;

    static Window multiply(const Window& a, const Window& b) {
        std::array<uint32_t, 61> product{};
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                product[i + j] += a[i] * b[j];
            }
        }
        for (size_t i = product.size() - 1; i >= a.size(); --i) {
            product[i - 3] += product[i];
            product[i - 31] += product[i];
        }
        Window result;
        std::copy_n(product.begin(), result.size(), result.begin());
        return result;
    }

    std::array<uint32_t, 32> state_{};
    unsigned cursor_ = 0;
#endif
};

void fill(double* values, size_t count, double maxVal, offset_t offset) {
    if (count == 0) {
        return;
    }
    BenchmarkRandom random(1, offset);
    for (size_t i = 0; i < count; ++i) {
        values[i] = maxVal * (random.next() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

struct RowRange {
    index_t first;
    index_t size;
};

RowRange partitionRows(index_t dim, int rank, int ranks) {
    const index_t base = dim / ranks;
    const index_t extra = dim % ranks;
    const auto id = static_cast<index_t>(rank);
    return {id * base + std::min(id, extra), base + (id < extra)};
}

// Generate this rank's CSR structure locally and return the global offset
// into the independently generated matrix value stream.
offset_t initRandomMatrix(std::vector<index_t>& cols, std::vector<offset_t>& rowDelimiters,
                          offset_t nItems, index_t dim, RowRange rows, int rank) {
    if (nItems == 0) {
        return 0;
    }

    const offset_t totalCells = static_cast<offset_t>(dim) * dim;
    const offset_t firstCell = static_cast<offset_t>(rows.first) * dim;
    if (nItems == totalCells) {
        cols.resize(static_cast<offset_t>(rows.size) * dim);
        for (index_t i = 0; i < rows.size; ++i) {
            std::iota(cols.data() + rowDelimiters[i], cols.data() + rowDelimiters[i] + dim, 0u);
            rowDelimiters[i + 1] = rowDelimiters[i] + dim;
        }
        return firstCell;
    }

    const double prob = static_cast<double>(nItems) / (static_cast<double>(dim) * dim);
    uint32_t threshold = static_cast<uint32_t>(prob * RAND_MAX);
    // Match the original floating-point comparison at the integer boundary.
    if (static_cast<double>(threshold) / RAND_MAX > prob) {
        --threshold;
    } else if (threshold < RAND_MAX && static_cast<double>(threshold + 1) / RAND_MAX <= prob) {
        ++threshold;
    }

    if (rows.size != 0) {
        cols.reserve(static_cast<offset_t>(nItems * (static_cast<double>(rows.size) / dim)));
        BenchmarkRandom random(8675309, firstCell);
        for (index_t i = 0; i < rows.size; ++i) {
            for (index_t j = 0; j < dim; ++j) {
                if (random.next() <= threshold) {
                    cols.push_back(j);
                }
            }
            rowDelimiters[i + 1] = cols.size();
        }
    }

    const offset_t localCandidates = cols.size();
    offset_t candidatesBefore = 0;
    MPI_Exscan(&localCandidates, &candidatesBefore, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        candidatesBefore = 0;
    }

    // The serial generator keeps candidates until it reaches the nonzero
    // quota, or takes every cell once the remaining cells must all be filled.
    const auto itemsBefore = [=](offset_t cell, offset_t localPrefix) {
        const offset_t forced = nItems - std::min(nItems, totalCells - cell);
        return std::min(nItems, std::max(candidatesBefore + localPrefix, forced));
    };

    const offset_t firstItem = itemsBefore(firstCell, 0);
    const offset_t endCell = firstCell + static_cast<offset_t>(rows.size) * dim;
    if (itemsBefore(endCell, localCandidates) - firstItem == localCandidates) {
        return firstItem;
    }

    // The quota can only add entries or only remove entries across the
    // entire matrix. Rebuild just the suffix whose row counts changed.
    index_t row = 0;
    offset_t cell = firstCell;
    while (itemsBefore(cell + dim, rowDelimiters[row + 1]) - firstItem == rowDelimiters[row + 1]) {
        ++row;
        cell += dim;
    }
    cols.resize(rowDelimiters[row]);
    cols.reserve(itemsBefore(endCell, localCandidates) - firstItem);
    offset_t assigned = firstItem + cols.size();
    BenchmarkRandom random(8675309, cell);
    for (; row < rows.size; ++row) {
        for (index_t j = 0; j < dim && assigned < nItems; ++j, ++cell) {
            if (totalCells - cell <= nItems - assigned || random.next() <= threshold) {
                cols.push_back(j);
                ++assigned;
            }
        }
        rowDelimiters[row + 1] = cols.size();
    }
    return firstItem;
}

inline double sumRow(const double* val, const index_t* cols, const double* vec,
                     offset_t begin, offset_t end, double sum = 0.0) {
    for (offset_t j = begin; j < end; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    return sum;
}

void spmvReference(const double* val, const index_t* cols, const offset_t* rowDelimiters,
                   const double* vec, index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        out[i] = sumRow(val, cols, vec, rowDelimiters[i], rowDelimiters[i + 1]);
    }
}

void spmvCpu(const double* val, const index_t* cols, const offset_t* rowDelimiters,
             const double* vec, index_t dim, double* out) {
    // Interleave independent accumulations while keeping each row's sum in CSR order.
    index_t i = 0;
    for (; i + 1 < dim; i += 2) {
        offset_t first = rowDelimiters[i];
        offset_t second = rowDelimiters[i + 1];
        const offset_t firstEnd = second;
        const offset_t secondEnd = rowDelimiters[i + 2];
        const offset_t pairedEnd = first + std::min(firstEnd - first, secondEnd - second);
        double firstSum = 0.0;
        double secondSum = 0.0;
        for (; first < pairedEnd; ++first, ++second) {
            firstSum += val[first] * vec[cols[first]];
            secondSum += val[second] * vec[cols[second]];
        }
        out[i] = sumRow(val, cols, vec, first, firstEnd, firstSum);
        out[i + 1] = sumRow(val, cols, vec, second, secondEnd, secondSum);
    }
    if (i < dim) {
        out[i] = sumRow(val, cols, vec, rowDelimiters[i], rowDelimiters[i + 1]);
    }
}

double relativeError(double reference, double result) {
    if (std::abs(reference) < 1e-10) {
        return std::abs(result);
    }
    return std::abs((result - reference) / reference);
}

bool verifyResults(const std::vector<double>& reference, const std::vector<double>& result,
                   RowRange rows, index_t numRows, int rank) {
    struct {
        int index;
        int rank;
    } local{static_cast<int>(numRows), rank}, first{};
    for (index_t i = 0; i < rows.size; ++i) {
        if (relativeError(reference[i], result[i]) > MAX_RELATIVE_ERROR) {
            local.index = rows.first + i;
            break;
        }
    }
    MPI_Allreduce(&local, &first, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    if (first.index == static_cast<int>(numRows)) {
        return true;
    }

    double values[2];
    if (rank == first.rank) {
        const index_t i = first.index - rows.first;
        values[0] = reference[i];
        values[1] = result[i];
    }
    MPI_Bcast(values, 2, MPI_DOUBLE, first.rank, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Validation failed at index %d: reference %.10e, got %.10e",
               first.index, values[0], values[1]);
        if (std::abs(values[0]) >= 1e-10) {
            printf(" (rel error: %.10e)", relativeError(values[0], values[1]));
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

    if (sparsity == 0 || numRows > std::numeric_limits<int>::max()) {
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be positive and the row count must fit in an MPI count.\n");
        }
        return 1;
    }

    const offset_t matrixCells = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = matrixCells / sparsity;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, matrixCells ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixCells) : 0.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    const RowRange rows = partitionRows(numRows, rank, ranks);
    std::vector<double> vec(numRows);
    if (rank == 0) {
        fill(vec.data(), vec.size(), maxVal, 0);
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<index_t> cols;
    std::vector<offset_t> rowDelimiters(static_cast<size_t>(rows.size) + 1);
    const offset_t firstItem = initRandomMatrix(cols, rowDelimiters, nItems, numRows, rows, rank);
    std::vector<double> val(cols.size());
    fill(val.data(), val.size(), maxVal, numRows + firstItem);
    std::vector<double> out(rows.size);

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
        }
        reference.resize(rows.size);
        spmvReference(val.data(), cols.data(), rowDelimiters.data(), vec.data(), rows.size,
                      reference.data());
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (rows.size != 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCpu(val.data(), cols.data(), rowDelimiters.data(), vec.data(), rows.size, out.data());
            // Keep each timed multiply observable to the compiler.
            asm volatile("" : : "g"(out.data()) : "memory");
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;
        const double gflops = duration > 0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> globalOut;
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            globalOut.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int i = 0; i < ranks; ++i) {
                const RowRange part = partitionRows(numRows, i, ranks);
                counts[i] = part.size;
                displacements[i] = part.first;
            }
        }
        MPI_Gatherv(out.data(), rows.size, MPI_DOUBLE, globalOut.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(globalOut, "OutputVector");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = verifyResults(reference, out, rows, numRows, rank);
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

    const int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
