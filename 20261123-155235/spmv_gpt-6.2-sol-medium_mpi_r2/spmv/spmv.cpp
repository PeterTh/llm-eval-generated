#include <algorithm>
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

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int ROOT = 0;
constexpr int TRANSFER_CHUNK = 1 << 18;

struct RowBlock {
    index_t first;
    index_t count;
};

RowBlock rowBlock(index_t rows, int rank, int ranks) {
    const index_t quotient = rows / ranks;
    const index_t remainder = rows % ranks;
    const auto peer = static_cast<index_t>(rank);
    return {peer * quotient + std::min(peer, remainder), quotient + (peer < remainder)};
}

template <typename T>
void sendBuffer(const T* data, offset_t size, MPI_Datatype type, int peer, int tag) {
    for (offset_t offset = 0; offset < size; offset += TRANSFER_CHUNK) {
        const int count = std::min<offset_t>(size - offset, TRANSFER_CHUNK);
        MPI_Send(data + offset, count, type, peer, tag, MPI_COMM_WORLD);
    }
}

template <typename T>
void receiveBuffer(T* data, offset_t size, MPI_Datatype type, int tag) {
    for (offset_t offset = 0; offset < size; offset += TRANSFER_CHUNK) {
        const int count = std::min<offset_t>(size - offset, TRANSFER_CHUNK);
        MPI_Recv(data + offset, count, type, ROOT, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void fill(double* A, const offset_t n, const double maxVal) {
    for (offset_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Visit blocks in global row order to preserve the original rand() sequence.
void initRandomRows(std::vector<index_t>& cols, std::vector<offset_t>& rowDelimiters,
                    index_t dim, RowBlock block, double probability, offset_t& needed) {
    cols.clear();
    rowDelimiters.resize(static_cast<offset_t>(block.count) + 1);
    for (index_t i = 0; i < block.count; ++i) {
        rowDelimiters[i] = cols.size();
        offset_t remaining = static_cast<offset_t>(dim - block.first - i) * dim;
        index_t j = 0;
        for (; j < dim && needed != 0 && remaining > needed; ++j, --remaining) {
            if (static_cast<double>(rand()) / RAND_MAX <= probability) {
                cols.push_back(j);
                --needed;
            }
        }
        // Once every remaining entry is required, the random draws are immaterial.
        if (needed != 0 && remaining == needed) {
            for (; j < dim; ++j) {
                cols.push_back(j);
                --needed;
            }
        }
    }
    rowDelimiters[block.count] = cols.size();
}

struct LocalMatrix {
    std::vector<offset_t> rowDelimiters;
    std::vector<index_t> cols;
    std::unique_ptr<double[]> val;
};

LocalMatrix initialize(index_t dim, offset_t nonzeros, double maxVal, RowBlock block,
                       int rank, int ranks, std::vector<double>& vec) {
    constexpr int ROWS_TAG = 0;
    constexpr int COLS_TAG = 1;
    constexpr int VALUES_TAG = 2;
    LocalMatrix matrix;
    std::vector<offset_t> nonzerosPerRank;

    if (rank == ROOT) {
        nonzerosPerRank.resize(ranks);
        const double probability = dim == 0 ? 0.0 :
            static_cast<double>(nonzeros) / (static_cast<double>(dim) * dim);
        offset_t needed = nonzeros;
        srand(8675309);
        matrix.cols.reserve(nonzeros / ranks);
        initRandomRows(matrix.cols, matrix.rowDelimiters, dim, block, probability, needed);
        nonzerosPerRank[ROOT] = matrix.cols.size();

        std::vector<index_t> cols;
        std::vector<offset_t> rowDelimiters;
        if (ranks > 1) {
            cols.reserve(nonzeros / ranks);
        }
        for (int peer = 1; peer < ranks; ++peer) {
            const RowBlock peerBlock = rowBlock(dim, peer, ranks);
            if (peerBlock.count == 0) {
                break;
            }
            initRandomRows(cols, rowDelimiters, dim, peerBlock, probability, needed);
            nonzerosPerRank[peer] = cols.size();
            sendBuffer(rowDelimiters.data(), rowDelimiters.size(), MPI_UINT64_T, peer, ROWS_TAG);
            sendBuffer(cols.data(), cols.size(), MPI_UINT32_T, peer, COLS_TAG);
        }
    } else {
        matrix.rowDelimiters.resize(static_cast<offset_t>(block.count) + 1);
        if (block.count != 0) {
            receiveBuffer(matrix.rowDelimiters.data(), matrix.rowDelimiters.size(),
                          MPI_UINT64_T, ROWS_TAG);
            matrix.cols.resize(matrix.rowDelimiters.back());
            receiveBuffer(matrix.cols.data(), matrix.cols.size(), MPI_UINT32_T, COLS_TAG);
        }
    }

    if (rank == ROOT) {
        // The original vector and matrix values shared the default rand() stream.
        srand(1);
        fill(vec.data(), dim, maxVal);
    }
    MPI_Bcast(vec.data(), static_cast<int>(dim), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    matrix.val = std::make_unique_for_overwrite<double[]>(matrix.cols.size());
    if (rank == ROOT) {
        fill(matrix.val.get(), matrix.cols.size(), maxVal);
        const auto bufferSize = ranks == 1 ? 0 : std::min<offset_t>(
            TRANSFER_CHUNK, *std::max_element(nonzerosPerRank.begin() + 1, nonzerosPerRank.end()));
        auto buffer = std::make_unique_for_overwrite<double[]>(bufferSize);
        for (int peer = 1; peer < ranks; ++peer) {
            for (offset_t offset = 0; offset < nonzerosPerRank[peer]; offset += TRANSFER_CHUNK) {
                const int count = std::min<offset_t>(nonzerosPerRank[peer] - offset, TRANSFER_CHUNK);
                fill(buffer.get(), count, maxVal);
                MPI_Send(buffer.get(), count, MPI_DOUBLE, peer, VALUES_TAG, MPI_COMM_WORLD);
            }
        }
    } else {
        receiveBuffer(matrix.val.get(), matrix.cols.size(), MPI_DOUBLE, VALUES_TAG);
    }
    return matrix;
}

// Each row keeps the original CSR accumulation order.
void spmvCpu(const double* val, const index_t* cols, const offset_t* rowDelimiters,
             const double* vec, const index_t rows, double* out) {
    for (index_t i = 0; i < rows; ++i) {
        double t = 0.0;
        for (offset_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

bool verifyResults(const double* reference, const double* result, RowBlock block, int rank) {
    constexpr index_t NO_MISMATCH = std::numeric_limits<index_t>::max();
    constexpr int MISMATCH_TAG = 3;
    index_t localMismatch = NO_MISMATCH;
    for (index_t i = 0; i < block.count; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            localMismatch = block.first + i;
            break;
        }
    }

    index_t mismatch = NO_MISMATCH;
    MPI_Allreduce(&localMismatch, &mismatch, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
    if (mismatch == NO_MISMATCH) {
        return true;
    }

    // Match the serial diagnostic by reporting only the first failing row.
    double values[2];
    if (localMismatch == mismatch) {
        const index_t i = mismatch - block.first;
        values[0] = reference[i];
        values[1] = result[i];
        if (rank != ROOT) {
            MPI_Send(values, 2, MPI_DOUBLE, ROOT, MISMATCH_TAG, MPI_COMM_WORLD);
        }
    }
    if (rank == ROOT) {
        if (localMismatch != mismatch) {
            MPI_Recv(values, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MISMATCH_TAG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        const double ref = values[0];
        const double res = values[1];
        if (std::abs(ref) < 1e-10) {
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", mismatch, ref, res);
        } else {
            printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                   mismatch, ref, res, std::abs((res - ref) / ref));
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
            if (rank == ROOT) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == ROOT) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (sparsity == 0 || numRows > std::numeric_limits<int>::max()) {
        if (rank == ROOT) {
            fprintf(stderr, "Sparsity must be positive and the row count must fit in an MPI count.\n");
        }
        return 1;
    }

    const offset_t entries = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = entries / sparsity;
    if (rank == ROOT) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    const RowBlock block = rowBlock(numRows, rank, ranks);
    std::vector<double> vec(numRows);
    const LocalMatrix matrix = initialize(numRows, nItems, maxVal, block, rank, ranks, vec);
    std::vector<double> out(block.count);
    std::vector<double> reference;
    if (validate) {
        if (rank == ROOT) {
            printf("Computing reference solution...\n");
        }
        reference.resize(block.count);
        spmvCpu(matrix.val.get(), matrix.cols.data(), matrix.rowDelimiters.data(),
                vec.data(), block.count, reference.data());
    }

    if (rank == ROOT) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (block.count != 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCpu(matrix.val.get(), matrix.cols.data(), matrix.rowDelimiters.data(),
                    vec.data(), block.count, out.data());
        }
    }

    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT) {
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations != 0 ? 1000.0 * duration / iterations : 0.0;
        printf("Computation time: %ld ms\n", static_cast<long>(1000.0 * duration));
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> result;
        if (rank == ROOT) {
            counts.resize(ranks);
            displacements.resize(ranks);
            result.resize(numRows);
            for (int peer = 0; peer < ranks; ++peer) {
                const RowBlock peerBlock = rowBlock(numRows, peer, ranks);
                counts[peer] = peerBlock.count;
                displacements[peer] = peerBlock.first;
            }
        }
        MPI_Gatherv(out.data(), static_cast<int>(block.count), MPI_DOUBLE, result.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if (rank == ROOT) {
            print_results(result, "OutputVector");
        }
    }

    if (validate) {
        if (rank == ROOT) {
            printf("Validating result...\n");
        }
        const bool valid = verifyResults(reference.data(), out.data(), block, rank);
        if (rank == ROOT) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 0;
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
