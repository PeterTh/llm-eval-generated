#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

namespace {

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

struct Block {
    size_t offset;
    size_t size;
};

// A remainder-aware one-dimensional block decomposition.
constexpr Block splitBlock(const size_t N, const int parts,
                           const int coordinate) noexcept {
    const size_t base = N / static_cast<size_t>(parts);
    const size_t remainder = N % static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    return {coord * base + std::min(coord, remainder),
            base + (coord < remainder ? 1U : 0U)};
}

// Each process owns a two-dimensional tile of C.  It needs only the matching
// row slab of A and column slab of B; deterministic input generation means
// neither slab has to be communicated from a root process.
void initLocalMatrices(std::vector<double>& A, std::vector<double>& B,
                       const size_t N, const Block rows, const Block columns) {
    for (size_t i = 0; i < rows.size; ++i) {
        const size_t globalRow = rows.offset + i;
        double* const aRow = A.data() + i * N;
        for (size_t k = 0; k < N; ++k) {
            aRow[k] = getPseudoRndValue(N, globalRow, k);
        }
    }

    for (size_t k = 0; k < N; ++k) {
        double* const bRow = B.data() + k * columns.size;
        for (size_t j = 0; j < columns.size; ++j) {
            bRow[j] = getPseudoRndValue(N, k, columns.offset + j);
        }
    }
}

// Register- and cache-blocked local GEMM.  On AVX2 hosts the 4x8 microkernel
// holds 32 outputs in registers while traversing K, amortizing each A/B load
// over multiple FMAs and eliminating intermediate loads/stores of C.
void matrixMultiplyLocal(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, const size_t N,
                         const size_t localRows, const size_t localColumns) {
#if defined(__AVX2__) && defined(__FMA__)
    const double* const aData = A.data();
    const double* const bData = B.data();
    double* const cData = C.data();
    size_t i = 0;

    for (; i + 3 < localRows; i += 4) {
        size_t j = 0;
        for (; j + 7 < localColumns; j += 8) {
            __m256d c00 = _mm256_setzero_pd();
            __m256d c01 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd();
            __m256d c11 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd();
            __m256d c21 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd();
            __m256d c31 = _mm256_setzero_pd();

            for (size_t k = 0; k < N; ++k) {
                __m256d b0 =
                    _mm256_loadu_pd(bData + k * localColumns + j);
                __m256d b1 =
                    _mm256_loadu_pd(bData + k * localColumns + j + 4);
#if defined(__GNUC__) || defined(__clang__)
                // Keep these reused operands in registers instead of folding
                // the same load into each of the four dependent FMAs.
                asm("" : "+x"(b0), "+x"(b1));
#endif
                const __m256d a0 = _mm256_broadcast_sd(aData + i * N + k);
                const __m256d a1 =
                    _mm256_broadcast_sd(aData + (i + 1) * N + k);
                const __m256d a2 =
                    _mm256_broadcast_sd(aData + (i + 2) * N + k);
                const __m256d a3 =
                    _mm256_broadcast_sd(aData + (i + 3) * N + k);
                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a0, b1, c01);
                c10 = _mm256_fmadd_pd(a1, b0, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a2, b0, c20);
                c21 = _mm256_fmadd_pd(a2, b1, c21);
                c30 = _mm256_fmadd_pd(a3, b0, c30);
                c31 = _mm256_fmadd_pd(a3, b1, c31);
            }

            _mm256_storeu_pd(cData + i * localColumns + j, c00);
            _mm256_storeu_pd(cData + i * localColumns + j + 4, c01);
            _mm256_storeu_pd(cData + (i + 1) * localColumns + j, c10);
            _mm256_storeu_pd(cData + (i + 1) * localColumns + j + 4, c11);
            _mm256_storeu_pd(cData + (i + 2) * localColumns + j, c20);
            _mm256_storeu_pd(cData + (i + 2) * localColumns + j + 4, c21);
            _mm256_storeu_pd(cData + (i + 3) * localColumns + j, c30);
            _mm256_storeu_pd(cData + (i + 3) * localColumns + j + 4, c31);
        }

        for (; j + 3 < localColumns; j += 4) {
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            for (size_t k = 0; k < N; ++k) {
                __m256d b =
                    _mm256_loadu_pd(bData + k * localColumns + j);
#if defined(__GNUC__) || defined(__clang__)
                asm("" : "+x"(b));
#endif
                c0 = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(aData + i * N + k), b, c0);
                c1 = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(aData + (i + 1) * N + k), b, c1);
                c2 = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(aData + (i + 2) * N + k), b, c2);
                c3 = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(aData + (i + 3) * N + k), b, c3);
            }
            _mm256_storeu_pd(cData + i * localColumns + j, c0);
            _mm256_storeu_pd(cData + (i + 1) * localColumns + j, c1);
            _mm256_storeu_pd(cData + (i + 2) * localColumns + j, c2);
            _mm256_storeu_pd(cData + (i + 3) * localColumns + j, c3);
        }

        for (; j < localColumns; ++j) {
            double c0 = 0.0;
            double c1 = 0.0;
            double c2 = 0.0;
            double c3 = 0.0;
            for (size_t k = 0; k < N; ++k) {
                const double b = bData[k * localColumns + j];
                c0 = std::fma(aData[i * N + k], b, c0);
                c1 = std::fma(aData[(i + 1) * N + k], b, c1);
                c2 = std::fma(aData[(i + 2) * N + k], b, c2);
                c3 = std::fma(aData[(i + 3) * N + k], b, c3);
            }
            cData[i * localColumns + j] = c0;
            cData[(i + 1) * localColumns + j] = c1;
            cData[(i + 2) * localColumns + j] = c2;
            cData[(i + 3) * localColumns + j] = c3;
        }
    }

    for (; i < localRows; ++i) {
        size_t j = 0;
        for (; j + 7 < localColumns; j += 8) {
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            for (size_t k = 0; k < N; ++k) {
                const __m256d a = _mm256_broadcast_sd(aData + i * N + k);
                c0 = _mm256_fmadd_pd(
                    a, _mm256_loadu_pd(bData + k * localColumns + j), c0);
                c1 = _mm256_fmadd_pd(
                    a, _mm256_loadu_pd(bData + k * localColumns + j + 4),
                    c1);
            }
            _mm256_storeu_pd(cData + i * localColumns + j, c0);
            _mm256_storeu_pd(cData + i * localColumns + j + 4, c1);
        }
        for (; j + 3 < localColumns; j += 4) {
            __m256d c = _mm256_setzero_pd();
            for (size_t k = 0; k < N; ++k) {
                c = _mm256_fmadd_pd(
                    _mm256_broadcast_sd(aData + i * N + k),
                    _mm256_loadu_pd(bData + k * localColumns + j), c);
            }
            _mm256_storeu_pd(cData + i * localColumns + j, c);
        }
        for (; j < localColumns; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum = std::fma(aData[i * N + k],
                               bData[k * localColumns + j], sum);
            }
            cData[i * localColumns + j] = sum;
        }
    }
#else
    constexpr size_t rowTile = 32;
    constexpr size_t columnTile = 128;
    constexpr size_t innerTile = 256;

    std::fill(C.begin(), C.end(), 0.0);

    for (size_t ii = 0; ii < localRows; ii += rowTile) {
        const size_t iEnd = std::min(ii + rowTile, localRows);
        for (size_t jj = 0; jj < localColumns; jj += columnTile) {
            const size_t jEnd = std::min(jj + columnTile, localColumns);
            for (size_t kk = 0; kk < N; kk += innerTile) {
                const size_t kEnd = std::min(kk + innerTile, N);
                for (size_t k = kk; k < kEnd; ++k) {
                    const double* const bRow = B.data() + k * localColumns;
                    for (size_t i = ii; i < iEnd; ++i) {
                        const double a = A[i * N + k];
                        double* const cRow = C.data() + i * localColumns;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
                        for (size_t j = jj; j < jEnd; ++j) {
                            cRow[j] += a * bRow[j];
                        }
                    }
                }
            }
        }
    }
#endif
}

struct ValidationFailure {
    uint64_t i = 0;
    uint64_t j = 0;
    double expected = 0.0;
    double actual = 0.0;
    double relativeError = 0.0;
};

bool validateResult(const std::vector<double>& C, const size_t N,
                    const Block rows, const Block columns, const int rank,
                    const int processCount, MPI_Comm communicator,
                    ValidationFailure& failure) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    bool localValid = true;

    for (const size_t pi : checkPoints) {
        const size_t i = pi % N;
        if (i < rows.offset || i >= rows.offset + rows.size) {
            continue;
        }
        for (const size_t pj : checkPoints) {
            const size_t j = pj % N;
            if (j < columns.offset || j >= columns.offset + columns.size) {
                continue;
            }

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) *
                            getPseudoRndValue(N, k, j);
            }

            const double actual =
                C[(i - rows.offset) * columns.size + (j - columns.offset)];
            const double relativeError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relativeError > 1e-6 && localValid) {
                localValid = false;
                failure = {static_cast<uint64_t>(i), static_cast<uint64_t>(j),
                           expected, actual, relativeError};
            }
        }
    }

    int failingRank = localValid ? processCount : rank;
    MPI_Allreduce(MPI_IN_PLACE, &failingRank, 1, MPI_INT, MPI_MIN,
                  communicator);
    if (failingRank == processCount) {
        return true;
    }

    MPI_Bcast(&failure, static_cast<int>(sizeof(failure)), MPI_BYTE,
              failingRank, communicator);
    return false;
}

// Assemble the distributed tiles directly into row-major order on rank zero.
// Derived datatypes avoid both a second N*N packing buffer and MPI's int-sized
// aggregate count limit.  This path runs only when result printing is asked for.
std::vector<double> gatherResult(const std::vector<double>& localC,
                                 const size_t N, const int rank,
                                 const int processCount, const int dims[2],
                                 const Block localRows,
                                 const Block localColumns,
                                 MPI_Comm communicator, bool& succeeded) {
    constexpr int resultTag = 1001;
    std::vector<double> result;
    std::vector<MPI_Request> requests;
    std::vector<MPI_Datatype> receiveTypes;
    int allocationSucceeded = 1;

    // Establish that rank zero can hold the requested diagnostic result before
    // any sender enters a potentially blocking transfer.
    if (rank == 0) {
        try {
            result.resize(N * N);
            requests.reserve(static_cast<size_t>(processCount - 1));
            receiveTypes.reserve(static_cast<size_t>(processCount - 1));
        } catch (...) {
            allocationSucceeded = 0;
        }
    }
    MPI_Bcast(&allocationSucceeded, 1, MPI_INT, 0, communicator);
    succeeded = allocationSucceeded != 0;
    if (!succeeded) {
        return result;
    }

    if (rank != 0) {
        if (localRows.size != 0 && localColumns.size != 0) {
            MPI_Datatype tileType = MPI_DATATYPE_NULL;
            MPI_Type_vector(static_cast<int>(localRows.size),
                            static_cast<int>(localColumns.size),
                            static_cast<int>(localColumns.size), MPI_DOUBLE,
                            &tileType);
            MPI_Type_commit(&tileType);
            MPI_Send(localC.data(), 1, tileType, 0, resultTag, communicator);
            MPI_Type_free(&tileType);
        }
        return result;
    }

    for (int source = 0; source < processCount; ++source) {
        int coordinates[2] = {source / dims[1], source % dims[1]};
        const Block sourceRows = splitBlock(N, dims[0], coordinates[0]);
        const Block sourceColumns = splitBlock(N, dims[1], coordinates[1]);
        if (sourceRows.size == 0 || sourceColumns.size == 0) {
            continue;
        }

        if (source == 0) {
            for (size_t i = 0; i < sourceRows.size; ++i) {
                std::copy_n(localC.data() + i * sourceColumns.size,
                            sourceColumns.size,
                            result.data() + (sourceRows.offset + i) * N +
                                sourceColumns.offset);
            }
            continue;
        }

        MPI_Datatype receiveType = MPI_DATATYPE_NULL;
        MPI_Type_vector(static_cast<int>(sourceRows.size),
                        static_cast<int>(sourceColumns.size),
                        static_cast<int>(N), MPI_DOUBLE, &receiveType);
        MPI_Type_commit(&receiveType);
        receiveTypes.push_back(receiveType);
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(result.data() + sourceRows.offset * N + sourceColumns.offset,
                  1, receiveTypes.back(), source, resultTag, communicator,
                  &requests.back());
    }

    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
    }
    for (MPI_Datatype& datatype : receiveTypes) {
        MPI_Type_free(&datatype);
    }
    return result;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf(
        "  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Options {
    uint64_t N = 512;
    int validate = 0;
    int printResults = 0;
    int status = 0;  // 0: run, 1: error, 2: help
};

Options parseOptions(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (argv[i][0] == '\0' || end == nullptr || *end != '\0' ||
                parsed == 0 ||
                parsed > static_cast<unsigned long long>(
                             std::numeric_limits<int>::max())) {
                std::printf("Invalid matrix size: %s\n", argv[i]);
                printUsage(argv[0]);
                options.status = 1;
                return options;
            }
            options.N = static_cast<uint64_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            options.status = 2;
            return options;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            options.status = 1;
            return options;
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    Options options;
    if (rank == 0) {
        options = parseOptions(argc, argv);
    }
    MPI_Bcast(&options, static_cast<int>(sizeof(options)), MPI_BYTE, 0,
              MPI_COMM_WORLD);
    if (options.status != 0) {
        MPI_Finalize();
        return options.status == 2 ? 0 : 1;
    }

    const size_t N = static_cast<size_t>(options.N);
    int dims[2] = {0, 0};
    MPI_Dims_create(processCount, 2, dims);
    const int coordinates[2] = {rank / dims[1], rank % dims[1]};
    const Block rows = splitBlock(N, dims[0], coordinates[0]);
    const Block columns = splitBlock(N, dims[1], coordinates[1]);
    const bool ownsOutput = rows.size != 0 && columns.size != 0;

    if (rank == 0) {
        std::printf("Matrix Multiplication Benchmark\n");
        std::printf("Matrix size: %zu x %zu\n", N, N);
        std::printf("Validation: %s\n",
                    options.validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d process grid)\n", processCount,
                    dims[0], dims[1]);
        std::printf("Initializing matrices...\n");
    }

    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    int allocationSucceeded = 1;
    try {
        if (ownsOutput) {
            A.resize(rows.size * N);
            B.resize(N * columns.size);
            C.resize(rows.size * columns.size);
            initLocalMatrices(A, B, N, rows, columns);
        }
    } catch (...) {
        allocationSucceeded = 0;
    }

    MPI_Allreduce(MPI_IN_PLACE, &allocationSucceeded, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!allocationSucceeded) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Failed to allocate distributed matrix storage.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (ownsOutput) {
        matrixMultiplyLocal(A, B, C, N, rows.size, columns.size);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long long elapsedMilliseconds =
            static_cast<long long>(elapsed * 1000.0);
        std::printf("Computation time: %lld ms\n", elapsedMilliseconds);
        const double gflops =
            elapsed > 0.0
                ? (2.0 * static_cast<double>(N) * static_cast<double>(N) *
                   static_cast<double>(N)) /
                      elapsed / 1e9
                : std::numeric_limits<double>::infinity();
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if (options.printResults) {
        bool gatherSucceeded = false;
        std::vector<double> result =
            gatherResult(C, N, rank, processCount, dims, rows, columns,
                         MPI_COMM_WORLD, gatherSucceeded);
        if (rank == 0) {
            if (gatherSucceeded) {
                print_results(result, "MatrixC");
            } else {
                std::fprintf(stderr,
                             "Failed to allocate the assembled result.\n");
                exitCode = 1;
            }
        }
    }

    if (options.validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        ValidationFailure failure;
        const bool valid = validateResult(C, N, rows, columns, rank,
                                          processCount, MPI_COMM_WORLD, failure);
        if (rank == 0) {
            if (valid) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf(
                    "Validation failed at (%llu, %llu): expected %.10f, got "
                    "%.10f (error: %.10e)\n",
                    static_cast<unsigned long long>(failure.i),
                    static_cast<unsigned long long>(failure.j), failure.expected,
                    failure.actual, failure.relativeError);
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
