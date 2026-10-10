#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate the same values using global matrix coordinates on every rank.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t start;
    size_t count;
};

Range partition(size_t N, int parts, int coordinate) {
    const size_t base = N / parts;
    const size_t extra = N % parts;
    const size_t c = static_cast<size_t>(coordinate);
    return {c * base + std::min(c, extra), base + (c < extra)};
}

constexpr size_t MR = 4;
constexpr size_t NR = 8;
constexpr size_t KB = 128;
constexpr size_t MB = 64;

size_t padded(size_t n, size_t block) {
    return ((n + block - 1) / block) * block;
}

size_t matrixElements(size_t rows, size_t cols) {
    if (cols && rows > std::vector<double>().max_size() / cols) {
        throw std::length_error("Matrix size exceeds addressable storage");
    }
    return rows * cols;
}

// A contains just this process row's input rows. B contains just this process
// column's input columns, packed in NR-wide panels for contiguous SIMD loads.
// The deterministic initialization eliminates input scattering/broadcasting.
void initMatrices(std::vector<double>& A, std::vector<double>& B,
                  size_t N, Range rows, Range cols) {
    for (size_t i = 0; i < rows.count; ++i) {
        for (size_t k = 0; k < N; ++k) {
            A[i * N + k] = getPseudoRndValue(N, rows.start + i, k);
        }
    }
    for (size_t j = 0; j < cols.count; j += NR) {
        for (size_t k = 0; k < N; ++k) {
            for (size_t lane = 0; lane < std::min(NR, cols.count - j); ++lane) {
                B[j * N + k * NR + lane] = getPseudoRndValue(N, k, cols.start + j + lane);
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, size_t N, size_t rows, size_t stride) {
    // Reuse an A row block in cache across column panels and reuse each B
    // panel across rows. SIMD spans independent output elements; each dot
    // product retains the original ascending k order.
    for (size_t kb = 0; kb < N; kb += KB) {
        const size_t kend = std::min(N, kb + KB);
        for (size_t ib = 0; ib < rows; ib += MB) {
            const size_t iend = std::min(rows, ib + MB);
            for (size_t j = 0; j < stride; j += NR) {
                for (size_t i = ib; i < iend; i += MR) {
                    double c0[NR], c1[NR], c2[NR], c3[NR];
                    double* out = C.data() + i * stride + j;
                    #pragma omp simd
                    for (size_t lane = 0; lane < NR; ++lane) {
                        c0[lane] = out[lane];
                        c1[lane] = out[stride + lane];
                        c2[lane] = out[2 * stride + lane];
                        c3[lane] = out[3 * stride + lane];
                    }
                    const double* a = A.data() + i * N;
                    const double* b = B.data() + j * N + kb * NR;
                    // Avoid unrolling k, which spills the SIMD accumulators.
                    #pragma GCC unroll 1
                    for (size_t k = kb; k < kend; ++k, b += NR) {
                        const double a0 = a[k];
                        const double a1 = a[N + k];
                        const double a2 = a[2 * N + k];
                        const double a3 = a[3 * N + k];
                        #pragma omp simd
                        for (size_t lane = 0; lane < NR; ++lane) {
                            c0[lane] += a0 * b[lane];
                            c1[lane] += a1 * b[lane];
                            c2[lane] += a2 * b[lane];
                            c3[lane] += a3 * b[lane];
                        }
                    }
                    #pragma omp simd
                    for (size_t lane = 0; lane < NR; ++lane) {
                        out[lane] = c0[lane];
                        out[stride + lane] = c1[lane];
                        out[2 * stride + lane] = c2[lane];
                        out[3 * stride + lane] = c3[lane];
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& C, size_t N, Range rows,
                    Range cols, size_t stride) {
    // Check the original 25 global positions on their owning ranks.
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = pi % N;
            const size_t j = pj % N;
            if (i < rows.start || i >= rows.start + rows.count ||
                j < cols.start || j >= cols.start + cols.count) {
                continue;
            }
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            const double actual = C[(i - rows.start) * stride + j - cols.start];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            if (!std::isfinite(actual) || relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

// Result collection is only needed for -r. Chunk transfers so MPI's int count
// limit does not constrain matrix size. Packing is already done by the kernel.
void printDistributedResults(const std::vector<double>& C, size_t N,
                             Range rows, size_t stride, int rank, int ranks,
                             const int dims[2]) {
    constexpr size_t chunkSize = 1 << 20;
    constexpr int tag = 0;
    if (rank != 0) {
        const size_t count = rows.count * stride;
        for (size_t offset = 0; offset < count; offset += chunkSize) {
            const int chunk = static_cast<int>(std::min(chunkSize, count - offset));
            MPI_Send(C.data() + offset, chunk, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
        return;
    }

    std::vector<double> result(matrixElements(N, N));
    std::vector<double> received;
    for (int source = 0; source < ranks; ++source) {
        const Range sourceRows = partition(N, dims[0], source / dims[1]);
        const Range sourceCols = partition(N, dims[1], source % dims[1]);
        const size_t sourceStride = padded(sourceCols.count, NR);
        const size_t count = sourceRows.count * sourceStride;
        const double* tile = C.data();
        if (source != 0) {
            received.resize(count);
            for (size_t offset = 0; offset < count; offset += chunkSize) {
                const int chunk = static_cast<int>(std::min(chunkSize, count - offset));
                MPI_Recv(received.data() + offset, chunk, MPI_DOUBLE, source, tag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            tile = received.data();
        }
        if (sourceCols.count == 0) {
            continue;
        }
        for (size_t i = 0; i < sourceRows.count; ++i) {
            std::copy_n(tile + i * sourceStride, sourceCols.count,
                        result.data() + (sourceRows.start + i) * N + sourceCols.start);
        }
    }
    print_results(result, "MatrixC");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) fprintf(stderr, "Matrix size must be a positive integer\n");
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    // Validate N*N even when no process allocates a full matrix, since the
    // pseudo-random generator uses that product as its modulus.
    matrixElements(N, N);
    int dims[2] = {0, 0};
    MPI_Dims_create(ranks, 2, dims);
    const Range rows = partition(N, dims[0], rank / dims[1]);
    const Range cols = partition(N, dims[1], rank % dims[1]);
    const size_t localRows = padded(rows.count, MR);
    const size_t stride = padded(cols.count, NR);
    const bool active = rows.count != 0 && cols.count != 0;

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    std::vector<double> A(active ? matrixElements(localRows, N) : 0);
    std::vector<double> B(active ? matrixElements(N, stride) : 0);
    std::vector<double> C(active ? matrixElements(localRows, stride) : 0);
    if (active) initMatrices(A, B, N, rows, cols);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) matrixMultiply(A, B, C, N, localRows, stride);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double n = static_cast<double>(N);
        printf("Performance: %.3f GFLOPS\n", (2.0 * n * n * n) / elapsed / 1e9);
    }
    if (printResults) {
        printDistributedResults(C, N, rows, stride, rank, ranks, dims);
    }
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = validateResult(C, N, rows, cols, stride) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // A failed allocation must not leave peers waiting in MPI collectives.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
