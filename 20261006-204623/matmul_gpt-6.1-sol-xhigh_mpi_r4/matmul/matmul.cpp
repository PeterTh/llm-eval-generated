#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate the same values as the serial benchmark, using global indices.
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

struct Range {
    size_t begin;
    size_t size;
};

Range partition(size_t N, int parts, int coordinate) {
    const size_t base = N / parts;
    const size_t extra = N % parts;
    return {base * coordinate + std::min(extra, static_cast<size_t>(coordinate)),
            base + (static_cast<size_t>(coordinate) < extra)};
}

size_t storageSize(size_t count, size_t stride) {
    if (stride != 0 && count > std::vector<double>().max_size() / stride) {
        throw std::length_error("Matrix storage is too large");
    }
    // A valid buffer is also supplied to MPI for zero-count broadcasts.
    return std::max(size_t{1}, count * stride);
}

struct Panels {
    size_t width;
    size_t count;
    size_t aStride;
    size_t bStride;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> aReceive[2];
    std::vector<double> bReceive[2];

    Panels(size_t N, Range rows, Range cols, const int dims[2], const int coords[2]) {
        const size_t paddedRows = (rows.size + 3) / 4 * 4;
        const size_t paddedCols = (cols.size + 7) / 8 * 8;
        // Keep enough panels to distribute initialization across the grid,
        // and bound every MPI count to the portable int-count interface.
        width = std::min(size_t{256}, (N - 1) / std::max(dims[0], dims[1]) + 1);
        const size_t maxRows = partition(N, dims[0], 0).size;
        const size_t maxCols = partition(N, dims[1], 0).size;
        const size_t maxExtent = std::max((maxRows + 3) / 4 * 4, (maxCols + 7) / 8 * 8);
        width = std::min(width, static_cast<size_t>(INT_MAX) /
                                std::max(size_t{1}, maxExtent));
        if (width == 0) {
            throw std::length_error("Matrix panel exceeds MPI count limits");
        }
        count = (N - 1) / width + 1;
        aStride = paddedRows * width;
        bStride = paddedCols * width;
        const size_t aCount = count / dims[1] + (static_cast<size_t>(coords[1]) < count % dims[1]);
        const size_t bCount = count / dims[0] + (static_cast<size_t>(coords[0]) < count % dims[0]);
        A.resize(storageSize(aCount, aStride));
        B.resize(storageSize(bCount, bStride));
        for (int slot = 0; slot < 2; ++slot) {
            aReceive[slot].resize(std::max(size_t{1}, aStride));
            bReceive[slot].resize(std::max(size_t{1}, bStride));
        }

        // A panels are cyclic over process columns, B panels over process
        // rows. Pack microtiles once so the inner kernel reads contiguous data.
        for (size_t panel = coords[1]; panel < count; panel += dims[1]) {
            double* packed = A.data() + (panel / dims[1]) * aStride;
            const size_t depth = std::min(width, N - panel * width);
            for (size_t i = 0; i < rows.size; i += 4) {
                for (size_t k = 0; k < depth; ++k) {
                    for (size_t r = 0; r < std::min(size_t{4}, rows.size - i); ++r) {
                        packed[i * width + k * 4 + r] =
                            getPseudoRndValue(N, rows.begin + i + r, panel * width + k);
                    }
                }
            }
        }
        for (size_t panel = coords[0]; panel < count; panel += dims[0]) {
            double* packed = B.data() + (panel / dims[0]) * bStride;
            const size_t depth = std::min(width, N - panel * width);
            for (size_t j = 0; j < cols.size; j += 8) {
                for (size_t k = 0; k < depth; ++k) {
                    for (size_t c = 0; c < std::min(size_t{8}, cols.size - j); ++c) {
                        packed[j * width + k * 8 + c] =
                            getPseudoRndValue(N, panel * width + k, cols.begin + j + c);
                    }
                }
            }
        }
    }
};

// SIMD runs across independent columns; each element accumulates k in its
// original order. No floating-point reassociation or partial-sum reduction.
template<int Rows>
inline void multiplyTile(const double* A, const double* B, double* C,
                         size_t stride, size_t depth) {
    double sums[Rows][8];
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < 8; ++c) {
            sums[r][c] = C[r * stride + c];
        }
    }
    for (size_t k = 0; k < depth; ++k) {
        for (int r = 0; r < Rows; ++r) {
            const double a = A[k * 4 + r];
            for (int c = 0; c < 8; ++c) {
                sums[r][c] += a * B[k * 8 + c];
            }
        }
    }
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < 8; ++c) {
            C[r * stride + c] = sums[r][c];
        }
    }
}

#if defined(__AVX2__)
inline __m256d multiplyAdd(__m256d a, __m256d b, __m256d c) {
#if defined(__FMA__)
    return _mm256_fmadd_pd(a, b, c);
#else
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
#endif
}

// Eight vector accumulators hide instruction latency without spilling.
// Keep the portable template above for other CPUs and partial row tiles.
template<>
inline void multiplyTile<4>(const double* A, const double* B, double* C,
                            size_t stride, size_t depth) {
    __m256d c00 = _mm256_loadu_pd(C);
    __m256d c01 = _mm256_loadu_pd(C + 4);
    __m256d c10 = _mm256_loadu_pd(C + stride);
    __m256d c11 = _mm256_loadu_pd(C + stride + 4);
    __m256d c20 = _mm256_loadu_pd(C + 2 * stride);
    __m256d c21 = _mm256_loadu_pd(C + 2 * stride + 4);
    __m256d c30 = _mm256_loadu_pd(C + 3 * stride);
    __m256d c31 = _mm256_loadu_pd(C + 3 * stride + 4);
    for (size_t k = 0; k < depth; ++k) {
        __m256d b0 = _mm256_loadu_pd(B + k * 8);
        __m256d b1 = _mm256_loadu_pd(B + k * 8 + 4);
#if defined(__GNUC__)
        // Keep B in registers rather than reload it for each output row.
        __asm__("" : "+x"(b0), "+x"(b1));
#endif
        const __m256d a0 = _mm256_broadcast_sd(A + k * 4);
        const __m256d a1 = _mm256_broadcast_sd(A + k * 4 + 1);
        const __m256d a2 = _mm256_broadcast_sd(A + k * 4 + 2);
        const __m256d a3 = _mm256_broadcast_sd(A + k * 4 + 3);
        c00 = multiplyAdd(a0, b0, c00);
        c01 = multiplyAdd(a0, b1, c01);
        c10 = multiplyAdd(a1, b0, c10);
        c11 = multiplyAdd(a1, b1, c11);
        c20 = multiplyAdd(a2, b0, c20);
        c21 = multiplyAdd(a2, b1, c21);
        c30 = multiplyAdd(a3, b0, c30);
        c31 = multiplyAdd(a3, b1, c31);
    }
    _mm256_storeu_pd(C, c00);
    _mm256_storeu_pd(C + 4, c01);
    _mm256_storeu_pd(C + stride, c10);
    _mm256_storeu_pd(C + stride + 4, c11);
    _mm256_storeu_pd(C + 2 * stride, c20);
    _mm256_storeu_pd(C + 2 * stride + 4, c21);
    _mm256_storeu_pd(C + 3 * stride, c30);
    _mm256_storeu_pd(C + 3 * stride + 4, c31);
}
#endif

void multiplyPanel(const double* A, const double* B, std::vector<double>& C,
                   size_t rows, size_t cols, size_t width, size_t depth,
                   MPI_Request nextRequests[2]) {
    const size_t stride = (cols + 7) / 8 * 8;
    for (size_t jb = 0; jb < cols; jb += 128) {
        for (size_t ib = 0; ib < rows; ib += 64) {
            const size_t iEnd = std::min(rows, ib + 64);
            for (size_t j = jb; j < std::min(cols, jb + 128); j += 8) {
                const double* b = B + j * width;
                size_t i = ib;
                for (; i + 4 <= iEnd; i += 4) {
                    multiplyTile<4>(A + i * width, b, C.data() + i * stride + j, stride, depth);
                }
                if (i != iEnd) {
                    const double* a = A + i * width;
                    double* c = C.data() + i * stride + j;
                    switch (iEnd - i) {
                        case 1: multiplyTile<1>(a, b, c, stride, depth); break;
                        case 2: multiplyTile<2>(a, b, c, stride, depth); break;
                        case 3: multiplyTile<3>(a, b, c, stride, depth); break;
                    }
                }
            }
            // Drive progress on MPI implementations without a progress thread.
            int complete;
            MPI_Testall(2, nextRequests, &complete, MPI_STATUSES_IGNORE);
        }
    }
}

void matrixMultiply(Panels& panels, std::vector<double>& C, size_t N,
                    Range rows, Range cols, const int dims[2], const int coords[2],
                    MPI_Comm rowComm, MPI_Comm colComm) {
    MPI_Request requests[2][2] = {{MPI_REQUEST_NULL, MPI_REQUEST_NULL},
                                  {MPI_REQUEST_NULL, MPI_REQUEST_NULL}};
    double* a[2];
    double* b[2];
    auto startPanel = [&](size_t panel, int slot) {
        const int aRoot = static_cast<int>(panel % dims[1]);
        const int bRoot = static_cast<int>(panel % dims[0]);
        a[slot] = coords[1] == aRoot ? panels.A.data() + (panel / dims[1]) * panels.aStride
                                    : panels.aReceive[slot].data();
        b[slot] = coords[0] == bRoot ? panels.B.data() + (panel / dims[0]) * panels.bStride
                                    : panels.bReceive[slot].data();
        MPI_Ibcast(a[slot], static_cast<int>(panels.aStride), MPI_DOUBLE,
                   aRoot, rowComm, &requests[slot][0]);
        MPI_Ibcast(b[slot], static_cast<int>(panels.bStride), MPI_DOUBLE,
                   bRoot, colComm, &requests[slot][1]);
    };

    startPanel(0, 0);
    for (size_t panel = 0; panel < panels.count; ++panel) {
        const int slot = static_cast<int>(panel % 2);
        MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);
        if (panel + 1 < panels.count) {
            startPanel(panel + 1, 1 - slot);
        }
        multiplyPanel(a[slot], b[slot], C, rows.size, cols.size, panels.width,
                      std::min(panels.width, N - panel * panels.width), requests[1 - slot]);
    }
}

// Gather only for -r, receiving directly into the original global row-major
// layout. Derived datatypes avoid an extra packing buffer and int-sized N*N.
std::vector<double> gatherResult(const std::vector<double>& C, size_t N,
                                 Range rows, Range cols, MPI_Comm grid, int rank, int size,
                                 const int dims[2]) {
    const int stride = static_cast<int>((cols.size + 7) / 8 * 8);
    std::vector<double> result;
    if (rank == 0) {
        result.resize(N * N);
        for (size_t i = 0; i < rows.size; ++i) {
            std::copy_n(C.data() + i * stride, cols.size,
                        result.data() + (rows.begin + i) * N + cols.begin);
        }
        for (int source = 1; source < size; ++source) {
            int coords[2];
            MPI_Cart_coords(grid, source, 2, coords);
            const Range r = partition(N, dims[0], coords[0]);
            const Range c = partition(N, dims[1], coords[1]);
            if (r.size == 0 || c.size == 0) {
                continue;
            }
            MPI_Datatype block;
            MPI_Type_vector(static_cast<int>(r.size), static_cast<int>(c.size),
                            static_cast<int>(N), MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data() + r.begin * N + c.begin, 1, block, source, 0,
                     grid, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    } else if (rows.size != 0 && cols.size != 0) {
        MPI_Datatype block;
        MPI_Type_vector(static_cast<int>(rows.size), static_cast<int>(cols.size),
                        stride, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(C.data(), 1, block, 0, 0, grid);
        MPI_Type_free(&block);
    }
    return result;
}

bool validateResult(const std::vector<double>& C, size_t N, Range rows, Range cols) {
    const size_t stride = (cols.size + 7) / 8 * 8;
    auto check = [&](size_t i, size_t j) {
        if (i < rows.begin || i >= rows.begin + rows.size ||
            j < cols.begin || j >= cols.begin + cols.size) {
            return true;
        }
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) {
            expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
        }
        const double actual = C[(i - rows.begin) * stride + j - cols.begin];
        const double relError = std::abs((actual - expected) / (expected + 1e-10));
        if (!std::isfinite(actual) || relError > 1e-6) {
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                   i, j, expected, actual, relError);
            return false;
        }
        return true;
    };
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            if (!check(i % N, j % N)) {
                return false;
            }
        }
    }
    // Also sample every nonempty rank, including the bottom/right edges.
    if (rows.size != 0 && cols.size != 0) {
        const size_t ri[] = {rows.begin, rows.begin + rows.size / 2, rows.begin + rows.size - 1};
        const size_t cj[] = {cols.begin, cols.begin + cols.size / 2, cols.begin + cols.size - 1};
        for (size_t i : ri) {
            for (size_t j : cj) {
                if (!check(i, j)) {
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int size) {
    std::uint64_t options[3] = {512, 0, 0};
    int argumentStatus = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* value = argv[++i];
                char* end;
                errno = 0;
                const unsigned long long parsed = std::strtoull(value, &end, 10);
                const size_t maxElements = std::vector<double>().max_size();
                if (value[0] < '0' || value[0] > '9' || *end != '\0' || errno != 0 ||
                    parsed == 0 || parsed > maxElements / parsed) {
                    printf("Invalid matrix size: %s\n", value);
                    argumentStatus = 1;
                    break;
                }
                options[0] = parsed;
            } else if (strcmp(argv[i], "-v") == 0) {
                options[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                argumentStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                argumentStatus = 1;
                break;
            }
        }
    }
    MPI_Bcast(&argumentStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (argumentStatus != 0) {
        return argumentStatus == 2 ? 0 : 1;
    }
    MPI_Bcast(options, 3, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const size_t N = static_cast<size_t>(options[0]);
    const bool validate = options[1] != 0;
    const bool printResults = options[2] != 0;

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm grid, rowComm, colComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    int rowRemain[2] = {0, 1};
    int colRemain[2] = {1, 0};
    MPI_Cart_sub(grid, rowRemain, &rowComm);
    MPI_Cart_sub(grid, colRemain, &colComm);
    const Range rows = partition(N, dims[0], coords[0]);
    const Range cols = partition(N, dims[1], coords[1]);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }
    Panels panels(N, rows, cols, dims, coords);
    std::vector<double> C(storageSize(rows.size, (cols.size + 7) / 8 * 8));
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    matrixMultiply(panels, C, N, rows, cols, dims, coords, rowComm, colComm);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
        const double gflops = 2.0 * N * N * N / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    if (printResults) {
        const std::vector<double> result = gatherResult(C, N, rows, cols, grid, rank, size, dims);
        if (rank == 0) {
            print_results(result, "MatrixC");
        }
    }
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const int localValid = validateResult(C, N, rows, cols) ? 1 : 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, grid);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&grid);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        status = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
