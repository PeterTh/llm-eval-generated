#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using wide = uint64_t;

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// The glibc rand() sequence is a lagged additive generator. Advancing
// by a polynomial in its shift operator lets each worker start at the exact
// position the serial generator would have reached, without replaying it.
class RandomStream {
    using State = std::array<uint32_t, 31>;
    State state{};
    unsigned head = 0;

    static State multiply(const State& a, const State& b) {
        uint32_t product[61]{};
        for (int i = 0; i < 31; ++i)
            for (int j = 0; j < 31; ++j)
                product[i + j] += a[i] * b[j];
        // s[i+31] = s[i+28] + s[i], with arithmetic modulo 2^32.
        for (int i = 60; i >= 31; --i) {
            product[i - 3] += product[i];
            product[i - 31] += product[i];
        }
        State result;
        std::copy_n(product, 31, result.begin());
        return result;
    }

    uint32_t step() {
        unsigned lag = head < 3 ? head + 28 : head - 3;
        uint32_t value = (state[head] += state[lag]);
        if (++head == 31) head = 0;
        return value;
    }

public:
    explicit RandomStream(uint32_t seed) {
        // Put the oldest word first (libc starts its write pointer at 3).
        for (int i = 0; i < 31; ++i) {
            state[(i + 28) % 31] = seed;
            seed = (wide(seed) * 16807) % 2147483647;
        }
        for (int i = 0; i < 310; ++i) step();
    }

    uint32_t next() { return step() >> 1; }

    void skip(wide distance) {
        if (!distance) return;
        static const auto powers = [] {
            std::array<State, 64> table{};
            table[0][1] = 1;
            for (int i = 1; i < 64; ++i)
                table[i] = multiply(table[i-1], table[i-1]);
            return table;
        }();
        State coefficients{};
        coefficients[0] = 1;
        for (int i = 0; distance; distance >>= 1, ++i)
            if (distance & 1) coefficients = multiply(coefficients, powers[i]);

        State advanced{};
        for (int j = 0; j < 31; ++j) {
            for (int i = 0; i < 31; ++i)
                advanced[j] += coefficients[i] * state[(head + i) % 31];
            step();
        }
        state = advanced;
        head = 0;
    }
};

struct Matrix {
    std::vector<wide> offsets;
    std::vector<index_t> columns;
    std::vector<double> values;
};

static Matrix initialize(index_t first, index_t rows, index_t dim, wide nnz,
                         double maximum, std::vector<double>& vec) {
    const wide cells = wide(dim) * dim;
    const double probability = cells ? double(nnz) / double(cells) : 0;
    // Match rand() / RAND_MAX <= probability, including boundary rounding.
    uint32_t cutoff = uint32_t(probability * 2147483647.0);
    if (double(cutoff) / 2147483647.0 > probability) --cutoff;
    if (double(cutoff + 1u) / 2147483647.0 <= probability) ++cutoff;
    Matrix a;
    a.offsets.resize(size_t(rows) + 1);
    #pragma omp parallel
    {
        int thread = omp_get_thread_num(), team = omp_get_num_threads();
        size_t begin = wide(rows) * thread / team, end = wide(rows) * (thread + 1) / team;
        RandomStream random(8675309);
        random.skip((wide(first) + begin) * dim);
        for (size_t r = begin; r < end; ++r) {
            wide count = 0;
            if (nnz == cells) count = dim;
            else if (nnz)
                for (index_t j = 0; j < dim; ++j) count += random.next() <= cutoff;
            a.offsets[r + 1] = count;
        }
    }
    for (size_t r = 0; r < rows; ++r) a.offsets[r + 1] += a.offsets[r];
    wide preceding = 0;
    MPI_Exscan(&a.offsets.back(), &preceding, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (!first) preceding = 0;
    // At any row boundary the serial count is bounded by the quota and
    // by the number of cells still available to satisfy it.
    auto assigned = [&](wide observed, wide processed) {
        wide remaining = cells - processed;
        return std::max(std::min(observed, nnz), nnz > remaining ? nnz - remaining : 0);
    };
    const wide base = assigned(preceding, wide(first) * dim);
    for (size_t r = 0; r <= rows; ++r)
        a.offsets[r] = assigned(preceding + a.offsets[r], (first + r) * dim) - base;
    a.columns.resize(a.offsets.back());
    a.values.resize(a.offsets.back());

    #pragma omp parallel
    {
        int thread = omp_get_thread_num(), team = omp_get_num_threads();
        size_t begin = wide(rows) * thread / team, end = wide(rows) * (thread + 1) / team;
        RandomStream position(8675309);
        position.skip((wide(first) + begin) * dim);
        RandomStream value(1);
        value.skip(wide(dim) + base + a.offsets[begin]);
        for (size_t r = begin; r < end; ++r) {
            wide write = a.offsets[r];
            wide left = nnz - base - write;
            wide remaining = cells - (first + r) * dim;
            if (!left) break;
            for (index_t j = 0; j < dim; ++j) {
                if (left == remaining || position.next() <= cutoff) {
                    a.columns[write] = j;
                    a.values[write++] = maximum * (value.next() / 2147483648.0);
                    if (!--left) break;
                }
                --remaining;
            }
        }
        size_t vbegin = wide(dim) * thread / team, vend = wide(dim) * (thread + 1) / team;
        RandomStream vectorRandom(1);
        vectorRandom.skip(vbegin);
        for (size_t i = vbegin; i < vend; ++i)
            vec[i] = maximum * (vectorRandom.next() / 2147483648.0);
    }
    return a;
}

template<class T>
static T* upload(const std::vector<T>& host) {
    if (host.empty()) return nullptr;
    T* device;
    check(cudaMalloc(&device, host.size() * sizeof(T)));
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
}

// Neighboring lanes load a group of consecutive CSR entries together. A
// shuffle broadcasts each product in column order to preserve the serial sum.
template<int lanes>
__global__ void spmvKernel(const wide* __restrict__ offsets, const index_t* __restrict__ columns,
                           const double* __restrict__ values, const double* __restrict__ vec, index_t rows,
                           double* out) {
    wide tid = wide(blockIdx.x) * blockDim.x + threadIdx.x;
    index_t r = tid / lanes;
    if (r >= rows) return;
    unsigned lane = threadIdx.x % lanes;
    unsigned mask = (0xffffffffu >> (32 - lanes)) << ((threadIdx.x % 32) - lane);
    wide j = offsets[r], end = offsets[r + 1];
    double sum = 0;
    for (; end - j >= lanes; j += lanes) {
        double product = __dmul_rn(values[j + lane], vec[columns[j + lane]]);
        #pragma unroll
        for (int i = 0; i < lanes; ++i)
            sum = __dadd_rn(sum, __shfl_sync(mask, product, i, lanes));
    }
    if (j < end) {
        double product = j + lane < end ? __dmul_rn(values[j + lane], vec[columns[j + lane]]) : 0;
        #pragma unroll
        for (int i = 0; i < lanes; ++i)
            sum = __dadd_rn(sum, __shfl_sync(mask, product, i, lanes));
    }
    if (!lane) out[r] = sum;
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

static int run(int argc, char** argv, int rank, int ranks) {
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
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }

    if (!sparsity || numRows > std::numeric_limits<int>::max()) {
        if (!rank) fprintf(stderr, "Invalid matrix dimension or sparsity\n");
        return 1;
    }
    const wide cells = wide(numRows) * numRows;
    const wide nnz = cells / sparsity;
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %llu (%.2f%% sparse)\n",
               static_cast<unsigned long long>(nnz), cells ? 100.0 * (1.0 - double(nnz) / cells) : 100.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(localRank % devices));

    auto boundary = [&](int i) { return index_t(wide(numRows) * i / ranks); };
    const index_t first = boundary(rank), rows = boundary(rank + 1) - first;
    // Bound team startup costs on small local partitions.
    omp_set_num_threads(std::min<wide>(omp_get_max_threads(), 1 + wide(rows) * numRows / 32768));
    std::vector<double> vec(numRows), result(validate || printResults ? rows : 0);
    std::vector<double> reference(validate ? rows : 0);
    auto matrix = initialize(first, rows, numRows, nnz, maxVal, vec);
    if (validate) {
        if (!rank) printf("Computing reference solution...\n");
        #pragma omp parallel for schedule(static)
        for (index_t r = 0; r < rows; ++r) {
            double sum = 0;
            for (wide j = matrix.offsets[r]; j < matrix.offsets[r + 1]; ++j)
                sum += matrix.values[j] * vec[matrix.columns[j]];
            reference[r] = sum;
        }
    }

    auto dOffsets = upload(matrix.offsets);
    auto dValues = upload(matrix.values);
    auto dColumns = upload(matrix.columns);
    matrix = {};
    auto dVec = upload(vec);
    double* dOut = nullptr;
    if (rows) check(cudaMalloc(&dOut, rows * sizeof(double)));
    if (!rank) printf("Computing SpMV...\n");
    const unsigned lanes = nnz <= wide(numRows) * 2 ? 1 : 8;
    auto kernel = lanes == 1 ? spmvKernel<1> : spmvKernel<8>;
    constexpr unsigned threads = 128;
    const unsigned blocks = (wide(rows) * lanes + threads - 1) / threads;
    cudaFuncAttributes attributes;
    check(cudaFuncGetAttributes(&attributes, kernel));
    check(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows && iterations) {
        for (index_t iter = 0; iter < iterations; ++iter)
            kernel<<<blocks, threads>>>(dOffsets, dColumns, dValues, dVec, rows, dOut);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("Performance: %.3f GFLOPS\n", seconds ? 2e-9 * nnz * iterations / seconds : 0);
    }

    if (rows && iterations && (validate || printResults))
        check(cudaMemcpy(result.data(), dOut, rows * sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaFree(dOut));
    check(cudaFree(dVec));
    check(cudaFree(dColumns));
    check(cudaFree(dValues));
    check(cudaFree(dOffsets));
    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> full;
        if (!rank) {
            full.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int i = 0; i < ranks; ++i) {
                counts[i] = boundary(i + 1) - boundary(i);
                displacements[i] = boundary(i);
            }
        }
        MPI_Gatherv(result.data(), rows, MPI_DOUBLE, full.data(), counts.data(),
                    displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(full, "OutputVector");
    }
    if (validate) {
        index_t bad = numRows;
        #pragma omp parallel for reduction(min:bad)
        for (index_t r = 0; r < rows; ++r) {
            double ref = reference[r], res = result[r];
            double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
            if (!(error <= 0.02)) bad = std::min(bad, first + r);
        }
        index_t globalBad;
        MPI_Allreduce(&bad, &globalBad, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validating result...\n");
        if (globalBad >= first && globalBad - first < rows)
            printf("Validation failed at index %u: reference %.10e, got %.10e\n", globalBad,
                   reference[globalBad-first], result[globalBad-first]);
        if (!rank) printf("Validation: %s\n", globalBad == numRows ? "PASSED" : "FAILED");
        return globalBad != numRows;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
