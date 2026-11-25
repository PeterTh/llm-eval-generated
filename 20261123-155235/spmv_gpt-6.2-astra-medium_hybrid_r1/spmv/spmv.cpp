#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <mutex>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;

static void fail(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

template<class T>
class DeviceArray {
    T* ptr = nullptr;
public:
    explicit DeviceArray(size_t count) {
        if (count) check(cudaMalloc(&ptr, count * sizeof(T)));
    }
    explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
        if (!host.empty())
            check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { if (ptr) cudaFree(ptr); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data() const { return ptr; }
};

// Each worker starts at the same point it would have reached in the serial
// rand() stream. On glibc the additive generator can jump there in O(log offset).
class RandomStream {
#ifdef __GLIBC__
    using Window = std::array<uint32_t, 31>;
    Window state{};
    unsigned head = 3;

    // The recurrence is a[t+31] = a[t+28] + a[t] modulo 2^32.
    static Window multiply(const Window& a, const Window& b) {
        uint32_t product[61]{};
        for (int i = 0; i < 31; ++i)
            for (int j = 0; j < 31; ++j)
                product[i+j] += a[i] * b[j];
        for (int i = 60; i >= 31; --i) {
            product[i-3] += product[i];
            product[i-31] += product[i];
        }
        Window result;
        std::copy_n(product, 31, result.begin());
        return result;
    }

public:
    RandomStream(unsigned seed, uint64_t offset) {
        state[0] = seed;
        for (int i = 1; i < 31; ++i)
            state[i] = (uint64_t(state[i-1]) * 16807) % 2147483647;
        for (int i = 0; i < 310; ++i) next();
        if (!offset) return;

        Window shift{}, power{};
        shift[0] = power[1] = 1;
        while (offset) {
            if (offset & 1) shift = multiply(shift, power);
            offset >>= 1;
            if (offset) power = multiply(power, power);
        }
        Window jumped{};
        for (auto& value : jumped) {
            for (unsigned j = 0; j < 31; ++j)
                value += shift[j] * state[(head+j) % 31];
            next();
        }
        head = 0;
        state = jumped;
    }
    uint32_t next() {
        uint32_t value = (state[head] += state[head >= 3 ? head-3 : head+28]);
        if (++head == 31) head = 0;
        return value >> 1;
    }
#else
    // Other C libraries may use a different sequence.
    static std::mutex& mutex() { static std::mutex m; return m; }
    std::unique_lock<std::mutex> lock{mutex()};
public:
    RandomStream(unsigned seed, uint64_t offset) {
        std::srand(seed);
        while (offset--) std::rand();
    }
    uint32_t next() { return std::rand(); }
#endif
};

static index_t partition(index_t size, int part, int parts) {
    return uint64_t(size) * part / parts;
}

// Avoid creating a large host team for small partitions.
static int teamSize(uint64_t work) {
    return int(std::min<uint64_t>(omp_get_max_threads(), 1 + work / 65536));
}

static void fill(std::vector<double>& values, uint64_t offset, double maxVal) {
    #pragma omp parallel num_threads(teamSize(values.size()))
    {
        index_t first = partition(values.size(), omp_get_thread_num(), omp_get_num_threads());
        index_t last = partition(values.size(), omp_get_thread_num()+1, omp_get_num_threads());
        RandomStream stream(1, offset + first);
        for (index_t i = first; i < last; ++i)
            values[i] = maxVal * (stream.next() / (double(RAND_MAX) + 1.0));
    }
}

constexpr index_t sliceWidth = 32;

struct LocalMatrix {
    std::vector<uint64_t> slices;
    std::vector<index_t> lengths, columns;
    std::vector<double> values, reference;

    LocalMatrix(index_t first, index_t rows, index_t sparsity, double maxVal,
                const std::vector<double>& vec, bool validate, int rank);
};

LocalMatrix::LocalMatrix(index_t first, index_t rows, index_t sparsity, double maxVal,
                         const std::vector<double>& vec, bool validate, int rank) {
    const index_t numRows = vec.size();
    const uint64_t cells = uint64_t(numRows) * numRows;
    const uint64_t nItems = cells / sparsity;
    // Count independent Bernoulli hits per row, then recover the serial
    // generator's global quota at each row boundary with one prefix scan.
    const double probability = cells ? double(nItems) / cells : 0.0;
    int threshold = int(probability * RAND_MAX);
    if (double(threshold) / RAND_MAX > probability) --threshold;
    if (threshold < RAND_MAX && double(threshold + 1) / RAND_MAX <= probability) ++threshold;
    const int groups = teamSize(uint64_t(rows) * numRows);
    std::vector<std::vector<index_t>> candidates(groups);
    std::vector<index_t> hitCounts(rows);
    uint64_t hits = 0;
    #pragma omp parallel for num_threads(groups) schedule(static) reduction(+:hits)
    for (int g = 0; g < groups; ++g) {
        index_t begin = partition(rows, g, groups);
        index_t end = partition(rows, g+1, groups);
        auto& cols = candidates[g];
        if (sparsity != 1)
            cols.reserve(uint64_t(end - begin) * numRows / sparsity);
        RandomStream stream(8675309, uint64_t(first + begin) * numRows);
        for (index_t r = begin; r < end; ++r) {
            size_t start = cols.size();
            if (sparsity != 1 && nItems) {
                for (index_t j = 0; j < numRows; ++j)
                    if (stream.next() <= uint32_t(threshold)) cols.push_back(j);
            }
            hitCounts[r] = sparsity == 1 ? numRows : cols.size() - start;
            hits += hitCounts[r];
        }
    }
    uint64_t prefix = 0;
    MPI_Exscan(&hits, &prefix, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) prefix = 0;
    auto assigned = [&](index_t row) -> uint64_t {
        uint64_t remaining = uint64_t(numRows - first - row) * numRows;
        uint64_t forced = remaining < nItems ? nItems - remaining : 0;
        return std::min<uint64_t>(nItems, std::max(forced, prefix));
    };
    const uint64_t base = assigned(0);
    lengths.resize(rows);
    std::vector<uint64_t> offsets(size_t(rows) + 1);
    for (index_t r = 0; r < rows; ++r) {
        prefix += hitCounts[r];
        offsets[r+1] = assigned(r+1) - base;
        lengths[r] = offsets[r+1] - offsets[r];
    }

    const index_t numSlices = (rows + sliceWidth - 1) / sliceWidth;
    slices.resize(size_t(numSlices) + 1);
    for (index_t s = 0; s < numSlices; ++s) {
        auto begin = lengths.begin() + s * sliceWidth;
        auto end = lengths.begin() + std::min(rows, (s+1) * sliceWidth);
        slices[s+1] = slices[s] + uint64_t(sliceWidth) * *std::max_element(begin, end);
    }
    values.resize(slices.back());
    columns.resize(slices.back());
    reference.resize(validate ? rows : 0);
    if (!rank && validate) printf("Computing reference solution...\n");
    #pragma omp parallel for num_threads(groups) schedule(static)
    for (int g = 0; g < groups; ++g) {
        index_t begin = partition(rows, g, groups);
        index_t end = partition(rows, g+1, groups);
        // Values follow the vector in the original, un-reseeded stream.
        RandomStream stream(1, uint64_t(numRows) + base + offsets[begin]);
        size_t start = 0;
        for (index_t r = begin; r < end; ++r) {
            uint64_t pos = slices[r / sliceWidth] + r % sliceWidth;
            double sum = 0.0;
            for (index_t k = 0; k < lengths[r]; ++k) {
                index_t col = k;
                if (sparsity != 1) {
                    // Force a dense suffix when every remaining cell is needed.
                    index_t latest = numRows - lengths[r] + k;
                    col = k < hitCounts[r] ? std::min(candidates[g][start+k], latest) : latest;
                }
                double v = maxVal * (stream.next() / (double(RAND_MAX) + 1.0));
                columns[pos] = col;
                values[pos] = v;
                if (validate) sum += v * vec[col];
                pos += sliceWidth;
            }
            if (validate) reference[r] = sum;
            start += hitCounts[r];
        }
        std::vector<index_t>().swap(candidates[g]);
    }
}

// Store each group of 32 rows column-wise so adjacent CUDA lanes read
// consecutive values and indices while accumulating in original CSR order.
__global__ void spmvKernel(index_t rows, const uint64_t* __restrict__ slices,
                           const index_t* __restrict__ lengths,
                           const index_t* __restrict__ columns,
                           const double* __restrict__ values,
                           const double* __restrict__ vector, double* output) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint64_t pos = slices[row / sliceWidth] + row % sliceWidth;
    double sum = 0.0;
    index_t remaining = lengths[row];
    // Issue several independent gathers per lane to hide vector-load latency.
    constexpr index_t batch = 8;
    while (remaining >= batch) {
        double products[batch];
        #pragma unroll
        for (index_t k = 0; k < batch; ++k) {
            uint64_t slot = pos + k * sliceWidth;
            products[k] = values[slot] * vector[columns[slot]];
        }
        #pragma unroll
        for (index_t k = 0; k < batch; ++k) sum += products[k];
        pos += batch * sliceWidth;
        remaining -= batch;
    }
    for (index_t k = 0; k < remaining; ++k, pos += sliceWidth)
        sum += values[pos] * vector[columns[pos]];
    output[row] = sum;
}

bool verifyResults(const double* reference, const double* result, const index_t size, index_t first) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", first + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       first + i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    const uint64_t cells = uint64_t(numRows) * numRows;
    if (!sparsity || numRows > uint32_t(INT_MAX)) {
        if (!rank) fprintf(stderr, "Invalid sparsity or matrix exceeds supported index range\n");
        return 1;
    }
    const uint64_t nItems = cells / sparsity;
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - double(nItems) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations,
               maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) fail("No visible CUDA device");
    check(cudaSetDevice(localRank % devices));

    const index_t first = partition(numRows, rank, ranks);
    const index_t rows = partition(numRows, rank + 1, ranks) - first;
    std::vector<double> vec(numRows);
    fill(vec, 0, maxVal);

    LocalMatrix matrix(first, rows, sparsity, maxVal, vec, validate, rank);
    DeviceArray<uint64_t> dSlices(matrix.slices);
    DeviceArray<index_t> dLengths(matrix.lengths), dCols(matrix.columns);
    DeviceArray<double> dValues(matrix.values), dVec(vec), dOut(rows);
    std::vector<double>().swap(matrix.values);
    std::vector<index_t>().swap(matrix.columns);
    const unsigned blockSize = rows <= 65536 ? 32 : 128;
    cudaFuncAttributes attributes;
    check(cudaFuncGetAttributes(&attributes, spmvKernel));
    check(cudaDeviceSynchronize());
    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows) {
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<<<(rows+blockSize-1)/blockSize, blockSize>>>(rows, dSlices.data(), dLengths.data(),
                dCols.data(), dValues.data(), dVec.data(), dOut.data());
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("Performance: %.3f GFLOPS\n", seconds ? 2.0 * nItems * iterations / seconds / 1e9 : 0);
    }

    std::vector<double> out(validate || printResults ? rows : 0);
    if (!out.empty() && iterations)
        check(cudaMemcpy(out.data(), dOut.data(), rows * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        std::vector<int> counts, starts;
        std::vector<double> global;
        if (!rank) {
            counts.resize(ranks);
            starts.resize(ranks);
            global.resize(numRows);
            for (int i = 0; i < ranks; ++i) {
                starts[i] = partition(numRows, i, ranks);
                counts[i] = partition(numRows, i+1, ranks) - starts[i];
            }
        }
        MPI_Gatherv(out.data(), rows, MPI_DOUBLE, global.data(), counts.data(),
                    starts.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(global, "OutputVector");
    }
    if (validate) {
        if (!rank) printf("Validating result...\n");
        int valid = verifyResults(matrix.reference.data(), out.data(), rows, first);
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!rank) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED required");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
