#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr int sliceWidth = 32;

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Copy a rank's contiguous partition without depending on MPI large-count APIs.
template <typename T>
void transfer(T* data, size_t count, MPI_Datatype type, int source, int dest, int rank) {
    while (count) {
        int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
        if (rank == source)
            MPI_Send(data, chunk, type, dest, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(data, chunk, type, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

struct Matrix {
    std::vector<uint64_t> offsets;
    std::vector<index_t> cols;
    std::vector<double> values;
};

size_t boundary(index_t dim, int partition, int ranks) {
    return uint64_t(dim) * partition / ranks;
}

// Stream the original seeded generator in row order. Root holds at most
// its own partition and the partition currently being distributed.
Matrix distribute(index_t dim, uint64_t entries, double maxVal, int rank, int ranks,
                  std::vector<double>& vec) {
    Matrix local;
    std::vector<size_t> counts(rank == 0 ? ranks : 0);
    if (rank == 0) {
        srand(8675309);
        const uint64_t cells = uint64_t(dim) * dim;
        const double probability = cells ? double(entries) / cells : 0.0;
        uint64_t remaining = entries;
        for (int dest = 0; dest < ranks; ++dest) {
            Matrix part;
            size_t begin = boundary(dim, dest, ranks);
            size_t end = boundary(dim, dest + 1, ranks);
            part.offsets.reserve(end - begin + 1);
            if (dim) part.cols.reserve((entries / dim) * (end - begin));
            part.offsets.push_back(0);
            for (size_t row = begin; row < end; ++row) {
                for (index_t col = 0; remaining && col < dim; ++col) {
                    if (cells - (row * uint64_t(dim) + col) == remaining ||
                        double(rand()) / RAND_MAX <= probability) {
                        part.cols.push_back(col);
                        --remaining;
                    }
                }
                part.offsets.push_back(part.cols.size());
            }
            counts[dest] = part.cols.size();
            if (dest == 0) {
                local = std::move(part);
            } else {
                transfer(part.offsets.data(), part.offsets.size(),
                         MPI_UINT64_T, 0, dest, rank);
                transfer(part.cols.data(), part.cols.size(), MPI_UINT32_T, 0, dest, rank);
            }
        }
        // The original fills the vector and values before reseeding for columns.
        srand(1);
        auto nextValue = [maxVal]() { return maxVal * (rand() / (double(RAND_MAX) + 1.0)); };
        std::generate(vec.begin(), vec.end(), nextValue);
        for (int dest = 0; dest < ranks; ++dest) {
            std::vector<double> values(counts[dest]);
            std::generate(values.begin(), values.end(), nextValue);
            if (dest == 0) local.values = std::move(values);
            else transfer(values.data(), values.size(), MPI_DOUBLE, 0, dest, rank);
        }
    } else {
        local.offsets.resize(boundary(dim, rank + 1, ranks) - boundary(dim, rank, ranks) + 1);
        transfer(local.offsets.data(), local.offsets.size(), MPI_UINT64_T, 0, rank, rank);
        local.cols.resize(local.offsets.back());
        transfer(local.cols.data(), local.cols.size(), MPI_UINT32_T, 0, rank, rank);
        local.values.resize(local.cols.size());
        transfer(local.values.data(), local.values.size(), MPI_DOUBLE, 0, rank, rank);
    }
    for (size_t start = 0; start < dim; start += INT_MAX) {
        int chunk = static_cast<int>(std::min<size_t>(dim - start, INT_MAX));
        MPI_Bcast(vec.data() + start, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    return local;
}

template <typename T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)));
    }
    explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
        if (!host.empty())
            check(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    DeviceArray(const DeviceArray&) = delete;
    ~DeviceArray() { if (data) check(cudaFree(data)); }
};

// Within each slice, neighboring lanes read neighboring entries, while
// each lane accumulates its own row in the original CSR order.
__global__ void multiply(size_t rows, const size_t* __restrict__ slices,
                         const index_t* __restrict__ lengths, const index_t* __restrict__ cols,
                         const double* __restrict__ values, const double* __restrict__ vec,
                         double* __restrict__ result) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t pos = slices[row / sliceWidth] + row % sliceWidth;
    double sum = 0.0;
    index_t left = lengths[row];
    // Issue independent gathers together to cover memory latency.
    constexpr int batch = 8;
    while (left >= batch) {
        double products[batch];
        #pragma unroll
        for (int i = 0; i < batch; ++i) {
            size_t j = pos + i * sliceWidth;
            products[i] = values[j] * vec[cols[j]];
        }
        #pragma unroll
        for (int i = 0; i < batch; ++i) sum += products[i];
        pos += batch * sliceWidth;
        left -= batch;
    }
    for (; left; --left, pos += sliceWidth)
        sum += values[pos] * vec[cols[pos]];
    result[row] = sum;
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

    if (sparsity == 0) {
        if (rank == 0) fprintf(stderr, "Sparsity must be positive\n");
        return 1;
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, devices;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) check(cudaErrorNoDevice);
    check(cudaSetDevice(nodeRank % devices));

    const uint64_t cells = uint64_t(numRows) * numRows;
    const uint64_t entries = cells / sparsity;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", entries,
               100.0 * (1.0 - double(entries) / cells));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> vec(numRows);
    Matrix matrix = distribute(numRows, entries, maxVal, rank, ranks, vec);
    size_t rows = matrix.offsets.size() - 1;
    size_t nslices = (rows + sliceWidth - 1) / sliceWidth;
    int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, nslices)));
    std::vector<index_t> lengths(rows);
    std::vector<size_t> slices(nslices + 1);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t s = 0; s < nslices; ++s) {
        index_t longest = 0;
        for (size_t r = s * sliceWidth; r < std::min(rows, (s + 1) * sliceWidth); ++r) {
            lengths[r] = matrix.offsets[r + 1] - matrix.offsets[r];
            longest = std::max(longest, lengths[r]);
        }
        slices[s + 1] = size_t(longest) * sliceWidth;
    }
    for (size_t s = 0; s < nslices; ++s) slices[s + 1] += slices[s];
    DeviceArray<size_t> dSlices(slices);
    DeviceArray<index_t> dLengths(lengths);
    DeviceArray<double> dVec(vec), dOut(rows);
    std::vector<index_t> cols(slices.back());
    std::vector<double> values(slices.back());
    std::vector<double> reference(validate ? rows : 0);
    if (validate && rank == 0) printf("Computing reference solution...\n");
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t r = 0; r < rows; ++r) {
        size_t pos = slices[r / sliceWidth] + r % sliceWidth;
        for (size_t j = matrix.offsets[r]; j < matrix.offsets[r + 1]; ++j, pos += sliceWidth) {
            cols[pos] = matrix.cols[j];
            values[pos] = matrix.values[j];
        }
        if (validate) {
            double sum = 0.0;
            for (size_t j = matrix.offsets[r]; j < matrix.offsets[r + 1]; ++j)
                sum += matrix.values[j] * vec[matrix.cols[j]];
            reference[r] = sum;
        }
    }
    matrix = {};
    DeviceArray<index_t> dCols(cols);
    DeviceArray<double> dValues(values);

    if (rank == 0) printf("Computing SpMV...\n");
    cudaFuncAttributes attributes;
    check(cudaFuncGetAttributes(&attributes, multiply));
    check(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows && iterations) {
        for (index_t i = 0; i < iterations; ++i)
            multiply<<<(rows + 127) / 128, 128>>>(rows, dSlices.data, dLengths.data,
                dCols.data, dValues.data, dVec.data, dOut.data);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("Performance: %.3f GFLOPS\n", seconds ? 2.0 * entries * iterations / seconds / 1e9 : 0);
    }

    std::vector<double> out((validate || printResults) ? rows : 0);
    if (!out.empty() && iterations)
        check(cudaMemcpy(out.data(), dOut.data, rows * sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        if (rank == 0) {
            out.resize(numRows);
            for (int src = 1; src < ranks && numRows; ++src) {
                size_t begin = boundary(numRows, src, ranks);
                transfer(out.data() + begin, boundary(numRows, src + 1, ranks) - begin,
                         MPI_DOUBLE, src, 0, rank);
            }
            print_results(out, "OutputVector");
        } else {
            transfer(out.data(), rows, MPI_DOUBLE, rank, 0, rank);
        }
    }
    if (validate) {
        int valid = 1;
        #pragma omp parallel for num_threads(threads) reduction(&:valid)
        for (size_t r = 0; r < rows; ++r) {
            double ref = reference[r];
            double error = std::abs(ref) < 1e-10 ? std::abs(out[r]) : std::abs((out[r] - ref) / ref);
            valid &= error <= 0.02;
        }
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validating result...\nValidation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
