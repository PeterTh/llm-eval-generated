#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned sliceWidth = 32;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#ifdef __GLIBC__
// glibc rand() uses a degree-31 additive generator. Jump to each worker's
// slice of the original stream using x^31 = x^28 + 1 modulo 2^32.
class RandomStream {
    using Polynomial = std::array<uint32_t, 31>;
    Polynomial state{};
    unsigned head = 0;

    static Polynomial square(const Polynomial& p) {
        std::array<uint32_t, 61> product{};
        for (int i = 0; i < 31; ++i)
            for (int j = 0; j < 31; ++j)
                product[i+j] += p[i] * p[j];
        for (int i = 60; i >= 31; --i) {
            product[i-3] += product[i];
            product[i-31] += product[i];
        }
        Polynomial result;
        std::copy_n(product.begin(), 31, result.begin());
        return result;
    }

public:
    RandomStream(unsigned seed, uint64_t offset) {
        state[0] = seed;
        for (unsigned i = 1; i < 31; ++i)
            state[i] = uint64_t(state[i-1]) * 16807 % 2147483647;
        // The first additive update uses the seed words at positions 3 and 0.
        std::rotate(state.begin(), state.begin()+3, state.end());
        for (int i = 0; i < 310; ++i) next();
        if (!offset) return;

        Polynomial p{};
        p[0] = 1;
        uint64_t bit = 1;
        while (bit <= offset / 2) bit <<= 1;
        for (; bit; bit >>= 1) {
            p = square(p);
            if (offset & bit) {
                uint32_t carry = p.back();
                std::move_backward(p.begin(), p.end()-1, p.end());
                p[0] = carry;
                p[28] += carry;
            }
        }
        Polynomial jumped;
        for (auto& word : jumped) {
            word = 0;
            for (unsigned j = 0; j < 31; ++j)
                word += p[j] * state[(head + j) % 31];
            next();
        }
        state = jumped;
        head = 0;
    }

    uint32_t next() {
        unsigned other = head < 3 ? head + 28 : head - 3;
        uint32_t value = state[head] += state[other];
        if (++head == 31) head = 0;
        return value >> 1;
    }
};
#else
// Other C libraries may implement rand() differently. Replay their native
// stream on one host thread per rank to retain the platform's input data.
class RandomStream {
public:
    RandomStream(unsigned seed, uint64_t offset) {
        srand(seed);
        while (offset--) rand();
    }
    unsigned next() { return rand(); }
};
#endif

template<class Function>
void randomChunks(index_t count, index_t drawsPerItem, Function work) {
#ifdef __GLIBC__
    // Amortize thread startup and the jump over a useful amount of work.
    uint64_t chunks = 1 + uint64_t(count) * drawsPerItem / 16384;
    int threads = std::max<uint64_t>(1, std::min<uint64_t>(
        omp_get_max_threads(), std::min<uint64_t>(chunks, count)));
    #pragma omp parallel num_threads(threads)
    {
        unsigned id = omp_get_thread_num(), team = omp_get_num_threads();
        work(uint64_t(count) * id / team, uint64_t(count) * (id + 1) / team);
    }
#else
    work(0, count);
#endif
}

// Each warp reads one slice of 32 rows, column by column. Work for a
// short row stops at its own length, even when another row is much longer.
__global__ void spmvKernel(const double* __restrict__ values, const index_t* __restrict__ cols,
                           const size_t* __restrict__ offsets, const index_t* __restrict__ lengths,
                           const double* __restrict__ vec, index_t rows, double* out) {
    index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t entry = offsets[row / sliceWidth] + row % sliceWidth;
    double sum = 0;
    #pragma unroll 8
    for (index_t j = 0; j < lengths[row]; ++j, entry += sliceWidth)
        sum += values[entry] * vec[cols[entry]];
    out[row] = sum;
}

template<class T>
T* upload(const std::vector<T>& data) {
    if (data.empty()) return nullptr;
    T* device;
    checkCuda(cudaMalloc(&device, data.size() * sizeof(T)));
    checkCuda(cudaMemcpy(device, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device;
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

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    uint64_t cells = uint64_t(numRows) * numRows;
    if (!sparsity || numRows > INT_MAX) {
        if (!rank) fprintf(stderr, "Require positive sparsity and at most INT_MAX rows.\n");
        MPI_Finalize();
        return 1;
    }
    uint64_t nItems = cells / sparsity;
    if (!rank) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - double(nItems) / double(cells)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices));

    auto firstRow = [&](int proc) { return index_t(uint64_t(numRows) * proc / ranks); };
    index_t first = firstRow(rank);
    index_t rows = firstRow(rank+1) - first;
    std::vector<double> vec(numRows);
    auto randomValue = [&](RandomStream& stream) {
        return maxVal * (stream.next() / (double(RAND_MAX) + 1.0));
    };
    if (!rank) {
        randomChunks(numRows, 1, [&](index_t begin, index_t end) {
            RandomStream stream(1, begin);
            for (index_t i = begin; i < end; ++i) vec[i] = randomValue(stream);
        });
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Count Bernoulli successes in each row before applying the original
    // cap and forced-tail rules. A single prefix exchange resolves both.
    const double probability = cells ? double(nItems) / double(cells) : 0;
    unsigned threshold = static_cast<unsigned>(probability * RAND_MAX);
    if (double(threshold) / RAND_MAX > probability) --threshold;
    if (threshold < RAND_MAX && double(threshold+1) / RAND_MAX <= probability) ++threshold;
    std::vector<uint64_t> prefix(size_t(rows)+1);
    if (nItems && nItems != cells) {
        randomChunks(rows, numRows, [&](index_t begin, index_t end) {
            RandomStream stream(8675309, uint64_t(first+begin) * numRows);
            for (index_t i = begin; i < end; ++i) {
                index_t count = 0;
                for (index_t j = 0; j < numRows; ++j) count += stream.next() <= threshold;
                prefix[i+1] = count;
            }
        });
    } else if (nItems) {
        std::fill(prefix.begin()+1, prefix.end(), numRows);
    }
    for (index_t i = 0; i < rows; ++i) prefix[i+1] += prefix[i];
    uint64_t preceding = 0;
    MPI_Exscan(&prefix.back(), &preceding, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (!rank) preceding = 0;
    for (index_t i = 0; i <= rows; ++i) {
        uint64_t remaining = cells - uint64_t(first+i) * numRows;
        uint64_t minimum = nItems > remaining ? nItems - remaining : 0;
        prefix[i] = std::clamp(preceding + prefix[i], minimum, nItems);
    }

    size_t slices = (size_t(rows) + sliceWidth-1) / sliceWidth;
    std::vector<size_t> offsets(slices+1);
    std::vector<index_t> lengths(rows);
    #pragma omp parallel for if(slices >= 128)
    for (size_t s = 0; s < slices; ++s) {
        size_t end = std::min<size_t>(rows, (s+1)*sliceWidth);
        index_t width = 0;
        for (size_t i = s*sliceWidth; i < end; ++i) {
            lengths[i] = prefix[i+1] - prefix[i];
            width = std::max(width, lengths[i]);
        }
        offsets[s+1] = width;
    }
    for (size_t s = 0; s < slices; ++s)
        offsets[s+1] = offsets[s] + offsets[s+1] * sliceWidth;
    std::vector<double> values(offsets.back()), reference(validate ? rows : 0);
    std::vector<index_t> cols(offsets.back());
    if (validate && !rank) printf("Computing reference solution...\n");
    randomChunks(rows, numRows, [&](index_t begin, index_t end) {
        if (prefix[begin] == prefix[end]) return;
        bool dense = nItems == cells;
        RandomStream positions(8675309, dense ? 0 : uint64_t(first+begin) * numRows);
#ifdef __GLIBC__
        RandomStream numbers(1, numRows + prefix[begin]);
#endif
        for (index_t i = begin; i < end; ++i) {
            size_t entry = offsets[i/sliceWidth] + i%sliceWidth;
            index_t left = lengths[i];
            double sum = 0;
            for (index_t j = 0; j < numRows; ++j) {
                bool hit = dense || positions.next() <= threshold;
                if (left && (hit || numRows-j == left)) {
                    cols[entry] = j;
#ifdef __GLIBC__
                    values[entry] = randomValue(numbers);
                    if (validate) sum += values[entry] * vec[j];
#endif
                    entry += sliceWidth;
                    --left;
                }
            }
            if (validate) reference[i] = sum;
        }
    });
#ifndef __GLIBC__
    // Native rand() shares process state. Generate the values separately.
    RandomStream numbers(1, numRows + prefix.front());
    for (index_t i = 0; i < rows; ++i) {
        size_t entry = offsets[i/sliceWidth] + i%sliceWidth;
        for (index_t j = 0; j < lengths[i]; ++j, entry += sliceWidth) {
            values[entry] = randomValue(numbers);
            if (validate) reference[i] += values[entry] * vec[cols[entry]];
        }
    }
#endif
    auto dOffsets = upload(offsets);
    auto dLengths = upload(lengths);
    auto dValues = upload(values);
    auto dCols = upload(cols);
    auto dVec = upload(vec);
    double* dOut = nullptr;
    if (rows) {
        checkCuda(cudaMalloc(&dOut, sizeof(double) * rows));
        if (!iterations) checkCuda(cudaMemset(dOut, 0, sizeof(double)*rows));
    }

    if (!rank) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rows) {
        unsigned threads = rows < 8192 ? 32 : 128;
        unsigned blocks = (rows + threads-1) / threads;
        for (index_t iter = 0; iter < iterations; ++iter)
            spmvKernel<<<blocks, threads>>>(dValues, dCols, dOffsets, dLengths, dVec, rows, dOut);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Average time per iteration: %.3f ms\n", iterations ? seconds * 1000 / iterations : 0);
        printf("Performance: %.3f GFLOPS\n", seconds ? 2e-9 * nItems * iterations / seconds : 0);
    }
    std::vector<double> out;
    if (printResults || validate) {
        out.resize(rows);
        if (rows) checkCuda(cudaMemcpy(out.data(), dOut, sizeof(double)*rows, cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        std::vector<double> global;
        std::vector<int> counts, starts;
        if (!rank) {
            global.resize(numRows);
            counts.resize(ranks);
            starts.resize(ranks);
            for (int p = 0; p < ranks; ++p) {
                starts[p] = firstRow(p);
                counts[p] = firstRow(p+1) - firstRow(p);
            }
        }
        MPI_Gatherv(out.data(), rows, MPI_DOUBLE, global.data(), counts.data(), starts.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(global, "OutputVector");
    }
    int failed = 0;
    if (validate) {
        index_t invalid = numRows;
        #pragma omp parallel for if(rows >= 4096) reduction(min:invalid)
        for (index_t i = 0; i < rows; ++i) {
            double diff = std::abs(out[i] - reference[i]);
            double error = std::abs(reference[i]) < 1e-10 ? std::abs(out[i]) : diff/std::abs(reference[i]);
            if (!(error <= MAX_RELATIVE_ERROR)) invalid = std::min(invalid, first+i);
        }
        index_t globalInvalid;
        MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
        failed = globalInvalid != numRows;
        if (failed && globalInvalid >= first && globalInvalid-first < rows) {
            index_t i = globalInvalid-first;
            printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                   globalInvalid, reference[i], out[i]);
        }
        if (!rank) printf("Validating result...\nValidation: %s\n", failed ? "FAILED" : "PASSED");
    }
    checkCuda(cudaFree(dOut));
    checkCuda(cudaFree(dVec));
    checkCuda(cudaFree(dCols));
    checkCuda(cudaFree(dValues));
    checkCuda(cudaFree(dLengths));
    checkCuda(cudaFree(dOffsets));
    MPI_Finalize();
    return failed;
}
