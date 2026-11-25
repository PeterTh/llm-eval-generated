#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Word = unsigned int;
constexpr Word INF = 1000000000;
// Every real distance starts at most 201 and only decreases. A padded
// vertex with weight 255 cannot improve a real path. Byte panels cut
// the bandwidth required by each pivot broadcast by a factor of four.
using PanelValue = unsigned char;
constexpr Word padding = UCHAR_MAX;
constexpr int B = 32;
constexpr int H = 8;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Save the pivot tile's row and column operands as they stood before
// each scalar k. Using these snapshots also preserves path tie ordering.
__global__ void factorPivot(const Word* pivot, size_t pitch, PanelValue* factors) {
    __shared__ Word tile[B][B];
    int x = threadIdx.x;
    for (int y = threadIdx.y; y < B; y += H)
        tile[y][x] = pivot[y * pitch + x];
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        if (threadIdx.y == 0) {
            factors[k * B + x] = tile[x][k];
        }
        for (int y = threadIdx.y; y < B; y += H) {
            if (y != k && x != k)
                tile[y][x] = min(tile[y][x], tile[y][k] + tile[k][x]);
        }
        __syncthreads();
    }
}

// Replay the pivot tile on an entire block row or column, emitting one
// immutable operand per element for the outer-product updates below.
template<bool Row>
__global__ void makePanel(const Word* base, size_t pitch, const PanelValue* factors,
                          size_t factorStride, PanelValue* panel) {
    __shared__ Word tile[B][B + 1];
    __shared__ Word coeff[B][B + 1];
    int x = threadIdx.x;
    size_t start = size_t(blockIdx.x) * B;
    for (int y = threadIdx.y; y < B; y += H) {
        tile[y][x] = Row ? base[y * pitch + start + x]
                         : base[(start + y) * pitch + x];
        coeff[y][x] = factors[y * factorStride + x];
    }
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        // The pivot line is immutable in this iteration (zero diagonal).
        if (threadIdx.y == 0) {
            if (Row) panel[k * pitch + start + x] = tile[k][x];
            else panel[(start + x) * B + k] = tile[x][k];
        }
        for (int y = threadIdx.y; y < B; y += H) {
            if (Row && y != k)
                tile[y][x] = min(tile[y][x], coeff[k][y] + tile[k][x]);
            if (!Row && x != k)
                tile[y][x] = min(tile[y][x], tile[y][k] + coeff[k][x]);
        }
        __syncthreads();
    }
}

__global__ void update(Word* dist, Word* path, size_t pitch, int firstK,
                       const PanelValue* rows, const PanelValue* cols, int localTiles) {
    __shared__ Word row[B][B];
    __shared__ Word col[B][B];
    int x = threadIdx.x;
    size_t j = size_t(blockIdx.x) * B + x;
    for (int t = blockIdx.y; t < localTiles; t += gridDim.y) {
        size_t i = size_t(t) * B;
        for (int y = threadIdx.y; y < B; y += H) {
            row[y][x] = rows[y * pitch + j];
            col[y][x] = cols[(i + y) * B + x];
        }
        __syncthreads();
        #pragma unroll
        for (int y = threadIdx.y; y < B; y += H) {
            size_t index = (i + y) * pitch + j;
            Word best = dist[index];
            int last = -1;
            #pragma unroll
            for (int k = 0; k < B; ++k) {
                Word candidate = col[y][k] + row[k][x];
                if (candidate < best) {
                    best = candidate;
                    last = k;
                }
            }
            dist[index] = best;
            if (last >= 0) path[index] = firstK + last;
        }
        __syncthreads();
    }
}

#ifdef __GLIBC__
// glibc rand_r advances its unsigned LCG three times per result.
// Exponentiate that affine transform to start each thread independently.
Word rowSeed(size_t offset) {
    Word a = 1103515245u, b = 12345u, seed = 42;
    for (size_t steps = offset * 3; steps; steps >>= 1) {
        if (steps & 1) seed = a * seed + b;
        b *= a + 1;
        a *= a;
    }
    return seed;
}
#endif

void initialize(std::vector<Word>& host, int first, int n, size_t pitch, int count) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int team = omp_get_num_threads();
        int begin = static_cast<long long>(tid) * count / team;
        int end = static_cast<long long>(tid + 1) * count / team;
        size_t offset = size_t(first + begin) * n;
#ifdef __GLIBC__
        Word seed = rowSeed(offset);
#else
        Word seed = 42;
        if (begin < end)
            for (size_t i = 0; i < offset; ++i) rand_r(&seed);
#endif
        for (int r = begin; r < end; ++r) {
            Word* row = host.data() + r * pitch;
            for (int j = 0; j < n; ++j)
                row[j] = 1 + Word(200.0 * rand_r(&seed) / double(RAND_MAX));
            row[first + r] = 0;
        }
    }
}

template<typename T>
T* deviceArray(size_t count) {
    T* ptr;
    check(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

__global__ void initializePaths(Word* path, size_t count, size_t pitch, int first) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x)
        path[i] = first + i / pitch;
}

// Keep classic MPI interfaces usable when a matrix exceeds INT_MAX elements.
template<typename Transfer>
void chunks(size_t count, Transfer transfer) {
    for (size_t pos = 0; pos < count;) {
        int length = static_cast<int>(std::min(count - pos, size_t(INT_MAX)));
        transfer(pos, length);
        pos += length;
    }
}

// Rank zero owns the first tile. Only its sampled rows, the first few
// local columns, and the local diagonal are needed for the original checks.
bool validateResult(const Word* dist, size_t pitch, int n, int first,
                    int count, int rank, MPI_Comm comm) {
    int sample = std::min(10, n);
    std::vector<Word> rows(size_t(sample) * n), cols(size_t(count) * sample);
    std::vector<Word> diagonal(count);
    if (!rank)
        check(cudaMemcpy2D(rows.data(), size_t(n) * sizeof(Word), dist,
                           pitch * sizeof(Word), size_t(n) * sizeof(Word),
                           sample, cudaMemcpyDeviceToHost));
    chunks(rows.size(), [&](size_t pos, int length) {
        MPI_Bcast(rows.data() + pos, length, MPI_UNSIGNED, 0, comm);
    });
    check(cudaMemcpy2D(cols.data(), sample * sizeof(Word), dist, pitch * sizeof(Word),
                       sample * sizeof(Word), count, cudaMemcpyDeviceToHost));
    check(cudaMemcpy2D(diagonal.data(), sizeof(Word), dist + first,
                       (pitch + 1) * sizeof(Word), sizeof(Word), count, cudaMemcpyDeviceToHost));
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int r = 0; r < count; ++r) {
        valid &= diagonal[r] == 0;
        for (int i = 0; i < sample; ++i) {
            Word left = rows[size_t(i) * n + first + r];
            for (int j = 0; j < sample; ++j) {
                Word right = cols[size_t(r) * sample + j];
                valid &= left >= INF || right >= INF || left + right >= rows[size_t(i) * n + j];
            }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
    return valid;
}

int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            long value = strtol(argv[++i], nullptr, 10);
            if (value < 0 || value > INT_MAX - B) {
                if (!rank) fprintf(stderr, "Invalid graph size\n");
                return 1;
            }
            n = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\nValidation: %s\nInitializing graph...\n",
               n, validate ? "enabled" : "disabled");
    }

    int tiles = (n + B - 1) / B;
    int workers = std::min(ranks, std::max(tiles, 1));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < workers ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= workers) return 0;
    auto firstTile = [&](int r) {
        return r * (tiles / workers) + std::min(r, tiles % workers);
    };
    int first = firstTile(rank) * B;
    int localTiles = firstTile(rank + 1) - firstTile(rank);
    int realRows = std::min(n - first, localTiles * B);
    size_t pitch = size_t(tiles) * B;
    size_t cells = pitch * localTiles * B;
    std::vector<Word> answer;
    bool valid = true;
    if (n) {
        MPI_Comm shared;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
        int localRank, devices;
        MPI_Comm_rank(shared, &localRank);
        MPI_Comm_free(&shared);
        check(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
            MPI_Abort(comm, 1);
        }
        check(cudaSetDevice(localRank % devices));
        Word* dist = deviceArray<Word>(cells);
        Word* path = deviceArray<Word>(cells);
        {
            std::vector<Word> host(cells, padding);
            initialize(host, first, n, pitch, realRows);
            check(cudaMemcpy(dist, host.data(), cells * sizeof(Word), cudaMemcpyHostToDevice));
        }
        initializePaths<<<std::min(size_t(65535), (cells + 255) / 256), 256>>>(
            path, cells, pitch, first);
        PanelValue* factors = deviceArray<PanelValue>(B * B);
        PanelValue* rows = deviceArray<PanelValue>(B * pitch);
        PanelValue* cols = deviceArray<PanelValue>(size_t(localTiles) * B * B);
        PanelValue* staging = nullptr;
        bool direct = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        direct = MPIX_Query_cuda_support() != 0;
#endif
        if (workers > 1 && !direct)
            check(cudaMallocHost(&staging, B * pitch * sizeof(PanelValue)));
        check(cudaDeviceSynchronize());
        if (!rank) printf("Computing shortest paths...\n");
        MPI_Barrier(comm);
        double start = MPI_Wtime();
        dim3 threads(B, H);
        int owner = 0;
        for (int tile = 0; tile < tiles; ++tile) {
            if (tile == firstTile(owner + 1)) ++owner;
            int k = tile * B;
            if (owner == rank) {
                Word* band = dist + size_t(k - first) * pitch;
                factorPivot<<<1, threads>>>(band + k, pitch, factors);
                makePanel<true><<<tiles, threads>>>(band, pitch, factors, B, rows);
            }
            if (workers > 1) {
                size_t bytes = B * pitch * sizeof(PanelValue);
                if (direct) check(cudaDeviceSynchronize());
                else if (owner == rank)
                    check(cudaMemcpy(staging, rows, bytes, cudaMemcpyDeviceToHost));
                chunks(B * pitch, [&](size_t pos, int count) {
                    MPI_Bcast((direct ? rows : staging) + pos, count, MPI_UNSIGNED_CHAR, owner, comm);
                });
                if (!direct && owner != rank)
                    check(cudaMemcpy(rows, staging, bytes, cudaMemcpyHostToDevice));
            }
            makePanel<false><<<localTiles, threads>>>(dist + k, pitch, rows + k, pitch, cols);
            dim3 grid(tiles, std::min(65535, localTiles));
            update<<<grid, threads>>>(dist, path, pitch, k, rows, cols, localTiles);
        }
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        double elapsed = MPI_Wtime() - start, slowest;
        MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) printf("Computation time: %.0f ms\nPerformance: %.3f GOPS\n",
                          slowest * 1000, double(n) * n * n / slowest / 1e9);

        if (validate) valid = validateResult(dist, pitch, n, first, realRows, rank, comm);
        if (results) {
            // Strip padding and concatenate rank slabs in original row-major order.
            answer.resize(size_t(rank ? realRows : n) * n);
            check(cudaMemcpy2D(answer.data(), size_t(n) * sizeof(Word), dist,
                               pitch * sizeof(Word), size_t(n) * sizeof(Word),
                               realRows, cudaMemcpyDeviceToHost));
            if (rank) {
                chunks(answer.size(), [&](size_t pos, int count) {
                    MPI_Send(answer.data() + pos, count, MPI_UNSIGNED, 0, 0, comm);
                });
            } else {
                for (int r = 1; r < workers; ++r) {
                    int begin = firstTile(r) * B;
                    int end = std::min(n, firstTile(r + 1) * B);
                    chunks(size_t(end - begin) * n, [&](size_t pos, int count) {
                        MPI_Recv(answer.data() + size_t(begin) * n + pos, count,
                                 MPI_UNSIGNED, r, 0, comm, MPI_STATUS_IGNORE);
                    });
                }
            }
        }
        if (staging) check(cudaFreeHost(staging));
        check(cudaFree(dist));
        check(cudaFree(path));
        check(cudaFree(factors));
        check(cudaFree(rows));
        check(cudaFree(cols));
    } else if (!rank) {
        printf("Computing shortest paths...\nComputation time: 0 ms\nPerformance: 0.000 GOPS\n");
    }
    MPI_Comm_free(&comm);
    if (!rank && results) print_results_int(answer, "DistanceMatrix");
    if (!rank && validate) {
        printf("Validating result...\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
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
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
