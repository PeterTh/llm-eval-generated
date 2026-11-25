#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

struct Range {
    int begin, end;
    int size() const { return end - begin; }
};

Range slab(int rank, int ranks, int length) {
    const int base = length / ranks;
    const int extra = length % ranks;
    const int start = rank * base + std::min(rank, extra);
    return {start, start + base + (rank < extra)};
}

struct Grid {
    int width, height, depth, axis;
    Range owned;

    size_t plane() const { return size_t(width) * height; }
    Range stored() const { return {std::max(0, owned.begin - 2), std::min(depth, owned.end + 2)}; }
    size_t offset(int z) const { return size_t(z - stored().begin) * plane(); }
};

struct Transfer {
    int peer, count;
    size_t offset, staging;
};

void addTransfers(std::vector<Transfer>& transfers, size_t& stagingSize,
                  int peer, Range planes, const Grid& grid) {
    if (planes.size() <= 0) return;
    const size_t begin = grid.offset(planes.begin);
    const size_t end = grid.offset(planes.end);
    for (size_t offset = begin; offset < end;) {
        const int count = static_cast<int>(std::min(end - offset, size_t(INT_MAX)));
        transfers.push_back({peer, count, offset, stagingSize});
        offset += count;
        stagingSize += count;
    }
}

class HaloExchange {
public:
    HaloExchange(const Grid& grid, int rank, int ranks, MPI_Comm comm) : comm_(comm) {
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        direct_ = MPIX_Query_cuda_support() != 0;
#endif
        size_t receiveSize = 0, sendSize = 0;
        // A two-plane halo can reach two peers when a neighbor owns only one plane.
        for (int peer = std::max(0, rank - 2); peer <= std::min(ranks - 1, rank + 2); ++peer) {
            if (peer == rank) continue;
            const Range other = slab(peer, ranks, grid.depth);
            addTransfers(receives_, receiveSize, peer,
                         {std::max(other.begin, grid.owned.begin - 2),
                          std::min(other.end, grid.owned.end + 2)}, grid);
            addTransfers(sends_, sendSize, peer,
                         {std::max(grid.owned.begin, other.begin - 2),
                          std::min(grid.owned.end, other.end + 2)}, grid);
        }
        requests_.resize(receives_.size() + sends_.size());
        if (requests_.empty()) return;

        CUDA_CHECK(cudaEventCreateWithFlags(&fieldReady_, cudaEventDisableTiming));
        if (!direct_) {
            CUDA_CHECK(cudaMallocHost(&sendBuffer_, sendSize * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&receiveBuffer_, receiveSize * sizeof(double)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&received_, cudaEventDisableTiming));
        }
    }

    ~HaloExchange() {
        if (requests_.empty()) return;
        CUDA_CHECK(cudaEventDestroy(fieldReady_));
        if (!direct_) {
            CUDA_CHECK(cudaEventDestroy(received_));
            CUDA_CHECK(cudaStreamDestroy(copy_));
            CUDA_CHECK(cudaFreeHost(receiveBuffer_));
            CUDA_CHECK(cudaFreeHost(sendBuffer_));
        }
    }

    void begin(double* field, cudaStream_t compute) {
        // The outgoing planes depend on the previous update. Interior work queued
        // after this event can run concurrently with copies and MPI.
        CUDA_CHECK(cudaEventRecord(fieldReady_, compute));
        if (!direct_) {
            CUDA_CHECK(cudaStreamWaitEvent(copy_, fieldReady_, 0));
            for (const auto& send : sends_) {
                CUDA_CHECK(cudaMemcpyAsync(sendBuffer_ + send.staging, field + send.offset,
                                           send.count * sizeof(double), cudaMemcpyDeviceToHost, copy_));
            }
            // Keep MPI from reusing staging memory while the last upload reads it.
            CUDA_CHECK(cudaEventSynchronize(received_));
        }
        for (size_t i = 0; i < receives_.size(); ++i) {
            const auto& recv = receives_[i];
            double* destination = direct_ ? field + recv.offset : receiveBuffer_ + recv.staging;
            MPI_Irecv(destination, recv.count, MPI_DOUBLE, recv.peer, 0, comm_, &requests_[i]);
        }
    }

    void finish(double* field, cudaStream_t compute) {
        if (direct_) {
            CUDA_CHECK(cudaEventSynchronize(fieldReady_));
        } else {
            CUDA_CHECK(cudaStreamSynchronize(copy_));
        }
        for (size_t i = 0; i < sends_.size(); ++i) {
            const auto& send = sends_[i];
            const double* source = direct_ ? field + send.offset : sendBuffer_ + send.staging;
            MPI_Isend(source, send.count, MPI_DOUBLE, send.peer, 0, comm_,
                      &requests_[receives_.size() + i]);
        }
        MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
        if (!direct_) {
            for (const auto& recv : receives_) {
                CUDA_CHECK(cudaMemcpyAsync(field + recv.offset, receiveBuffer_ + recv.staging,
                                           recv.count * sizeof(double), cudaMemcpyHostToDevice, copy_));
            }
            CUDA_CHECK(cudaEventRecord(received_, copy_));
            CUDA_CHECK(cudaStreamWaitEvent(compute, received_, 0));
        }
    }

private:
    MPI_Comm comm_;
    bool direct_ = false;
    std::vector<Transfer> receives_, sends_;
    std::vector<MPI_Request> requests_;
    double* sendBuffer_ = nullptr;
    double* receiveBuffer_ = nullptr;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t fieldReady_ = nullptr;
    cudaEvent_t received_ = nullptr;
};

template<int Axis>
__device__ double laplacian(const double* field, size_t index, double center,
                            int x, int y, int z, int width, int height, int depth, size_t plane) {
    // The spacing is one in every direction. Sum in the original X, Y, Z order.
    const double twice = 2.0 * center;
    const double across = (field[index + (x + 1 < width)] + field[index - (x > 0)]) - twice;
    const size_t row = width;
    const double down = (field[index + (y + 1 < height ? row : 0)] +
                         field[index - (y > 0 ? row : 0)]) - twice;
    const double through = (field[index + (z + 1 < depth ? plane : 0)] +
                            field[index - (z > 0 ? plane : 0)]) - twice;
    if constexpr (Axis == 0) return (through + across) + down;
    if constexpr (Axis == 1) return (across + through) + down;
    return (across + down) + through;
}

template<int Axis, bool Potential>
__global__ void stencil(const double* __restrict__ input, const double* __restrict__ cold,
                        double* __restrict__ output, int width, int height, int depth,
                        int firstRow, int firstPlane, int firstSlot, int planes, int split, int gap) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = firstRow + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= unsigned(width) || y >= unsigned(height)) return;

    const size_t plane = size_t(width) * height;
    for (size_t p = blockIdx.z; p < size_t(planes); p += gridDim.z) {
        const int relative = static_cast<int>(p) + (p >= size_t(split) ? gap : 0);
        const int z = firstPlane + relative;
        const size_t index = (size_t(firstSlot) + relative) * plane + size_t(y) * width + x;
        const double center = input[index];
        const double lap = laplacian<Axis>(input, index, center, x, y, z, width, height, depth, plane);
        if constexpr (Potential) {
            constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            constexpr double gamma = 0.5;
            double chemical = fma(center + 1.0, eAA, (center - 1.0) * eBB);
            chemical = fma(-2.0 * center, eAB, chemical);
            chemical = fma(4.5, chemical, 3.0 * center);
            chemical = fma(center * center, center, chemical);
            output[index] = fma(-gamma, lap, chemical);
        } else {
            constexpr double dt = 0.01, D = 1.0;
            output[index] = fma(dt * D, lap, cold[index]);
        }
    }
}

template<int Axis, bool Potential>
void launchStencil(const Grid& grid, Range first, Range second, const double* input,
                   const double* cold, double* output, cudaStream_t stream) {
    const int planes = first.size() + second.size();
    if (planes == 0) return;
    unsigned tileWidth = 32;
    while (tileWidth / 2 >= unsigned(grid.width)) tileWidth /= 2;
    const dim3 block(tileWidth, 128 / tileWidth);
    for (unsigned firstRow = 0; firstRow < unsigned(grid.height); firstRow += 65535 * block.y) {
        const dim3 blocks((unsigned(grid.width) + block.x - 1) / block.x,
                          std::min(65535u, (unsigned(grid.height) - firstRow + block.y - 1) / block.y),
                          std::min(planes, 65535));
        stencil<Axis, Potential><<<blocks, block, 0, stream>>>(
            input, cold, output, grid.width, grid.height, grid.depth, firstRow, first.begin,
            first.begin - grid.stored().begin, planes, first.size(), second.begin - first.end);
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

template<int Axis>
double evolve(const Grid& grid, int iterations, int rank, int ranks, MPI_Comm comm,
              double*& cold, double*& next, double* mu) {
    HaloExchange halo(grid, rank, ranks, comm);
    cudaStream_t compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    const auto [begin, end] = grid.owned;
    const Range needed{std::max(0, begin - 1), std::min(grid.depth, end + 1)};
    const Range interior{begin + (begin != 0),
                         std::max(begin + (begin != 0), end - (end != grid.depth))};
    const Range empty{end, end};
    // An interior update also needs the neighboring potentials to be ready.
    // For thin slabs, keep one update launch for the whole slab.
    const Range updateInterior{interior.begin + (begin != 0), interior.end - (end != grid.depth)};
    const bool splitUpdate = ranks > 1 && updateInterior.size() >= 8;

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        if (ranks > 1) {
            halo.begin(cold, compute);
            launchStencil<Axis, true>(grid, interior, empty, cold, nullptr, mu, compute);
            if (splitUpdate) {
                launchStencil<Axis, false>(grid, updateInterior, empty, mu, cold, next, compute);
            }
            halo.finish(cold, compute);
            launchStencil<Axis, true>(grid, {needed.begin, interior.begin},
                                      {interior.end, needed.end}, cold, nullptr, mu, compute);
        } else {
            launchStencil<Axis, true>(grid, needed, empty, cold, nullptr, mu, compute);
        }
        if (splitUpdate) {
            launchStencil<Axis, false>(grid, {begin, updateInterior.begin},
                                       {updateInterior.end, end}, mu, cold, next, compute);
        } else {
            launchStencil<Axis, false>(grid, grid.owned, empty, mu, cold, next, compute);
        }
        std::swap(cold, next);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    const double elapsed = MPI_Wtime() - start;
    CUDA_CHECK(cudaStreamDestroy(compute));
    return elapsed;
}

int hostThreads(size_t cells) {
    return static_cast<int>(std::min(size_t(omp_get_max_threads()), 1 + cells / 65536));
}

size_t globalIndex(const Grid& grid, size_t local, size_t nx, size_t ny) {
    const size_t z = local / grid.plane();
    const size_t y = (local / grid.width) % grid.height;
    const size_t x = local % grid.width;
    if (grid.axis == 0) return (y * ny + x) * nx + z;
    if (grid.axis == 1) return (y * ny + z) * nx + x;
    return local;
}

void initialize(double* device, const Grid& grid, size_t nx, size_t ny, size_t volume) {
    const size_t count = grid.owned.size() * grid.plane();
    const size_t first = grid.owned.begin * grid.plane();
    double* host = new double[count];
    const int threads = hostThreads(count);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < count; ++i) {
        const size_t index = globalIndex(grid, first + i, nx, ny);
        const double pseudo = ((index + 1) * 1299709 % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    CUDA_CHECK(cudaMemcpy(device + grid.offset(grid.owned.begin), host,
                          count * sizeof(double), cudaMemcpyHostToDevice));
    delete[] host;
}

void reportResults(const std::vector<double>& local, const Grid& grid, size_t nx, size_t ny,
                   size_t volume, int rank, int ranks, MPI_Comm comm) {
    const size_t localStart = grid.owned.begin * grid.plane();
    if (ranks == 1 && grid.axis == 2) {
        print_results(local, "Concentration");
        return;
    }

    std::vector<double> gathered(rank == 0 ? volume : 0);
    std::vector<int> counts(rank == 0 ? ranks : 0), offsets(counts.size());
    // MPI_Gatherv uses signed int counts and displacements. Large fields are
    // gathered in windows so the output still uses the full global ordering.
    for (size_t start = 0; start < volume;) {
        const size_t end = start + std::min(volume - start, size_t(INT_MAX));
        auto overlap = [&](size_t first, size_t last) {
            const size_t begin = std::max(first, start);
            const size_t finish = std::min(last, end);
            return std::pair{begin, finish > begin ? static_cast<int>(finish - begin) : 0};
        };
        const auto [sendStart, sendCount] = overlap(localStart, localStart + local.size());
        if (rank == 0) {
            for (int peer = 0; peer < ranks; ++peer) {
                const Range owned = slab(peer, ranks, grid.depth);
                const auto [begin, count] = overlap(owned.begin * grid.plane(), owned.end * grid.plane());
                counts[peer] = count;
                offsets[peer] = count ? static_cast<int>(begin - start) : 0;
            }
        }
        MPI_Gatherv(local.data() + (sendCount ? sendStart - localStart : 0), sendCount, MPI_DOUBLE,
                    rank == 0 ? gathered.data() + start : nullptr, counts.data(),
                    offsets.data(), MPI_DOUBLE, 0, comm);
        start = end;
    }
    if (rank == 0) {
        if (grid.axis == 2) {
            print_results(gathered, "Concentration");
        } else {
            std::vector<double> ordered(volume);
            const int threads = hostThreads(volume);
            #pragma omp parallel for schedule(static) num_threads(threads)
            for (size_t i = 0; i < volume; ++i) {
                ordered[globalIndex(grid, i, nx, ny)] = gathered[i];
            }
            print_results(ordered, "Concentration");
        }
    }
}

bool validateResult(const std::vector<double>& local, int rank, MPI_Comm comm) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    int invalid = 0;
    const int threads = hostThreads(local.size());
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal) \
                             reduction(|:invalid) num_threads(threads)
    for (size_t i = 0; i < local.size(); ++i) {
        const double value = local[i];
        if (std::isfinite(value)) {
            minVal = std::min(minVal, value);
            maxVal = std::max(maxVal, value);
        } else {
            invalid = 1;
        }
    }
    const double bounds[2] = {minVal, -maxVal};
    double globalBounds[2];
    int globalInvalid;
    MPI_Reduce(bounds, globalBounds, 2, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, 0, comm);
    int valid = 1;
    if (rank == 0) {
        if (globalInvalid) {
            printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            minVal = globalBounds[0];
            maxVal = -globalBounds[1];
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0) {
                printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int nx = 64, ny = 0, nz = 0, iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const int depth = worldSize == 1 ? nz : std::max({nx, ny, nz});
    const int axis = depth == nz ? 2 : depth == ny ? 1 : 0;
    const int width = axis == 0 ? ny : nx;
    const int height = axis == 2 ? ny : nz;
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx <= 0 || ny <= 0 || nz <= 0 || depth > INT_MAX - 4 ||
        maxCells / size_t(width) / size_t(height) < size_t(depth)) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t volume = size_t(nx) * ny * nz;

    // Exclude empty slabs from device work and the simulation collectives.
    const int ranks = std::min(worldSize, depth);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    const Grid grid{width, height, depth, axis, slab(rank, ranks, depth)};
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    const size_t bytes = grid.stored().size() * grid.plane() * sizeof(double);
    double *cold, *next, *mu;
    CUDA_CHECK(cudaMalloc(&cold, bytes));
    CUDA_CHECK(cudaMalloc(&next, bytes));
    CUDA_CHECK(cudaMalloc(&mu, bytes));
    initialize(cold, grid, nx, ny, volume);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    double localTime;
    switch (grid.axis) {
        case 0: localTime = evolve<0>(grid, iterations, rank, ranks, comm, cold, next, mu); break;
        case 1: localTime = evolve<1>(grid, iterations, rank, ranks, comm, cold, next, mu); break;
        default: localTime = evolve<2>(grid, iterations, rank, ranks, comm, cold, next, mu); break;
    }
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / elapsed / 1e6);
    }

    std::vector<double> local;
    if (printResults || validate) {
        local.resize(grid.owned.size() * grid.plane());
        CUDA_CHECK(cudaMemcpy(local.data(), cold + grid.offset(grid.owned.begin),
                              local.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(next));
    CUDA_CHECK(cudaFree(cold));
    if (printResults) reportResults(local, grid, nx, ny, volume, rank, ranks, comm);
    if (validate && rank == 0) printf("Validating result...\n");
    const bool valid = !validate || validateResult(local, rank, comm);
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int status = run(argc, argv);
    MPI_Finalize();
    return status;
}
