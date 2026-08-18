#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define CAHN_HILLIARD_HAS_MPI_EXT 1
#endif
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int THREADS_X = 32;
constexpr unsigned int THREADS_Y = 8;
constexpr int LOWER_TAG = 4100;
constexpr int UPPER_TAG = 4101;
constexpr double GRADIENT_COEFFICIENT = 0.5;
constexpr double DT_TIMES_DIFFUSION = 0.01;
constexpr double E_AA = -(2.0 / 9.0);
constexpr double E_BB = -(2.0 / 9.0);
constexpr double E_AB = 2.0 / 9.0;

void mpiCheck(const int error, const char* expression, const char* file, const int line) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: MPI error at %s:%d in %s: %.*s\n",
                 rank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

void cudaCheck(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error == cudaSuccess) {
        return;
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: CUDA error at %s:%d in %s: %s\n",
                 rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

#define MPI_CHECK(call) mpiCheck((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

bool checkedProduct(const std::size_t a, const std::size_t b, std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool parseSize(const char* text, std::size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (text[0] == '-' || end == text || *end != '\0' || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct Slab {
    std::size_t localNz;
    std::size_t globalZOffset;
    int lowerRank;
    int upperRank;
};

Slab decomposeZ(const std::size_t nz, const int rank, const int ranks) {
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t base = nz / rankCount;
    const std::size_t remainder = nz % rankCount;
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t localNz = base + (rankIndex < remainder ? 1 : 0);
    const std::size_t offset = rankIndex * base + std::min(rankIndex, remainder);
    return {localNz, offset,
            rank == 0 ? MPI_PROC_NULL : rank - 1,
            rank == ranks - 1 ? MPI_PROC_NULL : rank + 1};
}

// Both stencils operate on local z slabs with one ghost plane on either side.
// x and y retain the original clamped (zero normal derivative) boundaries.
__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ chemicalPotential,
                                        const std::size_t nx, const std::size_t ny,
                                        const std::size_t zBegin, const std::size_t zEnd) {
    const std::size_t plane = nx * ny;
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    const std::size_t inPlane = y * nx + x;
    for (std::size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        const std::size_t center = z * plane + inPlane;
        const std::size_t left = x == 0 ? center : center - 1;
        const std::size_t right = x + 1 == nx ? center : center + 1;
        const std::size_t down = y == 0 ? center : center - nx;
        const std::size_t up = y + 1 == ny ? center : center + nx;
        const double value = concentration[center];
        const double laplacian =
            (concentration[left] + concentration[right] - 2.0 * value) +
            (concentration[down] + concentration[up] - 2.0 * value) +
            (concentration[center - plane] + concentration[center + plane] - 2.0 * value);

        chemicalPotential[center] =
            4.5 * ((value + 1.0) * E_AA + (value - 1.0) * E_BB -
                   2.0 * value * E_AB) +
            3.0 * value + value * value * value - GRADIENT_COEFFICIENT * laplacian;
    }
}

__global__ void updateKernel(double* __restrict__ nextConcentration,
                             const double* __restrict__ concentration,
                             const double* __restrict__ chemicalPotential,
                             const std::size_t nx, const std::size_t ny,
                             const std::size_t zBegin, const std::size_t zEnd) {
    const std::size_t plane = nx * ny;
    const std::size_t x = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t y = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }

    const std::size_t inPlane = y * nx + x;
    for (std::size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
        const std::size_t center = z * plane + inPlane;
        const std::size_t left = x == 0 ? center : center - 1;
        const std::size_t right = x + 1 == nx ? center : center + 1;
        const std::size_t down = y == 0 ? center : center - nx;
        const std::size_t up = y + 1 == ny ? center : center + nx;
        const double value = chemicalPotential[center];
        const double laplacian =
            (chemicalPotential[left] + chemicalPotential[right] - 2.0 * value) +
            (chemicalPotential[down] + chemicalPotential[up] - 2.0 * value) +
            (chemicalPotential[center - plane] + chemicalPotential[center + plane] - 2.0 * value);

        nextConcentration[center] = concentration[center] + DT_TIMES_DIFFUSION * laplacian;
    }
}

void launchChemical(const double* concentration, double* chemicalPotential,
                    const std::size_t nx, const std::size_t ny,
                    const std::size_t zBegin, const std::size_t zEnd,
                    const int maxGridZ, cudaStream_t stream) {
    if (zBegin >= zEnd) {
        return;
    }
    const dim3 threads(THREADS_X, THREADS_Y, 1);
    const dim3 blocks(static_cast<unsigned int>((nx + THREADS_X - 1) / THREADS_X),
                      static_cast<unsigned int>((ny + THREADS_Y - 1) / THREADS_Y),
                      static_cast<unsigned int>(std::min<std::size_t>(zEnd - zBegin,
                                                                      maxGridZ)));
    chemicalPotentialKernel<<<blocks, threads, 0, stream>>>(
        concentration, chemicalPotential, nx, ny, zBegin, zEnd);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launchUpdate(double* nextConcentration, const double* concentration,
                  const double* chemicalPotential, const std::size_t nx,
                  const std::size_t ny, const std::size_t zBegin,
                  const std::size_t zEnd, const int maxGridZ, cudaStream_t stream) {
    if (zBegin >= zEnd) {
        return;
    }
    const dim3 threads(THREADS_X, THREADS_Y, 1);
    const dim3 blocks(static_cast<unsigned int>((nx + THREADS_X - 1) / THREADS_X),
                      static_cast<unsigned int>((ny + THREADS_Y - 1) / THREADS_Y),
                      static_cast<unsigned int>(std::min<std::size_t>(zEnd - zBegin,
                                                                      maxGridZ)));
    updateKernel<<<blocks, threads, 0, stream>>>(
        nextConcentration, concentration, chemicalPotential, nx, ny, zBegin, zEnd);
    CUDA_CHECK(cudaPeekAtLastError());
}

struct HaloRequests {
    MPI_Request requests[4];
};

struct HaloTransport {
    bool cudaAware;
    double* lowerReceive;
    double* upperReceive;
    double* lowerSend;
    double* upperSend;
};

bool useCudaAwareMpi() {
    const char* overrideValue = std::getenv("CAHN_HILLIARD_CUDA_AWARE_MPI");
    if (overrideValue != nullptr) {
        return std::strcmp(overrideValue, "0") != 0;
    }
#if defined(CAHN_HILLIARD_HAS_MPI_EXT) && defined(OMPI_HAVE_MPI_EXT_CUDA)
    return MPIX_Query_cuda_support() != 0;
#else
    // MPI has no portable capability query. Unknown implementations use the
    // safe pinned path unless the cluster environment explicitly enables it.
    return false;
#endif
}

HaloRequests postHaloExchange(double* field, const std::size_t plane,
                              const std::size_t localNz, const Slab& slab,
                              const HaloTransport& transport, cudaStream_t haloStream) {
    HaloRequests exchange{};
    const int count = static_cast<int>(plane);
    double* lowerReceive = field;
    double* upperReceive = field + (localNz + 1) * plane;
    double* lowerSend = field + plane;
    double* upperSend = field + localNz * plane;

    if (!transport.cudaAware) {
        if (slab.lowerRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(transport.lowerSend, lowerSend,
                                       plane * sizeof(double), cudaMemcpyDeviceToHost, haloStream));
        }
        if (slab.upperRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(transport.upperSend, upperSend,
                                       plane * sizeof(double), cudaMemcpyDeviceToHost, haloStream));
        }
        CUDA_CHECK(cudaStreamSynchronize(haloStream));
        lowerReceive = transport.lowerReceive;
        upperReceive = transport.upperReceive;
        lowerSend = transport.lowerSend;
        upperSend = transport.upperSend;
    }

    MPI_CHECK(MPI_Irecv(lowerReceive, count, MPI_DOUBLE, slab.lowerRank, UPPER_TAG,
                        MPI_COMM_WORLD, &exchange.requests[0]));
    MPI_CHECK(MPI_Irecv(upperReceive, count, MPI_DOUBLE, slab.upperRank, LOWER_TAG,
                        MPI_COMM_WORLD, &exchange.requests[1]));
    MPI_CHECK(MPI_Isend(lowerSend, count, MPI_DOUBLE, slab.lowerRank, LOWER_TAG,
                        MPI_COMM_WORLD, &exchange.requests[2]));
    MPI_CHECK(MPI_Isend(upperSend, count, MPI_DOUBLE, slab.upperRank,
                        UPPER_TAG, MPI_COMM_WORLD, &exchange.requests[3]));
    return exchange;
}

void finishHaloExchange(HaloRequests& exchange, double* field,
                        const std::size_t plane, const Slab& slab,
                        const HaloTransport& transport, cudaStream_t haloStream,
                        cudaStream_t computeStream, cudaEvent_t haloReady) {
    MPI_CHECK(MPI_Waitall(4, exchange.requests, MPI_STATUSES_IGNORE));

    if (slab.lowerRank != MPI_PROC_NULL && !transport.cudaAware) {
        CUDA_CHECK(cudaMemcpyAsync(field, transport.lowerReceive, plane * sizeof(double),
                                   cudaMemcpyHostToDevice, haloStream));
    } else if (slab.lowerRank == MPI_PROC_NULL) {
        // MPI_PROC_NULL leaves a ghost plane untouched, so reproduce the
        // original clamped z boundary from the nearest physical plane.
        CUDA_CHECK(cudaMemcpyAsync(field, field + plane, plane * sizeof(double),
                                   cudaMemcpyDeviceToDevice, haloStream));
    }
    if (slab.upperRank != MPI_PROC_NULL && !transport.cudaAware) {
        CUDA_CHECK(cudaMemcpyAsync(field + (slab.localNz + 1) * plane,
                                   transport.upperReceive, plane * sizeof(double),
                                   cudaMemcpyHostToDevice, haloStream));
    } else if (slab.upperRank == MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpyAsync(field + (slab.localNz + 1) * plane,
                                   field + slab.localNz * plane,
                                   plane * sizeof(double), cudaMemcpyDeviceToDevice, haloStream));
    }
    CUDA_CHECK(cudaEventRecord(haloReady, haloStream));
    CUDA_CHECK(cudaStreamWaitEvent(computeStream, haloReady, 0));
}

void initializeConcentration(std::vector<double>& concentration,
                             const std::size_t globalOffset,
                             const std::size_t globalVolume) {
    const long long cells = static_cast<long long>(concentration.size());
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < cells; ++local) {
        const std::size_t linearId = globalOffset + static_cast<std::size_t>(local);
        const double pseudo = ((linearId + 1) * std::size_t{1299709} % globalVolume) /
                              static_cast<double>(globalVolume);
        concentration[static_cast<std::size_t>(local)] = -1.0 + 2.0 * pseudo;
    }
}

bool validateDistributed(const std::vector<double>& localConcentration, const int rank) {
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localInvalid = 0;
    const long long cells = static_cast<long long>(localConcentration.size());

#pragma omp parallel for schedule(static) reduction(min : localMin) reduction(max : localMax) reduction(| : localInvalid)
    for (long long i = 0; i < cells; ++i) {
        const double value = localConcentration[static_cast<std::size_t>(i)];
        localMin = std::min(localMin, value);
        localMax = std::max(localMax, value);
        localInvalid |= !std::isfinite(value);
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    int globalInvalid = 0;
    MPI_CHECK(MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

    if (rank == 0) {
        if (globalInvalid != 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return globalInvalid == 0 && globalMax <= 10.0 && globalMin >= -10.0;
}

std::vector<double> gatherConcentration(const std::vector<double>& localConcentration,
                                        const std::size_t plane, const std::size_t nz,
                                        const int rank, const int ranks) {
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> globalConcentration;
    if (rank == 0) {
        counts.resize(static_cast<std::size_t>(ranks));
        displacements.resize(static_cast<std::size_t>(ranks));
        for (int source = 0; source < ranks; ++source) {
            const Slab sourceSlab = decomposeZ(nz, source, ranks);
            const std::size_t count = sourceSlab.localNz * plane;
            const std::size_t displacement = sourceSlab.globalZOffset * plane;
            if (count > static_cast<std::size_t>(INT_MAX) ||
                displacement > static_cast<std::size_t>(INT_MAX)) {
                std::fprintf(stderr, "Result output exceeds MPI_Gatherv's count range\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            counts[static_cast<std::size_t>(source)] = static_cast<int>(count);
            displacements[static_cast<std::size_t>(source)] = static_cast<int>(displacement);
        }
        globalConcentration.resize(plane * nz);
    }

    if (localConcentration.size() > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Local result output exceeds MPI_Gatherv's count range\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_CHECK(MPI_Gatherv(localConcentration.data(), static_cast<int>(localConcentration.size()),
                          MPI_DOUBLE, rank == 0 ? globalConcentration.data() : nullptr,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displacements.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));
    return globalConcentration;
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0;
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t nx = 64;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            argumentError |= !parseSize(argv[++i], nx);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            argumentError |= !parseSize(argv[++i], ny);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            argumentError |= !parseSize(argv[++i], nz);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentError |= !parseIterations(argv[++i], iterations);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
            argumentError = true;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (showHelp || argumentError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return argumentError ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (nz < static_cast<std::size_t>(ranks)) {
        if (rank == 0) {
            std::fprintf(stderr, "The z dimension (%zu) must be at least the MPI rank count (%d)\n", nz, ranks);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t plane = 0;
    std::size_t gridSize = 0;
    if (!checkedProduct(nx, ny, plane) || !checkedProduct(plane, nz, gridSize) ||
        plane > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions overflow the supported MPI/CUDA index range\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const Slab slab = decomposeZ(nz, rank, ranks);
    std::size_t allocatedCells = 0;
    std::size_t localCells = 0;
    if (slab.localNz > std::numeric_limits<std::size_t>::max() - 2 ||
        !checkedProduct(slab.localNz + 2, plane, allocatedCells) ||
        !checkedProduct(slab.localNz, plane, localCells) ||
        allocatedCells > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        localCells > static_cast<std::size_t>(LLONG_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Local slab size overflows the supported index range\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices are visible\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    int cudaAware = useCudaAwareMpi() ? 1 : 0;
    int allCudaAware = 0;
    MPI_CHECK(MPI_Allreduce(&cudaAware, &allCudaAware, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    HaloTransport haloTransport{allCudaAware != 0, nullptr, nullptr, nullptr, nullptr};
    double* hostHaloStorage = nullptr;
    if (!haloTransport.cudaAware) {
        CUDA_CHECK(cudaMallocHost(&hostHaloStorage, 4 * plane * sizeof(double)));
        haloTransport.lowerReceive = hostHaloStorage;
        haloTransport.upperReceive = hostHaloStorage + plane;
        haloTransport.lowerSend = hostHaloStorage + 2 * plane;
        haloTransport.upperSend = hostHaloStorage + 3 * plane;
    }

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s)/rank, CUDA GPUs\n",
                    ranks, omp_get_max_threads());
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        std::printf("MPI GPU transport: %s\n",
                    haloTransport.cudaAware ? "CUDA-aware direct" : "pinned-host staged");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> localConcentration(localCells);
    initializeConcentration(localConcentration, slab.globalZOffset * plane, gridSize);

    double* deviceConcentration = nullptr;
    double* deviceNext = nullptr;
    double* deviceChemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceConcentration, allocatedCells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceNext, allocatedCells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceChemicalPotential, allocatedCells * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(deviceConcentration + plane, localConcentration.data(),
                          localCells * sizeof(double), cudaMemcpyHostToDevice));

    cudaStream_t computeStream = nullptr;
    cudaStream_t haloStream = nullptr;
    cudaEvent_t haloReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&haloStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));

    if (rank == 0) {
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int step = 0; step < iterations; ++step) {
        // Ensure the previous update is visible before CUDA-aware MPI reads it.
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        HaloRequests concentrationExchange =
            postHaloExchange(deviceConcentration, plane, slab.localNz, slab,
                             haloTransport, haloStream);

        // z=[2, localNz) never consumes a ghost plane and overlaps halo traffic.
        launchChemical(deviceConcentration, deviceChemicalPotential, nx, ny, 2,
                       slab.localNz, deviceProperties.maxGridSize[2], computeStream);
        finishHaloExchange(concentrationExchange, deviceConcentration, plane, slab,
                           haloTransport, haloStream, computeStream, haloReady);
        launchChemical(deviceConcentration, deviceChemicalPotential, nx, ny, 1, 2,
                       deviceProperties.maxGridSize[2], computeStream);
        if (slab.localNz > 1) {
            launchChemical(deviceConcentration, deviceChemicalPotential, nx, ny,
                           slab.localNz, slab.localNz + 1,
                           deviceProperties.maxGridSize[2], computeStream);
        }

        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        HaloRequests potentialExchange =
            postHaloExchange(deviceChemicalPotential, plane, slab.localNz, slab,
                             haloTransport, haloStream);

        launchUpdate(deviceNext, deviceConcentration, deviceChemicalPotential,
                     nx, ny, 2, slab.localNz,
                     deviceProperties.maxGridSize[2], computeStream);
        finishHaloExchange(potentialExchange, deviceChemicalPotential, plane, slab,
                           haloTransport, haloStream, computeStream, haloReady);
        launchUpdate(deviceNext, deviceConcentration, deviceChemicalPotential,
                     nx, ny, 1, 2,
                     deviceProperties.maxGridSize[2], computeStream);
        if (slab.localNz > 1) {
            launchUpdate(deviceNext, deviceConcentration, deviceChemicalPotential,
                         nx, ny, slab.localNz, slab.localNz + 1,
                         deviceProperties.maxGridSize[2], computeStream);
        }
        std::swap(deviceConcentration, deviceNext);
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(localConcentration.data(), deviceConcentration + plane,
                              localCells * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        std::vector<double> globalConcentration =
            gatherConcentration(localConcentration, plane, nz, rank, ranks);
        if (rank == 0) {
            print_results(globalConcentration, "Concentration");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        valid = validateDistributed(localConcentration, rank);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaEventDestroy(haloReady));
    CUDA_CHECK(cudaStreamDestroy(haloStream));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    if (hostHaloStorage != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostHaloStorage));
    }
    CUDA_CHECK(cudaFree(deviceChemicalPotential));
    CUDA_CHECK(cudaFree(deviceNext));
    CUDA_CHECK(cudaFree(deviceConcentration));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
