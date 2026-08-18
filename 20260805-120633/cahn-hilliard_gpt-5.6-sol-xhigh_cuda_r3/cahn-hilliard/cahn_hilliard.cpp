#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 4;
constexpr unsigned int BLOCK_Z = 2;
constexpr unsigned int INIT_BLOCK_SIZE = 256;

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

void cudaCheck(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                     expression, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t plane) noexcept {
    return z * plane + y * nx + x;
}

__global__ void initializeConcentration(double* __restrict__ concentration, const size_t volume) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linearId = first; linearId < volume; linearId += stride) {
        const double pseudo = ((linearId + 1) * size_t{1299709} % volume) /
                              static_cast<double>(volume);
        concentration[linearId] = -1.0 + 2.0 * pseudo;
    }
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t nx, const size_t ny,
                                             const size_t nz, const size_t plane,
                                             const size_t x, const size_t y,
                                             const size_t z, const size_t center) {
    const size_t xp = x + static_cast<size_t>(x + 1 < nx);
    const size_t yp = y + static_cast<size_t>(y + 1 < ny);
    const size_t zp = z + static_cast<size_t>(z + 1 < nz);
    const size_t xn = x - static_cast<size_t>(x != 0);
    const size_t yn = y - static_cast<size_t>(y != 0);
    const size_t zn = z - static_cast<size_t>(z != 0);

    // dx, dy and dz are all one, so the three directional second derivatives
    // reduce to this equivalent seven-point stencil.
    return field[idx3(xp, y, z, nx, plane)] + field[idx3(xn, y, z, nx, plane)] +
           field[idx3(x, yp, z, nx, plane)] + field[idx3(x, yn, z, nx, plane)] +
           field[idx3(x, y, zp, nx, plane)] + field[idx3(x, y, zn, nx, plane)] -
           6.0 * field[center];
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void computeChemicalPotential(const double* __restrict__ concentration,
                              double* __restrict__ chemicalPotential,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = idx3(x, y, z, nx, plane);
    const double cv = concentration[index];
    const double lap = laplacian(concentration, nx, ny, nz, plane, x, y, z, index);

    constexpr double gamma = 0.5;
    // With eAA=eBB=-2/9 and eAB=2/9, the bulk free-energy derivative
    // simplifies exactly to c^3-c.
    chemicalPotential[index] = cv * cv * cv - cv - gamma * lap;
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void cahnHilliardUpdate(double* __restrict__ newConcentration,
                        const double* __restrict__ oldConcentration,
                        const double* __restrict__ chemicalPotential,
                        const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = idx3(x, y, z, nx, plane);
    constexpr double dtTimesD = 0.01;
    newConcentration[index] =
        oldConcentration[index] +
        dtTimesD * laplacian(chemicalPotential, nx, ny, nz, plane, x, y, z, index);
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto extrema = std::minmax_element(concentration.begin(), concentration.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *extrema.first, *extrema.second);
    if (*extrema.second > 10.0 || *extrema.first < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid and nonzero.\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Requested grid is too large.\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));
    std::printf("CUDA device: %s\n", deviceProperties.name);

    double* deviceConcentration[2] = {nullptr, nullptr};
    double* deviceChemicalPotential = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceConcentration[0], bytes));
    CUDA_CHECK(cudaMalloc(&deviceConcentration[1], bytes));
    CUDA_CHECK(cudaMalloc(&deviceChemicalPotential, bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("Initializing concentration field...\n");
    const size_t requiredInitBlocks = (gridSize + INIT_BLOCK_SIZE - 1) / INIT_BLOCK_SIZE;
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min(requiredInitBlocks,
                 static_cast<size_t>(deviceProperties.multiProcessorCount) * 32));
    initializeConcentration<<<initBlocks, INIT_BLOCK_SIZE, 0, stream>>>(
        deviceConcentration[0], gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExecutable = nullptr;
    int finalBuffer = 0;

    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int step = 0; step < iterations; ++step) {
            const int input = step & 1;
            const int output = input ^ 1;
            computeChemicalPotential<<<grid, block, 0, stream>>>(
                deviceConcentration[input], deviceChemicalPotential, nx, ny, nz);
            cahnHilliardUpdate<<<grid, block, 0, stream>>>(
                deviceConcentration[output], deviceConcentration[input],
                deviceChemicalPotential, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExecutable, graph, nullptr, nullptr, 0));
        finalBuffer = iterations & 1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(start, stream));
    if (iterations > 0) {
        CUDA_CHECK(cudaGraphLaunch(graphExecutable, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMilliseconds > 0.0F
        ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1000.0)
        : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> concentration;
    if (printResults || validate) {
        concentration.resize(gridSize);
        CUDA_CHECK(cudaMemcpyAsync(concentration.data(), deviceConcentration[finalBuffer], bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (printResults) {
        print_results(concentration, "Concentration");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(concentration);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    if (graphExecutable != nullptr) CUDA_CHECK(cudaGraphExecDestroy(graphExecutable));
    if (graph != nullptr) CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceChemicalPotential));
    CUDA_CHECK(cudaFree(deviceConcentration[1]));
    CUDA_CHECK(cudaFree(deviceConcentration[0]));
    return valid ? 0 : 1;
}
