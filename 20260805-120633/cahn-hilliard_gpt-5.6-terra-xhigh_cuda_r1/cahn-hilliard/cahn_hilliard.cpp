#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t cudaStatus = (call);                                                  \
        if (cudaStatus != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                         cudaGetErrorString(cudaStatus));                                       \
            std::exit(EXIT_FAILURE);                                                            \
        }                                                                                       \
    } while (false)

namespace {

// A 32-wide x dimension gives adjacent threads adjacent memory locations.  The
// remaining dimensions make a 256-thread block, which provides enough
// parallelism for the latency-bound double-precision stencil on Ampere GPUs.
constexpr dim3 kBlockSize{32, 4, 2};

inline dim3 makeGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t gx = (nx + kBlockSize.x - 1) / kBlockSize.x;
    const size_t gy = (ny + kBlockSize.y - 1) / kBlockSize.y;
    const size_t gz = (nz + kBlockSize.z - 1) / kBlockSize.z;

    if (gx > std::numeric_limits<unsigned int>::max() ||
        gy > std::numeric_limits<unsigned int>::max() ||
        gz > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Grid dimensions exceed CUDA launch limits\n");
        std::exit(EXIT_FAILURE);
    }
    return dim3(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy),
                static_cast<unsigned int>(gz));
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(const size_t count) {
        CUDA_CHECK(cudaMalloc(&data_, count * sizeof(*data_)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    double* get() noexcept { return data_; }
private:
    double* data_ = nullptr;
};

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t index, const size_t x,
                                             const size_t y, const size_t z,
                                             const size_t nx, const size_t ny,
                                             const size_t nz, const double invDx2,
                                             const double invDy2,
                                             const double invDz2) {
    const size_t plane = nx * ny;
    const size_t xp = x + 1 < nx ? index + 1 : index;
    const size_t xn = x > 0 ? index - 1 : index;
    const size_t yp = y + 1 < ny ? index + nx : index;
    const size_t yn = y > 0 ? index - nx : index;
    const size_t zp = z + 1 < nz ? index + plane : index;
    const size_t zn = z > 0 ? index - plane : index;
    const double center = field[index];

    const double cxx = (field[xp] + field[xn] - 2.0 * center) * invDx2;
    const double cyy = (field[yp] + field[yn] - 2.0 * center) * invDy2;
    const double czz = (field[zp] + field[zn] - 2.0 * center) * invDz2;
    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ concentration,
                                              const size_t nx, const size_t ny,
                                              const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t linearId = z * plane + y * nx + x;
    const size_t volume = plane * nz;
    const size_t pseudo = ((linearId + 1) * static_cast<size_t>(1299709)) % volume;
    concentration[linearId] = -1.0 + 2.0 * (static_cast<double>(pseudo) / static_cast<double>(volume));
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    const size_t nx, const size_t ny, const size_t nz, const double invDx2,
    const double invDy2, const double invDz2, const double gamma, const double eAA,
    const double eBB, const double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t index = z * (nx * ny) + y * nx + x;
    const double concentrationValue = concentration[index];
    chemicalPotential[index] =
        4.5 * ((concentrationValue + 1.0) * eAA + (concentrationValue - 1.0) * eBB -
               2.0 * concentrationValue * eAB) +
        3.0 * concentrationValue + concentrationValue * concentrationValue * concentrationValue -
        gamma * laplacian(concentration, index, x, y, z, nx, ny, nz, invDx2, invDy2, invDz2);
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ nextConcentration, const double* __restrict__ concentration,
    const double* __restrict__ chemicalPotential, const size_t nx, const size_t ny,
    const size_t nz, const double diffusionTimeStep, const double invDx2,
    const double invDy2, const double invDz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t index = z * (nx * ny) + y * nx + x;
    nextConcentration[index] = concentration[index] +
                               diffusionTimeStep * laplacian(chemicalPotential, index, x, y, z,
                                                             nx, ny, nz, invDx2, invDy2, invDz2);
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minValue = concentration[0];
    double maxValue = concentration[0];
    for (const double value : concentration) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
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

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);

    const size_t gridSize = nx * ny * nz;
    const dim3 grid = makeGrid(nx, ny, nz);

    DeviceBuffer concentration(gridSize);
    DeviceBuffer nextConcentration(gridSize);
    DeviceBuffer chemicalPotential(gridSize);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<grid, kBlockSize, 0, stream>>>(concentration.get(), nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("Running Cahn-Hilliard simulation...\n");
    double* current = concentration.get();
    double* next = nextConcentration.get();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    if (iterations > 0) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (int timeStep = 0; timeStep < iterations; ++timeStep) {
            computeChemicalPotentialKernel<<<grid, kBlockSize, 0, stream>>>(
                current, chemicalPotential.get(), nx, ny, nz, invDx2, invDy2, invDz2, gamma,
                eAA, eBB, eAB);
            CUDA_CHECK(cudaGetLastError());
            cahnHilliardUpdateKernel<<<grid, kBlockSize, 0, stream>>>(
                next, current, chemicalPotential.get(), nx, ny, nz, diffusion * dt, invDx2,
                invDy2, invDz2);
            CUDA_CHECK(cudaGetLastError());
            std::swap(current, next);
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        CUDA_CHECK(cudaEventRecord(startEvent, stream));
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
    } else {
        CUDA_CHECK(cudaEventRecord(startEvent, stream));
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
    }

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    const double durationMilliseconds = static_cast<double>(elapsedMilliseconds);
    std::printf("Computation time: %.3f ms\n", durationMilliseconds);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (static_cast<double>(durationMilliseconds) / 1000.0) / 1.0e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> hostConcentration(gridSize);
        CUDA_CHECK(cudaMemcpyAsync(hostConcentration.data(), current, gridSize * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (printResults) {
            print_results(hostConcentration, "Concentration");
        }
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(hostConcentration);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaEventDestroy(startEvent));
            CUDA_CHECK(cudaEventDestroy(stopEvent));
            if (graphExec != nullptr) {
                CUDA_CHECK(cudaGraphExecDestroy(graphExec));
            }
            if (graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph));
            }
            CUDA_CHECK(cudaStreamDestroy(stream));
            return valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    if (graphExec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    }
    if (graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
