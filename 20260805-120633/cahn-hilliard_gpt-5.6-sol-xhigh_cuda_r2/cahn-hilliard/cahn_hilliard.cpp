#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 4;
constexpr unsigned int BLOCK_Z = 2;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void checkCuda(const cudaError_t error, const char* expression,
                      const char* file, const int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, expression, file, line);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename Index>
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                            const Index index, const Index x,
                                            const Index y, const Index z,
                                            const Index nx, const Index ny,
                                            const Index nz, const Index plane) {
    // Reusing the center at an edge exactly implements the original clamped
    // (zero normal derivative) boundary condition.
    const Index xm = (x != 0) ? index - 1 : index;
    const Index xp = (x + 1 < nx) ? index + 1 : index;
    const Index ym = (y != 0) ? index - nx : index;
    const Index yp = (y + 1 < ny) ? index + nx : index;
    const Index zm = (z != 0) ? index - plane : index;
    const Index zp = (z + 1 < nz) ? index + plane : index;
    const double center = field[index];
    return field[xp] + field[xm] + field[yp] + field[ym]
         + field[zp] + field[zm] - 6.0 * center;
}

template <typename Index>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void initializeKernel(double* __restrict__ concentration, const Index count) {
    Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    while (index < count) {
        // The CPU expression uses size_t arithmetic. Widen before the
        // multiply even in the fast 32-bit-index specialization so non-power-
        // of-two grids produce precisely the same initial field.
        const unsigned long long wide_index = index;
        const unsigned long long wide_count = count;
        const unsigned long long value =
            ((wide_index + 1ULL) * 1299709ULL) % wide_count;
        concentration[index] = -1.0 + 2.0 * (static_cast<double>(value) /
                                             static_cast<double>(wide_count));
        if (count - index <= stride) {
            break;
        }
        index += stride;
    }
}

template <typename Index>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void chemicalPotentialKernel(const double* __restrict__ concentration,
                             double* __restrict__ chemical_potential,
                             const Index nx, const Index ny, const Index nz,
                             const Index plane) {
    const Index x = static_cast<Index>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const Index y = static_cast<Index>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const Index z = static_cast<Index>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const Index index = z * plane + y * nx + x;
    const double cv = concentration[index];
    const double lap = laplacian(concentration, index, x, y, z,
                                 nx, ny, nz, plane);

    // With the benchmark's fixed interaction energies, the bulk free-energy
    // derivative simplifies exactly to cv^3 - cv.
    chemical_potential[index] = cv * cv * cv - cv - 0.5 * lap;
}

template <typename Index>
__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void updateKernel(double* __restrict__ next,
                  const double* __restrict__ current,
                  const double* __restrict__ chemical_potential,
                  const Index nx, const Index ny, const Index nz,
                  const Index plane) {
    const Index x = static_cast<Index>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const Index y = static_cast<Index>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const Index z = static_cast<Index>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const Index index = z * plane + y * nx + x;
    next[index] = current[index]
        + 0.01 * laplacian(chemical_potential, index, x, y, z,
                           nx, ny, nz, plane);
}

// CUDA's Y/Z launch dimensions have lower architectural limits than X. These
// grid-stride variants preserve correctness for unusually long, skinny grids;
// normal 3-D domains use the division-free kernels above.
template <typename Index>
__global__ __launch_bounds__(256)
void chemicalPotentialLinearKernel(const double* __restrict__ concentration,
                                   double* __restrict__ chemical_potential,
                                   const Index nx, const Index ny,
                                   const Index nz, const Index plane,
                                   const Index count) {
    Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    while (index < count) {
        const Index yz = index / nx;
        const Index x = index - yz * nx;
        const Index z = yz / ny;
        const Index y = yz - z * ny;
        const double cv = concentration[index];
        const double lap = laplacian(concentration, index, x, y, z,
                                     nx, ny, nz, plane);
        chemical_potential[index] = cv * cv * cv - cv - 0.5 * lap;
        if (count - index <= stride) {
            break;
        }
        index += stride;
    }
}

template <typename Index>
__global__ __launch_bounds__(256)
void updateLinearKernel(double* __restrict__ next,
                        const double* __restrict__ current,
                        const double* __restrict__ chemical_potential,
                        const Index nx, const Index ny, const Index nz,
                        const Index plane, const Index count) {
    Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = static_cast<Index>(blockDim.x) * gridDim.x;
    while (index < count) {
        const Index yz = index / nx;
        const Index x = index - yz * nx;
        const Index z = yz / ny;
        const Index y = yz - z * ny;
        next[index] = current[index]
            + 0.01 * laplacian(chemical_potential, index, x, y, z,
                               nx, ny, nz, plane);
        if (count - index <= stride) {
            break;
        }
        index += stride;
    }
}

template <typename Index>
dim3 makeGrid(const Index nx, const Index ny, const Index nz,
              const cudaDeviceProp& properties) {
    const std::uint64_t blocks_x =
        (static_cast<std::uint64_t>(nx) + BLOCK_X - 1) / BLOCK_X;
    const std::uint64_t blocks_y =
        (static_cast<std::uint64_t>(ny) + BLOCK_Y - 1) / BLOCK_Y;
    const std::uint64_t blocks_z =
        (static_cast<std::uint64_t>(nz) + BLOCK_Z - 1) / BLOCK_Z;
    return dim3(static_cast<unsigned int>(std::min<std::uint64_t>(
                    blocks_x, properties.maxGridSize[0])),
                static_cast<unsigned int>(std::min<std::uint64_t>(
                    blocks_y, properties.maxGridSize[1])),
                static_cast<unsigned int>(std::min<std::uint64_t>(
                    blocks_z, properties.maxGridSize[2])));
}

template <typename Index>
void initializeDevice(double* concentration, const Index count,
                      const cudaDeviceProp& properties, cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    const std::uint64_t required_blocks =
        (static_cast<std::uint64_t>(count) + threads - 1) / threads;
    // A moderate grid gives every SM ample work while the grid-stride loop
    // avoids both launch-limit and very-large-domain special cases.
    const std::uint64_t useful_blocks =
        static_cast<std::uint64_t>(properties.multiProcessorCount) * 16;
    const unsigned int blocks = static_cast<unsigned int>(std::min(
        required_blocks, std::min<std::uint64_t>(useful_blocks,
                                                 properties.maxGridSize[0])));
    initializeKernel<Index><<<blocks, threads, 0, stream>>>(concentration, count);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Index>
double runSimulation(double*& current, double*& next,
                     double* chemical_potential, const Index nx,
                     const Index ny, const Index nz, const int iterations,
                     const cudaDeviceProp& properties, cudaStream_t stream) {
    if (iterations == 0) {
        return 0.0;
    }

    const Index plane = nx * ny;
    const Index count = plane * nz;
    const std::uint64_t blocks_x =
        (static_cast<std::uint64_t>(nx) + BLOCK_X - 1) / BLOCK_X;
    const std::uint64_t blocks_y =
        (static_cast<std::uint64_t>(ny) + BLOCK_Y - 1) / BLOCK_Y;
    const std::uint64_t blocks_z =
        (static_cast<std::uint64_t>(nz) + BLOCK_Z - 1) / BLOCK_Z;
    const bool regular_grid =
        blocks_x <= static_cast<std::uint64_t>(properties.maxGridSize[0])
        && blocks_y <= static_cast<std::uint64_t>(properties.maxGridSize[1])
        && blocks_z <= static_cast<std::uint64_t>(properties.maxGridSize[2]);
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid = makeGrid(nx, ny, nz, properties);
    constexpr unsigned int linear_threads = 256;
    const std::uint64_t required_linear_blocks =
        (static_cast<std::uint64_t>(count) + linear_threads - 1)
        / linear_threads;
    const unsigned int linear_blocks = static_cast<unsigned int>(std::min(
        required_linear_blocks,
        std::min<std::uint64_t>(
            static_cast<std::uint64_t>(properties.multiProcessorCount) * 16,
            properties.maxGridSize[0])));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable_graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int step = 0; step < iterations; ++step) {
        if (regular_grid) {
            chemicalPotentialKernel<Index><<<grid, block, 0, stream>>>(
                current, chemical_potential, nx, ny, nz, plane);
            updateKernel<Index><<<grid, block, 0, stream>>>(
                next, current, chemical_potential, nx, ny, nz, plane);
        } else {
            chemicalPotentialLinearKernel<Index>
                <<<linear_blocks, linear_threads, 0, stream>>>(
                    current, chemical_potential, nx, ny, nz, plane, count);
            updateLinearKernel<Index>
                <<<linear_blocks, linear_threads, 0, stream>>>(
                    next, current, chemical_potential, nx, ny, nz, plane,
                    count);
        }
        std::swap(current, next);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable_graph, graph, nullptr, nullptr, 0));

    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaGraphLaunch(executable_graph, stream));
    CUDA_CHECK(cudaEventRecord(finish, stream));
    CUDA_CHECK(cudaEventSynchronize(finish));

    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
    CUDA_CHECK(cudaEventDestroy(finish));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaGraphExecDestroy(executable_graph));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return static_cast<double>(milliseconds);
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto bounds = std::minmax_element(concentration.begin(),
                                            concentration.end());
    const double min_value = *bounds.first;
    const double max_value = *bounds.second;
    std::printf("Concentration range: [%.6f, %.6f]\n",
                min_value, max_value);
    if (max_value > 10.0 || min_value < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
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
    std::size_t nx = 64;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool print_results_requested = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_requested = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be positive and iterations must be non-negative.\n");
        return EXIT_FAILURE;
    }
    if (nx > std::numeric_limits<std::size_t>::max() / ny
        || nx * ny > std::numeric_limits<std::size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions overflow the address space.\n");
        return EXIT_FAILURE;
    }
    const std::size_t grid_size = nx * ny * nz;
    if (grid_size > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid allocation size overflows the address space.\n");
        return EXIT_FAILURE;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    double* device_current = nullptr;
    double* device_next = nullptr;
    double* device_mu = nullptr;
    const std::size_t bytes = grid_size * sizeof(double);
    CUDA_CHECK(cudaMalloc(&device_current, bytes));
    CUDA_CHECK(cudaMalloc(&device_next, bytes));
    CUDA_CHECK(cudaMalloc(&device_mu, bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::printf("Initializing concentration field...\n");
    const bool use_32_bit_indices =
        grid_size <= std::numeric_limits<std::uint32_t>::max()
        && nx <= std::numeric_limits<std::uint32_t>::max()
        && ny <= std::numeric_limits<std::uint32_t>::max()
        && nz <= std::numeric_limits<std::uint32_t>::max();
    if (use_32_bit_indices) {
        initializeDevice(device_current, static_cast<std::uint32_t>(grid_size),
                         properties, stream);
    } else {
        initializeDevice(device_current, grid_size, properties, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("Running Cahn-Hilliard simulation...\n");
    double elapsed_ms = 0.0;
    if (use_32_bit_indices) {
        elapsed_ms = runSimulation(device_current, device_next, device_mu,
                                   static_cast<std::uint32_t>(nx),
                                   static_cast<std::uint32_t>(ny),
                                   static_cast<std::uint32_t>(nz), iterations,
                                   properties, stream);
    } else {
        elapsed_ms = runSimulation(device_current, device_next, device_mu,
                                   nx, ny, nz, iterations, properties, stream);
    }

    std::printf("Computation time: %.3f ms\n", elapsed_ms);
    const double cell_updates = static_cast<double>(grid_size) * iterations;
    const double mcups = elapsed_ms > 0.0
        ? cell_updates / (elapsed_ms * 1000.0)
        : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> concentration;
    if (print_results_requested || validate) {
        concentration.resize(grid_size);
        CUDA_CHECK(cudaMemcpyAsync(concentration.data(), device_current, bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (print_results_requested) {
        print_results(concentration, "Concentration");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(concentration);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_mu));
    CUDA_CHECK(cudaFree(device_next));
    CUDA_CHECK(cudaFree(device_current));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
