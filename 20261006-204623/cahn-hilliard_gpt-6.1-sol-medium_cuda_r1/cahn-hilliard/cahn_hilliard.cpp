#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Fixed benchmark physical parameters, visible to the device compiler.
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gradientEnergy = 0.5;
constexpr double D = 1.0;
    
// Fail explicitly on CUDA errors: simulation always runs on the GPU.
void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each warp traverses contiguous x coordinates. Ordinary domains use direct
// 3D indexing; oversized dimensions use bounded, strided row tiles.
__device__ __forceinline__ double laplacian(
    const double* __restrict__ field, size_t i, size_t x, size_t y, size_t z,
    size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[i];
    const double cxx = (field[x + 1 < nx ? i + 1 : i] +
                        field[x > 0 ? i - 1 : i] - 2.0 * center) / (dx * dx);
    const double cyy = (field[y + 1 < ny ? i + nx : i] +
                        field[y > 0 ? i - nx : i] - 2.0 * center) / (dy * dy);
    const double czz = (field[z + 1 < nz ? i + plane : i] +
                        field[z > 0 ? i - plane : i] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += size_t(blockDim.x) * gridDim.x) {
        const double pseudo = (((i + 1) * size_t(1299709)) % vol) /
                              static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

// Template instantiations keep both passes specialized while sharing indexing.
// The chemical-potential field is completed before any update kernel starts.
template <bool ChemicalPotential, bool FlattenedRows>
__global__ void stencil(const double* __restrict__ cold,
                        const double* __restrict__ field,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t tilesY = (ny + blockDim.y - 1) / blockDim.y;
    const size_t tileCount = FlattenedRows ? tilesY * nz : gridDim.y;
    for (size_t tile = blockIdx.y; tile < tileCount; tile += gridDim.y) {
        size_t z, y;
        if constexpr (FlattenedRows) {
            z = tile / tilesY;
            y = (tile - z * tilesY) * blockDim.y + threadIdx.y;
        } else {
            z = blockIdx.z;
            y = tile * blockDim.y + threadIdx.y;
        }
        if (y >= ny) continue;
        const size_t i = (z * ny + y) * nx + x;
        const double lap = laplacian(field, i, x, y, z, nx, ny, nz);
        if constexpr (ChemicalPotential) {
            const double cv = field[i];
            output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                               2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gradientEnergy * lap;
        } else {
            output[i] = cold[i] + dt * D * lap;
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
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

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Invalid or overflowing grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    cudaDeviceProp properties;
    checkCuda(cudaGetDeviceProperties(&properties, 0));
    checkCuda(cudaSetDevice(0));
    // Adapt the x tile to narrow domains while retaining 128 threads per block.
    const unsigned tileX = nx >= 32 ? 32 : nx >= 16 ? 16 : nx >= 8 ? 8 :
                           nx >= 4 ? 4 : nx >= 2 ? 2 : 1;
    const dim3 block(tileX, 128 / tileX);
    const size_t blocksX = (nx + block.x - 1) / block.x;
    if (blocksX > static_cast<size_t>(properties.maxGridSize[0])) {
        fprintf(stderr, "Grid X dimension exceeds CUDA launch limits\n");
        return 1;
    }
    const size_t tilesY = (ny + block.y - 1) / block.y;
    const bool flattenedRows = tilesY > static_cast<size_t>(properties.maxGridSize[1]) ||
                               nz > static_cast<size_t>(properties.maxGridSize[2]);
    const dim3 grid(static_cast<unsigned>(blocksX),
        static_cast<unsigned>(flattenedRows ? std::min(tilesY * nz,
            static_cast<size_t>(properties.maxGridSize[1])) : tilesY),
        flattenedRows ? 1 : static_cast<unsigned>(nz));
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, bytes));
    checkCuda(cudaMalloc(&cnew, bytes));
    checkCuda(cudaMalloc(&mu, bytes));
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    printf("Initializing concentration field...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min(
        (gridSize + 255) / 256, size_t(properties.multiProcessorCount) * 32));
    initializeConcentration<<<initBlocks, 256, 0, stream>>>(cold, gridSize);
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));

    auto step = [&](double* input, double* output) {
        if (flattenedRows) {
            stencil<true, true><<<grid, block, 0, stream>>>(input, input, mu, nx, ny, nz);
            stencil<false, true><<<grid, block, 0, stream>>>(input, mu, output, nx, ny, nz);
        } else {
            stencil<true, false><<<grid, block, 0, stream>>>(input, input, mu, nx, ny, nz);
            stencil<false, false><<<grid, block, 0, stream>>>(input, mu, output, nx, ny, nz);
        }
    };
    // A two-step graph restores buffer parity and can be replayed without
    // modifying node parameters or building a graph proportional to iterations.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executableGraph = nullptr;
    if (iterations >= 2) {
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        step(cold, cnew);
        step(cnew, cold);
        checkCuda(cudaStreamEndCapture(stream, &graph));
        checkCuda(cudaGraphInstantiate(&executableGraph, graph, nullptr, nullptr, 0));
        checkCuda(cudaGraphUpload(executableGraph, stream));
        checkCuda(cudaStreamSynchronize(stream));
    }

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations / 2; ++t) {
        checkCuda(cudaGraphLaunch(executableGraph, stream));
    }
    if (iterations > 0 && iterations % 2 != 0) {
        step(cold, cnew);
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = seconds > 0 ? cellUpdates / seconds / 1e6 : 0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfer just the final field, and only when the CLI requests host output.
    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost));
    }
    if (executableGraph) checkCuda(cudaGraphExecDestroy(executableGraph));
    if (graph) checkCuda(cudaGraphDestroy(graph));
    checkCuda(cudaFree(mu));
    checkCuda(cudaFree(cnew));
    checkCuda(cudaFree(cold));
    checkCuda(cudaStreamDestroy(stream));

    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
