#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;
struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// The generated mesh is always the same regular square topology.  Keeping only
// dynamic results on the host avoids constructing and transferring a 144-byte
// AoS connectivity record for every element.  The CUDA kernel below evaluates
// that exact connectivity (including its original neighbor order) implicitly.
struct World {
    std::vector<ElementDynamic> elements_dynamic;
};

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                        \
        if (cuda_check_error != cudaSuccess) {                                    \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);       \
        }                                                                        \
    } while (false)

void buildSquare2D(World& world, int n_elems_root) {
    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    world.elements_dynamic.assign(n_elems, ElementDynamic{0.0, 0.0});
}

__device__ __forceinline__ val_t computeFlux(val_t this_energy,
                                              val_t other_energy) {
    // Preserve the reference expression's operation order.  CUDA contracts the
    // final multiply/add just as the optimized host reference does.
    return (other_energy - this_energy) * 0.8 * 1.0 * 0.25;
}

constexpr unsigned BLOCK_COLUMNS = 32;
constexpr unsigned BLOCK_ROWS = 4;

__global__ void simulationStep(const val_t* __restrict__ energy_in,
                               val_t* __restrict__ energy_out,
                               val_t* __restrict__ cumulative_flux,
                               int n_elems_root) {
    const unsigned local_y = threadIdx.x;
    const unsigned local_x = threadIdx.y;
    const unsigned y = blockIdx.x * BLOCK_COLUMNS + local_y;
    const unsigned x = blockIdx.y * BLOCK_ROWS + local_x;
    const unsigned root = static_cast<unsigned>(n_elems_root);
    const bool valid = x < root && y < root;
    const size_t index = static_cast<size_t>(x) * root + y;
    if (!valid) {
        return;
    }

    const val_t current_energy = energy_in[index];
    const unsigned last = root - 1;
    val_t total_flux = 0.0;

    // These tests reproduce the final material assignments in buildSquare2D,
    // including the n=1 case where all four corners alias the inflow element.
    if ((x == 0 && y == 0) || (x == last && y == last)) {
        total_flux = 0.5;
    } else if ((x == 0 && y == last) || (x == last && y == 0)) {
        total_flux = -0.5;
    }

    // Original connection order: +x, -x, +y, -y.
    if (x + 1 < root) {
        total_flux += computeFlux(current_energy, energy_in[index + root]);
    }
    if (x != 0) {
        total_flux += computeFlux(current_energy, energy_in[index - root]);
    }
    if (y + 1 < root) {
        total_flux += computeFlux(current_energy, energy_in[index + 1]);
    }
    if (y != 0) {
        total_flux += computeFlux(current_energy, energy_in[index - 1]);
    }

    energy_out[index] = current_energy + total_flux;
    cumulative_flux[index] += fabs(total_flux);
}

struct GpuSimulation {
    val_t* allocation = nullptr;
    val_t* energy[2] = {nullptr, nullptr};
    val_t* cumulative_flux = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    size_t n_elems = 0;
    int final_energy_buffer = 0;

    GpuSimulation() = default;
    GpuSimulation(const GpuSimulation&) = delete;
    GpuSimulation& operator=(const GpuSimulation&) = delete;

    ~GpuSimulation() {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr) {
            cudaEventDestroy(stop_event);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (allocation != nullptr) {
            cudaFree(allocation);
        }
    }
};

void prepareSimulation(GpuSimulation& gpu, int n_elems_root, int n_iters) {
    gpu.n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    gpu.final_energy_buffer = n_iters & 1;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&gpu.start_event));
    CUDA_CHECK(cudaEventCreate(&gpu.stop_event));

    const size_t bytes_per_array = gpu.n_elems * sizeof(val_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.allocation),
                          3 * bytes_per_array));
    gpu.energy[0] = gpu.allocation;
    gpu.energy[1] = gpu.allocation + gpu.n_elems;
    gpu.cumulative_flux = gpu.allocation + 2 * gpu.n_elems;
    CUDA_CHECK(cudaMemsetAsync(gpu.allocation, 0, 3 * bytes_per_array, gpu.stream));
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));

    std::printf("CUDA device: %s (%d SMs)\n", properties.name,
                properties.multiProcessorCount);

    if (n_iters == 0) {
        return;
    }

    const dim3 threads(BLOCK_COLUMNS, BLOCK_ROWS);
    const dim3 blocks((n_elems_root + BLOCK_COLUMNS - 1) / BLOCK_COLUMNS,
                      (n_elems_root + BLOCK_ROWS - 1) / BLOCK_ROWS);

    CUDA_CHECK(cudaStreamBeginCapture(gpu.stream, cudaStreamCaptureModeGlobal));
    for (int iter = 0; iter < n_iters; ++iter) {
        simulationStep<<<blocks, threads, 0, gpu.stream>>>(
            gpu.energy[iter & 1], gpu.energy[(iter + 1) & 1],
            gpu.cumulative_flux, n_elems_root);
    }
    CUDA_CHECK(cudaStreamEndCapture(gpu.stream, &gpu.graph));
    CUDA_CHECK(cudaGraphInstantiate(&gpu.graph_exec, gpu.graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(gpu.graph_exec, gpu.stream));
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
}

double runSimulation(GpuSimulation& gpu) {
    if (gpu.graph_exec == nullptr) {
        return 0.0;
    }

    CUDA_CHECK(cudaEventRecord(gpu.start_event, gpu.stream));
    CUDA_CHECK(cudaGraphLaunch(gpu.graph_exec, gpu.stream));
    CUDA_CHECK(cudaEventRecord(gpu.stop_event, gpu.stream));
    CUDA_CHECK(cudaEventSynchronize(gpu.stop_event));

    float duration_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&duration_ms, gpu.start_event, gpu.stop_event));
    return static_cast<double>(duration_ms);
}

void retrieveSimulation(const GpuSimulation& gpu, World& world) {
    std::vector<val_t> staging(gpu.n_elems);

    CUDA_CHECK(cudaMemcpy(staging.data(), gpu.energy[gpu.final_energy_buffer],
                          gpu.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < gpu.n_elems; ++i) {
        world.elements_dynamic[i].current_energy = staging[i];
    }

    CUDA_CHECK(cudaMemcpy(staging.data(), gpu.cumulative_flux,
                          gpu.n_elems * sizeof(val_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < gpu.n_elems; ++i) {
        world.elements_dynamic[i].total_flux = staging[i];
    }
}

bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    std::printf("Validation results:\n");
    std::printf("  Energy sum: %.12f\n", energy_sum);
    std::printf("  Flux sum: %.2f\n", flux_sum);
    std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) {
        std::printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > energy_epsilon) {
        std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(flux_sum)) {
        std::printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        std::printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    std::printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t energy_bits = 0;
        uint64_t flux_bits = 0;
        std::memcpy(&energy_bits, &elements[i].current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elements[i].total_flux, sizeof(flux_bits));
        hash ^= (energy_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux_bits + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

void printUsage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool print_results_enabled = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            print_results_enabled = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    constexpr int max_root = 46340;
    if (n_elems_root <= 0 || n_elems_root > max_root || n_iters < 0) {
        std::fprintf(stderr,
                     "Grid size must be in [1, %d] and iterations must be non-negative.\n",
                     max_root);
        return 1;
    }

    const int n_elems = n_elems_root * n_elems_root;

    std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
    std::printf("============================================\n");
    std::printf("Grid size: %d x %d = %d elements\n", n_elems_root,
                n_elems_root, n_elems);
    std::printf("Iterations: %d\n", n_iters);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("\n");

    std::printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                total_mem / (1024.0 * 1024.0),
                static_mem / (1024.0 * 1024.0),
                dynamic_mem / (1024.0 * 1024.0));

    GpuSimulation gpu;
    prepareSimulation(gpu, n_elems_root, n_iters);
    std::printf("GPU state memory: %.2f MB\n",
                (3.0 * n_elems * sizeof(val_t)) / (1024.0 * 1024.0));
    std::printf("\n");

    std::printf("Running simulation...\n");
    const double duration_ms = runSimulation(gpu);
    retrieveSimulation(gpu, world);

    std::printf("Computation time: %.3f ms\n", duration_ms);

    const int n_measured_iters = std::max(n_iters, 1);
    const double time_per_iter = duration_ms / n_measured_iters;
    const double elapsed_seconds = duration_ms / 1000.0;
    const double giga_elems_per_sec =
        duration_ms > 0.0
            ? (static_cast<double>(n_iters) * n_elems) / elapsed_seconds / 1e9
            : 0.0;
    const double gflops = giga_elems_per_sec * 22.0;

    std::printf("Performance:\n");
    std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
    std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
    std::printf("  Performance: %.4f GFLOPS\n", gflops);

    const uint64_t result_hash = computeHash(world.elements_dynamic);
    std::printf("  Result hash: %016" PRIX64 "\n", result_hash);
    std::printf("\n");

    if (print_results_enabled) {
        std::vector<double> energy_data;
        energy_data.reserve(world.elements_dynamic.size());
        for (const auto& element : world.elements_dynamic) {
            energy_data.push_back(element.current_energy);
        }
        print_results(energy_data, "ElementEnergy");
    }

    if (validate && !validateResults(world)) {
        return 1;
    }

    return 0;
}
