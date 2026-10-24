#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

constexpr uint64_t HASH_MULT_ENERGY = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t HASH_MULT_FLUX = 0xbf58476d1ce4e5b9ULL;

constexpr int TAG_X_TOP = 1001;
constexpr int TAG_X_BOTTOM = 1002;
constexpr int TAG_Y_LEFT = 1003;
constexpr int TAG_Y_RIGHT = 1004;

constexpr int CUDA_THREADS = 256;

static void checkMpi(int rc, const char* what) {
    if (rc != MPI_SUCCESS) {
        char err[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(rc, err, &len);
        fprintf(stderr, "MPI error in %s: %s\n", what, err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void checkCuda(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(rc));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

struct LocalMesh {
    MPI_Comm cart = MPI_COMM_NULL;
    int N = 0;
    int xs = 0, ys = 0;
    int lx = 0, ly = 0;
    int stride = 0;
    int total = 0;
    int xm = MPI_PROC_NULL, xp = MPI_PROC_NULL;
    int ym = MPI_PROC_NULL, yp = MPI_PROC_NULL;

    double* d_energy = nullptr;
    double* d_energy_swap = nullptr;
    double* d_flux = nullptr;
    double* d_flux_swap = nullptr;
    double* d_columns = nullptr;
    int* d_material = nullptr;
    Material* d_materials = nullptr;

    double* h_send = nullptr;
    double* h_recv = nullptr;

    cudaStream_t stream = nullptr;
};

__global__ void computeFluxKernel(const double* __restrict__ energy_in,
                                  const double* __restrict__ flux_in,
                                  double* __restrict__ energy_out,
                                  double* __restrict__ flux_out,
                                  const int* __restrict__ material_idx,
                                  const Material* __restrict__ materials,
                                  int local_nx, int local_ny, int stride,
                                  int xs, int ys, int N) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = local_nx * local_ny;
    if (t >= total) {
        return;
    }

    const int lx = t / local_ny;
    const int ly = t - lx * local_ny;
    const int L = lx + 1;
    const int C = ly + 1;
    const int gx = xs + lx;
    const int gy = ys + ly;
    const int li = L * stride + C;

    const Material mat = materials[material_idx[t]];
    const double transfer_coeff = mat.transfer_coeff;
    const double e = energy_in[li];

    double total_flux = mat.external_flow;

    if (gx + 1 < N) {
        total_flux += (energy_in[li + stride] - e) * transfer_coeff * 1.0 * 0.25;
    }
    if (gx - 1 >= 0) {
        total_flux += (energy_in[li - stride] - e) * transfer_coeff * 1.0 * 0.25;
    }
    if (gy + 1 < N) {
        total_flux += (energy_in[li + 1] - e) * transfer_coeff * 1.0 * 0.25;
    }
    if (gy - 1 >= 0) {
        total_flux += (energy_in[li - 1] - e) * transfer_coeff * 1.0 * 0.25;
    }

    energy_out[li] = e + total_flux;
    flux_out[li] = flux_in[li] + fabs(total_flux);
}

__global__ void packColumnsKernel(const double* __restrict__ energy,
                                  double* __restrict__ columns,
                                  int lx, int ly, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lx) {
        return;
    }
    const size_t row = static_cast<size_t>(i + 1) * stride;
    columns[i] = energy[row + 1];
    columns[lx + i] = energy[row + ly];
}

__global__ void unpackColumnsKernel(double* __restrict__ energy,
                                    const double* __restrict__ columns,
                                    int lx, int ly, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lx) {
        return;
    }
    const size_t row = static_cast<size_t>(i + 1) * stride;
    energy[row] = columns[i];
    energy[row + ly + 1] = columns[lx + i];
}

static void exchangeHalos(LocalMesh& mesh, double* energy, cudaStream_t stream) {
    const int lx = mesh.lx;
    const int ly = mesh.ly;
    if (lx == 0 || ly == 0) {
        return;
    }

    const bool have_x = (mesh.xm != MPI_PROC_NULL) || (mesh.xp != MPI_PROC_NULL);
    const bool have_y = (mesh.ym != MPI_PROC_NULL) || (mesh.yp != MPI_PROC_NULL);
    if (!have_x && !have_y) {
        return;
    }

    const size_t stride = static_cast<size_t>(mesh.stride);
    const size_t elem = sizeof(double);
    double* send = mesh.h_send;
    double* recv = mesh.h_recv;

    const size_t send_top = 0;
    const size_t send_bottom = send_top + ly;
    const size_t send_left = send_bottom + ly;
    const size_t send_right = send_left + lx;

    const size_t recv_top = 0;
    const size_t recv_bottom = recv_top + ly;
    const size_t recv_left = recv_bottom + ly;
    const size_t recv_right = recv_left + lx;

    const int column_blocks = (lx + CUDA_THREADS - 1) / CUDA_THREADS;

    if (have_x) {
        checkCuda(cudaMemcpyAsync(send + send_top,
                                  energy + stride + 1,
                                  static_cast<size_t>(ly) * elem,
                                  cudaMemcpyDeviceToHost, stream),
                  "copy top row D2H");
        checkCuda(cudaMemcpyAsync(send + send_bottom,
                                  energy + static_cast<size_t>(lx) * stride + 1,
                                  static_cast<size_t>(ly) * elem,
                                  cudaMemcpyDeviceToHost, stream),
                  "copy bottom row D2H");
    }
    if (have_y) {
        packColumnsKernel<<<column_blocks, CUDA_THREADS, 0, stream>>>(
            energy, mesh.d_columns, lx, ly, mesh.stride);
        checkCuda(cudaGetLastError(), "packColumnsKernel launch");
        checkCuda(cudaMemcpyAsync(send + send_left, mesh.d_columns,
                                  static_cast<size_t>(2 * lx) * elem,
                                  cudaMemcpyDeviceToHost, stream),
                  "copy columns D2H");
    }
    checkCuda(cudaStreamSynchronize(stream), "sync boundary D2H");

    if (have_x) {
        checkMpi(MPI_Sendrecv(send + send_top, ly, MPI_DOUBLE, mesh.xm, TAG_X_TOP,
                              recv + recv_bottom, ly, MPI_DOUBLE, mesh.xp, TAG_X_TOP,
                              mesh.cart, MPI_STATUS_IGNORE),
                 "MPI_Sendrecv x top");
        checkMpi(MPI_Sendrecv(send + send_bottom, ly, MPI_DOUBLE, mesh.xp, TAG_X_BOTTOM,
                              recv + recv_top, ly, MPI_DOUBLE, mesh.xm, TAG_X_BOTTOM,
                              mesh.cart, MPI_STATUS_IGNORE),
                 "MPI_Sendrecv x bottom");
    }
    if (have_y) {
        checkMpi(MPI_Sendrecv(send + send_left, lx, MPI_DOUBLE, mesh.ym, TAG_Y_LEFT,
                              recv + recv_right, lx, MPI_DOUBLE, mesh.yp, TAG_Y_LEFT,
                              mesh.cart, MPI_STATUS_IGNORE),
                 "MPI_Sendrecv y left");
        checkMpi(MPI_Sendrecv(send + send_right, lx, MPI_DOUBLE, mesh.yp, TAG_Y_RIGHT,
                              recv + recv_left, lx, MPI_DOUBLE, mesh.ym, TAG_Y_RIGHT,
                              mesh.cart, MPI_STATUS_IGNORE),
                 "MPI_Sendrecv y right");
    }

    if (have_x) {
        checkCuda(cudaMemcpyAsync(energy + 1,
                                  recv + recv_top,
                                  static_cast<size_t>(ly) * elem,
                                  cudaMemcpyHostToDevice, stream),
                  "copy top ghost H2D");
        checkCuda(cudaMemcpyAsync(energy + static_cast<size_t>(lx + 1) * stride + 1,
                                  recv + recv_bottom,
                                  static_cast<size_t>(ly) * elem,
                                  cudaMemcpyHostToDevice, stream),
                  "copy bottom ghost H2D");
    }
    if (have_y) {
        checkCuda(cudaMemcpyAsync(mesh.d_columns, recv + recv_left,
                                  static_cast<size_t>(2 * lx) * elem,
                                  cudaMemcpyHostToDevice, stream),
                  "copy columns H2D");
        unpackColumnsKernel<<<column_blocks, CUDA_THREADS, 0, stream>>>(
            energy, mesh.d_columns, lx, ly, mesh.stride);
        checkCuda(cudaGetLastError(), "unpackColumnsKernel launch");
    }
}

// Run simulation for n_iters iterations
static void runSimulation(LocalMesh& mesh, const int n_iters) {
    double* energy = mesh.d_energy;
    double* energy_swap = mesh.d_energy_swap;
    double* flux = mesh.d_flux;
    double* flux_swap = mesh.d_flux_swap;

    const int blocks = (mesh.total + CUDA_THREADS - 1) / CUDA_THREADS;

    for (int iter = 0; iter < n_iters; ++iter) {
        if (mesh.total > 0) {
            computeFluxKernel<<<blocks, CUDA_THREADS, 0, mesh.stream>>>(
                energy, flux, energy_swap, flux_swap,
                mesh.d_material, mesh.d_materials,
                mesh.lx, mesh.ly, mesh.stride, mesh.xs, mesh.ys, mesh.N);
            checkCuda(cudaGetLastError(), "computeFluxKernel launch");
        }

        if (iter + 1 < n_iters) {
            exchangeHalos(mesh, energy_swap, mesh.stream);
        }

        std::swap(energy, energy_swap);
        std::swap(flux, flux_swap);
    }

    checkCuda(cudaStreamSynchronize(mesh.stream), "sync simulation");

    mesh.d_energy = energy;
    mesh.d_energy_swap = energy_swap;
    mesh.d_flux = flux;
    mesh.d_flux_swap = flux_swap;
}

// Validate simulation results
static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    // Check for numerical issues
    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        // Don't fail validation as this can happen with external flows
    }

    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

static void chooseProcessGrid(int world_size, int N, int dims[2]) {
    if (world_size <= N && world_size <= 8) {
        dims[0] = world_size;
        dims[1] = 1;
        return;
    }

    int best_px = 1;
    int best_py = 1;
    int best_prod = 0;
    const int max_px = std::min(world_size, N);
    for (int px = 1; px <= max_px; ++px) {
        const int py = std::min(N, world_size / px);
        if (py < 1) {
            continue;
        }
        const int prod = px * py;
        const int diff = std::abs(px - py);
        const int best_diff = std::abs(best_px - best_py);
        const bool better =
            (prod > best_prod) ||
            (prod == best_prod && diff < best_diff) ||
            (prod == best_prod && diff == best_diff && px > best_px);
        if (better) {
            best_px = px;
            best_py = py;
            best_prod = prod;
        }
    }
    dims[0] = best_px;
    dims[1] = best_py;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    checkMpi(MPI_Init(&argc, &argv), "MPI_Init");

    int world_rank = 0;
    int world_size = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size), "MPI_Comm_size");

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (n_elems_root <= 0) {
        if (world_rank == 0) {
            printf("Grid size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int N = n_elems_root;
    const long long n_elems_ll = static_cast<long long>(N) * N;
    const size_t n_elems = static_cast<size_t>(n_elems_ll);

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm node_comm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                 MPI_INFO_NULL, &node_comm),
             "MPI_Comm_split_type");
    int local_rank = 0;
    int local_size = 1;
    checkMpi(MPI_Comm_rank(node_comm, &local_rank), "MPI_Comm_rank(node)");
    checkMpi(MPI_Comm_size(node_comm, &local_size), "MPI_Comm_size(node)");
    const int device_id = local_rank % device_count;
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
    checkMpi(MPI_Comm_free(&node_comm), "MPI_Comm_free(node)");

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_max_threads() / local_size));
    }

    int dims[2] = {1, 1};
    chooseProcessGrid(world_size, N, dims);
    const int active_ranks = dims[0] * dims[1];

    LocalMesh mesh;
    mesh.N = N;

    MPI_Comm active_comm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split(MPI_COMM_WORLD,
                            world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                            world_rank, &active_comm),
             "MPI_Comm_split(active)");

    if (active_comm != MPI_COMM_NULL) {
        int periods[2] = {0, 0};
        checkMpi(MPI_Cart_create(active_comm, 2, dims, periods, 0, &mesh.cart),
                 "MPI_Cart_create");
        checkMpi(MPI_Comm_free(&active_comm), "MPI_Comm_free(active)");

        int cart_rank = 0;
        checkMpi(MPI_Comm_rank(mesh.cart, &cart_rank), "MPI_Comm_rank(cart)");
        int coords[2] = {0, 0};
        checkMpi(MPI_Cart_coords(mesh.cart, cart_rank, 2, coords),
                 "MPI_Cart_coords");

        checkMpi(MPI_Cart_shift(mesh.cart, 0, 1, &mesh.xm, &mesh.xp),
                 "MPI_Cart_shift x");
        checkMpi(MPI_Cart_shift(mesh.cart, 1, 1, &mesh.ym, &mesh.yp),
                 "MPI_Cart_shift y");

        mesh.xs = static_cast<int>((static_cast<long long>(coords[0]) * N) / dims[0]);
        const int xe =
            static_cast<int>((static_cast<long long>(coords[0] + 1) * N) / dims[0]);
        mesh.ys = static_cast<int>((static_cast<long long>(coords[1]) * N) / dims[1]);
        const int ye =
            static_cast<int>((static_cast<long long>(coords[1] + 1) * N) / dims[1]);
        mesh.lx = xe - mesh.xs;
        mesh.ly = ye - mesh.ys;
    }

    mesh.stride = mesh.ly + 2;
    mesh.total = mesh.lx * mesh.ly;

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", N, N, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");

        const size_t static_mem = n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    std::vector<Material> materials;
    materials.emplace_back(Material{0.8, 0.0});    // Default material
    materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    std::vector<int> local_material(static_cast<size_t>(mesh.total));
#pragma omp parallel for collapse(2) schedule(static)
    for (int a = 0; a < mesh.lx; ++a) {
        for (int b = 0; b < mesh.ly; ++b) {
            const int gx = mesh.xs + a;
            const int gy = mesh.ys + b;
            int mat = static_cast<int>(DEFAULT_MAT_ID);
            if ((gx == 0 && gy == 0) || (gx == N - 1 && gy == N - 1)) {
                mat = static_cast<int>(INFLOW_MAT_ID);
            } else if ((gx == 0 && gy == N - 1) ||
                       (gx == N - 1 && gy == 0)) {
                mat = static_cast<int>(OUTFLOW_MAT_ID);
            }
            local_material[static_cast<size_t>(a) * mesh.ly + b] = mat;
        }
    }

    const size_t local_total =
        static_cast<size_t>(mesh.lx + 2) * static_cast<size_t>(mesh.stride);
    const size_t local_bytes = std::max<size_t>(local_total, 1) * sizeof(double);
    const size_t local_mat_bytes =
        std::max<size_t>(static_cast<size_t>(mesh.total), 1) * sizeof(int);

    checkCuda(cudaStreamCreateWithFlags(&mesh.stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_energy), local_bytes),
              "cudaMalloc energy");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_energy_swap), local_bytes),
              "cudaMalloc energy swap");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_flux), local_bytes),
              "cudaMalloc flux");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_flux_swap), local_bytes),
              "cudaMalloc flux swap");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_columns),
                         std::max<size_t>(2 * static_cast<size_t>(mesh.lx), 1) *
                             sizeof(double)),
              "cudaMalloc columns");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_material), local_mat_bytes),
              "cudaMalloc material");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&mesh.d_materials),
                         std::max<size_t>(materials.size(), 1) * sizeof(Material)),
              "cudaMalloc materials");

    checkCuda(cudaMemsetAsync(mesh.d_energy, 0, local_bytes, mesh.stream),
              "memset energy");
    checkCuda(cudaMemsetAsync(mesh.d_energy_swap, 0, local_bytes, mesh.stream),
              "memset energy swap");
    checkCuda(cudaMemsetAsync(mesh.d_flux, 0, local_bytes, mesh.stream),
              "memset flux");
    checkCuda(cudaMemsetAsync(mesh.d_flux_swap, 0, local_bytes, mesh.stream),
              "memset flux swap");
    if (mesh.total > 0) {
        checkCuda(cudaMemcpyAsync(mesh.d_material, local_material.data(),
                                  static_cast<size_t>(mesh.total) * sizeof(int),
                                  cudaMemcpyHostToDevice, mesh.stream),
                  "copy material ids");
    }
    checkCuda(cudaMemcpyAsync(mesh.d_materials, materials.data(),
                              materials.size() * sizeof(Material),
                              cudaMemcpyHostToDevice, mesh.stream),
              "copy materials");

    const size_t buf_count =
        std::max<size_t>(2 * (static_cast<size_t>(mesh.lx) + mesh.ly), 1);
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&mesh.h_send),
                            buf_count * sizeof(double), cudaHostAllocDefault),
              "cudaHostAlloc send");
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&mesh.h_recv),
                            buf_count * sizeof(double), cudaHostAllocDefault),
              "cudaHostAlloc recv");
    std::memset(mesh.h_send, 0, buf_count * sizeof(double));
    std::memset(mesh.h_recv, 0, buf_count * sizeof(double));
    checkCuda(cudaStreamSynchronize(mesh.stream), "sync initialization");

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(mesh, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double local_duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double duration_ms = 0.0;
    checkMpi(MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX,
                        0, MPI_COMM_WORLD),
             "MPI_Reduce timing");

    std::vector<double> host_energy(static_cast<size_t>(mesh.total));
    std::vector<double> host_flux(static_cast<size_t>(mesh.total));
    if (mesh.total > 0) {
        const size_t row_bytes = static_cast<size_t>(mesh.ly) * sizeof(double);
        checkCuda(cudaMemcpy2DAsync(host_energy.data(), row_bytes,
                                    mesh.d_energy + mesh.stride + 1,
                                    static_cast<size_t>(mesh.stride) * sizeof(double),
                                    row_bytes, mesh.lx,
                                    cudaMemcpyDeviceToHost, mesh.stream),
                  "copy owned energy D2H");
        checkCuda(cudaMemcpy2DAsync(host_flux.data(), row_bytes,
                                    mesh.d_flux + mesh.stride + 1,
                                    static_cast<size_t>(mesh.stride) * sizeof(double),
                                    row_bytes, mesh.lx,
                                    cudaMemcpyDeviceToHost, mesh.stream),
                  "copy owned flux D2H");
        checkCuda(cudaStreamSynchronize(mesh.stream), "sync owned data D2H");
    }

    uint64_t local_hash = 0;
#pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (int k = 0; k < mesh.total; ++k) {
        const int a = k / mesh.ly;
        const int b = k - a * mesh.ly;
        const uint64_t i =
            static_cast<uint64_t>(mesh.xs + a) * static_cast<uint64_t>(N) +
            static_cast<uint64_t>(mesh.ys + b);
        uint64_t e_bits = 0;
        uint64_t f_bits = 0;
        std::memcpy(&e_bits, &host_energy[static_cast<size_t>(k)], sizeof(e_bits));
        std::memcpy(&f_bits, &host_flux[static_cast<size_t>(k)], sizeof(f_bits));
        local_hash ^= (e_bits + i) * HASH_MULT_ENERGY;
        local_hash ^= (f_bits + i) * HASH_MULT_FLUX;
    }

    uint64_t global_hash = 0;
    checkMpi(MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
                        0, MPI_COMM_WORLD),
             "MPI_Reduce hash");

    int exit_code = 0;

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration_ms));

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) /
            (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    if (printResults || validate) {
        int meta[4] = {mesh.xs, mesh.ys, mesh.lx, mesh.ly};
        std::vector<int> all_meta;
        if (world_rank == 0) {
            all_meta.resize(static_cast<size_t>(world_size) * 4);
        }
        checkMpi(MPI_Gather(meta, 4, MPI_INT,
                            world_rank == 0 ? all_meta.data() : nullptr,
                            4, MPI_INT, 0, MPI_COMM_WORLD),
                 "MPI_Gather meta");

        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> recv_energy;
        std::vector<double> recv_flux;
        int gathered_total = 0;
        if (world_rank == 0) {
            counts.resize(static_cast<size_t>(world_size));
            displs.resize(static_cast<size_t>(world_size));
            int offset = 0;
            for (int r = 0; r < world_size; ++r) {
                const int c = all_meta[static_cast<size_t>(r) * 4 + 2] *
                              all_meta[static_cast<size_t>(r) * 4 + 3];
                counts[static_cast<size_t>(r)] = c;
                displs[static_cast<size_t>(r)] = offset;
                offset += c;
            }
            gathered_total = offset;
            recv_energy.resize(static_cast<size_t>(gathered_total));
            recv_flux.resize(static_cast<size_t>(gathered_total));
        }

        checkMpi(MPI_Gatherv(host_energy.data(), mesh.total, MPI_DOUBLE,
                             world_rank == 0 ? recv_energy.data() : nullptr,
                             world_rank == 0 ? counts.data() : nullptr,
                             world_rank == 0 ? displs.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv energy");
        checkMpi(MPI_Gatherv(host_flux.data(), mesh.total, MPI_DOUBLE,
                             world_rank == 0 ? recv_flux.data() : nullptr,
                             world_rank == 0 ? counts.data() : nullptr,
                             world_rank == 0 ? displs.data() : nullptr,
                             MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv flux");

        if (world_rank == 0) {
            std::vector<ElementDynamic> global(n_elems);
#pragma omp parallel for schedule(static)
            for (int r = 0; r < world_size; ++r) {
                const int rxs = all_meta[static_cast<size_t>(r) * 4 + 0];
                const int rys = all_meta[static_cast<size_t>(r) * 4 + 1];
                const int rlx = all_meta[static_cast<size_t>(r) * 4 + 2];
                const int rly = all_meta[static_cast<size_t>(r) * 4 + 3];
                const double* e = recv_energy.data() + displs[static_cast<size_t>(r)];
                const double* f = recv_flux.data() + displs[static_cast<size_t>(r)];
                for (int a = 0; a < rlx; ++a) {
                    const size_t base =
                        static_cast<size_t>(rxs + a) * static_cast<size_t>(N) +
                        static_cast<size_t>(rys);
                    const double* er = e + static_cast<size_t>(a) * rly;
                    const double* fr = f + static_cast<size_t>(a) * rly;
                    for (int b = 0; b < rly; ++b) {
                        global[base + static_cast<size_t>(b)] =
                            ElementDynamic{er[b], fr[b]};
                    }
                }
            }

            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(global.size());
                for (const auto& elem : global) {
                    energyData.push_back(elem.current_energy);
                }
                print_results(energyData, "ElementEnergy");
            }

            if (validate) {
                if (!validateResults(global)) {
                    exit_code = 1;
                }
            }
        }
    }

    checkCuda(cudaFreeHost(mesh.h_send), "cudaFreeHost send");
    checkCuda(cudaFreeHost(mesh.h_recv), "cudaFreeHost recv");
    checkCuda(cudaFree(mesh.d_materials), "cudaFree materials");
    checkCuda(cudaFree(mesh.d_material), "cudaFree material");
    checkCuda(cudaFree(mesh.d_columns), "cudaFree columns");
    checkCuda(cudaFree(mesh.d_flux_swap), "cudaFree flux swap");
    checkCuda(cudaFree(mesh.d_flux), "cudaFree flux");
    checkCuda(cudaFree(mesh.d_energy_swap), "cudaFree energy swap");
    checkCuda(cudaFree(mesh.d_energy), "cudaFree energy");
    checkCuda(cudaStreamDestroy(mesh.stream), "cudaStreamDestroy");
    if (mesh.cart != MPI_COMM_NULL) {
        checkMpi(MPI_Comm_free(&mesh.cart), "MPI_Comm_free(cart)");
    }

    MPI_Finalize();
    return exit_code;
}
