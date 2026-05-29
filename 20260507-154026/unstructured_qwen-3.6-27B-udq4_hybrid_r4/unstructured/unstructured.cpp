#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

__device__ inline val_t computeFlux(val_t transfer_coeff,
                                    val_t this_energy,
                                    val_t connection_flux,
                                    val_t other_energy) {
    return (other_energy - this_energy) * transfer_coeff * connection_flux * 0.25;
}

__global__ void simulate_kernel(
    const uint64_t* __restrict__ static_mat_idx,
    const uint64_t* __restrict__ static_num_conn,
    const uint64_t* __restrict__ static_conn_idx,
    const double*   __restrict__ static_conn_flux,
    const double*   __restrict__ dyn_energy,
    const double*   __restrict__ dyn_flux,
    double*         __restrict__ swap_energy,
    double*         __restrict__ swap_flux,
    const double*   __restrict__ mat_transfer_coeff,
    const double*   __restrict__ mat_external_flow,
    int local_x, int local_y, int pitch)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;

    if (lx >= local_x || ly >= local_y) return;

    int i = (lx + 1) * pitch + (ly + 1);
    int conn_base = i * MAX_CONNECTIONS;

    uint64_t mat_id = static_mat_idx[i];
    uint64_t n_conn = static_num_conn[i];
    double this_energy = dyn_energy[i];
    double this_flux   = dyn_flux[i];
    double tc = mat_transfer_coeff[mat_id];
    double ef = mat_external_flow[mat_id];

    double total_flux = ef;

    for (uint64_t j = 0; j < n_conn; ++j) {
        uint64_t ni = static_conn_idx[conn_base + j];
        double cf = static_conn_flux[conn_base + j];
        total_flux += computeFlux(tc, this_energy, cf, dyn_energy[ni]);
    }

    swap_energy[i] = this_energy + total_flux;
    swap_flux[i]   = this_flux + fabs(total_flux);
}

// ---------------------------------------------------------------------------
// Build local mesh for this MPI rank (OpenMP-parallelised).
// The local array has a 1-element halo on every side:
//   dimensions = (local_x+2) x (local_y+2),  pitch = local_y+2
//   owned cells live at lx in [1,local_x], ly in [1,local_y]
// ---------------------------------------------------------------------------
void buildLocalMesh(ElementStatic* elements_static,
                    ElementDynamic* elements_dynamic,
                    ElementDynamic* elements_dynamic_swap,
                    int n_elems_root, int local_x, int local_y,
                    int x_start, int y_start)
{
    int pitch = local_y + 2;

    // Initialise every cell (owned + halo)
    #pragma omp parallel for collapse(2)
    for (int lx = 0; lx <= local_x + 1; ++lx) {
        for (int ly = 0; ly <= local_y + 1; ++ly) {
            int i = lx * pitch + ly;
            elements_static[i].material_idx  = DEFAULT_MAT_ID;
            elements_static[i].num_connections = 0;
            elements_dynamic[i].current_energy = 0.0;
            elements_dynamic[i].total_flux     = 0.0;
            elements_dynamic_swap[i].current_energy = 0.0;
            elements_dynamic_swap[i].total_flux     = 0.0;
        }
    }

    // Build connectivity for owned cells only
    #pragma omp parallel for collapse(2)
    for (int lx = 1; lx <= local_x; ++lx) {
        for (int ly = 1; ly <= local_y; ++ly) {
            int i = lx * pitch + ly;
            int gx = x_start + lx - 1;
            int gy = y_start + ly - 1;

            ElementStatic& elem = elements_static[i];

            const int dx[4] = {1, -1, 0, 0};
            const int dy[4] = {0,  0, 1, -1};

            for (int n = 0; n < 4; ++n) {
                int ngx = gx + dx[n];
                int ngy = gy + dy[n];

                if (ngx >= 0 && ngx < n_elems_root &&
                    ngy >= 0 && ngy < n_elems_root) {
                    int nlx = ngx - x_start + 1;
                    int nly = ngy - y_start + 1;

                    if (nlx >= 0 && nlx <= local_x + 1 &&
                        nly >= 0 && nly <= local_y + 1) {
                        elem.connected_idx[elem.num_connections] =
                            static_cast<idx_t>(nlx * pitch + nly);
                        elem.connected_flux[elem.num_connections] = 1.0;
                        elem.num_connections++;
                    }
                }
            }
        }
    }

    // Mark corner elements as inflow / outflow
    int last = n_elems_root - 1;
    auto setMaterial = [&](int gx, int gy, idx_t mat_id) {
        int lx = gx - x_start + 1;
        int ly = gy - y_start + 1;
        if (lx >= 1 && lx <= local_x && ly >= 1 && ly <= local_y) {
            elements_static[lx * pitch + ly].material_idx = mat_id;
        }
    };
    setMaterial(0, 0, INFLOW_MAT_ID);
    setMaterial(0, last, OUTFLOW_MAT_ID);
    setMaterial(last, 0, OUTFLOW_MAT_ID);
    setMaterial(last, last, INFLOW_MAT_ID);
}

// ---------------------------------------------------------------------------
// Halo exchange – non-blocking MPI (Isend/Irecv/Waitall) for deadlock freedom.
// Rows are contiguous in memory; columns use separate temp buffers per direction.
// ---------------------------------------------------------------------------
void exchangeHalo(ElementDynamic* elements_dynamic,
                  int local_x, int local_y,
                  int rank_x, int rank_y,
                  int mpi_size_x, int mpi_size_y)
{
    int pitch = local_y + 2;

    // Compute neighbour ranks (non-periodic boundaries)
    int left   = (rank_x > 0)    ? (rank_y * mpi_size_x + (rank_x - 1)) : MPI_PROC_NULL;
    int right  = (rank_x < mpi_size_x - 1)  ? (rank_y * mpi_size_x + (rank_x + 1)) : MPI_PROC_NULL;
    int top    = (rank_y > 0)    ? ((rank_y - 1) * mpi_size_x + rank_x) : MPI_PROC_NULL;
    int bottom = (rank_y < mpi_size_y - 1)  ? ((rank_y + 1) * mpi_size_x + rank_x) : MPI_PROC_NULL;

    int row_count = (local_x + 2) * 2;   // ElementDynamic == 2 doubles
    int col_count = (local_y + 2) * 2;

    // 8 MPI requests max (send+recv for each of 4 directions)
    MPI_Request requests[8];
    int nreq = 0;

    // ---- row exchanges (contiguous) ----
    // Top halo: send ly=1 to top, receive into ly=0 from bottom
    if (top != MPI_PROC_NULL)
        MPI_Isend(elements_dynamic + pitch, row_count, MPI_DOUBLE,
                  top, 0, MPI_COMM_WORLD, &requests[nreq++]);
    if (bottom != MPI_PROC_NULL)
        MPI_Irecv(elements_dynamic, row_count, MPI_DOUBLE,
                  bottom, 0, MPI_COMM_WORLD, &requests[nreq++]);

    // Bottom halo: send ly=local_y to bottom, receive into ly=local_y+1 from top
    if (bottom != MPI_PROC_NULL)
        MPI_Isend(elements_dynamic + local_y * pitch, row_count, MPI_DOUBLE,
                  bottom, 0, MPI_COMM_WORLD, &requests[nreq++]);
    if (top != MPI_PROC_NULL)
        MPI_Irecv(elements_dynamic + (local_y + 1) * pitch, row_count, MPI_DOUBLE,
                  top, 0, MPI_COMM_WORLD, &requests[nreq++]);

    // ---- column exchanges (strided, via temp buffers) ----
    // Separate buffers for each direction to avoid aliasing
    ElementDynamic* send_col_l = new ElementDynamic[local_y + 2];
    ElementDynamic* send_col_r = new ElementDynamic[local_y + 2];
    ElementDynamic* recv_col_l = new ElementDynamic[local_y + 2];
    ElementDynamic* recv_col_r = new ElementDynamic[local_y + 2];

    // Pack left halo column (lx=1) for sending to left neighbour
    #pragma omp parallel for
    for (int j = 0; j <= local_y + 1; ++j)
        send_col_l[j] = elements_dynamic[j * pitch + 1];

    // Pack right halo column (lx=local_x) for sending to right neighbour
    #pragma omp parallel for
    for (int j = 0; j <= local_y + 1; ++j)
        send_col_r[j] = elements_dynamic[j * pitch + local_x];

    // Post non-blocking sends and receives
    if (left != MPI_PROC_NULL)
        MPI_Isend(send_col_l, col_count, MPI_DOUBLE, left, 0,
                  MPI_COMM_WORLD, &requests[nreq++]);
    if (right != MPI_PROC_NULL)
        MPI_Isend(send_col_r, col_count, MPI_DOUBLE, right, 0,
                  MPI_COMM_WORLD, &requests[nreq++]);
    if (right != MPI_PROC_NULL)
        MPI_Irecv(recv_col_r, col_count, MPI_DOUBLE, right, 0,
                  MPI_COMM_WORLD, &requests[nreq++]);
    if (left != MPI_PROC_NULL)
        MPI_Irecv(recv_col_l, col_count, MPI_DOUBLE, left, 0,
                  MPI_COMM_WORLD, &requests[nreq++]);

    // Wait for all communications to complete
    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    // Post-copy received column data into halo regions
    if (right != MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int j = 0; j <= local_y + 1; ++j)
            elements_dynamic[j * pitch] = recv_col_r[j];
    }
    if (left != MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int j = 0; j <= local_y + 1; ++j)
            elements_dynamic[j * pitch + (local_x + 1)] = recv_col_l[j];
    }

    delete[] send_col_l;
    delete[] send_col_r;
    delete[] recv_col_l;
    delete[] recv_col_r;
}

// ---------------------------------------------------------------------------
// Simulation loop: MPI halo exchange → CUDA kernel → buffer swap
// Uses SoA (Structure of Arrays) layout on GPU to avoid struct alignment issues.
// ---------------------------------------------------------------------------
void runSimulation(ElementStatic* elements_static,
                   ElementDynamic* elements_dynamic,
                   ElementDynamic* elements_dynamic_swap,
                   const Material* materials,
                   int n_iters, int local_x, int local_y,
                   int rank_x, int rank_y,
                   int mpi_size_x, int mpi_size_y)
{
    int pitch = local_y + 2;
    int total_local = (local_x + 2) * (local_y + 2);
    bool has_elements = (local_x > 0 && local_y > 0);

    if (!has_elements) {
        // No owned cells – still need to do halo exchanges and buffer swaps
        for (int iter = 0; iter < n_iters; ++iter) {
            exchangeHalo(elements_dynamic, local_x, local_y,
                         rank_x, rank_y, mpi_size_x, mpi_size_y);
            std::swap(elements_dynamic, elements_dynamic_swap);
        }
        return;
    }

    // ---- GPU SoA allocations ----
    uint64_t* d_mat_idx     = nullptr;
    uint64_t* d_num_conn    = nullptr;
    uint64_t* d_conn_idx    = nullptr;
    double*   d_conn_flux   = nullptr;
    double*   d_energy      = nullptr;
    double*   d_flux        = nullptr;
    double*   d_swap_energy = nullptr;
    double*   d_swap_flux   = nullptr;
    double*   d_mat_tc      = nullptr;
    double*   d_mat_ef      = nullptr;

    CUDA_CHECK(cudaMalloc(&d_mat_idx,     total_local * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_num_conn,    total_local * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_idx,    total_local * MAX_CONNECTIONS * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_conn_flux,   total_local * MAX_CONNECTIONS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_energy,      total_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flux,        total_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_swap_energy, total_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_swap_flux,   total_local * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mat_tc,      3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mat_ef,      3 * sizeof(double)));

    // ---- Upload static data (once) ----
    uint64_t* h_mat_idx  = new uint64_t[total_local];
    uint64_t* h_num_conn = new uint64_t[total_local];
    uint64_t* h_conn_idx = new uint64_t[total_local * MAX_CONNECTIONS];
    double*   h_conn_flux = new double[total_local * MAX_CONNECTIONS];

    #pragma omp parallel for
    for (int i = 0; i < total_local; ++i) {
        h_mat_idx[i]  = elements_static[i].material_idx;
        h_num_conn[i] = elements_static[i].num_connections;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            h_conn_idx[i * MAX_CONNECTIONS + j] = elements_static[i].connected_idx[j];
            h_conn_flux[i * MAX_CONNECTIONS + j] = elements_static[i].connected_flux[j];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_mat_idx,   h_mat_idx,   total_local * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_num_conn,  h_num_conn,  total_local * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_idx,  h_conn_idx,  total_local * MAX_CONNECTIONS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conn_flux, h_conn_flux, total_local * MAX_CONNECTIONS * sizeof(double), cudaMemcpyHostToDevice));

    double h_mat_tc[3], h_mat_ef[3];
    for (int m = 0; m < 3; ++m) {
        h_mat_tc[m] = materials[m].transfer_coeff;
        h_mat_ef[m] = materials[m].external_flow;
    }
    CUDA_CHECK(cudaMemcpy(d_mat_tc, h_mat_tc, 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat_ef, h_mat_ef, 3 * sizeof(double), cudaMemcpyHostToDevice));

    delete[] h_mat_idx;
    delete[] h_num_conn;
    delete[] h_conn_idx;
    delete[] h_conn_flux;

    // ---- Host-side SoA buffers for dynamic data (reused each iteration) ----
    double* h_energy      = new double[total_local];
    double* h_flux        = new double[total_local];
    double* h_swap_energy = new double[total_local];
    double* h_swap_flux   = new double[total_local];

    dim3 block(16, 16);
    dim3 grid((local_x + 15) / 16, (local_y + 15) / 16);

    for (int iter = 0; iter < n_iters; ++iter) {
        // 1. Exchange halo
        exchangeHalo(elements_dynamic, local_x, local_y,
                     rank_x, rank_y, mpi_size_x, mpi_size_y);

        // 2. Pack AoS -> SoA and upload
        #pragma omp parallel for
        for (int i = 0; i < total_local; ++i) {
            h_energy[i] = elements_dynamic[i].current_energy;
            h_flux[i]   = elements_dynamic[i].total_flux;
        }
        CUDA_CHECK(cudaMemcpy(d_energy, h_energy, total_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_flux,   h_flux,   total_local * sizeof(double), cudaMemcpyHostToDevice));

        // 3. Launch kernel
        simulate_kernel<<<grid, block>>>(
            d_mat_idx, d_num_conn, d_conn_idx, d_conn_flux,
            d_energy, d_flux,
            d_swap_energy, d_swap_flux,
            d_mat_tc, d_mat_ef,
            local_x, local_y, pitch);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 4. Download results
        CUDA_CHECK(cudaMemcpy(h_swap_energy, d_swap_energy, total_local * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_swap_flux,   d_swap_flux,   total_local * sizeof(double), cudaMemcpyDeviceToHost));

        // Unpack SoA -> AoS
        #pragma omp parallel for
        for (int i = 0; i < total_local; ++i) {
            elements_dynamic_swap[i].current_energy = h_swap_energy[i];
            elements_dynamic_swap[i].total_flux     = h_swap_flux[i];
        }

        // 5. Swap buffers
        std::swap(elements_dynamic, elements_dynamic_swap);
    }

    // Cleanup
    delete[] h_energy;
    delete[] h_flux;
    delete[] h_swap_energy;
    delete[] h_swap_flux;
    CUDA_CHECK(cudaFree(d_mat_idx));
    CUDA_CHECK(cudaFree(d_num_conn));
    CUDA_CHECK(cudaFree(d_conn_idx));
    CUDA_CHECK(cudaFree(d_conn_flux));
    CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_flux));
    CUDA_CHECK(cudaFree(d_swap_energy));
    CUDA_CHECK(cudaFree(d_swap_flux));
    CUDA_CHECK(cudaFree(d_mat_tc));
    CUDA_CHECK(cudaFree(d_mat_ef));
}

// ---------------------------------------------------------------------------
// Validation – OpenMP reduction on each rank, then MPI_Reduce to rank 0
// ---------------------------------------------------------------------------
bool validateResults(ElementDynamic* elements_dynamic,
                     int local_x, int local_y,
                     int n_elems, int mpi_rank, int mpi_size)
{
    int pitch = local_y + 2;
    int owned_count = local_x * local_y;

    val_t local_energy_sum = 0.0;
    val_t local_flux_sum   = 0.0;
    val_t local_energy_max = -std::numeric_limits<val_t>::infinity();
    val_t local_energy_min =  std::numeric_limits<val_t>::infinity();

    if (owned_count > 0) {
        #pragma omp parallel for reduction(+:local_energy_sum, local_flux_sum) \
                                 reduction(max:local_energy_max) \
                                 reduction(min:local_energy_min)
        for (int idx = 0; idx < owned_count; ++idx) {
            int lx = idx / local_y;
            int ly = idx % local_y;
            int i  = (lx + 1) * pitch + (ly + 1);

            local_energy_sum += elements_dynamic[i].current_energy;
            local_flux_sum   += elements_dynamic[i].total_flux;
            local_energy_max  = std::max(elements_dynamic[i].current_energy,
                                         local_energy_max);
            local_energy_min  = std::min(elements_dynamic[i].current_energy,
                                         local_energy_min);
        }
    }

    val_t g_energy_sum, g_flux_sum, g_energy_max, g_energy_min;
    MPI_Reduce(&local_energy_sum, &g_energy_sum, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &g_flux_sum, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &g_energy_max, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &g_energy_min, 1, MPI_DOUBLE, MPI_MIN,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", g_energy_sum);
        printf("  Flux sum: %.2f\n", g_flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", g_energy_min, g_energy_max);

        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(g_energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }
        if (std::abs(g_energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 "
                   "(expected conservation)\n");
        }
        if (!std::isfinite(g_flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            return false;
        }
        if (!std::isfinite(g_energy_max) || !std::isfinite(g_energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            return false;
        }
        printf("  Validation: PASSED\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// Hash – OpenMP XOR-reduction on each rank, then MPI custom-op reduction
// ---------------------------------------------------------------------------
void xor_func(void* input, void* inoutput, int* len, MPI_Datatype*) {
    for (int i = 0; i < *len; ++i)
        ((uint64_t*)inoutput)[i] ^= ((uint64_t*)input)[i];
}

uint64_t computeHash(ElementDynamic* elements_dynamic,
                     int local_x, int local_y,
                     int x_start, int y_start, int n_elems_root,
                     int mpi_rank, int mpi_size)
{
    int pitch = local_y + 2;
    int owned_count = local_x * local_y;
    uint64_t local_hash = 0;

    #pragma omp parallel for reduction(^:local_hash)
    for (int idx = 0; idx < owned_count; ++idx) {
        int lx = idx / local_y;
        int ly = idx % local_y;
        int i  = (lx + 1) * pitch + (ly + 1);
        int gx = x_start + lx;
        int gy = y_start + ly;
        idx_t global_idx = static_cast<idx_t>(gx) * n_elems_root + gy;

        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(
            &elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(
            &elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash;
    MPI_Op xor_op;
    MPI_Op_create(xor_func, true, &xor_op);
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UNSIGNED_LONG_LONG,
               xor_op, 0, MPI_COMM_WORLD);
    MPI_Op_free(&xor_op);
    return global_hash;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_COMM_THREAD_MULTIPLE, &provided);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Assign each MPI rank to a CUDA device (round-robin)
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(mpi_rank % device_count));

    int n_elems_root = 512;
    int n_iters      = 10;
    bool validate    = false;
    bool printResults = false;

    // Parse arguments (all ranks must parse)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n_elems_root = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)
            validate = true;
        else if (strcmp(argv[i], "-r") == 0)
            printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // ---- 2-D MPI domain decomposition ----
    int mpi_size_x = 1, mpi_size_y = 1;
    for (int i = 1; i * i <= mpi_size; ++i) {
        if (mpi_size % i == 0) {
            mpi_size_x = mpi_size / i;
            mpi_size_y = i;
        }
    }
    int rank_x = mpi_rank % mpi_size_x;
    int rank_y = mpi_rank / mpi_size_x;

    int base_x      = n_elems_root / mpi_size_x;
    int base_y      = n_elems_root / mpi_size_y;
    int remainder_x = n_elems_root % mpi_size_x;
    int remainder_y = n_elems_root % mpi_size_y;
    int local_x     = base_x + (rank_x < remainder_x ? 1 : 0);
    int local_y     = base_y + (rank_y < remainder_y ? 1 : 0);
    int x_start     = rank_x * base_x + std::min(rank_x, remainder_x);
    int y_start     = rank_y * base_y + std::min(rank_y, remainder_y);

    int pitch       = local_y + 2;
    int total_local = (local_x + 2) * (local_y + 2);
    int owned_count = local_x * local_y;
    const int n_elems = n_elems_root * n_elems_root;

    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("MPI ranks: %d (%d x %d)\n",
               mpi_size, mpi_size_x, mpi_size_y);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Materials (identical on every rank)
    Material materials[3] = {
        {0.8,  0.0},   // Default
        {0.8,  0.5},   // Inflow
        {0.8, -0.5},   // Outflow
    };

    // Allocate per-rank memory
    ElementStatic*   elements_static     =
        new ElementStatic[total_local];
    ElementDynamic*  elements_dynamic    =
        new ElementDynamic[total_local];
    ElementDynamic*  elements_dynamic_swap =
        new ElementDynamic[total_local];

    // ---- Build mesh (OpenMP) ----
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    buildLocalMesh(elements_static, elements_dynamic, elements_dynamic_swap,
                   n_elems_root, local_x, local_y, x_start, y_start);

    // Memory report
    const size_t static_mem  = total_local * sizeof(ElementStatic);
    const size_t dynamic_mem = total_local * sizeof(ElementDynamic) * 2;
    const size_t total_mem   = static_mem + dynamic_mem;

    if (mpi_rank == 0) {
        printf("Memory per rank: %.2f MB "
               "(static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
    }

    // ---- Run simulation (MPI halo + CUDA kernel) ----
    if (mpi_rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(elements_static, elements_dynamic, elements_dynamic_swap,
                  materials, n_iters, local_x, local_y,
                  rank_x, rank_y, mpi_size_x, mpi_size_y);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<
        std::chrono::milliseconds>(end - start).count();
    MPI_Barrier(MPI_COMM_WORLD);

    // Performance metrics (rank 0 prints)
    // computeHash uses MPI_Reduce so all ranks must call it
    uint64_t hash = computeHash(elements_dynamic,
                                local_x, local_y,
                                x_start, y_start, n_elems_root,
                                mpi_rank, mpi_size);

    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // ---- Print results (-r flag) ----
    if (printResults) {
        std::vector<double> local_energy(owned_count);
        for (int lx = 0; lx < local_x; ++lx) {
            for (int ly = 0; ly < local_y; ++ly) {
                int i = (lx + 1) * pitch + (ly + 1);
                local_energy[lx * local_y + ly] =
                    elements_dynamic[i].current_energy;
            }
        }

        if (mpi_rank == 0) {
            std::vector<double> all_energy(n_elems);
            MPI_Status status;

            for (int r = 0; r < mpi_size; ++r) {
                int rx = r % mpi_size_x;
                int ry = r / mpi_size_x;
                int lx = base_x + (rx < remainder_x ? 1 : 0);
                int ly = base_y + (ry < remainder_y ? 1 : 0);
                int xs = rx * base_x + std::min(rx, remainder_x);
                int ys = ry * base_y + std::min(ry, remainder_y);

                std::vector<double> recv_buf(lx * ly);
                if (r == 0) {
                    recv_buf = local_energy;
                } else {
                    MPI_Recv(recv_buf.data(), lx * ly, MPI_DOUBLE,
                             r, 0, MPI_COMM_WORLD, &status);
                }

                for (int i = 0; i < lx; ++i) {
                    for (int j = 0; j < ly; ++j) {
                        int gi = (xs + i) * n_elems_root + (ys + j);
                        all_energy[gi] = recv_buf[i * ly + j];
                    }
                }
            }
            print_results(all_energy, "ElementEnergy");
        } else {
            MPI_Send(local_energy.data(), owned_count, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
        }
    }

    // ---- Validation ----
    if (validate) {
        bool valid = validateResults(elements_dynamic,
                                     local_x, local_y,
                                     n_elems, mpi_rank, mpi_size);
        if (!valid) {
            delete[] elements_static;
            delete[] elements_dynamic;
            delete[] elements_dynamic_swap;
            MPI_Finalize();
            return 1;
        }
    }

    // Cleanup
    delete[] elements_static;
    delete[] elements_dynamic;
    delete[] elements_dynamic_swap;
    MPI_Finalize();

    return 0;
}
