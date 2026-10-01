#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each rank owns a contiguous slab in Z. Planes 0 and local_z+1 are ghosts.
__global__ void chemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                  size_t nx, size_t ny, size_t plane,
                                  double gamma, double e_AA, double e_BB, double e_AB) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    size_t z = static_cast<size_t>(blockIdx.z) + 1;
    size_t i = z * plane + y * nx + x;
    double cv = c[i];
    double cxx = c[i + (x + 1 < nx ? 1 : 0)] + c[i - (x > 0 ? 1 : 0)] - 2.0 * cv;
    double cyy = c[i + (y + 1 < ny ? nx : 0)] + c[i - (y > 0 ? nx : 0)] - 2.0 * cv;
    double czz = c[i + plane] + c[i - plane] - 2.0 * cv;
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
          + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
}

__global__ void updateConcentration(const double* __restrict__ c, const double* __restrict__ mu,
                                    double* __restrict__ next, size_t nx, size_t ny,
                                    size_t plane, double dtD) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    size_t z = static_cast<size_t>(blockIdx.z) + 1;
    size_t i = z * plane + y * nx + x;
    double value = mu[i];
    double xx = mu[i + (x + 1 < nx ? 1 : 0)] + mu[i - (x > 0 ? 1 : 0)] - 2.0 * value;
    double yy = mu[i + (y + 1 < ny ? nx : 0)] + mu[i - (y > 0 ? nx : 0)] - 2.0 * value;
    double zz = mu[i + plane] + mu[i - plane] - 2.0 * value;
    next[i] = c[i] + dtD * (xx + yy + zz);
}

// Portable MPI exchange: pinned staging allows MPI implementations without CUDA awareness.
// The interior slabs stay on the GPU for the entire simulation.
void exchangePlanes(double* field, size_t local_z, size_t plane, int rank, int ranks,
                    double* send_low, double* send_high, double* recv_low, double* recv_high) {
    const size_t bytes = plane * sizeof(double);
    if (ranks == 1) {
        CUDA_CHECK(cudaMemcpy(field, field + plane, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(field + (local_z + 1) * plane, field + local_z * plane,
                              bytes, cudaMemcpyDeviceToDevice));
        return;
    }
    int previous = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    if (previous != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(send_low, field + plane, bytes, cudaMemcpyDeviceToHost));
    if (next != MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(send_high, field + local_z * plane, bytes, cudaMemcpyDeviceToHost));

    MPI_Request requests[4];
    int count = 0;
    if (previous != MPI_PROC_NULL) {
        MPI_Irecv(recv_low, static_cast<int>(plane), MPI_DOUBLE, previous, 1, MPI_COMM_WORLD, &requests[count++]);
        MPI_Isend(send_low, static_cast<int>(plane), MPI_DOUBLE, previous, 2, MPI_COMM_WORLD, &requests[count++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Irecv(recv_high, static_cast<int>(plane), MPI_DOUBLE, next, 2, MPI_COMM_WORLD, &requests[count++]);
        MPI_Isend(send_high, static_cast<int>(plane), MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[count++]);
    }
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    CUDA_CHECK(cudaMemcpy(field, previous == MPI_PROC_NULL ? field + plane : recv_low,
                          bytes, previous == MPI_PROC_NULL ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(field + (local_z + 1) * plane,
                          next == MPI_PROC_NULL ? field + local_z * plane : recv_high,
                          bytes, next == MPI_PROC_NULL ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
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
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz < static_cast<size_t>(ranks) ||
        nx > SIZE_MAX / ny || nx * ny > static_cast<size_t>(INT_MAX) ||
        nx * ny > SIZE_MAX / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or more MPI ranks than Z planes\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t local_z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t z_start = (nz / ranks) * rank + std::min(static_cast<size_t>(rank), nz % ranks);
    const size_t local_cells = plane * local_z;
    if (local_cells > static_cast<size_t>(INT_MAX) ||
        (local_z + 2) > SIZE_MAX / plane ||
        (local_z + 2) * plane > SIZE_MAX / sizeof(double) ||
        (printResults || validate) && gridSize > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Grid exceeds MPI message size limits\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "A CUDA device is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    const size_t allocated = (local_z + 2) * plane;
    double *cold, *cnew, *mu;
    CUDA_CHECK(cudaMalloc(&cold, allocated * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, allocated * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, allocated * sizeof(double)));
    std::vector<double> initial(local_cells);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_cells; ++i) {
        size_t global = z_start * plane + i;
        double pseudo = (((global + 1) * static_cast<size_t>(1299709)) % gridSize) /
                        static_cast<double>(gridSize);
        initial[i] = -1.0 + 2.0 * pseudo;
    }
    CUDA_CHECK(cudaMemcpy(cold + plane, initial.data(), local_cells * sizeof(double), cudaMemcpyHostToDevice));
    initial.clear();
    initial.shrink_to_fit();

    double *send_low, *send_high, *recv_low, *recv_high;
    CUDA_CHECK(cudaMallocHost(&send_low, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&send_high, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recv_low, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recv_high, plane * sizeof(double)));

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    const dim3 block(32, 8);
    if (local_z > 65535 || (nx + block.x - 1) / block.x > static_cast<size_t>(INT_MAX) ||
        (ny + block.y - 1) / block.y > 65535) {
        if (rank == 0) fprintf(stderr, "Grid exceeds CUDA launch limits\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const dim3 blocks(static_cast<unsigned>((nx + block.x - 1) / block.x),
                      static_cast<unsigned>((ny + block.y - 1) / block.y),
                      static_cast<unsigned>(local_z));
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (int t = 0; t < iterations; ++t) {
        exchangePlanes(cold, local_z, plane, rank, ranks, send_low, send_high, recv_low, recv_high);
        chemicalPotential<<<blocks, block>>>(cold, mu, nx, ny, plane,
                                                                      0.5, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        exchangePlanes(mu, local_z, plane, rank, ranks, send_low, send_high, recv_low, recv_high);
        updateConcentration<<<blocks, block>>>(cold, mu, cnew, nx, ny, plane, 0.01 * 1.0);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double local_seconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", gridSize * static_cast<double>(iterations) / seconds / 1e6);
    }

    int validation_failed = 0;
    if (printResults || validate) {
        std::vector<double> local(local_cells);
        CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, local_cells * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<int> counts, offsets;
        std::vector<double> result;
        if (rank == 0) {
            counts.resize(ranks);
            offsets.resize(ranks);
            result.resize(gridSize);
            for (int r = 0; r < ranks; ++r) {
                size_t rz = nz / ranks + (static_cast<size_t>(r) < nz % ranks);
                size_t start_z = (nz / ranks) * r + std::min(static_cast<size_t>(r), nz % ranks);
                counts[r] = static_cast<int>(rz * plane);
                offsets[r] = static_cast<int>(start_z * plane);
            }
        }
        MPI_Gatherv(local.data(), static_cast<int>(local_cells), MPI_DOUBLE, result.data(),
                    counts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results(result, "Concentration");
            if (validate) {
                printf("Validating result...\n");
                double minimum = std::numeric_limits<double>::infinity();
                double maximum = -std::numeric_limits<double>::infinity();
                int nonfinite = 0;
                #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:nonfinite)
                for (size_t i = 0; i < gridSize; ++i) {
                    double value = result[i];
                    if (!std::isfinite(value)) nonfinite = 1;
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
                if (nonfinite) printf("Validation failed: found NaN or Inf value\n");
                else printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
                if (!nonfinite && (maximum > 10.0 || minimum < -10.0))
                    printf("Validation failed: values out of expected range\n");
                validation_failed = nonfinite || maximum > 10.0 || minimum < -10.0;
                printf("Validation: %s\n", validation_failed ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_Bcast(&validation_failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFreeHost(send_low));
    CUDA_CHECK(cudaFreeHost(send_high));
    CUDA_CHECK(cudaFreeHost(recv_low));
    CUDA_CHECK(cudaFreeHost(recv_high));
    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    MPI_Finalize();
    return validation_failed ? 1 : 0;
}
