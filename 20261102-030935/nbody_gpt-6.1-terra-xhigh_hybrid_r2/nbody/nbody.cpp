#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

struct Body {
    double x, y, z, vx, vy, vz;
};

// Each thread owns one target body. All blocks read the same, immutable
// position snapshot; the separate output buffer makes integration race-free.
__global__ void advance(const double4* positions, double4* next,
                        double4* velocities, int n, int offset, int local_n) {
    __shared__ double4 tile[BLOCK_SIZE];
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local_i < local_n;
    double4 p = make_double4(0, 0, 0, 0);
    if (active) p = positions[offset + local_i];
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
                const double dist2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inv = 1.0 / sqrt(dist2);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        double4 v = velocities[local_i];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        p.x += v.x * DT;
        p.y += v.y * DT;
        p.z += v.z * DT;
        velocities[local_i] = v;
        next[local_i] = p;
    }
}

// One row per body counts each potential-energy pair exactly once.
__global__ void energyRows(const double4* positions, const double4* velocities,
                           double* rows, int n, int offset, int local_n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    const double4 p = positions[offset + i];
    const double4 v = velocities[i];
    double e = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    for (int j = offset + i + 1; j < n; ++j) {
        const double4 q = positions[j];
        const double dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    rows[i] = e;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI's variable-count collectives use int counts, in units of doubles.
    if (n < 0 || n > std::numeric_limits<int>::max() / 4) {
        if (rank == 0) fprintf(stderr, "Invalid number of bodies: %d\n", n);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, devices;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local_comm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0, start = 0; r < ranks; ++r) {
        const int count = n / ranks + (r < n % ranks);
        counts[r] = count * 4;
        displs[r] = start * 4;
        start += count;
    }
    const int local_n = counts[rank] / 4, offset = displs[rank] / 4;
    const size_t full_bytes = std::max<size_t>(1, n) * sizeof(double4);
    const size_t local_bytes = std::max<size_t>(1, local_n) * sizeof(double4);
    double4 *host_pos, *d_pos, *d_next, *d_vel;
    CUDA_CHECK(cudaMallocHost(&host_pos, full_bytes));
    CUDA_CHECK(cudaMalloc(&d_pos, full_bytes));
    // A full-sized second buffer permits pointer swapping for the single-rank case.
    CUDA_CHECK(cudaMalloc(&d_next, ranks == 1 ? full_bytes : local_bytes));
    CUDA_CHECK(cudaMalloc(&d_vel, local_bytes));

    // rand_r is sequential: generate the original stream exactly, then pack in parallel.
    std::vector<Body> initial(n);
    unsigned int seed = 42;
    for (Body& b : initial) {
        b.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
    std::vector<double4> host_vel(local_n);
    #pragma omp parallel for schedule(static) if(n >= 4096)
    for (int i = 0; i < n; ++i) {
        const Body& b = initial[i];
        host_pos[i] = make_double4(b.x, b.y, b.z, 0);
        if (i >= offset && i < offset + local_n)
            host_vel[i - offset] = make_double4(b.vx, b.vy, b.vz, 0);
    }
    std::vector<Body>().swap(initial);
    if (n) CUDA_CHECK(cudaMemcpy(d_pos, host_pos, n * sizeof(double4), cudaMemcpyHostToDevice));
    if (local_n) CUDA_CHECK(cudaMemcpy(d_vel, host_vel.data(), local_n * sizeof(double4), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        if (local_n) {
            advance<<<(local_n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_pos, d_next, d_vel, n, offset, local_n);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(d_pos, d_next);
        } else if (step + 1 < steps || validate || printResults) {
            if (local_n) CUDA_CHECK(cudaMemcpy(host_pos + offset, d_next, local_n * sizeof(double4), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_pos, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (n) CUDA_CHECK(cudaMemcpy(d_pos, host_pos, n * sizeof(double4), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000));

    if (validate || printResults) {
        if (ranks == 1 && n) CUDA_CHECK(cudaMemcpy(host_pos, d_pos, n * sizeof(double4), cudaMemcpyDeviceToHost));
        if (local_n) CUDA_CHECK(cudaMemcpy(host_vel.data(), d_vel, local_n * sizeof(double4), cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        std::vector<double4> all_vel(rank == 0 ? n : 0);
        MPI_Gatherv(host_vel.data(), counts[rank], MPI_DOUBLE, all_vel.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> data(static_cast<size_t>(n) * 6);
            #pragma omp parallel for schedule(static) if(n >= 4096)
            for (int i = 0; i < n; ++i) {
                data[6ULL*i] = host_pos[i].x; data[6ULL*i+1] = host_pos[i].y; data[6ULL*i+2] = host_pos[i].z;
                data[6ULL*i+3] = all_vel[i].x; data[6ULL*i+4] = all_vel[i].y; data[6ULL*i+5] = all_vel[i].z;
            }
            print_results(data, "Bodies");
        }
    }

    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int first_bad = n;
        #pragma omp parallel for reduction(min:first_bad) schedule(static) if(local_n >= 4096)
        for (int i = 0; i < local_n; ++i) {
            const double4 p = host_pos[offset + i], v = host_vel[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
                std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6 ||
                std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6)
                first_bad = std::min(first_bad, offset + i);
        }
        int global_bad;
        MPI_Allreduce(&first_bad, &global_bad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (global_bad != n) {
            int reason = 0;
            if (global_bad >= offset && global_bad < offset + local_n) {
                const double4 p = host_pos[global_bad], v = host_vel[global_bad - offset];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) reason = 1;
                else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) reason = 2;
                else reason = 3;
            }
            int global_reason;
            MPI_Reduce(&reason, &global_reason, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                const char* messages[] = {"", "found NaN or Inf value in body state", "body position exceeds reasonable bounds", "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\nValidation: FAILED\n", messages[global_reason]);
            }
            result = 1;
        } else {
            double *d_rows = nullptr;
            std::vector<double> rows(local_n);
            if (local_n) {
                CUDA_CHECK(cudaMalloc(&d_rows, local_n * sizeof(double)));
                energyRows<<<(local_n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_pos, d_vel, d_rows, n, offset, local_n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(rows.data(), d_rows, local_n * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(d_rows));
            }
            double local_energy = 0;
            #pragma omp parallel for reduction(+:local_energy) schedule(static) if(local_n >= 4096)
            for (int i = 0; i < local_n; ++i) local_energy += rows[i];
            double energy = 0;
            MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Final energy: %.6f\nValidation: PASSED\n", energy);
        }
    }
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFreeHost(host_pos));
    MPI_Finalize();
    return result;
}
