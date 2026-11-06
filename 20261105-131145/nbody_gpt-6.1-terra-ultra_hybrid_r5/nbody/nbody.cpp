#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed for MPI");

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Every thread reads the same old position snapshot. The updated positions are
// written separately, so blocks can integrate immediately after their force sum.
__global__ void advance(const Vec3* __restrict__ positions, Vec3* __restrict__ next,
                        Vec3* __restrict__ velocities, int n, int first, int local_n) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Vec3 p = {0.0, 0.0, 0.0};
    if (i < local_n) p = positions[first + i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        int count = min(BLOCK_SIZE, n - base);
        if (i < local_n) {
            for (int k = 0; k < count; ++k) {
                double dx = tile[k].x - p.x;
                double dy = tile[k].y - p.y;
                double dz = tile[k].z - p.z;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (i < local_n) {
        Vec3 v = velocities[i];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[i] = v;
        next[i] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

bool validateSimulation(const std::vector<Body>& bodies) {
    // Find the first invalid body, preserving the original diagnostic ordering.
    size_t first = bodies.size();
    #pragma omp parallel for reduction(min:first) schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Body& b = bodies[i];
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
            std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6 ||
            std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6)
            first = std::min(first, i);
    }
    if (first == bodies.size()) return true;
    const Body& b = bodies[first];
    if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
        !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z))
        std::printf("Validation failed: found NaN or Inf value in body state\n");
    else if (std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6)
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
    else std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return false;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation (checks energy conservation)\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 1024, steps = 10;
    bool validate = false, results = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) results = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); parse_status = 1; break; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parse_status = 2; break; }
    }
    if (parse_status) { MPI_Finalize(); return parse_status == 1 ? 0 : 1; }
    if (n < 0 || n > INT_MAX / 3) {
        if (rank == 0) std::fprintf(stderr, "Number of bodies must be between 0 and %d\n", INT_MAX / 3);
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, device_count;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) { std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0, offset = 0; r < ranks; ++r) {
        int count = n / ranks + (r < n % ranks);
        counts[r] = 3 * count;
        displs[r] = 3 * offset;
        offset += count;
    }
    int local_n = counts[rank] / 3, first = displs[rank] / 3;
    // Each rank reproduces the original serial random stream exactly.
    std::vector<Body> initial(n);
    randomizeBodies(initial);
    // Pinned buffers avoid an extra pageable-memory staging copy for every
    // position exchange. MPI uses host pointers and does not require CUDA-aware MPI.
    Vec3 *positions = nullptr, *local_positions = nullptr;
    if (n > 0) CUDA_CHECK(cudaMallocHost(&positions, size_t(n) * sizeof(Vec3)));
    if (local_n > 0 && ranks > 1)
        CUDA_CHECK(cudaMallocHost(&local_positions, size_t(local_n) * sizeof(Vec3)));
    std::vector<Vec3> velocities(local_n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) positions[i] = initial[i].pos;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_n; ++i) velocities[i] = initial[first + i].vel;
    initial.clear(); initial.shrink_to_fit();

    Vec3 *d_positions = nullptr, *d_next = nullptr, *d_velocities = nullptr;
    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_positions, size_t(n) * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d_positions, positions, size_t(n) * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    if (local_n > 0) {
        CUDA_CHECK(cudaMalloc(&d_next, size_t(local_n) * sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&d_velocities, size_t(local_n) * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d_velocities, velocities.data(), size_t(local_n) * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (local_n > 0) {
            advance<<<(local_n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_positions, d_next, d_velocities, n, first, local_n);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(d_positions, d_next);
        } else {
            if (local_n > 0) CUDA_CHECK(cudaMemcpy(local_positions, d_next, size_t(local_n) * sizeof(Vec3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(local_positions, counts[rank], MPI_DOUBLE, positions, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (step + 1 < steps && n > 0)
                CUDA_CHECK(cudaMemcpy(d_positions, positions, size_t(n) * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }
    if (ranks == 1 && local_n > 0 && steps > 0) CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(max_elapsed));

    if (results || validate) {
        if (ranks == 1 && n > 0 && steps > 0)
            CUDA_CHECK(cudaMemcpy(positions, d_positions, size_t(n) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        if (local_n > 0)
            CUDA_CHECK(cudaMemcpy(velocities.data(), d_velocities, size_t(local_n) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        std::vector<Vec3> all_velocities(rank == 0 ? n : 0);
        MPI_Gatherv(velocities.data(), counts[rank], MPI_DOUBLE, all_velocities.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<Body> bodies(rank == 0 ? n : 0);
        if (rank == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) bodies[i] = {positions[i], all_velocities[i]};
            if (results) {
                std::vector<double> data(size_t(n) * 6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    size_t k = size_t(i) * 6;
                    data[k] = bodies[i].pos.x; data[k+1] = bodies[i].pos.y; data[k+2] = bodies[i].pos.z;
                    data[k+3] = bodies[i].vel.x; data[k+4] = bodies[i].vel.y; data[k+5] = bodies[i].vel.z;
                }
                print_results(data, "Bodies");
            }
        }
        if (validate) {
            int valid = 1;
            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                valid = validateSimulation(bodies);
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (valid) {
                double energy = 0.0;
                #pragma omp parallel for reduction(+:energy) schedule(static)
                for (int i = 0; i < local_n; ++i) {
                    const Vec3& v = velocities[i];
                    energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
                }
                // Cyclic distribution balances the triangular potential-energy
                // calculation across ranks as well as across OpenMP threads.
                #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
                for (int i = rank; i < n; i += ranks) {
                    const Vec3& p = positions[i];
                    for (int j = i + 1; j < n; ++j) {
                        double dx = positions[j].x - p.x, dy = positions[j].y - p.y, dz = positions[j].z - p.z;
                        energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    }
                }
                double total = 0.0;
                MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) std::printf("Final energy: %.6f\nValidation: PASSED\n", total);
            } else if (rank == 0) std::printf("Validation: FAILED\n");
            if (!valid) {
                if (d_positions) CUDA_CHECK(cudaFree(d_positions));
                if (d_next) CUDA_CHECK(cudaFree(d_next));
                if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
                if (positions) CUDA_CHECK(cudaFreeHost(positions));
                if (local_positions) CUDA_CHECK(cudaFreeHost(local_positions));
                MPI_Finalize(); return 1;
            }
        }
    }
    if (d_positions) CUDA_CHECK(cudaFree(d_positions));
    if (d_next) CUDA_CHECK(cudaFree(d_next));
    if (d_velocities) CUDA_CHECK(cudaFree(d_velocities));
    if (positions) CUDA_CHECK(cudaFreeHost(positions));
    if (local_positions) CUDA_CHECK(cudaFreeHost(local_positions));
    MPI_Finalize();
    return 0;
}
