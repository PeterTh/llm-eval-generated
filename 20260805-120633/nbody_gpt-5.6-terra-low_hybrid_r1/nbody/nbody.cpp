#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 256;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

#define CUDA_CHECK(call) do {                                                       \
    const cudaError_t error_ = (call);                                              \
    if (error_ != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(error_));                                        \
        MPI_Abort(MPI_COMM_WORLD, 1);                                               \
    }                                                                               \
} while (0)

// Every rank owns a contiguous set of target bodies.  Sources remain replicated,
// which keeps the O(N^2) calculation embarrassingly parallel on each GPU.
__global__ void advanceBodies(Body* bodies, int first, int count, int total) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const int i = first + local;
    const Vec3 position = active ? bodies[i].pos : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    __shared__ Body sourceTile[THREADS_PER_BLOCK];

    for (int tile = 0; tile < total; tile += THREADS_PER_BLOCK) {
        const int source = tile + threadIdx.x;
        if (source < total) sourceTile[threadIdx.x] = bodies[source];
        __syncthreads();
        const int tileCount = min(THREADS_PER_BLOCK, total - tile);
        if (active) {
            for (int k = 0; k < tileCount; ++k) {
                const double dx = sourceTile[k].pos.x - position.x;
                const double dy = sourceTile[k].pos.y - position.y;
                const double dz = sourceTile[k].pos.z - position.z;
                const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    Body updated = bodies[i];
    updated.vel.x += DT * fx;
    updated.vel.y += DT * fy;
    updated.vel.z += DT * fz;
    updated.pos.x += DT * updated.vel.x;
    updated.pos.y += DT * updated.vel.y;
    updated.pos.z += DT * updated.vel.z;
    bodies[i] = updated;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Preserve the original deterministic stream exactly, so all ranks begin alike.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i)
        kinetic += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + bodies[i].vel.y * bodies[i].vel.y + bodies[i].vel.z * bodies[i].vel.z);
    double potential = 0.0;
#pragma omp parallel for reduction(+:potential) schedule(dynamic)
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        valid &= std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                 std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z) &&
                 std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                 std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
    }
    return valid != 0;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\n  -n <num>     Number of bodies (default: 1024)\n"
           "  -s <num>     Number of simulation steps (default: 10)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies <= 0 || numSteps < 0) { if (!rank) fprintf(stderr, "Body count must be positive and steps non-negative\n"); MPI_Finalize(); return 1; }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const int first = (numBodies * rank) / ranks;
    const int last = (numBodies * (rank + 1)) / ranks;
    const int localCount = last - first;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = ((numBodies * (r + 1)) / ranks - (numBodies * r) / ranks) * static_cast<int>(sizeof(Body)); offsets[r] = (numBodies * r) / ranks * static_cast<int>(sizeof(Body)); }

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    Body* deviceBodies = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)));
    CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body), cudaMemcpyHostToDevice));

    if (!rank) {
        printf("N-Body Simulation (MPI ranks: %d, CUDA devices/rank assignment: %d, OpenMP threads: %d)\n", ranks, devices, omp_get_max_threads());
        printf("Number of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) advanceBodies<<<(localCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(deviceBodies, first, localCount, numBodies);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(bodies.data() + first, deviceBodies + first, static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(), offsets.data(), MPI_BYTE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), static_cast<size_t>(numBodies) * sizeof(Body), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Simulation time: %.3f ms\n", maxElapsed * 1000.0);

    int result = 0;
    if (!rank && printResults) {
        std::vector<double> data(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) { const Body& b = bodies[i]; double* d = data.data() + 6LL * i; d[0]=b.pos.x; d[1]=b.pos.y; d[2]=b.pos.z; d[3]=b.vel.x; d[4]=b.vel.y; d[5]=b.vel.z; }
        print_results(data, "Bodies");
    }
    if (!rank && validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
        else { printf("Validation: FAILED\n"); result = 1; }
    }
    CUDA_CHECK(cudaFree(deviceBodies));
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
