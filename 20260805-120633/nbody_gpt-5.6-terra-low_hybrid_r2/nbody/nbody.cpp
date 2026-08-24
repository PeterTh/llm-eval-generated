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
constexpr int STATE_WIDTH = 6;
constexpr int THREADS = 256;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

[[noreturn]] static void cudaFail(cudaError_t error, const char* expression,
                                  const char* file, int line) {
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", expression, file, line,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}
#define CUDA_CHECK(expr) do { const cudaError_t _error = (expr); if (_error != cudaSuccess) cudaFail(_error, #expr, __FILE__, __LINE__); } while (0)

// Each MPI rank owns [first, first + count).  All positions are replicated on
// its GPU, so the force calculation has exactly the original all-pairs semantics.
__global__ void advanceBodies(const double* __restrict__ current,
                              double* __restrict__ next, int n, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = first + local;
    const bool active = local < count;
    const double xi = active ? current[STATE_WIDTH * i] : 0.0;
    const double yi = active ? current[STATE_WIDTH * i + 1] : 0.0;
    const double zi = active ? current[STATE_WIDTH * i + 2] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    __shared__ double positionTile[THREADS * 3];

    // A block reuses each coalesced position load for 256 target bodies.
    // Tiles are traversed in ascending j order, preserving the original sum order.
    for (int base = 0; base < n; base += THREADS) {
        const int source = base + threadIdx.x;
        if (source < n) {
            const int sourceOffset = STATE_WIDTH * source;
            positionTile[3 * threadIdx.x] = current[sourceOffset];
            positionTile[3 * threadIdx.x + 1] = current[sourceOffset + 1];
            positionTile[3 * threadIdx.x + 2] = current[sourceOffset + 2];
        }
        __syncthreads();
        const int limit = min(THREADS, n - base);
        if (active) {
            for (int k = 0; k < limit; ++k) {
                const double dx = positionTile[3 * k] - xi;
                const double dy = positionTile[3 * k + 1] - yi;
                const double dz = positionTile[3 * k + 2] - zi;
                const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        const int offset = STATE_WIDTH * i;
        const double vx = current[offset + 3] + DT * fx;
        const double vy = current[offset + 4] + DT * fy;
        const double vz = current[offset + 5] + DT * fz;
        next[offset] = xi + vx * DT;
        next[offset + 1] = yi + vy * DT;
        next[offset + 2] = zi + vz * DT;
        next[offset + 3] = vx;
        next[offset + 4] = vy;
        next[offset + 5] = vz;
    }
}

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Body& a = bodies[static_cast<size_t>(i)];
        energy += 0.5 * (a.vel.x * a.vel.x + a.vel.y * a.vel.y + a.vel.z * a.vel.z);
        for (long long j = i + 1; j < n; ++j) {
            const Body& b = bodies[static_cast<size_t>(j)];
            const double dx = b.pos.x - a.pos.x, dy = b.pos.y - a.pos.y, dz = b.pos.z - a.pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& b = bodies[static_cast<size_t>(i)];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) && std::isfinite(b.pos.z) &&
                            std::isfinite(b.vel.x) && std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 && std::abs(b.pos.z) <= 1e6 &&
                             std::abs(b.vel.x) <= 1e6 && std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        invalid |= !(finite && bounded);
    }
    if (invalid) std::printf("Validation failed: found non-finite or out-of-bounds body state\n");
    return invalid == 0;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Comm_free(&localComm); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Comm_free(&localComm); MPI_Finalize(); return 1; }
    }
    if (numBodies < 1 || numSteps < 0) { if (!rank) std::fprintf(stderr, "-n must be positive and -s non-negative\n"); MPI_Comm_free(&localComm); MPI_Finalize(); return 1; }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    // Use the node-local MPI rank, so every node gets the same GPU mapping.
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    if (!rank) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * STATE_WIDTH, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = numBodies / ranks + (r < numBodies % ranks); displs[r] = (numBodies / ranks) * r + (r < numBodies % ranks ? r : numBodies % ranks); }
    const int localCount = counts[rank], first = displs[rank];
    std::vector<int> doubleCounts(ranks), doubleDispls(ranks);
    for (int r = 0; r < ranks; ++r) { doubleCounts[r] = counts[r] * STATE_WIDTH; doubleDispls[r] = displs[r] * STATE_WIDTH; }

    double *deviceCurrent = nullptr, *deviceNext = nullptr;
    const size_t bytes = static_cast<size_t>(numBodies) * STATE_WIDTH * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceCurrent, bytes));
    CUDA_CHECK(cudaMalloc(&deviceNext, bytes));
    CUDA_CHECK(cudaMemcpy(deviceCurrent, bodies.data(), bytes, cudaMemcpyHostToDevice));

    if (!rank) { std::printf("N-Body Simulation (MPI ranks: %d, CUDA devices: %d, OpenMP threads: %d)\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", ranks, devices, omp_get_max_threads(), numBodies, numSteps, validate ? "enabled" : "disabled"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) advanceBodies<<<(localCount + THREADS - 1) / THREADS, THREADS>>>(deviceCurrent, deviceNext, numBodies, first, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(bodies.data() + static_cast<size_t>(first), deviceNext + static_cast<size_t>(first) * STATE_WIDTH,
                              static_cast<size_t>(localCount) * STATE_WIDTH * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), doubleCounts.data(), doubleDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceCurrent, bodies.data(), bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long long localMilliseconds = static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long simulationMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &simulationMilliseconds, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Simulation time: %lld ms\n", simulationMilliseconds);

    if (!rank && printResults) {
        std::vector<double> data; data.reserve(static_cast<size_t>(numBodies) * STATE_WIDTH);
        for (const Body& b : bodies) { data.insert(data.end(), {b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z}); }
        print_results(data, "Bodies");
    }
    int result = 0;
    if (!rank && validate) { std::printf("Validating simulation results...\n"); if (validateSimulation(bodies)) std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies)); else { std::printf("Validation: FAILED\n"); result = 1; } }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(deviceCurrent)); CUDA_CHECK(cudaFree(deviceNext));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return result;
}
