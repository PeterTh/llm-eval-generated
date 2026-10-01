#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void checkCuda(cudaError_t status, int rank, const char* action) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", rank, action, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
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

// Each thread owns one body. Shared memory broadcasts each position tile to
// all threads in the block, while the interaction sum retains the original order.
__global__ void advanceBodies(Body* __restrict__ local, Vec3* __restrict__ localPos,
                              const Vec3* __restrict__ allPos, int localCount, int n) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Body body;
    if (active) body = local[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int jload = base + threadIdx.x;
        if (jload < n) tile[threadIdx.x] = allPos[jload];
        __syncthreads();
        const int limit = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int j = 0; j < limit; ++j) {
                const double dx = tile[j].x - body.pos.x;
                const double dy = tile[j].y - body.pos.y;
                const double dz = tile[j].z - body.pos.z;
                const double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inv = 1.0 / sqrt(d2);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        local[i] = body;
        localPos[i] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0, potential = 0.0;
    #pragma omp parallel for reduction(+:kinetic,potential) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& a = bodies[i];
        kinetic += 0.5 * (a.vel.x * a.vel.x + a.vel.y * a.vel.y + a.vel.z * a.vel.z);
        for (int j = i + 1; j < n; ++j) {
            const Vec3& b = bodies[j].pos;
            const double dx = b.x - a.pos.x;
            const double dy = b.y - a.pos.y;
            const double dz = b.z - a.pos.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
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
    if (!valid) printf("Validation failed: non-finite or out-of-bounds body state\n");
    return valid != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    int parseResult = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { parseResult = 2; break; }
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); parseResult = 1; break; }
    }
    if (numBodies < 0 || numSteps < 0 || numBodies > INT_MAX / static_cast<int>(sizeof(Body))) {
        if (rank == 0) fprintf(stderr, "Invalid body count or step count\n");
        parseResult = 1;
    }
    if (parseResult) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseResult == 2 ? 0 : 1;
    }

    // Choose a different GPU for each local rank when enough devices are present.
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank, deviceCount = 0;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_free(&shared);
    checkCuda(cudaGetDeviceCount(&deviceCount), rank, "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), rank, "cudaSetDevice");

    std::vector<int> counts(ranks), offsets(ranks), bodyBytes(ranks), bodyOffsets(ranks);
    std::vector<int> posBytes(ranks), posOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = static_cast<int>((static_cast<long long>(numBodies) * r) / ranks);
        const int end = static_cast<int>((static_cast<long long>(numBodies) * (r + 1)) / ranks);
        counts[r] = end - offsets[r];
        bodyBytes[r] = counts[r] * static_cast<int>(sizeof(Body));
        bodyOffsets[r] = offsets[r] * static_cast<int>(sizeof(Body));
        posBytes[r] = counts[r] * static_cast<int>(sizeof(Vec3));
        posOffsets[r] = offsets[r] * static_cast<int>(sizeof(Vec3));
    }
    const int localCount = counts[rank];
    std::vector<Body> full;
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
        full.resize(numBodies);
        randomizeBodies(full);
    }
    std::vector<Body> local(localCount);
    std::vector<Vec3> positions(numBodies), localPositions(localCount);
    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) positions[i] = full[i].pos;
    }
    MPI_Bcast(positions.data(), numBodies * static_cast<int>(sizeof(Vec3)), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full.data() : nullptr, bodyBytes.data(), bodyOffsets.data(), MPI_BYTE,
                 local.data(), bodyBytes[rank], MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) full.clear();

    Body* gpuLocal = nullptr;
    Vec3* gpuPositions = nullptr;
    Vec3* gpuLocalPositions = nullptr;
    checkCuda(cudaMalloc(&gpuLocal, static_cast<size_t>(localCount ? localCount : 1) * sizeof(Body)), rank, "cudaMalloc bodies");
    checkCuda(cudaMalloc(&gpuPositions, static_cast<size_t>(numBodies ? numBodies : 1) * sizeof(Vec3)), rank, "cudaMalloc positions");
    checkCuda(cudaMalloc(&gpuLocalPositions, static_cast<size_t>(localCount ? localCount : 1) * sizeof(Vec3)), rank, "cudaMalloc local positions");
    checkCuda(cudaMemcpy(gpuLocal, local.data(), static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyHostToDevice), rank, "copy bodies to GPU");
    checkCuda(cudaMemcpy(gpuPositions, positions.data(), static_cast<size_t>(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice), rank, "copy positions to GPU");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            advanceBodies<<<(localCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(gpuLocal, gpuLocalPositions, gpuPositions, localCount, numBodies);
            checkCuda(cudaGetLastError(), rank, "launch advanceBodies");
        }
        // The next step must see the new positions of every body.
        if (step + 1 < numSteps) {
            checkCuda(cudaMemcpy(localPositions.data(), gpuLocalPositions, static_cast<size_t>(localCount) * sizeof(Vec3), cudaMemcpyDeviceToHost), rank, "copy positions from GPU");
            MPI_Allgatherv(localPositions.data(), posBytes[rank], MPI_BYTE, positions.data(),
                           posBytes.data(), posOffsets.data(), MPI_BYTE, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(gpuPositions, positions.data(), static_cast<size_t>(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice), rank, "update GPU positions");
        }
    }
    checkCuda(cudaDeviceSynchronize(), rank, "finish simulation");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDuration = static_cast<long>(duration.count());
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(local.data(), gpuLocal, static_cast<size_t>(localCount) * sizeof(Body), cudaMemcpyDeviceToHost), rank, "copy final bodies from GPU");
        if (rank == 0) full.resize(numBodies);
        MPI_Gatherv(local.data(), bodyBytes[rank], MPI_BYTE, rank == 0 ? full.data() : nullptr,
                    bodyBytes.data(), bodyOffsets.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const Body& body : full) {
                bodyData.push_back(body.pos.x); bodyData.push_back(body.pos.y); bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x); bodyData.push_back(body.vel.y); bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(full)) {
                printf("Final energy: %.6f\n", computeTotalEnergy(full));
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    checkCuda(cudaFree(gpuLocal), rank, "cudaFree bodies");
    checkCuda(cudaFree(gpuPositions), rank, "cudaFree positions");
    checkCuda(cudaFree(gpuLocalPositions), rank, "cudaFree local positions");
    MPI_Finalize();
    return result;
}
