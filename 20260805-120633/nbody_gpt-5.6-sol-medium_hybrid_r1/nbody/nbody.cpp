#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must contain exactly six doubles");

[[noreturn]] void fail(const char* operation, const char* detail, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) fail(operation, cudaGetErrorString(status), rank);
}

void checkMpi(int status, const char* operation, int rank) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        message[length] = '\0';
        fail(operation, message, rank);
    }
}

// Generate the identical rand_r stream used by the original program.  The
// conversion into Body objects is parallel because it has no dependencies.
void randomizeBodies(Body* bodies, int count, unsigned int seed = 42) {
    std::vector<double> values(static_cast<size_t>(count) * 6);
    for (double& value : values)
        value = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        const double* v = values.data() + static_cast<size_t>(i) * 6;
        bodies[i] = {{v[0], v[1], v[2]}, {v[3], v[4], v[5]}};
    }
}

// Each block reuses a tile of source positions from shared memory.  A rank
// computes only its target bodies, while every rank holds the current global
// snapshot.  Force and integration are fused; positions are not changed until
// all forces for that target have been accumulated, preserving the original
// step semantics.
__global__ void advanceBodies(const Body* __restrict__ allBodies,
                              Body* __restrict__ localBodies,
                              int numBodies, int firstBody, int localCount) {
    __shared__ double sx[CUDA_BLOCK_SIZE];
    __shared__ double sy[CUDA_BLOCK_SIZE];
    __shared__ double sz[CUDA_BLOCK_SIZE];

    const int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIndex < localCount;
    const int globalIndex = firstBody + localIndex;

    double px = 0.0, py = 0.0, pz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (active) {
        const Body body = allBodies[globalIndex];
        px = body.pos.x; py = body.pos.y; pz = body.pos.z;
        vx = body.vel.x; vy = body.vel.y; vz = body.vel.z;
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int tile = 0; tile < numBodies; tile += CUDA_BLOCK_SIZE) {
        const int source = tile + threadIdx.x;
        if (source < numBodies) {
            sx[threadIdx.x] = allBodies[source].pos.x;
            sy[threadIdx.x] = allBodies[source].pos.y;
            sz[threadIdx.x] = allBodies[source].pos.z;
        }
        __syncthreads();

        const int tileSize = min(CUDA_BLOCK_SIZE, numBodies - tile);
        if (active) {
#pragma unroll 8
            for (int j = 0; j < tileSize; ++j) {
                const double dx = sx[j] - px;
                const double dy = sy[j] - py;
                const double dz = sz[j] - pz;
                const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        vx += DT * fx; vy += DT * fy; vz += DT * fz;
        localBodies[localIndex] = {{px + vx * DT, py + vy * DT, pz + vz * DT},
                                   {vx, vy, vz}};
    }
}

double computeTotalEnergy(const Body* bodies, int count) {
    double energy = 0.0;
#pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < count; ++i) {
        const Vec3 v = bodies[i].vel;
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
#pragma omp parallel for reduction(+:energy) schedule(dynamic, 8)
    for (int i = 0; i < count; ++i) {
        double local = 0.0;
        for (int j = i + 1; j < count; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            local -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy += local;
    }
    return energy;
}

bool validateSimulation(const Body* bodies, int count) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < count; ++i) {
        const Body& b = bodies[i];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) &&
                            std::isfinite(b.pos.z) && std::isfinite(b.vel.x) &&
                            std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 &&
                             std::abs(b.pos.z) <= 1e6 && std::abs(b.vel.x) <= 1e6 &&
                             std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    if (!valid) std::printf("Validation failed: found non-finite or out-of-bounds body state\n");
    return valid != 0;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("  -n <num>  Number of bodies (default: 1024)\n");
    std::printf("  -s <num>  Number of simulation steps (default: 10)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size", rank);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_Init_thread", "insufficient thread support", rank);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false, help = false;
    int argumentsValid = 1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else argumentsValid = 0;
    }
    if (numBodies <= 0 || numSteps < 0) argumentsValid = 0;
    if (help || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return help ? 0 : 1;
    }

    // Select GPUs by node-local MPI rank so the same policy works across nodes.
    MPI_Comm localComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm), "MPI_Comm_split_type", rank);
    int localRank = 0;
    checkMpi(MPI_Comm_rank(localComm, &localRank), "local MPI_Comm_rank", rank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) fail("CUDA setup", "no CUDA devices found", rank);
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    checkCuda(cudaFree(nullptr), "CUDA context creation", rank);

    const int base = numBodies / ranks;
    const int remainder = numBodies % ranks;
    const int localCount = base + (rank < remainder ? 1 : 0);
    const int firstBody = rank * base + (rank < remainder ? rank : remainder);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        offsets[r] = r * base + (r < remainder ? r : remainder);
    }

    Body* hostBodies = nullptr;
    Body* hostLocal = nullptr;
    checkCuda(cudaMallocHost(&hostBodies, static_cast<size_t>(numBodies) * sizeof(Body)),
              "cudaMallocHost(global)", rank);
    const size_t localAllocation = static_cast<size_t>(localCount > 0 ? localCount : 1) * sizeof(Body);
    checkCuda(cudaMallocHost(&hostLocal, localAllocation),
              "cudaMallocHost(local)", rank);
    if (rank == 0) randomizeBodies(hostBodies, numBodies);

    MPI_Datatype mpiBody;
    checkMpi(MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody), "MPI_Type_contiguous", rank);
    checkMpi(MPI_Type_commit(&mpiBody), "MPI_Type_commit", rank);
    checkMpi(MPI_Bcast(hostBodies, numBodies, mpiBody, 0, MPI_COMM_WORLD), "MPI_Bcast", rank);

    Body* deviceBodies = nullptr;
    Body* deviceLocal = nullptr;
    checkCuda(cudaMalloc(&deviceBodies, static_cast<size_t>(numBodies) * sizeof(Body)),
              "cudaMalloc(global)", rank);
    checkCuda(cudaMalloc(&deviceLocal, localAllocation),
              "cudaMalloc(local)", rank);
    checkCuda(cudaMemcpy(deviceBodies, hostBodies, static_cast<size_t>(numBodies) * sizeof(Body),
                         cudaMemcpyHostToDevice), "initial cudaMemcpy", rank);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n", numBodies, numSteps);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier", rank);
    const auto start = std::chrono::high_resolution_clock::now();

    const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(deviceBodies, deviceLocal, numBodies,
                                                       firstBody, localCount);
            checkCuda(cudaGetLastError(), "advanceBodies launch", rank);
            checkCuda(cudaMemcpy(hostLocal, deviceLocal, static_cast<size_t>(localCount) * sizeof(Body),
                                 cudaMemcpyDeviceToHost), "result cudaMemcpy", rank);
        }
        checkMpi(MPI_Allgatherv(hostLocal, localCount, mpiBody, hostBodies, counts.data(),
                                offsets.data(), mpiBody, MPI_COMM_WORLD), "MPI_Allgatherv", rank);
        if (step + 1 < numSteps)
            checkCuda(cudaMemcpy(deviceBodies, hostBodies, static_cast<size_t>(numBodies) * sizeof(Body),
                                 cudaMemcpyHostToDevice), "state cudaMemcpy", rank);
    }

    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "final MPI_Barrier", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::printf("Simulation time: %lld ms\n", static_cast<long long>(ms));

        if (printResults) {
            std::vector<double> data(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const Body& b = hostBodies[i];
                double* out = data.data() + static_cast<size_t>(i) * 6;
                out[0] = b.pos.x; out[1] = b.pos.y; out[2] = b.pos.z;
                out[3] = b.vel.x; out[4] = b.vel.y; out[5] = b.vel.z;
            }
            print_results(data, "Bodies");
        }
    }

    int validationFailed = 0;
    if (validate && rank == 0) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(hostBodies, numBodies)) {
            std::printf("Final energy: %.6f\n", computeTotalEnergy(hostBodies, numBodies));
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            validationFailed = 1;
        }
    }
    checkMpi(MPI_Bcast(&validationFailed, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "validation MPI_Bcast", rank);

    cudaFree(deviceLocal);
    cudaFree(deviceBodies);
    cudaFreeHost(hostLocal);
    cudaFreeHost(hostBodies);
    MPI_Type_free(&mpiBody);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return validationFailed;
}
