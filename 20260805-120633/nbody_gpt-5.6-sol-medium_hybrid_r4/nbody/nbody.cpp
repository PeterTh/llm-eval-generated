#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0,
                                      const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(std::is_trivially_copyable_v<Body> && sizeof(Body) == 6 * sizeof(double));

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error_ = (call);                                      \
        if (error_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(error_));                  \
            MPI_Abort(MPI_COMM_WORLD, 2);                                       \
        }                                                                       \
    } while (false)

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original generator and ordering so initial conditions are
    // independent of the number of MPI ranks and OpenMP threads.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void advanceBodies(Body* __restrict__ localBodies,
                              const Vec3* __restrict__ globalPositions,
                              Vec3* __restrict__ newLocalPositions,
                              int localCount, int numBodies) {
    __shared__ Vec3 tile[CUDA_BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    Vec3 pi;
    if (i < localCount) pi = localBodies[i].pos;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < numBodies; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < numBodies) tile[threadIdx.x] = globalPositions[j];
        __syncthreads();

        if (i < localCount) {
            const int tileCount = min(blockDim.x, numBodies - base);
#pragma unroll 8
            for (int k = 0; k < tileCount; ++k) {
                const double dx = tile[k].x - pi.x;
                const double dy = tile[k].y - pi.y;
                const double dz = tile[k].z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < localCount) {
        Body body = localBodies[i];
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        localBodies[i] = body;
        newLocalPositions[i] = body.pos;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double kinetic = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const Vec3 v = bodies[static_cast<size_t>(i)].vel;
        kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    double potential = 0.0;
#pragma omp parallel for reduction(+:potential) schedule(dynamic, 8)
    for (long long i = 0; i < n; ++i) {
        const Vec3 pi = bodies[static_cast<size_t>(i)].pos;
        for (long long j = i + 1; j < n; ++j) {
            const Vec3 pj = bodies[static_cast<size_t>(j)].pos;
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int valid = 1;
#pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& b = bodies[static_cast<size_t>(i)];
        const bool finite = std::isfinite(b.pos.x) && std::isfinite(b.pos.y) &&
                            std::isfinite(b.pos.z) && std::isfinite(b.vel.x) &&
                            std::isfinite(b.vel.y) && std::isfinite(b.vel.z);
        const bool bounded = std::abs(b.pos.x) <= 1e6 && std::abs(b.pos.y) <= 1e6 &&
                             std::abs(b.pos.z) <= 1e6 && std::abs(b.vel.x) <= 1e6 &&
                             std::abs(b.vel.y) <= 1e6 && std::abs(b.vel.z) <= 1e6;
        valid &= finite && bounded;
    }
    if (!valid) std::printf("Validation failed: non-finite or out-of-bounds body state\n");
    return valid != 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false, help = false;
    bool argsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else argsValid = false;
    }
    if (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / 6)
        argsValid = false;
    if (help || !argsValid) {
        if (rank == 0) {
            if (!argsValid) std::fprintf(stderr, "Invalid command-line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argsValid ? 0 : 1;
    }

    // Block distribution keeps each rank's bodies contiguous in global order.
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks ? 1 : 0);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int localCount = counts[rank];

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n", numBodies, numSteps);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Datatype bodyType, vecType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    std::vector<Body> allBodies(rank == 0 ? static_cast<size_t>(numBodies) : 0);
    if (rank == 0) randomizeBodies(allBodies);
    std::vector<Body> localBodies(static_cast<size_t>(localCount));
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, counts.data(), offsets.data(), bodyType,
                 localBodies.data(), localCount, bodyType, 0, MPI_COMM_WORLD);

    std::vector<Vec3> globalPositions(static_cast<size_t>(numBodies));
    std::vector<Vec3> localPositions(static_cast<size_t>(localCount));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) localPositions[static_cast<size_t>(i)] = localBodies[static_cast<size_t>(i)].pos;
    MPI_Allgatherv(localPositions.data(), localCount, vecType, globalPositions.data(),
                   counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);

    // Page-lock the MPI staging buffers.  This retains compatibility with
    // ordinary MPI implementations while making both transfer directions DMA
    // capable; CUDA-aware MPI can also recognize these buffers efficiently.
    if (localCount)
        CUDA_CHECK(cudaHostRegister(localPositions.data(), localPositions.size() * sizeof(Vec3),
                                    cudaHostRegisterPortable));
    if (numBodies)
        CUDA_CHECK(cudaHostRegister(globalPositions.data(), globalPositions.size() * sizeof(Vec3),
                                    cudaHostRegisterPortable));

    Body* dBodies = nullptr;
    Vec3 *dGlobalPositions = nullptr, *dLocalPositions = nullptr;
    CUDA_CHECK(cudaMalloc(&dBodies, std::max<size_t>(1, localBodies.size()) * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&dGlobalPositions, std::max<size_t>(1, globalPositions.size()) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&dLocalPositions, std::max<size_t>(1, localPositions.size()) * sizeof(Vec3)));
    if (localCount) CUDA_CHECK(cudaMemcpy(dBodies, localBodies.data(), localBodies.size() * sizeof(Body), cudaMemcpyHostToDevice));
    if (numBodies) CUDA_CHECK(cudaMemcpy(dGlobalPositions, globalPositions.data(), globalPositions.size() * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        if (localCount) {
            const int blocks = (localCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            advanceBodies<<<blocks, CUDA_BLOCK_SIZE>>>(dBodies, dGlobalPositions,
                                                       dLocalPositions, localCount, numBodies);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localPositions.data(), dLocalPositions,
                                  localPositions.size() * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(localPositions.data(), localCount, vecType, globalPositions.data(),
                       counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
        if (numBodies) CUDA_CHECK(cudaMemcpy(dGlobalPositions, globalPositions.data(),
                                              globalPositions.size() * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localCount) CUDA_CHECK(cudaMemcpy(localBodies.data(), dBodies, localBodies.size() * sizeof(Body), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localBodies.data(), localCount, bodyType,
                rank == 0 ? allBodies.data() : nullptr, counts.data(), offsets.data(), bodyType,
                0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dLocalPositions));
    CUDA_CHECK(cudaFree(dGlobalPositions));
    CUDA_CHECK(cudaFree(dBodies));
    if (numBodies) CUDA_CHECK(cudaHostUnregister(globalPositions.data()));
    if (localCount) CUDA_CHECK(cudaHostUnregister(localPositions.data()));
    MPI_Type_free(&vecType);
    MPI_Type_free(&bodyType);

    int returnCode = 0;
    if (rank == 0) {
        std::printf("Simulation time: %.3f ms\n", elapsed * 1000.0);
        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const Body& b = allBodies[static_cast<size_t>(i)];
                const size_t k = static_cast<size_t>(i) * 6;
                bodyData[k] = b.pos.x; bodyData[k + 1] = b.pos.y; bodyData[k + 2] = b.pos.z;
                bodyData[k + 3] = b.vel.x; bodyData[k + 4] = b.vel.y; bodyData[k + 5] = b.vel.z;
            }
            print_results(bodyData, "Bodies");
        }
        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(allBodies)) {
                std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(allBodies));
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
