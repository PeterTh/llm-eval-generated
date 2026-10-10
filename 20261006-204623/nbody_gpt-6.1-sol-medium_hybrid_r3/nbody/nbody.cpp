#include <algorithm>
#include <exception>
#include <type_traits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

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

// All CUDA failures terminate the communicator so peers cannot hang in collectives.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

constexpr int TILE = 128;

// Each thread owns one target. Shared-memory tiles reuse the source positions
// across all targets in the block. Source indices are accumulated in their
// original order, including the softened self interaction.
template<bool Energy>
__global__ void evaluate(const Vec3* __restrict__ positions,
                         Vec3* __restrict__ velocities,
                         double* __restrict__ energies,
                         int n, int first, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const size_t local = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < static_cast<size_t>(count);
    const size_t target = size_t(first) + local;
    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        px = positions[target].x;
        py = positions[target].y;
        pz = positions[target].z;
    }
    double fx = 0.0, fy = 0.0, fz = 0.0, potential = 0.0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE) {
        for (int lane = threadIdx.x; lane < TILE; lane += blockDim.x) {
            const size_t source = base + lane;
            if (source < static_cast<size_t>(n)) {
                sx[lane] = positions[source].x;
                sy[lane] = positions[source].y;
                sz[lane] = positions[source].z;
            }
        }
        __syncthreads();
        if (active) {
            const int length = static_cast<size_t>(n) - base < TILE
                ? static_cast<int>(static_cast<size_t>(n) - base) : TILE;
            #pragma unroll 8
            for (int j = 0; j < length; ++j) {
                const double dx = sx[j] - px;
                const double dy = sy[j] - py;
                const double dz = sz[j] - pz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                if constexpr (Energy) {
                    if (base + j != static_cast<size_t>(target))
                        potential -= 0.5 * invDist;
                } else {
                    const double invDist3 = invDist * invDist * invDist;
                    fx += dx * invDist3;
                    fy += dy * invDist3;
                    fz += dz * invDist3;
                }
            }
        }
        // Even inactive threads must reach both barriers for partial blocks.
        __syncthreads();
    }
    if (active) {
        if constexpr (Energy) {
            const Vec3 v = velocities[local];
            energies[local] = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z) + potential;
        } else {
            velocities[local].x += DT * fx;
            velocities[local].y += DT * fy;
            velocities[local].z += DT * fz;
        }
    }
}

// A separate kernel ensures every force used the same pre-integration snapshot.
__global__ void integrateBodies(Vec3* positions, const Vec3* velocities,
                                int first, int count) {
    const size_t local = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local < static_cast<size_t>(count)) {
        positions[first + local].x += velocities[local].x * DT;
        positions[first + local].y += velocities[local].y * DT;
        positions[first + local].z += velocities[local].z * DT;
    }
}

// Validate that simulation produces finite, reasonable values
// Return a bit mask so OpenMP threads and MPI ranks can combine failures.
int validateSimulation(const std::vector<Body>& bodies) {
    int errors = 0;
    #pragma omp parallel for reduction(|:errors) schedule(static)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z))
            errors |= 1;
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6)
            errors |= 2;
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6)
            errors |= 4;
    }
    return errors;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks finite values and bounds)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Respect scheduler CUDA_VISIBLE_DEVICES masks. With a shared mask, map
    // node-local ranks to distinct GPUs; with one visible GPU, select device 0.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank = 0, devices = 0;
    MPI_Comm_rank(node, &nodeRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = nodeRank % devices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    MPI_Comm_free(&node);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Counts are in whole Vec3/Body elements, avoiding MPI int overflow for
    // the three/six double components of each body.
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 6 * sizeof(double));
    static_assert(std::is_trivially_copyable<Body>::value);
    MPI_Datatype positionType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &positionType);
    MPI_Type_commit(&positionType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = (numBodies / ranks) * r + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    const size_t positionBytes = std::max(size_t(1), size_t(numBodies)) * sizeof(Vec3);
    const size_t localBytes = std::max(size_t(1), size_t(count)) * sizeof(Vec3);
    Vec3 *positions = nullptr, *velocities = nullptr;
    Vec3 *devicePositions = nullptr, *deviceVelocities = nullptr;
    CUDA_CHECK(cudaMallocHost(&positions, positionBytes));
    CUDA_CHECK(cudaMallocHost(&velocities, localBytes));
    CUDA_CHECK(cudaMalloc(&devicePositions, positionBytes));
    CUDA_CHECK(cudaMalloc(&deviceVelocities, localBytes));

    std::vector<Body> bodies(rank == 0 ? numBodies : 0), localBodies(count);
    if (rank == 0) {
        // Preserve the original rand_r stream exactly, independent of ranks.
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
    }
    MPI_Bcast(positions, numBodies, positionType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) velocities[i] = localBodies[i].vel;
    CUDA_CHECK(cudaMemcpyAsync(devicePositions, positions, size_t(numBodies) * sizeof(Vec3),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceVelocities, velocities, size_t(count) * sizeof(Vec3),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Portable pinned-host staging works with MPI installations that do not
    // support CUDA device pointers. Only positions cross the network.
    auto exchangePositions = [&]() {
        CUDA_CHECK(cudaMemcpyAsync(positions + first, devicePositions + first,
                                   size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, positions, counts.data(),
                       offsets.data(), positionType, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(devicePositions, positions, size_t(numBodies) * sizeof(Vec3),
                                   cudaMemcpyHostToDevice, stream));
    };
    // Smaller target blocks spread a rank's shrinking workload across more
    // SMs during strong scaling, while retaining 128-source shared tiles.
    const int threads = count >= properties.multiProcessorCount * TILE / 2 ? TILE : TILE / 2;
    const unsigned blocks = (static_cast<unsigned>(count) + threads - 1) / threads;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        // With one rank the full simulation stays on the GPU. The final
        // snapshot needs no exchange unless energy validation requests it.
        if (step > 0 && ranks > 1) exchangePositions();
        if (count > 0) {
            evaluate<false><<<blocks, threads, 0, stream>>>(devicePositions, deviceVelocities,
                                                        nullptr, numBodies, first, count);
            CUDA_CHECK(cudaGetLastError());
            integrateBodies<<<blocks, threads, 0, stream>>>(devicePositions, deviceVelocities, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(duration * 1000.0));

    int result = 0;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpyAsync(positions + first, devicePositions + first, size_t(count) * sizeof(Vec3),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(velocities, deviceVelocities, size_t(count) * sizeof(Vec3),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            localBodies[i].pos = positions[first + i];
            localBodies[i].vel = velocities[i];
        }
    }
    if (printResults) {
        MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(), counts.data(),
                    offsets.data(), bodyType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const size_t k = size_t(i) * 6;
                bodyData[k] = bodies[i].pos.x;
                bodyData[k + 1] = bodies[i].pos.y;
                bodyData[k + 2] = bodies[i].pos.z;
                bodyData[k + 3] = bodies[i].vel.x;
                bodyData[k + 4] = bodies[i].vel.y;
                bodyData[k + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }
    }
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        const int localErrors = validateSimulation(localBodies);
        int errors = 0;
        MPI_Allreduce(&localErrors, &errors, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
        if (errors) {
            result = 1;
            if (rank == 0) {
                if (errors & 1) printf("Validation failed: found NaN or Inf value in body state\n");
                if (errors & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
                if (errors & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
                printf("Validation: FAILED\n");
            }
        } else {
            if (ranks > 1 && numSteps > 0) exchangePositions();
            double* deviceEnergy = nullptr;
            CUDA_CHECK(cudaMalloc(&deviceEnergy, std::max(size_t(1), size_t(count)) * sizeof(double)));
            std::vector<double> energies(count);
            if (count > 0) {
                evaluate<true><<<blocks, threads, 0, stream>>>(devicePositions, deviceVelocities,
                                                          deviceEnergy, numBodies, first, count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(energies.data(), deviceEnergy, size_t(count) * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double localEnergy = 0.0, finalEnergy = 0.0;
            #pragma omp parallel for reduction(+:localEnergy) schedule(static)
            for (int i = 0; i < count; ++i) localEnergy += energies[i];
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            CUDA_CHECK(cudaFree(deviceEnergy));
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }
    CUDA_CHECK(cudaFree(devicePositions));
    CUDA_CHECK(cudaFree(deviceVelocities));
    CUDA_CHECK(cudaFreeHost(positions));
    CUDA_CHECK(cudaFreeHost(velocities));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&positionType);
    MPI_Type_free(&bodyType);
    return result;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Avoid a full-node OpenMP team for every GPU rank by default. Explicit
    // OMP_NUM_THREADS settings always take precedence.
    if (!getenv("OMP_NUM_THREADS")) {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRanks = 1;
        MPI_Comm_size(node, &localRanks);
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localRanks)));
        MPI_Comm_free(&node);
    }
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
