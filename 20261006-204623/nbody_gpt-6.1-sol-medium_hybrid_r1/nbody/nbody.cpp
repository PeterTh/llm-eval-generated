#include <algorithm>
#include <climits>
#include <exception>
#include <utility>
#include <cuda_runtime.h>
#include <mpi.h>
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

// Each rank owns a contiguous range of targets. Source positions are replicated,
// but velocities never leave their owning GPU during the simulation.
static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void checkMPI(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// RAII also releases allocations on the help/error and validation paths.
struct DeviceVectors {
    Vec3* data = nullptr;
    explicit DeviceVectors(int count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data),
                             std::max(count, 1) * sizeof(Vec3)), "cudaMalloc");
    }
    ~DeviceVectors() { cudaFree(data); }
    DeviceVectors(const DeviceVectors&) = delete;
    DeviceVectors& operator=(const DeviceVectors&) = delete;
};

struct HostVectors {
    Vec3* data = nullptr;
    explicit HostVectors(int count) {
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&data),
                                 std::max(count, 1) * sizeof(Vec3)), "cudaMallocHost");
    }
    ~HostVectors() { cudaFreeHost(data); }
    HostVectors(const HostVectors&) = delete;
    HostVectors& operator=(const HostVectors&) = delete;
};

constexpr int TILE = 128;

__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ nextPositions,
                              int n, int offset, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    double px = 0, py = 0, pz = 0;
    if (active) {
        const Vec3 p = positions[offset + local];
        px = p.x; py = p.y; pz = p.z;
    }
    double fx = 0, fy = 0, fz = 0;
    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            const Vec3 p = positions[j];
            sx[threadIdx.x] = p.x;
            sy[threadIdx.x] = p.y;
            sz[threadIdx.x] = p.z;
        }
        __syncthreads();
        if (active) {
            const int length = min(TILE, n - base);
            // Keep the original ascending source order and precise double sqrt.
            // FMA contraction is disabled by the build to retain its arithmetic.
            for (int k = 0; k < length; ++k) {
                const double dx = sx[k] - px;
                const double dy = sy[k] - py;
                const double dz = sz[k] - pz;
                const double distance = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inverse = 1.0 / sqrt(distance);
                const double inverse3 = inverse * inverse * inverse;
                fx += dx * inverse3;
                fy += dy * inverse3;
                fz += dz * inverse3;
            }
        }
        __syncthreads();
    }
    if (active) {
        Vec3 v = velocities[local];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        velocities[local] = v;
        // All forces read the old position buffer; integration writes another.
        nextPositions[local].x = px + v.x * DT;
        nextPositions[local].y = py + v.y * DT;
        nextPositions[local].z = pz + v.z * DT;
    }
}

// The optional O(N^2) energy diagnostic is distributed as well, so validation
// does not introduce a serial all-pairs computation on the root process.
__global__ void energyBodies(const Vec3* __restrict__ positions,
                             const Vec3* __restrict__ velocities,
                             Vec3* __restrict__ contributions,
                             int n, int offset, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int target = offset + local;
    const bool active = local < count;
    double px = 0, py = 0, pz = 0, energy = 0;
    if (active) {
        const Vec3 p = positions[target], v = velocities[local];
        px = p.x; py = p.y; pz = p.z;
        energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    // Skip source tiles that precede every target in this block.
    for (int base = (offset + blockIdx.x * blockDim.x) / TILE * TILE;
         base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            const Vec3 p = positions[j];
            sx[threadIdx.x] = p.x;
            sy[threadIdx.x] = p.y;
            sz[threadIdx.x] = p.z;
        }
        __syncthreads();
        if (active) {
            for (int k = max(0, target + 1 - base); k < min(TILE, n - base); ++k) {
                const double dx = sx[k] - px;
                const double dy = sy[k] - py;
                const double dz = sz[k] - pz;
                energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            }
        }
        __syncthreads();
    }
    if (active) {
        contributions[local].x = energy;
        contributions[local].y = 0;
        contributions[local].z = 0;
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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

int runSimulation(int argc, char** argv, int rank, int ranks) {
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
    // MPI counts/displacements use int, including the six-double body gather.
    if (numBodies < 0 || numBodies > INT_MAX / 6 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Invalid body or step count\n");
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPUs using node-local rank; CUDA_VISIBLE_DEVICES is respected.
    MPI_Comm node;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &node));
    int localRank, devices;
    checkMPI(MPI_Comm_rank(node, &localRank));
    checkMPI(MPI_Comm_free(&node));
    checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA GPU\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), "cudaSetDevice");

    std::vector<int> counts(ranks), offsets(ranks), bodyCounts(ranks), bodyOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int count = numBodies / ranks + (r < numBodies % ranks);
        const int offset = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
        counts[r] = 3 * count;
        offsets[r] = 3 * offset;
        bodyCounts[r] = 6 * count;
        bodyOffsets[r] = 6 * offset;
    }
    const int count = counts[rank] / 3;
    const int offset = offsets[rank] / 3;
    std::vector<Body> bodies(rank == 0 ? numBodies : 0), localBodies(count);
    HostVectors positions(numBodies), velocities(count);
    if (rank == 0) {
        // rand_r is stateful: retain the serial sequence exactly.
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) positions.data[i] = bodies[i].pos;
    }
    checkMPI(MPI_Bcast(positions.data, 3 * numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    checkMPI(MPI_Scatterv(bodies.data(), bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                          localBodies.data(), 6 * count, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) velocities.data[i] = localBodies[i].vel;
    // Release the root's initial full body array during the simulation.
    std::vector<Body>().swap(bodies);

    DeviceVectors devicePositions(numBodies), deviceVelocities(count), nextPositions(count);
    checkCuda(cudaMemcpy(devicePositions.data, positions.data, numBodies * sizeof(Vec3),
                         cudaMemcpyHostToDevice), "upload positions");
    checkCuda(cudaMemcpy(deviceVelocities.data, velocities.data, count * sizeof(Vec3),
                         cudaMemcpyHostToDevice), "upload velocities");
    checkMPI(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            advanceBodies<<<(count + TILE - 1) / TILE, TILE>>>(
                devicePositions.data, deviceVelocities.data, nextPositions.data,
                numBodies, offset, count);
            checkCuda(cudaGetLastError(), "advanceBodies launch");
        }
        if (ranks == 1) {
            // No host round trip or MPI collective is needed for a single GPU.
            std::swap(devicePositions.data, nextPositions.data);
        } else {
            checkCuda(cudaMemcpy(positions.data + offset, nextPositions.data,
                                 count * sizeof(Vec3), cudaMemcpyDeviceToHost),
                      "download updated positions");
            // Portable staging works with MPI implementations lacking CUDA support.
            checkMPI(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, positions.data,
                                    counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD));
            if (step + 1 < numSteps) {
                checkCuda(cudaMemcpy(devicePositions.data, positions.data,
                                     numBodies * sizeof(Vec3), cudaMemcpyHostToDevice),
                          "upload exchanged positions");
            }
        }
    }
    checkCuda(cudaDeviceSynchronize(), "finish simulation");
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0;
    checkMPI(MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(maximumElapsed * 1000));

    double finalEnergy = 0;
    if (validate) {
        if (ranks > 1 && numSteps > 0) {
            checkCuda(cudaMemcpy(devicePositions.data, positions.data, numBodies * sizeof(Vec3),
                                 cudaMemcpyHostToDevice), "upload final diagnostic positions");
        }
        // Reuse the now-unused integration buffer for per-target contributions.
        if (count > 0) {
            energyBodies<<<(count + TILE - 1) / TILE, TILE>>>(
                devicePositions.data, deviceVelocities.data, nextPositions.data,
                numBodies, offset, count);
            checkCuda(cudaGetLastError(), "energyBodies launch");
        }
        checkCuda(cudaMemcpy(velocities.data, nextPositions.data, count * sizeof(Vec3),
                             cudaMemcpyDeviceToHost), "download energy contributions");
        double localEnergy = 0;
        #pragma omp parallel for reduction(+:localEnergy) schedule(static)
        for (int i = 0; i < count; ++i) localEnergy += velocities.data[i].x;
        checkMPI(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
    }

    if (printResults || validate) {
        if (ranks == 1) {
            checkCuda(cudaMemcpy(positions.data, devicePositions.data, numBodies * sizeof(Vec3),
                                 cudaMemcpyDeviceToHost), "download final positions");
        }
        checkCuda(cudaMemcpy(velocities.data, deviceVelocities.data, count * sizeof(Vec3),
                             cudaMemcpyDeviceToHost), "download final velocities");
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            localBodies[i].pos = positions.data[offset + i];
            localBodies[i].vel = velocities.data[i];
        }
        if (rank == 0) bodies.resize(numBodies);
        checkMPI(MPI_Gatherv(localBodies.data(), 6 * count, MPI_DOUBLE, bodies.data(),
                             bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }
    int result = 0;
    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const auto& body = bodies[i];
                const size_t j = static_cast<size_t>(i) * 6;
                bodyData[j] = body.pos.x; bodyData[j + 1] = body.pos.y;
                bodyData[j + 2] = body.pos.z; bodyData[j + 3] = body.vel.x;
                bodyData[j + 4] = body.vel.y; bodyData[j + 5] = body.vel.z;
            }
            print_results(bodyData, "Bodies");
        }
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    checkMPI(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return result;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    checkMPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int result = 1;
    try {
        result = runSimulation(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkMPI(MPI_Finalize());
    return result;
}
