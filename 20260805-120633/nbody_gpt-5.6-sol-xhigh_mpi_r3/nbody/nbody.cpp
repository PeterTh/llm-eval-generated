#include <mpi.h>

#include <algorithm>
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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(std::is_trivially_copyable_v<Body>);
static_assert(sizeof(Body) == 6 * sizeof(double));

// Structure-of-arrays storage gives the force kernel contiguous, non-aliasing
// streams while only retaining the rank's owned bodies.
struct LocalBodies {
    explicit LocalBodies(const size_t size)
        : posX(size), posY(size), posZ(size), velX(size), velY(size), velZ(size) {}

    size_t size() const noexcept { return posX.size(); }

    std::vector<double> posX, posY, posZ;
    std::vector<double> velX, velY, velZ;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int& seed) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

std::vector<int> makeCounts(const int total, const int ranks) {
    std::vector<int> counts(ranks, total / ranks);
    for (int rank = 0; rank < total % ranks; ++rank) {
        ++counts[rank];
    }
    return counts;
}

std::vector<int> makeDisplacements(const std::vector<int>& counts) {
    std::vector<int> displacements(counts.size(), 0);
    for (size_t rank = 1; rank < counts.size(); ++rank) {
        displacements[rank] = displacements[rank - 1] + counts[rank - 1];
    }
    return displacements;
}

void unpackBodies(const std::vector<Body>& packed, LocalBodies& bodies) {
    for (size_t i = 0; i < packed.size(); ++i) {
        bodies.posX[i] = packed[i].pos.x;
        bodies.posY[i] = packed[i].pos.y;
        bodies.posZ[i] = packed[i].pos.z;
        bodies.velX[i] = packed[i].vel.x;
        bodies.velY[i] = packed[i].vel.y;
        bodies.velZ[i] = packed[i].vel.z;
    }
}

std::vector<Body> packBodies(const LocalBodies& bodies) {
    std::vector<Body> packed(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        packed[i].pos = Vec3(bodies.posX[i], bodies.posY[i], bodies.posZ[i]);
        packed[i].vel = Vec3(bodies.velX[i], bodies.velY[i], bodies.velZ[i]);
    }
    return packed;
}

// Apply one globally contiguous source range to all local target bodies. Target
// blocking keeps the six target/accumulator streams in cache, and the inner
// loop is SIMD-friendly without changing the source summation order of any
// body.
void accumulateForces(const LocalBodies& bodies, const double* source,
                      const int sourceCount, std::vector<double>& forceX,
                      std::vector<double>& forceY, std::vector<double>& forceZ) {
    constexpr size_t TARGET_BLOCK_SIZE = 256;
    const size_t localCount = bodies.size();
    const double* const sourceX = source;
    const double* const sourceY = source + sourceCount;
    const double* const sourceZ = source + 2 * sourceCount;

    const double* const __restrict targetX = bodies.posX.data();
    const double* const __restrict targetY = bodies.posY.data();
    const double* const __restrict targetZ = bodies.posZ.data();
    double* const __restrict resultX = forceX.data();
    double* const __restrict resultY = forceY.data();
    double* const __restrict resultZ = forceZ.data();

    for (size_t first = 0; first < localCount; first += TARGET_BLOCK_SIZE) {
        const size_t last = std::min(first + TARGET_BLOCK_SIZE, localCount);
        for (int j = 0; j < sourceCount; ++j) {
            const double sx = sourceX[j];
            const double sy = sourceY[j];
            const double sz = sourceZ[j];

#pragma omp simd
            for (size_t i = first; i < last; ++i) {
                const double dx = sx - targetX[i];
                const double dy = sy - targetY[i];
                const double dz = sz - targetZ[i];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                resultX[i] += dx * invDist3;
                resultY[i] += dy * invDist3;
                resultZ[i] += dz * invDist3;
            }
        }
    }
}

// Stream source-position blocks in global body order. The next nonblocking
// broadcast progresses while the current block is evaluated. This uses
// O(N/P) storage per rank instead of replicating the full system, and retaining
// global block order preserves the original force summation order.
void computeForces(LocalBodies& bodies, const std::vector<int>& counts,
                   const int rank, MPI_Comm communicator,
                   std::vector<double>& forceX, std::vector<double>& forceY,
                   std::vector<double>& forceZ, std::vector<double>& buffer0,
                   std::vector<double>& buffer1) {
    std::fill(forceX.begin(), forceX.end(), 0.0);
    std::fill(forceY.begin(), forceY.end(), 0.0);
    std::fill(forceZ.begin(), forceZ.end(), 0.0);

    std::vector<double>* buffers[2] = {&buffer0, &buffer1};
    int currentBuffer = 0;
    MPI_Request currentRequest = MPI_REQUEST_NULL;

    const auto beginBroadcast = [&](const int owner, const int bufferIndex,
                                    MPI_Request* request) {
        const int count = counts[owner];
        double* const data = buffers[bufferIndex]->data();
        if (rank == owner) {
            std::copy(bodies.posX.begin(), bodies.posX.end(), data);
            std::copy(bodies.posY.begin(), bodies.posY.end(), data + count);
            std::copy(bodies.posZ.begin(), bodies.posZ.end(), data + 2 * count);
        }
        MPI_Ibcast(data, 3 * count, MPI_DOUBLE, owner, communicator, request);
    };

    beginBroadcast(0, currentBuffer, &currentRequest);
    for (int owner = 0; owner < static_cast<int>(counts.size()); ++owner) {
        MPI_Wait(&currentRequest, MPI_STATUS_IGNORE);

        const int nextBuffer = currentBuffer ^ 1;
        MPI_Request nextRequest = MPI_REQUEST_NULL;
        if (owner + 1 < static_cast<int>(counts.size())) {
            beginBroadcast(owner + 1, nextBuffer, &nextRequest);
        }

        accumulateForces(bodies, buffers[currentBuffer]->data(), counts[owner],
                         forceX, forceY, forceZ);
        currentBuffer = nextBuffer;
        currentRequest = nextRequest;
    }

    // Keep the multiply and add as two rounded operations, matching the
    // original velocity update even on targets where contraction is enabled.
#pragma omp simd
    for (size_t i = 0; i < bodies.size(); ++i) {
        forceX[i] *= DT;
        forceY[i] *= DT;
        forceZ[i] *= DT;
    }
#pragma omp simd
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies.velX[i] += forceX[i];
        bodies.velY[i] += forceY[i];
        bodies.velZ[i] += forceZ[i];
    }
}

void integrateBodies(LocalBodies& bodies) {
    const size_t count = bodies.size();
#pragma omp simd
    for (size_t i = 0; i < count; ++i) {
        bodies.posX[i] += bodies.velX[i] * DT;
        bodies.posY[i] += bodies.velY[i] * DT;
        bodies.posZ[i] += bodies.velZ[i] * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double MAX_POSITION = 1e6;
        constexpr double MAX_VELOCITY = 1e6;
        if (std::abs(body.pos.x) > MAX_POSITION ||
            std::abs(body.pos.y) > MAX_POSITION ||
            std::abs(body.pos.z) > MAX_POSITION) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > MAX_VELOCITY ||
            std::abs(body.vel.y) > MAX_VELOCITY ||
            std::abs(body.vel.z) > MAX_VELOCITY) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
            break;
        }
    }

    // MPI collectives use int counts; each streamed position block contains
    // three doubles per body.
    if (numBodies < 0 || numBodies > std::numeric_limits<int>::max() / 3) {
        if (rank == 0) {
            std::printf("Number of bodies must be between 0 and %d\n",
                        std::numeric_limits<int>::max() / 3);
        }
        argumentsValid = false;
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Datatype bodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType);
    MPI_Type_commit(&bodyType);

    const std::vector<int> counts = makeCounts(numBodies, rankCount);
    const std::vector<int> displacements = makeDisplacements(counts);
    std::vector<Body> localPacked(counts[rank]);

    // Generate the same deterministic stream as the serial implementation,
    // but send each rank's block as soon as it is produced so rank zero also
    // uses O(N/P) initialization memory.
    constexpr int INITIAL_DATA_TAG = 1;
    if (rank == 0) {
        unsigned int seed = 42;
        randomizeBodies(localPacked, seed);
        for (int destination = 1; destination < rankCount; ++destination) {
            std::vector<Body> block(counts[destination]);
            randomizeBodies(block, seed);
            if (!block.empty()) {
                MPI_Send(block.data(), counts[destination], bodyType, destination,
                         INITIAL_DATA_TAG, MPI_COMM_WORLD);
            }
        }
    } else if (!localPacked.empty()) {
        MPI_Recv(localPacked.data(), counts[rank], bodyType, 0, INITIAL_DATA_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    LocalBodies bodies(localPacked.size());
    unpackBodies(localPacked, bodies);
    localPacked.clear();
    localPacked.shrink_to_fit();

    const int maximumBlock = counts.empty() ? 0 : counts.front();
    const size_t bufferSize = std::max<size_t>(1, 3ULL * maximumBlock);
    std::vector<double> sourceBuffer0(bufferSize);
    std::vector<double> sourceBuffer1(bufferSize);
    std::vector<double> forceX(bodies.size());
    std::vector<double> forceY(bodies.size());
    std::vector<double> forceZ(bodies.size());

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, counts, rank, MPI_COMM_WORLD, forceX, forceY, forceZ,
                      sourceBuffer0, sourceBuffer1);
        integrateBodies(bodies);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const auto durationMilliseconds = static_cast<long long>(duration * 1000.0);
        std::printf("Simulation time: %lld ms\n", durationMilliseconds);
    }

    std::vector<Body> gatheredBodies;
    if (printResults || validate) {
        localPacked = packBodies(bodies);
        if (rank == 0) {
            gatheredBodies.resize(numBodies);
        }
        MPI_Gatherv(localPacked.data(), counts[rank], bodyType, gatheredBodies.data(),
                    counts.data(), displacements.data(), bodyType, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : gatheredBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(gatheredBodies)) {
            const double finalEnergy = computeTotalEnergy(gatheredBodies);
            std::printf("Final energy: %.6f\n", finalEnergy);
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);
    MPI_Finalize();
    return exitCode;
}
