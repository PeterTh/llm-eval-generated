#include <algorithm>
#include <array>
#include <climits>
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
constexpr int CUDA_BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must remain tightly packed for MPI byte collectives");

[[noreturn]] void abortWithMessage(const char* message, MPI_Comm communicator = MPI_COMM_WORLD) {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const char* call) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", call, length, error);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* call) {
    if (status == cudaSuccess) {
        return;
    }

    char message[512]{};
    std::snprintf(message, sizeof(message), "%s failed: %s", call, cudaGetErrorString(status));
    abortWithMessage(message);
}

#define CUDA_CHECK(call) checkCuda((call), #call)

int partitionBegin(const int total, const int part, const int parts) {
    return static_cast<int>((static_cast<long long>(total) * part) / parts);
}

struct PhysicalDevice {
    std::array<char, 16> uuid{};
    std::vector<int> ranks;
    std::vector<int> localDeviceIndices;
};

// Schedulers often give every local MPI rank a different CUDA_VISIBLE_DEVICES view, while a raw mpirun
// commonly exposes the full node to every rank.  Assign each physical GPU exactly once in either case.
std::vector<int> assignDevicesToRank(const MPI_Comm localCommunicator, const int localRank, const int localSize,
                                     const int visibleDevices) {
    constexpr int uuidBytes = 16;
    if (visibleDevices <= 0) {
        abortWithMessage("no CUDA devices are visible to this MPI rank");
    }

    std::vector<char> localUuids(static_cast<size_t>(visibleDevices) * uuidBytes);
    for (int device = 0; device < visibleDevices; ++device) {
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::memcpy(localUuids.data() + static_cast<size_t>(device) * uuidBytes, properties.uuid.bytes, uuidBytes);
    }

    std::vector<int> visibleCounts(localSize);
    checkMpi(MPI_Allgather(&visibleDevices, 1, MPI_INT, visibleCounts.data(), 1, MPI_INT, localCommunicator),
             "MPI_Allgather(visible CUDA devices)");

    std::vector<int> uuidCounts(localSize);
    std::vector<int> uuidDisplacements(localSize);
    int totalUuidBytes = 0;
    for (int peer = 0; peer < localSize; ++peer) {
        uuidCounts[peer] = visibleCounts[peer] * uuidBytes;
        uuidDisplacements[peer] = totalUuidBytes;
        totalUuidBytes += uuidCounts[peer];
    }
    std::vector<char> allUuids(totalUuidBytes);
    checkMpi(MPI_Allgatherv(localUuids.data(), static_cast<int>(localUuids.size()), MPI_BYTE, allUuids.data(),
                            uuidCounts.data(), uuidDisplacements.data(), MPI_BYTE, localCommunicator),
             "MPI_Allgatherv(visible CUDA UUIDs)");

    std::vector<PhysicalDevice> physicalDevices;
    for (int peer = 0; peer < localSize; ++peer) {
        for (int device = 0; device < visibleCounts[peer]; ++device) {
            const char* const uuid = allUuids.data() + uuidDisplacements[peer] + device * uuidBytes;
            int physicalIndex = 0;
            while (physicalIndex < static_cast<int>(physicalDevices.size()) &&
                   std::memcmp(physicalDevices[physicalIndex].uuid.data(), uuid, uuidBytes) != 0) {
                ++physicalIndex;
            }
            if (physicalIndex == static_cast<int>(physicalDevices.size())) {
                PhysicalDevice physical;
                std::memcpy(physical.uuid.data(), uuid, uuidBytes);
                physicalDevices.push_back(physical);
            }
            physicalDevices[physicalIndex].ranks.push_back(peer);
            physicalDevices[physicalIndex].localDeviceIndices.push_back(device);
        }
    }

    if (static_cast<int>(physicalDevices.size()) < localSize) {
        abortWithMessage("fewer physical CUDA devices than local MPI ranks; reduce ranks per node or request more GPUs");
    }

    const auto isVisibleTo = [&physicalDevices](const int physical, const int candidateRank) {
        const auto& ranks = physicalDevices[physical].ranks;
        return std::find(ranks.begin(), ranks.end(), candidateRank) != ranks.end();
    };

    // First find a matching that gives every local rank a GPU.  The remaining GPUs are balanced afterward.
    std::vector<int> owner(physicalDevices.size(), -1);
    const auto assignOne = [&owner, &physicalDevices, &isVisibleTo](auto&& self, const int candidateRank,
                                                                      std::vector<int>& seen) -> bool {
        for (int physical = 0; physical < static_cast<int>(physicalDevices.size()); ++physical) {
            if (seen[physical] != 0 || !isVisibleTo(physical, candidateRank)) {
                continue;
            }
            seen[physical] = 1;
            if (owner[physical] == -1 || self(self, owner[physical], seen)) {
                owner[physical] = candidateRank;
                return true;
            }
        }
        return false;
    };
    for (int candidateRank = 0; candidateRank < localSize; ++candidateRank) {
        std::vector<int> seen(physicalDevices.size());
        if (!assignOne(assignOne, candidateRank, seen)) {
            abortWithMessage("CUDA device visibility cannot provide a non-overlapping GPU to every local MPI rank");
        }
    }

    std::vector<int> assignedCounts(localSize);
    for (const int assignedRank : owner) {
        if (assignedRank >= 0) {
            ++assignedCounts[assignedRank];
        }
    }
    for (int physical = 0; physical < static_cast<int>(physicalDevices.size()); ++physical) {
        if (owner[physical] != -1) {
            continue;
        }
        int selectedRank = -1;
        for (const int candidateRank : physicalDevices[physical].ranks) {
            if (selectedRank == -1 || assignedCounts[candidateRank] < assignedCounts[selectedRank] ||
                (assignedCounts[candidateRank] == assignedCounts[selectedRank] && candidateRank < selectedRank)) {
                selectedRank = candidateRank;
            }
        }
        owner[physical] = selectedRank;
        ++assignedCounts[selectedRank];
    }

    std::vector<int> assignedDevices;
    for (int physical = 0; physical < static_cast<int>(physicalDevices.size()); ++physical) {
        if (owner[physical] != localRank) {
            continue;
        }
        const auto& ranks = physicalDevices[physical].ranks;
        const auto rankIterator = std::find(ranks.begin(), ranks.end(), localRank);
        const size_t visibilityIndex = static_cast<size_t>(rankIterator - ranks.begin());
        assignedDevices.push_back(physicalDevices[physical].localDeviceIndices[visibilityIndex]);
    }
    std::sort(assignedDevices.begin(), assignedDevices.end());
    return assignedDevices;
}

void randomizeBodies(Body* bodies, const int numBodies, unsigned int seed = 42) {
    // Keep the original rand_r stream and field order so the initial state is unchanged.
    for (int i = 0; i < numBodies; ++i) {
        Body& body = bodies[i];
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

// Each block advances independent target bodies and reuses a tile of source positions from shared memory.
// The inner loop preserves increasing j order for every target body, matching the original direct method.
__global__ void advanceBodies(Body* __restrict__ bodies, const int begin, const int count, const int numBodies) {
    extern __shared__ double sourcePositions[];
    double* const sourceX = sourcePositions;
    double* const sourceY = sourceX + blockDim.x;
    double* const sourceZ = sourceY + blockDim.x;

    const int lane = threadIdx.x;
    const int bodyIndex = begin + blockIdx.x * blockDim.x + lane;
    const bool active = bodyIndex < begin + count;

    double posX = 0.0;
    double posY = 0.0;
    double posZ = 0.0;
    double velX = 0.0;
    double velY = 0.0;
    double velZ = 0.0;
    double forceX = 0.0;
    double forceY = 0.0;
    double forceZ = 0.0;

    if (active) {
        const Body target = bodies[bodyIndex];
        posX = target.pos.x;
        posY = target.pos.y;
        posZ = target.pos.z;
        velX = target.vel.x;
        velY = target.vel.y;
        velZ = target.vel.z;
    }

    for (int tileBegin = 0; tileBegin < numBodies; tileBegin += blockDim.x) {
        const int sourceIndex = tileBegin + lane;
        if (sourceIndex < numBodies) {
            sourceX[lane] = bodies[sourceIndex].pos.x;
            sourceY[lane] = bodies[sourceIndex].pos.y;
            sourceZ[lane] = bodies[sourceIndex].pos.z;
        }
        __syncthreads();

        const int tileCount = min(static_cast<int>(blockDim.x), numBodies - tileBegin);
        if (active) {
            #pragma unroll 4
            for (int source = 0; source < tileCount; ++source) {
                const double dx = sourceX[source] - posX;
                const double dy = sourceY[source] - posY;
                const double dz = sourceZ[source] - posZ;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                forceX += dx * invDist3;
                forceY += dy * invDist3;
                forceZ += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        velX += DT * forceX;
        velY += DT * forceY;
        velZ += DT * forceZ;

        Body& target = bodies[bodyIndex];
        target.vel.x = velX;
        target.vel.y = velY;
        target.vel.z = velZ;
        target.pos.x = posX + velX * DT;
        target.pos.y = posY + velY * DT;
        target.pos.z = posZ + velZ * DT;
    }
}

struct DeviceWorker {
    int device = -1;
    int begin = 0;
    int count = 0;
    Body* bodies = nullptr;
    cudaStream_t stream = nullptr;
};

void initializeWorker(DeviceWorker& worker, Body* hostBodies, const int numBodies) {
    CUDA_CHECK(cudaSetDevice(worker.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&worker.stream, cudaStreamNonBlocking));
    if (numBodies > 0) {
        CUDA_CHECK(cudaMalloc(&worker.bodies, static_cast<size_t>(numBodies) * sizeof(Body)));
        CUDA_CHECK(cudaMemcpyAsync(worker.bodies, hostBodies, static_cast<size_t>(numBodies) * sizeof(Body),
                                   cudaMemcpyHostToDevice, worker.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(worker.stream));
}

void finalizeWorker(DeviceWorker& worker) {
    CUDA_CHECK(cudaSetDevice(worker.device));
    if (worker.bodies != nullptr) {
        CUDA_CHECK(cudaFree(worker.bodies));
    }
    if (worker.stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(worker.stream));
    }
}

bool validateSimulation(const Body* bodies, const int numBodies) {
    for (int i = 0; i < numBodies; ++i) {
        const Body& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        constexpr double maxPosition = 1e6;
        constexpr double maxVelocity = 1e6;
        if (std::abs(body.pos.x) > maxPosition || std::abs(body.pos.y) > maxPosition ||
            std::abs(body.pos.z) > maxPosition) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVelocity || std::abs(body.vel.y) > maxVelocity ||
            std::abs(body.vel.z) > maxVelocity) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

double computeTotalEnergy(const Body* bodies, const int numBodies) {
    double energy = 0.0;

    for (int i = 0; i < numBodies; ++i) {
        const Body& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    for (int i = 0; i < numBodies; ++i) {
        for (int j = i + 1; j < numBodies; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks state and reports final energy)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parsePositiveOrZero(const char* value, int& destination) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    destination = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    checkMpi(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel), "MPI_Init_thread");
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI implementation does not provide the required MPI_THREAD_FUNNELED support");
    }

    int rank = 0;
    int worldSize = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");

    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;
    int parseStatus = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                if (!parsePositiveOrZero(argv[++i], numBodies)) {
                    parseStatus = 1;
                }
            } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                if (!parsePositiveOrZero(argv[++i], numSteps)) {
                    parseStatus = 1;
                }
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
            } else {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
            }
        }
        if (static_cast<long long>(numBodies) * sizeof(Body) > INT_MAX) {
            std::fprintf(stderr, "Number of bodies is too large for MPI byte counts\n");
            parseStatus = 1;
        }
    }

    int options[5] = {parseStatus, numBodies, numSteps, validate, printResults};
    checkMpi(MPI_Bcast(options, 5, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast(options)");
    parseStatus = options[0];
    numBodies = options[1];
    numSteps = options[2];
    validate = options[3];
    printResults = options[4];
    if (parseStatus != 0) {
        checkMpi(MPI_Finalize(), "MPI_Finalize");
        return parseStatus == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type");
    int localRank = 0;
    int localSize = 1;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank), "MPI_Comm_rank(local)");
    checkMpi(MPI_Comm_size(localCommunicator, &localSize), "MPI_Comm_size(local)");

    int visibleDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visibleDevices));
    const std::vector<int> assignedDevices = assignDevicesToRank(localCommunicator, localRank, localSize, visibleDevices);
    const int workerCount = static_cast<int>(assignedDevices.size());
    if (workerCount <= 0) {
        abortWithMessage("no CUDA device was assigned to this MPI rank");
    }

    const int rankBegin = partitionBegin(numBodies, rank, worldSize);
    const int rankEnd = partitionBegin(numBodies, rank + 1, worldSize);
    const int rankCount = rankEnd - rankBegin;

    std::vector<int> receiveCounts(worldSize);
    std::vector<int> receiveDisplacements(worldSize);
    for (int sourceRank = 0; sourceRank < worldSize; ++sourceRank) {
        const int begin = partitionBegin(numBodies, sourceRank, worldSize);
        const int end = partitionBegin(numBodies, sourceRank + 1, worldSize);
        receiveCounts[sourceRank] = (end - begin) * static_cast<int>(sizeof(Body));
        receiveDisplacements[sourceRank] = begin * static_cast<int>(sizeof(Body));
    }

    Body* hostBodies = nullptr;
    if (numBodies > 0) {
        CUDA_CHECK(cudaHostAlloc(&hostBodies, static_cast<size_t>(numBodies) * sizeof(Body), cudaHostAllocPortable));
    }
    if (rank == 0 && numBodies > 0) {
        randomizeBodies(hostBodies, numBodies);
    }
    checkMpi(MPI_Bcast(hostBodies, numBodies * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD),
             "MPI_Bcast(initial bodies)");

    std::vector<DeviceWorker> workers(workerCount);
    for (int worker = 0; worker < workerCount; ++worker) {
        workers[worker].device = assignedDevices[worker];
        workers[worker].begin = rankBegin + partitionBegin(rankCount, worker, workerCount);
        const int end = rankBegin + partitionBegin(rankCount, worker + 1, workerCount);
        workers[worker].count = end - workers[worker].begin;
    }

    double localSeconds = 0.0;
    #pragma omp parallel num_threads(workerCount) shared(workers, hostBodies, numBodies, numSteps, receiveCounts, receiveDisplacements, localSeconds)
    {
        const int workerIndex = omp_get_thread_num();
        DeviceWorker& worker = workers[workerIndex];
        initializeWorker(worker, hostBodies, numBodies);
        #pragma omp barrier
        #pragma omp master
        {
            checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
            localSeconds = MPI_Wtime();
        }
        #pragma omp barrier

        for (int step = 0; step < numSteps; ++step) {
            if (worker.count > 0) {
                const int gridSize = (worker.count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
                const size_t sharedBytes = 3 * static_cast<size_t>(CUDA_BLOCK_SIZE) * sizeof(double);
                advanceBodies<<<gridSize, CUDA_BLOCK_SIZE, sharedBytes, worker.stream>>>(
                    worker.bodies, worker.begin, worker.count, numBodies);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(hostBodies + worker.begin, worker.bodies + worker.begin,
                                           static_cast<size_t>(worker.count) * sizeof(Body), cudaMemcpyDeviceToHost,
                                           worker.stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(worker.stream));
            #pragma omp barrier

            // MPI_IN_PLACE publishes the contiguous range produced by this rank without an extra host copy.
            #pragma omp master
            {
                checkMpi(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostBodies, receiveCounts.data(),
                                        receiveDisplacements.data(), MPI_BYTE, MPI_COMM_WORLD),
                         "MPI_Allgatherv(updated bodies)");
            }
            #pragma omp barrier

            // The next kernel uses the same stream, so it waits for this asynchronous refresh without a
            // host-side synchronize.  The final refresh is unnecessary because the host copy is authoritative.
            if (step + 1 < numSteps && numBodies > 0) {
                CUDA_CHECK(cudaMemcpyAsync(worker.bodies, hostBodies, static_cast<size_t>(numBodies) * sizeof(Body),
                                           cudaMemcpyHostToDevice, worker.stream));
            }
            #pragma omp barrier
        }

        #pragma omp master
        {
            localSeconds = MPI_Wtime() - localSeconds;
        }
        #pragma omp barrier
        finalizeWorker(worker);
    }

    // Report the slowest rank because it is the wall-clock completion time of the distributed simulation.
    double simulationSeconds = 0.0;
    checkMpi(MPI_Reduce(&localSeconds, &simulationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
             "MPI_Reduce(simulation time)");

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate != 0 ? "enabled" : "disabled");
        std::printf("Simulation time: %ld ms\n", static_cast<long>(simulationSeconds * 1000.0));

        if (printResults != 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                const Body& body = hostBodies[i];
                const size_t offset = static_cast<size_t>(i) * 6;
                bodyData[offset] = body.pos.x;
                bodyData[offset + 1] = body.pos.y;
                bodyData[offset + 2] = body.pos.z;
                bodyData[offset + 3] = body.vel.x;
                bodyData[offset + 4] = body.vel.y;
                bodyData[offset + 5] = body.vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate != 0) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(hostBodies, numBodies)) {
                std::printf("Final energy: %.6f\n", computeTotalEnergy(hostBodies, numBodies));
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                if (hostBodies != nullptr) {
                    CUDA_CHECK(cudaFreeHost(hostBodies));
                }
                checkMpi(MPI_Comm_free(&localCommunicator), "MPI_Comm_free");
                checkMpi(MPI_Finalize(), "MPI_Finalize");
                return EXIT_FAILURE;
            }
        }
    }

    if (hostBodies != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostBodies));
    }
    checkMpi(MPI_Comm_free(&localCommunicator), "MPI_Comm_free");
    checkMpi(MPI_Finalize(), "MPI_Finalize");
    return EXIT_SUCCESS;
}
