#include <algorithm>
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

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

constexpr int TILE_SIZE = 128;

// Each thread owns a target body. Source positions are loaded once per block
// and reused from shared memory. All threads, including the partial last block,
// participate in both barriers. Updates go to a separate buffer so every force
// uses the same pre-integration positions.
__global__ void advanceBodies(const Vec3* __restrict__ positions,
                              Vec3* __restrict__ velocities,
                              Vec3* __restrict__ nextPositions,
                              int n, int offset, int count) {
    __shared__ double tileX[TILE_SIZE], tileY[TILE_SIZE], tileZ[TILE_SIZE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const Vec3 position = active ? positions[offset + local] : Vec3();
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE_SIZE) {
        const size_t source = base + threadIdx.x;
        if (source < static_cast<size_t>(n)) {
            const Vec3 p = positions[source];
            tileX[threadIdx.x] = p.x;
            tileY[threadIdx.x] = p.y;
            tileZ[threadIdx.x] = p.z;
        }
        __syncthreads();
        const int length = static_cast<int>(static_cast<size_t>(n) - base < TILE_SIZE
                                           ? static_cast<size_t>(n) - base : TILE_SIZE);
        if (active) {
            #pragma unroll 8
            for (int j = 0; j < length; ++j) {
                const double dx = tileX[j] - position.x;
                const double dy = tileY[j] - position.y;
                const double dz = tileZ[j] - position.z;
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
    if (active) {
        Vec3 velocity = velocities[local];
        velocity.x += DT * fx;
        velocity.y += DT * fy;
        velocity.z += DT * fz;
        velocities[local] = velocity;
        nextPositions[local] = Vec3(position.x + velocity.x * DT,
                                    position.y + velocity.y * DT,
                                    position.z + velocity.z * DT);
    }
}

double computeTotalEnergy(const Vec3* positions, const std::vector<Vec3>& velocities,
                          int n, int rank, int ranks) {
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static) if(velocities.size() >= 4096)
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3 v = velocities[i];
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Cyclic ownership balances the triangular potential-energy calculation.
    // Each pair is counted exactly once, across both ranks and host threads.
    #pragma omp parallel for reduction(+:energy) schedule(static, 1) if(n >= 256)
    for (size_t i = rank; i < static_cast<size_t>(n); i += ranks) {
        double potential = 0.0;
        for (size_t j = i + 1; j < static_cast<size_t>(n); ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
        energy += potential;
    }
    double total = 0.0;
    MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Vec3* positions, const std::vector<Vec3>& velocities) {
    int invalid = 0;
    #pragma omp parallel for reduction(|:invalid) schedule(static) if(velocities.size() >= 4096)
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Body body{positions[i], velocities[i]};
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            invalid |= 1;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            invalid |= 2;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            invalid |= 4;
        }
    }
    int globalInvalid = 0;
    MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        if (globalInvalid & 1) printf("Validation failed: found NaN or Inf value in body state\n");
        if (globalInvalid & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
        if (globalInvalid & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return globalInvalid == 0;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Finalize MPI on every normal exit, including command-line errors/help.
    struct FinalizeMPI {
        ~FinalizeMPI() { MPI_Finalize(); }
    } finalizeMPI;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative\n");
        return 1;
    }

    // Node-local ranks select distinct visible devices where possible. This
    // also supports launchers that expose just one assigned GPU to each rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localRanks);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "query CUDA devices");
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "select CUDA device");
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::min(8, std::max(1, omp_get_max_threads() / localRanks)));
    }
    MPI_Comm_free(&nodeComm);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Contiguous target ownership minimizes communication: only positions are
    // exchanged each step; velocities stay on their owning rank and device.
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], offset = offsets[rank];
    static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed");
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    const size_t positionBytes = std::max(size_t(1), static_cast<size_t>(numBodies)) * sizeof(Vec3);
    const size_t localBytes = std::max(size_t(1), static_cast<size_t>(count)) * sizeof(Vec3);
    Vec3* positions = nullptr;
    // Pinned staging works with standard MPI; CUDA-aware MPI is not required.
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&positions), positionBytes), "allocate pinned positions");
    std::vector<Vec3> velocities(count), allVelocities;
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies); // Keep the original rand_r sequence exactly.
        allVelocities.resize(numBodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = bodies[i].pos;
            allVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(positions, numBodies, vecType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(allVelocities.data(), counts.data(), offsets.data(), vecType,
                 velocities.data(), count, vecType, 0, MPI_COMM_WORLD);
    std::vector<Vec3>().swap(allVelocities);

    Vec3 *devicePositions = nullptr, *deviceVelocities = nullptr, *deviceNext = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&devicePositions), positionBytes), "allocate device positions");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceVelocities), localBytes), "allocate device velocities");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceNext), localBytes), "allocate next positions");
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create CUDA stream");
    cudaCheck(cudaMemcpyAsync(devicePositions, positions, static_cast<size_t>(numBodies) * sizeof(Vec3),
                              cudaMemcpyHostToDevice, stream), "upload positions");
    if (count > 0) {
        cudaCheck(cudaMemcpyAsync(deviceVelocities, velocities.data(), static_cast<size_t>(count) * sizeof(Vec3),
                                  cudaMemcpyHostToDevice, stream), "upload velocities");
    }
    cudaCheck(cudaStreamSynchronize(stream), "finish initialization");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            advanceBodies<<<(static_cast<size_t>(count) + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE, 0, stream>>>(
                devicePositions, deviceVelocities, deviceNext, numBodies, offset, count);
            cudaCheck(cudaGetLastError(), "launch force/integration kernel");
            cudaCheck(cudaMemcpyAsync(positions + offset, deviceNext, static_cast<size_t>(count) * sizeof(Vec3),
                                      cudaMemcpyDeviceToHost, stream), "download updated positions");
        }
        cudaCheck(cudaStreamSynchronize(stream), "finish timestep");
        // This collective is also the timestep dependency: no rank evaluates
        // the next force until all new positions have become available.
        MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(),
                       vecType, MPI_COMM_WORLD);
        if (step + 1 < numSteps) {
            if (ranks == 1) {
                std::swap(devicePositions, deviceNext);
            } else {
                cudaCheck(cudaMemcpyAsync(devicePositions, positions, static_cast<size_t>(numBodies) * sizeof(Vec3),
                                          cudaMemcpyHostToDevice, stream), "upload exchanged positions");
            }
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(duration * 1000.0));

    if ((printResults || validate) && count > 0) {
        cudaCheck(cudaMemcpyAsync(velocities.data(), deviceVelocities, static_cast<size_t>(count) * sizeof(Vec3),
                                  cudaMemcpyDeviceToHost, stream), "download final velocities");
    }
    cudaCheck(cudaStreamSynchronize(stream), "finish final downloads");
    cudaCheck(cudaFree(devicePositions), "free device positions");
    cudaCheck(cudaFree(deviceVelocities), "free device velocities");
    cudaCheck(cudaFree(deviceNext), "free next positions");
    cudaCheck(cudaStreamDestroy(stream), "destroy CUDA stream");
    
    // Print results for external validation
    if (printResults) {
        if (rank == 0) allVelocities.resize(numBodies);
        MPI_Gatherv(velocities.data(), count, vecType, allVelocities.data(), counts.data(),
                    offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
            #pragma omp parallel for schedule(static) if(numBodies >= 4096)
            for (int i = 0; i < numBodies; ++i) {
                const size_t base = static_cast<size_t>(i) * 6;
                bodyData[base] = positions[i].x;
                bodyData[base + 1] = positions[i].y;
                bodyData[base + 2] = positions[i].z;
                bodyData[base + 3] = allVelocities[i].x;
                bodyData[base + 4] = allVelocities[i].y;
                bodyData[base + 5] = allVelocities[i].z;
            }
            print_results(bodyData, "Bodies");
        }
    }
    
    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        if (validateSimulation(positions + offset, velocities)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(positions, velocities, numBodies, rank, ranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            result = 1;
        }
    }
    cudaCheck(cudaFreeHost(positions), "free pinned positions");
    MPI_Type_free(&vecType);
    return result;
}
