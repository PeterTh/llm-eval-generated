#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 256;
// Small target blocks retain enough independent blocks per GPU when MPI
// partitions the targets, while reusing a larger shared-memory source tile.
constexpr int BLOCK = 64;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double));
static_assert(sizeof(Body) == 6 * sizeof(double));

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(message, length));
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
template<class T> struct HostBuffer {
    T* data = nullptr;
    explicit HostBuffer(size_t count) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~HostBuffer() { cudaFreeHost(data); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

// Each thread owns one target. All threads, including padding threads, take
// part in both barriers so partial tiles and ranks with few bodies are safe.
// Sources are visited in the original order; positions remain immutable until
// every target has completed the velocity update.
template<bool Energy>
__global__ void interact(const Vec3* __restrict__ positions,
                         Vec3* __restrict__ velocities,
                         Vec3* __restrict__ nextPositions,
                         double* __restrict__ energies,
                         int n, int begin, int count) {
    __shared__ Vec3 sources[TILE];
    const size_t local = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < size_t(count);
    const int target = active ? begin + static_cast<int>(local) : 0;
    const Vec3 p = active ? positions[target] : Vec3{0, 0, 0};
    double fx = 0, fy = 0, fz = 0, potential = 0;
    for (int base = 0; base < n;) {
        const int width = min(TILE, n - base);
        for (int j = threadIdx.x; j < width; j += blockDim.x)
            sources[j] = positions[base + j];
        __syncthreads();
        if (active) {
            #pragma unroll 4
            for (int j = 0; j < width; ++j) {
                if (!Energy || base + j > target) {
                    const double dx = sources[j].x - p.x;
                    const double dy = sources[j].y - p.y;
                    const double dz = sources[j].z - p.z;
                    const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    if (Energy) {
                        potential -= inv;
                    } else {
                        const double inv3 = inv * inv * inv;
                        fx += dx * inv3;
                        fy += dy * inv3;
                        fz += dz * inv3;
                    }
                }
            }
        }
        __syncthreads();
        base += width;
    }
    if (active) {
        Vec3 v = velocities[local];
        if (Energy) {
            energies[local] = potential + 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
        } else {
            v.x += DT * fx;
            v.y += DT * fy;
            v.z += DT * fz;
            velocities[local] = v;
            nextPositions[local] = Vec3{p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
        }
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int ranks) {
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            const bool bodiesOption = strcmp(argv[i], "-n") == 0;
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX) {
                if (rank == 0) fprintf(stderr, "Body and step counts must be nonnegative integers <= INT_MAX.\n");
                return 1;
            }
            (bodiesOption ? numBodies : numSteps) = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }

    // Honor scheduler CUDA visibility; otherwise spread node-local ranks over
    // the visible GPUs. Multiple ranks may share a GPU when oversubscribed.
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, devices;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    mpiCheck(MPI_Comm_free(&node));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("A CUDA GPU is required on every MPI rank");
    cudaCheck(cudaSetDevice(localRank % devices));

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], begin = offsets[rank];
    MPI_Datatype positionType, bodyType;
    mpiCheck(MPI_Type_contiguous(3, MPI_DOUBLE, &positionType));
    mpiCheck(MPI_Type_commit(&positionType));
    mpiCheck(MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType));
    mpiCheck(MPI_Type_commit(&bodyType));

    std::vector<Body> bodies(rank == 0 ? numBodies : 0), localBodies(count);
    HostBuffer<Vec3> hostPositions(numBodies), hostVelocities(count);
    DeviceBuffer<Vec3> positions(numBodies), nextPositions(ranks == 1 ? numBodies : count), velocities(count);
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               numBodies, numSteps, validate ? "enabled" : "disabled");
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static) if(numBodies > 1024)
        for (int i = 0; i < numBodies; ++i) hostPositions.data[i] = bodies[i].pos;
    }
    mpiCheck(MPI_Bcast(hostPositions.data, numBodies, positionType, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                         localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD));
    #pragma omp parallel for schedule(static) if(count > 1024)
    for (int i = 0; i < count; ++i) hostVelocities.data[i] = localBodies[i].vel;
    cudaCheck(cudaMemcpy(positions.data, hostPositions.data, size_t(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(velocities.data, hostVelocities.data, size_t(count) * sizeof(Vec3), cudaMemcpyHostToDevice));
    // Resolve CUDA lazy module loading before timing the simulation.
    cudaFuncAttributes attributes;
    cudaCheck(cudaFuncGetAttributes(&attributes, interact<false>));
    cudaCheck(cudaDeviceSynchronize());
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            interact<false><<< (size_t(count) + BLOCK - 1) / BLOCK, BLOCK >>>
                (positions.data, velocities.data, nextPositions.data, nullptr, numBodies, begin, count);
            cudaCheck(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(positions.data, nextPositions.data);
        } else {
            // Pinned staging supports MPI implementations without CUDA-aware
            // transport. Only positions are communicated; velocities stay local.
            cudaCheck(cudaMemcpy(hostPositions.data + begin, nextPositions.data,
                                 size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost));
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, positionType, hostPositions.data,
                                   counts.data(), offsets.data(), positionType, MPI_COMM_WORLD));
            cudaCheck(cudaMemcpy(positions.data, hostPositions.data,
                                 size_t(numBodies) * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0;
    mpiCheck(MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(maximumElapsed * 1000));

    int failed = 0;
    if (validate || printResults) {
        cudaCheck(cudaMemcpy(hostPositions.data + begin, positions.data + begin,
                             size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hostVelocities.data, velocities.data,
                             size_t(count) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static) reduction(|:failed) if(count > 1024)
        for (int i = 0; i < count; ++i) {
            const Vec3 p = hostPositions.data[begin + i], v = hostVelocities.data[i];
            localBodies[i] = Body{p, v};
            if (validate && (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                             !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
                             std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6 ||
                             std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6)) failed = 1;
        }
    }
    if (printResults) {
        mpiCheck(MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(),
                            counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static) if(numBodies > 1024)
            for (int i = 0; i < numBodies; ++i) {
                const size_t k = size_t(i) * 6;
                bodyData[k] = bodies[i].pos.x; bodyData[k+1] = bodies[i].pos.y; bodyData[k+2] = bodies[i].pos.z;
                bodyData[k+3] = bodies[i].vel.x; bodyData[k+4] = bodies[i].vel.y; bodyData[k+5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }
    }
    if (validate) {
        mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
        double finalEnergy = 0;
        if (!failed) {
            DeviceBuffer<double> energies(count);
            HostBuffer<double> hostEnergies(count);
            if (count) {
                interact<true><<< (size_t(count) + BLOCK - 1) / BLOCK, BLOCK >>>
                    (positions.data, velocities.data, nullptr, energies.data, numBodies, begin, count);
                cudaCheck(cudaGetLastError());
            }
            cudaCheck(cudaMemcpy(hostEnergies.data, energies.data, size_t(count) * sizeof(double), cudaMemcpyDeviceToHost));
            double localEnergy = 0;
            #pragma omp parallel for schedule(static) reduction(+:localEnergy) if(count > 1024)
            for (int i = 0; i < count; ++i) localEnergy += hostEnergies.data[i];
            mpiCheck(MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
        }
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (!failed) printf("Final energy: %.6f\n", finalEnergy);
            else printf("Validation failed: found nonfinite values or unreasonable body state\n");
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    mpiCheck(MPI_Type_free(&positionType));
    mpiCheck(MPI_Type_free(&bodyType));
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int result;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
