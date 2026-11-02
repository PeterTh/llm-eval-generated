#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

// All ranks participate in MPI; a CUDA failure on any one must terminate them all.
static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct Body {
    double3 pos;
    double3 vel;
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

// Positions are read from a complete snapshot and written to a separate buffer.
// The source order is the same as in the serial code, including self interaction.
__global__ void advanceBodies(const double3* __restrict__ positions,
                              double3* __restrict__ velocities,
                              double3* __restrict__ nextPositions,
                              int n, int offset, int localCount) {
    __shared__ double3 tile[BLOCK_SIZE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < localCount;
    double3 p = make_double3(0, 0, 0);
    if (active) p = positions[offset + local];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
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
        double3 v = velocities[local];
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        nextPositions[local] = make_double3(p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT);
    }
}

// Sum each unordered pair once. Each rank handles the same target bodies as
// in the simulation, and blocks reduce their partial energies on the device.
__global__ void energyBlocks(const double3* positions, const double3* velocities,
                             double2* blockSums, int n, int offset, int localCount) {
    __shared__ double3 tile[BLOCK_SIZE];
    __shared__ double kinetic[BLOCK_SIZE], potential[BLOCK_SIZE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = offset + local;
    const bool active = local < localCount;
    double3 p = make_double3(0, 0, 0);
    double ke = 0.0, pe = 0.0;
    if (active) {
        p = positions[i];
        const double3 v = velocities[local];
        ke = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
    for (int base = (offset + blockIdx.x * blockDim.x) / BLOCK_SIZE * BLOCK_SIZE;
         base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                if (base + k > i) {
                    const double dx = tile[k].x - p.x;
                    const double dy = tile[k].y - p.y;
                    const double dz = tile[k].z - p.z;
                    pe -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
            }
        }
        __syncthreads();
    }
    kinetic[threadIdx.x] = ke;
    potential[threadIdx.x] = pe;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            kinetic[threadIdx.x] += kinetic[threadIdx.x + stride];
            potential[threadIdx.x] += potential[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) blockSums[blockIdx.x] = make_double2(kinetic[0], potential[0]);
}

bool validateSimulation(const std::vector<Body>& bodies) {
    size_t firstBad = bodies.size();
    #pragma omp parallel for reduction(min:firstBad) if(bodies.size() > 1024)
    for (size_t i = 0; i < bodies.size(); ++i) {
        const auto& b = bodies[i];
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
            std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6 ||
            std::abs(b.vel.x) > 1e6 || std::abs(b.vel.y) > 1e6 || std::abs(b.vel.z) > 1e6)
            firstBad = std::min(firstBad, i);
    }
    if (firstBad == bodies.size()) return true;
    const auto& b = bodies[firstBad];
    if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
        !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z))
        std::printf("Validation failed: found NaN or Inf value in body state\n");
    else if (std::abs(b.pos.x) > 1e6 || std::abs(b.pos.y) > 1e6 || std::abs(b.pos.z) > 1e6)
        std::printf("Validation failed: body position exceeds reasonable bounds\n");
    else
        std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return false;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI counts and displacements below are expressed in doubles.
    if (n < 0 || n > INT_MAX / 3) {
        if (rank == 0) std::fprintf(stderr, "Invalid number of bodies\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int count = n / ranks + (r < n % ranks ? 1 : 0);
        const int offset = r * (n / ranks) + std::min(r, n % ranks);
        counts[r] = 3 * count;
        displs[r] = 3 * offset;
    }
    const int localCount = counts[rank] / 3;
    const int offset = displs[rank] / 3;
    static_assert(sizeof(double3) == 3 * sizeof(double), "double3 must be packed");
    const size_t globalBytes = static_cast<size_t>(n) * sizeof(double3);
    const size_t localBytes = static_cast<size_t>(localCount) * sizeof(double3);

    double3* hostPositions;
    CUDA_CHECK(cudaMallocHost(&hostPositions, std::max(globalBytes, sizeof(double3))));
    std::vector<Body> bodies(rank == 0 ? n : 0);
    std::vector<double3> allVelocities(rank == 0 ? n : 0);
    std::vector<double3> localVelocities(localCount);
    if (rank == 0) {
        randomizeBodies(bodies);
        #pragma omp parallel for if(n > 1024)
        for (int i = 0; i < n; ++i) {
            hostPositions[i] = bodies[i].pos;
            allVelocities[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(hostPositions, 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(allVelocities.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 localVelocities.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double3 *devicePositions, *deviceNext, *deviceVelocities;
    CUDA_CHECK(cudaMalloc(&devicePositions, std::max(globalBytes, sizeof(double3))));
    CUDA_CHECK(cudaMalloc(&deviceNext, std::max(localBytes, sizeof(double3))));
    CUDA_CHECK(cudaMalloc(&deviceVelocities, std::max(localBytes, sizeof(double3))));
    if (globalBytes) CUDA_CHECK(cudaMemcpy(devicePositions, hostPositions, globalBytes, cudaMemcpyHostToDevice));
    if (localBytes) CUDA_CHECK(cudaMemcpy(deviceVelocities, localVelocities.data(), localBytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (localCount) {
            advanceBodies<<<(localCount + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                devicePositions, deviceVelocities, deviceNext, n, offset, localCount);
            CUDA_CHECK(cudaGetLastError());
        }
        if (step + 1 < steps) {
            if (ranks == 1) {
                std::swap(devicePositions, deviceNext);
            } else {
                if (localBytes) CUDA_CHECK(cudaMemcpy(hostPositions + offset, deviceNext, localBytes, cudaMemcpyDeviceToHost));
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostPositions,
                               counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                if (globalBytes) CUDA_CHECK(cudaMemcpy(devicePositions, hostPositions, globalBytes, cudaMemcpyHostToDevice));
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed));

    if (validate || printResults) {
        if (steps > 0 && localBytes)
            CUDA_CHECK(cudaMemcpy(hostPositions + offset, deviceNext, localBytes, cudaMemcpyDeviceToHost));
        if (localBytes)
            CUDA_CHECK(cudaMemcpy(localVelocities.data(), deviceVelocities, localBytes, cudaMemcpyDeviceToHost));
        if (steps > 0) {
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hostPositions + offset, counts[rank], MPI_DOUBLE,
                        hostPositions, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Gatherv(localVelocities.data(), counts[rank], MPI_DOUBLE, allVelocities.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            #pragma omp parallel for if(n > 1024)
            for (int i = 0; i < n; ++i) {
                bodies[i].pos = hostPositions[i];
                bodies[i].vel = allVelocities[i];
            }
            if (printResults) {
                std::vector<double> bodyData(static_cast<size_t>(n) * 6);
                #pragma omp parallel for if(n > 1024)
                for (int i = 0; i < n; ++i) {
                    const Body& b = bodies[i];
                    const size_t k = static_cast<size_t>(i) * 6;
                    bodyData[k] = b.pos.x; bodyData[k + 1] = b.pos.y; bodyData[k + 2] = b.pos.z;
                    bodyData[k + 3] = b.vel.x; bodyData[k + 4] = b.vel.y; bodyData[k + 5] = b.vel.z;
                }
                print_results(bodyData, "Bodies");
            }
        }
    }
    int result = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating simulation results...\n");
            if (!validateSimulation(bodies)) result = 1;
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!result) {
            // The final positions were gathered only to rank zero above.
            if (steps > 0) MPI_Bcast(hostPositions, 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (globalBytes) CUDA_CHECK(cudaMemcpy(devicePositions, hostPositions, globalBytes, cudaMemcpyHostToDevice));
            const int blocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<double2> sums(blocks);
            if (blocks) {
                double2* deviceSums;
                CUDA_CHECK(cudaMalloc(&deviceSums, static_cast<size_t>(blocks) * sizeof(double2)));
                energyBlocks<<<blocks, BLOCK_SIZE>>>(devicePositions, deviceVelocities, deviceSums, n, offset, localCount);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(sums.data(), deviceSums, static_cast<size_t>(blocks) * sizeof(double2), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(deviceSums));
            }
            double kinetic = 0.0, potential = 0.0;
            #pragma omp parallel for reduction(+:kinetic,potential) if(blocks > 128)
            for (int i = 0; i < blocks; ++i) {
                kinetic += sums[i].x;
                potential += sums[i].y;
            }
            double localEnergy[2] = {kinetic, potential}, totalEnergy[2] = {0.0, 0.0};
            MPI_Reduce(localEnergy, totalEnergy, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
                std::printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy[0] + totalEnergy[1]);
        } else if (rank == 0) {
            std::printf("Validation: FAILED\n");
        }
    }
    CUDA_CHECK(cudaFree(devicePositions));
    CUDA_CHECK(cudaFree(deviceNext));
    CUDA_CHECK(cudaFree(deviceVelocities));
    CUDA_CHECK(cudaFreeHost(hostPositions));
    MPI_Finalize();
    return result;
}
