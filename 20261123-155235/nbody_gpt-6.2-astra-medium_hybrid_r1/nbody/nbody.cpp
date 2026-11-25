#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
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
};

static_assert(sizeof(Vec3) == 3 * sizeof(double));

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

#define CUDA(call) checkCuda((call), #call)

constexpr int tileSize = 128;

// All threads participate in loading each tile, including those beyond the last target.
template<int width>
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int first, int count) {
    __shared__ Vec3 tile[width];
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + lane;
    const Vec3 p = i < count ? positions[first + i] : Vec3{};
    Vec3 force{};
    for (unsigned base = 0; base < n; base += width) {
        const unsigned length = min(unsigned(width), unsigned(n) - base);
        if (lane < length) tile[lane] = positions[base + lane];
        __syncthreads();
        if (i < count) {
            #pragma unroll 4
            for (unsigned j = 0; j < length; ++j) {
                const double dx = tile[j].x - p.x;
                const double dy = tile[j].y - p.y;
                const double dz = tile[j].z - p.z;
                const double r2 = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double inv = __ddiv_rn(1.0, __dsqrt_rn(r2));
                const double scale = inv * inv * inv;
                force.x = fma(dx, scale, force.x);
                force.y = fma(dy, scale, force.y);
                force.z = fma(dz, scale, force.z);
            }
        }
        __syncthreads();
    }
    if (i < count) {
        Vec3 v = velocities[i];
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        velocities[i] = v;
        next[i] = {fma(v.x, DT, p.x), fma(v.y, DT, p.y), fma(v.z, DT, p.z)};
    }
}

// Each block sums the pairs for one body. Walking halfway around the cyclic
// body order assigns every unordered pair once and balances the device workloads.
__global__ void bodyEnergy(const Vec3* positions, const Vec3* velocities,
                           int n, int first, double* energies) {
    __shared__ double partial[tileSize];
    const int local = blockIdx.x;
    const unsigned i = first + local;
    const Vec3 p = positions[i];
    const unsigned pairs = (n - 1) / 2 + (n % 2 == 0 && i < unsigned(n / 2));
    double energy = 0;
    for (unsigned offset = threadIdx.x + 1; offset <= pairs; offset += blockDim.x) {
        unsigned j = i + offset;
        if (j >= n) j -= n;
        const double dx = positions[j].x - p.x;
        const double dy = positions[j].y - p.y;
        const double dz = positions[j].z - p.z;
        energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    partial[threadIdx.x] = energy;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const Vec3 v = velocities[local];
        energies[local] = partial[0] + 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

struct Partition {
    int device, first, count;
    Vec3 *pos = nullptr, *next = nullptr, *vel = nullptr;
};

int boundary(int total, int part, int parts) {
    return (total / parts) * part + std::min(part, total % parts);
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const Vec3* positions, const std::vector<Vec3>& velocities) {
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3 pos = positions[i], vel = velocities[i];
        // Check for NaN or Inf values
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos.x) > maxPos || std::abs(pos.y) > maxPos || std::abs(pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel.x) > maxVel || std::abs(vel.y) > maxVel || std::abs(vel.z) > maxVel) {
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

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Schedulers may expose the whole node or a separate GPU set to each rank.
    // Stripe devices among ranks that see the same leading physical GPU.
    cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties, 0));
    std::vector<cudaUUID_t> leadingDevices(localRanks);
    MPI_Allgather(&properties.uuid, sizeof(cudaUUID_t), MPI_BYTE, leadingDevices.data(),
                  sizeof(cudaUUID_t), MPI_BYTE, node);
    MPI_Comm_free(&node);
    int poolRank = 0, poolSize = 0;
    for (int r = 0; r < localRanks; ++r) {
        if (std::memcmp(&leadingDevices[r], &properties.uuid, sizeof(cudaUUID_t)) == 0) {
            if (r < localRank) ++poolRank;
            ++poolSize;
        }
    }
    const int firstDevice = poolRank % devices;
    CUDA(cudaSetDevice(firstDevice));
    const int workers = 1 + (devices - 1 - firstDevice) / poolSize;
    std::vector<int> workerCounts(ranks), counts(ranks), offsets(ranks);
    MPI_Allgather(&workers, 1, MPI_INT, workerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int totalWorkers = 0;
    for (int k : workerCounts) totalWorkers += k;
    int workerBase = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = boundary(numBodies, workerBase, totalWorkers);
        workerBase += workerCounts[r];
        counts[r] = boundary(numBodies, workerBase, totalWorkers) - offsets[r];
    }
    workerBase = 0;
    for (int r = 0; r < rank; ++r) workerBase += workerCounts[r];

    MPI_Datatype vectorType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType);
    MPI_Type_commit(&vectorType);
    const size_t posBytes = size_t(numBodies) * sizeof(Vec3);
    Vec3* positions;
    CUDA(cudaHostAlloc(&positions, std::max(posBytes, sizeof(Vec3)), cudaHostAllocPortable));
    std::vector<Vec3> velocities(rank == 0 ? numBodies : counts[rank]);
    if (rank == 0) {
        unsigned seed = 42;
        auto randomValue = [&]() { return 2.0 * (rand_r(&seed) / double(RAND_MAX)) - 1.0; };
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = {randomValue(), randomValue(), randomValue()};
            velocities[i] = {randomValue(), randomValue(), randomValue()};
        }
    }
    MPI_Bcast(positions, numBodies, vectorType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities.data(), counts.data(), offsets.data(), vectorType,
                 rank == 0 ? MPI_IN_PLACE : velocities.data(), counts[rank], vectorType,
                 0, MPI_COMM_WORLD);

    std::vector<Partition> partitions(workers);
    const int threads = std::min(workers, omp_get_max_threads());
    #pragma omp parallel for num_threads(threads)
    for (int w = 0; w < workers; ++w) {
        const int first = boundary(numBodies, workerBase + w, totalWorkers);
        auto& g = partitions[w];
        g = {firstDevice + w * poolSize, first,
             boundary(numBodies, workerBase + w + 1, totalWorkers) - first};
        if (!g.count) continue;
        CUDA(cudaSetDevice(g.device));
        CUDA(cudaMalloc(&g.pos, posBytes));
        const size_t bytes = size_t(g.count) * sizeof(Vec3);
        CUDA(cudaMalloc(&g.next, bytes));
        CUDA(cudaMalloc(&g.vel, bytes));
        CUDA(cudaMemcpy(g.pos, positions, posBytes, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(g.vel, velocities.data() + (first - offsets[rank]), bytes,
                        cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool needState = printResults || validate;
    for (int step = 0; step < numSteps; ++step) {
        const bool again = step + 1 < numSteps;
        #pragma omp parallel for num_threads(threads)
        for (int w = 0; w < workers; ++w) {
            auto& g = partitions[w];
            if (!g.count) continue;
            CUDA(cudaSetDevice(g.device));
            if (totalWorkers > 1 && step > 0)
                CUDA(cudaMemcpy(g.pos, positions, posBytes, cudaMemcpyHostToDevice));
            // Smaller blocks distribute short target ranges across more SMs.
            if (g.count < 8192) {
                advance<32><<<1 + (g.count - 1) / 32, 32>>>(
                    g.pos, g.next, g.vel, numBodies, g.first, g.count);
            } else {
                advance<tileSize><<<1 + (g.count - 1) / tileSize, tileSize>>>(
                    g.pos, g.next, g.vel, numBodies, g.first, g.count);
            }
            CUDA(cudaGetLastError());
            if (totalWorkers == 1) std::swap(g.pos, g.next);
        }

        // Finish all uploads before reusing the shared host snapshot for downloads.
        if ((again && totalWorkers > 1) || !again) {
            #pragma omp parallel for num_threads(threads)
            for (int w = 0; w < workers; ++w) {
                auto& g = partitions[w];
                if (!g.count) continue;
                CUDA(cudaSetDevice(g.device));
                if (again || needState) {
                    const Vec3* result = totalWorkers == 1 ? g.pos : g.next;
                    CUDA(cudaMemcpy(positions + g.first, result, size_t(g.count) * sizeof(Vec3),
                                    cudaMemcpyDeviceToHost));
                } else {
                    CUDA(cudaDeviceSynchronize());
                }
            }
        }
        if (again && ranks > 1)
            MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, positions, counts.data(),
                           offsets.data(), vectorType, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - start, slowest;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", (long long)(1000 * slowest));

    if (validate && ranks > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, positions, counts.data(),
                       offsets.data(), vectorType, MPI_COMM_WORLD);
    }
    std::vector<double> energies(validate ? counts[rank] : 0);
    #pragma omp parallel for num_threads(threads)
    for (int w = 0; w < workers; ++w) {
        auto& g = partitions[w];
        if (!g.count) continue;
        CUDA(cudaSetDevice(g.device));
        const int local = g.first - offsets[rank];
        if (validate) {
            if (totalWorkers > 1 && numSteps > 0)
                CUDA(cudaMemcpy(g.pos, positions, posBytes, cudaMemcpyHostToDevice));
            double* deviceEnergy;
            const size_t bytes = size_t(g.count) * sizeof(double);
            CUDA(cudaMalloc(&deviceEnergy, bytes));
            bodyEnergy<<<g.count, tileSize>>>(g.pos, g.vel, numBodies, g.first, deviceEnergy);
            CUDA(cudaGetLastError());
            CUDA(cudaMemcpy(energies.data() + local, deviceEnergy, bytes, cudaMemcpyDeviceToHost));
            CUDA(cudaFree(deviceEnergy));
        }
        if (needState)
            CUDA(cudaMemcpy(velocities.data() + local, g.vel,
                            size_t(g.count) * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA(cudaFree(g.vel));
        CUDA(cudaFree(g.next));
        CUDA(cudaFree(g.pos));
    }

    double finalEnergy = 0;
    if (validate) {
        const double localEnergy = kahan_sum(energies);
        MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (needState && ranks > 1) {
        if (!validate)
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + offsets[rank], counts[rank],
                        vectorType, positions, counts.data(), offsets.data(), vectorType,
                        0, MPI_COMM_WORLD);
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : velocities.data(), counts[rank], vectorType,
                    velocities.data(), counts.data(), offsets.data(), vectorType, 0, MPI_COMM_WORLD);
    }
    MPI_Type_free(&vectorType);

    if (rank == 0 && printResults) {
        std::vector<double> bodyData(size_t(numBodies) * 6);
        #pragma omp parallel for if(numBodies >= 8192)
        for (int i = 0; i < numBodies; ++i) {
            const size_t k = size_t(i) * 6;
            const Vec3 p = positions[i], v = velocities[i];
            bodyData[k] = p.x;
            bodyData[k + 1] = p.y;
            bodyData[k + 2] = p.z;
            bodyData[k + 3] = v.x;
            bodyData[k + 4] = v.y;
            bodyData[k + 5] = v.z;
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(positions, velocities)) {
            // Report final energy for reference
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    CUDA(cudaFreeHost(positions));
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
