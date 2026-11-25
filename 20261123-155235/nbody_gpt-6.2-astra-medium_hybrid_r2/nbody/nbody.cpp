#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
};
static_assert(sizeof(Vec3) == 3 * sizeof(double));

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::abort();
    }
}

struct Partition {
    int device, first, count;
    Vec3* positions = nullptr;
    Vec3* next = nullptr;
    Vec3* velocity = nullptr;
};

int boundary(int n, long long part, long long parts) {
    return part * (n / parts) + std::min(part, n % parts);
}

__device__ double inverseDistance(Vec3 delta) {
    // Keep the rounding of the original optimized double-precision expression.
    double xy = fma(delta.x, delta.x, delta.y * delta.y);
    double squared = fma(delta.z, delta.z, xy) + SOFTENING;
    return __ddiv_rn(1.0, __dsqrt_rn(squared));
}

// One thread owns a target. A block reuses each tile of source positions.
// Inactive lanes in the last block still participate in loading the tile.
template<bool Energy, int Tile>
__global__ void interact(const Vec3* __restrict__ positions, Vec3* next,
                         Vec3* velocity, double* energy, int n, int first, int count) {
    __shared__ Vec3 sources[Tile];
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = local < count;
    int target = active ? first + local : 0;
    Vec3 pos = positions[target];
    Vec3 force{};
    double potential = 0;

    for (size_t base = Energy ? size_t(first + blockIdx.x * blockDim.x) : 0;
         base < size_t(n); base += Tile) {
        int length = min(size_t(Tile), size_t(n) - base);
        if (threadIdx.x < length) sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncthreads();
        if (active) {
            int begin = Energy && base <= target ? target - int(base) + 1 : 0;
            #pragma unroll 4
            for (int j = begin; j < length; ++j) {
                Vec3 d{sources[j].x - pos.x, sources[j].y - pos.y, sources[j].z - pos.z};
                double inv = inverseDistance(d);
                if constexpr (Energy) {
                    potential -= inv;
                } else {
                    double scale = inv * inv * inv;
                    force.x = fma(d.x, scale, force.x);
                    force.y = fma(d.y, scale, force.y);
                    force.z = fma(d.z, scale, force.z);
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    Vec3 v = velocity[local];
    if constexpr (Energy) {
        energy[local] = potential + 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
    } else {
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        velocity[local] = v;
        next[local] = {fma(DT, v.x, pos.x), fma(DT, v.y, pos.y), fma(DT, v.z, pos.z)};
    }
}

template<bool Energy>
void launch(const Partition& p, int n, double* energy = nullptr) {
    // Smaller blocks spread short target ranges over more multiprocessors.
    if (p.count < 8192) {
        interact<Energy, 32><<<1 + (p.count - 1) / 32, 32>>>(
            p.positions, p.next, p.velocity, energy, n, p.first, p.count);
    } else {
        interact<Energy, 128><<<1 + (p.count - 1) / 128, 128>>>(
            p.positions, p.next, p.velocity, energy, n, p.first, p.count);
    }
    check(cudaGetLastError());
}

Vec3 randomVector(unsigned& seed) {
    auto sample = [&]() { return 2.0 * (rand_r(&seed) / double(RAND_MAX)) - 1.0; };
    return {sample(), sample(), sample()};
}

int validateBody(Vec3 pos, Vec3 vel) {
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
        !std::isfinite(vel.x) || !std::isfinite(vel.y) || !std::isfinite(vel.z)) return 1;
    if (std::max({std::abs(pos.x), std::abs(pos.y), std::abs(pos.z)}) > 1e6) return 2;
    if (std::max({std::abs(vel.x), std::abs(vel.y), std::abs(vel.z)}) > 1e6) return 3;
    return 0;
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (numBodies < 0) return 1;
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, nodeSize, devices;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_size(node, &nodeSize);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // A launcher may expose a separate GPU set to each task. Share device
    // ordinals only among node-local ranks with the same first visible GPU.
    cudaDeviceProp firstGPU;
    check(cudaGetDeviceProperties(&firstGPU, 0));
    std::vector<cudaUUID_t> leaders(nodeSize);
    MPI_Allgather(&firstGPU.uuid, sizeof(cudaUUID_t), MPI_BYTE, leaders.data(),
                  sizeof(cudaUUID_t), MPI_BYTE, node);
    MPI_Comm_free(&node);
    int groupRank = 0, groupSize = 0;
    for (int r = 0; r < nodeSize; ++r) {
        if (std::memcmp(&leaders[r], &firstGPU.uuid, sizeof(cudaUUID_t)) == 0) {
            if (r < nodeRank) ++groupRank;
            ++groupSize;
        }
    }

    // Weight the MPI partition by the number of GPUs driven by each rank.
    int available = groupRank < devices ? 1 + (devices - 1 - groupRank) / groupSize : 1;
    std::vector<int> gpuCounts(ranks), offsets(ranks), counts(ranks);
    MPI_Allgather(&available, 1, MPI_INT, gpuCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long totalGPUs = 0;
    for (int gpus : gpuCounts) totalGPUs += gpus;
    long long gpuOffset = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = boundary(numBodies, gpuOffset, totalGPUs);
        gpuOffset += gpuCounts[r];
        counts[r] = boundary(numBodies, gpuOffset, totalGPUs) - offsets[r];
    }
    const int first = offsets[rank], count = counts[rank];
    std::vector<Partition> parts;
    int workers = std::min(count, available);
    for (int k = 0; k < workers; ++k) {
        int begin = boundary(count, k, workers);
        parts.push_back({(groupRank % devices + k * groupSize) % devices, first + begin,
                         boundary(count, k + 1, workers) - begin});
    }
    const bool distributed = ranks > 1 || workers > 1;
    const size_t positionBytes = size_t(numBodies) * sizeof(Vec3);
    Vec3* positions;
    check(cudaMallocHost(&positions, std::max(positionBytes, sizeof(Vec3))));
    std::vector<Vec3> velocities(rank == 0 ? numBodies : count);
    if (rank == 0) {
        unsigned seed = 42;
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = randomVector(seed);
            velocities[i] = randomVector(seed);
        }
    }
    MPI_Datatype vec;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec);
    MPI_Type_commit(&vec);
    MPI_Bcast(positions, numBodies, vec, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities.data(), counts.data(), offsets.data(), vec,
                 rank == 0 ? MPI_IN_PLACE : velocities.data(), count, vec, 0, MPI_COMM_WORLD);

    int threads = std::max(1, std::min(workers, omp_get_max_threads()));
    #pragma omp parallel for num_threads(threads)
    for (int k = 0; k < workers; ++k) {
        auto& p = parts[k];
        check(cudaSetDevice(p.device));
        size_t bytes = size_t(p.count) * sizeof(Vec3);
        check(cudaMalloc(&p.positions, positionBytes));
        check(cudaMalloc(&p.next, bytes));
        check(cudaMalloc(&p.velocity, bytes));
        check(cudaMemcpy(p.positions, positions, positionBytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(p.velocity, velocities.data() + (p.first - first), bytes, cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        bool collect = distributed && (step + 1 < numSteps || validate || printResults);
        #pragma omp parallel for num_threads(threads)
        for (int k = 0; k < workers; ++k) {
            auto& p = parts[k];
            check(cudaSetDevice(p.device));
            if (distributed && step > 0)
                check(cudaMemcpy(p.positions, positions, positionBytes, cudaMemcpyHostToDevice));
            launch<false>(p, numBodies);
            if (!distributed) std::swap(p.next, p.positions);
        }
        // Finish all uploads before any worker overwrites the shared host snapshot.
        if (collect) {
            #pragma omp parallel for num_threads(threads)
            for (int k = 0; k < workers; ++k) {
                auto& p = parts[k];
                check(cudaSetDevice(p.device));
                check(cudaMemcpy(positions + p.first, p.next, size_t(p.count) * sizeof(Vec3),
                                 cudaMemcpyDeviceToHost));
            }
        }
        if (ranks > 1 && collect)
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec, positions, counts.data(), offsets.data(),
                           vec, MPI_COMM_WORLD);
    }
    #pragma omp parallel for num_threads(threads)
    for (int k = 0; k < workers; ++k) {
        check(cudaSetDevice(parts[k].device));
        check(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", (long long)(maxElapsed * 1000));

    std::vector<double> energies(validate ? count : 0);
    #pragma omp parallel for num_threads(threads)
    for (int k = 0; k < workers; ++k) {
        auto& p = parts[k];
        check(cudaSetDevice(p.device));
        if (validate || printResults) {
            check(cudaMemcpy(velocities.data() + (p.first - first), p.velocity,
                             size_t(p.count) * sizeof(Vec3), cudaMemcpyDeviceToHost));
            if (!distributed)
                check(cudaMemcpy(positions, p.positions, positionBytes, cudaMemcpyDeviceToHost));
        }
        if (validate) {
            if (distributed && numSteps > 0)
                check(cudaMemcpy(p.positions, positions, positionBytes, cudaMemcpyHostToDevice));
            size_t bytes = size_t(p.count) * sizeof(double);
            double* deviceEnergy;
            check(cudaMalloc(&deviceEnergy, bytes));
            launch<true>(p, numBodies, deviceEnergy);
            check(cudaMemcpy(energies.data() + (p.first - first), deviceEnergy, bytes,
                             cudaMemcpyDeviceToHost));
            check(cudaFree(deviceEnergy));
        }
        check(cudaFree(p.positions));
        check(cudaFree(p.next));
        check(cudaFree(p.velocity));
    }

    if (printResults) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : velocities.data(), count, vec, velocities.data(),
                    counts.data(), offsets.data(), vec, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> data(size_t(numBodies) * 6);
            for (size_t i = 0; i < size_t(numBodies); ++i) {
                std::memcpy(&data[6*i], &positions[i], sizeof(Vec3));
                std::memcpy(&data[6*i+3], &velocities[i], sizeof(Vec3));
            }
            print_results(data, "Bodies");
        }
    }
    int failed = 0;
    if (validate) {
        double energy = 0, total = 0;
        int invalid = 0;
        for (int i = 0; i < count; ++i) {
            energy += energies[i];
            if (!invalid) invalid = validateBody(positions[first+i], velocities[i]);
        }
        MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&invalid, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (failed) {
                const char* messages[] = {"", "found NaN or Inf value in body state",
                    "body position exceeds reasonable bounds", "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\nValidation: FAILED\n", messages[failed]);
            } else {
                printf("Final energy: %.6f\nValidation: PASSED\n", total);
            }
        }
    }
    check(cudaFreeHost(positions));
    MPI_Type_free(&vec);
    return failed ? 1 : 0;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
