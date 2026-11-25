#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 128;

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

template<class T>
T* deviceArray(size_t count) {
    T* data;
    check(cudaMalloc(&data, sizeof(T) * std::max<size_t>(count, 1)));
    return data;
}

__device__ double inverseDistance(double dx, double dy, double dz) {
    double squared = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
    return 1.0 / sqrt(squared);
}

// Every block reuses each source tile across its target bodies. The output
// buffer keeps all reads on the same time slice during the fused update.
__global__ void advance(const Vec3* __restrict__ pos, Vec3* __restrict__ next,
                        Vec3* __restrict__ vel, int n, int first, int count) {
    __shared__ Vec3 tile[TILE];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * TILE + lane;
    Vec3 p = i < count ? pos[first + i] : Vec3{};
    Vec3 force{};
    for (size_t base = 0; base < static_cast<size_t>(n); base += TILE) {
        int length = min(static_cast<size_t>(TILE), static_cast<size_t>(n) - base);
        if (lane < length) tile[lane] = pos[base + lane];
        __syncthreads();
        if (i < count) {
            #pragma unroll 4
            for (int j = 0; j < length; ++j) {
                double dx = tile[j].x - p.x;
                double dy = tile[j].y - p.y;
                double dz = tile[j].z - p.z;
                double inv = inverseDistance(dx, dy, dz);
                double cube = inv * inv * inv;
                force.x = fma(dx, cube, force.x);
                force.y = fma(dy, cube, force.y);
                force.z = fma(dz, cube, force.z);
            }
        }
        __syncthreads();
    }
    if (i < count) {
        Vec3 v = vel[i];
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        vel[i] = v;
        next[i] = {fma(DT, v.x, p.x), fma(DT, v.y, p.y), fma(DT, v.z, p.z)};
    }
}

// Pair long and short rows of the upper triangle to balance potential
// energy work across devices. Kinetic energy follows velocity ownership.
__global__ void energyRows(const Vec3* pos, const Vec3* vel, double* energy,
                           int n, int first) {
    const int index = first + blockIdx.x;
    const int i = index % 2 ? n - 1 - index / 2 : index / 2;
    const Vec3 p = pos[i];
    double sum = 0.0;
    for (size_t j = size_t(i) + 1 + threadIdx.x; j < size_t(n); j += TILE) {
        Vec3 q = pos[j];
        sum -= inverseDistance(q.x - p.x, q.y - p.y, q.z - p.z);
    }
    __shared__ double partial[TILE];
    partial[threadIdx.x] = sum;
    for (int stride = TILE / 2; stride; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    if (threadIdx.x == 0) {
        Vec3 v = vel[blockIdx.x];
        energy[blockIdx.x] = partial[0] + 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
    }
}

struct Device {
    int id, first, count;
    Vec3 *positions, *next, *velocities;
};

int split(int n, int part, int parts) {
    return (n / parts) * part + std::min(part, n % parts);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm node;
    int localRank, localRanks, visible;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    check(cudaGetDeviceCount(&visible));
    if (visible == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Launchers may expose all node GPUs or a private GPU set to each rank.
    // Share devices only among ranks with the same leading visible GPU.
    cudaDeviceProp props;
    check(cudaGetDeviceProperties(&props, 0));
    std::vector<cudaUUID_t> leading(localRanks);
    MPI_Allgather(&props.uuid, sizeof(cudaUUID_t), MPI_BYTE, leading.data(),
                  sizeof(cudaUUID_t), MPI_BYTE, node);
    int groupRank = 0, groupSize = 0;
    for (int r = 0; r < localRanks; ++r) {
        if (std::memcmp(&leading[r], &props.uuid, sizeof(cudaUUID_t)) == 0) {
            groupRank += r < localRank;
            ++groupSize;
        }
    }
    MPI_Comm_free(&node);
    const int firstDevice = groupRank % visible;
    const int assigned = 1 + (visible - 1 - firstDevice) / groupSize;
    std::vector<int> shares(ranks), counts(ranks), offsets(ranks);
    MPI_Allgather(&assigned, 1, MPI_INT, shares.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int totalShares = 0;
    for (int share : shares) totalShares += share;
    int startShare = 0;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = split(numBodies, startShare, totalShares);
        startShare += shares[r];
        counts[r] = split(numBodies, startShare, totalShares) - offsets[r];
    }
    const int localCount = counts[rank];
    const int devices = std::min(assigned, std::max(1, localCount));
    const int workers = std::min(devices, omp_get_max_threads());
    const bool resident = localCount == numBodies && devices == 1;
    // A pinned snapshot works with ordinary MPI transports on and off node.
    Vec3* positions;
    check(cudaMallocHost(&positions, sizeof(Vec3) * std::max(numBodies, 1)));
    std::vector<Vec3> velocities(rank == 0 ? numBodies : 0);
    std::vector<Vec3> localVel(std::max(1, localCount));
    if (rank == 0) {
        unsigned seed = 42;
        auto sample = [&]() { return 2.0 * (rand_r(&seed) / double(RAND_MAX)) - 1.0; };
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = {sample(), sample(), sample()};
            velocities[i] = {sample(), sample(), sample()};
        }
    }

    MPI_Datatype triple;
    MPI_Type_contiguous(3, MPI_DOUBLE, &triple);
    MPI_Type_commit(&triple);
    MPI_Bcast(positions, numBodies, triple, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities.data(), counts.data(), offsets.data(), triple,
                 localVel.data(), localCount, triple, 0, MPI_COMM_WORLD);
    if (!printResults) std::vector<Vec3>().swap(velocities);

    std::vector<Device> gpu(devices);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < devices; ++g) {
        Device& d = gpu[g];
        d.id = firstDevice + g * groupSize;
        const int offset = split(localCount, g, devices);
        d.first = offsets[rank] + offset;
        d.count = split(localCount, g + 1, devices) - offset;
        check(cudaSetDevice(d.id));
        d.positions = deviceArray<Vec3>(d.count ? numBodies : 0);
        d.velocities = deviceArray<Vec3>(d.count);
        d.next = deviceArray<Vec3>(d.count);
        if (d.count) {
            check(cudaMemcpy(d.positions, positions, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
            check(cudaMemcpy(d.velocities, localVel.data() + offset,
                             sizeof(Vec3) * d.count, cudaMemcpyHostToDevice));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        #pragma omp parallel for num_threads(workers) schedule(static)
        for (int g = 0; g < devices; ++g) {
            Device& d = gpu[g];
            if (!d.count) continue;
            check(cudaSetDevice(d.id));
            advance<<<1 + (d.count - 1) / TILE, TILE>>>(d.positions, d.next, d.velocities,
                                                         numBodies, d.first, d.count);
            check(cudaGetLastError());
            if (resident) {
                std::swap(d.positions, d.next);
            } else if (step + 1 < numSteps) {
                check(cudaMemcpy(positions + d.first, d.next, sizeof(Vec3) * d.count, cudaMemcpyDeviceToHost));
            }
        }
        if (ranks > 1 && numBodies > 1 && step + 1 < numSteps)
            MPI_Allgatherv(MPI_IN_PLACE, 0, triple, positions, counts.data(), offsets.data(), triple, MPI_COMM_WORLD);
        if (!resident && step + 1 < numSteps) {
            // Finish all uploads before any GPU overwrites its host partition.
            #pragma omp parallel for num_threads(workers) schedule(static)
            for (int g = 0; g < devices; ++g) {
                Device& d = gpu[g];
                if (!d.count) continue;
                check(cudaSetDevice(d.id));
                check(cudaMemcpy(d.positions, positions, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
            }
        }
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < devices; ++g) {
        check(cudaSetDevice(gpu[g].id));
        check(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(1000 * maxElapsed));

    if (validate || printResults) {
        #pragma omp parallel for num_threads(workers) schedule(static)
        for (int g = 0; g < devices; ++g) {
            Device& d = gpu[g];
            if (!d.count) continue;
            check(cudaSetDevice(d.id));
            if (numSteps > 0) {
                const Vec3* finalPositions = resident ? d.positions : d.next;
                check(cudaMemcpy(positions + d.first, finalPositions, sizeof(Vec3) * d.count, cudaMemcpyDeviceToHost));
            }
            check(cudaMemcpy(localVel.data() + (d.first - offsets[rank]), d.velocities,
                             sizeof(Vec3) * d.count, cudaMemcpyDeviceToHost));
        }
        if (ranks > 1 && numSteps > 0) {
            if (validate)
                MPI_Allgatherv(MPI_IN_PLACE, 0, triple, positions, counts.data(), offsets.data(), triple, MPI_COMM_WORLD);
            else
                MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + offsets[rank], localCount, triple,
                            positions, counts.data(), offsets.data(), triple, 0, MPI_COMM_WORLD);
        }
    }
    if (printResults) {
        MPI_Gatherv(localVel.data(), localCount, triple, velocities.data(), counts.data(),
                    offsets.data(), triple, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(size_t(numBodies) * 6);
            #pragma omp parallel for schedule(static) if(numBodies >= 16384)
            for (int i = 0; i < numBodies; ++i) {
                Vec3 p = positions[i], v = velocities[i];
                double* out = bodyData.data() + size_t(i) * 6;
                out[0] = p.x;
                out[1] = p.y;
                out[2] = p.z;
                out[3] = v.x;
                out[4] = v.y;
                out[5] = v.z;
            }
            print_results(bodyData, "Bodies");
        }
    }

    int failure = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int localFailure = 0;
        for (int i = 0; i < localCount && !localFailure; ++i) {
            Vec3 p = positions[offsets[rank] + i], v = localVel[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                localFailure = 1;
            else if (std::max({fabs(p.x), fabs(p.y), fabs(p.z)}) > 1e6)
                localFailure = 2;
            else if (std::max({fabs(v.x), fabs(v.y), fabs(v.z)}) > 1e6)
                localFailure = 3;
        }
        // Contiguous rank partitions make the first failing rank the owner
        // of the first invalid body in the original ordering.
        int localIssue[2] = {localFailure ? rank : ranks, localFailure};
        int firstIssue[2];
        MPI_Allreduce(localIssue, firstIssue, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        failure = firstIssue[1];
        if (!failure) {
            double localEnergy = 0.0, total;
            #pragma omp parallel for num_threads(workers) schedule(static) reduction(+:localEnergy)
            for (int g = 0; g < devices; ++g) {
                Device& d = gpu[g];
                if (!d.count) continue;
                check(cudaSetDevice(d.id));
                if (!resident && numSteps > 0)
                    check(cudaMemcpy(d.positions, positions, sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
                double* rows = deviceArray<double>(d.count);
                energyRows<<<d.count, TILE>>>(d.positions, d.velocities, rows, numBodies, d.first);
                check(cudaGetLastError());
                std::vector<double> values(d.count);
                check(cudaMemcpy(values.data(), rows, sizeof(double) * d.count, cudaMemcpyDeviceToHost));
                for (double value : values) localEnergy += value;
                check(cudaFree(rows));
            }
            MPI_Reduce(&localEnergy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Final energy: %.6f\nValidation: PASSED\n", total);
        } else if (rank == 0) {
            const char* reasons[] = {"", "found NaN or Inf value in body state",
                                     "body position exceeds reasonable bounds",
                                     "body velocity exceeds reasonable bounds"};
            printf("Validation failed: %s\nValidation: FAILED\n", reasons[failure]);
        }
    }

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < devices; ++g) {
        Device& d = gpu[g];
        check(cudaSetDevice(d.id));
        check(cudaFree(d.positions));
        check(cudaFree(d.next));
        check(cudaFree(d.velocities));
    }
    check(cudaFreeHost(positions));
    MPI_Type_free(&triple);
    MPI_Finalize();
    return failure ? 1 : 0;
}
