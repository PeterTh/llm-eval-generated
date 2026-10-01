#include <algorithm>
#include <climits>
#include <cstddef>
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

// MPI communicates the three-double position/velocity representation directly.
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed");
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed");

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

constexpr int TILE = 128;

// One thread owns a target: source traversal and accumulation retain the
// original j order. All threads, including inactive tail lanes, load tiles.
template<bool Energy>
__global__ void interactions(const Vec3* positions, Vec3* velocities,
                             int n, int first, int count, double* energies) {
    __shared__ Vec3 sources[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    const int i = active ? first + local : 0;
    const Vec3 p = active ? positions[i] : Vec3{};
    double fx = 0, fy = 0, fz = 0, potential = 0;
    for (int base = 0; base < n;) {
        const int length = min(TILE, n - base);
        if (threadIdx.x < length) sources[threadIdx.x] = positions[base + threadIdx.x];
        __syncthreads();
        if (active) {
            for (int j = 0; j < length; ++j) {
                if constexpr (Energy) {
                    if (base + j <= i) continue;
                }
                const double dx = sources[j].x - p.x;
                const double dy = sources[j].y - p.y;
                const double dz = sources[j].z - p.z;
                const double distance = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inv = 1.0 / sqrt(distance);
                if constexpr (Energy) {
                    potential -= inv;
                } else {
                    const double inv3 = inv * inv * inv;
                    fx += dx * inv3;
                    fy += dy * inv3;
                    fz += dz * inv3;
                }
            }
        }
        __syncthreads();
        base += length;
    }
    if (active) {
        Vec3 v = velocities[local];
        if constexpr (Energy) {
            energies[local] = potential + 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
        } else {
            v.x += DT * fx;
            v.y += DT * fy;
            v.z += DT * fz;
            velocities[local] = v;
        }
    }
}

// A separate kernel is essential: every force must read the old positions.
__global__ void integrate(Vec3* positions, const Vec3* velocities, int first, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        positions[first+i].x += velocities[i].x * DT;
        positions[first+i].y += velocities[i].y * DT;
        positions[first+i].z += velocities[i].z * DT;
    }
}

struct DeviceState {
    int id, first, count;
    Vec3* positions = nullptr;
    Vec3* velocities = nullptr;
    double* energies = nullptr;
    cudaStream_t stream = nullptr;
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int n = 1024, steps = 10;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-n") || !strcmp(argv[i], "-s")) && i+1 < argc) {
            const bool bodies = !strcmp(argv[i], "-n");
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX) {
                if (!rank) fprintf(stderr, "Invalid nonnegative integer: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            (bodies ? n : steps) = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    // Divide the GPUs visible on each node among its ranks. A single rank
    // drives all visible GPUs with OpenMP; launch one rank/GPU if preferred.
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, deviceCount;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        if (!rank) fprintf(stderr, "This benchmark requires a CUDA GPU on every rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> deviceIds;
    for (int id = localRank; id < deviceCount; id += localRanks) deviceIds.push_back(id);
    if (deviceIds.empty()) deviceIds.push_back(localRank % deviceCount);
    MPI_Comm_free(&node);

    // Weight rank partitions by their GPU counts (e.g. three ranks on a
    // four-GPU node), so each GPU receives approximately equal work.
    const int available = static_cast<int>(deviceIds.size());
    std::vector<int> weights(ranks), counts(ranks), offsets(ranks);
    MPI_Allgather(&available, 1, MPI_INT, weights.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long totalDevices = 0, prefix = 0;
    for (int weight : weights) totalDevices += weight;
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = static_cast<int>((n / totalDevices)*prefix + std::min(prefix, n % totalDevices));
        prefix += weights[r];
        counts[r] = static_cast<int>((n / totalDevices)*prefix + std::min(prefix, n % totalDevices)) - offsets[r];
    }
    const int count = counts[rank], first = offsets[rank];
    // Avoid idle GPU contexts when the problem is smaller than the GPU count.
    const int workers = std::min(static_cast<int>(deviceIds.size()), std::max(count, 1));
    const bool resident = ranks == 1 && workers == 1;
    MPI_Datatype vectorType, bodyType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vectorType); MPI_Type_commit(&vectorType);
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType); MPI_Type_commit(&bodyType);
    Vec3 *positions, *velocities;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&positions), std::max(size_t(n), size_t(1))*sizeof(Vec3)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&velocities), std::max(size_t(count), size_t(1))*sizeof(Vec3)));
    std::vector<Body> bodies(rank == 0 ? n : 0), localBodies(count);
    if (!rank) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               n, steps, validate ? "enabled" : "disabled");
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) positions[i] = bodies[i].pos;
    }
    MPI_Bcast(positions, n, vectorType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(bodies.data(), counts.data(), offsets.data(), bodyType,
                 localBodies.data(), count, bodyType, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) velocities[i] = localBodies[i].vel;
    std::vector<DeviceState> devices(workers);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < workers; ++g) {
        auto& d = devices[g];
        d.id = deviceIds[g];
        d.first = first + static_cast<int>(static_cast<long long>(count)*g/workers);
        d.count = static_cast<int>(static_cast<long long>(count)*(g+1)/workers) - (d.first-first);
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d.positions, std::max(size_t(n), size_t(1))*sizeof(Vec3)));
        CUDA_CHECK(cudaMalloc(&d.velocities, std::max(d.count, 1)*sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpyAsync(d.positions, positions, size_t(n)*sizeof(Vec3), cudaMemcpyHostToDevice, d.stream));
        CUDA_CHECK(cudaMemcpyAsync(d.velocities, velocities+d.first-first, size_t(d.count)*sizeof(Vec3), cudaMemcpyHostToDevice, d.stream));
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    // Keep the OpenMP team alive across timesteps. Only the primary thread
    // calls MPI, as required by MPI_THREAD_FUNNELED.
    #pragma omp parallel num_threads(workers)
    {
        for (int step = 0; step < steps; ++step) {
            #pragma omp for schedule(static)
            for (int g = 0; g < workers; ++g) {
                auto& d = devices[g];
                CUDA_CHECK(cudaSetDevice(d.id));
                if (step && !resident)
                    CUDA_CHECK(cudaMemcpyAsync(d.positions, positions, size_t(n)*sizeof(Vec3), cudaMemcpyHostToDevice, d.stream));
                if (d.count) {
                    const int blocks = (d.count-1)/TILE+1;
                    interactions<false><<<blocks, TILE, 0, d.stream>>>(d.positions, d.velocities, n, d.first, d.count, nullptr);
                    CUDA_CHECK(cudaGetLastError());
                    integrate<<<blocks, TILE, 0, d.stream>>>(d.positions, d.velocities, d.first, d.count);
                    CUDA_CHECK(cudaGetLastError());
                }
                if (!resident) {
                    CUDA_CHECK(cudaMemcpyAsync(positions+d.first, d.positions+d.first, size_t(d.count)*sizeof(Vec3), cudaMemcpyDeviceToHost, d.stream));
                    CUDA_CHECK(cudaStreamSynchronize(d.stream));
                }
            }
            #pragma omp master
            {
                if (ranks > 1)
                    MPI_Allgatherv(MPI_IN_PLACE, 0, vectorType, positions, counts.data(), offsets.data(), vectorType, MPI_COMM_WORLD);
            }
            #pragma omp barrier
        }
        #pragma omp for schedule(static)
        for (int g = 0; g < workers; ++g) {
            CUDA_CHECK(cudaSetDevice(devices[g].id));
            CUDA_CHECK(cudaStreamSynchronize(devices[g].stream));
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("Simulation time: %lld ms\n", static_cast<long long>(duration*1000));

    double localEnergy = 0;
    std::vector<double> energies(validate ? count : 0);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < workers; ++g) {
        auto& d = devices[g];
        CUDA_CHECK(cudaSetDevice(d.id));
        if (resident)
            CUDA_CHECK(cudaMemcpyAsync(positions, d.positions, size_t(n)*sizeof(Vec3), cudaMemcpyDeviceToHost, d.stream));
        if (validate || results)
            CUDA_CHECK(cudaMemcpyAsync(velocities+d.first-first, d.velocities, size_t(d.count)*sizeof(Vec3), cudaMemcpyDeviceToHost, d.stream));
        if (validate && d.count) {
            if (!resident)
                CUDA_CHECK(cudaMemcpyAsync(d.positions, positions, size_t(n)*sizeof(Vec3), cudaMemcpyHostToDevice, d.stream));
            CUDA_CHECK(cudaMalloc(&d.energies, size_t(d.count)*sizeof(double)));
            interactions<true><<<(d.count-1)/TILE+1, TILE, 0, d.stream>>>(d.positions, d.velocities, n, d.first, d.count, d.energies);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(energies.data()+d.first-first, d.energies, size_t(d.count)*sizeof(double), cudaMemcpyDeviceToHost, d.stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    int valid = 1;
    if (validate || results) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) localBodies[i] = {positions[first+i], velocities[i]};
        if (results) {
            MPI_Gatherv(localBodies.data(), count, bodyType, bodies.data(), counts.data(), offsets.data(), bodyType, 0, MPI_COMM_WORLD);
            if (!rank) {
                std::vector<double> data(size_t(n)*6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < n; ++i) {
                    const auto& b = bodies[i];
                    const size_t k = size_t(i)*6;
                    data[k] = b.pos.x; data[k+1] = b.pos.y; data[k+2] = b.pos.z;
                    data[k+3] = b.vel.x; data[k+4] = b.vel.y; data[k+5] = b.vel.z;
                }
                print_results(data, "Bodies");
            }
        }
        if (validate) {
            const int localValid = validateSimulation(localBodies);
            MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            #pragma omp parallel for reduction(+:localEnergy) schedule(static)
            for (int i = 0; i < count; ++i) localEnergy += energies[i];
            double totalEnergy;
            MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!rank) {
                printf("Validating simulation results...\n");
                if (valid) printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int g = 0; g < workers; ++g) {
        auto& d = devices[g];
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaFree(d.energies));
        CUDA_CHECK(cudaFree(d.velocities));
        CUDA_CHECK(cudaFree(d.positions));
        CUDA_CHECK(cudaStreamDestroy(d.stream));
    }
    CUDA_CHECK(cudaFreeHost(velocities)); CUDA_CHECK(cudaFreeHost(positions));
    MPI_Type_free(&vectorType); MPI_Type_free(&bodyType);
    MPI_Finalize();
    return valid ? 0 : 1;
}
