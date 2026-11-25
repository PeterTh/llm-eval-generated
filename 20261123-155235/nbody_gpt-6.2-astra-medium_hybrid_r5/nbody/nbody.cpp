#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// rand_r on glibc advances a 32-bit LCG three times per draw. Jump
// directly to a thread's first body without changing the serial RNG stream.
unsigned int bodySeed(size_t first) {
    unsigned int seed = 42;
#ifdef __GLIBC__
    unsigned int multiplier = 1103515245u, increment = 12345u;
    for (uint64_t steps = uint64_t(first) * 18; steps; steps >>= 1) {
        if (steps & 1) seed = seed * multiplier + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
#else
    for (uint64_t i = 0; i < uint64_t(first) * 6; ++i) rand_r(&seed);
#endif
    return seed;
}

Vec3 randomVector(unsigned int& seed) {
    auto draw = [&] { return 2.0 * (rand_r(&seed) / double(RAND_MAX)) - 1.0; };
    return {draw(), draw(), draw()};
}

void initialize(Vec3* positions, Vec3* velocities, int first, int count) {
    #pragma omp parallel
    {
        const int threads = omp_get_num_threads(), thread = omp_get_thread_num();
        const int begin = static_cast<int>(int64_t(count) * thread / threads);
        const int end = static_cast<int>(int64_t(count) * (thread + 1) / threads);
        if (begin < end) {
            unsigned int seed = bodySeed(first + begin);
            for (int i = begin; i < end; ++i) {
                positions[first + i] = randomVector(seed);
                velocities[i] = randomVector(seed);
            }
        }
    }
}

static_assert(sizeof(Vec3) == 3 * sizeof(double));

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

constexpr int tileSize = 128;

__device__ inline void integrate(Vec3 pos, Vec3 force, Vec3& velocity, Vec3& next) {
    Vec3 v = velocity;
    v.x += __dmul_rn(DT, force.x);
    v.y += __dmul_rn(DT, force.y);
    v.z += __dmul_rn(DT, force.z);
    velocity = v;
    next = {fma(DT, v.x, pos.x), fma(DT, v.y, pos.y), fma(DT, v.z, pos.z)};
}

__device__ inline double inverseCube(double dx, double dy, double dz) {
    const double r2 = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
    const double inv = __ddiv_rn(1.0, __dsqrt_rn(r2));
    return inv * inv * inv;
}

// With few targets per GPU, give each target a warp. The lanes evaluate
// distances concurrently, then lane zero accumulates in source order.
__global__ void advanceWarps(const Vec3* positions, Vec3* next, Vec3* velocities,
                             int n, int first, int count) {
    constexpr int batch = 64;
    __shared__ double4 terms[4][batch];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp;
    if (i >= count) return;
    const Vec3 pos = positions[first + i];
    Vec3 force{};
    for (size_t base = 0; base < n; base += batch) {
        const int length = min(batch, int(n - base));
        for (int k = lane; k < length; k += 32) {
            const Vec3 source = positions[base + k];
            const double dx = source.x - pos.x, dy = source.y - pos.y, dz = source.z - pos.z;
            terms[warp][k] = make_double4(dx, dy, dz, inverseCube(dx, dy, dz));
        }
        __syncwarp();
        if (lane == 0) {
            #pragma unroll 4
            for (int k = 0; k < length; ++k) {
                const double4 term = terms[warp][k];
                force.x = fma(term.x, term.w, force.x);
                force.y = fma(term.y, term.w, force.y);
                force.z = fma(term.z, term.w, force.z);
            }
        }
        __syncwarp();
    }
    if (lane == 0) integrate(pos, force, velocities[i], next[i]);
}

// All lanes, including inactive targets in the last block, load source tiles.
__global__ void advance(const Vec3* __restrict__ positions, Vec3* __restrict__ next,
                        Vec3* __restrict__ velocities, int n, int first, int count) {
    __shared__ Vec3 tile[tileSize];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * tileSize + lane;
    const Vec3 pos = i < count ? positions[first + i] : Vec3{};
    Vec3 force{};
    for (size_t base = 0; base < n; base += tileSize) {
        const int length = min(tileSize, int(n - base));
        if (lane < length) tile[lane] = positions[base + lane];
        __syncthreads();
        if (i < count) {
            #pragma unroll 8
            for (int j = 0; j < length; ++j) {
                const double dx = tile[j].x - pos.x;
                const double dy = tile[j].y - pos.y;
                const double dz = tile[j].z - pos.z;
                const double inv3 = inverseCube(dx, dy, dz);
                force.x = fma(dx, inv3, force.x);
                force.y = fma(dy, inv3, force.y);
                force.z = fma(dz, inv3, force.z);
            }
        }
        __syncthreads();
    }
    if (i < count) integrate(pos, force, velocities[i], next[i]);
}

// Mirror the triangular pair range so every rank has a balanced energy sum.
double potentialEnergy(const Vec3* positions, int n, int first, int count) {
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int k = first; k < first + count; ++k) {
        const int i = (k % 2 == 0) ? k / 2 : n - 1 - k / 2;
        const Vec3 p = positions[i];
        double row = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = positions[j].x - p.x;
            const double dy = positions[j].y - p.y;
            const double dz = positions[j].z - p.z;
            row -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        energy += row;
    }
    return energy;
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

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices));
    int deviceCollectives = 0;
#if defined(OPEN_MPI) && defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    deviceCollectives = MPIX_Query_cuda_support();
    MPI_Allreduce(MPI_IN_PLACE, &deviceCollectives, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    MPI_Datatype xyz;
    MPI_Type_contiguous(3, MPI_DOUBLE, &xyz);
    MPI_Type_commit(&xyz);

    const size_t globalBytes = sizeof(Vec3) * numBodies;
    const size_t localBytes = sizeof(Vec3) * count;
    Vec3 *positions, *current, *next, *velocity;
    checkCuda(cudaMallocHost(&positions, std::max(globalBytes, sizeof(Vec3))));
    checkCuda(cudaMalloc(&current, std::max(globalBytes, sizeof(Vec3))));
    checkCuda(cudaMalloc(&next, std::max(localBytes, sizeof(Vec3))));
    checkCuda(cudaMalloc(&velocity, std::max(localBytes, sizeof(Vec3))));
    std::vector<Vec3> localVelocity(count);
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    initialize(positions, localVelocity.data(), first, count);
    MPI_Allgatherv(MPI_IN_PLACE, 0, xyz, positions, counts.data(), offsets.data(), xyz, MPI_COMM_WORLD);
    checkCuda(cudaMemcpy(current, positions, globalBytes, cudaMemcpyHostToDevice));
    if (count) checkCuda(cudaMemcpy(velocity, localVelocity.data(), localBytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        if (count) {
            if (count <= 2048) {
                advanceWarps<<<1 + (count - 1) / 4, 128>>>(current, next, velocity, numBodies, first, count);
            } else {
                advance<<<1 + (count - 1) / tileSize, tileSize>>>(current, next, velocity, numBodies, first, count);
            }
            checkCuda(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(current, next);
        } else if (step + 1 < numSteps) {
            if (deviceCollectives) {
                checkCuda(cudaDeviceSynchronize());
                MPI_Allgatherv(next, count, xyz, current, counts.data(), offsets.data(), xyz, MPI_COMM_WORLD);
            } else {
                checkCuda(cudaMemcpy(positions + first, next, localBytes, cudaMemcpyDeviceToHost));
                MPI_Allgatherv(MPI_IN_PLACE, 0, xyz, positions, counts.data(), offsets.data(), xyz, MPI_COMM_WORLD);
                checkCuda(cudaMemcpy(current, positions, globalBytes, cudaMemcpyHostToDevice));
            }
        }
    }
    checkCuda(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, longest;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %lld ms\n", static_cast<long long>(1000 * longest));

    if (printResults || validate) {
        if (numSteps > 0) {
            checkCuda(cudaMemcpy(positions + first, ranks == 1 ? current : next, localBytes, cudaMemcpyDeviceToHost));
            if (validate) {
                MPI_Allgatherv(MPI_IN_PLACE, 0, xyz, positions, counts.data(), offsets.data(), xyz, MPI_COMM_WORLD);
            } else {
                MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + first, count, xyz,
                            positions, counts.data(), offsets.data(), xyz, 0, MPI_COMM_WORLD);
            }
            if (count) checkCuda(cudaMemcpy(localVelocity.data(), velocity, localBytes, cudaMemcpyDeviceToHost));
        }
        if (printResults) {
            std::vector<Vec3> allVelocity(rank == 0 ? numBodies : 0);
            MPI_Gatherv(localVelocity.data(), count, xyz, allVelocity.data(), counts.data(), offsets.data(),
                        xyz, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::vector<double> data(size_t(numBodies) * 6);
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    const Vec3 p = positions[i], v = allVelocity[i];
                    double* out = data.data() + size_t(i) * 6;
                    out[0] = p.x; out[1] = p.y; out[2] = p.z;
                    out[3] = v.x; out[4] = v.y; out[5] = v.z;
                }
                print_results(data, "Bodies");
            }
        }
    }

    int failed = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int localFailure = 0;
        double kinetic = 0.0;
        #pragma omp parallel for reduction(max:localFailure) reduction(+:kinetic)
        for (int i = 0; i < count; ++i) {
            const Vec3 p = positions[first + i], v = localVelocity[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                localFailure = 3;
            else if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) > 1e6)
                localFailure = std::max(localFailure, 2);
            else if (std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)}) > 1e6)
                localFailure = std::max(localFailure, 1);
            kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
        }
        MPI_Allreduce(&localFailure, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!failed) {
            double energy = kinetic + potentialEnergy(positions, numBodies, first, count), total;
            MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Final energy: %.6f\nValidation: PASSED\n", total);
        } else if (rank == 0) {
            const char* messages[] = {"", "body velocity exceeds reasonable bounds",
                                      "body position exceeds reasonable bounds", "found NaN or Inf value in body state"};
            printf("Validation failed: %s\nValidation: FAILED\n", messages[failed]);
        }
    }
    checkCuda(cudaFree(velocity));
    checkCuda(cudaFree(next));
    checkCuda(cudaFree(current));
    checkCuda(cudaFreeHost(positions));
    MPI_Type_free(&xyz);
    MPI_Finalize();
    return failed ? 1 : 0;
}
