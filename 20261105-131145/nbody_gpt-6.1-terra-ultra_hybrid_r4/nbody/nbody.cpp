#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be packed for MPI");

static void cudaCheck(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

// Every thread accumulates sources in their original order. All threads, including
// those outside the final partial block, participate in the tile barriers.
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int first, int count) {
    extern __shared__ Vec3 tile[];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    Vec3 p = active ? positions[first + local] : Vec3{0, 0, 0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int size = min((int)blockDim.x, n - base);
        if (active) {
            for (int k = 0; k < size; ++k) {
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
        Vec3 v = velocities[local];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        p.x += v.x * DT; p.y += v.y * DT; p.z += v.z * DT;
        velocities[local] = v;
        next[first + local] = p;
    }
}

static void printUsage(const char* progName) {
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI_Allgatherv counts and displacements are expressed in doubles.
    if (n < 0 || n > INT_MAX / 3) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be between 0 and %d\n", INT_MAX / 3);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        int amount = n / ranks + (r < n % ranks);
        int offset = r * (n / ranks) + std::min(r, n % ranks);
        counts[r] = 3 * amount;
        displs[r] = 3 * offset;
    }
    const int count = counts[rank] / 3, first = displs[rank] / 3;
    const size_t positionBytes = size_t(n) * sizeof(Vec3);
    const size_t velocityBytes = size_t(count) * sizeof(Vec3);

    // Pinned staging buffers work with both CUDA-aware and ordinary MPI stacks.
    // Allocate at least one element to keep zero-count MPI buffers valid.
    Vec3 *positions, *velocities;
    CUDA_CHECK(cudaMallocHost(&positions, std::max(positionBytes, sizeof(Vec3))));
    CUDA_CHECK(cudaMallocHost(&velocities, std::max(velocityBytes, sizeof(Vec3))));
    std::vector<Body> initial(n);
    unsigned int seed = 42;
    for (int i = 0; i < n; ++i) {
        Vec3 &p = initial[i].pos, &v = initial[i].vel;
        p.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        p.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        p.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        positions[i] = initial[i].pos;
        if (i >= first && i < first + count) velocities[i - first] = initial[i].vel;
    }
    std::vector<Body>().swap(initial);
    Vec3 *dPos, *dNext, *dVel;
    CUDA_CHECK(cudaMalloc(&dPos, std::max(positionBytes, sizeof(Vec3))));
    CUDA_CHECK(cudaMalloc(&dNext, std::max(positionBytes, sizeof(Vec3))));
    CUDA_CHECK(cudaMalloc(&dVel, std::max(velocityBytes, sizeof(Vec3))));
    if (n) CUDA_CHECK(cudaMemcpy(dPos, positions, positionBytes, cudaMemcpyHostToDevice));
    if (count) CUDA_CHECK(cudaMemcpy(dVel, velocities, velocityBytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    constexpr int blockSize = 128;
    for (int step = 0; step < steps; ++step) {
        if (count) {
            advance<<<(count + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(Vec3)>>>(
                dPos, dNext, dVel, n, first, count);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(dPos, dNext);
        } else {
            if (count) CUDA_CHECK(cudaMemcpy(positions + first, dNext + first, velocityBytes, cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, positions, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            if (n && step + 1 < steps) CUDA_CHECK(cudaMemcpy(dPos, positions, positionBytes, cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", (long)(maxElapsed * 1000.0));

    if (printResults || validate) {
        if (ranks == 1 && n && steps > 0) CUDA_CHECK(cudaMemcpy(positions, dPos, positionBytes, cudaMemcpyDeviceToHost));
        if (count) CUDA_CHECK(cudaMemcpy(velocities, dVel, velocityBytes, cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        std::vector<Vec3> allVel(rank == 0 ? n : 0);
        MPI_Gatherv(velocities, 3 * count, MPI_DOUBLE, allVel.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> bodyData(size_t(n) * 6);
            #pragma omp parallel for if(n >= 4096)
            for (int i = 0; i < n; ++i) {
                bodyData[6 * size_t(i)] = positions[i].x;
                bodyData[6 * size_t(i) + 1] = positions[i].y;
                bodyData[6 * size_t(i) + 2] = positions[i].z;
                bodyData[6 * size_t(i) + 3] = allVel[i].x;
                bodyData[6 * size_t(i) + 4] = allVel[i].y;
                bodyData[6 * size_t(i) + 5] = allVel[i].z;
            }
            print_results(bodyData, "Bodies");
        }
    }
    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int firstBad = n;
        #pragma omp parallel for reduction(min:firstBad) if(count >= 4096)
        for (int i = 0; i < count; ++i) {
            const Vec3 p = positions[first + i], v = velocities[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z) ||
                std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6 ||
                std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6)
                firstBad = std::min(firstBad, first + i);
        }
        int globalBad;
        MPI_Allreduce(&firstBad, &globalBad, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (globalBad < n) {
            int reason = 0;
            if (globalBad >= first && globalBad < first + count) {
                Vec3 p = positions[globalBad], v = velocities[globalBad - first];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                    !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) reason = 1;
                else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) reason = 2;
                else reason = 3;
            }
            int globalReason;
            MPI_Reduce(&reason, &globalReason, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                const char* messages[] = {"", "found NaN or Inf value in body state", "body position exceeds reasonable bounds", "body velocity exceeds reasonable bounds"};
                printf("Validation failed: %s\nValidation: FAILED\n", messages[globalReason]);
            }
            status = 1;
        } else {
            double energy = 0.0;
            #pragma omp parallel for reduction(+:energy) if(count >= 64)
            for (int i = 0; i < count; ++i) {
                const Vec3 v = velocities[i];
                energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
            }
            // Interleave pair rows across ranks and threads to balance the
            // triangular potential-energy calculation.
            #pragma omp parallel for reduction(+:energy) schedule(static,1) if(n >= 64)
            for (int i = rank; i < n; i += ranks) {
                const Vec3 p = positions[i];
                double e = 0.0;
                for (int j = i + 1; j < n; ++j) {
                    const double dx = positions[j].x - p.x, dy = positions[j].y - p.y, dz = positions[j].z - p.z;
                    e -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
                energy += e;
            }
            double totalEnergy;
            MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy);
        }
    }
    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dNext));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(positions));
    CUDA_CHECK(cudaFreeHost(velocities));
    MPI_Finalize();
    return status;
}
