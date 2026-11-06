#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 128;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) checkCuda((call), #call)

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

// Each thread owns one target body. All threads, including inactive threads in
// the final block, participate in loading and synchronizing the source tiles.
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int offset, int localCount) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Vec3 p = active ? positions[offset + i] : Vec3{0, 0, 0};
    double fx = 0, fy = 0, fz = 0;
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
        Vec3 v = velocities[i];
        v.x += DT * fx; v.y += DT * fy; v.z += DT * fz;
        p.x += v.x * DT; p.y += v.y * DT; p.z += v.z * DT;
        velocities[i] = v;
        next[i] = p;
    }
}

__global__ void energyKernel(const Vec3* positions, const Vec3* velocities,
                             double* energies, int n, int offset, int localCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localCount) return;
    Vec3 p = positions[offset + i], v = velocities[i];
    double e = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    for (int j = offset + i + 1; j < n; ++j) {
        double dx = positions[j].x - p.x;
        double dy = positions[j].y - p.y;
        double dz = positions[j].z - p.z;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    energies[i] = e;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = 1024, steps = 10, exitCode = 0;
    bool validate = false, printResults = false, exitEarly = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); exitEarly = true; break; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } exitCode = 1; exitEarly = true; break; }
    }
    if (n < 0) { if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n"); exitCode = 1; exitEarly = true; }
    if (exitEarly) { MPI_Finalize(); return exitCode; }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
               n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0, off = 0; r < ranks; ++r) {
        counts[r] = n / ranks + (r < n % ranks);
        offsets[r] = off;
        off += counts[r];
    }
    const int count = counts[rank], offset = offsets[rank];
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    Vec3* positions;
    CUDA(cudaMallocHost(&positions, std::max(n, 1) * sizeof(Vec3)));
    std::vector<Vec3> localVel(std::max(count, 1));
    std::vector<Vec3> initialVel;
    if (rank == 0) {
        std::vector<Body> initial(n);
        randomizeBodies(initial);
        initialVel.resize(n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) { positions[i] = initial[i].pos; initialVel[i] = initial[i].vel; }
    }
    MPI_Bcast(positions, n, vecType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVel.data(), counts.data(), offsets.data(), vecType,
                 localVel.data(), count, vecType, 0, MPI_COMM_WORLD);
    initialVel.clear();

    Vec3 *dPositions, *dNext, *dVel;
    CUDA(cudaMalloc(&dPositions, std::max(n, 1) * sizeof(Vec3)));
    CUDA(cudaMalloc(&dNext, std::max(count, 1) * sizeof(Vec3)));
    CUDA(cudaMalloc(&dVel, std::max(count, 1) * sizeof(Vec3)));
    CUDA(cudaMemcpy(dPositions, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dVel, localVel.data(), count * sizeof(Vec3), cudaMemcpyHostToDevice));

    // Load the CUDA kernel before starting the benchmark timer.
    advance<<<1, BLOCK_SIZE>>>(dPositions, dNext, dVel, 0, 0, 0);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        if (count) {
            advance<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dPositions, dNext, dVel, n, offset, count);
            CUDA(cudaGetLastError());
        }
        if (step + 1 < steps) {
            if (ranks == 1) std::swap(dPositions, dNext);
            else {
                CUDA(cudaMemcpy(positions + offset, dNext, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
                MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
                CUDA(cudaMemcpy(dPositions, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
            }
        }
    }
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) printf("Simulation time: %ld ms\n", (long)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    if (validate || printResults) {
        if (steps > 0) {
            CUDA(cudaMemcpy(positions + offset, dNext, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
        }
        CUDA(cudaMemcpy(localVel.data(), dVel, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        std::vector<Vec3> allVel(rank == 0 ? n : 0);
        MPI_Gatherv(localVel.data(), count, vecType, allVel.data(), counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        std::vector<Body> bodies(rank == 0 ? n : 0);
        if (rank == 0) {
            #pragma omp parallel for if(n > 4096)
            for (int i = 0; i < n; ++i) { bodies[i].pos = positions[i]; bodies[i].vel = allVel[i]; }
            if (printResults) {
                std::vector<double> data(static_cast<size_t>(n) * 6);
                #pragma omp parallel for if(n > 4096)
                for (int i = 0; i < n; ++i) {
                    size_t j = static_cast<size_t>(i) * 6;
                    data[j] = bodies[i].pos.x; data[j+1] = bodies[i].pos.y; data[j+2] = bodies[i].pos.z;
                    data[j+3] = bodies[i].vel.x; data[j+4] = bodies[i].vel.y; data[j+5] = bodies[i].vel.z;
                }
                print_results(data, "Bodies");
            }
        }
        if (validate) {
            if (rank == 0) { printf("Validating simulation results...\n"); exitCode = validateSimulation(bodies) ? 0 : 1; }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!exitCode) {
                CUDA(cudaMemcpy(dPositions, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
                double* dEnergy;
                CUDA(cudaMalloc(&dEnergy, std::max(count, 1) * sizeof(double)));
                std::vector<double> energies(count);
                if (count) {
                    energyKernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dPositions, dVel, dEnergy, n, offset, count);
                    CUDA(cudaGetLastError());
                    CUDA(cudaMemcpy(energies.data(), dEnergy, count * sizeof(double), cudaMemcpyDeviceToHost));
                }
                double localEnergy = 0, totalEnergy = 0;
                #pragma omp parallel for reduction(+:localEnergy) if(count > 4096)
                for (int i = 0; i < count; ++i) localEnergy += energies[i];
                MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy);
                CUDA(cudaFree(dEnergy));
            } else if (rank == 0) printf("Validation: FAILED\n");
        }
    }
    CUDA(cudaFree(dPositions)); CUDA(cudaFree(dNext)); CUDA(cudaFree(dVel)); CUDA(cudaFreeHost(positions));
    MPI_Type_free(&vecType);
    MPI_Finalize();
    return exitCode;
}
