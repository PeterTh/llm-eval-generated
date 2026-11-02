#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
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
static_assert(sizeof(Vec3) == 3 * sizeof(double), "MPI position datatype requires packed coordinates");

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
        std::fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// The output positions are separate from the input snapshot: no body can see
// another body's newly integrated position during the force calculation.
__global__ void advance(const Vec3* positions, Vec3* next, Vec3* velocities,
                        int n, int offset, int localCount) {
    __shared__ Vec3 tile[BLOCK_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < localCount;
    Vec3 p = active ? positions[offset + i] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += BLOCK_SIZE) {
        size_t j = base + threadIdx.x;
        if (j < static_cast<size_t>(n)) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int count = static_cast<int>(min(static_cast<size_t>(BLOCK_SIZE), static_cast<size_t>(n) - base));
        if (active) {
            for (int k = 0; k < count; ++k) {
                double dx = tile[k].x - p.x;
                double dy = tile[k].y - p.y;
                double dz = tile[k].z - p.z;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
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
        velocities[i] = v;
        next[i] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
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
        if (rank == 0) std::fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 1024, steps = 10, status = 0;
    bool validate = false, results = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) results = true;
        else if (std::strcmp(argv[i], "-h") == 0) { help = true; break; }
        else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            status = 1; break;
        }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize(); return status;
    }
    if (n < 0) {
        if (rank == 0) std::fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize(); return 1;
    }
    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    n, steps, validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = n / ranks + (r < n % ranks);
        offsets[r] = r * (n / ranks) + std::min(r, n % ranks);
    }
    const int localCount = counts[rank], offset = offsets[rank];
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    Vec3* positions;
    CUDA_CHECK(cudaMallocHost(&positions, std::max(1, n) * sizeof(Vec3)));
    std::vector<Vec3> initialVel(rank == 0 ? n : 0);
    std::vector<Vec3> localVel(std::max(1, localCount));
    if (rank == 0) {
        // rand_r is a sequential stream; preserve the exact original order.
        unsigned int seed = 42;
        std::vector<Body> bodies(n);
        for (int i = 0; i < n; ++i) {
            bodies[i].pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            bodies[i].pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            bodies[i].vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
        #pragma omp parallel for schedule(static) if(n >= 4096)
        for (int i = 0; i < n; ++i) {
            positions[i] = bodies[i].pos;
            initialVel[i] = bodies[i].vel;
        }
    }
    MPI_Bcast(positions, n, vecType, 0, MPI_COMM_WORLD);
    MPI_Scatterv(initialVel.data(), counts.data(), offsets.data(), vecType,
                 localVel.data(), localCount, vecType, 0, MPI_COMM_WORLD);
    initialVel.clear();

    Vec3 *dPositions, *dNext, *dVel;
    CUDA_CHECK(cudaMalloc(&dPositions, std::max(1, n) * sizeof(Vec3)));
    // For a single rank, both position buffers contain the entire system.
    CUDA_CHECK(cudaMalloc(&dNext, std::max(1, localCount) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&dVel, std::max(1, localCount) * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(dPositions, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, localVel.data(), localCount * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        if (localCount) {
            advance<<<(localCount - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dPositions, dNext, dVel, n, offset, localCount);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(dPositions, dNext);
        } else if (step + 1 < steps || validate || results) {
            CUDA_CHECK(cudaMemcpy(positions + offset, dNext, localCount * sizeof(Vec3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
            if (step + 1 < steps)
                CUDA_CHECK(cudaMemcpy(dPositions, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    if (validate || results) {
        if (ranks == 1 && steps > 0)
            CUDA_CHECK(cudaMemcpy(positions, dPositions, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localVel.data(), dVel, localCount * sizeof(Vec3), cudaMemcpyDeviceToHost));
        std::vector<Vec3> allVel(rank == 0 ? n : 0);
        MPI_Gatherv(localVel.data(), localCount, vecType, allVel.data(), counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (results && rank == 0) {
            std::vector<double> data(static_cast<size_t>(n) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                data[6ull*i] = positions[i].x; data[6ull*i+1] = positions[i].y; data[6ull*i+2] = positions[i].z;
                data[6ull*i+3] = allVel[i].x; data[6ull*i+4] = allVel[i].y; data[6ull*i+5] = allVel[i].z;
            }
            print_results(data, "Bodies");
        }
        if (validate) {
            int failure = 0;
            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                // Preserve the original first-failure ordering and messages.
                for (int i = 0; i < n && !failure; ++i) {
                    const Vec3 p = positions[i], v = allVel[i];
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                        !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) failure = 1;
                    else if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) failure = 2;
                    else if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) failure = 3;
                }
                if (failure == 1) std::printf("Validation failed: found NaN or Inf value in body state\n");
                else if (failure == 2) std::printf("Validation failed: body position exceeds reasonable bounds\n");
                else if (failure == 3) std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            }
            MPI_Bcast(&failure, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!failure) {
                double energy = 0.0;
                #pragma omp parallel for reduction(+:energy) schedule(static)
                for (int i = 0; i < localCount; ++i) {
                    Vec3 v = localVel[i];
                    energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
                }
                // Cyclic rows balance the triangular potential-energy calculation
                // across ranks; dynamic scheduling balances it across CPU threads.
                #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16)
                for (long long i = rank; i < n; i += ranks) {
                    Vec3 p = positions[i];
                    for (long long j = i + 1; j < n; ++j) {
                        double dx = positions[j].x - p.x, dy = positions[j].y - p.y, dz = positions[j].z - p.z;
                        energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    }
                }
                double totalEnergy = 0.0;
                MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) std::printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy);
            } else {
                if (rank == 0) std::printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }
    CUDA_CHECK(cudaFree(dPositions)); CUDA_CHECK(cudaFree(dNext)); CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFreeHost(positions));
    MPI_Type_free(&vecType);
    MPI_Finalize();
    return status;
}
