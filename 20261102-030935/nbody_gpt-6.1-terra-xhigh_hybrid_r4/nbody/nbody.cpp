#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 128;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };
static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be contiguous");

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// All threads take part in loading each tile, including threads beyond the
// local body count, so the barriers are valid for partially filled blocks.
__global__ void stepKernel(const Vec3* positions, Vec3* newPositions,
                           Vec3* velocities, int n, int first, int count) {
    __shared__ Vec3 tile[TILE];
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = local < count;
    Vec3 p = active ? positions[first + local] : Vec3{0.0, 0.0, 0.0};
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = positions[j];
        __syncthreads();
        const int length = min(TILE, n - base);
        if (active) {
            for (int k = 0; k < length; ++k) {
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
        v.x += DT * fx;
        v.y += DT * fy;
        v.z += DT * fz;
        velocities[local] = v;
        newPositions[local] = Vec3{p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
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

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); parseStatus = 1; break; }
        else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            parseStatus = 2;
            break;
        }
    }
    if (n < 0) {
        if (rank == 0) std::fprintf(stderr, "Number of bodies must be nonnegative\n");
        parseStatus = 2;
    }
    if (parseStatus) { MPI_Finalize(); return parseStatus == 1 ? 0 : 1; }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = n / ranks + (r < n % ranks ? 1 : 0);
        offsets[r] = r * (n / ranks) + std::min(r, n % ranks);
    }
    const int count = counts[rank], first = offsets[rank];
    MPI_Datatype vecType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vecType);
    MPI_Type_commit(&vecType);

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    n, steps, validate ? "enabled" : "disabled");
    }
    // Generate the original sequence on every rank, including velocities for
    // other ranks, to retain the exact rand_r initialization order.
    Vec3* positions;
    CUDA(cudaMallocHost(&positions, std::max(n, 1) * sizeof(Vec3)));
    std::vector<Vec3> localVel(std::max(count, 1));
    std::vector<Body> initial(n);
    unsigned int seed = 42;
    for (int i = 0; i < n; ++i) {
        Vec3& p = initial[i].pos;
        Vec3& v = initial[i].vel;
        p.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        p.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        p.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        v.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
    #pragma omp parallel for schedule(static) if(n > 4096)
    for (int i = 0; i < n; ++i) {
        positions[i] = initial[i].pos;
        if (i >= first && i - first < count) localVel[i - first] = initial[i].vel;
    }
    std::vector<Body>().swap(initial);
    Vec3 *devicePos, *deviceNext, *deviceVel;
    CUDA(cudaMalloc(&devicePos, std::max(n, 1) * sizeof(Vec3)));
    CUDA(cudaMalloc(&deviceNext, std::max(count, 1) * sizeof(Vec3)));
    CUDA(cudaMalloc(&deviceVel, std::max(count, 1) * sizeof(Vec3)));
    if (n) CUDA(cudaMemcpy(devicePos, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
    if (count) CUDA(cudaMemcpy(deviceVel, localVel.data(), count * sizeof(Vec3), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < steps; ++step) {
        if (count) {
            stepKernel<<<(count - 1) / TILE + 1, TILE>>>(devicePos, deviceNext, deviceVel, n, first, count);
            CUDA(cudaGetLastError());
        }
        if (ranks == 1) {
            std::swap(devicePos, deviceNext);
        } else if (step + 1 < steps) {
            if (count) CUDA(cudaMemcpy(positions + first, deviceNext, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
            if (n) CUDA(cudaMemcpy(devicePos, positions, n * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) std::printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    int status = 0;
    if (printResults || validate) {
        if (count) {
            const Vec3* finalPos = (steps > 0 && ranks > 1) ? deviceNext : devicePos + first;
            CUDA(cudaMemcpy(positions + first, finalPos, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(localVel.data(), deviceVel, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) MPI_Allgatherv(MPI_IN_PLACE, 0, vecType, positions, counts.data(), offsets.data(), vecType, MPI_COMM_WORLD);
        std::vector<Vec3> allVel(rank == 0 ? std::max(n, 1) : 1);
        MPI_Gatherv(localVel.data(), count, vecType, allVel.data(), counts.data(), offsets.data(), vecType, 0, MPI_COMM_WORLD);
        if (rank == 0 && printResults) {
            std::vector<double> data(static_cast<size_t>(n) * 6);
            #pragma omp parallel for schedule(static) if(n > 4096)
            for (int i = 0; i < n; ++i) {
                data[6ULL*i] = positions[i].x; data[6ULL*i+1] = positions[i].y; data[6ULL*i+2] = positions[i].z;
                data[6ULL*i+3] = allVel[i].x; data[6ULL*i+4] = allVel[i].y; data[6ULL*i+5] = allVel[i].z;
            }
            print_results(data, "Bodies");
        }
        if (validate) {
            if (rank == 0) {
                std::printf("Validating simulation results...\n");
                for (int i = 0; i < n; ++i) {
                    const Vec3 p = positions[i], v = allVel[i];
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                        !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                        std::printf("Validation failed: found NaN or Inf value in body state\n"); status = 1; break;
                    }
                    if (std::abs(p.x) > 1e6 || std::abs(p.y) > 1e6 || std::abs(p.z) > 1e6) {
                        std::printf("Validation failed: body position exceeds reasonable bounds\n"); status = 1; break;
                    }
                    if (std::abs(v.x) > 1e6 || std::abs(v.y) > 1e6 || std::abs(v.z) > 1e6) {
                        std::printf("Validation failed: body velocity exceeds reasonable bounds\n"); status = 1; break;
                    }
                }
            }
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (!status) {
                double energy = 0.0;
                #pragma omp parallel for reduction(+:energy) schedule(static) if(count > 256)
                for (int i = 0; i < count; ++i) {
                    const Vec3 v = localVel[i];
                    energy += 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
                }
                // Round-robin rows keep the triangular potential-energy work
                // balanced across ranks; dynamic scheduling does so for threads.
                #pragma omp parallel for reduction(+:energy) schedule(dynamic, 16) if(n > 256)
                for (int i = rank; i < n; i += ranks) {
                    for (int j = i + 1; j < n; ++j) {
                        const double dx = positions[j].x - positions[i].x;
                        const double dy = positions[j].y - positions[i].y;
                        const double dz = positions[j].z - positions[i].z;
                        energy -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                    }
                }
                double totalEnergy = 0.0;
                MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) std::printf("Final energy: %.6f\nValidation: PASSED\n", totalEnergy);
            } else if (rank == 0) std::printf("Validation: FAILED\n");
        }
    }
    CUDA(cudaFree(devicePos)); CUDA(cudaFree(deviceNext)); CUDA(cudaFree(deviceVel));
    CUDA(cudaFreeHost(positions));
    MPI_Type_free(&vecType);
    MPI_Finalize();
    return status;
}
