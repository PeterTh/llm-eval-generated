#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// (kept for reference semantics; init is inlined in main)
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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

constexpr int BLOCK = 128;

// Forces on bodies [offset, offset+count) from all n bodies; updates their velocity.
// pos: all bodies (x,y,z,pad); vel: local slice only.
__global__ void forceKernel(const double4* __restrict__ pos, double4* __restrict__ vel,
                            int n, int offset, int count) {
    __shared__ double4 sh[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = li < count;
    const int gi = offset + li;
    double4 pi = active ? pos[gi] : make_double4(0, 0, 0, 0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) sh[threadIdx.x] = pos[j];
        __syncthreads();
        const int lim = min(BLOCK, n - base);
        if (lim == BLOCK) {
#pragma unroll 8
            for (int k = 0; k < BLOCK; ++k) {
                const double dx = sh[k].x - pi.x;
                const double dy = sh[k].y - pi.y;
                const double dz = sh[k].z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
            for (int k = 0; k < lim; ++k) {
                const double dx = sh[k].x - pi.x;
                const double dy = sh[k].y - pi.y;
                const double dz = sh[k].z - pi.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }
    if (active) {
        double4 v = vel[li];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[li] = v;
    }
}

__global__ void integrateKernel(double4* __restrict__ pos, const double4* __restrict__ vel,
                                int offset, int count) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= count) return;
    double4 p = pos[offset + li];
    const double4 v = vel[li];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    pos[offset + li] = p;
}

// Potential energy: rows i = rank, rank+size, ... summed over j > i.
__global__ void potentialKernel(const double4* __restrict__ pos, int n, int rank, int size,
                                double* __restrict__ out) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * size + rank;
    double e = 0.0;
    if (i < n) {
        const double4 pi = pos[i];
        for (int j = i + 1; j < n; ++j) {
            const double4 pj = pos[j];
            const double dx = pj.x - pi.x, dy = pj.y - pi.y, dz = pj.z - pi.z;
            e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    for (int o = 16; o > 0; o >>= 1) e += __shfl_down_sync(0xffffffffu, e, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, e);
}

bool validateSimulation(const std::vector<double4>& pos, const std::vector<double4>& vel) {
    const size_t n = pos.size();
    int bad = 0; // 0 ok, 1 nonfinite, 2 pos bounds, 3 vel bounds
    const double maxPos = 1e6, maxVel = 1e6;
#pragma omp parallel for reduction(max : bad)
    for (size_t i = 0; i < n; ++i) {
        const double4 p = pos[i], v = vel[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
            bad = std::max(bad, 1);
        else if (std::abs(p.x) > maxPos || std::abs(p.y) > maxPos || std::abs(p.z) > maxPos)
            bad = std::max(bad, 2);
        else if (std::abs(v.x) > maxVel || std::abs(v.y) > maxVel || std::abs(v.z) > maxVel)
            bad = std::max(bad, 3);
    }
    if (bad == 1) printf("Validation failed: found NaN or Inf value in body state\n");
    if (bad == 2) printf("Validation failed: body position exceeds reasonable bounds\n");
    if (bad == 3) printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return bad == 0;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind one GPU per rank (round-robin among ranks on the same node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Initialize bodies (identical on every rank; same RNG stream as the original)
    const size_t n = numBodies;
    std::vector<double4> hpos(n), hvel(n);
    {
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            double r[6];
            for (int k = 0; k < 6; ++k) r[k] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            hpos[i] = make_double4(r[0], r[1], r[2], 0.0);
            hvel[i] = make_double4(r[3], r[4], r[5], 0.0);
        }
    }

    // Block partition of bodies over ranks
    std::vector<int> counts(size), displs(size), cnt4(size), disp4(size);
    for (int r = 0; r < size; ++r) {
        const size_t base = n / size, rem = n % size;
        counts[r] = (int)(base + (r < (int)rem ? 1 : 0));
        displs[r] = (int)(r * base + std::min<size_t>(r, rem));
        cnt4[r] = counts[r] * 4;
        disp4[r] = displs[r] * 4;
    }
    const int myCount = counts[rank], myOff = displs[rank];

    double4 *dpos = nullptr, *dvel = nullptr;
    CUDA_CHECK(cudaMalloc(&dpos, std::max<size_t>(n, 1) * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&dvel, std::max<size_t>(myCount, 1) * sizeof(double4)));
    CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), n * sizeof(double4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvel, hvel.data() + myOff, myCount * sizeof(double4), cudaMemcpyHostToDevice));

    const int blocks = (myCount + BLOCK - 1) / BLOCK;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (blocks > 0) {
            forceKernel<<<blocks, BLOCK>>>(dpos, dvel, (int)n, myOff, myCount);
            integrateKernel<<<blocks, BLOCK>>>(dpos, dvel, myOff, myCount);
        }
        // Exchange updated positions of all ranks
        CUDA_CHECK(cudaMemcpy(hpos.data() + myOff, dpos + myOff, myCount * sizeof(double4), cudaMemcpyDeviceToHost));
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hpos.data(), cnt4.data(), disp4.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            // Upload only the remote parts
            if (myOff > 0)
                CUDA_CHECK(cudaMemcpy(dpos, hpos.data(), (size_t)myOff * sizeof(double4), cudaMemcpyHostToDevice));
            const size_t tail = n - myOff - myCount;
            if (tail > 0)
                CUDA_CHECK(cudaMemcpy(dpos + myOff + myCount, hpos.data() + myOff + myCount,
                                      tail * sizeof(double4), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gather final velocities on rank 0 (positions are already replicated)
    CUDA_CHECK(cudaMemcpy(hvel.data() + myOff, dvel, myCount * sizeof(double4), cudaMemcpyDeviceToHost));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_ms);

    if (size > 1) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : hvel.data() + myOff, rank == 0 ? 0 : cnt4[rank],
                    rank == 0 ? MPI_DATATYPE_NULL : MPI_DOUBLE, hvel.data(), cnt4.data(), disp4.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData(n * 6);
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            bodyData[6 * i + 0] = hpos[i].x;
            bodyData[6 * i + 1] = hpos[i].y;
            bodyData[6 * i + 2] = hpos[i].z;
            bodyData[6 * i + 3] = hvel[i].x;
            bodyData[6 * i + 4] = hvel[i].y;
            bodyData[6 * i + 5] = hvel[i].z;
        }
        print_results(bodyData, "Bodies");
    }

    int status = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int ok = 1;
        if (rank == 0) ok = validateSimulation(hpos, hvel) ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Distributed potential energy on GPU
            double* dE;
            CUDA_CHECK(cudaMalloc(&dE, sizeof(double)));
            CUDA_CHECK(cudaMemset(dE, 0, sizeof(double)));
            const int rows = (int)((n + size - 1) / size);
            if (rows > 0) potentialKernel<<<(rows + 127) / 128, 128>>>(dpos, (int)n, rank, size, dE);
            double pot = 0.0, totalPot = 0.0;
            CUDA_CHECK(cudaMemcpy(&pot, dE, sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(dE));
            MPI_Reduce(&pot, &totalPot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                double kin = 0.0;
#pragma omp parallel for reduction(+ : kin)
                for (size_t i = 0; i < n; ++i)
                    kin += 0.5 * (hvel[i].x * hvel[i].x + hvel[i].y * hvel[i].y + hvel[i].z * hvel[i].z);
                printf("Final energy: %.6f\n", kin + totalPot);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            status = 1;
        }
    }

    cudaFree(dpos);
    cudaFree(dvel);
    MPI_Finalize();
    return status;
}
