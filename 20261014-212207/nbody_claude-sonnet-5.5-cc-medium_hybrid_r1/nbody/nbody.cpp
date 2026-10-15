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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

constexpr int BLOCK = 128;

// Positions are stored as double4 (w unused) for vectorized, coalesced access.
// Each rank owns bodies [lo, lo+cnt). Forces on owned bodies are computed from
// all positions (tiled through shared memory), velocities and positions of the
// owned bodies are then updated.
__global__ void stepKernel(const double4* __restrict__ pos, double4* __restrict__ newPos,
                           double4* __restrict__ vel, int n, int lo, int cnt) {
    __shared__ double4 tile[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const int gi = lo + li;
    double4 pi = make_double4(0, 0, 0, 0);
    if (li < cnt) pi = pos[gi];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        __syncthreads();
        if (j < n) tile[threadIdx.x] = pos[j];
        __syncthreads();
        const int m = min(BLOCK, n - base);
#pragma unroll 8
        for (int k = 0; k < m; ++k) {
            const double dx = tile[k].x - pi.x;
            const double dy = tile[k].y - pi.y;
            const double dz = tile[k].z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
    }

    if (li < cnt) {
        double4 v = vel[li];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[li] = v;
        pi.x += v.x * DT;
        pi.y += v.y * DT;
        pi.z += v.z * DT;
        newPos[li] = pi;
    }
}

// Per-body potential energy contribution (each pair counted from both sides, halved).
__global__ void energyKernel(const double4* __restrict__ pos, double* __restrict__ out,
                             int n, int lo, int cnt) {
    __shared__ double4 tile[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const int gi = lo + li;
    double4 pi = make_double4(0, 0, 0, 0);
    if (li < cnt) pi = pos[gi];
    double e = 0.0;
    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        __syncthreads();
        if (j < n) tile[threadIdx.x] = pos[j];
        __syncthreads();
        const int m = min(BLOCK, n - base);
        for (int k = 0; k < m; ++k) {
            if (base + k == gi) continue;
            const double dx = tile[k].x - pi.x;
            const double dy = tile[k].y - pi.y;
            const double dz = tile[k].z - pi.z;
            e += 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    if (li < cnt) out[li] = -0.5 * e;
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
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind one GPU per rank (round-robin over ranks on the same node)
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

    // Initialize bodies (identical on every rank; the RNG stream is sequential)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block partition of bodies over ranks
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const long long n = numBodies;
        const long long a = n * r / size, b = n * (r + 1) / size;
        displs[r] = (int)a;
        counts[r] = (int)(b - a);
    }
    const int lo = displs[rank], cnt = counts[rank];
    std::vector<int> mpiCounts(size), mpiDispls(size);  // in units of double4 (32 B)
    for (int r = 0; r < size; ++r) {
        mpiCounts[r] = counts[r] * 4;
        mpiDispls[r] = displs[r] * 4;
    }

    const size_t n = numBodies;
    double4* hPos;   // full position array (pinned)
    double4* hVel;   // owned velocities (pinned)
    CUDA_CHECK(cudaMallocHost(&hPos, (n + 1) * sizeof(double4)));
    CUDA_CHECK(cudaMallocHost(&hVel, (cnt + 1) * sizeof(double4)));
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < numBodies; ++i) {
        hPos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < cnt; ++i) {
        const Body& b = bodies[lo + i];
        hVel[i] = make_double4(b.vel.x, b.vel.y, b.vel.z, 0.0);
    }

    double4 *dPos, *dNewPos, *dVel;
    CUDA_CHECK(cudaMalloc(&dPos, (n + 1) * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&dNewPos, (cnt + 1) * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&dVel, (cnt + 1) * sizeof(double4)));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, n * sizeof(double4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, cnt * sizeof(double4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int grid = (cnt + BLOCK - 1) / BLOCK;
    for (int step = 0; step < numSteps; ++step) {
        if (cnt > 0) stepKernel<<<grid, BLOCK>>>(dPos, dNewPos, dVel, numBodies, lo, cnt);
        if (size == 1) {
            CUDA_CHECK(cudaMemcpy(dPos, dNewPos, cnt * sizeof(double4), cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(hPos + lo, dNewPos, cnt * sizeof(double4), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, mpiCounts.data(),
                           mpiDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(dPos, hPos, n * sizeof(double4), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Final state: positions are replicated; gather velocities on rank 0
    CUDA_CHECK(cudaMemcpy(hVel, dVel, cnt * sizeof(double4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPos, dPos, n * sizeof(double4), cudaMemcpyDeviceToHost));
    std::vector<double4> allVel;
    if (rank == 0) allVel.resize(n + 1);
    MPI_Gatherv(hVel, cnt * 4, MPI_DOUBLE, allVel.data(), mpiCounts.data(), mpiDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Potential energy: distributed over ranks, computed on GPU
    double potential = 0.0;
    if (validate) {
        double* dE;
        CUDA_CHECK(cudaMalloc(&dE, (cnt + 1) * sizeof(double)));
        if (cnt > 0) energyKernel<<<grid, BLOCK>>>(dPos, dE, numBodies, lo, cnt);
        std::vector<double> e(cnt);
        CUDA_CHECK(cudaMemcpy(e.data(), dE, cnt * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dE));
        double local = 0.0;
#pragma omp parallel for reduction(+ : local)
        for (int i = 0; i < cnt; ++i) local += e[i];
        MPI_Reduce(&local, &potential, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    int rc = 0;
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hPos[i].x, hPos[i].y, hPos[i].z);
            bodies[i].vel = Vec3(allVel[i].x, allVel[i].y, allVel[i].z);
        }

        // Print results for external validation
        if (printResults) {
            std::vector<double> bodyData(n * 6);
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < numBodies; ++i) {
                bodyData[6 * i + 0] = bodies[i].pos.x;
                bodyData[6 * i + 1] = bodies[i].pos.y;
                bodyData[6 * i + 2] = bodies[i].pos.z;
                bodyData[6 * i + 3] = bodies[i].vel.x;
                bodyData[6 * i + 4] = bodies[i].vel.y;
                bodyData[6 * i + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                double kinetic = 0.0;
#pragma omp parallel for reduction(+ : kinetic)
                for (long long i = 0; i < numBodies; ++i) {
                    const Vec3& v = bodies[i].vel;
                    kinetic += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
                }
                printf("Final energy: %.6f\n", kinetic + potential);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    cudaFree(dPos);
    cudaFree(dNewPos);
    cudaFree(dVel);
    cudaFreeHost(hPos);
    cudaFreeHost(hVel);
    MPI_Finalize();
    return rc;
}
