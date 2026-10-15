#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

// Computes forces on bodies [offset, offset+count) from all n bodies (positions packed xyz),
// updates their velocities and integrates their positions. Same j-order as the serial code.
__global__ void stepKernel(const double* __restrict__ allPos, double* __restrict__ newPos,
                           double* __restrict__ vel, int n, int offset, int count) {
    __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = li < count;
    const int gi = offset + (active ? li : 0);
    const double px = allPos[3 * gi], py = allPos[3 * gi + 1], pz = allPos[3 * gi + 2];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = allPos[3 * j];
            sy[threadIdx.x] = allPos[3 * j + 1];
            sz[threadIdx.x] = allPos[3 * j + 2];
        }
        __syncthreads();
        const int lim = min(BLOCK, n - base);
#pragma unroll 4
        for (int k = 0; k < lim; ++k) {
            const double dx = sx[k] - px;
            const double dy = sy[k] - py;
            const double dz = sz[k] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (active) {
        const double vx = vel[3 * li] + DT * Fx;
        const double vy = vel[3 * li + 1] + DT * Fy;
        const double vz = vel[3 * li + 2] + DT * Fz;
        vel[3 * li] = vx;
        vel[3 * li + 1] = vy;
        vel[3 * li + 2] = vz;
        newPos[3 * li] = px + vx * DT;
        newPos[3 * li + 1] = py + vy * DT;
        newPos[3 * li + 2] = pz + vz * DT;
    }
}

// Potential + kinetic energy of the full system; each rank handles a strided share of rows,
// OpenMP threads share the rows within a rank. pos is packed xyz (all bodies).
double computeTotalEnergy(const std::vector<double>& pos, const std::vector<double>& vel,
                          size_t n, int rank, int nranks) {
    double energy = 0.0;
    if (rank == 0) {
#pragma omp parallel for reduction(+ : energy) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            energy += 0.5 * (vel[3 * i] * vel[3 * i] + vel[3 * i + 1] * vel[3 * i + 1] +
                             vel[3 * i + 2] * vel[3 * i + 2]);
        }
    }
    double pot = 0.0;
#pragma omp parallel for reduction(+ : pot) schedule(dynamic, 16)
    for (size_t i = rank; i < n; i += nranks) {
        double local = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = pos[3 * j] - pos[3 * i];
            const double dy = pos[3 * j + 1] - pos[3 * i + 1];
            const double dz = pos[3 * j + 2] - pos[3 * i + 2];
            local -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        pot += local;
    }
    energy += pot;
    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<double>& data) {
    const double maxVal = 1e6;  // bodies shouldn't fly off to infinity
    const size_t n = data.size() / 6;
    int bad = 0;
#pragma omp parallel for reduction(max : bad)
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 6; ++k) {
            const double v = data[6 * i + k];
            if (!std::isfinite(v)) bad = std::max(bad, 1);
            else if (std::abs(v) > maxVal) bad = std::max(bad, k < 3 ? 2 : 3);
        }
    }
    if (bad == 1) printf("Validation failed: found NaN or Inf value in body state\n");
    else if (bad == 2) printf("Validation failed: body position exceeds reasonable bounds\n");
    else if (bad == 3) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    // Select GPU: one per rank, round-robin over the node-local ranks
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank, ndev;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Initialize bodies (identically on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block distribution of bodies
    const int n = numBodies;
    std::vector<int> counts(nranks), displs(nranks);  // in doubles (3 per body)
    std::vector<int> bcounts(nranks), boffs(nranks);  // in bodies
    for (int r = 0; r < nranks; ++r) {
        const int base = n / nranks, rem = n % nranks;
        bcounts[r] = base + (r < rem ? 1 : 0);
        boffs[r] = r * base + std::min(r, rem);
        counts[r] = 3 * bcounts[r];
        displs[r] = 3 * boffs[r];
    }
    const int myCount = bcounts[rank], myOff = boffs[rank];

    std::vector<double> allPos(3 * (size_t)n), myVel(3 * (size_t)myCount);
    for (int i = 0; i < n; ++i) {
        allPos[3 * i] = bodies[i].pos.x;
        allPos[3 * i + 1] = bodies[i].pos.y;
        allPos[3 * i + 2] = bodies[i].pos.z;
    }
    for (int i = 0; i < myCount; ++i) {
        myVel[3 * i] = bodies[myOff + i].vel.x;
        myVel[3 * i + 1] = bodies[myOff + i].vel.y;
        myVel[3 * i + 2] = bodies[myOff + i].vel.z;
    }

    double *dAll = nullptr, *dNew = nullptr, *dVel = nullptr;
    CUDA_CHECK(cudaMalloc(&dAll, sizeof(double) * (3 * (size_t)n + 1)));
    CUDA_CHECK(cudaMalloc(&dNew, sizeof(double) * (3 * (size_t)myCount + 1)));
    CUDA_CHECK(cudaMalloc(&dVel, sizeof(double) * (3 * (size_t)myCount + 1)));
    CUDA_CHECK(cudaMemcpy(dVel, myVel.data(), sizeof(double) * myVel.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAll, allPos.data(), sizeof(double) * allPos.size(), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int grid = (myCount + BLOCK - 1) / BLOCK;
    for (int step = 0; step < numSteps; ++step) {
        if (grid > 0)
            stepKernel<<<grid, BLOCK, 0, stream>>>(dAll, dNew, dVel, n, myOff, myCount);
        CUDA_CHECK(cudaGetLastError());
        if (myCount > 0)
            CUDA_CHECK(cudaMemcpyAsync(allPos.data() + 3 * (size_t)myOff, dNew,
                                       sizeof(double) * 3 * (size_t)myCount,
                                       cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (nranks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, allPos.data(), counts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
        if (step + 1 < numSteps || validate || printResults)
            CUDA_CHECK(cudaMemcpyAsync(dAll, allPos.data(), sizeof(double) * allPos.size(),
                                       cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);

    // Gather velocities (positions are already replicated everywhere)
    CUDA_CHECK(cudaMemcpy(myVel.data(), dVel, sizeof(double) * myVel.size(), cudaMemcpyDeviceToHost));
    std::vector<double> allVel(3 * (size_t)n);
    MPI_Allgatherv(myVel.data(), 3 * myCount, MPI_DOUBLE, allVel.data(), counts.data(),
                   displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dAll));
    CUDA_CHECK(cudaFree(dNew));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::vector<double> bodyData;
    if (rank == 0 && (printResults || validate)) {
        bodyData.resize(6 * (size_t)n);
#pragma omp parallel for
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                bodyData[6 * (size_t)i + k] = allPos[3 * (size_t)i + k];
                bodyData[6 * (size_t)i + 3 + k] = allVel[3 * (size_t)i + k];
            }
    }

    // Print results for external validation
    if (printResults && rank == 0) print_results(bodyData, "Bodies");

    // Validation: check that simulation produces finite, reasonable values
    int rc = 0;
    if (validate) {
        int ok = 1;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(bodyData) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            const double finalEnergy = computeTotalEnergy(allPos, allVel, n, rank, nranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
