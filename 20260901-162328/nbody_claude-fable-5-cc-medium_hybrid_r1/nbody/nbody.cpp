#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),        \
                    __FILE__, __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

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

// Force kernel: each thread owns one local body i and accumulates the force
// from all n bodies in ascending j order (same summation order as the serial
// code), staging position tiles through shared memory.
__global__ void forcesKernel(const double3* __restrict__ pos, double3* __restrict__ vel,
                             int n, int offset, int count) {
    extern __shared__ double3 tile[];
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < count;

    double3 p = active ? pos[offset + li] : make_double3(0.0, 0.0, 0.0);
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + (int)threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = pos[j];
        }
        __syncthreads();

        const int m = min((int)blockDim.x, n - base);
        if (active) {
            for (int k = 0; k < m; ++k) {
                const double dx = tile[k].x - p.x;
                const double dy = tile[k].y - p.y;
                const double dz = tile[k].z - p.z;
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
        vel[li].x += DT * Fx;
        vel[li].y += DT * Fy;
        vel[li].z += DT * Fz;
    }
}

__global__ void integrateKernel(double3* __restrict__ pos, const double3* __restrict__ vel,
                                int offset, int count) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < count) {
        pos[offset + li].x += vel[li].x * DT;
        pos[offset + li].y += vel[li].y * DT;
        pos[offset + li].z += vel[li].z * DT;
    }
}

// Distributed total energy: each rank sums its slice of the pairwise
// potential with OpenMP; kinetic energy is added on rank 0.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int offset, int count) {
    const long long n = (long long)bodies.size();
    double energy = 0.0;

    if (rank == 0) {
        // Kinetic energy (assuming unit mass)
        double kinetic = 0.0;
#pragma omp parallel for reduction(+ : kinetic) schedule(static)
        for (long long i = 0; i < n; ++i) {
            const auto& body = bodies[i];
            kinetic += 0.5 * (body.vel.x * body.vel.x +
                              body.vel.y * body.vel.y +
                              body.vel.z * body.vel.z);
        }
        energy += kinetic;
    }

    // Potential energy (assuming unit mass for all bodies)
    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 64)
    for (long long i = offset; i < offset + count; ++i) {
        for (long long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }
    energy += potential;

    double totalEnergy = 0.0;
    MPI_Reduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return totalEnergy;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU (round-robin over the node-local devices).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int localRank = rank;
    if (const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
        localRank = atoi(lr);
    } else if (const char* slr = getenv("SLURM_LOCALID")) {
        localRank = atoi(slr);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Block decomposition: rank owns bodies [offset, offset + count).
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const long long lo = (long long)numBodies * r / nranks;
        const long long hi = (long long)numBodies * (r + 1) / nranks;
        displs[r] = (int)lo;
        counts[r] = (int)(hi - lo);
    }
    const int offset = displs[rank];
    const int count = counts[rank];
    // Counts/displacements in doubles for exchanging packed double3 positions.
    std::vector<int> counts3(nranks), displs3(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts3[r] = counts[r] * 3;
        displs3[r] = displs[r] * 3;
    }

    // Initialize bodies (identically on every rank; the RNG is sequential).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Split state into packed position/velocity arrays for the GPU and MPI.
    std::vector<double> pos(3ULL * numBodies), vel(3ULL * (count > 0 ? count : 1));
    for (int i = 0; i < numBodies; ++i) {
        pos[3ULL * i + 0] = bodies[i].pos.x;
        pos[3ULL * i + 1] = bodies[i].pos.y;
        pos[3ULL * i + 2] = bodies[i].pos.z;
    }
    for (int i = 0; i < count; ++i) {
        vel[3ULL * i + 0] = bodies[offset + i].vel.x;
        vel[3ULL * i + 1] = bodies[offset + i].vel.y;
        vel[3ULL * i + 2] = bodies[offset + i].vel.z;
    }

    double3* d_pos = nullptr;
    double3* d_vel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(double3) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_vel, sizeof(double3) * (count > 0 ? count : 1)));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), sizeof(double3) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, vel.data(), sizeof(double3) * (count > 0 ? count : 1),
                          cudaMemcpyHostToDevice));

    constexpr int BLOCK = 256;
    const int grid = (count + BLOCK - 1) / BLOCK;
    const size_t shmem = sizeof(double3) * BLOCK;

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            forcesKernel<<<grid, BLOCK, shmem>>>(d_pos, d_vel, numBodies, offset, count);
            integrateKernel<<<grid, BLOCK>>>(d_pos, d_vel, offset, count);
            CUDA_CHECK(cudaGetLastError());
            // Fetch the positions this rank just advanced.
            CUDA_CHECK(cudaMemcpy(pos.data() + 3ULL * offset, d_pos + offset,
                                  sizeof(double3) * count, cudaMemcpyDeviceToHost));
        }
        // Exchange updated positions with all ranks, then refresh the device copy.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(),
                       counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), sizeof(double3) * numBodies,
                              cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather final velocities to every rank and rebuild the body array.
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(vel.data(), d_vel, sizeof(double3) * count, cudaMemcpyDeviceToHost));
    }
    std::vector<double> allVel(3ULL * numBodies);
    MPI_Allgatherv(vel.data(), 3 * count, MPI_DOUBLE, allVel.data(),
                   counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(pos[3ULL * i + 0], pos[3ULL * i + 1], pos[3ULL * i + 2]);
        bodies[i].vel = Vec3(allVel[3ULL * i + 0], allVel[3ULL * i + 1], allVel[3ULL * i + 2]);
    }

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve((size_t)numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = (rank == 0) ? (int)validateSimulation(bodies) : 1;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference (distributed across ranks)
            double finalEnergy = computeTotalEnergy(bodies, rank, offset, count);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
