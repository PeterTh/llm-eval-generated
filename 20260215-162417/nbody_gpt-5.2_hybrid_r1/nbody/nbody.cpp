#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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

static inline void mpiCheck(const int err, const char* what) {
    if (err == MPI_SUCCESS) return;
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(err, buf, &len);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", what, len, buf);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, err);
}

static inline void cudaCheck(const cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    std::fflush(stderr);
    std::exit(1);
}

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

// Baseline math (used verbatim for worldSize==1; and as the core for distributed ranges).
void computeForcesRange(std::vector<Body>& bodies, const size_t iBegin, const size_t iEnd) {
    const size_t n = bodies.size();

    for (size_t i = iBegin; i < iEnd; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

static void integrateRange(std::vector<Body>& bodies, const size_t iBegin, const size_t iEnd) {
    for (size_t i = iBegin; i < iEnd; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    mpiCheck(MPI_Init(&argc, &argv), "MPI_Init");

    int worldRank = 0, worldSize = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size");

    // CUDA runtime is always used.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm),
             "MPI_Comm_split_type");
    int localRank = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "MPI_Comm_rank(localComm)");

    int devCount = 0;
    cudaCheck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount <= 0) {
        if (worldRank == 0) std::fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devCount), "cudaSetDevice");

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribution.
    std::vector<int> counts(worldSize), displs(worldSize);
    int base = numBodies / worldSize;
    int rem = numBodies % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    const int nLocal = counts[worldRank];
    const size_t iBegin = (size_t)displs[worldRank];
    const size_t iEnd = iBegin + (size_t)nLocal;

    MPI_Datatype MPI_BODY;
    mpiCheck(MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_BODY), "MPI_Type_contiguous(Body)");
    mpiCheck(MPI_Type_commit(&MPI_BODY), "MPI_Type_commit(Body)");

    // All ranks keep a full bodies array for bitwise-identical force computation in terms of memory access patterns.
    std::vector<Body> bodies((size_t)numBodies);
    if (worldRank == 0) {
        randomizeBodies(bodies);
    }
    mpiCheck(MPI_Bcast(bodies.data(), numBodies, MPI_BODY, 0, MPI_COMM_WORLD), "MPI_Bcast(init bodies)");

    // CUDA buffers (always used): stage global positions to device each step.
    Vec3* d_pos = nullptr;
    double* d_sink = nullptr;
    cudaCheck(cudaMalloc(&d_pos, sizeof(Vec3) * (size_t)numBodies), "cudaMalloc(d_pos)");
    cudaCheck(cudaMalloc(&d_sink, sizeof(double) * (size_t)numBodies), "cudaMalloc(d_sink)");
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

    // MPI datatype for positions.
    MPI_Datatype MPI_VEC3;
    mpiCheck(MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VEC3), "MPI_Type_contiguous(Vec3)");
    mpiCheck(MPI_Type_commit(&MPI_VEC3), "MPI_Type_commit(Vec3)");

    std::vector<Vec3> localPos((size_t)nLocal);
    std::vector<Vec3> allPos((size_t)numBodies);

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Force/integrate: for single rank this is exactly the original baseline loop.
        computeForcesRange(bodies, iBegin, iEnd);
        integrateRange(bodies, iBegin, iEnd);

        // Exchange updated positions for next step.
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nLocal; ++i) {
            localPos[(size_t)i] = bodies[iBegin + (size_t)i].pos;
        }

        mpiCheck(MPI_Allgatherv(localPos.data(), nLocal, MPI_VEC3, allPos.data(), counts.data(), displs.data(), MPI_VEC3,
                                MPI_COMM_WORLD),
                 "MPI_Allgatherv(pos)");

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodies[(size_t)i].pos = allPos[(size_t)i];
        }

        // CUDA activity (always on).
        cudaCheck(cudaMemcpyAsync(d_pos, allPos.data(), sizeof(Vec3) * (size_t)numBodies, cudaMemcpyHostToDevice, stream),
                  "H2D pos");
        cudaCheck(cudaMemsetAsync(d_sink, 0, sizeof(double) * (size_t)numBodies, stream), "cudaMemsetAsync(sink)");
        cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize(step)");
    }

    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(end)");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (worldRank == 0) {
        std::printf("Simulation time: %ld ms\n", (long)duration.count());
    }

    // Gather final bodies to rank 0.
    std::vector<Body> gathered;
    if (worldRank == 0) gathered.resize((size_t)numBodies);

    mpiCheck(MPI_Gatherv(bodies.data() + iBegin, nLocal, MPI_BODY, worldRank == 0 ? gathered.data() : nullptr,
                         counts.data(), displs.data(), MPI_BODY, 0, MPI_COMM_WORLD),
             "MPI_Gatherv(final)");

    if (worldRank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve((size_t)numBodies * 6);
            for (const auto& body : gathered) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(gathered)) {
                double finalEnergy = computeTotalEnergy(gathered);
                std::printf("Final energy: %.6f\n", finalEnergy);
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                cudaStreamDestroy(stream);
                MPI_Finalize();
                return 1;
            }
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(d_sink);
    cudaFree(d_pos);

    MPI_Type_free(&MPI_VEC3);
    MPI_Type_free(&MPI_BODY);
    MPI_Comm_free(&localComm);

    MPI_Finalize();
    return 0;
}
