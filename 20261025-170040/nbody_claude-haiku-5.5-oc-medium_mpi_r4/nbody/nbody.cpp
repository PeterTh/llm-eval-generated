#include <mpi.h>

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

// Number of target bodies processed together by the force kernel. Each lane keeps
// its own accumulator and sums over j in the same order as the serial algorithm,
// so results are bitwise identical regardless of how many ranks are used.
constexpr size_t TILE = 8;

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

// Computes forces acting on bodies [begin, end) from all n bodies and updates
// their velocities. Positions of all bodies are supplied in SoA form (xs, ys, zs).
// localVel[k] corresponds to body begin + k.
void computeForcesLocal(const double* __restrict xs, const double* __restrict ys,
                        const double* __restrict zs, const size_t n,
                        const size_t begin, const size_t end, Vec3* __restrict localVel) {
    for (size_t i0 = begin; i0 < end; i0 += TILE) {
        const size_t cnt = std::min(TILE, end - i0);

        double xi[TILE], yi[TILE], zi[TILE];
        double Fx[TILE], Fy[TILE], Fz[TILE];
        for (size_t t = 0; t < TILE; ++t) {
            const size_t idx = i0 + (t < cnt ? t : 0);  // padding lanes duplicate a valid body
            xi[t] = xs[idx];
            yi[t] = ys[idx];
            zi[t] = zs[idx];
            Fx[t] = 0.0;
            Fy[t] = 0.0;
            Fz[t] = 0.0;
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = xs[j];
            const double yj = ys[j];
            const double zj = zs[j];
            for (size_t t = 0; t < TILE; ++t) {
                const double dx = xj - xi[t];
                const double dy = yj - yi[t];
                const double dz = zj - zi[t];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[t] += dx * invDist3;
                Fy[t] += dy * invDist3;
                Fz[t] += dz * invDist3;
            }
        }

        for (size_t t = 0; t < cnt; ++t) {
            Vec3& v = localVel[i0 - begin + t];
            v.x += DT * Fx[t];
            v.y += DT * Fy[t];
            v.z += DT * Fz[t];
        }
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
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

int runBenchmark(int argc, char** argv, int rank, int size) {
    const bool isRoot = (rank == 0);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on every rank)
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
            if (isRoot) printUsage(argv[0]);
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // Initialize bodies. The generator is deterministic, so every rank produces the
    // same initial state and no broadcast is needed.
    std::vector<Body> init(n);
    randomizeBodies(init);

    // Block partition of bodies across ranks
    const size_t un = static_cast<size_t>(size);
    const size_t ur = static_cast<size_t>(rank);
    const size_t base = n / un;
    const size_t rem = n % un;
    const size_t begin = ur * base + std::min(ur, rem);
    const size_t end = begin + base + (ur < rem ? 1 : 0);
    const size_t localCount = end - begin;

    // Full position array (kept in sync on all ranks) and velocities of local bodies
    std::vector<Vec3> allPos(n);
    for (size_t i = 0; i < n; ++i) allPos[i] = init[i].pos;
    std::vector<Vec3> localVel(localCount);
    for (size_t k = 0; k < localCount; ++k) localVel[k] = init[begin + k].vel;

    // Per-rank counts/displacements (in units of Vec3)
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t ub = static_cast<size_t>(r);
        const size_t cnt = base + (ub < rem ? 1 : 0);
        const size_t disp = ub * base + std::min(ub, rem);
        counts[r] = static_cast<int>(cnt);
        displs[r] = static_cast<int>(disp);
    }

    // Datatype representing one Vec3 (3 contiguous doubles)
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    std::vector<double> xs(n), ys(n), zs(n);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        for (size_t i = 0; i < n; ++i) {
            xs[i] = allPos[i].x;
            ys[i] = allPos[i].y;
            zs[i] = allPos[i].z;
        }

        computeForcesLocal(xs.data(), ys.data(), zs.data(), n, begin, end, localVel.data());

        for (size_t i = begin; i < end; ++i) {
            const Vec3& v = localVel[i - begin];
            allPos[i].x += v.x * DT;
            allPos[i].y += v.y * DT;
            allPos[i].z += v.z * DT;
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, allPos.data(), counts.data(), displs.data(),
                       vec3Type, MPI_COMM_WORLD);
    }

    auto end_t = std::chrono::high_resolution_clock::now();
    const double localSec = std::chrono::duration<double>(end_t - start).count();
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather final velocities on the root
    std::vector<Vec3> allVel(isRoot ? n : 0);
    MPI_Gatherv(localVel.data(), static_cast<int>(localCount), vec3Type,
                allVel.data(), counts.data(), displs.data(), vec3Type, 0, MPI_COMM_WORLD);

    MPI_Type_free(&vec3Type);

    if (!isRoot) {
        return 0;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(maxSec));
    printf("Simulation time: %ld ms\n", duration.count());

    std::vector<Body> bodies(n);
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = allPos[i];
        bodies[i].vel = allVel[i];
    }
    
    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
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
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ret = runBenchmark(argc, argv, rank, size);

    // Propagate the root's exit status to all ranks
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return ret;
}
