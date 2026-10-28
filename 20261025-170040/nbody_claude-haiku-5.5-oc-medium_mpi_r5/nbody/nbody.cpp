#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Number of target bodies processed together by the force kernel (SIMD lanes / independent chains).
// Each target keeps its own accumulation chain, so the summation order for every body is unchanged.
constexpr size_t kIBlock = 8;

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

// Updates velocities of bodies [i0, i0 + B). Each body sums its contributions over j = 0..n-1 in
// increasing order, exactly as the serial reference, so the results are bit-identical regardless of
// how bodies are partitioned. Lanes are independent, so the B targets are processed as SIMD lanes
// (packed sqrt/div) without reordering any per-body sum.
// The multiply-adds are explicit std::fma calls, matching the fused operations of the reference.
template <size_t B>
inline void forceBlock(const Vec3* pos, Vec3* vel, size_t n, size_t i0) {
    double xi[B], yi[B], zi[B];
    double Fx[B] = {}, Fy[B] = {}, Fz[B] = {};
    for (size_t k = 0; k < B; ++k) {
        xi[k] = pos[i0 + k].x;
        yi[k] = pos[i0 + k].y;
        zi[k] = pos[i0 + k].z;
    }

    for (size_t j = 0; j < n; ++j) {
        const double xj = pos[j].x;
        const double yj = pos[j].y;
        const double zj = pos[j].z;
        for (size_t k = 0; k < B; ++k) {
            const double dx = xj - xi[k];
            const double dy = yj - yi[k];
            const double dz = zj - zi[k];
            const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[k] = std::fma(dx, invDist3, Fx[k]);
            Fy[k] = std::fma(dy, invDist3, Fy[k]);
            Fz[k] = std::fma(dz, invDist3, Fz[k]);
        }
    }

    for (size_t k = 0; k < B; ++k) {
        vel[i0 + k].x += DT * Fx[k];
        vel[i0 + k].y += DT * Fy[k];
        vel[i0 + k].z += DT * Fz[k];
    }
}

// Computes forces on the locally owned bodies [begin, end) using the full position array.
void computeForcesRange(const std::vector<Vec3>& pos, std::vector<Vec3>& vel, size_t begin, size_t end) {
    const size_t n = pos.size();
    const Vec3* p = pos.data();
    Vec3* v = vel.data();

    size_t i = begin;
    for (; i + kIBlock <= end; i += kIBlock) {
        forceBlock<kIBlock>(p, v, n, i);
    }
    if (i + 4 <= end) {
        forceBlock<4>(p, v, n, i);
        i += 4;
    }
    for (; i < end; ++i) {
        forceBlock<1>(p, v, n, i);
    }
}

void integrateRange(std::vector<Vec3>& pos, const std::vector<Vec3>& vel, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        pos[i].x = std::fma(vel[i].x, DT, pos[i].x);
        pos[i].y = std::fma(vel[i].y, DT, pos[i].y);
        pos[i].z = std::fma(vel[i].z, DT, pos[i].z);
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

// Runs the benchmark on all ranks. Returns the process exit code (identical on every rank).
int runBenchmark(int argc, char** argv) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (every rank parses identically)
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            printf("Number of bodies must be non-negative\n");
            printUsage(argv[0]);
        }
        return 1;
    }
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = static_cast<size_t>(numBodies);

    // MPI datatype for Vec3 (three contiguous doubles)
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    // Block partition of bodies across ranks (remainder spread over the first ranks)
    std::vector<int> counts(size), displs(size);
    {
        const size_t base = n / static_cast<size_t>(size);
        const size_t rem = n % static_cast<size_t>(size);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
            displs[r] = offset;
            offset += counts[r];
        }
    }
    const size_t lo = static_cast<size_t>(displs[rank]);
    const size_t hi = lo + static_cast<size_t>(counts[rank]);

    // Initialize bodies (deterministic, so every rank starts from the same state).
    // Positions are kept fully replicated; velocities are only meaningful for owned bodies.
    std::vector<Body> initial(n);
    randomizeBodies(initial);
    std::vector<Vec3> pos(n), vel(n);
    for (size_t i = 0; i < n; ++i) {
        pos[i] = initial[i].pos;
        vel[i] = initial[i].vel;
    }
    std::vector<Body>().swap(initial);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesRange(pos, vel, lo, hi);
        integrateRange(pos, vel, lo, hi);
        MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, pos.data(), counts.data(), displs.data(), vec3Type,
                       MPI_COMM_WORLD);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
    }

    int rc = 0;
    const bool needFullState = printResults || validate;

    if (needFullState) {
        // Gather all velocities onto rank 0 (positions are already replicated everywhere)
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, vec3Type, vel.data(), counts.data(), displs.data(), vec3Type, 0,
                        MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(vel.data() + lo, counts[rank], vec3Type, nullptr, nullptr, nullptr, vec3Type, 0,
                        MPI_COMM_WORLD);
        }

        if (rank == 0) {
            std::vector<Body> bodies(n);
            for (size_t i = 0; i < n; ++i) {
                bodies[i].pos = pos[i];
                bodies[i].vel = vel[i];
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
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&vec3Type);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int rc = runBenchmark(argc, argv);
    MPI_Finalize();
    return rc;
}
