#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be packed as six doubles");

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

// Computes the force accumulators for B consecutive bodies starting at index i0.
// Each body's sum is accumulated over j = 0..n-1 in increasing order, exactly as in the
// serial reference, so results are bit-identical regardless of how bodies are distributed.
// Processing B bodies per pass lets the compiler vectorize across bodies (not across j,
// which would reorder the floating-point reduction).
// Every body goes through this same kernel (partial blocks are padded), so the generated
// arithmetic, including any FP contraction, is identical for all bodies.
template <size_t B>
static inline void forceBlock(const double* __restrict px, const double* __restrict py,
                              const double* __restrict pz, size_t n, const size_t* idx,
                              double* __restrict Fx, double* __restrict Fy, double* __restrict Fz) {
    double xi[B], yi[B], zi[B];
    double fx[B], fy[B], fz[B];
    for (size_t k = 0; k < B; ++k) {
        xi[k] = px[idx[k]];
        yi[k] = py[idx[k]];
        zi[k] = pz[idx[k]];
        fx[k] = 0.0;
        fy[k] = 0.0;
        fz[k] = 0.0;
    }

    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j];
        const double yj = py[j];
        const double zj = pz[j];
        for (size_t k = 0; k < B; ++k) {
            const double dx = xj - xi[k];
            const double dy = yj - yi[k];
            const double dz = zj - zi[k];
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            fx[k] += dx * invDist3;
            fy[k] += dy * invDist3;
            fz[k] += dz * invDist3;
        }
    }

    for (size_t k = 0; k < B; ++k) {
        Fx[k] = fx[k];
        Fy[k] = fy[k];
        Fz[k] = fz[k];
    }
}

// Updates velocities of bodies in [first, first + count) using the positions of all bodies.
// bodies must hold the globally consistent state at the start of the step.
void computeForces(std::vector<Body>& bodies, size_t first, size_t count,
                   const std::vector<double>& px, const std::vector<double>& py,
                   const std::vector<double>& pz) {
    const size_t n = bodies.size();
    constexpr size_t BLOCK = 8;
    size_t idx[BLOCK];
    double Fx[BLOCK], Fy[BLOCK], Fz[BLOCK];

    for (size_t local = 0; local < count; local += BLOCK) {
        const size_t valid = (count - local < BLOCK) ? (count - local) : BLOCK;
        // Pad a partial block with repeats of the last valid body; padded results are discarded.
        for (size_t k = 0; k < BLOCK; ++k) {
            idx[k] = first + local + (k < valid ? k : valid - 1);
        }
        forceBlock<BLOCK>(px.data(), py.data(), pz.data(), n, idx, Fx, Fy, Fz);
        for (size_t k = 0; k < valid; ++k) {
            Body& b = bodies[idx[k]];
            b.vel.x += DT * Fx[k];
            b.vel.y += DT * Fy[k];
            b.vel.z += DT * Fz[k];
        }
    }
}

void integrateBodies(std::vector<Body>& bodies, size_t first, size_t count) {
    for (size_t i = first; i < first + count; ++i) {
        Body& body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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
bool validateSimulation(const std::vector<Body>& bodies, bool report) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            if (report) printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            if (report) printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            if (report) printf("Validation failed: body velocity exceeds reasonable bounds\n");
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

// Contiguous block partition of n items over `size` ranks.
void blockPartition(size_t n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    const size_t base = n / size;
    const size_t rem = n % size;
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(offset);
        offset += c;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numBodies < 0) {
        if (root) printf("Number of bodies must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Every rank holds the full, globally consistent body state. Initialization is
    // deterministic, so all ranks generate identical data without communication.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Each rank owns a contiguous block of bodies whose forces/velocities it computes.
    std::vector<int> counts, displs;
    blockPartition(static_cast<size_t>(numBodies), size, counts, displs);
    const size_t first = static_cast<size_t>(displs[rank]);
    const size_t count = static_cast<size_t>(counts[rank]);

    // Element-count arrays for exchanging full Body state (6 doubles per body).
    std::vector<int> dcounts(size), ddispls(size);
    for (int r = 0; r < size; ++r) {
        dcounts[r] = counts[r] * 6;
        ddispls[r] = displs[r] * 6;
    }

    std::vector<double> px(numBodies), py(numBodies), pz(numBodies);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Gather a structure-of-arrays copy of positions for vectorizable inner loops.
        for (int i = 0; i < numBodies; ++i) {
            px[i] = bodies[i].pos.x;
            py[i] = bodies[i].pos.y;
            pz[i] = bodies[i].pos.z;
        }

        computeForces(bodies, first, count, px, py, pz);
        integrateBodies(bodies, first, count);

        // Share updated state of every rank's block with all ranks.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE,
                       bodies.data(), dcounts.data(), ddispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) printf("Simulation time: %ld ms\n", global_ms);

    // Print results for external validation
    if (printResults && root) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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

    // Validation: check that simulation produces finite, reasonable values.
    // The state is identical on all ranks, so every rank reaches the same verdict.
    if (validate) {
        if (root) printf("Validating simulation results...\n");

        if (validateSimulation(bodies, root)) {
            // Report final energy for reference
            if (root) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
