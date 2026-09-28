#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Number of bodies whose force accumulation is interleaved in the inner loop.
// Each of the LANES accumulators sums over j in ascending order exactly as the
// scalar version does, so results are bit-identical while the compiler is free
// to place the lanes into SIMD registers.
constexpr int LANES = 8;

// Block decomposition of [0, n) over `size` ranks: the first (n % size) ranks
// get one extra element.
struct Partition {
    std::vector<int> counts; // number of bodies owned by each rank
    std::vector<int> displs; // first body index owned by each rank
};

static Partition partitionBodies(const int n, const int size) {
    Partition p;
    p.counts.resize(size);
    p.displs.resize(size);
    const int base = n / size;
    const int rem = n % size;
    int off = 0;
    for (int r = 0; r < size; ++r) {
        p.counts[r] = base + (r < rem ? 1 : 0);
        p.displs[r] = off;
        off += p.counts[r];
    }
    return p;
}

// Positions are kept replicated (interleaved x,y,z) on every rank; velocities
// only for the locally owned block of bodies.
void randomizeBodies(double* pos, double* vel, const int n, const int begin, const int localN,
                     unsigned int seed = 42) {
    // The RNG stream is sequential, so every rank replays it in full and keeps
    // the positions plus its own slice of the velocities.
    for (int i = 0; i < n; ++i) {
        pos[3 * i + 0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[3 * i + 1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[3 * i + 2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const double vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        const int l = i - begin;
        if (l >= 0 && l < localN) {
            vel[3 * l + 0] = vx;
            vel[3 * l + 1] = vy;
            vel[3 * l + 2] = vz;
        }
    }
}

void computeForces(const double* __restrict__ pos, const int n, const int begin, const int localN,
                   double* __restrict__ vel) {
    for (int l = 0; l < localN; l += LANES) {
        // Lanes past the end of the local block duplicate the last body; their
        // results are simply not stored. Keeping a single code path for all
        // bodies makes the result independent of the block size.
        const int m = std::min(LANES, localN - l);

        double xi[LANES], yi[LANES], zi[LANES];
        double Fx[LANES] = {}, Fy[LANES] = {}, Fz[LANES] = {};

        for (int k = 0; k < LANES; ++k) {
            const int i = begin + l + (k < m ? k : m - 1);
            xi[k] = pos[3 * i + 0];
            yi[k] = pos[3 * i + 1];
            zi[k] = pos[3 * i + 2];
        }

        for (int j = 0; j < n; ++j) {
            const double xj = pos[3 * j + 0];
            const double yj = pos[3 * j + 1];
            const double zj = pos[3 * j + 2];

            for (int k = 0; k < LANES; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }

        for (int k = 0; k < m; ++k) {
            // The volatile temporaries keep the compiler from fusing the
            // multiply into the following add, matching the rounding of the
            // scalar reference implementation. This is outside the O(n) inner
            // loop, so it costs nothing measurable.
            const volatile double dvx = DT * Fx[k];
            const volatile double dvy = DT * Fy[k];
            const volatile double dvz = DT * Fz[k];
            vel[3 * (l + k) + 0] += dvx;
            vel[3 * (l + k) + 1] += dvy;
            vel[3 * (l + k) + 2] += dvz;
        }
    }
}

void integrateBodies(double* __restrict__ pos, const int begin, const int localN, const double* __restrict__ vel) {
    for (int l = 0; l < localN; ++l) {
        pos[3 * (begin + l) + 0] += vel[3 * l + 0] * DT;
        pos[3 * (begin + l) + 1] += vel[3 * l + 1] * DT;
        pos[3 * (begin + l) + 2] += vel[3 * l + 2] * DT;
    }
}

// Distributed total energy: every rank accumulates the contributions of its own
// block of bodies (in ascending index order, as in the serial code) and the
// block sums are added up in rank order on rank 0.
double computeTotalEnergy(const double* pos, const int n, const int begin, const int localN, const double* vel,
                          const int rank, const int size, MPI_Comm comm) {
    double kinetic = 0.0;
    for (int l = 0; l < localN; ++l) {
        kinetic += 0.5 * (vel[3 * l + 0] * vel[3 * l + 0] + vel[3 * l + 1] * vel[3 * l + 1] +
                          vel[3 * l + 2] * vel[3 * l + 2]);
    }

    // The kinetic part is completed in body order, exactly as in the serial
    // code, and rank 0 then keeps accumulating the potential terms on top of it.
    std::vector<double> partials(rank == 0 ? (size_t)size : 0);
    MPI_Gather(&kinetic, 1, MPI_DOUBLE, partials.data(), 1, MPI_DOUBLE, 0, comm);
    double running = 0.0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            running += partials[r];
        }
    }

    // Potential energy (assuming unit mass for all bodies). The pair (i, j)
    // with i < j is evaluated by the owner of i.
    double potential = running;
    for (int l = 0; l < localN; ++l) {
        const int i = begin + l;
        const double xi = pos[3 * i + 0];
        const double yi = pos[3 * i + 1];
        const double zi = pos[3 * i + 2];
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos[3 * j + 0] - xi;
            const double dy = pos[3 * j + 1] - yi;
            const double dz = pos[3 * j + 2] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }

    MPI_Gather(&potential, 1, MPI_DOUBLE, partials.data(), 1, MPI_DOUBLE, 0, comm);

    double energy = 0.0;
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            energy += partials[r];
        }
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values.
// Returns the index of the first offending body and the kind of violation;
// checks the locally owned bodies only.
static void validateLocal(const double* pos, const int begin, const int localN, const double* vel, int& firstBad,
                          int& reason) {
    firstBad = std::numeric_limits<int>::max();
    reason = 0;

    for (int l = 0; l < localN; ++l) {
        const double px = pos[3 * (begin + l) + 0], py = pos[3 * (begin + l) + 1], pz = pos[3 * (begin + l) + 2];
        const double vx = vel[3 * l + 0], vy = vel[3 * l + 1], vz = vel[3 * l + 2];

        // Check for NaN or Inf values
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) || !std::isfinite(vx) ||
            !std::isfinite(vy) || !std::isfinite(vz)) {
            firstBad = begin + l;
            reason = 1;
            return;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px) > maxPos || std::abs(py) > maxPos || std::abs(pz) > maxPos) {
            firstBad = begin + l;
            reason = 2;
            return;
        }
        if (std::abs(vx) > maxVel || std::abs(vy) > maxVel || std::abs(vz) > maxVel) {
            firstBad = begin + l;
            reason = 3;
            return;
        }
    }
}

bool validateSimulation(const double* pos, const int begin, const int localN, const double* vel, const int rank,
                        MPI_Comm comm) {
    int firstBad = 0, reason = 0;
    validateLocal(pos, begin, localN, vel, firstBad, reason);

    // Report the violation of the lowest-numbered body, as the serial code does.
    int in[2] = {firstBad, reason};
    int out[2] = {0, 0};
    MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MINLOC, comm);

    if (out[0] == std::numeric_limits<int>::max()) {
        return true;
    }
    if (rank == 0) {
        if (out[1] == 1) {
            printf("Validation failed: found NaN or Inf value in body state\n");
        } else if (out[1] == 2) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
        } else {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
        }
    }
    return false;
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

    int rank = 0, size = 1;
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (numBodies < 0) numBodies = 0;
    // More ranks than bodies: the surplus ranks idle in the collectives.
    const Partition part = partitionBodies(numBodies, size);
    const int begin = part.displs[rank];
    const int localN = part.counts[rank];

    // Communication of the replicated position array: one element per body.
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    // Initialize bodies
    std::vector<double> pos((size_t)numBodies * 3);
    std::vector<double> vel((size_t)localN * 3);
    randomizeBodies(pos.data(), vel.data(), numBodies, begin, localN);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), numBodies, begin, localN, vel.data());
        integrateBodies(pos.data(), begin, localN, vel.data());
        // Republish the locally updated positions to all ranks.
        MPI_Allgatherv(MPI_IN_PLACE, localN, vec3Type, pos.data(), part.counts.data(), part.displs.data(), vec3Type,
                       MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Print results for external validation
    if (printResults) {
        // Collect the distributed velocities on rank 0
        std::vector<double> allVel(rank == 0 ? (size_t)numBodies * 3 : 0);
        MPI_Gatherv(vel.data(), localN, vec3Type, allVel.data(), part.counts.data(), part.displs.data(), vec3Type, 0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve((size_t)numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(pos[3 * i + 0]);
                bodyData.push_back(pos[3 * i + 1]);
                bodyData.push_back(pos[3 * i + 2]);
                bodyData.push_back(allVel[3 * i + 0]);
                bodyData.push_back(allVel[3 * i + 1]);
                bodyData.push_back(allVel[3 * i + 2]);
            }
            print_results(bodyData, "Bodies");
        }
    }

    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        if (validateSimulation(pos.data(), begin, localN, vel.data(), rank, MPI_COMM_WORLD)) {
            // Report final energy for reference
            double finalEnergy =
                computeTotalEnergy(pos.data(), numBodies, begin, localN, vel.data(), rank, size, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            status = 1;
        }
    }

    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return status;
}
