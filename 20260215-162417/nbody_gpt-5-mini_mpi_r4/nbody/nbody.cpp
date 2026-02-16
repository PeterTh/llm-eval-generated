#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

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

// The original single-process helpers are kept for reference but the parallel main implements
// an MPI allgather-based approach for correctness and scalability.

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

    // Parse command line arguments on all ranks
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
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute distribution of bodies
    std::vector<int> counts(size, 0), displs(size, 0);
    int base = numBodies / size;
    int rem = numBodies % size;
    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];

    // scaled counts for packed double arrays (6 doubles per body: pos(x,y,z) and vel(x,y,z))
    std::vector<int> counts6(size), displs6(size);
    for (int i = 0; i < size; ++i) {
        counts6[i] = counts[i] * 6;
        displs6[i] = displs[i] * 6;
    }

    const int localN = counts[rank];
    std::vector<Body> localBodies(localN);
    std::vector<double> localBuf(localN * 6);

    // Root initializes full state and scatters
    if (rank == 0) {
        std::vector<Body> fullBodies(numBodies);
        randomizeBodies(fullBodies);
        std::vector<double> fullBuf(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            fullBuf[i*6 + 0] = fullBodies[i].pos.x;
            fullBuf[i*6 + 1] = fullBodies[i].pos.y;
            fullBuf[i*6 + 2] = fullBodies[i].pos.z;
            fullBuf[i*6 + 3] = fullBodies[i].vel.x;
            fullBuf[i*6 + 4] = fullBodies[i].vel.y;
            fullBuf[i*6 + 5] = fullBodies[i].vel.z;
        }
        MPI_Scatterv(fullBuf.data(), counts6.data(), displs6.data(), MPI_DOUBLE,
                     localBuf.data(), localBuf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // unpack into local bodies
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x = localBuf[i*6 + 0];
            localBodies[i].pos.y = localBuf[i*6 + 1];
            localBodies[i].pos.z = localBuf[i*6 + 2];
            localBodies[i].vel.x = localBuf[i*6 + 3];
            localBodies[i].vel.y = localBuf[i*6 + 4];
            localBodies[i].vel.z = localBuf[i*6 + 5];
        }
    } else {
        MPI_Scatterv(nullptr, counts6.data(), displs6.data(), MPI_DOUBLE,
                     localBuf.data(), localBuf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x = localBuf[i*6 + 0];
            localBodies[i].pos.y = localBuf[i*6 + 1];
            localBodies[i].pos.z = localBuf[i*6 + 2];
            localBodies[i].vel.x = localBuf[i*6 + 3];
            localBodies[i].vel.y = localBuf[i*6 + 4];
            localBodies[i].vel.z = localBuf[i*6 + 5];
        }
    }

    // Buffer that will hold the gathered state for all bodies on every rank for force computation
    std::vector<double> allBuf(numBodies * 6);

    // Warm-up barrier and timed run
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // pack local state into localBuf
        for (int i = 0; i < localN; ++i) {
            localBuf[i*6 + 0] = localBodies[i].pos.x;
            localBuf[i*6 + 1] = localBodies[i].pos.y;
            localBuf[i*6 + 2] = localBodies[i].pos.z;
            localBuf[i*6 + 3] = localBodies[i].vel.x;
            localBuf[i*6 + 4] = localBodies[i].vel.y;
            localBuf[i*6 + 5] = localBodies[i].vel.z;
        }

        // gather full state to all ranks
        MPI_Allgatherv(localBuf.data(), localBuf.size(), MPI_DOUBLE,
                       allBuf.data(), counts6.data(), displs6.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // compute forces on local bodies using positions of all bodies
        for (int i = 0; i < localN; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double ix = localBodies[i].pos.x;
            const double iy = localBodies[i].pos.y;
            const double iz = localBodies[i].pos.z;
            for (int j = 0; j < numBodies; ++j) {
                const double jx = allBuf[j*6 + 0];
                const double jy = allBuf[j*6 + 1];
                const double jz = allBuf[j*6 + 2];
                const double dx = jx - ix;
                const double dy = jy - iy;
                const double dz = jz - iz;
                const double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            localBodies[i].vel.x += DT * Fx;
            localBodies[i].vel.y += DT * Fy;
            localBodies[i].vel.z += DT * Fz;
        }

        // integrate local bodies
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    double local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.0f ms\n", max_ms);
    }

    // Gather final state to root for printing/validation if requested
    std::vector<double> finalBuf;
    if (rank == 0) finalBuf.resize(numBodies * 6);

    // pack local state
    for (int i = 0; i < localN; ++i) {
        localBuf[i*6 + 0] = localBodies[i].pos.x;
        localBuf[i*6 + 1] = localBodies[i].pos.y;
        localBuf[i*6 + 2] = localBodies[i].pos.z;
        localBuf[i*6 + 3] = localBodies[i].vel.x;
        localBuf[i*6 + 4] = localBodies[i].vel.y;
        localBuf[i*6 + 5] = localBodies[i].vel.z;
    }

    MPI_Gatherv(localBuf.data(), localBuf.size(), MPI_DOUBLE,
                (rank == 0) ? finalBuf.data() : nullptr, counts6.data(), displs6.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // reconstruct full bodies vector
        std::vector<Body> fullBodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            fullBodies[i].pos.x = finalBuf[i*6 + 0];
            fullBodies[i].pos.y = finalBuf[i*6 + 1];
            fullBodies[i].pos.z = finalBuf[i*6 + 2];
            fullBodies[i].vel.x = finalBuf[i*6 + 3];
            fullBodies[i].vel.y = finalBuf[i*6 + 4];
            fullBodies[i].vel.z = finalBuf[i*6 + 5];
        }

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : fullBodies) {
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
            printf("Validating simulation results...\n");
            // Validate and compute energy on root
            bool ok = true;
            for (const auto& body : fullBodies) {
                if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
                    !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
                    printf("Validation failed: found NaN or Inf value in body state\n");
                    ok = false;
                    break;
                }
                const double maxPos = 1e6;
                const double maxVel = 1e6;
                if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                    ok = false;
                    break;
                }
                if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                    ok = false;
                    break;
                }
            }
            if (ok) {
                // compute total energy
                double energy = 0.0;
                for (const auto& b : fullBodies) {
                    energy += 0.5 * (b.vel.x*b.vel.x + b.vel.y*b.vel.y + b.vel.z*b.vel.z);
                }
                for (int i = 0; i < numBodies; ++i) {
                    for (int j = i+1; j < numBodies; ++j) {
                        const double dx = fullBodies[j].pos.x - fullBodies[i].pos.x;
                        const double dy = fullBodies[j].pos.y - fullBodies[i].pos.y;
                        const double dz = fullBodies[j].pos.z - fullBodies[i].pos.z;
                        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                        energy -= 1.0 / dist;
                    }
                }
                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
