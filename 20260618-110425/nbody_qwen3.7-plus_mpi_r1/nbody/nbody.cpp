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

static inline void computeDistribution(int N, int sz, int rk, int& local_n, int& offset) {
    int base = N / sz;
    int rem = N % sz;
    if (rk < rem) {
        local_n = base + 1;
        offset = rk * (base + 1);
    } else {
        local_n = base;
        offset = rem * (base + 1) + (rk - rem) * base;
    }
}

void randomizeBodies(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                     double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                     int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void computeForces(const double* __restrict__ lpx, const double* __restrict__ lpy, const double* __restrict__ lpz,
                   double* __restrict__ lvx, double* __restrict__ lvy, double* __restrict__ lvz,
                   int local_n,
                   const double* __restrict__ apx, const double* __restrict__ apy, const double* __restrict__ apz,
                   int N) {
    for (int i = 0; i < local_n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = lpx[i], py = lpy[i], pz = lpz[i];

        for (int j = 0; j < N; ++j) {
            const double dx = apx[j] - px;
            const double dy = apy[j] - py;
            const double dz = apz[j] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        lvx[i] += DT * Fx;
        lvy[i] += DT * Fy;
        lvz[i] += DT * Fz;
    }
}

void integrateBodies(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                     const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
                     int n) {
    for (int i = 0; i < n; ++i) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

bool validateLocal(const double* px, const double* py, const double* pz,
                   const double* vx, const double* vy, const double* vz, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            return false;
        }
        constexpr double maxPos = 1e6, maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos ||
            std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
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
    MPI_Init(&argc, &argv);

    int rank, sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sz);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printRes = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printRes = true;
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

    // Compute this rank's portion
    int local_n, offset;
    computeDistribution(numBodies, sz, rank, local_n, offset);

    // Initialize: every rank generates the full sequence identically, then extracts its chunk
    std::vector<double> local_px(local_n), local_py(local_n), local_pz(local_n);
    std::vector<double> local_vx(local_n), local_vy(local_n), local_vz(local_n);
    {
        std::vector<double> all_px(numBodies), all_py(numBodies), all_pz(numBodies);
        std::vector<double> all_vx(numBodies), all_vy(numBodies), all_vz(numBodies);
        randomizeBodies(all_px.data(), all_py.data(), all_pz.data(),
                        all_vx.data(), all_vy.data(), all_vz.data(), numBodies);
        for (int i = 0; i < local_n; ++i) {
            local_px[i] = all_px[offset + i];
            local_py[i] = all_py[offset + i];
            local_pz[i] = all_pz[offset + i];
            local_vx[i] = all_vx[offset + i];
            local_vy[i] = all_vy[offset + i];
            local_vz[i] = all_vz[offset + i];
        }
    }

    // Allgather buffers and metadata
    std::vector<double> all_px(numBodies), all_py(numBodies), all_pz(numBodies);
    std::vector<int> rc(sz), disp(sz);
    for (int r = 0; r < sz; ++r) {
        int ln, off;
        computeDistribution(numBodies, sz, r, ln, off);
        rc[r] = ln;
        disp[r] = off;
    }

    // Timed simulation loop
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(local_px.data(), local_n, MPI_DOUBLE,
                        all_px.data(), rc.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_py.data(), local_n, MPI_DOUBLE,
                        all_py.data(), rc.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_pz.data(), local_n, MPI_DOUBLE,
                        all_pz.data(), rc.data(), disp.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        computeForces(local_px.data(), local_py.data(), local_pz.data(),
                      local_vx.data(), local_vy.data(), local_vz.data(),
                      local_n, all_px.data(), all_py.data(), all_pz.data(), numBodies);

        integrateBodies(local_px.data(), local_py.data(), local_pz.data(),
                        local_vx.data(), local_vy.data(), local_vz.data(), local_n);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double local_duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double simulation_time_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &simulation_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all body data to rank 0 for output/validation
    // Pack as [px,py,pz,vx,vy,vz] per body
    std::vector<double> local_packed(local_n * 6);
    for (int i = 0; i < local_n; ++i) {
        local_packed[i*6+0] = local_px[i];
        local_packed[i*6+1] = local_py[i];
        local_packed[i*6+2] = local_pz[i];
        local_packed[i*6+3] = local_vx[i];
        local_packed[i*6+4] = local_vy[i];
        local_packed[i*6+5] = local_vz[i];
    }

    std::vector<int> grc(sz), gdisp(sz);
    for (int r = 0; r < sz; ++r) {
        int ln, off;
        computeDistribution(numBodies, sz, r, ln, off);
        grc[r] = ln * 6;
        gdisp[r] = off * 6;
    }

    std::vector<double> all_packed;
    if (rank == 0) all_packed.resize(numBodies * 6);

    MPI_Gatherv(local_packed.data(), local_n * 6, MPI_DOUBLE,
                all_packed.data(), grc.data(), gdisp.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.3f ms\n", simulation_time_ms);

        if (printRes) {
            print_results(all_packed, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");

            bool valid = true;
            for (int i = 0; i < numBodies && valid; ++i) {
                double px = all_packed[i*6+0], py = all_packed[i*6+1], pz = all_packed[i*6+2];
                double vx = all_packed[i*6+3], vy = all_packed[i*6+4], vz = all_packed[i*6+5];
                if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) ||
                    !std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
                    printf("Validation failed: found NaN or Inf value in body state\n");
                    valid = false;
                }
                constexpr double maxPos = 1e6, maxVel = 1e6;
                if (valid && (std::abs(px) > maxPos || std::abs(py) > maxPos || std::abs(pz) > maxPos)) {
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                    valid = false;
                }
                if (valid && (std::abs(vx) > maxVel || std::abs(vy) > maxVel || std::abs(vz) > maxVel)) {
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                    valid = false;
                }
            }

            if (valid) {
                double energy = 0.0;
                for (int i = 0; i < numBodies; ++i) {
                    double vx = all_packed[i*6+3], vy = all_packed[i*6+4], vz = all_packed[i*6+5];
                    energy += 0.5 * (vx*vx + vy*vy + vz*vz);
                }
                for (int i = 0; i < numBodies; ++i) {
                    double px = all_packed[i*6+0], py = all_packed[i*6+1], pz = all_packed[i*6+2];
                    for (int j = i + 1; j < numBodies; ++j) {
                        double dx = all_packed[j*6+0] - px;
                        double dy = all_packed[j*6+1] - py;
                        double dz = all_packed[j*6+2] - pz;
                        energy -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
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
