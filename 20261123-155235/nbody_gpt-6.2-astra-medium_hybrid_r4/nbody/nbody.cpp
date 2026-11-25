#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
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
};
static_assert(sizeof(Vec3) == 3 * sizeof(double));

void check(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::abort(); // MPI calls are confined to the main thread.
    }
}

template<class T>
T* deviceBuffer(size_t count) {
    T* ptr;
    check(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

// All threads, including lanes past the last target, participate in tile loads.
template<int Tile>
__global__ void advance(const Vec3* __restrict__ positions, Vec3* next,
                        Vec3* velocities, int first, int count, int total) {
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * blockDim.x + lane;
    __shared__ Vec3 tile[Tile];
    const Vec3 p = i < count ? positions[first + i] : Vec3{};
    Vec3 force{};
    for (unsigned start = 0; start < total; start += Tile) {
        if (start + lane < total) tile[lane] = positions[start + lane];
        __syncthreads();
        const int length = min(unsigned(Tile), total - start);
        if (i < count) {
            #pragma unroll 4
            for (int j = 0; j < length; ++j) {
                const double dx = tile[j].x - p.x;
                const double dy = tile[j].y - p.y;
                const double dz = tile[j].z - p.z;
                const double dist2 = fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
                const double inv = 1.0 / sqrt(dist2);
                const double inv3 = inv * inv * inv;
                force.x += dx * inv3;
                force.y += dy * inv3;
                force.z += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (i < count) {
        Vec3 v = velocities[i];
        v.x += __dmul_rn(DT, force.x);
        v.y += __dmul_rn(DT, force.y);
        v.z += __dmul_rn(DT, force.z);
        velocities[i] = v;
        next[i] = {p.x + DT * v.x, p.y + DT * v.y, p.z + DT * v.z};
    }
}

__global__ void energyRows(const Vec3* positions, const Vec3* velocities,
                           double* energies, int first, int count, int total) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const Vec3 p = positions[first + i], v = velocities[i];
    double energy = 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    for (unsigned j = first + i + 1; j < total; ++j) {
        const double dx = positions[j].x - p.x;
        const double dy = positions[j].y - p.y;
        const double dz = positions[j].z - p.z;
        energy -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    energies[i] = energy;
}

struct Device {
    int id, first, count, multiprocessors;
    Vec3 *pos, *next, *vel;
};

int boundary(int n, int part, int parts) {
    return static_cast<long long>(n) * part / parts;
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
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n",
               numBodies, numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks, visible;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&visible));
    if (!visible) MPI_Abort(MPI_COMM_WORLD, 1);

    std::vector<int> counts(ranks), starts(ranks);
    for (int r = 0; r < ranks; ++r) {
        starts[r] = boundary(numBodies, r, ranks);
        counts[r] = boundary(numBodies, r + 1, ranks) - starts[r];
    }
    const int count = counts[rank], first = starts[rank];
    // Interleave visible GPUs among node-local ranks. A scheduler may expose
    // just one GPU per rank, in which case device zero is already the assigned GPU.
    const int assigned = localRank < visible ? 1 + (visible - 1 - localRank) / localRanks : 1;
    const int workers = std::min(std::max(count, 1), assigned);
    const bool exchange = ranks > 1 || workers > 1;
    std::vector<Device> devices(workers);
    MPI_Datatype xyz;
    MPI_Type_contiguous(3, MPI_DOUBLE, &xyz);
    MPI_Type_commit(&xyz);

    Vec3* positions;
    check(cudaMallocHost(&positions, sizeof(Vec3) * std::max(numBodies, 1)));
    std::vector<Vec3> velocities(rank == 0 ? numBodies : 0), localVel(count);
    if (rank == 0) {
        unsigned seed = 42;
        auto sample = [&]() { return 2.0 * (rand_r(&seed) / double(RAND_MAX)) - 1.0; };
        for (int i = 0; i < numBodies; ++i) {
            positions[i] = {sample(), sample(), sample()};
            velocities[i] = {sample(), sample(), sample()};
        }
    }
    MPI_Bcast(positions, numBodies, xyz, 0, MPI_COMM_WORLD);
    MPI_Scatterv(velocities.data(), counts.data(), starts.data(), xyz,
                 localVel.data(), count, xyz, 0, MPI_COMM_WORLD);

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int w = 0; w < workers; ++w) {
        Device& d = devices[w];
        d.id = (localRank + w * localRanks) % visible;
        d.first = first + boundary(count, w, workers);
        d.count = first + boundary(count, w + 1, workers) - d.first;
        check(cudaSetDevice(d.id));
        check(cudaDeviceGetAttribute(&d.multiprocessors, cudaDevAttrMultiProcessorCount, d.id));
        d.pos = deviceBuffer<Vec3>(numBodies);
        d.next = deviceBuffer<Vec3>(d.count);
        d.vel = deviceBuffer<Vec3>(d.count);
        check(cudaMemcpy(d.pos, positions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
        if (d.count) check(cudaMemcpy(d.vel, localVel.data() + (d.first - first),
                                      d.count * sizeof(Vec3), cudaMemcpyHostToDevice));
        check(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        // Finish all reads of the shared host snapshot before reusing its slices.
        if (exchange && step > 0) {
            #pragma omp parallel for num_threads(workers) schedule(static)
            for (int w = 0; w < workers; ++w) {
                Device& d = devices[w];
                if (!d.count) continue;
                check(cudaSetDevice(d.id));
                check(cudaMemcpy(d.pos, positions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
            }
        }
        #pragma omp parallel for num_threads(workers) schedule(static)
        for (int w = 0; w < workers; ++w) {
            Device& d = devices[w];
            if (!d.count) continue;
            check(cudaSetDevice(d.id));
            // Smaller blocks spread short target ranges over more multiprocessors.
            if (d.count < 128 * d.multiprocessors) {
                advance<32><<<1 + (d.count - 1) / 32, 32>>>(
                    d.pos, d.next, d.vel, d.first, d.count, numBodies);
            } else {
                advance<128><<<1 + (d.count - 1) / 128, 128>>>(
                    d.pos, d.next, d.vel, d.first, d.count, numBodies);
            }
            check(cudaGetLastError());
            if (!exchange) {
                std::swap(d.pos, d.next);
            } else if (step + 1 < numSteps || validate || printResults) {
                check(cudaMemcpy(positions + d.first, d.next, d.count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            }
            if (step + 1 == numSteps) check(cudaDeviceSynchronize());
        }
        if (ranks > 1 && (step + 1 < numSteps || validate))
            MPI_Allgatherv(MPI_IN_PLACE, 0, xyz, positions, counts.data(), starts.data(), xyz, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - start, longest;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(1000 * longest));

    std::vector<double> energy(validate ? count : 0);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int w = 0; w < workers; ++w) {
        Device& d = devices[w];
        check(cudaSetDevice(d.id));
        if (d.count && (printResults || validate)) {
            check(cudaMemcpy(localVel.data() + (d.first - first), d.vel,
                             d.count * sizeof(Vec3), cudaMemcpyDeviceToHost));
            if (!exchange)
                check(cudaMemcpy(positions, d.pos, numBodies * sizeof(Vec3), cudaMemcpyDeviceToHost));
            if (validate) {
                if (exchange && numSteps > 0)
                    check(cudaMemcpy(d.pos, positions, numBodies * sizeof(Vec3), cudaMemcpyHostToDevice));
                double* rows = deviceBuffer<double>(d.count);
                energyRows<<<1 + (d.count - 1) / 128, 128>>>(d.pos, d.vel, rows, d.first, d.count, numBodies);
                check(cudaGetLastError());
                check(cudaMemcpy(energy.data() + (d.first - first), rows,
                                 d.count * sizeof(double), cudaMemcpyDeviceToHost));
                check(cudaFree(rows));
            }
        }
        check(cudaFree(d.vel));
        check(cudaFree(d.next));
        check(cudaFree(d.pos));
    }
    if (printResults) {
        if (ranks > 1 && !validate && numSteps > 0)
            MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : positions + first, count, xyz,
                        positions, counts.data(), starts.data(), xyz, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localVel.data(), count, xyz, velocities.data(), counts.data(),
                    starts.data(), xyz, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<double> data(size_t(numBodies) * 6);
            #pragma omp parallel for if(numBodies >= 8192)
            for (int i = 0; i < numBodies; ++i) {
                size_t k = size_t(i) * 6;
                const Vec3 p = positions[i], v = velocities[i];
                data[k] = p.x; data[k+1] = p.y; data[k+2] = p.z;
                data[k+3] = v.x; data[k+4] = v.y; data[k+5] = v.z;
            }
            print_results(data, "Bodies");
        }
    }
    int failure = 0;
    if (validate) {
        int localFailure = 0;
        double localEnergy = 0.0, totalEnergy;
        #pragma omp parallel for reduction(max:localFailure) reduction(+:localEnergy) if(count >= 8192)
        for (int i = 0; i < count; ++i) {
            const Vec3 p = positions[first + i], v = localVel[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
                localFailure = 3;
            else if (std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) > 1e6)
                localFailure = std::max(localFailure, 2);
            else if (std::max({std::abs(v.x), std::abs(v.y), std::abs(v.z)}) > 1e6)
                localFailure = std::max(localFailure, 1);
            localEnergy += energy[i];
        }
        MPI_Allreduce(&localFailure, &failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Reduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            puts("Validating simulation results...");
            if (!failure) printf("Final energy: %.6f\n", totalEnergy);
            if (failure == 3) puts("Validation failed: found NaN or Inf value in body state");
            if (failure == 2) puts("Validation failed: body position exceeds reasonable bounds");
            if (failure == 1) puts("Validation failed: body velocity exceeds reasonable bounds");
            printf("Validation: %s\n", failure ? "FAILED" : "PASSED");
        }
    }
    check(cudaFreeHost(positions));
    MPI_Type_free(&xyz);
    MPI_Finalize();
    return failure ? 1 : 0;
}
