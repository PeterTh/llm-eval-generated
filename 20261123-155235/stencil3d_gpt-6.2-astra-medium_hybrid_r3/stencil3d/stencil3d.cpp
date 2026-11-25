#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

static void fail(const char* message) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

// A warp spans adjacent x coordinates. Reuse the two preceding z loads
// in registers as each thread advances through a short column.
constexpr int zTile = 8;
__global__ void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                        size_t nx, size_t ny, size_t first, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t y = 1 + size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t start = first + size_t(blockIdx.z) * zTile;
             start < end; start += size_t(gridDim.z) * zTile) {
            size_t i = start * plane + y * nx + x;
            Real below = input[i - plane];
            Real center = input[i];
            for (int dz = 0; dz < zTile && start + dz < end; ++dz, i += plane) {
                Real above = input[i + plane];
                output[i] = (center + input[i-1] + input[i+1] + input[i-nx]
                             + input[i+nx] + below + above) / 7.0;
                below = center;
                center = above;
            }
        }
    }
}

static void update(const Real* input, Real* output, size_t nx, size_t ny,
                   size_t first, size_t end, cudaStream_t stream) {
    if (first >= end) return;
    dim3 block(32, 4);
    dim3 blocks((nx + 31) / 32, std::min<size_t>((ny + 1) / 4, 65535),
                std::min<size_t>((end - first + zTile - 1) / zTile, 65535));
    stencil<<<blocks, block, 0, stream>>>(input, output, nx, ny, first, end);
    check(cudaGetLastError());
}

static size_t slabStart(int rank, int ranks, size_t nz) {
    return rank * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
}

struct Face {
    int peer;
    size_t owned, ghost;
    Real *send = nullptr, *recv = nullptr;
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI thread support is insufficient");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (!nx || !ny || !nz || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx > limit / ny || nx * ny > limit / nz || iterations < 0)
        fail("Invalid grid size or iteration count");

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    const size_t plane = nx * ny;
    const size_t offset = slabStart(rank, ranks, nz);
    const size_t depth = slabStart(rank + 1, ranks, nz) - offset;
    const size_t cells = depth * plane;
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\nInitializing grid...\n",
               iterations, validate ? "enabled" : "disabled");
    }

    Real *input = nullptr, *output = nullptr;
    cudaStream_t compute{}, transfer{};
    cudaEvent_t ready{};
    // Use device buffers directly when MPI advertises CUDA support.
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    const bool deviceMPI = MPIX_Query_cuda_support() != 0;
#else
    const bool deviceMPI = false;
#endif
    std::vector<Face> faces;
    if (depth) {
        int devices;
        check(cudaGetDeviceCount(&devices));
        if (!devices) fail("CUDA device required");
        check(cudaSetDevice(localRank % devices));
        if (depth + 2 > limit / plane) fail("Local slab is too large");
        const size_t bytes = (depth + 2) * plane * sizeof(Real);
        check(cudaMalloc(&input, bytes));
        check(cudaMalloc(&output, bytes));
        check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));

        // Initialize both buffers, including the initial neighbor values.
        size_t first = offset ? 0 : 1;
        size_t planes = depth + (offset != 0) + (offset + depth < nz);
        std::vector<Real> host(planes * plane);
        size_t base = (offset + first - 1) * plane;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < host.size(); ++i) host[i] = Real((base + i) % 19);
        check(cudaMemcpy(input + first * plane, host.data(), host.size() * sizeof(Real),
                         cudaMemcpyHostToDevice));
        check(cudaMemcpy(output + first * plane, input + first * plane,
                         host.size() * sizeof(Real), cudaMemcpyDeviceToDevice));
        if (offset) faces.push_back({rank - 1, 1, 0});
        if (offset + depth < nz) faces.push_back({rank + 1, depth, depth + 1});
        if (!deviceMPI) {
            for (auto& face : faces) {
                check(cudaMallocHost(&face.send, plane * sizeof(Real)));
                check(cudaMallocHost(&face.recv, plane * sizeof(Real)));
            }
        }
        check(cudaDeviceSynchronize());
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    std::vector<MPI_Request> requests;
    const size_t first = offset ? 1 : 2;
    const size_t end = depth + (offset + depth < nz);
    if (depth && nx > 2 && ny > 2 && nz > 2) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter == 0 || faces.empty()) {
                update(input, output, nx, ny, first, end, compute);
            } else {
                // Transfer current faces concurrently with the independent interior.
                check(cudaEventRecord(ready, compute));
                if (!deviceMPI) {
                    check(cudaStreamWaitEvent(transfer, ready));
                    for (auto& face : faces)
                        check(cudaMemcpyAsync(face.send, input + face.owned * plane,
                                              plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                }
                update(input, output, nx, ny, 2, depth, compute);
                if (deviceMPI) check(cudaEventSynchronize(ready));
                else check(cudaStreamSynchronize(transfer));
                requests.clear();
                for (auto& face : faces) {
                    Real* recv = deviceMPI ? input + face.ghost * plane : face.recv;
                    Real* send = deviceMPI ? input + face.owned * plane : face.send;
                    for (size_t i = 0; i < plane; i += INT_MAX) {
                        int count = std::min<size_t>(plane - i, INT_MAX);
                        requests.emplace_back();
                        MPI_Irecv(recv + i, count, MPI_DOUBLE, face.peer, 0,
                                  MPI_COMM_WORLD, &requests.back());
                        requests.emplace_back();
                        MPI_Isend(send + i, count, MPI_DOUBLE, face.peer, 0,
                                  MPI_COMM_WORLD, &requests.back());
                    }
                }
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                if (!deviceMPI) {
                    for (auto& face : faces)
                        check(cudaMemcpyAsync(input + face.ghost * plane, face.recv,
                                              plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
                    check(cudaEventRecord(ready, transfer));
                    check(cudaStreamWaitEvent(compute, ready));
                }
                if (first == 1 && first < end)
                    update(input, output, nx, ny, 1, 2, compute);
                if (depth >= first && depth < end && depth != 1)
                    update(input, output, nx, ny, depth, depth + 1, compute);
            }
            std::swap(input, output);
        }
        check(cudaStreamSynchronize(compute));
    }
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double updates = nx > 2 && ny > 2 && nz > 2 ?
            double(nx - 2) * (ny - 2) * (nz - 2) * iterations : 0;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds ? updates / seconds / 1e6 : 0.0);
    }

    std::vector<Real> result;
    if (printResults || validate) {
        result.resize(printResults && rank == 0 ? nz * plane : cells);
        if (cells) check(cudaMemcpy(result.data(), input + plane, cells * sizeof(Real),
                                    cudaMemcpyDeviceToHost));
    }
    if (depth) {
        if (!deviceMPI) {
            for (auto& face : faces) {
                check(cudaFreeHost(face.recv));
                check(cudaFreeHost(face.send));
            }
        }
        check(cudaEventDestroy(ready));
        check(cudaStreamDestroy(transfer));
        check(cudaStreamDestroy(compute));
        check(cudaFree(output));
        check(cudaFree(input));
    }

    if (printResults) {
        // Restore global index order for the original Kahan sum and byte hash.
        if (rank == 0) {
            for (int peer = 1; peer < ranks; ++peer) {
                size_t pos = slabStart(peer, ranks, nz) * plane;
                size_t stop = slabStart(peer + 1, ranks, nz) * plane;
                while (pos < stop) {
                    int count = std::min<size_t>(stop - pos, INT_MAX);
                    MPI_Recv(result.data() + pos, count, MPI_DOUBLE, peer, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += count;
                }
            }
            print_results(result, "Grid");
        } else {
            for (size_t pos = 0; pos < cells; pos += INT_MAX) {
                int count = std::min<size_t>(cells - pos, INT_MAX);
                MPI_Send(result.data() + pos, count, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
    }
    int invalid = 0;
    if (validate) {
        Real low = std::numeric_limits<Real>::infinity(), high = -low;
        #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:invalid)
        for (size_t i = 0; i < cells; ++i) {
            low = std::min(low, result[i]);
            high = std::max(high, result[i]);
            invalid |= !std::isfinite(result[i]);
        }
        Real globalLow, globalHigh;
        MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (invalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", globalLow, globalHigh);
                if (globalLow < -1e6 || globalHigh > 1e6) {
                    invalid = 1;
                    printf("Validation failed: values out of expected range\n");
                }
            }
            printf("Validation: %s\n", invalid ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&invalid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return invalid;
}
