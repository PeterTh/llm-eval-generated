#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using Real = double;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Consecutive lanes traverse X; each thread rolls through a short Z column,
// reusing the previous/current/next values in registers. Both buffers retain
// the fixed global boundary values installed during initialization.
__global__ void stencil(const Real* __restrict__ in, Real* __restrict__ out,
                        size_t nx, size_t ny, size_t begin, size_t end) {
    const size_t plane = nx * ny;
    const size_t columns = (nx - 2) * (ny - 2);
    const size_t tiles = (end - begin + 7) / 8;
    for (size_t task = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         task < columns * tiles; task += size_t(gridDim.x) * blockDim.x) {
        const size_t column = task % columns;
        const size_t first = begin + (task / columns) * 8;
        const size_t last = first + 8 < end ? first + 8 : end;
        size_t i = first * plane + (column / (nx - 2) + 1) * nx + column % (nx - 2) + 1;
        Real bottom = in[i - plane], center = in[i];
        for (size_t z = first; z < last; ++z, i += plane) {
            const Real top = in[i + plane];
            out[i] = (center + in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx] + bottom + top) / 7.0;
            bottom = center;
            center = top;
        }
    }
}

static void launch(const Real* in, Real* out, size_t nx, size_t ny,
                   size_t begin, size_t end, cudaStream_t stream) {
    if (nx < 3 || ny < 3 || begin >= end) return;
    const size_t work = (nx - 2) * (ny - 2) * ((end - begin + 7) / 8);
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((work + 255) / 256, 65535));
    stencil<<<blocks, 256, 0, stream>>>(in, out, nx, ny, begin, end);
    CUDA(cudaGetLastError());
}

// Chunk transfers so grids are not limited by MPI's int element counts.
static void exchange(Real* data, size_t count, int peer, int tag, bool send,
                     MPI_Comm comm, std::vector<MPI_Request>& requests) {
    if (peer == MPI_PROC_NULL) return;
    for (size_t offset = 0; offset < count;) {
        const int n = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        MPI_Request request;
        if (send) MPI_Isend(data + offset, n, MPI_DOUBLE, peer, tag, comm, &request);
        else MPI_Irecv(data + offset, n, MPI_DOUBLE, peer, tag, comm, &request);
        requests.push_back(request);
        offset += n;
    }
}

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    

    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny ||
        nx * ny > limit / nz || limit / (nx * ny) < 3 ||
        nz > limit / (nx * ny) - 2 || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
               nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    // Excess ranks participate in final world collectives without allocating a GPU.
    const int active = static_cast<int>(std::min<size_t>(ranks, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    double elapsed = 0;
    int valid = 1;
    if (rank < active) {
        MPI_Comm local;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "A CUDA device is required on every active rank\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        const size_t plane = nx * ny;
        const size_t depth = nz / active + (size_t(rank) < nz % active);
        const size_t startZ = size_t(rank) * (nz / active) + std::min<size_t>(rank, nz % active);
        const size_t count = (depth + 2) * plane;
        std::vector<Real> host(count);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            const size_t z = i / plane;
            if (startZ + z > 0 && startZ + z <= nz)
                host[i] = Real(((startZ + z - 1) * plane + i % plane) % 19);
        }
        Real *input, *output;
        CUDA(cudaMalloc(&input, count * sizeof(Real)));
        CUDA(cudaMalloc(&output, count * sizeof(Real)));
        CUDA(cudaMemcpy(input, host.data(), count * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(output, input, count * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA(cudaDeviceSynchronize());
        cudaStream_t compute, transfer;
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        Real* halos = nullptr;
        if (active > 1) CUDA(cudaMallocHost(&halos, 4 * plane * sizeof(Real)));
        const int below = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int above = rank == active - 1 ? MPI_PROC_NULL : rank + 1;
        const size_t first = startZ == 0 ? 2 : 1;
        const size_t end = startZ + depth == nz ? depth : depth + 1;
        std::vector<MPI_Request> requests;
        requests.reserve(8);
        if (rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        const double beginTime = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            if (active == 1 || nx < 3 || ny < 3 || nz < 3) {
                launch(input, output, nx, ny, first, end, compute);
            } else {
                requests.clear();
                exchange(halos + 2 * plane, plane, below, 1, false, comm, requests);
                exchange(halos + 3 * plane, plane, above, 0, false, comm, requests);
                if (below != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(halos, input + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                if (above != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(halos + plane, input + depth * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, transfer));
                // The interior does not depend on incoming halo planes.
                launch(input, output, nx, ny, std::max<size_t>(first, 2), std::min(end, depth), compute);
                CUDA(cudaStreamSynchronize(transfer));
                exchange(halos, plane, below, 0, true, comm, requests);
                exchange(halos + plane, plane, above, 1, true, comm, requests);
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
                if (below != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(input, halos + 2 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
                if (above != MPI_PROC_NULL)
                    CUDA(cudaMemcpyAsync(input + (depth + 1) * plane, halos + 3 * plane, plane * sizeof(Real), cudaMemcpyHostToDevice, transfer));
                CUDA(cudaStreamSynchronize(transfer));
                if (first <= 1 && end > 1) launch(input, output, nx, ny, 1, 2, compute);
                if (depth > 1 && depth >= first && depth < end)
                    launch(input, output, nx, ny, depth, depth + 1, compute);
                CUDA(cudaStreamSynchronize(compute));
            }
            std::swap(input, output);
        }
        CUDA(cudaStreamSynchronize(compute));
        elapsed = MPI_Wtime() - beginTime;
        if (validate || printResults)
            CUDA(cudaMemcpy(host.data(), input + plane, depth * plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) {
            std::vector<Real> global;
            if (rank == 0) {
                global.resize(nx * ny * nz);
                std::copy_n(host.data(), depth * plane, global.data());
            }
            // Preserve the exact global ordering for the original Kahan sum/hash.
            for (int source = 1; source < active; ++source) {
                const size_t sourceDepth = nz / active + (size_t(source) < nz % active);
                const size_t sourceStart = size_t(source) * (nz / active) + std::min<size_t>(source, nz % active);
                for (size_t offset = 0; offset < sourceDepth * plane;) {
                    const int n = static_cast<int>(std::min<size_t>(sourceDepth * plane - offset, INT_MAX));
                    if (rank == source) MPI_Send(host.data() + offset, n, MPI_DOUBLE, 0, 2, comm);
                    if (rank == 0) MPI_Recv(global.data() + sourceStart * plane + offset, n, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                    offset += n;
                }
            }
            if (rank == 0) print_results(global, "Grid");
        }
        if (validate) {
            Real low = std::numeric_limits<Real>::infinity(), high = -low;
            int finite = 1;
            #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(&:finite) schedule(static)
            for (size_t i = 0; i < depth * plane; ++i) {
                finite &= std::isfinite(host[i]);
                low = std::min(low, host[i]);
                high = std::max(high, host[i]);
            }
            Real globalLow, globalHigh;
            MPI_Allreduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, comm);
            valid = valid && globalLow >= -1e6 && globalHigh <= 1e6;
            if (rank == 0) {
                printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalLow, globalHigh);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        if (halos) CUDA(cudaFreeHost(halos));
        CUDA(cudaFree(input));
        CUDA(cudaFree(output));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(transfer));
        MPI_Comm_free(&comm);
    }
    double totalTime;
    MPI_Reduce(&elapsed, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double updates = double(nx > 2 ? nx - 2 : 0) * double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        printf("Computation time: %ld ms\n", static_cast<long>(totalTime * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", totalTime > 0 ? updates / totalTime / 1e6 : 0);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
