#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include "../common/results_output.hpp"

using Real = double;

static void fail(const char* message) {
    fprintf(stderr, "stencil3d: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

// A warp follows X. Each thread rolls three Z values through registers,
// reusing two of them when it moves to the next plane.
constexpr unsigned zTile = 8;
__global__ void stencil(const Real* __restrict__ src, Real* __restrict__ dst,
                        size_t nx, size_t ny, size_t first, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x + 1 >= nx) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y + 1 < ny; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t z = first + size_t(blockIdx.z) * zTile; z < end;
             z += size_t(gridDim.z) * zTile) {
            size_t i = z * plane + y * nx + x;
            Real below = src[i - plane];
            Real center = src[i];
            for (unsigned k = 0; k < zTile && z + k < end; ++k, i += plane) {
                Real above = src[i + plane];
                dst[i] = (center + src[i-1] + src[i+1] + src[i-nx] +
                          src[i+nx] + below + above) / 7.0;
                below = center;
                center = above;
            }
        }
    }
}

static void update(const Real* src, Real* dst, size_t nx, size_t ny,
                   size_t first, size_t end, cudaStream_t stream) {
    if (first >= end) return;
    dim3 block(32, 4);
    dim3 grid((nx + 31) / 32, std::min<size_t>((ny + 1) / 4, 65535),
              std::min<size_t>((end - first + zTile - 1) / zTile, 65535));
    stencil<<<grid, block, 0, stream>>>(src, dst, nx, ny, first, end);
    check(cudaGetLastError());
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
    int provided, rank, world;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
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
    
    if (!nx || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx * ny > std::numeric_limits<size_t>::max() / sizeof(Real) / (nz + 2))
        fail("invalid grid size");

    // Give every active rank at least one plane, including for thin grids.
    const int ranks = std::min<size_t>(world, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) fail("a CUDA device is required");
    check(cudaSetDevice(localRank % devices));

    const auto offset = [&](int r) -> size_t {
        return (nz / ranks) * r + std::min<size_t>(r, nz % ranks);
    };
    const size_t start = offset(rank);
    const size_t depth = offset(rank + 1) - start;
    const size_t plane = nx * ny, bytes = plane * sizeof(Real);
    const size_t count = (depth + 2) * plane;
    const int neighbors[] = {rank ? rank - 1 : MPI_PROC_NULL,
                             rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL};

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\nInitializing grid...\n",
               iterations, validate ? "enabled" : "disabled");
    }
    Real *src, *dst;
    check(cudaMalloc(&src, count * sizeof(Real)));
    check(cudaMalloc(&dst, count * sizeof(Real)));
    {
        // Initialize both copies of the fixed exterior and the initial halos.
        std::vector<Real> host(count);
        #pragma omp parallel for schedule(static)
        for (size_t z = 0; z < depth + 2; ++z) {
            size_t globalZ = start + z ? std::min(nz - 1, start + z - 1) : 0;
            for (size_t i = 0; i < plane; ++i)
                host[z * plane + i] = (globalZ * plane + i) % 19;
        }
        check(cudaMemcpy(src, host.data(), count * sizeof(Real), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dst, src, count * sizeof(Real), cudaMemcpyDeviceToDevice));
    }
    MPI_Datatype rowType, planeType;
    MPI_Type_contiguous(static_cast<int>(nx), MPI_DOUBLE, &rowType);
    MPI_Type_contiguous(static_cast<int>(ny), rowType, &planeType);
    MPI_Type_commit(&planeType);
    MPI_Type_free(&rowType);

    cudaStream_t compute, transfer;
    check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    cudaEvent_t ready;
    check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    // MPI implementations without a CUDA capability query use pinned buffers.
    bool deviceMPI = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
    Real *send[2]{}, *recv[2]{};
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    for (int s = 0; s < 2; ++s) {
        if (deviceMPI || neighbors[s] == MPI_PROC_NULL || iterations < 2 || !hasInterior) continue;
        check(cudaMallocHost(&send[s], bytes));
        check(cudaMallocHost(&recv[s], bytes));
    }
    const size_t first = start == 0 ? 2 : 1;
    const size_t end = depth + (rank + 1 < ranks ? 1 : 0);
    if (rank == 0) printf("Running stencil computation...\n");
    check(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    double begin = MPI_Wtime();
    for (int iter = 0; hasInterior && iter < iterations; ++iter) {
        if (ranks == 1 || iter + 1 == iterations) {
            update(src, dst, nx, ny, first, end, compute);
        } else {
            // Produce outgoing faces first. A one-plane slab has just one face
            // to compute, although it can have two neighbors to send it to.
            if (first == 1 && first < end) update(src, dst, nx, ny, 1, 2, compute);
            if (depth > 1 && depth < end)
                update(src, dst, nx, ny, depth, depth + 1, compute);
            check(cudaEventRecord(ready, compute));
            update(src, dst, nx, ny, 2, depth, compute);
            Real* face[] = {dst + plane, dst + depth * plane};
            Real* ghost[] = {dst, dst + (depth + 1) * plane};

            if (deviceMPI) {
                check(cudaEventSynchronize(ready));
            } else {
                check(cudaStreamWaitEvent(transfer, ready));
                for (int s = 0; s < 2; ++s) {
                    if (send[s]) check(cudaMemcpyAsync(send[s], face[s], bytes,
                                                       cudaMemcpyDeviceToHost, transfer));
                }
                check(cudaStreamSynchronize(transfer));
            }
            MPI_Request requests[4];
            int n = 0;
            for (int s = 0; s < 2; ++s) {
                if (neighbors[s] == MPI_PROC_NULL) continue;
                MPI_Irecv(deviceMPI ? ghost[s] : recv[s], 1, planeType,
                          neighbors[s], 0, comm, &requests[n++]);
                MPI_Isend(deviceMPI ? face[s] : send[s], 1, planeType,
                          neighbors[s], 0, comm, &requests[n++]);
            }
            MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
            if (!deviceMPI) {
                for (int s = 0; s < 2; ++s) {
                    if (recv[s]) check(cudaMemcpyAsync(ghost[s], recv[s], bytes,
                                                       cudaMemcpyHostToDevice, transfer));
                }
                check(cudaEventRecord(ready, transfer));
                check(cudaStreamWaitEvent(compute, ready));
            }
        }
        std::swap(src, dst);
    }
    check(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime() - begin, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        double updates = hasInterior ? double(nx-2) * (ny-2) * (nz-2) * iterations : 0;
        printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",
               seconds * 1000, seconds > 0 ? updates / seconds / 1e6 : 0);
    }

    std::vector<Real> result;
    if (validate || printResults) {
        result.resize((printResults && rank == 0 ? nz : depth) * plane);
        check(cudaMemcpy(result.data(), src + plane, depth * bytes, cudaMemcpyDeviceToHost));
    }
    if (printResults) {
        if (rank == 0) {
            for (int r = 1; r < ranks; ++r) {
                MPI_Recv(result.data() + offset(r) * plane, offset(r+1) - offset(r),
                         planeType, r, 1, comm, MPI_STATUS_IGNORE);
            }
            print_results(result, "Grid");
        } else {
            MPI_Send(result.data(), depth, planeType, 0, 1, comm);
        }
    }
    int valid = 1;
    if (validate) {
        Real low = std::numeric_limits<Real>::infinity(), high = -low;
        int finite = 1;
        #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(&:finite)
        for (size_t i = 0; i < depth * plane; ++i) {
            finite &= std::isfinite(result[i]);
            low = std::min(low, result[i]);
            high = std::max(high, result[i]);
        }
        Real globalLow, globalHigh;
        MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        valid = finite && low >= -1e6 && high <= 1e6;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\nValidation: %s\n",
                   globalLow, globalHigh, valid ? "PASSED" : "FAILED");
        }
    }
    for (int s = 0; s < 2; ++s) {
        if (send[s]) check(cudaFreeHost(send[s]));
        if (recv[s]) check(cudaFreeHost(recv[s]));
    }
    check(cudaEventDestroy(ready));
    check(cudaStreamDestroy(transfer));
    check(cudaStreamDestroy(compute));
    check(cudaFree(src));
    check(cudaFree(dst));
    MPI_Type_free(&planeType);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
