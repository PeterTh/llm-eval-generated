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
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

[[noreturn]] static void fail(const char* message) {
    fprintf(stderr, "stencil3d: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

// Each warp covers adjacent x coordinates. Sweep a short column in z so
// that two of the three z values come from registers on successive planes.
constexpr unsigned tileZ = 8;
__global__ void stencil(const Real* __restrict__ src, Real* __restrict__ dst,
                        size_t nx, size_t ny, size_t first, size_t end) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t z = first + size_t(blockIdx.z) * tileZ; z < end;
             z += size_t(gridDim.z) * tileZ) {
            size_t i = z * plane + y * nx + x;
            Real below = src[i - plane];
            Real center = src[i];
            for (unsigned k = 0; k < tileZ && z + k < end; ++k, i += plane) {
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
    dim3 grid((nx - 3) / block.x + 1,
              std::min<size_t>((ny - 3) / block.y + 1, 65535),
              std::min<size_t>((end - first - 1) / tileZ + 1, 65535));
    stencil<<<grid, block, 0, stream>>>(src, dst, nx, ny, first, end);
    check(cudaGetLastError());
}

// Use pinned staging so the MPI implementation need not accept device pointers.
struct Halo {
    int peer = MPI_PROC_NULL;
    size_t owned, ghost;
    Real *send = nullptr, *recv = nullptr;

    void allocate(size_t bytes) {
        if (peer == MPI_PROC_NULL) return;
        check(cudaMallocHost(&send, bytes));
        check(cudaMallocHost(&recv, bytes));
    }
};

static void postPlane(Real* buffer, size_t count, int peer, bool receive,
                      MPI_Comm comm, std::vector<MPI_Request>& requests) {
    while (count) {
        int part = static_cast<int>(std::min<size_t>(count, INT_MAX));
        requests.emplace_back();
        if (receive)
            MPI_Irecv(buffer, part, MPI_DOUBLE, peer, 0, comm, &requests.back());
        else
            MPI_Isend(buffer, part, MPI_DOUBLE, peer, 0, comm, &requests.back());
        buffer += part;
        count -= part;
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
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
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
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\nInitializing grid...\n",
               iterations, validate ? "enabled" : "disabled");
    }
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxCells / ny ||
        nx * ny > maxCells / nz || iterations < 0)
        fail("Invalid grid dimensions or iteration count");

    const size_t interior = nz > 2 ? nz - 2 : 0;
    const int workers = std::min<size_t>(ranks, std::max<size_t>(1, interior));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < workers ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= workers) {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) fail("No CUDA device available");
    check(cudaSetDevice(localRank % devices));

    // Each slab includes one ghost or physical boundary plane on each side.
    auto slabStart = [&](int r) -> size_t {
        return r * (interior / workers) + std::min<size_t>(r, interior % workers);
    };
    const size_t start = slabStart(rank);
    const size_t depth = slabStart(rank + 1) - start;
    const size_t planes = interior ? depth + 2 : nz;
    const size_t plane = nx * ny;
    const size_t cells = planes * plane;
    const size_t bytes = cells * sizeof(Real);
    Real *src, *dst;
    check(cudaMalloc(&src, bytes));
    check(cudaMalloc(&dst, bytes));
    {
        std::unique_ptr<Real[]> initial(new Real[cells]);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells; ++i)
            initial[i] = static_cast<Real>((start * plane + i) % 19);
        check(cudaMemcpy(src, initial.get(), bytes, cudaMemcpyHostToDevice));
    }
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));

    Halo halos[2] = {{rank ? rank - 1 : MPI_PROC_NULL, 1, 0},
                     {rank + 1 < workers ? rank + 1 : MPI_PROC_NULL, depth, depth + 1}};
    const size_t planeBytes = plane * sizeof(Real);
    const bool exchange = workers > 1 && iterations > 1 && nx > 2 && ny > 2;
    bool deviceMPI = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
    if (exchange && !deviceMPI)
        for (auto& h : halos) h.allocate(planeBytes);
    cudaStream_t edge, bulk;
    check(cudaStreamCreateWithFlags(&edge, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&bulk, cudaStreamNonBlocking));
    check(cudaDeviceSynchronize());
    if (rank == 0) printf("Running stencil computation...\n");
    std::vector<MPI_Request> requests;
    MPI_Barrier(comm);
    double begin = MPI_Wtime();
    for (int iter = 0; iter < iterations && depth && nx > 2 && ny > 2; ++iter) {
        if (exchange && iter + 1 < iterations) {
            requests.clear();
            size_t first = 1, end = depth + 1;
            for (auto& h : halos) {
                if (h.peer == MPI_PROC_NULL) continue;
                Real* incoming = deviceMPI ? dst + h.ghost * plane : h.recv;
                postPlane(incoming, plane, h.peer, true, comm, requests);
                // A one-plane slab sends the same updated plane to both peers.
                if (h.owned >= first && h.owned < end) {
                    update(src, dst, nx, ny, h.owned, h.owned + 1, edge);
                    if (h.owned == first) ++first;
                    else --end;
                }
                if (!deviceMPI)
                    check(cudaMemcpyAsync(h.send, dst + h.owned * plane, planeBytes,
                                      cudaMemcpyDeviceToHost, edge));
            }
            update(src, dst, nx, ny, first, end, bulk);
            check(cudaStreamSynchronize(edge));
            for (auto& h : halos) {
                if (h.peer == MPI_PROC_NULL) continue;
                Real* outgoing = deviceMPI ? dst + h.owned * plane : h.send;
                postPlane(outgoing, plane, h.peer, false, comm, requests);
            }
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            if (!deviceMPI) {
                for (auto& h : halos)
                    if (h.peer != MPI_PROC_NULL)
                        check(cudaMemcpyAsync(dst + h.ghost * plane, h.recv, planeBytes,
                                              cudaMemcpyHostToDevice, edge));
                check(cudaStreamSynchronize(edge));
            }
            check(cudaStreamSynchronize(bulk));
        } else {
            update(src, dst, nx, ny, 1, depth + 1, bulk);
        }
        std::swap(src, dst);
    }
    check(cudaStreamSynchronize(bulk));
    double elapsed = MPI_Wtime() - begin, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        double updates = nx > 2 && ny > 2 ? double(nx-2) * (ny-2) * interior * iterations : 0;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", updates / seconds / 1e6);
    }

    bool valid = true;
    if (validate || printResults) {
        auto ownedStart = [&](int r) -> size_t {
            if (r == 0) return 0;
            if (r == workers) return nz;
            return slabStart(r) + 1;
        };
        const size_t count = (ownedStart(rank + 1) - ownedStart(rank)) * plane;
        std::vector<Real> data(printResults && rank == 0 ? nz * plane : count);
        check(cudaMemcpy(data.data(), src + (ownedStart(rank) - start) * plane,
                         count * sizeof(Real), cudaMemcpyDeviceToHost));
        if (printResults) {
            if (rank == 0) {
                for (int r = 1; r < workers; ++r) {
                    size_t i = ownedStart(r) * plane, end = ownedStart(r + 1) * plane;
                    while (i < end) {
                        int part = static_cast<int>(std::min<size_t>(end - i, INT_MAX));
                        MPI_Recv(data.data() + i, part, MPI_DOUBLE, r, 1, comm, MPI_STATUS_IGNORE);
                        i += part;
                    }
                }
                print_results(data, "Grid");
            } else {
                for (size_t i = 0; i < count;) {
                    int part = static_cast<int>(std::min<size_t>(count - i, INT_MAX));
                    MPI_Send(data.data() + i, part, MPI_DOUBLE, 0, 1, comm);
                    i += part;
                }
            }
        }
        if (validate) {
            Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
            int bad = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
            for (size_t i = 0; i < count; ++i) {
                bad |= !std::isfinite(data[i]);
                lo = std::min(lo, data[i]);
                hi = std::max(hi, data[i]);
            }
            Real limits[] = {lo, -hi}, global[2];
            int nonfinite;
            MPI_Allreduce(limits, global, 2, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&bad, &nonfinite, 1, MPI_INT, MPI_MAX, comm);
            valid = !nonfinite && global[0] >= -1e6 && -global[1] <= 1e6;
            if (rank == 0) {
                printf("Validating result...\n");
                if (nonfinite) printf("Validation failed: found NaN or Inf value\n");
                else printf("Value range: [%.6f, %.6f]\n", global[0], -global[1]);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    for (auto& h : halos) {
        if (h.send) check(cudaFreeHost(h.send));
        if (h.recv) check(cudaFreeHost(h.recv));
    }
    check(cudaStreamDestroy(edge));
    check(cudaStreamDestroy(bulk));
    check(cudaFree(src));
    check(cudaFree(dst));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
