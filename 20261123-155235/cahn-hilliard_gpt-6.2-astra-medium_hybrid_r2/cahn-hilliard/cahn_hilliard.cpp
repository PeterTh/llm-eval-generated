#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

struct Slab {
    int nx, ny, depth;
    size_t plane;
    int lower, upper;
};

// Threads access adjacent x coordinates. Each thread reuses its z neighbors
// in registers across a short column. Local planes 0 and depth+1 are halos.
template<bool potential>
__global__ void stencil(const double* __restrict__ field, const double* cold,
                        double* __restrict__ out, Slab slab, int first, int last,
                        bool faces) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= slab.nx) return;
    for (long long y = blockIdx.y * blockDim.y + threadIdx.y; y < slab.ny;
         y += gridDim.y * blockDim.y) {
        for (long long start = first + (faces ? blockIdx.z : 4LL * blockIdx.z);
             start <= last; start += 4LL * gridDim.z) {
            const int z0 = faces && blockIdx.z ? last : static_cast<int>(start);
            const int end = faces ? z0 : static_cast<int>(min(start + 3, (long long)last));
            size_t i = z0 * slab.plane + y * slab.nx + x;
            double v = field[i];
            double below = z0 == 1 && slab.lower == MPI_PROC_NULL ? v : field[i - slab.plane];
            for (int z = z0; z <= end; ++z, i += slab.plane) {
                const double above = z == slab.depth && slab.upper == MPI_PROC_NULL ? v : field[i + slab.plane];
                const double twice = 2.0 * v;
                const double xx = (field[i + (x+1 < slab.nx)] + field[i - (x > 0)]) - twice;
                const double yy = (field[i + (y+1 < slab.ny ? slab.nx : 0)] +
                                   field[i - (y > 0 ? slab.nx : 0)]) - twice;
                const double lap = (xx + yy) + ((above + below) - twice);
                if (potential) {
                    constexpr double e = 2.0 / 9.0;
                    double interaction = fma(v + 1.0, -e, __dmul_rn(v - 1.0, -e));
                    interaction = fma(-twice, e, interaction);
                    double bulk = fma(4.5, interaction, __dmul_rn(3.0, v));
                    bulk = fma(v*v, v, bulk);
                    out[i] = fma(-0.5, lap, bulk);
                } else {
                    out[i] = fma(0.01, lap, cold[i]);
                }
                below = v;
                v = above;
            }
            if (faces) break;
        }
    }
}

template<bool potential>
void launch(const double* field, const double* cold, double* out, Slab slab,
            int first, int last, bool faces, cudaStream_t stream) {
    if (first > last) return;
    dim3 block(32, 4);
    unsigned planes = last - first + 1;
    dim3 grid((slab.nx + 31u)/32, std::min(65535u, (slab.ny + 3u)/4),
              faces ? std::min(2u, planes) : std::min(65535u, (planes + 3)/4));
    stencil<potential><<<grid, block, 0, stream>>>(field, cold, out, slab, first, last, faces);
    check(cudaGetLastError());
}

struct Exchange {
    cudaStream_t bulk, edge;
    cudaEvent_t bulkDone, edgeDone;
    double* staging = nullptr;
    bool direct = false;
};

// The edge stream owns the outermost plane on each side. Its output can
// enter MPI while the bulk stream is still finishing the preceding phase.
template<bool potential>
void phase(double* field, double* cold, double* out, Slab s, Exchange& ex) {
    if (s.lower == MPI_PROC_NULL && s.upper == MPI_PROC_NULL) {
        launch<potential>(field, cold, out, s, 1, s.depth, false, ex.bulk);
        return;
    }

    const int neighbors[] = {s.lower, s.upper};
    double* owned[] = {field + s.plane, field + s.depth*s.plane};
    double* ghost[] = {field, field + (s.depth+1)*s.plane};
    double* send[2];
    double* recv[2];
    const size_t bytes = s.plane * sizeof(double);
    for (int side = 0; side < 2; ++side) {
        send[side] = ex.direct ? owned[side] : ex.staging + side*s.plane;
        recv[side] = ex.direct ? ghost[side] : ex.staging + (2+side)*s.plane;
        if (!ex.direct && neighbors[side] != MPI_PROC_NULL)
            check(cudaMemcpyAsync(send[side], owned[side], bytes, cudaMemcpyDeviceToHost, ex.edge));
    }

    // Capture both preceding completion events before recording the new ones.
    check(cudaStreamWaitEvent(ex.bulk, ex.edgeDone));
    launch<potential>(field, cold, out, s, 2, s.depth-1, false, ex.bulk);
    check(cudaStreamSynchronize(ex.edge));
    check(cudaStreamWaitEvent(ex.edge, ex.bulkDone));
    check(cudaEventRecord(ex.bulkDone, ex.bulk));

    for (size_t offset = 0; offset < s.plane; offset += INT_MAX) {
        int count = static_cast<int>(std::min<size_t>(INT_MAX, s.plane-offset));
        MPI_Request requests[4];
        for (int side = 0; side < 2; ++side) {
            MPI_Irecv(recv[side]+offset, count, MPI_DOUBLE, neighbors[side], 1-side,
                      MPI_COMM_WORLD, &requests[side]);
            MPI_Isend(send[side]+offset, count, MPI_DOUBLE, neighbors[side], side,
                      MPI_COMM_WORLD, &requests[2+side]);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    }

    for (int side = 0; side < 2; ++side)
        if (!ex.direct && neighbors[side] != MPI_PROC_NULL)
            check(cudaMemcpyAsync(ghost[side], recv[side], bytes, cudaMemcpyHostToDevice, ex.edge));
    launch<potential>(field, cold, out, s, 1, s.depth, true, ex.edge);
    check(cudaEventRecord(ex.edgeDone, ex.edge));
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}


int run(int argc, char** argv, int rank, int ranks) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > INT_MAX || ny > INT_MAX || nz >= INT_MAX ||
        nx * ny > SIZE_MAX / sizeof(double) / (nz+2)) {
        if (rank == 0) fprintf(stderr, "Invalid or oversized grid\n");
        return 1;
    }
    const int active = std::min<int>(ranks, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(localRank % devices));

    const auto slabStart = [=](int r) {
        return (nz/active)*r + std::min<size_t>(r, nz%active);
    };
    Slab s{int(nx), int(ny), int(slabStart(rank+1)-slabStart(rank)), nx*ny,
           rank ? rank-1 : MPI_PROC_NULL, rank+1 < active ? rank+1 : MPI_PROC_NULL};
    const size_t count = s.plane*s.depth;
    const size_t volume = nx*ny*nz;
    const size_t offset = slabStart(rank)*s.plane;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> local(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const double pseudo = (((offset+i+1)*1299709) % volume) / double(volume);
        local[i] = -1.0 + 2.0*pseudo;
    }

    double *cold, *next, *mu;
    const size_t bytes = (s.depth+2ULL)*s.plane*sizeof(double);
    check(cudaMalloc(&cold, bytes));
    check(cudaMalloc(&next, bytes));
    check(cudaMalloc(&mu, bytes));
    check(cudaMemcpy(cold+s.plane, local.data(), count*sizeof(double), cudaMemcpyHostToDevice));
    if (!validate && !printResults) std::vector<double>().swap(local);

    Exchange ex;
    check(cudaStreamCreateWithFlags(&ex.bulk, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&ex.edge, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&ex.bulkDone, cudaEventDisableTiming));
    check(cudaEventCreateWithFlags(&ex.edgeDone, cudaEventDisableTiming));
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    ex.direct = MPIX_Query_cuda_support() != 0;
#endif
    if (active > 1 && !ex.direct)
        check(cudaMallocHost(&ex.staging, 4*s.plane*sizeof(double)));
    check(cudaDeviceSynchronize());
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        phase<true>(cold, cold, mu, s, ex);
        phase<false>(mu, cold, next, s, ex);
        std::swap(cold, next);
    }
    check(cudaDeviceSynchronize());
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", long(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/seconds/1e6);
    }
    if (validate || printResults)
        check(cudaMemcpy(local.data(), cold+s.plane, count*sizeof(double), cudaMemcpyDeviceToHost));
    check(cudaFree(cold));
    check(cudaFree(next));
    check(cudaFree(mu));
    if (ex.staging) check(cudaFreeHost(ex.staging));
    check(cudaEventDestroy(ex.edgeDone));
    check(cudaEventDestroy(ex.bulkDone));
    check(cudaStreamDestroy(ex.edge));
    check(cudaStreamDestroy(ex.bulk));

    if (printResults) {
        // Preserve global element order for Kahan summation and the byte hash.
        if (rank == 0) {
            local.resize(volume);
            for (int r = 1; r < active; ++r) {
                size_t end = slabStart(r+1)*s.plane;
                for (size_t i = slabStart(r)*s.plane; i < end; i += INT_MAX)
                    MPI_Recv(local.data()+i, std::min<size_t>(INT_MAX, end-i), MPI_DOUBLE,
                             r, 0, comm, MPI_STATUS_IGNORE);
            }
            print_results(local, "Concentration");
        } else {
            for (size_t i = 0; i < count; i += INT_MAX)
                MPI_Send(local.data()+i, std::min<size_t>(INT_MAX, count-i), MPI_DOUBLE, 0, 0, comm);
        }
    }
    int valid = 1;
    if (validate) {
        double lo = local[0], hi = local[0];
        int bad = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
        for (size_t i = 0; i < count; ++i) {
            lo = std::min(lo, local[i]);
            hi = std::max(hi, local[i]);
            bad |= !std::isfinite(local[i]);
        }
        double globalLo, globalHi;
        int globalBad;
        MPI_Reduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (globalBad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                if (globalHi > 10.0 || globalLo < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            valid = !globalBad && globalHi <= 10.0 && globalLo >= -10.0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
