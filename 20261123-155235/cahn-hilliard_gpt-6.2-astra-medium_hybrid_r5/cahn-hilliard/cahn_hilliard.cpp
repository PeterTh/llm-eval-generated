#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI)
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

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

struct Slab {
    int nx, ny, depth;
    bool first, last;
    size_t plane;
};

// Each warp reads consecutive x coordinates. Each field has one ghost plane
// at either end, with the owned planes numbered 1 through depth.
template<bool potential, bool edges>
__global__ void stencil(Slab s, const double* __restrict__ input,
                        const double* concentration, double* __restrict__ output,
                        int begin, int count) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= s.nx) return;
    for (unsigned row = blockIdx.y * blockDim.y + threadIdx.y; row < s.ny;
         row += gridDim.y * blockDim.y) {
        for (unsigned k = blockIdx.z; k < count; k += gridDim.z) {
            const int z = edges && k == 1 ? s.depth : begin + k;
            const size_t i = z * s.plane + row * size_t(s.nx) + x;
            const double c = input[i];
            const double twice = 2.0 * c;
            const double xx = input[i + (x < s.nx-1)] + input[i - (x > 0)] - twice;
            const double yy = input[i + (row < s.ny-1 ? s.nx : 0)]
                            + input[i - (row > 0 ? s.nx : 0)] - twice;
            const double zz = input[i + (s.last && z == s.depth ? 0 : s.plane)]
                            + input[i - (s.first && z == 1 ? 0 : s.plane)] - twice;
            const double lap = xx + yy + zz;
            if constexpr (potential) {
                constexpr double e = 2.0 / 9.0;
                double interaction = fma(c + 1.0, -e, (c - 1.0) * -e);
                interaction = fma(-twice, e, interaction);
                const double bulk = fma(c*c, c, fma(4.5, interaction, 3.0*c));
                output[i] = fma(-0.5, lap, bulk);
            } else {
                output[i] = fma(0.01, lap, concentration[i]);
            }
        }
    }
}

template<bool potential, bool edges = false>
void launch(Slab s, const double* input, const double* c, double* output,
            int begin, int count, cudaStream_t stream) {
    if (count <= 0) return;
    const dim3 block(32, 4);
    const dim3 grid((s.nx-1)/32 + 1, std::min((s.ny-1)/4 + 1, 65535),
                    std::min(count, 65535));
    stencil<potential, edges><<<grid, block, 0, stream>>>(s, input, c, output, begin, count);
    check(cudaGetLastError());
}

struct Halo {
    MPI_Comm comm;
    int peer[2];
    double* staging = nullptr;
    cudaStream_t compute, transfer;
    cudaEvent_t ready;
};

template<bool potential>
void advance(Slab s, Halo& h, double* input, const double* c, double* output) {
    if (s.first && s.last) {
        launch<potential>(s, input, c, output, 1, s.depth, h.compute);
        return;
    }
    double* send[] = {input + s.plane, input + s.depth*s.plane};
    double* recv[] = {input, input + (size_t(s.depth)+1)*s.plane};
    check(cudaEventRecord(h.ready, h.compute));
    check(cudaStreamWaitEvent(h.transfer, h.ready, 0));
    launch<potential>(s, input, c, output, 2, s.depth-2, h.compute);
    for (int side = 0; side < 2; ++side) {
        if (h.peer[side] == MPI_PROC_NULL || !h.staging) continue;
        double* buffer = h.staging + side*s.plane;
        check(cudaMemcpyAsync(buffer, send[side], s.plane*sizeof(double),
                              cudaMemcpyDeviceToHost, h.transfer));
        send[side] = buffer;
        recv[side] = buffer + 2*s.plane;
    }
    check(cudaStreamSynchronize(h.transfer));
    for (size_t offset = 0; offset < s.plane; offset += INT_MAX) {
        int count = std::min<size_t>(INT_MAX, s.plane-offset);
        MPI_Request requests[4];
        for (int side = 0; side < 2; ++side) {
            MPI_Irecv(recv[side]+offset, count, MPI_DOUBLE, h.peer[side], 1-side,
                      h.comm, &requests[side]);
            MPI_Isend(send[side]+offset, count, MPI_DOUBLE, h.peer[side], side,
                      h.comm, &requests[2+side]);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    }
    if (h.staging) {
        for (int side = 0; side < 2; ++side) {
            if (h.peer[side] == MPI_PROC_NULL) continue;
            const size_t ghost = side == 0 ? 0 : size_t(s.depth)+1;
            check(cudaMemcpyAsync(input + ghost*s.plane, recv[side], s.plane*sizeof(double),
                                  cudaMemcpyHostToDevice, h.transfer));
        }
        check(cudaEventRecord(h.ready, h.transfer));
        check(cudaStreamWaitEvent(h.compute, h.ready, 0));
    }
    launch<potential, true>(s, input, c, output, 1, std::min(s.depth, 2), h.compute);
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
    
    if (!nx || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx*ny > std::numeric_limits<size_t>::max()/sizeof(double)/(nz+2)) {
        if (!rank) fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const int workers = std::min<size_t>(ranks, nz);
    Halo halo{};
    MPI_Comm_split(MPI_COMM_WORLD, rank < workers ? 0 : MPI_UNDEFINED, rank, &halo.comm);
    if (rank >= workers) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(halo.comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, devices;
    MPI_Comm_rank(node, &nodeRank);
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(nodeRank % devices));
    MPI_Comm_free(&node);

    auto firstPlane = [=](int r) {
        return r*(nz/workers) + std::min<size_t>(r, nz%workers);
    };
    const size_t plane = nx*ny, volume = plane*nz;
    const Slab slab{int(nx), int(ny), int(firstPlane(rank+1)-firstPlane(rank)),
                    rank == 0, rank+1 == workers, plane};
    const size_t localSize = slab.depth*plane;
    const size_t bytes = (localSize + 2*plane)*sizeof(double);
    double *cold, *cnew, *mu;
    check(cudaMalloc(&cold, bytes));
    check(cudaMalloc(&cnew, bytes));
    check(cudaMalloc(&mu, bytes));
    check(cudaStreamCreateWithFlags(&halo.compute, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&halo.transfer, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&halo.ready, cudaEventDisableTiming));
    halo.peer[0] = slab.first ? MPI_PROC_NULL : rank-1;
    halo.peer[1] = slab.last ? MPI_PROC_NULL : rank+1;
    bool direct = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    direct = MPIX_Query_cuda_support();
#endif
    if (workers > 1 && !direct)
        check(cudaMallocHost(&halo.staging, 4*plane*sizeof(double)));

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> local(localSize);
    const size_t offset = plane*firstPlane(rank);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localSize; ++i) {
        const double pseudo = ((offset+i+1)*1299709 % volume)/double(volume);
        local[i] = -1.0 + 2.0*pseudo;
    }
    check(cudaMemcpy(cold+plane, local.data(), localSize*sizeof(double), cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(halo.comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        advance<true>(slab, halo, cold, nullptr, mu);
        advance<false>(slab, halo, mu, cold, cnew);
        std::swap(cold, cnew);
    }
    check(cudaStreamSynchronize(halo.compute));
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, halo.comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", long(duration*1000));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/duration/1e6);
    }

    if (printResults || validate)
        check(cudaMemcpy(local.data(), cold+plane, localSize*sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        // The external hash and Kahan sum consume values in global linear order.
        if (!rank) {
            local.resize(volume);
            for (int r = 1; r < workers; ++r) {
                size_t end = firstPlane(r+1)*plane;
                for (size_t i = firstPlane(r)*plane; i < end; i += INT_MAX) {
                    int n = std::min<size_t>(INT_MAX, end-i);
                    MPI_Recv(local.data()+i, n, MPI_DOUBLE, r, 2, halo.comm, MPI_STATUS_IGNORE);
                }
            }
            print_results(local, "Concentration");
        } else {
            for (size_t i = 0; i < localSize; i += INT_MAX) {
                int n = std::min<size_t>(INT_MAX, localSize-i);
                MPI_Send(local.data()+i, n, MPI_DOUBLE, 0, 2, halo.comm);
            }
        }
    }
    int failed = 0;
    if (validate) {
        double lo = local[0], hi = local[0];
        int bad = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
        for (size_t i = 0; i < localSize; ++i) {
            lo = std::min(lo, local[i]);
            hi = std::max(hi, local[i]);
            bad |= !std::isfinite(local[i]);
        }
        double minVal, maxVal;
        int invalid;
        MPI_Reduce(&lo, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, halo.comm);
        MPI_Reduce(&hi, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, halo.comm);
        MPI_Reduce(&bad, &invalid, 1, MPI_INT, MPI_MAX, 0, halo.comm);
        if (!rank) {
            printf("Validating result...\n");
            if (invalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    invalid = 1;
                }
            }
            failed = invalid;
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, halo.comm);
    }
    if (halo.staging) check(cudaFreeHost(halo.staging));
    check(cudaEventDestroy(halo.ready));
    check(cudaStreamDestroy(halo.transfer));
    check(cudaStreamDestroy(halo.compute));
    check(cudaFree(mu));
    check(cudaFree(cnew));
    check(cudaFree(cold));
    MPI_Comm_free(&halo.comm);
    return failed;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
