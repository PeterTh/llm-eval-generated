#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

inline constexpr size_t idx3(size_t x, size_t y, size_t z,
                             size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// z is local and includes one ghost plane on either side.  Every owned cell is
// written, so physical boundary values remain invariant without a second pass.
__global__ void stencil_kernel(const Real* __restrict__ in,
                               Real* __restrict__ out,
                               size_t nx, size_t ny, size_t local_nz,
                               size_t global_z0, size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || lz > local_nz) return;

    const size_t gz = global_z0 + lz - 1;
    const size_t i = (lz * ny + y) * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        gz == 0 || gz + 1 == global_nz) {
        out[i] = in[i];
    } else {
        const size_t plane = nx * ny;
        out[i] = (in[i] + in[i - 1] + in[i + 1] +
                  in[i - nx] + in[i + nx] +
                  in[i - plane] + in[i + plane]) / Real(7.0);
    }
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n", p);
    std::printf("  -x <num>  X size (default 128)\n"
                "  -y <num>  Y size (default X)\n"
                "  -z <num>  Z size (default X)\n"
                "  -i <num>  iterations (default 10)\n"
                "  -v        validate\n  -r        print results\n"
                "  -h        help\n");
}

static bool validate_result(const std::vector<Real>& a) {
    int bad = 0;
    Real lo = std::numeric_limits<Real>::infinity();
    Real hi = -std::numeric_limits<Real>::infinity();
#pragma omp parallel for reduction(+:bad) reduction(min:lo) reduction(max:hi)
    for (long long i = 0; i < static_cast<long long>(a.size()); ++i) {
        const Real v = a[static_cast<size_t>(i)];
        bad += !std::isfinite(v);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    std::printf("Value range: [%.6f, %.6f]\n", lo, hi);
    if (bad || hi > 1e6 || lo < -1e6) {
        std::printf("Validation failed: invalid or out-of-range value\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, results = false, help = false, parse_ok = true;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else parse_ok = false;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (help) {
        if (!rank) usage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (!parse_ok || nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        static_cast<size_t>(nranks) > nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (!rank) {
            std::fprintf(stderr, "Invalid arguments (dimensions >= 2 and MPI ranks <= Z are required).\n");
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Block distribution; the first remainder ranks own one additional plane.
    const size_t base = nz / static_cast<size_t>(nranks);
    const size_t rem = nz % static_cast<size_t>(nranks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem);
    const size_t z0 = static_cast<size_t>(rank) * base +
                      std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny;
    const size_t local_elems = (local_nz + 2) * plane;

    // Bind ranks on the same node round-robin to visible accelerators.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        if (!rank) std::fprintf(stderr, "The hybrid benchmark requires a CUDA device.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));
    MPI_Comm_free(&local_comm);

    std::vector<Real> host(local_elems, 0.0);
#pragma omp parallel for collapse(2) schedule(static)
    for (long long lz = 1; lz <= static_cast<long long>(local_nz); ++lz)
        for (long long y = 0; y < static_cast<long long>(ny); ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = z0 + static_cast<size_t>(lz) - 1;
                const size_t gi = (gz * ny + static_cast<size_t>(y)) * nx + x;
                host[idx3(x, static_cast<size_t>(y), static_cast<size_t>(lz), nx, ny)] = Real(gi % 19);
            }

    Real *d_a = nullptr, *d_b = nullptr;
    Real *send_lo = nullptr, *send_hi = nullptr, *recv_lo = nullptr, *recv_hi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_b, local_elems * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_a, host.data(), local_elems * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(Real)));

    if (!rank) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        std::printf("Parallelism: %d MPI rank(s), OpenMP, CUDA\nValidation: %s\n", nranks, validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    Real* in = d_a;
    Real* out = d_b;
    const int below = rank ? rank - 1 : MPI_PROC_NULL;
    const int above = rank + 1 < nranks ? rank + 1 : MPI_PROC_NULL;
    const int plane_count = static_cast<int>(plane);
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "XY plane exceeds MPI count range.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    const dim3 block(32, 4, 2);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);
    for (int iter = 0; iter < iterations; ++iter) {
        // Pinned staging works with every MPI implementation, including those
        // without CUDA-aware transport. Transfers are contiguous full planes.
        if (below != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(send_lo, in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (above != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(send_hi, in + local_nz * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Request req[4]; int nreq = 0;
        if (below != MPI_PROC_NULL) {
            MPI_Irecv(recv_lo, plane_count, MPI_DOUBLE, below, 11, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(send_lo, plane_count, MPI_DOUBLE, below, 12, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (above != MPI_PROC_NULL) {
            MPI_Irecv(recv_hi, plane_count, MPI_DOUBLE, above, 12, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Isend(send_hi, plane_count, MPI_DOUBLE, above, 11, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (nreq) MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        if (below != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(in, recv_lo, plane * sizeof(Real), cudaMemcpyHostToDevice));
        if (above != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(in + (local_nz + 1) * plane, recv_hi, plane * sizeof(Real), cudaMemcpyHostToDevice));
        stencil_kernel<<<grid, block>>>(in, out, nx, ny, local_nz, z0, nz);
        CUDA_CHECK(cudaGetLastError());
        std::swap(in, out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displs;
    std::vector<Real> global;
    if (results || validate) {
        CUDA_CHECK(cudaMemcpy(host.data() + plane, in + plane,
                              local_nz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
        if (!rank) {
            counts.resize(nranks); displs.resize(nranks);
            global.resize(nx * ny * nz);
            for (int r = 0; r < nranks; ++r) {
                const size_t rn = base + (static_cast<size_t>(r) < rem);
                const size_t rz = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
                if (rn * plane > static_cast<size_t>(std::numeric_limits<int>::max()) || rz * plane > static_cast<size_t>(std::numeric_limits<int>::max()))
                    MPI_Abort(MPI_COMM_WORLD, 3);
                counts[r] = static_cast<int>(rn * plane);
                displs[r] = static_cast<int>(rz * plane);
            }
        }
        MPI_Gatherv(host.data() + plane, static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank ? nullptr : global.data(), rank ? nullptr : counts.data(),
                    rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int rc = 0;
    if (!rank) {
        const double updates = double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
        if (results) print_results(global, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            rc = validate_result(global) ? 0 : 1;
            std::printf("Validation: %s\n", rc ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFreeHost(send_lo); cudaFreeHost(send_hi); cudaFreeHost(recv_lo); cudaFreeHost(recv_hi);
    cudaFree(d_a); cudaFree(d_b);
    MPI_Finalize();
    return rc;
}
