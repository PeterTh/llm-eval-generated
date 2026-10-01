#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"
using Real = double;

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); }
}
static void check(int e, const char* what) {
    if (e != MPI_SUCCESS) { fprintf(stderr, "%s: MPI error %d\n", what, e); MPI_Abort(MPI_COMM_WORLD, 1); }
}

__global__ void stencil(const Real* __restrict__ a, Real* __restrict__ b,
                        size_t nx, size_t ny, size_t nz, size_t z0, size_t first, size_t last) {
    size_t plane = nx * ny;
    size_t count = (last - first + 1) * plane;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < count;
         t += size_t(gridDim.x) * blockDim.x) {
        size_t z = first + t / plane, p = t % plane, y = p / nx, x = p % nx;
        size_t i = z * plane + p;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z0 + z == 1 || z0 + z == nz)
            b[i] = a[i];
        else {
            Real v = a[i];
            v += a[i-1]; v += a[i+1]; v += a[i-nx]; v += a[i+nx];
            v += a[i-plane]; v += a[i+plane];
            b[i] = v / 7.0;
        }
    }
}
static void launch(const Real* a, Real* b, size_t nx, size_t ny, size_t nz,
                   size_t z0, size_t first, size_t last, cudaStream_t stream) {
    if (first > last) return;
    size_t count = (last - first + 1) * nx * ny;
    int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 65535));
    stencil<<<blocks, 256, 0, stream>>>(a, b, nx, ny, nz, z0, first, last);
    check(cudaGetLastError(), "stencil launch");
}
static void usage(const char* program) {
    printf("Usage: %s [options]\n  -x <num> Grid size X (default: 128)\n  -y <num> Grid size Y (default: X)\n  -z <num> Grid size Z (default: X)\n  -i <num> Iterations (default: 10)\n  -v Validate\n  -r Print results\n  -h Help\n", program);
}
static bool number(const char* text, size_t& value) {
    if (*text == '-' || *text == '+') return false;
    char* end = nullptr;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (end == text || *end || parsed == 0 || parsed > SIZE_MAX) return false;
    value = static_cast<size_t>(parsed); return true;
}
int main(int argc, char** argv) {
    check(MPI_Init(&argc, &argv), "MPI_Init");
    int rank, ranks;
    check(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    check(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, results = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (strlen(argv[i]) == 2 && argv[i][0] == '-' &&
            (argv[i][1] == 'x' || argv[i][1] == 'y' || argv[i][1] == 'z' || argv[i][1] == 'i') && i+1 < argc) {
            char option = argv[i][1]; size_t n = 0;
            if (option == 'i') {
                char* end = nullptr; long v = strtol(argv[++i], &end, 10);
                if (*end || v < 0 || v > INT_MAX) bad = true; else iterations = static_cast<int>(v);
            } else if (!number(argv[++i], n)) bad = true;
            else if (option == 'x') nx = n; else if (option == 'y') ny = n; else nz = n;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (rank == 0 && (help || bad)) usage(argv[0]);
    if (help || bad) { MPI_Finalize(); return bad ? 1 : 0; }
    if (nx < 2 || ny < 2 || nz < 2 || nx > SIZE_MAX / ny || nx*ny > SIZE_MAX/nz || nx*ny > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid dimensions or MPI plane too large\n");
        MPI_Finalize(); return 1;
    }
    int active = static_cast<int>(std::min<size_t>(nz, ranks));
    MPI_Comm comm;
    check(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm), "MPI_Comm_split");
    if (rank >= active) { MPI_Finalize(); return 0; }
    size_t plane = nx*ny;
    size_t localZ = nz/active + (static_cast<size_t>(rank) < nz%active);
    size_t z0 = size_t(rank)*(nz/active) + std::min<size_t>(rank, nz%active);
    size_t localCount = (localZ+2)*plane;
    if (localCount > SIZE_MAX/sizeof(Real)) { fprintf(stderr, "Local grid too large\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int devices = 0; check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm shared;
    check(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared), "MPI_Comm_split_type");
    int localRank; check(MPI_Comm_rank(shared, &localRank), "shared rank");
    check(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&shared);
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; CUDA devices per node: %d; OpenMP threads: %d\nInitializing grid...\n", active, devices, omp_get_max_threads());
    }
    std::vector<Real> initial(localCount, 0.0);
#pragma omp parallel for schedule(static)
    for (size_t z = 1; z <= localZ; ++z)
        for (size_t j = 0; j < plane; ++j)
            initial[z*plane+j] = static_cast<Real>(((z0+z-1)*plane+j)%19);
    Real *a = nullptr, *b = nullptr;
    check(cudaMalloc(&a, localCount*sizeof(Real)), "cudaMalloc(a)");
    check(cudaMalloc(&b, localCount*sizeof(Real)), "cudaMalloc(b)");
    check(cudaMemcpy(a, initial.data(), localCount*sizeof(Real), cudaMemcpyHostToDevice), "initial copy");
    std::vector<Real>().swap(initial);
    Real *sendLow = nullptr, *sendHigh = nullptr, *recvLow = nullptr, *recvHigh = nullptr;
    size_t bytes = plane*sizeof(Real);
    if (active > 1) {
        check(cudaMallocHost(&sendLow, bytes), "sendLow allocation");
        check(cudaMallocHost(&sendHigh, bytes), "sendHigh allocation");
        check(cudaMallocHost(&recvLow, bytes), "recvLow allocation");
        check(cudaMallocHost(&recvHigh, bytes), "recvHigh allocation");
    }
    cudaStream_t compute, transfer;
    check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking), "compute stream");
    check(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking), "transfer stream");
    if (rank == 0) printf("Running stencil computation...\n");
    check(MPI_Barrier(comm), "MPI_Barrier");
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (active == 1) launch(a, b, nx, ny, nz, z0, 1, localZ, compute);
        else {
            MPI_Request req[4]; int nreq = 0;
            if (rank > 0) check(MPI_Irecv(recvLow, static_cast<int>(plane), MPI_DOUBLE, rank-1, 1, comm, &req[nreq++]), "Irecv low");
            if (rank+1 < active) check(MPI_Irecv(recvHigh, static_cast<int>(plane), MPI_DOUBLE, rank+1, 0, comm, &req[nreq++]), "Irecv high");
            if (localZ > 2) launch(a, b, nx, ny, nz, z0, 2, localZ-1, compute);
            if (rank > 0) check(cudaMemcpyAsync(sendLow, a+plane, bytes, cudaMemcpyDeviceToHost, transfer), "send low copy");
            if (rank+1 < active) check(cudaMemcpyAsync(sendHigh, a+localZ*plane, bytes, cudaMemcpyDeviceToHost, transfer), "send high copy");
            check(cudaStreamSynchronize(transfer), "send plane copies");
            if (rank > 0) check(MPI_Isend(sendLow, static_cast<int>(plane), MPI_DOUBLE, rank-1, 0, comm, &req[nreq++]), "Isend low");
            if (rank+1 < active) check(MPI_Isend(sendHigh, static_cast<int>(plane), MPI_DOUBLE, rank+1, 1, comm, &req[nreq++]), "Isend high");
            check(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE), "MPI_Waitall");
            if (rank > 0) check(cudaMemcpyAsync(a, recvLow, bytes, cudaMemcpyHostToDevice, transfer), "low halo copy");
            if (rank+1 < active) check(cudaMemcpyAsync(a+(localZ+1)*plane, recvHigh, bytes, cudaMemcpyHostToDevice, transfer), "high halo copy");
            check(cudaStreamSynchronize(transfer), "receive halo copies");
            launch(a, b, nx, ny, nz, z0, 1, 1, compute);
            if (localZ > 1) launch(a, b, nx, ny, nz, z0, localZ, localZ, compute);
        }
        check(cudaStreamSynchronize(compute), "stencil computation");
        std::swap(a, b);
    }
    double localTime = MPI_Wtime()-start, elapsed = 0;
    check(MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm), "time reduction");
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed*1000);
        double updates = double(nx-2)*double(ny-2)*double(nz-2)*iterations;
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates/elapsed/1e6 : 0.0);
    }
    if (validate || results) {
        std::vector<Real> local(localZ*plane);
        check(cudaMemcpy(local.data(), a+plane, local.size()*sizeof(Real), cudaMemcpyDeviceToHost), "final copy");
        if (results) {
            if (rank == 0) {
                std::vector<Real> global(nx*ny*nz);
                std::copy(local.begin(), local.end(), global.begin());
                for (int src = 1; src < active; ++src) {
                    size_t n = nz/active + (size_t(src) < nz%active);
                    size_t first = size_t(src)*(nz/active) + std::min<size_t>(src, nz%active);
                    for (size_t offset = 0; offset < n*plane;) {
                        int count = static_cast<int>(std::min<size_t>(n*plane-offset, INT_MAX));
                        check(MPI_Recv(global.data()+first*plane+offset, count, MPI_DOUBLE, src, 2, comm, MPI_STATUS_IGNORE), "result receive");
                        offset += count;
                    }
                }
                print_results(global, "Grid");
            } else {
                for (size_t offset = 0; offset < local.size();) {
                    int count = static_cast<int>(std::min<size_t>(local.size()-offset, INT_MAX));
                    check(MPI_Send(local.data()+offset, count, MPI_DOUBLE, 0, 2, comm), "result send");
                    offset += count;
                }
            }
        }
        if (validate) {
            double low = std::numeric_limits<double>::infinity(), high = -low;
            int invalid = 0;
#pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:invalid)
            for (size_t i = 0; i < local.size(); ++i) {
                double v = local[i];
                if (!std::isfinite(v)) invalid = 1;
                else { low = std::min(low, v); high = std::max(high, v); }
            }
            double globalLow, globalHigh; int globalInvalid;
            check(MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm), "minimum reduction");
            check(MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm), "maximum reduction");
            check(MPI_Reduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, 0, comm), "invalid reduction");
            if (rank == 0) {
                bool valid = !globalInvalid && globalLow >= -1e6 && globalHigh <= 1e6;
                printf("Validating result...\nValue range: [%.6f, %.6f]\nValidation: %s\n", globalLow, globalHigh, valid ? "PASSED" : "FAILED");
                invalid = !valid;
            }
            check(MPI_Bcast(&invalid, 1, MPI_INT, 0, comm), "validation broadcast");
            if (invalid) MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    check(cudaStreamDestroy(compute), "destroy compute stream");
    check(cudaStreamDestroy(transfer), "destroy transfer stream");
    if (active > 1) { check(cudaFreeHost(sendLow), "free sendLow"); check(cudaFreeHost(sendHigh), "free sendHigh");
                      check(cudaFreeHost(recvLow), "free recvLow"); check(cudaFreeHost(recvHigh), "free recvHigh"); }
    check(cudaFree(a), "free a"); check(cudaFree(b), "free b");
    MPI_Comm_free(&comm); MPI_Finalize(); return 0;
}
