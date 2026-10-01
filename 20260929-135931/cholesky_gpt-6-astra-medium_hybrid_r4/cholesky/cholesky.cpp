#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cerrno>
#include <vector>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/results_output.hpp"

namespace {
constexpr int block = 256;
int rankId, ranks;
void cudaCheck(cudaError_t s) {
    if (s != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", rankId, cudaGetErrorString(s));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void blasCheck(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Rank %d: cuBLAS error %d\n", rankId, int(s));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
struct Device {
    double* p = nullptr;
    explicit Device(size_t count) { cudaCheck(cudaMalloc(&p, std::max(size_t(1), count)*sizeof(double))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) { cudaCheck(cudaMallocHost(&p, std::max(size_t(1), count)*sizeof(double))); }
    ~Pinned() { cudaFreeHost(p); }
};
std::vector<int> rowMap(int n, int r) {
    std::vector<int> rows;
    for (int k = r*block; k < n; k += ranks*block)
        for (int i = k; i < std::min(n, k+block); ++i) rows.push_back(i);
    return rows;
}
// Gather only for requested output/validation; the timed factorization stays distributed.
std::vector<double> gather(const Device& a, int n, const std::vector<int>& rows) {
    std::vector<double> local(size_t(rows.size())*n);
    cudaCheck(cudaMemcpy(local.data(), a.p, local.size()*sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> result;
    if (rankId == 0) result.resize(size_t(n)*n);
    for (int r = 0; r < ranks; ++r) {
        auto map = rowMap(n, r);
        size_t count = size_t(map.size())*n;
        std::vector<double> received;
        if (rankId == 0 && r != 0) received.resize(count);
        double* data = r == 0 ? local.data() : received.data();
        for (size_t off = 0; off < count; off += INT_MAX) {
            int chunk = int(std::min(size_t(INT_MAX), count-off));
            if (rankId == r && r != 0) MPI_Send(local.data()+off, chunk, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
            if (rankId == 0 && r != 0) MPI_Recv(data+off, chunk, MPI_DOUBLE, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rankId == 0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < int(map.size()); ++i)
                for (int j = 0; j < n; ++j) result[size_t(map[i])*n+j] = data[i+size_t(j)*map.size()];
        }
    }
    return result;
}

void generate(Device& a, int n, const std::vector<int>& rows, cublasHandle_t handle) {
    const int m = int(rows.size()), ld = std::max(1, m);
    // Replay the original stream without replicating the dense input on each rank.
    // Only owned rows and one column tile are retained, keeping memory O(n^2/P+n*b).
    std::vector<double> local(size_t(m)*n);
    unsigned seed = 42;
    int owned = 0;
    for (int i = 0; i < n; ++i) {
        bool keep = owned < m && rows[owned] == i;
        for (int j = 0; j < n; ++j) {
            double x = rand_r(&seed)/double(RAND_MAX)-0.5;
            if (keep) local[owned+size_t(j)*m] = x;
        }
        if (keep) ++owned;
    }
    Device db(local.size()), tile(size_t(n)*block);
    cudaCheck(cudaMemcpy(db.p, local.data(), local.size()*sizeof(double), cudaMemcpyHostToDevice));
    Pinned packed(size_t(n)*block);
    const double one = 1, zero = 0;
    seed = 42;
    for (int j = 0; j < n; j += block) {
        int width = std::min(block, n-j);
        // A row-major block of B is already a column-major block of B^T.
        for (size_t t = 0; t < size_t(width)*n; ++t)
            packed.p[t] = rand_r(&seed)/double(RAND_MAX)-0.5;
        cudaCheck(cudaMemcpy(tile.p, packed.p, size_t(width)*n*sizeof(double), cudaMemcpyHostToDevice));
        if (m) blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, width, n,
            &one, db.p, ld, tile.p, n, &zero, a.p+size_t(j)*ld, ld));
    }
    cudaCheck(cudaMemcpy(local.data(), a.p, local.size()*sizeof(double), cudaMemcpyDeviceToHost));
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) local[i+size_t(rows[i])*m] += n;
    cudaCheck(cudaMemcpy(a.p, local.data(), local.size()*sizeof(double), cudaMemcpyHostToDevice));
}

bool factor(Device& a, int n, const std::vector<int>& rows, cublasHandle_t handle) {
    const int m = int(rows.size()), ld = std::max(1, m);
    Device diagonal(block*block), panel(size_t(n)*block);
    Pinned diag(block*block), send(size_t(m)*block), received(size_t(n)*block), packed(size_t(n)*block);
    std::vector<std::vector<int>> maps(ranks);
    for (int r = 0; r < ranks; ++r) maps[r] = rowMap(n, r);
    std::vector<int> counts(ranks), offsets(ranks), starts(ranks);
    const double one = 1, minus = -1;
    for (int k = 0; k < n; k += block) {
        int b = std::min(block, n-k), owner = (k/block)%ranks, info = 0;
        if (rankId == owner) {
            int first = int(std::lower_bound(rows.begin(), rows.end(), k)-rows.begin());
            cudaCheck(cudaMemcpy2D(diag.p, b*sizeof(double), a.p+first+size_t(k)*ld,
                ld*sizeof(double), b*sizeof(double), b, cudaMemcpyDeviceToHost));
            // Small diagonal panel on the CPU; all cubic trailing work is on GPUs.
            for (int j = 0; j < b; ++j) {
                double d = diag.p[j+size_t(j)*b];
                for (int t = 0; t < j; ++t) d -= diag.p[j+size_t(t)*b]*diag.p[j+size_t(t)*b];
                if (!(d > 0) || !std::isfinite(d)) { info = k+j+1; break; }
                diag.p[j+size_t(j)*b] = std::sqrt(d);
                for (int i = j+1; i < b; ++i) {
                    double v = diag.p[i+size_t(j)*b];
                    for (int t = 0; t < j; ++t) v -= diag.p[i+size_t(t)*b]*diag.p[j+size_t(t)*b];
                    diag.p[i+size_t(j)*b] = v/diag.p[j+size_t(j)*b];
                }
            }
            if (!info) cudaCheck(cudaMemcpy2D(a.p+first+size_t(k)*ld, ld*sizeof(double),
                diag.p, b*sizeof(double), b*sizeof(double), b, cudaMemcpyHostToDevice));
        }
        MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (info) {
            if (!rankId) printf("Error: Matrix is not positive definite at diagonal element %d\n", info-1);
            return false;
        }
        int trailing = n-k-b;
        if (!trailing) break;
        MPI_Bcast(diag.p, b*b, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(diagonal.p, diag.p, size_t(b)*b*sizeof(double), cudaMemcpyHostToDevice));
        int start = int(std::lower_bound(rows.begin(), rows.end(), k+b)-rows.begin()), remaining = m-start;
        if (remaining) {
            blasCheck(cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, remaining, b, &one, diagonal.p, b,
                a.p+start+size_t(k)*ld, ld));
            cudaCheck(cudaMemcpy2D(send.p, remaining*sizeof(double), a.p+start+size_t(k)*ld,
                ld*sizeof(double), remaining*sizeof(double), b, cudaMemcpyDeviceToHost));
        }
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            starts[r] = int(std::lower_bound(maps[r].begin(), maps[r].end(), k+b)-maps[r].begin());
            counts[r] = (int(maps[r].size())-starts[r])*b;
            offsets[r] = total; total += counts[r];
        }
        MPI_Allgatherv(send.p, remaining*b, MPI_DOUBLE, received.p, counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j < b; ++j)
            for (int r = 0; r < ranks; ++r) {
                int nr = counts[r]/b;
                for (int i = 0; i < nr; ++i)
                    packed.p[maps[r][starts[r]+i]-k-b+size_t(j)*trailing] = received.p[offsets[r]+i+size_t(j)*nr];
            }
        if (remaining) {
            cudaCheck(cudaMemcpy(panel.p, packed.p, size_t(trailing)*b*sizeof(double), cudaMemcpyHostToDevice));
            blasCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remaining, trailing, b,
                &minus, a.p+start+size_t(k)*ld, ld, panel.p, trailing, &one,
                a.p+start+size_t(k+b)*ld, ld));
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    return true;
}

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n, cublasHandle_t handle) {
    Device dl(l.size()), product(l.size());
    cudaCheck(cudaMemcpy(dl.p, l.data(), l.size()*sizeof(double), cudaMemcpyHostToDevice));
    const double one = 1, zero = 0;
    blasCheck(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one,
        dl.p, n, dl.p, n, &zero, product.p, n));
    std::vector<double> reconstructed(l.size());
    cudaCheck(cudaMemcpy(reconstructed.data(), product.p, l.size()*sizeof(double), cudaMemcpyDeviceToHost));
    double maxError = 0, relError = 0;
    int finite = 1;
    #pragma omp parallel for reduction(max:maxError,relError) reduction(&:finite) schedule(static)
    for (size_t i = 0; i < l.size(); ++i) {
        double error = std::fabs(reconstructed[i]-original[i]);
        finite &= std::isfinite(error);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error/(std::fabs(original[i])+1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return finite && relError <= 1e-6;
}
void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}
int run(int argc, char** argv) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1 < argc) {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || *end || value <= 0 || value > INT_MAX/block) {
                if (!rankId) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rankId) printUsage(argv[0]); return 0; }
        else { if (!rankId) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } return 1; }
    }
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &shared);
    int localRank, localSize, devices;
    MPI_Comm_rank(shared, &localRank);
    MPI_Comm_size(shared, &localSize);
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs()/localSize)));
    MPI_Comm_free(&shared);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: a CUDA GPU is required\n", rankId); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank%devices));
    cublasHandle_t handle;
    blasCheck(cublasCreate(&handle));
    auto rows = rowMap(n, rankId);
    Device a(size_t(rows.size())*n);
    if (!rankId) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }
    generate(a, n, rows, handle);
    std::vector<double> original;
    if (validate) original = gather(a, n, rows);
    cudaCheck(cudaDeviceSynchronize());
    if (!rankId) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool success = factor(a, n, rows, handle);
    double elapsed = MPI_Wtime()-start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int result = success ? 0 : 1;
    if (!rankId) {
        if (!success) printf("Cholesky decomposition failed\n");
        else printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n",
            static_cast<long long>(seconds*1000), double(n)*n*n/(3*seconds*1e9));
    }
    if (success && (printResults || validate)) {
        auto l = gather(a, n, rows);
        if (!rankId) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                for (int j = i+1; j < n; ++j) l[size_t(i)*n+j] = 0;
            if (printResults) print_results(l, "CholeskyL");
            if (validate) {
                printf("Validating result...\n");
                result = validateCholesky(l, original, n, handle) ? 0 : 1;
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    blasCheck(cublasDestroy(handle));
    return result;
}
} // namespace
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rankId, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
