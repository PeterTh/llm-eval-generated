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

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void clear_upper(double *a, size_t n, size_t first, size_t rows) {
    size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * n;
    if (q < total) {
        size_t lr = q / n, j = q - lr * n;
        if (j > first + lr) a[q] = 0.0;
    }
}

__global__ void scale_column(double *a, size_t n, size_t first, size_t rows,
                             size_t k, double diagonal) {
    size_t lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr < rows) {
        size_t i = first + lr;
        if (i > k) a[lr * n + k] /= diagonal;
    }
}

__global__ void extract_column(const double *a, double *part, size_t n,
                               size_t first, size_t rows, size_t k) {
    size_t lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr < rows) part[lr] = a[lr * n + k];
}

__global__ void trailing_update(double *a, const double *column, size_t n,
                                size_t first, size_t rows, size_t k) {
    size_t j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    size_t lr = blockIdx.y * blockDim.y + threadIdx.y;
    if (lr < rows && j < n) {
        size_t i = first + lr;
        if (i > k && j <= i) a[lr * n + j] -= column[i] * column[j];
    }
}

static void partition(size_t n, int p, std::vector<int>& rows,
                      std::vector<int>& matrix_counts, std::vector<int>& matrix_displs) {
    rows.resize(p); matrix_counts.resize(p); matrix_displs.resize(p);
    size_t off = 0;
    for (int r = 0; r < p; ++r) {
        size_t nr = n / p + (static_cast<size_t>(r) < n % p);
        if (nr > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            nr * n > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "Matrix is too large for MPI counts\n"); MPI_Abort(MPI_COMM_WORLD, 3);
        }
        rows[r] = static_cast<int>(nr);
        matrix_counts[r] = static_cast<int>(nr * n);
        matrix_displs[r] = static_cast<int>(off * n);
        off += nr;
    }
}

static int owner_of(size_t row, const std::vector<int>& rows) {
    size_t off = 0;
    for (int r = 0; r < static_cast<int>(rows.size()); ++r) {
        if (row < off + static_cast<size_t>(rows[r])) return r;
        off += rows[r];
    }
    return static_cast<int>(rows.size()) - 1;
}

static bool distributed_cholesky(std::vector<double>& local, size_t n, size_t first,
                                 const std::vector<int>& rows, int rank) {
    const size_t nr = rows[rank];
    double *d_a = nullptr, *d_part = nullptr, *d_column = nullptr;
    CUDA_OK(cudaMalloc(&d_a, std::max<size_t>(1, nr * n) * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_part, std::max<size_t>(1, nr) * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_column, std::max<size_t>(1, n) * sizeof(double)));
    if (nr) CUDA_OK(cudaMemcpy(d_a, local.data(), nr * n * sizeof(double), cudaMemcpyHostToDevice));
    if (nr) {
        clear_upper<<<static_cast<unsigned>((nr*n + 255)/256), 256>>>(d_a, n, first, nr);
        CUDA_OK(cudaGetLastError());
    }

    std::vector<int> displs(rows.size(), 0);
    for (size_t r = 1; r < rows.size(); ++r) displs[r] = displs[r-1] + rows[r-1];
    std::vector<double> part(nr), column(n);
    bool ok = true;
    for (size_t k = 0; k < n; ++k) {
        int owner = owner_of(k, rows);
        double diagonal = 0.0;
        if (rank == owner) {
            size_t lk = k - static_cast<size_t>(displs[rank]);
            CUDA_OK(cudaMemcpy(&diagonal, d_a + lk*n + k, sizeof(double), cudaMemcpyDeviceToHost));
            if (diagonal > 0.0) {
                diagonal = std::sqrt(diagonal);
                CUDA_OK(cudaMemcpy(d_a + lk*n + k, &diagonal, sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&diagonal, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) { ok = false; break; }

        if (nr) {
            scale_column<<<static_cast<unsigned>((nr + 255)/256), 256>>>(d_a, n, first, nr, k, diagonal);
            extract_column<<<static_cast<unsigned>((nr + 255)/256), 256>>>(d_a, d_part, n, first, nr, k);
            CUDA_OK(cudaGetLastError());
        }
        if (nr) CUDA_OK(cudaMemcpy(part.data(), d_part, nr*sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(part.data(), rows[rank], MPI_DOUBLE, column.data(), rows.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_OK(cudaMemcpy(d_column, column.data(), n*sizeof(double), cudaMemcpyHostToDevice));
        dim3 block(32, 8), grid(static_cast<unsigned>((n-k-1+block.x-1)/block.x),
                               static_cast<unsigned>((nr+block.y-1)/block.y));
        if (k + 1 < n && nr) trailing_update<<<grid, block>>>(d_a, d_column, n, first, nr, k);
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaDeviceSynchronize());
    if (nr) CUDA_OK(cudaMemcpy(local.data(), d_a, nr*n*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_column); cudaFree(d_part); cudaFree(d_a);
    return ok;
}

static void generate_matrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n);
    // Retain the original benchmark's deterministic random-number stream.
    unsigned int seed = 42;
    for (size_t i = 0; i < n*n; ++i) {
        B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) sum += B[i*n+k] * B[j*n+k];
            A[i*n+j] = sum;
        }
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) A[i*n+i] += n;
}

static bool validate(const std::vector<double>& L, const std::vector<double>& A, size_t n) {
    double max_abs = 0.0, max_rel = 0.0;
    #pragma omp parallel for reduction(max:max_abs,max_rel) schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(n); ++ii)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0; size_t lim = std::min(static_cast<size_t>(ii), j);
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= lim; ++k) sum += L[ii*n+k] * L[j*n+k];
            double e = std::fabs(sum-A[ii*n+j]);
            max_abs = std::max(max_abs, e); max_rel = std::max(max_rel, e/(std::fabs(A[ii*n+j])+1e-10));
        }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_abs, max_rel);
    return max_rel <= 1e-6;
}

static void usage(const char *p) { printf("Usage: %s [-n num] [-v] [-r] [-h]\n", p); }

int main(int argc, char **argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank); int devices = 0; CUDA_OK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "A CUDA GPU is required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_OK(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local_comm);

    size_t n = 512; bool do_validate = false, results = false; int status = 0;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1<argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-v")) do_validate=true;
        else if (!strcmp(argv[i], "-r")) results=true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else status=1;
    }
    if (!n || status) { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    std::vector<int> rows, counts, displs; partition(n, size, rows, counts, displs);
    size_t first = static_cast<size_t>(displs[rank]) / n;
    std::vector<double> full, original;
    if (!rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
               n,n,do_validate?"enabled":"disabled",size,omp_get_max_threads());
        full.resize(n*n); printf("Generating positive definite matrix...\n"); generate_matrix(full,n);
        if (do_validate) original=full;
    }
    std::vector<double> local(static_cast<size_t>(counts[rank]));
    MPI_Scatterv(rank ? nullptr : full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (!rank) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    bool ok=distributed_cholesky(local,n,first,rows,rank);
    MPI_Barrier(MPI_COMM_WORLD); double elapsed=MPI_Wtime()-start;
    double max_elapsed=0.0; MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    int all_ok=0, local_ok=ok; MPI_Allreduce(&local_ok,&all_ok,1,MPI_INT,MPI_LAND,MPI_COMM_WORLD);
    MPI_Gatherv(local.data(),counts[rank],MPI_DOUBLE,rank?nullptr:full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (!rank) {
        if (!all_ok) { fprintf(stderr,"Cholesky decomposition failed: matrix is not positive definite\n"); status=1; }
        else {
            printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",max_elapsed*1000.0,
                   (static_cast<double>(n)*n*n/3.0)/max_elapsed/1e9);
            if (results) print_results(full,"CholeskyL");
            if (do_validate) { printf("Validating result...\n"); bool valid=validate(full,original,n);
                printf("Validation: %s\n",valid?"PASSED":"FAILED"); if(!valid) status=1; }
        }
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return status;
}
