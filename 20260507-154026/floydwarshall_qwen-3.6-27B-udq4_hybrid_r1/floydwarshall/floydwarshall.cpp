#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

/* ------------------------------------------------------------------ */
/*  CUDA kernel – FW update step: dist[i][j] = min(dist[i][j],        */
/*  colK[i]+rowK[j]).  Row-major layout, each rank owns rows          */
/*  [gRowOff, gRowOff+localRows).                                      */
/* ------------------------------------------------------------------ */
__global__ void fwUpdateKernel(
    unsigned int* dist, const unsigned int* rowK, const unsigned int* colK,
    unsigned int* path,
    size_t localRows, size_t numNodes, size_t gRowOff,
    unsigned int k)
{
    size_t li = blockIdx.y * blockDim.y + threadIdx.y;
    size_t lj = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < localRows && lj < numNodes) {
        size_t gI  = gRowOff + li;
        size_t idx = li * numNodes + lj;
        unsigned int nd = colK[gI] + rowK[lj];
        if (nd < dist[idx]) {
            dist[idx] = nd;
            path[idx] = k;
        }
    }
}

/* Gather a single column from row-major matrix into contiguous buffer */
__global__ void gatherColumnKernel(const unsigned int* dist,
                                   unsigned int* colOut,
                                   size_t localRows, size_t numNodes,
                                   size_t colIdx)
{
    size_t li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < localRows)
        colOut[li] = dist[li * numNodes + colIdx];
}

static void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    /* ── parse CLI (rank 0) ─────────────────────────────────────── */
    size_t numNodes = 512;
    int validate = 0, printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                numNodes = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [-n nodes] [-v] [-r] [-h]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize(); return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
        }
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Bcast(&numNodes,    1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,    1, MPI_INT,           0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults,1, MPI_INT,           0, MPI_COMM_WORLD);

    /* ── row-wise MPI distribution ──────────────────────────────── */
    size_t baseLocalRows = numNodes / numRanks;
    size_t extraRows     = numNodes % numRanks;
    size_t localRows     = baseLocalRows + (rank < (int)extraRows ? 1 : 0);
    size_t gRowOff       = baseLocalRows * rank + std::min((size_t)rank, extraRows);
    size_t localSize     = localRows * numNodes;

    /* ── local buffers (row-major: lDist[li*N + lj] = dist[gRowOff+li][lj]) */
    std::vector<unsigned int> lDist(localSize), lPath(localSize);

    /* ── initialise (rank 0 builds full matrices → scatter) ────── */
    if (rank == 0) {
        printf("Initializing graph...\n");

        std::vector<unsigned int> fDist(numNodes * numNodes);
        std::vector<unsigned int> fPath(numNodes * numNodes);

        /* distance matrix – same seed/sequence as original */
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        for (size_t i = 0; i < numNodes * numNodes; ++i)
            fDist[i] = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        for (size_t i = 0; i < numNodes; ++i)
            fDist[i * numNodes + i] = 0;

        /* path matrix – equivalent to original, parallelized with OpenMP */
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t j = 0; j < numNodes; ++j)
            for (size_t i = 0; i < numNodes; ++i)
                fPath[i * numNodes + j] = (unsigned int)i;

        /* keep rank-0 local rows */
        for (size_t li = 0; li < localRows; ++li)
            for (size_t lj = 0; lj < numNodes; ++lj) {
                lDist[li * numNodes + lj] = fDist[li * numNodes + lj];
                lPath[li * numNodes + lj] = fPath[li * numNodes + lj];
            }

        /* scatter to other ranks */
        for (int r = 1; r < numRanks; ++r) {
            size_t gOff = baseLocalRows * r + std::min((size_t)r, extraRows);
            size_t lr   = baseLocalRows + (r < (int)extraRows ? 1 : 0);
            size_t sz   = lr * numNodes;
            std::vector<unsigned int> blk(sz);
            for (size_t li = 0; li < lr; ++li)
                for (size_t lj = 0; lj < numNodes; ++lj)
                    blk[li * numNodes + lj] = fDist[(gOff + li) * numNodes + lj];
            MPI_Send(blk.data(), (int)sz, MPI_UNSIGNED, r, 100, MPI_COMM_WORLD);

            for (size_t li = 0; li < lr; ++li)
                for (size_t lj = 0; lj < numNodes; ++lj)
                    blk[li * numNodes + lj] = fPath[(gOff + li) * numNodes + lj];
            MPI_Send(blk.data(), (int)sz, MPI_UNSIGNED, r, 101, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(lDist.data(), (int)localSize, MPI_UNSIGNED, 0, 100,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(lPath.data(), (int)localSize, MPI_UNSIGNED, 0, 101,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── CUDA device allocation ─────────────────────────────────── */
    unsigned int *d_dist = nullptr, *d_path = nullptr,
                 *d_rowK = nullptr, *d_colK = nullptr,
                 *d_colK_local = nullptr;
    cudaStream_t stream = 0;

    if (localSize > 0) {
        int devCount = 0;
        cudaGetDeviceCount(&devCount);
        if (devCount > 0) cudaSetDevice(rank % devCount);

        cudaCheck(cudaMalloc(&d_dist, localSize * sizeof(unsigned int)), "d_dist");
        cudaCheck(cudaMalloc(&d_path, localSize * sizeof(unsigned int)), "d_path");
        cudaCheck(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)), "d_rowK");
        cudaCheck(cudaMalloc(&d_colK, numNodes * sizeof(unsigned int)), "d_colK");
        cudaCheck(cudaMalloc(&d_colK_local, localRows * sizeof(unsigned int)), "d_colK_local");
        cudaCheck(cudaStreamCreate(&stream), "stream");
        cudaCheck(cudaMemcpy(d_dist, lDist.data(),
                             localSize * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "h2d dist");
        cudaCheck(cudaMemcpy(d_path, lPath.data(),
                             localSize * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "h2d path");
    }

    /* ── precompute kOwner and Allgatherv parameters ────────────── */
    std::vector<int> kOwner(numNodes);
    {
        size_t off = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t lr = baseLocalRows + (r < (int)extraRows ? 1 : 0);
            for (size_t i = off; i < off + lr; ++i)
                kOwner[i] = r;
            off += lr;
        }
    }

    std::vector<int> rcounts(numRanks), displs(numRanks);
    {
        size_t d = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t lr = baseLocalRows + (r < (int)extraRows ? 1 : 0);
            rcounts[r] = (int)lr;
            displs[r]  = (int)d;
            d += lr;
        }
    }

    std::vector<unsigned int> rowK(numNodes), colK(numNodes);
    std::vector<unsigned int> colK_local(localRows);

    /* ── main FW loop ───────────────────────────────────────────── */
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        int k_owner = kOwner[k];

        /* ── broadcast row k from the owning rank ─────────────── */
        if (rank == k_owner) {
            if (localSize > 0) {
                size_t li = k - gRowOff;
                cudaCheck(cudaMemcpyAsync(rowK.data(),
                    d_dist + li * numNodes,
                    numNodes * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost, stream), "d2h rowK");
                cudaCheck(cudaStreamSynchronize(stream), "sync rowK");
            }
        }
        MPI_Bcast(rowK.data(), (int)numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);

        /* ── gather column k from all ranks ───────────────────── */
        if (localSize > 0) {
            int blk = 256;
            int grds = (localRows + blk - 1) / blk;
            gatherColumnKernel<<<grds, blk, 0, stream>>>(
                d_dist, d_colK_local, localRows, numNodes, k);
            cudaCheck(cudaMemcpyAsync(colK_local.data(), d_colK_local,
                localRows * sizeof(unsigned int),
                cudaMemcpyDeviceToHost, stream), "d2h colK");
            cudaCheck(cudaStreamSynchronize(stream), "sync colK");
        }
        MPI_Allgatherv(colK_local.data(), (int)localRows, MPI_UNSIGNED,
                       colK.data(), rcounts.data(), displs.data(),
                       MPI_UNSIGNED, MPI_COMM_WORLD);

        /* ── GPU update: dist[i][j] = min(dist[i][j], colK[i] + rowK[j]) */
        if (localSize > 0) {
            cudaCheck(cudaMemcpyAsync(d_rowK, rowK.data(),
                                      numNodes * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice, stream), "h2d rowK");
            cudaCheck(cudaMemcpyAsync(d_colK, colK.data(),
                                      numNodes * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice, stream), "h2d colK");

            dim3 bs(16, 16);
            dim3 gs((numNodes + 15) / 16, (localRows + 15) / 16);
            fwUpdateKernel<<<gs, bs, 0, stream>>>(d_dist, d_rowK, d_colK, d_path,
                                                    localRows, numNodes, gRowOff,
                                                    (unsigned int)k);
            cudaCheck(cudaGetLastError(), "kernel launch");
            cudaCheck(cudaStreamSynchronize(stream), "sync kernel");
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double local_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms = 0.0;
    MPI_Reduce(&local_ms, &ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ── copy results back from GPU ─────────────────────────────── */
    if (localSize > 0) {
        cudaCheck(cudaMemcpy(lDist.data(), d_dist,
                             localSize * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "d2h dist");
        cudaCheck(cudaMemcpy(lPath.data(), d_path,
                             localSize * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "d2h path");
        cudaFree(d_dist); cudaFree(d_path);
        cudaFree(d_rowK); cudaFree(d_colK);
    }

    /* ── gather full matrix to rank 0 ───────────────────────────── */
    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        for (size_t li = 0; li < localRows; ++li)
            for (size_t lj = 0; lj < numNodes; ++lj)
                fullDist[li * numNodes + lj] = lDist[li * numNodes + lj];

        for (int r = 1; r < numRanks; ++r) {
            size_t gOff = baseLocalRows * r + std::min((size_t)r, extraRows);
            size_t lr   = baseLocalRows + (r < (int)extraRows ? 1 : 0);
            size_t sz   = lr * numNodes;
            std::vector<unsigned int> blk(sz);
            MPI_Recv(blk.data(), (int)sz, MPI_UNSIGNED, r, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t li = 0; li < lr; ++li)
                for (size_t lj = 0; lj < numNodes; ++lj)
                    fullDist[(gOff + li) * numNodes + lj] = blk[li * numNodes + lj];
        }
    } else {
        MPI_Send(lDist.data(), (int)localSize, MPI_UNSIGNED,
                 0, 200, MPI_COMM_WORLD);
    }

    /* ── timing & results (rank 0) ──────────────────────────────── */
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", ms);
        double ops = (double)numNodes * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / (ms / 1000.0) / 1e9);

        if (printResults)
            print_results_int(fullDist, "DistanceMatrix");

        if (validate) {
            printf("Validating result...\n");
            bool ok = true;
            for (size_t i = 0; i < numNodes && ok; ++i)
                if (fullDist[i * numNodes + i] != 0) {
                    printf("Validation failed: diagonal [%zu,%zu] != 0\n", i, i);
                    ok = false;
                }
            if (ok) {
                for (size_t i = 0; i < std::min(numNodes, (size_t)10) && ok; ++i)
                    for (size_t j = 0; j < std::min(numNodes, (size_t)10) && ok; ++j)
                        for (size_t kk = 0; kk < numNodes; ++kk) {
                            unsigned int dIJ = fullDist[i * numNodes + j];
                            unsigned int dIK = fullDist[i * numNodes + kk];
                            unsigned int dKJ = fullDist[kk * numNodes + j];
                            if (dIK < INF && dKJ < INF && dIK + dKJ < dIJ) {
                                printf("Validation failed: triangle inequality at [%zu,%zu,%zu]\n",
                                       i, j, kk);
                                ok = false;
                            }
                        }
            }
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
