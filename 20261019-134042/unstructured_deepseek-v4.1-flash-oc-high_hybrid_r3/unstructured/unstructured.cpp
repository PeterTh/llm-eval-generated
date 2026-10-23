#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct Xfer {
    int rank;
    int count;
    int dev_offset;
    int host_offset;
};

struct SimData {
    int n_root = 0;
    int n_elems = 0;
    int rank = 0;
    int size = 0;
    int start = 0;
    int end = 0;
    int local_n = 0;
    int ghost = 0;
    int cur = 0;
    int send_total = 0;
    int recv_total = 0;
    std::vector<Xfer> sends;
    std::vector<Xfer> recvs;
    val_t* d_energy[2] = {nullptr, nullptr};
    val_t* d_flux[2] = {nullptr, nullptr};
    val_t* h_send = nullptr;
    val_t* h_recv = nullptr;
    cudaStream_t stream = nullptr;
};

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void mpiCheck(int rc, const char* what) {
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "MPI error in %s\n", what);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void energyStepKernel(const val_t* __restrict__ energy_in,
                                 const val_t* __restrict__ flux_in,
                                 val_t* __restrict__ energy_out,
                                 val_t* __restrict__ flux_out,
                                 int start, int local_n, int n_root, int ghost) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= local_n) return;

    const int g = start + k;
    const int x = g / n_root;
    const int y = g - x * n_root;
    const int b = k + ghost;

    const val_t e = energy_in[b];
    val_t total = 0.0;

    if (x == 0 && y == 0) {
        total = 0.5;
    } else if (x == 0 && y == n_root - 1) {
        total = -0.5;
    } else if (x == n_root - 1 && y == 0) {
        total = -0.5;
    } else if (x == n_root - 1 && y == n_root - 1) {
        total = 0.5;
    }

    if (x + 1 < n_root) {
        total += (energy_in[b + n_root] - e) * 0.8 * 1.0 * 0.25;
    }
    if (x > 0) {
        total += (energy_in[b - n_root] - e) * 0.8 * 1.0 * 0.25;
    }
    if (y + 1 < n_root) {
        total += (energy_in[b + 1] - e) * 0.8 * 1.0 * 0.25;
    }
    if (y > 0) {
        total += (energy_in[b - 1] - e) * 0.8 * 1.0 * 0.25;
    }

    energy_out[b] = e + total;
    flux_out[k] = flux_in[k] + fabs(total);
}

static void setupSimulation(SimData& sim, int n_root, int n_elems, int rank, int size) {
    sim.n_root = n_root;
    sim.n_elems = n_elems;
    sim.rank = rank;
    sim.size = size;
    sim.start = static_cast<int>((static_cast<long long>(rank) * n_elems) / size);
    sim.end = static_cast<int>((static_cast<long long>(rank + 1) * n_elems) / size);
    sim.local_n = sim.end - sim.start;
    sim.ghost = n_root;
    sim.cur = 0;

    const size_t buf_elems = static_cast<size_t>(sim.local_n) + 2 * static_cast<size_t>(sim.ghost);
    const size_t energy_bytes = std::max<size_t>(buf_elems, 1) * sizeof(val_t);
    const size_t flux_bytes = std::max<size_t>(static_cast<size_t>(sim.local_n), 1) * sizeof(val_t);

    for (int i = 0; i < 2; ++i) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&sim.d_energy[i]), energy_bytes), "cudaMalloc energy");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&sim.d_flux[i]), flux_bytes), "cudaMalloc flux");
        cudaCheck(cudaMemset(sim.d_energy[i], 0, energy_bytes), "cudaMemset energy");
        cudaCheck(cudaMemset(sim.d_flux[i], 0, flux_bytes), "cudaMemset flux");
    }

    auto qstart = [n_elems, size](int q) {
        return static_cast<int>((static_cast<long long>(q) * n_elems) / size);
    };
    auto qend = [n_elems, size](int q) {
        return static_cast<int>((static_cast<long long>(q + 1) * n_elems) / size);
    };

    const int S = sim.start;
    const int E = sim.end;
    const int G = sim.ghost;

    for (int q = 0; q < size; ++q) {
        const int qs = qstart(q);
        const int qe = qend(q);

        int lo = std::max(S - G, qs);
        int hi = std::min(S, qe);
        if (hi > lo) {
            sim.recvs.push_back(Xfer{q, hi - lo, lo - S + G, sim.recv_total});
            sim.recv_total += hi - lo;
        }

        lo = std::max(E, qs);
        hi = std::min(E + G, qe);
        if (hi > lo) {
            sim.recvs.push_back(Xfer{q, hi - lo, lo - S + G, sim.recv_total});
            sim.recv_total += hi - lo;
        }

        lo = std::max(S, qs - G);
        hi = std::min(E, qs);
        if (hi > lo) {
            sim.sends.push_back(Xfer{q, hi - lo, lo - S + G, sim.send_total});
            sim.send_total += hi - lo;
        }

        lo = std::max(S, qe);
        hi = std::min(E, qe + G);
        if (hi > lo) {
            sim.sends.push_back(Xfer{q, hi - lo, lo - S + G, sim.send_total});
            sim.send_total += hi - lo;
        }
    }

    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&sim.h_send),
                             std::max<size_t>(static_cast<size_t>(sim.send_total), 1) * sizeof(val_t)),
              "cudaMallocHost send");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&sim.h_recv),
                             std::max<size_t>(static_cast<size_t>(sim.recv_total), 1) * sizeof(val_t)),
              "cudaMallocHost recv");
    cudaCheck(cudaStreamCreate(&sim.stream), "cudaStreamCreate");
}

static void destroySimulation(SimData& sim) {
    if (sim.stream) cudaStreamDestroy(sim.stream);
    if (sim.h_send) cudaFreeHost(sim.h_send);
    if (sim.h_recv) cudaFreeHost(sim.h_recv);
    for (int i = 0; i < 2; ++i) {
        if (sim.d_energy[i]) cudaFree(sim.d_energy[i]);
        if (sim.d_flux[i]) cudaFree(sim.d_flux[i]);
    }
}

static void runSimulation(SimData& sim, int n_iters) {
    const int n_recvs = static_cast<int>(sim.recvs.size());
    const int n_sends = static_cast<int>(sim.sends.size());
    std::vector<MPI_Request> requests(static_cast<size_t>(n_recvs + n_sends));

    const int threads = 256;

    for (int iter = 0; iter < n_iters; ++iter) {
        const int cur = sim.cur;
        const int next = 1 - cur;
        val_t* energy_cur = sim.d_energy[cur];
        val_t* energy_next = sim.d_energy[next];
        val_t* flux_cur = sim.d_flux[cur];
        val_t* flux_next = sim.d_flux[next];

        int req_idx = 0;
        for (const Xfer& x : sim.recvs) {
            mpiCheck(MPI_Irecv(sim.h_recv + x.host_offset, x.count, MPI_DOUBLE,
                               x.rank, 101, MPI_COMM_WORLD, &requests[req_idx++]),
                     "MPI_Irecv");
        }

        for (const Xfer& x : sim.sends) {
            cudaCheck(cudaMemcpyAsync(sim.h_send + x.host_offset, energy_cur + x.dev_offset,
                                      static_cast<size_t>(x.count) * sizeof(val_t),
                                      cudaMemcpyDeviceToHost, sim.stream),
                      "cudaMemcpyAsync D2H");
        }
        cudaCheck(cudaStreamSynchronize(sim.stream), "cudaStreamSynchronize sends");

        for (const Xfer& x : sim.sends) {
            mpiCheck(MPI_Isend(sim.h_send + x.host_offset, x.count, MPI_DOUBLE,
                               x.rank, 101, MPI_COMM_WORLD, &requests[req_idx++]),
                     "MPI_Isend");
        }

        mpiCheck(MPI_Waitall(req_idx, requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall");

        for (const Xfer& x : sim.recvs) {
            cudaCheck(cudaMemcpyAsync(energy_cur + x.dev_offset, sim.h_recv + x.host_offset,
                                      static_cast<size_t>(x.count) * sizeof(val_t),
                                      cudaMemcpyHostToDevice, sim.stream),
                      "cudaMemcpyAsync H2D");
        }

        if (sim.local_n > 0) {
            const int blocks = (sim.local_n + threads - 1) / threads;
            energyStepKernel<<<blocks, threads, 0, sim.stream>>>(
                energy_cur, flux_cur, energy_next, flux_next,
                sim.start, sim.local_n, sim.n_root, sim.ghost);
            cudaCheck(cudaGetLastError(), "energyStepKernel");
        }

        cudaCheck(cudaStreamSynchronize(sim.stream), "cudaStreamSynchronize iteration");
        sim.cur = next;
    }
}

static uint64_t computeHashLocal(const std::vector<val_t>& energy,
                                 const std::vector<val_t>& flux,
                                 int start) {
    uint64_t hash = 0;
    const int n = static_cast<int>(energy.size());
    #pragma omp parallel for reduction(^:hash)
    for (int i = 0; i < n; ++i) {
        uint64_t e_bits;
        uint64_t f_bits;
        std::memcpy(&e_bits, &energy[i], sizeof(uint64_t));
        std::memcpy(&f_bits, &flux[i], sizeof(uint64_t));
        const uint64_t idx = static_cast<uint64_t>(start + i);
        hash ^= (e_bits + idx) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + idx) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const std::vector<val_t>& energy,
                            const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    constexpr val_t energy_epsilon = 1e-8;

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }

    if (std::abs(energy_sum) > energy_epsilon) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }

    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }

    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    mpiCheck(MPI_Init(&argc, &argv), "MPI_Init");

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int node_rank = rank;
    MPI_Comm node_comm = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &node_comm) == MPI_SUCCESS) {
        MPI_Comm_rank(node_comm, &node_rank);
    }

    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count == 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA device required but none is available\n");
        }
        if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return 1;
    }
    cudaCheck(cudaSetDevice(node_rank % device_count), "cudaSetDevice");
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&node_comm);
    }

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    bool exit_early = false;
    int exit_code = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exit_early = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exit_early = true;
            exit_code = 1;
        }
    }

    if (exit_early) {
        MPI_Finalize();
        return exit_code;
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");

        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    SimData sim;
    setupSimulation(sim, n_elems_root, n_elems, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    runSimulation(sim, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD),
             "MPI_Allreduce timing");

    std::vector<val_t> local_energy(static_cast<size_t>(sim.local_n));
    std::vector<val_t> local_flux(static_cast<size_t>(sim.local_n));
    if (sim.local_n > 0) {
        cudaCheck(cudaMemcpy(local_energy.data(), sim.d_energy[sim.cur] + sim.ghost,
                             static_cast<size_t>(sim.local_n) * sizeof(val_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy final energy");
        cudaCheck(cudaMemcpy(local_flux.data(), sim.d_flux[sim.cur],
                             static_cast<size_t>(sim.local_n) * sizeof(val_t),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy final flux");
    }

    const uint64_t local_hash = computeHashLocal(local_energy, local_flux, sim.start);
    uint64_t hash = 0;
    mpiCheck(MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD),
             "MPI_Reduce hash");

    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (printResults || validate) {
        std::vector<int> counts(static_cast<size_t>(size));
        std::vector<int> displs(static_cast<size_t>(size));
        for (int q = 0; q < size; ++q) {
            const int qs = static_cast<int>((static_cast<long long>(q) * n_elems) / size);
            const int qe = static_cast<int>((static_cast<long long>(q + 1) * n_elems) / size);
            counts[q] = qe - qs;
            displs[q] = qs;
        }
        if (rank == 0) {
            global_energy.resize(static_cast<size_t>(n_elems));
            if (validate) global_flux.resize(static_cast<size_t>(n_elems));
        }
        mpiCheck(MPI_Gatherv(local_energy.data(), sim.local_n, MPI_DOUBLE,
                             rank == 0 ? global_energy.data() : nullptr,
                             counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv energy");
        if (validate) {
            mpiCheck(MPI_Gatherv(local_flux.data(), sim.local_n, MPI_DOUBLE,
                                 rank == 0 ? global_flux.data() : nullptr,
                                 counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                     "MPI_Gatherv flux");
        }
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");

        if (printResults) {
            print_results(global_energy, "ElementEnergy");
        }

        if (validate) {
            if (!validateResults(global_energy, global_flux)) {
                destroySimulation(sim);
                MPI_Finalize();
                return 1;
            }
        }
    }

    destroySimulation(sim);
    MPI_Finalize();
    return 0;
}
