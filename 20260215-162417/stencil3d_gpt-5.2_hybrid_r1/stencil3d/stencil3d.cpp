#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (global layout: x fastest, then y, then z)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        std::fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void stencil7_update_kernel(const Real* __restrict__ in,
                                      Real* __restrict__ out,
                                      int nx, int ny,
                                      int global_nz,
                                      int z_start_global,
                                      int local_z_begin,
                                      int local_z_end) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int lz = (int)(local_z_begin + (int)blockIdx.z); // local z index in [1..local_z_count]

    if (x >= nx || y >= ny || lz > local_z_end) return;

    const int gz = z_start_global + (lz - 1);
    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t idx = ((size_t)lz * (size_t)ny + (size_t)y) * (size_t)nx + (size_t)x;

    const Real center = in[idx];

    // Copy global or x/y boundaries unchanged (equivalent semantics).
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || gz == 0 || gz == global_nz - 1) {
        out[idx] = center;
        return;
    }

    const Real left = in[idx - 1];
    const Real right = in[idx + 1];
    const Real front = in[idx - (size_t)nx];
    const Real back = in[idx + (size_t)nx];
    const Real bottom = in[idx - plane];
    const Real top = in[idx + plane];

    out[idx] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
}

static void printUsage(const char* progName) {
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

static void decompose_z(int nz, int size, int rank, int& z_start, int& z_count) {
    const int base = nz / size;
    const int rem = nz % size;
    z_count = base + (rank < rem ? 1 : 0);
    z_start = rank * base + (rank < rem ? rank : rem);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks must see same argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = (size_t)atoi(argv[++i]);
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

    const int inx = (int)nx;
    const int iny = (int)ny;
    const int inz = (int)nz;

    if (inx < 2 || iny < 2 || inz < 2) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be >= 2\n");
        MPI_Finalize();
        return 1;
    }

    // Select CUDA device (simple mapping; assumes MPI ranks are bound reasonably).
    int devCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&devCount);
    if (devErr != cudaSuccess || devCount <= 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Finalize();
        return 1;
    }
    cudaCheck(cudaSetDevice(rank % devCount), "cudaSetDevice");

    // Z-slab decomposition across MPI ranks.
    int z_start = 0, z_count = 0;
    decompose_z(inz, size, rank, z_start, z_count);
    if (z_count <= 0) {
        if (rank == 0) std::fprintf(stderr, "MPI decomposition invalid: too many ranks for nz=%d\n", inz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int prev = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

    const size_t plane = nx * ny;
    const size_t local_planes_with_halo = (size_t)z_count + 2;
    const size_t local_elems_with_halo = local_planes_with_halo * plane;

    // Host initialization (owned planes only live in [1..z_count] in the halo layout).
    std::vector<Real> h_init(local_elems_with_halo, 0.0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int lz0 = 0; lz0 < z_count; ++lz0) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = (size_t)z_start + (size_t)lz0;
                const size_t gidx = idx3(x, y, gz, nx, ny);
                const size_t lidx = idx3(x, y, (size_t)(lz0 + 1), nx, ny);
                h_init[lidx] = (Real)((gidx % 19) * 1.0);
            }
        }
    }

    Real* d_a = nullptr;
    Real* d_b = nullptr;
    cudaCheck(cudaMalloc((void**)&d_a, local_elems_with_halo * sizeof(Real)), "cudaMalloc d_a");
    cudaCheck(cudaMalloc((void**)&d_b, local_elems_with_halo * sizeof(Real)), "cudaMalloc d_b");
    cudaCheck(cudaMemcpy(d_a, h_init.data(), local_elems_with_halo * sizeof(Real), cudaMemcpyHostToDevice), "H2D init");

    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_comm = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking), "cudaStreamCreate compute");
    cudaCheck(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking), "cudaStreamCreate comm");

    // Pinned host buffers for halo exchange (1 plane each direction).
    Real *h_send_prev = nullptr, *h_recv_prev = nullptr;
    Real *h_send_next = nullptr, *h_recv_next = nullptr;
    if (prev != MPI_PROC_NULL) {
        cudaCheck(cudaHostAlloc((void**)&h_send_prev, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc send_prev");
        cudaCheck(cudaHostAlloc((void**)&h_recv_prev, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc recv_prev");
    }
    if (next != MPI_PROC_NULL) {
        cudaCheck(cudaHostAlloc((void**)&h_send_next, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc send_next");
        cudaCheck(cudaHostAlloc((void**)&h_recv_next, plane * sizeof(Real), cudaHostAllocDefault), "cudaHostAlloc recv_next");
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads (max): %d\n", omp_get_max_threads());
        printf("CUDA devices visible: %d\n", devCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Run stencil iterations
    const int TAG_DOWN = 100; // from rank+1 to rank
    const int TAG_UP = 101;   // from rank-1 to rank

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_a : d_b;
        Real* d_out = (iter % 2 == 0) ? d_b : d_a;

        MPI_Request reqs[4];
        int nreq = 0;

        if (prev != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_prev, (int)plane, MPI_DOUBLE, prev, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (next != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_next, (int)plane, MPI_DOUBLE, next, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // Start D2H copies of boundary planes for sends.
        if (prev != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(h_send_prev, d_in + plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm), "D2H send_prev");
        }
        if (next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(h_send_next, d_in + (size_t)z_count * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm), "D2H send_next");
        }

        // Compute bulk planes that do not depend on MPI halos.
        const bool lower_dep = (prev != MPI_PROC_NULL) && (z_start > 0) && (z_start < inz - 1);
        const bool upper_dep = (next != MPI_PROC_NULL) && ((z_start + z_count - 1) < inz - 1) && ((z_start + z_count - 1) > 0);

        int bulk_begin = 1 + (lower_dep ? 1 : 0);
        int bulk_end = z_count - (upper_dep ? 1 : 0);

        if (bulk_begin <= bulk_end) {
            dim3 block(32, 4, 1);
            dim3 grid((inx + block.x - 1) / block.x,
                      (iny + block.y - 1) / block.y,
                      (unsigned)(bulk_end - bulk_begin + 1));
            stencil7_update_kernel<<<grid, block, 0, stream_compute>>>(
                d_in, d_out, inx, iny, inz, z_start, bulk_begin, bulk_end);
            cudaCheck(cudaGetLastError(), "kernel launch (bulk)");
        }

        // Ensure send buffers are ready, then post sends.
        cudaCheck(cudaStreamSynchronize(stream_comm), "sync comm before Isend");
        if (prev != MPI_PROC_NULL) {
            MPI_Isend(h_send_prev, (int)plane, MPI_DOUBLE, prev, TAG_UP, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (next != MPI_PROC_NULL) {
            MPI_Isend(h_send_next, (int)plane, MPI_DOUBLE, next, TAG_DOWN, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        if (nreq > 0) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Copy received halos back to device.
        if (prev != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(d_in + 0, h_recv_prev, plane * sizeof(Real), cudaMemcpyHostToDevice, stream_comm), "H2D recv_prev");
        }
        if (next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(d_in + (size_t)(z_count + 1) * plane, h_recv_next, plane * sizeof(Real), cudaMemcpyHostToDevice, stream_comm), "H2D recv_next");
        }
        cudaCheck(cudaStreamSynchronize(stream_comm), "sync comm after H2D halos");

        // Compute boundary planes that require halos.
        if (lower_dep || upper_dep) {
            dim3 block(32, 4, 1);

            if (lower_dep && upper_dep && z_count == 1) {
                dim3 grid((inx + block.x - 1) / block.x,
                          (iny + block.y - 1) / block.y,
                          1);
                stencil7_update_kernel<<<grid, block, 0, stream_compute>>>(
                    d_in, d_out, inx, iny, inz, z_start, 1, 1);
                cudaCheck(cudaGetLastError(), "kernel launch (both deps)");
            } else {
                if (lower_dep) {
                    dim3 grid((inx + block.x - 1) / block.x,
                              (iny + block.y - 1) / block.y,
                              1);
                    stencil7_update_kernel<<<grid, block, 0, stream_compute>>>(
                        d_in, d_out, inx, iny, inz, z_start, 1, 1);
                    cudaCheck(cudaGetLastError(), "kernel launch (lower dep)");
                }
                if (upper_dep) {
                    dim3 grid((inx + block.x - 1) / block.x,
                              (iny + block.y - 1) / block.y,
                              1);
                    stencil7_update_kernel<<<grid, block, 0, stream_compute>>>(
                        d_in, d_out, inx, iny, inz, z_start, z_count, z_count);
                    cudaCheck(cudaGetLastError(), "kernel launch (upper dep)");
                }
            }
        }

        cudaCheck(cudaStreamSynchronize(stream_compute), "sync compute end-iter");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Use max wall time across ranks for performance reporting.
    const double local_ms = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_ms);
        const double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * (double)iterations;
        const double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 only if needed.
    if (validate || printResults) {
        Real* d_final = ((iterations % 2) == 0) ? d_a : d_b;
        std::vector<Real> h_local((size_t)z_count * plane);
        cudaCheck(cudaMemcpy(h_local.data(), d_final + plane, (size_t)z_count * plane * sizeof(Real), cudaMemcpyDeviceToHost), "D2H final");

        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<Real> full;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                int zs = 0, zc = 0;
                decompose_z(inz, size, r, zs, zc);
                counts[r] = (int)((size_t)zc * plane);
                displs[r] = (int)((size_t)zs * plane);
            }
            full.resize((size_t)inz * plane);
        }

        MPI_Gatherv(h_local.data(), (int)h_local.size(), MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(full, "Grid");
            }

            if (validate) {
                // Same validation semantics as baseline: no NaN/Inf and reasonable bounds.
                for (const auto& val : full) {
                    if (std::isnan(val) || std::isinf(val)) {
                        printf("Validation failed: found NaN or Inf value\n");
                        MPI_Finalize();
                        return 1;
                    }
                }

                Real minVal = full[0];
                Real maxVal = full[0];
                for (const auto& val : full) {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }

                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    MPI_Finalize();
                    return 1;
                }

                printf("Validation: PASSED\n");
            }
        }
    }

    if (h_send_prev) cudaCheck(cudaFreeHost(h_send_prev), "cudaFreeHost send_prev");
    if (h_recv_prev) cudaCheck(cudaFreeHost(h_recv_prev), "cudaFreeHost recv_prev");
    if (h_send_next) cudaCheck(cudaFreeHost(h_send_next), "cudaFreeHost send_next");
    if (h_recv_next) cudaCheck(cudaFreeHost(h_recv_next), "cudaFreeHost recv_next");

    cudaCheck(cudaStreamDestroy(stream_compute), "cudaStreamDestroy compute");
    cudaCheck(cudaStreamDestroy(stream_comm), "cudaStreamDestroy comm");

    cudaCheck(cudaFree(d_a), "cudaFree d_a");
    cudaCheck(cudaFree(d_b), "cudaFree d_b");

    MPI_Finalize();
    return 0;
}
