#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void cudaCheck(cudaError_t e, const char* expr, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s (%d): %s at %s:%d\n", expr, (int)e, cudaGetErrorString(e), file, line);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

__global__ void stencil7_interior_xy(const Real* __restrict__ in,
                                    Real* __restrict__ out,
                                    int nx, int ny, int plane,
                                    int zStart, int zCount) {
    const int x = 1 + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int localZ = zStart + (int)blockIdx.z;

    if (x >= nx - 1 || y >= ny - 1 || (int)blockIdx.z >= zCount) return;

    const int idx = localZ * plane + y * nx + x;
    const Real center = in[idx];
    const Real left = in[idx - 1];
    const Real right = in[idx + 1];
    const Real front = in[idx - nx];
    const Real back = in[idx + nx];
    const Real bottom = in[idx - plane];
    const Real top = in[idx + plane];
    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

__global__ void copy_x_edges(const Real* __restrict__ in,
                            Real* __restrict__ out,
                            int nx, int ny, int plane,
                            int zStart, int zCount) {
    const int y = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int localZ = zStart + (int)blockIdx.y;
    if (y >= ny || (int)blockIdx.y >= zCount) return;

    const int base = localZ * plane + y * nx;
    out[base] = in[base];
    out[base + (nx - 1)] = in[base + (nx - 1)];
}

__global__ void copy_y_edges(const Real* __restrict__ in,
                            Real* __restrict__ out,
                            int nx, int ny, int plane,
                            int zStart, int zCount) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int localZ = zStart + (int)blockIdx.y;
    if (x >= nx || (int)blockIdx.y >= zCount) return;

    const int base0 = localZ * plane + x;
    const int base1 = localZ * plane + (ny - 1) * nx + x;
    out[base0] = in[base0];
    out[base1] = in[base1];
}

static void initializeLocalOwned(std::vector<Real>& localWithHalos, size_t nx, size_t ny, size_t local_nz, size_t z0_global) {
    const size_t plane = nx * ny;
    std::fill(localWithHalos.begin(), localWithHalos.end(), 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gz = z0_global + lz;
            const size_t zOff = (lz + 1) * plane;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = idx3(x, y, gz, nx, ny);
                localWithHalos[zOff + y * nx + x] = (gidx % 19) * 1.0;
            }
        }
    }
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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

    if (nx < 3 || ny < 3 || nz < 3) {
        if (rank == 0) {
            printf("Grid must be at least 3x3x3 for stencil.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Select GPU based on shared-memory (node-local) rank.
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmComm);
    int localRank = 0;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_free(&shmComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t plane = nx * ny;

    // Z-slab decomposition.
    const size_t base = nz / (size_t)size;
    const size_t rem = nz % (size_t)size;
    const size_t local_nz = base + ((size_t)rank < rem ? 1 : 0);
    const size_t z0 = (size_t)rank * base + std::min((size_t)rank, rem);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Local grids include 2 halo planes in Z.
    const size_t localWithHalosSize = (local_nz + 2) * plane;
    std::vector<Real> h_a(localWithHalosSize);
    std::vector<Real> h_b(localWithHalosSize);
    initializeLocalOwned(h_a, nx, ny, local_nz, z0);
    std::fill(h_b.begin(), h_b.end(), 0.0);

    Real* d_a = nullptr;
    Real* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, localWithHalosSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_b, localWithHalosSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), localWithHalosSize * sizeof(Real), cudaMemcpyHostToDevice));

    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    cudaEvent_t evt_out_ready;
    cudaEvent_t evt_halo_ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_out_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_halo_ready, cudaEventDisableTiming));
    // Initial input ready.
    CUDA_CHECK(cudaEventRecord(evt_out_ready, stream_compute));

    Real* h_send_down = nullptr;
    Real* h_send_up = nullptr;
    Real* h_recv_down = nullptr;
    Real* h_recv_up = nullptr;

    const int down = (local_nz > 0 && z0 > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (local_nz > 0 && (z0 + local_nz) < nz) ? rank + 1 : MPI_PROC_NULL;

    if (down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaHostAlloc(&h_send_down, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv_down, plane * sizeof(Real), cudaHostAllocDefault));
    }
    if (up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaHostAlloc(&h_send_up, plane * sizeof(Real), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_recv_up, plane * sizeof(Real), cudaHostAllocDefault));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in = (iter % 2 == 0) ? d_a : d_b;
        Real* d_out = (iter % 2 == 0) ? d_b : d_a;

        // Ensure d_in (previous iteration output) is visible to comm stream before D2H halo sends.
        CUDA_CHECK(cudaStreamWaitEvent(stream_comm, evt_out_ready, 0));

        MPI_Request reqs[4];
        int nreq = 0;

        if (down != MPI_PROC_NULL && local_nz > 0) {
            MPI_Irecv(h_recv_down, (int)plane, MPI_DOUBLE, down, 100, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (up != MPI_PROC_NULL && local_nz > 0) {
            MPI_Irecv(h_recv_up, (int)plane, MPI_DOUBLE, up, 101, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // Overlap interior compute with halo exchange.
        if (local_nz > 2) {
            const int zStart = 2;
            const int zEnd = (int)local_nz - 1;
            const int zCount = zEnd - zStart + 1;
            dim3 block(32, 8, 1);
            dim3 grid((unsigned int)(((nx - 2) + block.x - 1) / block.x),
                      (unsigned int)(((ny - 2) + block.y - 1) / block.y),
                      (unsigned int)zCount);
            stencil7_interior_xy<<<grid, block, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, zStart, zCount);

            dim3 bx(256, 1, 1);
            dim3 gx((unsigned int)((ny + bx.x - 1) / bx.x), (unsigned int)zCount, 1);
            copy_x_edges<<<gx, bx, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, zStart, zCount);

            dim3 by(256, 1, 1);
            dim3 gy((unsigned int)((nx + by.x - 1) / by.x), (unsigned int)zCount, 1);
            copy_y_edges<<<gy, by, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, zStart, zCount);
        }

        // Global Z-boundary planes are pure copy.
        if (local_nz > 0) {
            const size_t zLast = z0 + local_nz - 1;
            if (z0 == 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_out + plane * 1, d_in + plane * 1, plane * sizeof(Real), cudaMemcpyDeviceToDevice, stream_compute));
            }
            if (zLast == nz - 1) {
                CUDA_CHECK(cudaMemcpyAsync(d_out + plane * (size_t)local_nz, d_in + plane * (size_t)local_nz, plane * sizeof(Real), cudaMemcpyDeviceToDevice, stream_compute));
            }
        }

        // D2H boundary planes for MPI sends.
        if (local_nz > 0) {
            if (down != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_down, d_in + plane * 1, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));
            }
            if (up != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(h_send_up, d_in + plane * (size_t)local_nz, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_comm));

        if (down != MPI_PROC_NULL && local_nz > 0) {
            MPI_Isend(h_send_down, (int)plane, MPI_DOUBLE, down, 101, MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (up != MPI_PROC_NULL && local_nz > 0) {
            MPI_Isend(h_send_up, (int)plane, MPI_DOUBLE, up, 100, MPI_COMM_WORLD, &reqs[nreq++]);
        }

        if (nreq > 0) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // H2D halo planes.
        if (local_nz > 0) {
            if (down != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in + plane * 0, h_recv_down, plane * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
            }
            if (up != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(d_in + plane * (local_nz + 1), h_recv_up, plane * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
            }
        }

        CUDA_CHECK(cudaEventRecord(evt_halo_ready, stream_comm));
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, evt_halo_ready, 0));

        // Compute the boundary-owned planes (if they are not global Z boundaries).
        if (local_nz > 0) {
            const size_t zLast = z0 + local_nz - 1;
            auto launchPlane = [&](int localZ) {
                dim3 block(32, 8, 1);
                dim3 grid((unsigned int)(((nx - 2) + block.x - 1) / block.x),
                          (unsigned int)(((ny - 2) + block.y - 1) / block.y),
                          1u);
                stencil7_interior_xy<<<grid, block, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, localZ, 1);

                dim3 bx(256, 1, 1);
                dim3 gx((unsigned int)((ny + bx.x - 1) / bx.x), 1u, 1u);
                copy_x_edges<<<gx, bx, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, localZ, 1);

                dim3 by(256, 1, 1);
                dim3 gy((unsigned int)((nx + by.x - 1) / by.x), 1u, 1u);
                copy_y_edges<<<gy, by, 0, stream_compute>>>(d_in, d_out, (int)nx, (int)ny, (int)plane, localZ, 1);
            };

            const bool firstIsInterior = (z0 > 0 && z0 < nz - 1);
            const bool lastIsInterior = (zLast > 0 && zLast < nz - 1);

            if (firstIsInterior) {
                launchPlane(1);
            }
            if (local_nz > 1 && lastIsInterior) {
                launchPlane((int)local_nz);
            }
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evt_out_ready, stream_compute));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(end - start).count();
    double elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", elapsed_max);
        const double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * (double)iterations;
        const double mcups = cellUpdates / elapsed_max / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy final owned region back for optional validation/printing.
    const bool needHostFinal = validate || printResults;
    std::vector<Real> h_final_owned;
    if (needHostFinal && local_nz > 0) {
        h_final_owned.resize(local_nz * plane);
        Real* d_final = (iterations % 2 == 0) ? d_a : d_b;
        CUDA_CHECK(cudaMemcpy(h_final_owned.data(), d_final + plane * 1, local_nz * plane * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    if (validate) {
        // Distributed validation: NaN/Inf check + global min/max.
        int local_bad = 0;
        Real local_min = std::numeric_limits<Real>::infinity();
        Real local_max = -std::numeric_limits<Real>::infinity();

        for (size_t i = 0; i < h_final_owned.size(); ++i) {
            const Real v = h_final_owned[i];
            if (std::isnan(v) || std::isinf(v)) {
                local_bad = 1;
                break;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }

        int any_bad = 0;
        MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        Real gmin = 0.0, gmax = 0.0;
        MPI_Allreduce(&local_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            if (any_bad) {
                printf("Validation failed: found NaN or Inf value\n");
                printf("Validation: FAILED\n");
            } else {
                printf("Value range: [%.6f, %.6f]\n", gmin, gmax);
                if (gmax > 1e6 || gmin < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    printf("Validation: FAILED\n");
                } else {
                    printf("Validation: PASSED\n");
                }
            }
        }

        if (any_bad || gmax > 1e6 || gmin < -1e6) {
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_b));
            if (h_send_down) CUDA_CHECK(cudaFreeHost(h_send_down));
            if (h_send_up) CUDA_CHECK(cudaFreeHost(h_send_up));
            if (h_recv_down) CUDA_CHECK(cudaFreeHost(h_recv_down));
            if (h_recv_up) CUDA_CHECK(cudaFreeHost(h_recv_up));
            CUDA_CHECK(cudaEventDestroy(evt_out_ready));
            CUDA_CHECK(cudaEventDestroy(evt_halo_ready));
            CUDA_CHECK(cudaStreamDestroy(stream_compute));
            CUDA_CHECK(cudaStreamDestroy(stream_comm));
            MPI_Finalize();
            return 1;
        }
    }

    if (printResults) {
        // Gather full grid to rank 0 and print.
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
        }
        const int myCount = (int)(local_nz * plane);
        MPI_Gather(&myCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<Real> full;
        if (rank == 0) {
            displs.resize(size);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                displs[r] = offset;
                offset += counts[r];
            }
            full.resize(nx * ny * nz);
        }

        MPI_Gatherv(h_final_owned.data(), myCount, MPI_DOUBLE,
                    rank == 0 ? full.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(full, "Grid");
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    if (h_send_down) CUDA_CHECK(cudaFreeHost(h_send_down));
    if (h_send_up) CUDA_CHECK(cudaFreeHost(h_send_up));
    if (h_recv_down) CUDA_CHECK(cudaFreeHost(h_recv_down));
    if (h_recv_up) CUDA_CHECK(cudaFreeHost(h_recv_up));
    CUDA_CHECK(cudaEventDestroy(evt_out_ready));
    CUDA_CHECK(cudaEventDestroy(evt_halo_ready));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_comm));

    MPI_Finalize();
    return 0;
}
