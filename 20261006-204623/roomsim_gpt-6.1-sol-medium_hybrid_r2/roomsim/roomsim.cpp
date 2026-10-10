/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cstdarg>
#include <cfloat>

#include "../common/results_output.hpp"

// MPI calls stay on the main thread (MPI_THREAD_FUNNELED).
static int worldRank = 0, worldSize = 1;
static int rootPrintf(const char* format, ...) {
    if (worldRank != 0) return 0;
    va_list args;
    va_start(args, format);
    int result = vprintf(format, args);
    va_end(args);
    return result;
}
#define printf rootPrintf

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "Rank %d: MPI operation failed\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    void selectDevice() {
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                     worldRank, MPI_INFO_NULL, &local));
        int localRank, localSize, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        mpiCheck(MPI_Comm_size(local, &localSize));
        // Respect explicit OpenMP settings; otherwise share CPU cores among
        // local ranks and cap this memory-intensive sampling workload.
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(16, omp_get_num_procs() / localSize)));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA device is required\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaCheck(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&local));
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;
    void allocate(size_t n) {
        count = n;
        if (n) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    void upload(const T* src, size_t n) {
        if (n) cudaCheck(cudaMemcpy(data, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* dst, size_t n) const {
        if (n) cudaCheck(cudaMemcpy(dst, data, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }

};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(uint64_t n) { rng.discard(n); }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

struct FlatNode {
    Vec3 center, halfExtent;
    size_t first, count, escape;
};

static size_t flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                            std::vector<size_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(),
                     tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (int child = 0; child < 8; ++child)
        if (tree.children[child]) flattenOctree(*tree.children[child], nodes, indices);
    nodes[index].escape = nodes.size();
    return index;
}

__device__ bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y ||
        fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z - d.z*c.y) > h.y*ad.z + h.z*ad.y + EPSILON) return false;
    if (fabsf(d.z*c.x - d.x*c.z) > h.z*ad.x + h.x*ad.z + EPSILON) return false;
    if (fabsf(d.x*c.y - d.y*c.x) > h.x*ad.y + h.y*ad.x + EPSILON) return false;
    return true;
}

__device__ bool blocked(const Vec3& from, const Vec3& to, size_t i, size_t j,
                        const Triangle* tris, const FlatNode* nodes,
                        size_t nodeCount, const size_t* indices) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    Vec3 normalized = dir / length;
    // Preorder escape links avoid recursion and any fixed-size traversal stack.
    for (size_t k = 0; k < nodeCount;) {
        const FlatNode& node = nodes[k];
        // The original traversal does not test the root box.
        if (k && !intersectsBox(node, from, to)) { k = node.escape; continue; }
        for (size_t q = 0; q < node.count; ++q) {
            size_t idx = indices[node.first + q];
            if (idx == i || idx == j) continue;
            const Triangle& tri = tris[idx];
            val_t distance = rayTriangleIntersect(from, normalized, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__global__ void formFactorKernel(const Triangle* tris, size_t n, size_t first,
                                 size_t rows, size_t batchFirst, size_t batchRows,
                                 const val_t* random, const FlatNode* nodes,
                                 size_t nodeCount, const size_t* indices, val_t* kij) {
    size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= batchRows * n) return;
    size_t row = pair / n, j = pair % n, i = first + batchFirst + row;
    const Triangle& ti = tris[i];
    const Triangle& tj = tris[j];
    val_t sum = ZERO;
    if (i != j && ti.normal().dot(tj.normal()) <= 0.99f) {
        // Sample-major layout coalesces reads across adjacent pair threads.
        for (int r = 0; r < NUM_RAYS; ++r) {
            size_t stride = batchRows * n;
            Vec3 pi = sampledPoint(ti, random[(4*r)*stride + pair],
                                      random[(4*r+1)*stride + pair]);
            Vec3 pj = sampledPoint(tj, random[(4*r+2)*stride + pair],
                                      random[(4*r+3)*stride + pair]);
            if (blocked(pi, pj, i, j, tris, nodes, nodeCount, indices)) continue;
            Vec3 v = pj - pi;
            val_t d2 = v.squaredNorm();
            if (d2 < EPSILON) continue;
            val_t ci = cosPhi(v, ti.normal()), cj = cosPhi(-v, tj.normal());
            if (ci <= ZERO || cj <= ZERO) continue;
            sum += (ci * cj) / (PI * d2);
        }
    }
    // Column-major local rows coalesce propagation reads across receivers.
    kij[j * rows + batchFirst + row] = sum * INV_NUM_RAYS;
}

__global__ void countFactorsKernel(const val_t* kij, size_t pairs,
                                    unsigned long long* count) {
    __shared__ unsigned int counts[256];
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    counts[threadIdx.x] = index < pairs && kij[index] > EPSILON ? 1 : 0;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(count, static_cast<unsigned long long>(counts[0]));
}

__global__ void delayKernel(const Triangle* tris, size_t n, size_t first,
                            size_t rows, int* tau) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * n) return;
    size_t i = first + index % rows, j = index / rows;
    tau[index] = i == j ? 0 : computeTau(tris[i], tris[j]);
}

__global__ void propagationKernel(size_t n, size_t first, size_t rows, size_t t,
                                  size_t source, size_t timeOff, const val_t* kij,
                                  const int* tau, const val_t* areas,
                                  const val_t* rho, val_t* radiosity) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t i = first + row;
    val_t sum = ZERO;
    // Distinct triangles in this mesh have strictly positive delays, so rows
    // at a timestep are independent. Preserve the emitter accumulation order.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t index = j * rows + row;
        int delay = tau[index];
        if (t < static_cast<size_t>(delay)) continue;
        val_t k = kij[index];
        if (k <= ZERO) continue;
        val_t rad = radiosity[(t - delay) * n + j];
        if (rad <= ZERO) continue;
        sum += fminf(k * areas[j], ONE) * rad;
    }
    radiosity[t * n + i] = rho[i] * sum + (i == source && t < timeOff ? ONE : ZERO);
}

__global__ void correlationKernel(size_t n, size_t times, size_t first, size_t rows,
                                  size_t source, const val_t* radiosity, val_t* corr) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * times) return;
    size_t row = index % rows, lag = index / rows;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < times; ++tt)
        sum += radiosity[(tt-lag)*n + source] * radiosity[tt*n + first + row];
    corr[index] = sum;
}

__global__ void distanceKernel(size_t times, size_t rows, const val_t* corr, val_t* distances) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    val_t best = ZERO;
    size_t lag = 0;
    for (size_t t = 0; t < times; ++t) {
        val_t value = corr[t * rows + row];
        if (value > best) { best = value; lag = t; }
    }
    distances[row] = WAVE_SPEED * static_cast<val_t>(lag);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t firstRow, localRows;
    std::vector<int> counts, offsets;
    uint64_t globalNonZeroKij = 0;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<FlatNode> deviceNodes;
    DeviceBuffer<size_t> deviceIndices;
    DeviceBuffer<val_t> deviceKij, deviceAreas, deviceRho, deviceRad;
    DeviceBuffer<int> deviceTau;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Distribute receiver rows; only the O(N*T) history is replicated.
    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t begin = state.numTriangles * rank / worldSize;
        size_t end = state.numTriangles * (rank + 1) / worldSize;
        state.offsets[rank] = static_cast<int>(begin);
        state.counts[rank] = static_cast<int>(end - begin);
    }
    state.firstRow = state.offsets[worldRank];
    state.localRows = state.counts[worldRank];
    state.deviceKij.allocate(state.localRows * state.numTriangles);
    state.deviceTau.allocate(state.localRows * state.numTriangles);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    state.deviceTriangles.allocate(state.numTriangles);
    state.deviceTriangles.upload(state.triangles.data(), state.numTriangles);
    state.deviceNodes.allocate(nodes.size());
    state.deviceNodes.upload(nodes.data(), nodes.size());
    state.deviceIndices.allocate(indices.size());
    state.deviceIndices.upload(indices.data(), indices.size());
    state.deviceAreas.allocate(state.numTriangles);
    state.deviceAreas.upload(state.areas.data(), state.numTriangles);
    state.deviceRho.allocate(state.numTriangles);
    state.deviceRho.upload(state.rho.data(), state.numTriangles);
    state.deviceRad.allocate(state.radB.size());
    if (state.radB.size()) cudaCheck(cudaMemset(state.deviceRad.data, 0, state.radB.size()*sizeof(val_t)));

}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (MPI/OpenMP/CUDA)...\n");
    const size_t n = state.numTriangles, rows = state.localRows;
    // Snapshot the original mt19937 stream at row boundaries. Culling consumes
    // no random numbers in the original implementation, so offsets count only
    // surviving pairs. Copies allow OpenMP workers to reproduce exact samples.
    std::vector<uint64_t> draws(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = state.firstRow; i < state.firstRow + rows; ++i) {
        uint64_t active = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f)
                ++active;
        draws[i] = active * NUM_RAYS * 4;
    }
    mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_UINT64_T, draws.data(),
        state.counts.data(), state.offsets.data(), MPI_UINT64_T, MPI_COMM_WORLD));
    RandomGenerator stream(42);
    uint64_t prefix = 0;
    for (size_t i = 0; i < state.firstRow; ++i) prefix += draws[i];
    if (rows) stream.discard(prefix);
    // Bound staging memory, even for the largest supported mesh.
    const size_t batchCapacity = std::min(rows, std::max(size_t(1), size_t(262144) / n));
    if (rows) {
        val_t* samples = nullptr;
        const size_t capacity = batchCapacity * n * NUM_RAYS * 4;
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&samples), capacity*sizeof(val_t)));
        DeviceBuffer<val_t> deviceSamples;
        deviceSamples.allocate(capacity);
        for (size_t begin = 0; begin < rows; begin += batchCapacity) {
            size_t batch = std::min(batchCapacity, rows - begin);
            std::vector<RandomGenerator> generators;
            generators.reserve(batch);
            for (size_t row = 0; row < batch; ++row) {
                generators.push_back(stream);
                stream.discard(draws[state.firstRow + begin + row]);
            }
            #pragma omp parallel for schedule(static) num_threads(std::min(omp_get_max_threads(), static_cast<int>(batch)))
            for (size_t row = 0; row < batch; ++row) {
                size_t i = state.firstRow + begin + row;
                auto& rng = generators[row];
                for (size_t j = 0; j < n; ++j) {
                    if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f)
                        continue;
                    for (int k = 0; k < NUM_RAYS*4; ++k)
                        samples[size_t(k)*batch*n + row*n + j] = rng.rand();
                }
            }
            deviceSamples.upload(samples, batch*n*NUM_RAYS*4);
            formFactorKernel<<<(batch*n+127)/128, 128>>>(state.deviceTriangles.data,
                n, state.firstRow, rows, begin, batch, deviceSamples.data,
                state.deviceNodes.data, state.deviceNodes.count, state.deviceIndices.data,
                state.deviceKij.data);
            cudaCheck(cudaGetLastError());
            // The synchronous upload in the next batch orders reuse of staging.
        }
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaFreeHost(samples));
    }
    unsigned long long deviceCount = 0;
    DeviceBuffer<unsigned long long> count;
    count.allocate(1);
    count.upload(&deviceCount, 1);
    size_t pairs = rows * n;
    if (pairs) {
        countFactorsKernel<<<(pairs+255)/256, 256>>>(state.deviceKij.data, pairs, count.data);
        cudaCheck(cudaGetLastError());
    }
    count.download(&deviceCount, 1);
    uint64_t local = deviceCount;
    mpiCheck(MPI_Allreduce(&local, &state.globalNonZeroKij, 1, MPI_UINT64_T,
                          MPI_SUM, MPI_COMM_WORLD));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (CUDA)...\n");
    size_t pairs = state.localRows * state.numTriangles;
    if (pairs) {
        delayKernel<<<(pairs+255)/256, 256>>>(state.deviceTriangles.data,
            state.numTriangles, state.firstRow, state.localRows, state.deviceTau.data);
        cudaCheck(cudaGetLastError());
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation (MPI/CUDA)...\n");
    const size_t n = state.numTriangles, rows = state.localRows;
    val_t* exchange = nullptr;
    if (worldSize > 1)
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&exchange), n*sizeof(val_t)));
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (rows) {
            propagationKernel<<<(rows+127)/128, 128>>>(n, state.firstRow, rows, t,
                state.sourceIndex, state.numTimesteps/2, state.deviceKij.data,
                state.deviceTau.data, state.deviceAreas.data, state.deviceRho.data,
                state.deviceRad.data);
            cudaCheck(cudaGetLastError());
        }
        if (worldSize > 1) {
            // Only the newly computed time slice crosses the network. Pinned
            // host staging works with MPI libraries without CUDA awareness.
            if (rows) cudaCheck(cudaMemcpy(exchange + state.firstRow,
                state.deviceRad.data + t*n + state.firstRow, rows*sizeof(val_t),
                cudaMemcpyDeviceToHost));
            mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange,
                                   state.counts.data(), state.offsets.data(), MPI_FLOAT,
                                   MPI_COMM_WORLD));
            cudaCheck(cudaMemcpy(state.deviceRad.data + t*n, exchange, n*sizeof(val_t),
                                 cudaMemcpyHostToDevice));
        }
    }
    state.deviceRad.download(state.radB.data(), state.radB.size());
    if (exchange) cudaCheck(cudaFreeHost(exchange));
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation (MPI/CUDA)...\n");
    size_t rows = state.localRows, times = state.numTimesteps;
    DeviceBuffer<val_t> correlations, distances;
    correlations.allocate(rows*times);
    distances.allocate(rows);
    if (rows) {
        if (times) {
            correlationKernel<<<(rows*times+255)/256, 256>>>(state.numTriangles,
                times, state.firstRow, rows, state.sourceIndex, state.deviceRad.data,
                correlations.data);
            cudaCheck(cudaGetLastError());
        }
        distanceKernel<<<(rows+255)/256, 256>>>(times, rows, correlations.data, distances.data);
        cudaCheck(cudaGetLastError());
        distances.download(state.distances.data() + state.firstRow, rows);
    }
    mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(),
        state.counts.data(), state.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    uint64_t nonZeroKij = state.globalNonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0) {
        printf("Timesteps must be non-negative\n");
        return 1;
    }
    runtime.selectDevice();
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &preDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &simDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &distDuration, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", n);
    printf("  Timesteps: %zu\n", t);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Rank 0 matrix/history memory (host + device): %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int valid = 1;
    if (validate && worldRank == 0) valid = validateResults(state) ? 1 : 0;
    mpiCheck(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (!valid) return 1;

    return 0;
}
