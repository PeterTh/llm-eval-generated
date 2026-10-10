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
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// MPI calls remain on the main thread (MPI_THREAD_FUNNELED).
int worldRank = 0, worldSize = 1;
#define ROOT_PRINT(...) do { if (worldRank == 0) printf(__VA_ARGS__); } while (0)
void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void mpiCheck(int result) {
    if (result != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, result);
}
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    }
    void selectDevice() {
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                    MPI_INFO_NULL, &local));
        int localRank, localSize, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        mpiCheck(MPI_Comm_size(local, &localSize));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs()/localSize)));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        cudaCheck(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&local));
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t size = 0;
    void allocate(size_t n) {
        size = n;
        if (n) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    void upload(const T* p, size_t n) {
        if (n) cudaCheck(cudaMemcpy(data, p, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* p, size_t n) const {
        if (n) cudaCheck(cudaMemcpy(p, data, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
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
                Octree* child = children[i].get();
                #pragma omp task firstprivate(child, i, childMin, childMax) shared(childIndices) if(indices.size() > 128)
                child->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
    return cosine > 0.0f ? cosine : 0.0f;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Depth-first threaded octree: escape skips an entire subtree without a stack.
struct FlatNode {
    Vec3 center, halfExtent;
    int first, count, escape;
    __device__ bool intersects(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f, c = p1 + d - center;
        Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
        if (fabsf(c.x) > halfExtent.x + ad.x ||
            fabsf(c.y) > halfExtent.y + ad.y ||
            fabsf(c.z) > halfExtent.z + ad.z) return false;
        return !(fabsf(d.y*c.z-d.z*c.y) > halfExtent.y*ad.z+halfExtent.z*ad.y+EPSILON ||
                 fabsf(d.z*c.x-d.x*c.z) > halfExtent.z*ad.x+halfExtent.x*ad.z+EPSILON ||
                 fabsf(d.x*c.y-d.y*c.x) > halfExtent.x*ad.y+halfExtent.y*ad.x+EPSILON);
    }
};
void flattenTree(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<idx_t>& indices) {
    size_t node = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size()), 0});
    for (size_t index : tree.triangleIndices) indices.push_back(static_cast<idx_t>(index));
    for (const auto& child : tree.children) if (child) flattenTree(*child, nodes, indices);
    nodes[node].escape = static_cast<int>(nodes.size());
}
__device__ bool blocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                        const FlatNode* nodes, int nodeCount, const idx_t* indices,
                        size_t i, size_t j) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    for (int node = 0; node < nodeCount;) {
        const FlatNode& n = nodes[node];
        // The original traversal tests boxes only before entering children.
        if (node && !n.intersects(from, to)) { node = n.escape; continue; }
        for (int k = 0; k < n.count; ++k) {
            idx_t index = indices[n.first + k];
            if (index == i || index == j) continue;
            const Triangle& tri = triangles[index];
            val_t dist = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}
__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE-u; v = ONE-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}
__global__ void formKernel(const Triangle* triangles, const FlatNode* nodes, int nodeCount,
                           const idx_t* indices, const val_t* randoms, const size_t* offsets,
                           val_t* kij, int* tau, size_t n, size_t begin, size_t local,
                           size_t batchBegin, size_t batchRows) {
    size_t p = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p >= n*batchRows) return;
    size_t row = p/n, j = p%n, i = batchBegin+row;
    size_t output = j*local+i-begin;
    val_t result = ZERO;
    int delay = 0;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    if (i != j) {
        delay = static_cast<int>(ceilf((a.center()-b.center()).norm()*INV_WAVE_SPEED));
        size_t offset = offsets[p];
        if (offset != ~size_t(0)) {
            for (int r = 0; r < NUM_RAYS; ++r) {
                const val_t* values = randoms+offset+4*r;
                Vec3 pi = samplePoint(a, values[0], values[1]);
                Vec3 pj = samplePoint(b, values[2], values[3]);
                if (blocked(pi, pj, triangles, nodes, nodeCount, indices, i, j)) continue;
                Vec3 v = pj-pi;
                val_t d = v.squaredNorm();
                if (d < EPSILON) continue;
                val_t ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
                if (ci <= ZERO || cj <= ZERO) continue;
                result += ci*cj/(PI*d);
            }
            result *= INV_NUM_RAYS;
        }
    }
    kij[output] = result;
    tau[output] = delay;
}
__global__ void waveKernel(const val_t* kij, const int* tau, const val_t* areas,
                           val_t* history, size_t n, size_t local, size_t begin,
                           size_t t, size_t steps, size_t source, val_t rho) {
    size_t row = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= local) return;
    size_t i = begin+row;
    val_t sum = ZERO;
    // Keep the original increasing-j accumulation order and float arithmetic.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t p = j*local+row;
        int delay = tau[p];
        if (t < static_cast<size_t>(delay)) continue;
        val_t k = kij[p];
        if (k <= ZERO) continue;
        val_t rad = history[(t-delay)*n+j];
        if (rad <= ZERO) continue;
        sum += fminf(k*areas[j], ONE)*rad;
    }
    history[t*n+i] = rho*sum + ((i == source && t < steps/2) ? ONE : ZERO);
}
__global__ void correlationKernel(const val_t* history, val_t* correlations, size_t n,
                                  size_t steps, size_t begin, size_t local, size_t source) {
    size_t p = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p >= local*steps) return;
    size_t lag = p/local, i = begin+p%local;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < steps; ++tt)
        sum += history[(tt-lag)*n+source]*history[tt*n+i];
    correlations[p] = sum;
}
__global__ void distanceKernel(const val_t* correlations, val_t* distances,
                               size_t steps, size_t local) {
    size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= local) return;
    val_t best = ZERO;
    size_t lag = 0;
    for (size_t t = 0; t < steps; ++t) {
        val_t value = correlations[t*local+i];
        if (value > best) { best = value; lag = t; }
    }
    distances[i] = WAVE_SPEED*static_cast<val_t>(lag);
}

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t rowBegin, localRows;
    unsigned long long nonZeroFactors = 0;
    int exchangeSteps = 1;
    val_t reflectivity;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<FlatNode> deviceNodes;
    DeviceBuffer<idx_t> deviceIndices;
    DeviceBuffer<val_t> deviceKij, deviceAreas, deviceHistory;
    DeviceBuffer<int> deviceTau;
    std::vector<int> counts, displacements;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

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

    ROOT_PRINT("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    ROOT_PRINT("Building octree...\n");
    #pragma omp parallel
    {
        #pragma omp single
        state.octree.build(state.triangles);
    }
    state.rowBegin = state.numTriangles * worldRank / worldSize;
    state.localRows = state.numTriangles * (worldRank + 1) / worldSize - state.rowBegin;
    state.reflectivity = reflectivity;
    state.counts.resize(worldSize);
    state.displacements.resize(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        state.displacements[r] = static_cast<int>(state.numTriangles*r/worldSize);
        state.counts[r] = static_cast<int>(state.numTriangles*(r+1)/worldSize)-state.displacements[r];
    }
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flattenTree(state.octree, nodes, indices);
    state.deviceTriangles.allocate(state.numTriangles);
    state.deviceTriangles.upload(state.triangles.data(), state.numTriangles);
    state.deviceNodes.allocate(nodes.size());
    state.deviceNodes.upload(nodes.data(), nodes.size());
    state.deviceIndices.allocate(indices.size());
    state.deviceIndices.upload(indices.data(), indices.size());

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Quadratic geometry storage is distributed; only time history is replicated.
    state.deviceKij.allocate(state.localRows * state.numTriangles);
    state.deviceTau.allocate(state.localRows * state.numTriangles);
    state.deviceAreas.allocate(state.numTriangles);
    state.deviceAreas.upload(state.areas.data(), state.numTriangles);
    state.deviceHistory.allocate(timesteps * state.numTriangles);
    if (state.deviceHistory.size)
        cudaCheck(cudaMemset(state.deviceHistory.data, 0, state.deviceHistory.size*sizeof(val_t)));
    state.distances.resize(state.numTriangles, ZERO);

}

// ============================================================================
// Precomputation Phase
// ============================================================================

// The standard MT19937 stream, with blockwise discard that avoids tempering
// samples preceding this rank or the next OpenMP row checkpoint.
class SampleGenerator {
    std::array<uint32_t, 624> words;
    size_t position = 624;
    static uint32_t mix(uint32_t a, uint32_t b) {
        uint32_t value = (a & 0x80000000U) | (b & 0x7fffffffU);
        return (value >> 1) ^ ((value & 1) ? 0x9908b0dfU : 0);
    }
    void twist() {
        for (size_t i = 0; i < 227; ++i)
            words[i] = words[i+397] ^ mix(words[i], words[i+1]);
        for (size_t i = 227; i < 623; ++i)
            words[i] = words[i-227] ^ mix(words[i], words[i+1]);
        words[623] = words[396] ^ mix(words[623], words[0]);
        position = 0;
    }

public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }
    explicit SampleGenerator(uint32_t seed = 42) {
        words[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253U*(words[i-1] ^ (words[i-1] >> 30))+i;
    }
    result_type operator()() {
        if (position == 624) twist();
        uint32_t value = words[position++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680U;
        value ^= (value << 15) & 0xefc60000U;
        return value ^ (value >> 18);
    }
    void discard(unsigned long long count) {
        while (count) {
            if (position == 624) twist();
            size_t step = static_cast<size_t>(std::min(count, static_cast<unsigned long long>(624-position)));
            position += step;
            count -= step;
        }
    }
};

void computeFormFactors(SimulationState& state) {
    ROOT_PRINT("Computing distributed form factors and time delays...\n");
    const size_t n = state.numTriangles;
    // Each row starts at exactly the same mt19937 position as in the serial code.
    // Culled and diagonal pairs consume no random numbers.
    std::vector<size_t> draws(n, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f)
                draws[i] += 4*NUM_RAYS;
    SampleGenerator generator(42);
    unsigned long long skip = 0;
    for (size_t i = 0; i < state.rowBegin; ++i) skip += draws[i];
    generator.discard(skip);
    // Bound temporary sample storage independently of the local matrix size.
    const size_t batchSize = std::max(size_t(1), std::min(size_t(16), size_t(1048576)/(n*4*NUM_RAYS)));
    DeviceBuffer<val_t> samples;
    DeviceBuffer<size_t> offsets;
    samples.allocate(batchSize*n*4*NUM_RAYS);
    offsets.allocate(batchSize*n);
    std::vector<val_t> hostSamples(samples.size);
    std::vector<size_t> hostOffsets(offsets.size);
    for (size_t first = state.rowBegin; first < state.rowBegin+state.localRows; first += batchSize) {
        size_t rows = std::min(batchSize, state.rowBegin+state.localRows-first);
        std::vector<SampleGenerator> rowGenerators(rows);
        std::vector<size_t> rowOffsets(rows+1, 0);
        for (size_t r = 0; r < rows; ++r) {
            rowGenerators[r] = generator;
            generator.discard(draws[first+r]);
            rowOffsets[r+1] = rowOffsets[r]+draws[first+r];
        }
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            std::uniform_real_distribution<val_t> distribution(0.0f, 1.0f);
            auto& rng = rowGenerators[r];
            size_t offset = rowOffsets[r], i = first+r;
            for (size_t j = 0; j < n; ++j) {
                bool active = i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f;
                hostOffsets[r*n+j] = active ? offset : ~size_t(0);
                if (active) {
                    for (int d = 0; d < 4*NUM_RAYS; ++d) hostSamples[offset++] = distribution(rng);
                }
            }
        }
        samples.upload(hostSamples.data(), rowOffsets[rows]);
        offsets.upload(hostOffsets.data(), rows*n);
        formKernel<<<(rows*n+255)/256, 256>>>(state.deviceTriangles.data,
            state.deviceNodes.data, static_cast<int>(state.deviceNodes.size), state.deviceIndices.data,
            samples.data, offsets.data, state.deviceKij.data, state.deviceTau.data,
            n, state.rowBegin, state.localRows, first, rows);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    // Positive delays permit multiple timesteps between MPI exchanges.
    int minimum = INT_MAX;
    #pragma omp parallel for reduction(min:minimum) schedule(static)
    for (size_t i = state.rowBegin; i < state.rowBegin+state.localRows; ++i)
        for (size_t j = 0; j < state.numTriangles; ++j)
            if (i != j) minimum = std::min(minimum, computeTau(state.triangles[i], state.triangles[j]));
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (minimum <= 0) {
        fprintf(stderr, "Nonpositive inter-triangle delay cannot be propagated in parallel\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    state.exchangeSteps = std::min(minimum, 16);
}

void runSimulation(SimulationState& state) {
    ROOT_PRINT("Running distributed CUDA wave propagation...\n");
    const size_t n = state.numTriangles, local = state.localRows;
    // With a single rank the complete history already resides on this GPU.
    if (worldSize == 1) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            waveKernel<<<(local+255)/256, 256>>>(state.deviceKij.data, state.deviceTau.data,
                state.deviceAreas.data, state.deviceHistory.data, n, local, state.rowBegin,
                t, state.numTimesteps, state.sourceIndex, state.reflectivity);
            cudaCheck(cudaGetLastError());
        }
        cudaCheck(cudaDeviceSynchronize());
        return;
    }
    const size_t capacity = static_cast<size_t>(state.exchangeSteps)*n;
    val_t *send, *receive, *history;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&send), std::max(size_t(1), capacity)*sizeof(val_t)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&receive), std::max(size_t(1), capacity)*sizeof(val_t)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&history), std::max(size_t(1), capacity)*sizeof(val_t)));
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (size_t first = 0; first < state.numTimesteps; first += state.exchangeSteps) {
        size_t batch = std::min(static_cast<size_t>(state.exchangeSteps), state.numTimesteps-first);
        for (size_t t = first; t < first+batch; ++t) {
            if (local) {
                waveKernel<<<(local+255)/256, 256>>>(state.deviceKij.data, state.deviceTau.data,
                    state.deviceAreas.data, state.deviceHistory.data, n, local, state.rowBegin,
                    t, state.numTimesteps, state.sourceIndex, state.reflectivity);
                cudaCheck(cudaGetLastError());
            }
        }
        if (local) cudaCheck(cudaMemcpy2D(send, local*sizeof(val_t),
            state.deviceHistory.data+first*n+state.rowBegin, n*sizeof(val_t),
            local*sizeof(val_t), batch, cudaMemcpyDeviceToHost));
        for (int r = 0; r < worldSize; ++r) {
            counts[r] = static_cast<int>(batch)*state.counts[r];
            displacements[r] = static_cast<int>(batch)*state.displacements[r];
        }
        mpiCheck(MPI_Allgatherv(send, static_cast<int>(batch*local), MPI_FLOAT,
                               receive, counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < worldSize; ++r)
            for (size_t t = 0; t < batch; ++t)
                std::copy_n(receive+displacements[r]+t*state.counts[r], state.counts[r],
                            history+t*n+state.displacements[r]);
        cudaCheck(cudaMemcpy(state.deviceHistory.data+first*n, history,
                             batch*n*sizeof(val_t), cudaMemcpyHostToDevice));
    }
    cudaCheck(cudaFreeHost(send));
    cudaCheck(cudaFreeHost(receive));
    cudaCheck(cudaFreeHost(history));
}

void computeDistances(SimulationState& state) {
    ROOT_PRINT("Computing CUDA cross-correlations...\n");
    const size_t local = state.localRows;
    DeviceBuffer<val_t> correlations, distances;
    correlations.allocate(local*state.numTimesteps);
    distances.allocate(local);
    if (local) {
        if (state.numTimesteps) {
            correlationKernel<<<(correlations.size+255)/256, 256>>>(state.deviceHistory.data,
                correlations.data, state.numTriangles, state.numTimesteps,
                state.rowBegin, local, state.sourceIndex);
            cudaCheck(cudaGetLastError());
        }
        distanceKernel<<<(local+255)/256, 256>>>(correlations.data, distances.data, state.numTimesteps, local);
        cudaCheck(cudaGetLastError());
    }
    std::vector<val_t> localDistances(local);
    distances.download(localDistances.data(), local);
    mpiCheck(MPI_Gatherv(localDistances.data(), static_cast<int>(local), MPI_FLOAT,
        state.distances.data(), state.counts.data(), state.displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
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

    // Check Kij matrix (globally reduced count from distributed storage)
    unsigned long long nonZeroKij = state.nonZeroFactors;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij,
           state.numTriangles*state.numTriangles,
           100.0*nonZeroKij/(static_cast<double>(state.numTriangles)*state.numTriangles));

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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            ROOT_PRINT("Unknown option: %s\n", argv[i]);
            if (worldRank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0) {
        if (worldRank == 0) fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
    }
    runtime.selectDevice();
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINT("Room Response Simulation Benchmark\n");
    ROOT_PRINT("===================================\n");
    ROOT_PRINT("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINT("Timesteps: %d\n", timesteps);
    ROOT_PRINT("Source triangle: %d\n", sourceIdx);
    ROOT_PRINT("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINT("MPI ranks: %d, OpenMP threads per rank: %d\n", worldSize, omp_get_max_threads());
    ROOT_PRINT("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINT("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINT("\n");

    // Precomputation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &preDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));

    ROOT_PRINT("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINT("\n");

    // Simulation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &simDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));

    ROOT_PRINT("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINT("\n");

    // Distance computation
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &distDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));

    ROOT_PRINT("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINT("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    ROOT_PRINT("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINT("\nPerformance:\n");
    ROOT_PRINT("  Triangles: %zu\n", n);
    ROOT_PRINT("  Timesteps: %zu\n", t);
    ROOT_PRINT("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINT("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINT("  Distance computations: %.2e\n", distOps);
    ROOT_PRINT("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINT("  Matrix/history memory per rank (rank 0): %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINT("  Result hash: %016lX\n", hash);
    ROOT_PRINT("\n");
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Collect validation data only when requested, outside timed computation.
    int success = 1;
    if (validate) {
        std::vector<val_t> localKij(state.deviceKij.size);
        state.deviceKij.download(localKij.data(), localKij.size());
        unsigned long long nonzero = 0;
        #pragma omp parallel for reduction(+:nonzero) schedule(static)
        for (size_t i = 0; i < localKij.size(); ++i) if (localKij[i] > EPSILON) ++nonzero;
        mpiCheck(MPI_Reduce(&nonzero, &state.nonZeroFactors, 1, MPI_UNSIGNED_LONG_LONG,
                            MPI_SUM, 0, MPI_COMM_WORLD));
        if (worldRank == 0) {
            state.radB.resize(state.numTimesteps*state.numTriangles);
            state.deviceHistory.download(state.radB.data(), state.radB.size());
            success = validateResults(state) ? 1 : 0;
        }
        mpiCheck(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    if (!success) return 1;

    return 0;
}
