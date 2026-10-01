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

int worldRank = 0, worldSize = 1;

void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// MPI calls are confined to the main thread (MPI_THREAD_FUNNELED).
struct Runtime {
    Runtime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    void selectDevice() {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "A CUDA device is required on every rank.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaCheck(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
    }
    ~Runtime() { MPI_Finalize(); }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t n) { if (n) cudaCheck(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

#include "../common/results_output.hpp"

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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(uint64_t count) { rng.discard(count); }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

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
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// A preorder octree with escape indices permits stackless device traversal.
struct FlatNode {
    Vec3 center, halfExtent;
    int escape, begin, count;
};

void flattenTree(const Octree& tree, std::vector<FlatNode>& nodes,
                 std::vector<idx_t>& indices) {
    int index = static_cast<int>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<int>(indices.size()), static_cast<int>(tree.triangleIndices.size())});
    for (size_t t : tree.triangleIndices) indices.push_back(static_cast<idx_t>(t));
    for (const auto& child : tree.children) if (child) flattenTree(*child, nodes, indices);
    nodes[index].escape = static_cast<int>(nodes.size());
}

__device__ bool intersectsBox(const FlatNode& node, Vec3 p1, Vec3 p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(Vec3 from, Vec3 to, const Triangle* triangles,
                        const FlatNode* nodes, const idx_t* indices, size_t i, size_t j) {
    Vec3 dir = to - from;
    float len = dir.norm();
    if (len < EPSILON) return true;
    dir = dir / len;
    int node = 0;
    while (node < nodes[0].escape) {
        FlatNode current = nodes[node];
        if (node != 0 && !intersectsBox(current, from, to)) {
            node = current.escape;
            continue;
        }
        for (int k = 0; k < current.count; ++k) {
            idx_t id = indices[current.begin + k];
            if (id == i || id == j) continue;
            const Triangle& tri = triangles[id];
            float d = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < len - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

// Each lane computes one pair, retaining the original order of its 16 samples.
__global__ void formKernel(const Triangle* triangles, const FlatNode* nodes,
                           const idx_t* indices, const float* randoms, float* kij, int* tau,
                           size_t n, size_t first, size_t local, size_t batchFirst, size_t rows) {
    size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= rows * n) return;
    size_t row = pair / n, j = pair % n, i = first + batchFirst + row;
    size_t out = j * local + batchFirst + row;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    tau[out] = i == j ? 0 : computeTau(a, b);
    float sum = 0;
    if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
        // Sample-major layout coalesces reads across adjacent triangle pairs.
        for (int r = 0; r < NUM_RAYS; ++r) {
            Vec3 p = sampledPoint(a, randoms[(4*r)*rows*n+pair], randoms[(4*r+1)*rows*n+pair]);
            Vec3 q = sampledPoint(b, randoms[(4*r+2)*rows*n+pair], randoms[(4*r+3)*rows*n+pair]);
            if (blocked(p, q, triangles, nodes, indices, i, j)) continue;
            Vec3 v = q - p;
            float d = v.squaredNorm();
            if (d < EPSILON) continue;
            float ca = cosPhi(v, a.normal()), cb = cosPhi(-v, b.normal());
            if (ca <= ZERO || cb <= ZERO) continue;
            sum += (ca * cb) / (PI * d);
        }
    }
    kij[out] = sum * INV_NUM_RAYS;
}

// Adjacent lanes own adjacent receivers. The transposed local matrices provide
// coalesced reads while accumulation over emitters stays in serial order.
__global__ void waveKernel(const float* kij, const int* tau, const float* areas,
                           float* history, float rho, size_t n, size_t first, size_t local,
                           size_t t, size_t source, size_t timeOff) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= local) return;
    size_t i = first + row;
    float sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t p = j * local + row;
        int delay = tau[p];
        if (t < static_cast<size_t>(delay) || kij[p] <= ZERO) continue;
        float rad = history[(t - delay) * n + j];
        if (rad <= ZERO) continue;
        sum += fminf(kij[p] * areas[j], ONE) * rad;
    }
    history[t*n+i] = rho * sum + ((i == source && t < timeOff) ? ONE : ZERO);
}

__global__ void correlationKernel(const float* history, float* correlations, size_t n,
                                  size_t first, size_t local, size_t steps, size_t source) {
    size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= local * steps) return;
    size_t row = pair % local, t = pair / local;
    float sum = ZERO;
    for (size_t tt = t; tt < steps; ++tt)
        sum += history[(tt-t)*n+source] * history[tt*n+first+row];
    correlations[pair] = sum;
}

__global__ void distanceKernel(const float* correlations, float* distances,
                               size_t local, size_t steps) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= local) return;
    float best = ZERO;
    size_t bestT = 0;
    for (size_t t = 0; t < steps; ++t) {
        float sum = correlations[t*local+row];
        if (sum > best) { best = sum; bestT = t; }
    }
    distances[row] = WAVE_SPEED * static_cast<float>(bestT);
}

__global__ void countFactors(const float* factors, size_t count, unsigned long long* total) {
    unsigned int sum = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < count; i += static_cast<size_t>(gridDim.x)*blockDim.x)
        sum += factors[i] > EPSILON;
    __shared__ unsigned int partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x/2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, static_cast<unsigned long long>(partial[0]));
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
    size_t first, local;
    std::vector<int> counts, offsets;
    DeviceBuffer<float> deviceKij, deviceAreas, deviceHistory;
    DeviceBuffer<int> deviceTau;
    unsigned long long nonZeroKij = 0;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
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

    if (worldRank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (worldRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t lo = state.numTriangles * rank / worldSize;
        size_t hi = state.numTriangles * (rank + 1) / worldSize;
        state.counts[rank] = static_cast<int>(hi - lo);
        state.offsets[rank] = static_cast<int>(lo);
    }
    state.first = state.offsets[worldRank];
    state.local = state.counts[worldRank];
    size_t pairs = state.local * state.numTriangles;
    state.deviceKij.allocate(pairs);
    state.deviceTau.allocate(pairs);
    state.deviceAreas.allocate(state.numTriangles);
    cudaCheck(cudaMemcpy(state.deviceAreas.data, state.areas.data(),
                         state.numTriangles * sizeof(float), cudaMemcpyHostToDevice));
    state.deviceHistory.allocate(timesteps * state.numTriangles);
    if (timesteps) cudaCheck(cudaMemset(state.deviceHistory.data, 0, timesteps * state.numTriangles * sizeof(float)));
    state.distances.resize(state.numTriangles, ZERO);

}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (worldRank == 0) printf("Computing distributed form factors and time delays...\n");
    const size_t n = state.numTriangles;
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flattenTree(state.octree, nodes, indices);
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<FlatNode> deviceNodes;
    DeviceBuffer<idx_t> deviceIndices;
    triangles.allocate(n);
    deviceNodes.allocate(nodes.size());
    deviceIndices.allocate(indices.size());
    cudaCheck(cudaMemcpy(triangles.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceIndices.data, indices.data(), indices.size()*sizeof(idx_t), cudaMemcpyHostToDevice));

    // Count only pairs that consumed random numbers in the sequential program.
    std::vector<uint64_t> draws(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        uint64_t active = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) ++active;
        draws[i] = active * (4 * NUM_RAYS);
    }
    RandomGenerator rng(42);
    uint64_t skip = 0;
    for (size_t i = 0; i < state.first; ++i) skip += draws[i];
    rng.discard(skip);

    // Double-buffer bounded pinned batches to overlap OpenMP sampling and GPU
    // visibility work. Matrix rows are disjoint between the two CUDA streams.
    size_t batchSize = std::min(state.local, std::max(size_t(1), (32*1024*1024) / (n*4*NUM_RAYS*sizeof(float))));
    size_t sampleCount = batchSize*n*4*NUM_RAYS;
    float* hostSamples[2] = {nullptr, nullptr};
    DeviceBuffer<float> deviceSamples[2];
    cudaStream_t streams[2];
    for (int slot = 0; slot < 2; ++slot) {
        cudaCheck(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
        if (sampleCount) cudaCheck(cudaMallocHost(&hostSamples[slot], sampleCount*sizeof(float)));
        deviceSamples[slot].allocate(sampleCount);
    }
    for (size_t begin = 0, batch = 0; begin < state.local; begin += batchSize, ++batch) {
        int slot = batch % 2;
        cudaCheck(cudaStreamSynchronize(streams[slot]));
        float* samples = hostSamples[slot];
        size_t rows = std::min(batchSize, state.local - begin);
        std::vector<RandomGenerator> rowRng;
        rowRng.reserve(rows);
        for (size_t r = 0; r < rows; ++r) {
            rowRng.push_back(rng);
            rng.discard(draws[state.first + begin + r]);
        }
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; ++r) {
            size_t i = state.first + begin + r;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                for (int k = 0; k < 4*NUM_RAYS; ++k)
                    samples[k*rows*n+r*n+j] = rowRng[r].rand();
            }
        }
        cudaCheck(cudaMemcpyAsync(deviceSamples[slot].data, samples, rows*n*4*NUM_RAYS*sizeof(float),
                                  cudaMemcpyHostToDevice, streams[slot]));
        formKernel<<<(rows*n+127)/128,128,0,streams[slot]>>>(triangles.data, deviceNodes.data, deviceIndices.data,
            deviceSamples[slot].data, state.deviceKij.data, state.deviceTau.data,
            n, state.first, state.local, begin, rows);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    for (int slot = 0; slot < 2; ++slot) {
        if (hostSamples[slot]) cudaCheck(cudaFreeHost(hostSamples[slot]));
        cudaCheck(cudaStreamDestroy(streams[slot]));
    }
    // Only counts are needed by validation; never gather the quadratic matrix.
    unsigned long long localCount = 0;
    DeviceBuffer<unsigned long long> deviceCount;
    deviceCount.allocate(1);
    cudaCheck(cudaMemset(deviceCount.data, 0, sizeof(localCount)));
    if (state.local) {
        countFactors<<<256,256>>>(state.deviceKij.data, n*state.local, deviceCount.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(&localCount, deviceCount.data, sizeof(localCount), cudaMemcpyDeviceToHost));
    }
    MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

void runSimulation(SimulationState& state) {
    if (worldRank == 0) printf("Running distributed wave propagation...\n");
    const size_t n = state.numTriangles;
    // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
    float* row;
    cudaCheck(cudaMallocHost(&row, n*sizeof(float)));
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.local) {
            waveKernel<<<(state.local+127)/128,128>>>(state.deviceKij.data, state.deviceTau.data,
                state.deviceAreas.data, state.deviceHistory.data, state.rho[0], n,
                state.first, state.local, t, state.sourceIndex, state.numTimesteps/2);
            cudaCheck(cudaGetLastError());
        }
        if (worldSize > 1) {
            if (state.local) cudaCheck(cudaMemcpy(row+state.first,
                state.deviceHistory.data+t*n+state.first, state.local*sizeof(float), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, row, state.counts.data(), state.offsets.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            cudaCheck(cudaMemcpy(state.deviceHistory.data+t*n, row, n*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFreeHost(row));
}

void computeDistances(SimulationState& state) {
    if (worldRank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    DeviceBuffer<float> distances, correlations;
    distances.allocate(state.local);
    correlations.allocate(state.local * state.numTimesteps);
    if (state.local) {
        if (state.numTimesteps) {
            correlationKernel<<<(state.local*state.numTimesteps+127)/128,128>>>(state.deviceHistory.data,
                correlations.data, state.numTriangles, state.first, state.local, state.numTimesteps, state.sourceIndex);
            cudaCheck(cudaGetLastError());
        }
        distanceKernel<<<(state.local+127)/128,128>>>(correlations.data, distances.data,
            state.local, state.numTimesteps);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(state.distances.data()+state.first, distances.data,
                             state.local*sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (worldRank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.counts.data(),
                    state.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(state.distances.data()+state.first, static_cast<int>(state.local), MPI_FLOAT,
                    nullptr, nullptr, nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    if (worldRank == 0) printf("\nValidation:\n");

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
            if (worldRank == 0) printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            if (worldRank == 0) printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    if (worldRank == 0) printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    if (worldRank == 0) printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    if (worldRank == 0) printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        if (worldRank == 0) printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    if (worldRank == 0) printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        if (worldRank == 0) printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
    if (worldRank == 0) printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        if (worldRank == 0) printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    if (worldRank == 0) printf("  Validation: PASSED\n");
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
    if (worldRank == 0) printf("Usage: %s [options]\n", progName);
    if (worldRank == 0) printf("Options:\n");
    if (worldRank == 0) printf("  -n <num>     Target number of triangles (default: 320)\n");
    if (worldRank == 0) printf("               Actual count will be rounded to nearest icosphere level:\n");
    if (worldRank == 0) printf("               20, 80, 320, 1280, 5120, 20480\n");
    if (worldRank == 0) printf("  -t <num>     Number of timesteps (default: 50)\n");
    if (worldRank == 0) printf("  -s <num>     Source triangle index (default: 0)\n");
    if (worldRank == 0) printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    if (worldRank == 0) printf("  -v           Enable validation\n");
    if (worldRank == 0) printf("  -o           Print results for external validation\n");
    if (worldRank == 0) printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    Runtime runtime(argc, argv);
    try {
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
                if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
                if (worldRank == 0) printUsage(argv[0]);
                return 1;
            }
        }

        if (timesteps < 0) {
            if (worldRank == 0) fprintf(stderr, "Timesteps must be non-negative.\n");
            return 1;
        }
        runtime.selectDevice();
        int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

        if (worldRank == 0) printf("Room Response Simulation Benchmark\n");
        if (worldRank == 0) printf("===================================\n");
        if (worldRank == 0) printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        if (worldRank == 0) printf("Timesteps: %d\n", timesteps);
        if (worldRank == 0) printf("Source triangle: %d\n", sourceIdx);
        if (worldRank == 0) printf("Reflectivity: %.2f\n", reflectivity);
        if (worldRank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
        if (worldRank == 0) printf("\n");

        // Initialize
        SimulationState state;
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity);

        if (worldRank == 0) printf("\n");

        // Precomputation
        MPI_Barrier(MPI_COMM_WORLD);
        auto startPre = std::chrono::high_resolution_clock::now();

        computeFormFactors(state);

        auto endPre = std::chrono::high_resolution_clock::now();
        auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

        if (worldRank == 0) printf("Precomputation time: %ld ms\n", preDuration);
        if (worldRank == 0) printf("\n");

        // Simulation
        auto startSim = std::chrono::high_resolution_clock::now();

        runSimulation(state);

        MPI_Barrier(MPI_COMM_WORLD);
        auto endSim = std::chrono::high_resolution_clock::now();
        auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

        if (worldRank == 0) printf("Simulation time: %ld ms\n", simDuration);
        if (worldRank == 0) printf("\n");

        // Distance computation
        auto startDist = std::chrono::high_resolution_clock::now();

        computeDistances(state);

        MPI_Barrier(MPI_COMM_WORLD);
        auto endDist = std::chrono::high_resolution_clock::now();
        auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

        if (worldRank == 0) printf("Distance computation time: %ld ms\n", distDuration);
        if (worldRank == 0) printf("\n");

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        if (worldRank == 0) printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        if (worldRank == 0) printf("\nPerformance:\n");
        if (worldRank == 0) printf("  Triangles: %zu\n", n);
        if (worldRank == 0) printf("  Timesteps: %zu\n", t);
        if (worldRank == 0) printf("  Form factor computations: %.2e\n", kijOps);
        if (worldRank == 0) printf("  Simulation operations: %.2e\n", simOps);
        if (worldRank == 0) printf("  Distance computations: %.2e\n", distOps);
        if (worldRank == 0) printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = state.local * n * sizeof(val_t);
        size_t memTau = state.local * n * sizeof(int);
        size_t memRad = t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        if (worldRank == 0) printf("  Matrix/history memory per rank (rank 0): %.2f MB\n", totalMem / (1024.0 * 1024.0));

        if (worldRank != 0) {
            int status;
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return status;
        }

        // Hash
        uint64_t hash = computeHash(state);
        if (worldRank == 0) printf("  Result hash: %016lX\n", hash);
        if (worldRank == 0) printf("\n");
    
        // Print results for external validation
        if (printResults) {
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        // Validation
        int status = 0;
        if (validate) {
            state.radB.resize(t*n);
            if (t) cudaCheck(cudaMemcpy(state.radB.data(), state.deviceHistory.data, t*n*sizeof(float), cudaMemcpyDeviceToHost));
            if (!validateResults(state)) status = 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}

