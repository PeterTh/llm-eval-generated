/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI + OpenMP + CUDA implementation of room impulse response
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
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
// CUDA data layout and flattened octree
// ============================================================================

// The pointer-based host octree is flattened once so the same visibility
// algorithm can run efficiently on every rank's GPU.  Leaves retain the exact
// triangle lists used by the host tree, including overlap duplicates.
struct DeviceVec3 {
    val_t x, y, z;
};

struct DeviceTriangle {
    DeviceVec3 a, b, c, normal;
};

struct DeviceOctreeNode {
    DeviceVec3 center;
    DeviceVec3 halfExtent;
    int children[8];
    int triangleOffset;
    int triangleCount;
};

struct FlatOctree {
    std::vector<DeviceOctreeNode> nodes;
    std::vector<idx_t> triangleIndices;
};

DeviceVec3 toDeviceVec3(const Vec3& v) {
    return {v.x, v.y, v.z};
}

int flattenOctree(const Octree& octree, FlatOctree& flat) {
    DeviceOctreeNode node{};
    node.center = toDeviceVec3(octree.center);
    node.halfExtent = toDeviceVec3(octree.halfExtent);
    node.triangleOffset = 0;
    node.triangleCount = 0;
    for (int& child : node.children) child = -1;

    const int nodeIndex = static_cast<int>(flat.nodes.size());
    flat.nodes.push_back(node);

    if (!octree.triangleIndices.empty()) {
        flat.nodes[nodeIndex].triangleOffset = static_cast<int>(flat.triangleIndices.size());
        flat.nodes[nodeIndex].triangleCount = static_cast<int>(octree.triangleIndices.size());
        for (size_t triangleIndex : octree.triangleIndices) {
            flat.triangleIndices.push_back(static_cast<idx_t>(triangleIndex));
        }
    } else {
        for (int child = 0; child < 8; ++child) {
            if (octree.children[child]) {
                flat.nodes[nodeIndex].children[child] = flattenOctree(*octree.children[child], flat);
            }
        }
    }
    return nodeIndex;
}

__device__ __forceinline__ DeviceVec3 dadd(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ DeviceVec3 dsub(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ DeviceVec3 dmul(const DeviceVec3& a, val_t s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ val_t ddot(const DeviceVec3& a, const DeviceVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ DeviceVec3 dcross(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ val_t dnorm2(const DeviceVec3& a) {
    return ddot(a, a);
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Stateless samples make Monte-Carlo form factors reproducible regardless of
// MPI partitioning, GPU scheduling, or OpenMP thread count.
__device__ __forceinline__ val_t uniformSample(unsigned long long key) {
    return static_cast<val_t>((mix64(key) >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ DeviceVec3 randomPointInTriangleDevice(
    const DeviceTriangle& triangle, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return dadd(triangle.a, dadd(dmul(dsub(triangle.b, triangle.a), u),
                                 dmul(dsub(triangle.c, triangle.a), v)));
}

__device__ __forceinline__ bool segmentIntersectsBox(
    const DeviceVec3& p1, const DeviceVec3& p2, const DeviceOctreeNode& node) {
    const DeviceVec3 d = dmul(dsub(p2, p1), 0.5f);
    const DeviceVec3 c = dsub(dadd(p1, d), node.center);
    const DeviceVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + 1.0e-6f) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + 1.0e-6f) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + 1.0e-6f) return false;
    return true;
}

__device__ __forceinline__ val_t rayTriangleIntersectDevice(
    const DeviceVec3& origin, const DeviceVec3& direction, const DeviceTriangle& triangle) {
    const DeviceVec3 e1 = dsub(triangle.b, triangle.a);
    const DeviceVec3 e2 = dsub(triangle.c, triangle.a);
    const DeviceVec3 pvec = dcross(direction, e2);
    const val_t det = ddot(e1, pvec);
    if (fabsf(det) < 1.0e-6f) return 3.402823466e+38F;

    const val_t invDet = 1.0f / det;
    const DeviceVec3 tvec = dsub(origin, triangle.a);
    const val_t u = ddot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38F;

    const DeviceVec3 qvec = dcross(tvec, e1);
    const val_t v = ddot(direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e+38F;
    return ddot(e2, qvec) * invDet;
}

__device__ bool isRayBlockedDevice(const DeviceVec3& from, const DeviceVec3& to,
                                   const DeviceTriangle* triangles,
                                   const DeviceOctreeNode* nodes,
                                   const idx_t* triangleIndices,
                                   int sourceTriangle, int destinationTriangle) {
    const DeviceVec3 direction = dsub(to, from);
    const val_t rayLength = sqrtf(dnorm2(direction));
    if (rayLength < 1.0e-6f) return true;
    const DeviceVec3 normalizedDirection = dmul(direction, 1.0f / rayLength);

    // The generated room's fixed minimum leaf size limits the depth to seven;
    // 64 entries cover all deferred siblings while keeping traversal local state small.
    int stack[64];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize > 0) {
        const DeviceOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount != 0) {
            for (int offset = 0; offset < node.triangleCount; ++offset) {
                const int triangleIndex = static_cast<int>(triangleIndices[node.triangleOffset + offset]);
                if (triangleIndex == sourceTriangle || triangleIndex == destinationTriangle) continue;
                const val_t distance = rayTriangleIntersectDevice(
                    from, normalizedDirection, triangles[triangleIndex]);
                if (distance > 1.0e-6f && distance < rayLength - 1.0e-6f) return true;
            }
        } else {
            for (int child = 7; child >= 0; --child) {
                const int childIndex = node.children[child];
                if (childIndex >= 0 && segmentIntersectsBox(from, to, nodes[childIndex])) {
                    // The construction bounds the stack well below 64 entries.
                    if (stackSize < 64) stack[stackSize++] = childIndex;
                }
            }
        }
    }
    return false;
}

// Two independent (i, j) pairs share each 32-thread warp.  Each half warp
// samples one pair's 16 rays and reduces them in ray order.
__global__ void computeFormFactorsKernel(const DeviceTriangle* triangles,
                                         const DeviceOctreeNode* nodes,
                                         const idx_t* triangleIndices,
                                         val_t* kij, int numTriangles,
                                         int firstRow, size_t pairCount) {
    const int thread = threadIdx.x;
    const int lane = thread & (NUM_RAYS - 1);
    const size_t localPair = static_cast<size_t>(blockIdx.x) * 2 + static_cast<size_t>(thread / NUM_RAYS);
    __shared__ val_t contributions[2 * NUM_RAYS];

    val_t contribution = ZERO;
    int receiver = 0;
    int emitter = 0;
    const bool active = localPair < pairCount;
    if (active) {
        receiver = firstRow + static_cast<int>(localPair / static_cast<size_t>(numTriangles));
        emitter = static_cast<int>(localPair % static_cast<size_t>(numTriangles));
        if (receiver != emitter) {
            const DeviceTriangle receiverTriangle = triangles[receiver];
            const DeviceTriangle emitterTriangle = triangles[emitter];
            if (ddot(receiverTriangle.normal, emitterTriangle.normal) <= 0.99f) {
                const unsigned long long sampleBase =
                    ((static_cast<unsigned long long>(receiver) * static_cast<unsigned long long>(numTriangles) +
                      static_cast<unsigned long long>(emitter)) * NUM_RAYS +
                     static_cast<unsigned long long>(lane)) * 4ULL + 42ULL;
                const DeviceVec3 receiverPoint = randomPointInTriangleDevice(
                    receiverTriangle, uniformSample(sampleBase), uniformSample(sampleBase + 1ULL));
                const DeviceVec3 emitterPoint = randomPointInTriangleDevice(
                    emitterTriangle, uniformSample(sampleBase + 2ULL), uniformSample(sampleBase + 3ULL));

                if (!isRayBlockedDevice(receiverPoint, emitterPoint, triangles, nodes, triangleIndices,
                                        receiver, emitter)) {
                    const DeviceVec3 v = dsub(emitterPoint, receiverPoint);
                    const val_t distanceSquared = dnorm2(v);
                    if (distanceSquared >= 1.0e-6f) {
                        const val_t distance = sqrtf(distanceSquared);
                        const val_t cosPhiI = fmaxf(ZERO, ddot(v, receiverTriangle.normal) / distance);
                        const val_t cosPhiJ = fmaxf(ZERO, -ddot(v, emitterTriangle.normal) / distance);
                        if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                            contribution = (cosPhiI * cosPhiJ) / (PI * distanceSquared);
                        }
                    }
                }
            }
        }
    }

    contributions[thread] = contribution;
    __syncthreads();
    for (int stride = NUM_RAYS / 2; stride > 0; stride >>= 1) {
        if (lane < stride) contributions[thread] += contributions[thread + stride];
        __syncthreads();
    }
    if (lane == 0 && active) {
        kij[static_cast<size_t>(receiver) * numTriangles + emitter] = contributions[thread] * INV_NUM_RAYS;
    }
}

__global__ void computeTimeDelaysKernel(const DeviceTriangle* triangles, int* tau,
                                        int numTriangles, int firstRow, size_t pairCount) {
    const size_t localPair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localPair >= pairCount) return;
    const int receiver = firstRow + static_cast<int>(localPair / static_cast<size_t>(numTriangles));
    const int emitter = static_cast<int>(localPair % static_cast<size_t>(numTriangles));
    int delay = 0;
    if (receiver != emitter) {
        const DeviceTriangle receiverTriangle = triangles[receiver];
        const DeviceTriangle emitterTriangle = triangles[emitter];
        const DeviceVec3 receiverCenter = dmul(dadd(dadd(receiverTriangle.a, receiverTriangle.b), receiverTriangle.c), 1.0f / 3.0f);
        const DeviceVec3 emitterCenter = dmul(dadd(dadd(emitterTriangle.a, emitterTriangle.b), emitterTriangle.c), 1.0f / 3.0f);
        delay = static_cast<int>(ceilf(sqrtf(dnorm2(dsub(receiverCenter, emitterCenter))) * INV_WAVE_SPEED));
    }
    tau[static_cast<size_t>(receiver) * numTriangles + emitter] = delay;
}

__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
                                const val_t* rho, val_t* radiosity, int numTriangles,
                                int firstRow, int rowCount, int timestep,
                                int sourceTriangle, int timeOff) {
    const int localReceiver = blockIdx.x;
    if (localReceiver >= rowCount) return;
    const int receiver = firstRow + localReceiver;
    const int thread = threadIdx.x;
    val_t sum = ZERO;
    const size_t matrixRow = static_cast<size_t>(receiver) * numTriangles;
    for (int emitter = thread; emitter < numTriangles; emitter += blockDim.x) {
        if (receiver == emitter) continue;
        const int delay = tau[matrixRow + emitter];
        if (timestep < delay) continue;
        const val_t formFactor = kij[matrixRow + emitter];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radiosity[
            static_cast<size_t>(timestep - delay) * numTriangles + emitter];
        if (sourceRadiosity <= ZERO) continue;
        sum += fminf(formFactor * areas[emitter], ONE) * sourceRadiosity;
    }

    __shared__ val_t partial[256];
    partial[thread] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }
    if (thread == 0) {
        radiosity[static_cast<size_t>(timestep) * numTriangles + receiver] =
            rho[receiver] * partial[0] + ((receiver == sourceTriangle && timestep < timeOff) ? ONE : ZERO);
    }
}

__global__ void computeDistancesKernel(const val_t* radiosity, val_t* distances,
                                       int numTriangles, int numTimesteps, int firstRow,
                                       int rowCount, int sourceTriangle) {
    const int localReceiver = blockIdx.x;
    if (localReceiver >= rowCount) return;
    const int receiver = firstRow + localReceiver;
    const int thread = threadIdx.x;
    __shared__ val_t partial[256];
    __shared__ val_t maxCorrelation;
    __shared__ int bestTime;
    if (thread == 0) {
        maxCorrelation = ZERO;
        bestTime = 0;
    }
    __syncthreads();

    for (int delay = 0; delay < numTimesteps; ++delay) {
        val_t sum = ZERO;
        for (int time = delay + thread; time < numTimesteps; time += blockDim.x) {
            sum += radiosity[static_cast<size_t>(time) * numTriangles + receiver] *
                   radiosity[static_cast<size_t>(time - delay) * numTriangles + sourceTriangle];
        }
        partial[thread] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride) partial[thread] += partial[thread + stride];
            __syncthreads();
        }
        if (thread == 0 && partial[0] > maxCorrelation) {
            maxCorrelation = partial[0];
            bestTime = delay;
        }
        __syncthreads();
    }
    if (thread == 0) distances[receiver] = WAVE_SPEED * static_cast<val_t>(bestTime);
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

struct ParallelContext {
    int rank = 0;
    int worldSize = 1;
    int localRank = 0;
    int device = 0;

    bool isRoot() const { return rank == 0; }
};

struct RowPartition {
    size_t first = 0;
    size_t count = 0;
};

RowPartition partitionRows(size_t totalRows, int rank, int worldSize) {
    const size_t baseRows = totalRows / static_cast<size_t>(worldSize);
    const size_t remainder = totalRows % static_cast<size_t>(worldSize);
    const size_t count = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t first = static_cast<size_t>(rank) * baseRows +
                         std::min(static_cast<size_t>(rank), remainder);
    return {first, count};
}

[[noreturn]] void abortWithCudaError(cudaError_t error, const char* expression,
                                     const ParallelContext& parallel) {
    fprintf(stderr, "MPI rank %d: CUDA failure in %s: %s\n", parallel.rank, expression,
            cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 2);
    std::abort();
}

void checkCuda(cudaError_t error, const char* expression, const ParallelContext& parallel) {
    if (error != cudaSuccess) abortWithCudaError(error, expression, parallel);
}

#define CUDA_CHECK(parallel, expression) checkCuda((expression), #expression, (parallel))

template <typename T>
T* allocateDevice(size_t count, const ParallelContext& parallel) {
    T* pointer = nullptr;
    CUDA_CHECK(parallel, cudaMalloc(reinterpret_cast<void**>(&pointer),
                                    std::max<size_t>(count, 1) * sizeof(T)));
    return pointer;
}

struct DeviceBuffers {
    DeviceTriangle* triangles = nullptr;
    DeviceOctreeNode* octreeNodes = nullptr;
    idx_t* octreeTriangleIndices = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* radiosity = nullptr;
    val_t* distances = nullptr;

    void release() {
        // Cleanup does not hide an earlier computation error and must also work
        // after MPI_Finalize, so cudaFree errors are intentionally ignored here.
        if (triangles) cudaFree(triangles);
        if (octreeNodes) cudaFree(octreeNodes);
        if (octreeTriangleIndices) cudaFree(octreeTriangleIndices);
        if (kij) cudaFree(kij);
        if (tau) cudaFree(tau);
        if (areas) cudaFree(areas);
        if (rho) cudaFree(rho);
        if (radiosity) cudaFree(radiosity);
        if (distances) cudaFree(distances);
        *this = {};
    }
};

template <typename T>
void allgatherRowsInPlace(T* globalValues, size_t totalRows, size_t rowWidth,
                          MPI_Datatype datatype, const ParallelContext& parallel) {
    const size_t maximumCount = static_cast<size_t>(std::numeric_limits<int>::max());
    if (rowWidth == 0 || rowWidth > maximumCount) {
        fprintf(stderr, "MPI rank %d: unsupported MPI row width %zu\n", parallel.rank, rowWidth);
        MPI_Abort(MPI_COMM_WORLD, 3);
        std::abort();
    }
    const size_t rowsPerCollective = std::max<size_t>(1, maximumCount / rowWidth);
    std::vector<int> receiveCounts(static_cast<size_t>(parallel.worldSize));
    std::vector<int> displacements(static_cast<size_t>(parallel.worldSize));

    for (size_t chunkFirst = 0; chunkFirst < totalRows; chunkFirst += rowsPerCollective) {
        const size_t chunkLast = std::min(totalRows, chunkFirst + rowsPerCollective);
        for (int rank = 0; rank < parallel.worldSize; ++rank) {
            const RowPartition partition = partitionRows(totalRows, rank, parallel.worldSize);
            const size_t first = std::max(chunkFirst, partition.first);
            const size_t last = std::min(chunkLast, partition.first + partition.count);
            const size_t count = last > first ? (last - first) * rowWidth : 0;
            receiveCounts[static_cast<size_t>(rank)] = static_cast<int>(count);
            displacements[static_cast<size_t>(rank)] =
                static_cast<int>((first - chunkFirst) * rowWidth);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, datatype, globalValues + chunkFirst * rowWidth,
                       receiveCounts.data(), displacements.data(), datatype, MPI_COMM_WORLD);
    }
}

DeviceBuffers initializeDeviceBuffers(const SimulationState& state,
                                      const ParallelContext& parallel) {
    FlatOctree flatOctree;
    flatOctree.nodes.reserve(state.numTriangles * 2);
    flattenOctree(state.octree, flatOctree);

    std::vector<DeviceTriangle> hostTriangles(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const Triangle& triangle = state.triangles[static_cast<size_t>(i)];
        hostTriangles[static_cast<size_t>(i)] = {
            toDeviceVec3(triangle.a), toDeviceVec3(triangle.b), toDeviceVec3(triangle.c),
            toDeviceVec3(triangle.normal())};
    }

    const size_t matrixElements = state.numTriangles * state.numTriangles;
    DeviceBuffers device;
    device.triangles = allocateDevice<DeviceTriangle>(state.numTriangles, parallel);
    device.octreeNodes = allocateDevice<DeviceOctreeNode>(flatOctree.nodes.size(), parallel);
    device.octreeTriangleIndices = allocateDevice<idx_t>(flatOctree.triangleIndices.size(), parallel);
    device.kij = allocateDevice<val_t>(matrixElements, parallel);
    device.tau = allocateDevice<int>(matrixElements, parallel);
    device.areas = allocateDevice<val_t>(state.numTriangles, parallel);
    device.rho = allocateDevice<val_t>(state.numTriangles, parallel);
    device.radiosity = allocateDevice<val_t>(state.numTriangles * state.numTimesteps, parallel);
    device.distances = allocateDevice<val_t>(state.numTriangles, parallel);

    CUDA_CHECK(parallel, cudaMemcpy(device.triangles, hostTriangles.data(),
                                    hostTriangles.size() * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(parallel, cudaMemcpy(device.octreeNodes, flatOctree.nodes.data(),
                                    flatOctree.nodes.size() * sizeof(DeviceOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(parallel, cudaMemcpy(device.octreeTriangleIndices, flatOctree.triangleIndices.data(),
                                    flatOctree.triangleIndices.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(parallel, cudaMemcpy(device.areas, state.areas.data(),
                                    state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(parallel, cudaMemcpy(device.rho, state.rho.data(),
                                    state.rho.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(parallel, cudaMemset(device.radiosity, 0,
                                    state.numTriangles * state.numTimesteps * sizeof(val_t)));
    return device;
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          const ParallelContext& parallel) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (parallel.isRoot()) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (parallel.isRoot()) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // OpenMP prepares the rank-local host representation while the accelerator
    // later computes the dense matrices and wave propagation.
    state.areas.resize(state.numTriangles);
    state.rho.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();
        state.rho[static_cast<size_t>(i)] = reflectivity;
    }

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state, DeviceBuffers& device,
                       const RowPartition& rows, const ParallelContext& parallel) {
    if (parallel.isRoot()) printf("Computing time delays (Tau)...\n");
    const size_t localPairs = rows.count * state.numTriangles;
    if (localPairs != 0) {
        constexpr int threads = 256;
        const size_t blocks = (localPairs + threads - 1) / threads;
        computeTimeDelaysKernel<<<static_cast<unsigned int>(blocks), threads>>>(
            device.triangles, device.tau, static_cast<int>(state.numTriangles),
            static_cast<int>(rows.first), localPairs);
        CUDA_CHECK(parallel, cudaGetLastError());
        CUDA_CHECK(parallel, cudaMemcpy(state.tau.data() + rows.first * state.numTriangles,
                                        device.tau + rows.first * state.numTriangles,
                                        localPairs * sizeof(int), cudaMemcpyDeviceToHost));
    }
    allgatherRowsInPlace(state.tau.data(), state.numTriangles, state.numTriangles, MPI_INT, parallel);
    CUDA_CHECK(parallel, cudaMemcpy(device.tau, state.tau.data(),
                                    state.tau.size() * sizeof(int), cudaMemcpyHostToDevice));
}

void computeFormFactors(SimulationState& state, DeviceBuffers& device,
                        const RowPartition& rows, const ParallelContext& parallel) {
    if (parallel.isRoot()) printf("Computing form factors (Kij)...\n");
    const size_t localPairs = rows.count * state.numTriangles;
    if (localPairs != 0) {
        const size_t blocks = (localPairs + 1) / 2;
        computeFormFactorsKernel<<<static_cast<unsigned int>(blocks), 2 * NUM_RAYS>>>(
            device.triangles, device.octreeNodes, device.octreeTriangleIndices, device.kij,
            static_cast<int>(state.numTriangles), static_cast<int>(rows.first), localPairs);
        CUDA_CHECK(parallel, cudaGetLastError());
        CUDA_CHECK(parallel, cudaMemcpy(state.kij.data() + rows.first * state.numTriangles,
                                        device.kij + rows.first * state.numTriangles,
                                        localPairs * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    allgatherRowsInPlace(state.kij.data(), state.numTriangles, state.numTriangles, MPI_FLOAT, parallel);
    CUDA_CHECK(parallel, cudaMemcpy(device.kij, state.kij.data(),
                                    state.kij.size() * sizeof(val_t), cudaMemcpyHostToDevice));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, DeviceBuffers& device,
                   const RowPartition& rows, const ParallelContext& parallel) {
    if (parallel.isRoot()) printf("Running wave propagation simulation...\n");
    constexpr int threads = 256;
    const int numTriangles = static_cast<int>(state.numTriangles);
    const int timeOff = static_cast<int>(state.numTimesteps / 2);
    for (int timestep = 0; timestep < static_cast<int>(state.numTimesteps); ++timestep) {
        if (rows.count != 0) {
            propagateKernel<<<static_cast<unsigned int>(rows.count), threads>>>(
                device.kij, device.tau, device.areas, device.rho, device.radiosity,
                numTriangles, static_cast<int>(rows.first), static_cast<int>(rows.count), timestep,
                static_cast<int>(state.sourceIndex), timeOff);
            CUDA_CHECK(parallel, cudaGetLastError());
            CUDA_CHECK(parallel, cudaMemcpy(
                state.radB.data() + static_cast<size_t>(timestep) * state.numTriangles + rows.first,
                device.radiosity + static_cast<size_t>(timestep) * state.numTriangles + rows.first,
                rows.count * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        // Each receiver row is independent within a timestep, but future time
        // delays can read any emitter.  Gathering this slice preserves the
        // original causality while allowing every rank to own only its rows.
        allgatherRowsInPlace(state.radB.data() + static_cast<size_t>(timestep) * state.numTriangles,
                             state.numTriangles, 1, MPI_FLOAT, parallel);
        CUDA_CHECK(parallel, cudaMemcpy(device.radiosity + static_cast<size_t>(timestep) * state.numTriangles,
                                        state.radB.data() + static_cast<size_t>(timestep) * state.numTriangles,
                                        state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (parallel.isRoot() && ((timestep + 1) % 10 == 0 ||
                                  timestep + 1 == static_cast<int>(state.numTimesteps))) {
            printf("  Timestep %d/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, DeviceBuffers& device,
                      const RowPartition& rows, const ParallelContext& parallel) {
    if (parallel.isRoot()) printf("Computing distances via cross-correlation...\n");
    if (rows.count != 0) {
        computeDistancesKernel<<<static_cast<unsigned int>(rows.count), 256>>>(
            device.radiosity, device.distances, static_cast<int>(state.numTriangles),
            static_cast<int>(state.numTimesteps), static_cast<int>(rows.first),
            static_cast<int>(rows.count), static_cast<int>(state.sourceIndex));
        CUDA_CHECK(parallel, cudaGetLastError());
        CUDA_CHECK(parallel, cudaMemcpy(state.distances.data() + rows.first,
                                        device.distances + rows.first, rows.count * sizeof(val_t),
                                        cudaMemcpyDeviceToHost));
    }
    allgatherRowsInPlace(state.distances.data(), state.numTriangles, 1, MPI_FLOAT, parallel);
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
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
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
    #pragma omp parallel for schedule(static) reduction(^:hash)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + static_cast<uint64_t>(i)) * 0x9e3779b97f4a7c15ULL;
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

struct RuntimeOptions {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIndex = 0;
    val_t reflectivity = 0.8f;
    int validate = 0;
    int printResults = 0;
};

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        fprintf(stderr, "Unable to initialize MPI.\n");
        return 1;
    }

    ParallelContext parallel;
    MPI_Comm_rank(MPI_COMM_WORLD, &parallel.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallel.worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (parallel.isRoot()) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support.\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, parallel.rank, MPI_INFO_NULL,
                        &localCommunicator);
    MPI_Comm_rank(localCommunicator, &parallel.localRank);
    MPI_Comm_free(&localCommunicator);

    int deviceCount = 0;
    CUDA_CHECK(parallel, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (parallel.isRoot()) fprintf(stderr, "No CUDA accelerator is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    parallel.device = parallel.localRank % deviceCount;
    CUDA_CHECK(parallel, cudaSetDevice(parallel.device));

    RuntimeOptions options;
    int parseStatus = 0;
    int showHelp = 0;
    if (parallel.isRoot()) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                options.targetTriangles = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                options.timesteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                options.sourceIndex = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
                options.reflectivity = static_cast<val_t>(atof(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                options.validate = 1;
            } else if (strcmp(argv[i], "-o") == 0) {
                options.printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                parseStatus = 1;
            }
        }
        if (!showHelp && (options.targetTriangles <= 0 || options.timesteps <= 0 ||
                          options.sourceIndex < 0 || options.reflectivity < ZERO ||
                          options.reflectivity > ONE)) {
            fprintf(stderr, "Invalid simulation parameters.\n");
            parseStatus = 1;
        }
    }
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (showHelp || parseStatus != 0) {
        if (parallel.isRoot()) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus;
    }
    MPI_Bcast(&options, static_cast<int>(sizeof(options)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int subdivisions = getSubdivisionsForTriangleCount(options.targetTriangles);
    if (parallel.isRoot()) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", options.targetTriangles, subdivisions);
        printf("Timesteps: %d\n", options.timesteps);
        printf("Source triangle: %d\n", options.sourceIndex);
        printf("Reflectivity: %.2f\n", options.reflectivity);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA devices per node: %d\n",
               parallel.worldSize, omp_get_max_threads(), deviceCount);
        printf("\n");
    }

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(options.timesteps),
                         static_cast<size_t>(options.sourceIndex), options.reflectivity, parallel);
    if (state.numTriangles > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (parallel.isRoot()) fprintf(stderr, "Triangle count exceeds CUDA kernel index range.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
        return 3;
    }
    const RowPartition rows = partitionRows(state.numTriangles, parallel.rank, parallel.worldSize);
    DeviceBuffers device = initializeDeviceBuffers(state, parallel);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startPre = std::chrono::high_resolution_clock::now();
    computeTimeDelays(state, device, rows, parallel);
    computeFormFactors(state, device, rows, parallel);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endPre = std::chrono::high_resolution_clock::now();
    long long preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long long maxPreDuration = 0;
    MPI_Reduce(&preDuration, &maxPreDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (parallel.isRoot()) printf("Precomputation time: %lld ms\n\n", maxPreDuration);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startSimulation = std::chrono::high_resolution_clock::now();
    runSimulation(state, device, rows, parallel);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endSimulation = std::chrono::high_resolution_clock::now();
    long long simulationDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endSimulation - startSimulation).count();
    long long maxSimulationDuration = 0;
    MPI_Reduce(&simulationDuration, &maxSimulationDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (parallel.isRoot()) printf("Simulation time: %lld ms\n\n", maxSimulationDuration);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto startDistance = std::chrono::high_resolution_clock::now();
    computeDistances(state, device, rows, parallel);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endDistance = std::chrono::high_resolution_clock::now();
    long long distanceDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(endDistance - startDistance).count();
    long long maxDistanceDuration = 0;
    MPI_Reduce(&distanceDuration, &maxDistanceDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int validationStatus = 0;
    if (parallel.isRoot()) {
        printf("Distance computation time: %lld ms\n\n", maxDistanceDuration);
        const long long totalTime = maxPreDuration + maxSimulationDuration + maxDistanceDuration;
        printf("Total computation time: %lld ms\n", totalTime);

        const size_t n = state.numTriangles;
        const size_t t = state.numTimesteps;
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = kijOps * static_cast<double>(t);
        const double distanceOps = static_cast<double>(n) * static_cast<double>(t) * static_cast<double>(t);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distanceOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        const size_t memKij = n * n * sizeof(val_t);
        const size_t memTau = n * n * sizeof(int);
        const size_t memRad = 2 * t * n * sizeof(val_t);
        const size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(computeHash(state)));

        if (options.printResults) {
            std::vector<double> distanceData(state.distances.begin(), state.distances.end());
            print_results(distanceData, "Distances");
        }
        if (options.validate && !validateResults(state)) validationStatus = 1;
    }

    MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    device.release();
    MPI_Finalize();
    return validationStatus;
}
