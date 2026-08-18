/**
 * Room Response Simulation Benchmark
 * 
 * This CUDA implementation of room impulse response simulation uses
 * radiosity-based wave propagation. It models how sound/light
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
constexpr int CUDA_BLOCK_SIZE = 256;

// CUDA errors are fatal: this benchmark deliberately has no CPU execution path.
inline void cudaCheck(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

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
// CUDA representation and kernels
// ============================================================================
// The host octree owns dynamic C++ objects, so it is flattened in depth-first
// order before upload.  An escape index lets a ray walk that layout without a
// per-thread traversal stack, which keeps the form-factor kernel scalable.

struct DeviceVec3 {
    val_t x, y, z;
};

struct DeviceTriangle {
    DeviceVec3 a, b, c;
    DeviceVec3 normal;
};

struct DeviceOctreeNode {
    DeviceVec3 center;
    DeviceVec3 halfExtent;
    int triangleOffset;
    int triangleCount;
    int escapeIndex;
};

inline DeviceVec3 toDeviceVec3(const Vec3& v) {
    return {v.x, v.y, v.z};
}

inline DeviceTriangle toDeviceTriangle(const Triangle& tri) {
    return {toDeviceVec3(tri.a), toDeviceVec3(tri.b), toDeviceVec3(tri.c),
            toDeviceVec3(tri.normal())};
}

void flattenOctree(const Octree& octree, std::vector<DeviceOctreeNode>& nodes,
                   std::vector<int>& triangleIndices) {
    const int nodeIndex = static_cast<int>(nodes.size());
    nodes.push_back({toDeviceVec3(octree.center), toDeviceVec3(octree.halfExtent),
                     0, 0, 0});

    if (!octree.triangleIndices.empty()) {
        DeviceOctreeNode& node = nodes[nodeIndex];
        node.triangleOffset = static_cast<int>(triangleIndices.size());
        node.triangleCount = static_cast<int>(octree.triangleIndices.size());
        for (size_t triIndex : octree.triangleIndices) {
            triangleIndices.push_back(static_cast<int>(triIndex));
        }
    } else {
        for (const auto& child : octree.children) {
            if (child) flattenOctree(*child, nodes, triangleIndices);
        }
    }

    nodes[nodeIndex].escapeIndex = static_cast<int>(nodes.size());
}

__device__ __forceinline__ DeviceVec3 dAdd(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ DeviceVec3 dSub(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ DeviceVec3 dMul(const DeviceVec3& v, val_t s) {
    return {v.x * s, v.y * s, v.z * s};
}

__device__ __forceinline__ DeviceVec3 dNeg(const DeviceVec3& v) {
    return {-v.x, -v.y, -v.z};
}

__device__ __forceinline__ val_t dDot(const DeviceVec3& a, const DeviceVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ DeviceVec3 dCross(const DeviceVec3& a, const DeviceVec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ val_t dSquaredNorm(const DeviceVec3& v) {
    return dDot(v, v);
}

__device__ __forceinline__ DeviceVec3 dCenter(const DeviceTriangle& tri) {
    return dMul(dAdd(dAdd(tri.a, tri.b), tri.c), 1.0f / 3.0f);
}

__device__ __forceinline__ bool dRayIntersectsBox(const DeviceVec3& p1,
                                                    const DeviceVec3& p2,
                                                    const DeviceOctreeNode& node) {
    const DeviceVec3 d = dMul(dSub(p2, p1), 0.5f);
    const DeviceVec3 c = dSub(dAdd(p1, d), node.center);
    const DeviceVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ bool dRayTriangleBlocks(const DeviceVec3& origin,
                                                     const DeviceVec3& direction,
                                                     val_t rayLength,
                                                     const DeviceTriangle& tri) {
    const DeviceVec3 e1 = dSub(tri.b, tri.a);
    const DeviceVec3 e2 = dSub(tri.c, tri.a);
    const DeviceVec3 pvec = dCross(direction, e2);
    const val_t det = dDot(e1, pvec);
    if (fabsf(det) < EPSILON) return false;

    const val_t invDet = 1.0f / det;
    const DeviceVec3 tvec = dSub(origin, tri.a);
    const val_t u = dDot(tvec, pvec) * invDet;
    if (u < ZERO || u > ONE) return false;

    const DeviceVec3 qvec = dCross(tvec, e1);
    const val_t v = dDot(direction, qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    const val_t distance = dDot(e2, qvec) * invDet;
    return distance > EPSILON && distance < rayLength - EPSILON;
}

__device__ __forceinline__ bool dIsRayBlocked(
    const DeviceVec3& from, const DeviceVec3& to,
    const DeviceTriangle* __restrict__ triangles,
    const DeviceOctreeNode* __restrict__ nodes,
    const int* __restrict__ leafTriangleIndices,
    int sourceTriangle, int destinationTriangle) {
    const DeviceVec3 direction = dSub(to, from);
    const val_t rayLength = sqrtf(dSquaredNorm(direction));
    if (rayLength < EPSILON) return true;
    const DeviceVec3 directionNormal = dMul(direction, 1.0f / rayLength);

    // Nodes are stored in preorder.  A miss skips the complete subtree, while
    // a hit advances to the next node, reproducing the original child order.
    int nodeIndex = 0;
    const int nodeEnd = nodes[0].escapeIndex;
    while (nodeIndex < nodeEnd) {
        const DeviceOctreeNode node = nodes[nodeIndex];
        if (nodeIndex != 0 && !dRayIntersectsBox(from, to, node)) {
            nodeIndex = node.escapeIndex;
            continue;
        }

        if (node.triangleCount != 0) {
            for (int k = 0; k < node.triangleCount; ++k) {
                const int triangleIndex = leafTriangleIndices[node.triangleOffset + k];
                if (triangleIndex != sourceTriangle && triangleIndex != destinationTriangle &&
                    dRayTriangleBlocks(from, directionNormal, rayLength, triangles[triangleIndex])) {
                    return true;
                }
            }
        }

        ++nodeIndex;
        // The final node is always a leaf, so the next iteration exits at the
        // root's escape index.  Internal nodes without children are also handled.
    }
    return false;
}

__device__ __forceinline__ uint32_t dRandomU32(uint32_t& state) {
    // A small per-pair generator avoids serial RNG state while retaining the
    // independent uniform samples used by the Monte-Carlo form-factor estimate.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ val_t dRandomUniform(uint32_t& state) {
    return static_cast<val_t>(dRandomU32(state) >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ DeviceVec3 dRandomPointInTriangle(const DeviceTriangle& tri,
                                                               uint32_t& state) {
    val_t u = dRandomUniform(state);
    val_t v = dRandomUniform(state);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return dAdd(tri.a, dAdd(dMul(dSub(tri.b, tri.a), u), dMul(dSub(tri.c, tri.a), v)));
}

__global__ void computeTimeDelaysKernel(const DeviceTriangle* __restrict__ triangles,
                                        int numTriangles, int* __restrict__ tau) {
    const size_t linearIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(numTriangles) * numTriangles;
    if (linearIndex >= total) return;

    const int i = static_cast<int>(linearIndex / numTriangles);
    const int j = static_cast<int>(linearIndex - static_cast<size_t>(i) * numTriangles);
    if (i == j) {
        tau[linearIndex] = 0;
        return;
    }
    const DeviceVec3 delta = dSub(dCenter(triangles[i]), dCenter(triangles[j]));
    tau[linearIndex] = static_cast<int>(ceilf(sqrtf(dSquaredNorm(delta)) * INV_WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(
    const DeviceTriangle* __restrict__ triangles, int numTriangles,
    const DeviceOctreeNode* __restrict__ nodes,
    const int* __restrict__ leafTriangleIndices, val_t* __restrict__ kij) {
    const size_t linearIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(numTriangles) * numTriangles;
    if (linearIndex >= total) return;

    const int i = static_cast<int>(linearIndex / numTriangles);
    const int j = static_cast<int>(linearIndex - static_cast<size_t>(i) * numTriangles);
    if (i == j) {
        kij[linearIndex] = ZERO;
        return;
    }

    const DeviceTriangle triI = triangles[i];
    const DeviceTriangle triJ = triangles[j];
    if (dDot(triI.normal, triJ.normal) > 0.99f) {
        kij[linearIndex] = ZERO;
        return;
    }

    uint32_t randomState = static_cast<uint32_t>(linearIndex) * 747796405u + 2891336453u;
    randomState ^= static_cast<uint32_t>(linearIndex >> 32) * 277803737u;
    randomState ^= 42u;
    if (randomState == 0) randomState = 1;

    val_t result = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const DeviceVec3 pI = dRandomPointInTriangle(triI, randomState);
        const DeviceVec3 pJ = dRandomPointInTriangle(triJ, randomState);
        if (dIsRayBlocked(pI, pJ, triangles, nodes, leafTriangleIndices, i, j)) continue;

        const DeviceVec3 v = dSub(pJ, pI);
        const val_t distanceSquared = dSquaredNorm(v);
        if (distanceSquared < EPSILON) continue;
        const val_t inverseDistance = 1.0f / sqrtf(distanceSquared);
        const val_t cosPhiI = fmaxf(ZERO, dDot(v, triI.normal) * inverseDistance);
        const val_t cosPhiJ = fmaxf(ZERO, dDot(dNeg(v), triJ.normal) * inverseDistance);
        if (cosPhiI > ZERO && cosPhiJ > ZERO) {
            result += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
        }
    }
    kij[linearIndex] = result * INV_NUM_RAYS;
}

__global__ void propagateTimestepKernel(
    int timestep, int numTriangles, const val_t* __restrict__ areas,
    const val_t* __restrict__ rho, const val_t* __restrict__ kij,
    const int* __restrict__ tau, const val_t* __restrict__ radE,
    val_t* __restrict__ radB) {
    const int receiver = blockIdx.x;
    if (receiver >= numTriangles) return;

    val_t partialSum = ZERO;
    const size_t rowOffset = static_cast<size_t>(receiver) * numTriangles;
    for (int emitter = threadIdx.x; emitter < numTriangles; emitter += blockDim.x) {
        if (emitter == receiver) continue;
        const int delay = tau[rowOffset + emitter];
        if (timestep < delay) continue;
        const val_t formFactor = kij[rowOffset + emitter];
        if (formFactor <= ZERO) continue;
        const size_t sourceOffset = static_cast<size_t>(timestep - delay) * numTriangles + emitter;
        const val_t sourceRadiosity = radB[sourceOffset];
        if (sourceRadiosity <= ZERO) continue;
        partialSum += fminf(formFactor * areas[emitter], ONE) * sourceRadiosity;
    }

    __shared__ val_t sums[CUDA_BLOCK_SIZE];
    sums[threadIdx.x] = partialSum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const size_t outputOffset = static_cast<size_t>(timestep) * numTriangles + receiver;
        radB[outputOffset] = rho[receiver] * sums[0] + radE[outputOffset];
    }
}

__global__ void computeDistancesKernel(const val_t* __restrict__ radB, int numTriangles,
                                       int numTimesteps, int sourceIndex,
                                       val_t* __restrict__ distances) {
    const int triangle = blockIdx.x;
    if (triangle >= numTriangles) return;

    val_t bestCorrelation = ZERO;
    int bestTime = 0;
    for (int lag = threadIdx.x; lag < numTimesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (int time = lag; time < numTimesteps; ++time) {
            const val_t response = radB[static_cast<size_t>(time) * numTriangles + triangle];
            const val_t source = radB[static_cast<size_t>(time - lag) * numTriangles + sourceIndex];
            sum += source * response;
        }
        if (sum > bestCorrelation) {
            bestCorrelation = sum;
            bestTime = lag;
        }
    }

    __shared__ val_t correlations[CUDA_BLOCK_SIZE];
    __shared__ int times[CUDA_BLOCK_SIZE];
    correlations[threadIdx.x] = bestCorrelation;
    times[threadIdx.x] = bestTime;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t otherCorrelation = correlations[threadIdx.x + stride];
            const int otherTime = times[threadIdx.x + stride];
            if (otherCorrelation > correlations[threadIdx.x] ||
                (otherCorrelation == correlations[threadIdx.x] && otherTime < times[threadIdx.x])) {
                correlations[threadIdx.x] = otherCorrelation;
                times[threadIdx.x] = otherTime;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[triangle] = WAVE_SPEED * static_cast<val_t>(times[0]);
    }
}

class CudaSimulation {
public:
    explicit CudaSimulation(const SimulationState& state)
        : numTriangles_(static_cast<int>(state.numTriangles)),
          numTimesteps_(static_cast<int>(state.numTimesteps)),
          matrixSize_(state.numTriangles * state.numTriangles),
          radSize_(state.numTriangles * state.numTimesteps) {
        if (state.numTriangles > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            state.numTimesteps > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "CUDA implementation requires triangle and timestep counts fitting in int.\n");
            std::exit(EXIT_FAILURE);
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            std::fprintf(stderr, "No CUDA device is available.\n");
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<DeviceTriangle> deviceTriangles;
        deviceTriangles.reserve(state.numTriangles);
        for (const Triangle& tri : state.triangles) deviceTriangles.push_back(toDeviceTriangle(tri));

        std::vector<DeviceOctreeNode> deviceNodes;
        std::vector<int> leafTriangleIndices;
        flattenOctree(state.octree, deviceNodes, leafTriangleIndices);

        allocateAndCopy(dTriangles_, deviceTriangles.data(), deviceTriangles.size());
        allocateAndCopy(dNodes_, deviceNodes.data(), deviceNodes.size());
        allocateAndCopy(dLeafTriangleIndices_, leafTriangleIndices.data(), leafTriangleIndices.size());
        allocateAndCopy(dAreas_, state.areas.data(), state.areas.size());
        allocateAndCopy(dRho_, state.rho.data(), state.rho.size());
        allocateAndCopy(dRadE_, state.radE.data(), radSize_);
        allocate(dRadB_, radSize_);
        allocate(dKij_, matrixSize_);
        allocate(dTau_, matrixSize_);
        allocate(dDistances_, state.numTriangles);
        if (radSize_ != 0) CUDA_CHECK(cudaMemset(dRadB_, 0, radSize_ * sizeof(val_t)));
    }

    CudaSimulation(const CudaSimulation&) = delete;
    CudaSimulation& operator=(const CudaSimulation&) = delete;

    ~CudaSimulation() {
        cudaFree(dDistances_);
        cudaFree(dTau_);
        cudaFree(dKij_);
        cudaFree(dRadB_);
        cudaFree(dRadE_);
        cudaFree(dRho_);
        cudaFree(dAreas_);
        cudaFree(dLeafTriangleIndices_);
        cudaFree(dNodes_);
        cudaFree(dTriangles_);
    }

    void computeTimeDelays(SimulationState& state) {
        launchMatrixKernel(dTriangles_, numTriangles_, dTau_);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!state.tau.empty()) copyToHost(state.tau.data(), dTau_, matrixSize_);
    }

    void computeFormFactors(SimulationState& state) {
        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid(static_cast<unsigned int>((matrixSize_ + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE));
        computeFormFactorsKernel<<<grid, block>>>(dTriangles_, numTriangles_, dNodes_,
                                                   dLeafTriangleIndices_, dKij_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!state.kij.empty()) copyToHost(state.kij.data(), dKij_, matrixSize_);
    }

    void runSimulation(SimulationState& state) {
        if (numTimesteps_ == 0) return;
        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid(numTriangles_);
        for (int timestep = 0; timestep < numTimesteps_; ++timestep) {
            propagateTimestepKernel<<<grid, block>>>(timestep, numTriangles_, dAreas_, dRho_, dKij_,
                                                      dTau_, dRadE_, dRadB_);
            CUDA_CHECK(cudaGetLastError());
            if ((timestep + 1) % 10 == 0 || timestep + 1 == numTimesteps_) {
                printf("  Timestep %d/%d\n", timestep + 1, numTimesteps_);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!state.radB.empty()) copyToHost(state.radB.data(), dRadB_, radSize_);
    }

    void computeDistances(SimulationState& state) {
        if (numTimesteps_ == 0) {
            CUDA_CHECK(cudaMemset(dDistances_, 0, state.numTriangles * sizeof(val_t)));
        } else {
            computeDistancesKernel<<<numTriangles_, CUDA_BLOCK_SIZE>>>(
                dRadB_, numTriangles_, numTimesteps_, static_cast<int>(state.sourceIndex), dDistances_);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        copyToHost(state.distances.data(), dDistances_, state.numTriangles);
    }

private:
    int numTriangles_;
    int numTimesteps_;
    size_t matrixSize_;
    size_t radSize_;
    DeviceTriangle* dTriangles_ = nullptr;
    DeviceOctreeNode* dNodes_ = nullptr;
    int* dLeafTriangleIndices_ = nullptr;
    val_t* dAreas_ = nullptr;
    val_t* dRho_ = nullptr;
    val_t* dKij_ = nullptr;
    int* dTau_ = nullptr;
    val_t* dRadE_ = nullptr;
    val_t* dRadB_ = nullptr;
    val_t* dDistances_ = nullptr;

    template<typename T>
    static void allocate(T*& devicePointer, size_t count) {
        if (count != 0) CUDA_CHECK(cudaMalloc(&devicePointer, count * sizeof(T)));
    }

    template<typename T>
    static void allocateAndCopy(T*& devicePointer, const T* hostPointer, size_t count) {
        allocate(devicePointer, count);
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(devicePointer, hostPointer, count * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    template<typename T>
    static void copyToHost(T* hostPointer, const T* devicePointer, size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMemcpy(hostPointer, devicePointer, count * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    static void launchMatrixKernel(const DeviceTriangle* triangles, int numTriangles, int* tau) {
        const size_t matrixSize = static_cast<size_t>(numTriangles) * numTriangles;
        const dim3 block(CUDA_BLOCK_SIZE);
        const dim3 grid(static_cast<unsigned int>((matrixSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE));
        computeTimeDelaysKernel<<<grid, block>>>(triangles, numTriangles, tau);
        CUDA_CHECK(cudaGetLastError());
    }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          bool retainValidationData) {
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
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    if (retainValidationData) {
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.tau.resize(state.numTriangles * state.numTriangles, 0);
        state.radB.resize(timesteps * state.numTriangles, ZERO);
    }
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
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
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
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
                         static_cast<size_t>(sourceIdx), reflectivity, validate);

    printf("\n");

    // All numerical phases execute on CUDA.  The host state is only retained
    // for input construction, reporting, and the existing validation routine.
    CudaSimulation cudaSimulation(state);

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    printf("Computing time delays (Tau)...\n");
    cudaSimulation.computeTimeDelays(state);
    printf("Computing form factors (Kij)...\n");
    cudaSimulation.computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    printf("Running wave propagation simulation...\n");
    cudaSimulation.runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    printf("Computing distances via cross-correlation...\n");
    cudaSimulation.computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

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
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
