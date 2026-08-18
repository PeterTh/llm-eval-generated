/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
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
#include <stdexcept>
#include <string>
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

#define ROOMSIM_HD __host__ __device__
#define ROOMSIM_DEVICE __device__

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    ROOMSIM_HD constexpr Vec3() : x(0), y(0), z(0) {}
    ROOMSIM_HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    ROOMSIM_HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    ROOMSIM_HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    ROOMSIM_HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    ROOMSIM_HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    ROOMSIM_HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    ROOMSIM_HD Vec3 operator-() const { return {-x, -y, -z}; }

    ROOMSIM_HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    ROOMSIM_HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    ROOMSIM_HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    ROOMSIM_HD val_t norm() const { return sqrtf(squaredNorm()); }
    ROOMSIM_HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    ROOMSIM_HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    ROOMSIM_HD Triangle() = default;
    ROOMSIM_HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    ROOMSIM_HD Vec3 center() const { return (a + b + c) / 3.0f; }
    ROOMSIM_HD Vec3 normal() const { return _normal; }

    ROOMSIM_HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    ROOMSIM_HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

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
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
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
// CUDA implementation
// ============================================================================

// The host octree uses owning pointers and vectors, which are convenient while
// building it but cannot be traversed by a device kernel.  This is its compact,
// index-based representation for GPU traversal.
struct DeviceOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    uint32_t firstTriangle;
    uint32_t triangleCount;
};

constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_OCTREE_STACK_SIZE = 128;

[[noreturn]] void throwCudaError(cudaError_t error, const char* expression,
                                 const char* file, int line) {
    std::string message = "CUDA error from ";
    message += expression;
    message += " at ";
    message += file;
    message += ":";
    message += std::to_string(line);
    message += ": ";
    message += cudaGetErrorString(error);
    throw std::runtime_error(message);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        cudaError_t roomsimCudaError = (expression);                            \
        if (roomsimCudaError != cudaSuccess) {                                   \
            throwCudaError(roomsimCudaError, #expression, __FILE__, __LINE__);   \
        }                                                                        \
    } while (false)

template <typename T>
T* cudaAllocate(size_t count) {
    if (count == 0) return nullptr;
    T* result = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)));
    return result;
}

// Flattening needs child indices before recursion can append later siblings.
// This wrapper performs the same traversal while retaining each child's root
// index, avoiding pointers in device memory.
void flattenOctreeIndexed(const Octree& source,
                          std::vector<DeviceOctreeNode>& nodes,
                          std::vector<uint32_t>& leafTriangles) {
    const size_t nodeIndex = nodes.size();
    DeviceOctreeNode flat{};
    flat.center = source.center;
    flat.halfExtent = source.halfExtent;
    for (int i = 0; i < 8; ++i) flat.children[i] = -1;
    nodes.push_back(flat);

    if (!source.triangleIndices.empty()) {
        nodes[nodeIndex].firstTriangle = static_cast<uint32_t>(leafTriangles.size());
        nodes[nodeIndex].triangleCount = static_cast<uint32_t>(source.triangleIndices.size());
        for (size_t triangleIndex : source.triangleIndices) {
            leafTriangles.push_back(static_cast<uint32_t>(triangleIndex));
        }
        return;
    }

    for (int child = 0; child < 8; ++child) {
        if (!source.children[child]) continue;
        const size_t childIndex = nodes.size();
        nodes[nodeIndex].children[child] = static_cast<int>(childIndex);
        flattenOctreeIndexed(*source.children[child], nodes, leafTriangles);
    }
}

ROOMSIM_DEVICE __forceinline__ bool deviceRayIntersectsBox(
    const Vec3& p1, const Vec3& p2, const DeviceOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

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

ROOMSIM_DEVICE __forceinline__ val_t deviceRayTriangleIntersect(
    const Vec3& orig, const Vec3& dir, const Triangle& triangle) {
    Vec3 e1 = triangle.b - triangle.a;
    Vec3 e2 = triangle.c - triangle.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return 1.0e30f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - triangle.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1.0e30f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1.0e30f;
    return e2.dot(qvec) * invDet;
}

ROOMSIM_DEVICE bool deviceRayBlocked(
    const Vec3& from, const Vec3& to, const Triangle* triangles,
    const DeviceOctreeNode* nodes, const uint32_t* leafTriangles,
    uint32_t sourceIndex, uint32_t destinationIndex) {
    Vec3 direction = to - from;
    val_t rayLength = direction.norm();
    if (rayLength < EPSILON) return true;
    Vec3 directionNormalized = direction / rayLength;

    int stack[CUDA_OCTREE_STACK_SIZE];
    int stackSize = 1;
    stack[0] = 0;

    while (stackSize != 0) {
        const DeviceOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount != 0) {
            for (uint32_t k = 0; k < node.triangleCount; ++k) {
                uint32_t triangleIndex = leafTriangles[node.firstTriangle + k];
                if (triangleIndex == sourceIndex || triangleIndex == destinationIndex) continue;

                val_t distance = deviceRayTriangleIntersect(
                    from, directionNormalized, triangles[triangleIndex]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }

        // Push in reverse so the stack visits children in the same order as
        // the original recursive traversal (0 through 7).
        for (int child = 7; child >= 0; --child) {
            int childIndex = node.children[child];
            if (childIndex >= 0 &&
                deviceRayIntersectsBox(from, to, nodes[childIndex])) {
                if (stackSize < CUDA_OCTREE_STACK_SIZE) stack[stackSize++] = childIndex;
            }
        }
    }
    return false;
}

ROOMSIM_DEVICE __forceinline__ uint32_t nextPairRandom(uint32_t& state) {
    // A private counter-based stream removes the sequential RNG bottleneck
    // while retaining the same uniform barycentric sampling semantics.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

ROOMSIM_DEVICE __forceinline__ val_t pairRandom01(uint32_t& state) {
    return static_cast<val_t>(nextPairRandom(state)) * (1.0f / 4294967296.0f);
}

ROOMSIM_DEVICE Vec3 deviceRandomPointInTriangle(
    const Triangle& triangle, uint32_t& randomState) {
    val_t u = pairRandom01(randomState);
    val_t v = pairRandom01(randomState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u +
           (triangle.c - triangle.a) * v;
}

ROOMSIM_DEVICE __forceinline__ val_t deviceCosPhi(const Vec3& vector,
                                                   const Vec3& normal) {
    val_t vectorNorm = vector.norm();
    if (vectorNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, vector.dot(normal) / vectorNorm);
}

ROOMSIM_DEVICE val_t deviceComputeKij(
    uint32_t sourceIndex, uint32_t destinationIndex, const Triangle* triangles,
    const DeviceOctreeNode* nodes, const uint32_t* leafTriangles,
    uint64_t pairIndex) {
    const Triangle& receiver = triangles[sourceIndex];
    const Triangle& emitter = triangles[destinationIndex];

    if (receiver.normal().dot(emitter.normal()) > 0.99f) return ZERO;

    uint32_t randomState = static_cast<uint32_t>(pairIndex) ^
                           static_cast<uint32_t>(pairIndex >> 32) ^ 42u;
    randomState ^= randomState >> 16;
    randomState *= 0x7feb352du;
    randomState ^= randomState >> 15;
    randomState *= 0x846ca68bu;
    randomState ^= randomState >> 16;
    if (randomState == 0) randomState = 0x6d2b79f5u;

    val_t formFactor = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        Vec3 pointReceiver = deviceRandomPointInTriangle(receiver, randomState);
        Vec3 pointEmitter = deviceRandomPointInTriangle(emitter, randomState);

        if (deviceRayBlocked(pointReceiver, pointEmitter, triangles, nodes,
                             leafTriangles, sourceIndex, destinationIndex)) continue;

        Vec3 vector = pointEmitter - pointReceiver;
        val_t distanceSquared = vector.squaredNorm();
        if (distanceSquared < EPSILON) continue;

        val_t cosReceiver = deviceCosPhi(vector, receiver.normal());
        val_t cosEmitter = deviceCosPhi(-vector, emitter.normal());
        if (cosReceiver <= ZERO || cosEmitter <= ZERO) continue;

        formFactor += (cosReceiver * cosEmitter) / (PI * distanceSquared);
    }
    return formFactor * INV_NUM_RAYS;
}

__global__ void formFactorKernel(
    const Triangle* triangles, uint32_t triangleCount,
    const DeviceOctreeNode* nodes, const uint32_t* leafTriangles,
    val_t* formFactors) {
    const uint64_t totalPairs = static_cast<uint64_t>(triangleCount) * triangleCount;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t pair = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < totalPairs; pair += stride) {
        uint32_t receiver = static_cast<uint32_t>(pair / triangleCount);
        uint32_t emitter = static_cast<uint32_t>(pair % triangleCount);
        formFactors[pair] = receiver == emitter
            ? ZERO
            : deviceComputeKij(receiver, emitter, triangles, nodes,
                               leafTriangles, pair);
    }
}

__global__ void timeDelayKernel(const Vec3* centers, uint32_t triangleCount,
                                int* timeDelays) {
    const uint64_t totalPairs = static_cast<uint64_t>(triangleCount) * triangleCount;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t pair = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < totalPairs; pair += stride) {
        uint32_t receiver = static_cast<uint32_t>(pair / triangleCount);
        uint32_t emitter = static_cast<uint32_t>(pair % triangleCount);
        if (receiver == emitter) {
            timeDelays[pair] = 0;
            continue;
        }
        timeDelays[pair] = static_cast<int>(ceilf(
            (centers[receiver] - centers[emitter]).norm() * INV_WAVE_SPEED));
    }
}

__global__ void radiosityKernel(
    const val_t* formFactors, const int* timeDelays, const val_t* areas,
    const val_t* reflectivity, const val_t* emission, val_t* reflected,
    uint32_t triangleCount, uint32_t timestep) {
    const uint32_t receiver = blockIdx.x;
    if (receiver >= triangleCount) return;

    val_t sum = ZERO;
    for (uint32_t emitter = threadIdx.x; emitter < triangleCount;
         emitter += blockDim.x) {
        if (receiver == emitter) continue;
        const uint64_t pair = static_cast<uint64_t>(receiver) * triangleCount + emitter;
        const int tau = timeDelays[pair];
        if (static_cast<int>(timestep) < tau) continue;

        val_t formFactor = formFactors[pair];
        if (formFactor <= ZERO) continue;
        const uint64_t source = static_cast<uint64_t>(timestep - tau) * triangleCount + emitter;
        val_t sourceRadiosity = reflected[source];
        if (sourceRadiosity <= ZERO) continue;
        sum += fminf(formFactor * areas[emitter], ONE) * sourceRadiosity;
    }

    __shared__ val_t partialSums[CUDA_BLOCK_SIZE];
    partialSums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        if (threadIdx.x < offset) partialSums[threadIdx.x] += partialSums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        reflected[static_cast<uint64_t>(timestep) * triangleCount + receiver] =
            reflectivity[receiver] * partialSums[0] +
            emission[static_cast<uint64_t>(timestep) * triangleCount + receiver];
    }
}

__global__ void distanceKernel(const val_t* reflected, uint32_t triangleCount,
                               uint32_t timestepCount, uint32_t sourceIndex,
                               val_t* distances) {
    const uint32_t triangle = blockIdx.x;
    if (triangle >= triangleCount) return;

    val_t bestCorrelation = ZERO;
    int bestTimestep = 0;
    for (uint32_t delay = threadIdx.x; delay < timestepCount;
         delay += blockDim.x) {
        val_t correlation = ZERO;
        for (uint32_t time = delay; time < timestepCount; ++time) {
            const uint64_t receiver = static_cast<uint64_t>(time) * triangleCount + triangle;
            const uint64_t source = static_cast<uint64_t>(time - delay) * triangleCount + sourceIndex;
            correlation += reflected[source] * reflected[receiver];
        }
        if (correlation > bestCorrelation ||
            (correlation == bestCorrelation && delay < static_cast<uint32_t>(bestTimestep))) {
            bestCorrelation = correlation;
            bestTimestep = static_cast<int>(delay);
        }
    }

    __shared__ val_t partialCorrelations[CUDA_BLOCK_SIZE];
    __shared__ int partialBestTimesteps[CUDA_BLOCK_SIZE];
    partialCorrelations[threadIdx.x] = bestCorrelation;
    partialBestTimesteps[threadIdx.x] = bestTimestep;
    __syncthreads();

    for (uint32_t offset = blockDim.x / 2; offset != 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            val_t otherCorrelation = partialCorrelations[threadIdx.x + offset];
            int otherTimestep = partialBestTimesteps[threadIdx.x + offset];
            if (otherCorrelation > partialCorrelations[threadIdx.x] ||
                (otherCorrelation == partialCorrelations[threadIdx.x] &&
                 otherTimestep < partialBestTimesteps[threadIdx.x])) {
                partialCorrelations[threadIdx.x] = otherCorrelation;
                partialBestTimesteps[threadIdx.x] = otherTimestep;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[triangle] = WAVE_SPEED * static_cast<val_t>(partialBestTimesteps[0]);
    }
}

class CudaSimulation {
public:
    explicit CudaSimulation(const SimulationState& state)
        : triangleCount_(state.numTriangles), timestepCount_(state.numTimesteps) {
        if (triangleCount_ == 0) {
            throw std::runtime_error("roomsim requires a non-empty mesh");
        }
        if (triangleCount_ > std::numeric_limits<uint32_t>::max() ||
            timestepCount_ > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("roomsim dimensions exceed CUDA kernel index limits");
        }

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) throw std::runtime_error("roomsim requires a CUDA device");
        CUDA_CHECK(cudaSetDevice(0));

        std::vector<DeviceOctreeNode> nodes;
        std::vector<uint32_t> leafTriangles;
        nodes.reserve(triangleCount_);
        leafTriangles.reserve(triangleCount_);
        flattenOctreeIndexed(state.octree, nodes, leafTriangles);

        std::vector<Vec3> centers(triangleCount_);
        for (size_t i = 0; i < triangleCount_; ++i) centers[i] = state.triangles[i].center();

        dTriangles_ = cudaAllocate<Triangle>(triangleCount_);
        dCenters_ = cudaAllocate<Vec3>(triangleCount_);
        dNodes_ = cudaAllocate<DeviceOctreeNode>(nodes.size());
        dLeafTriangles_ = cudaAllocate<uint32_t>(leafTriangles.size());
        dAreas_ = cudaAllocate<val_t>(triangleCount_);
        dReflectivity_ = cudaAllocate<val_t>(triangleCount_);
        dFormFactors_ = cudaAllocate<val_t>(triangleCount_ * triangleCount_);
        dTimeDelays_ = cudaAllocate<int>(triangleCount_ * triangleCount_);
        dEmission_ = cudaAllocate<val_t>(timestepCount_ * triangleCount_);
        dReflected_ = cudaAllocate<val_t>(timestepCount_ * triangleCount_);
        dDistances_ = cudaAllocate<val_t>(triangleCount_);

        CUDA_CHECK(cudaMemcpy(dTriangles_, state.triangles.data(),
                              triangleCount_ * sizeof(Triangle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCenters_, centers.data(),
                              triangleCount_ * sizeof(Vec3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dNodes_, nodes.data(),
                              nodes.size() * sizeof(DeviceOctreeNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dLeafTriangles_, leafTriangles.data(),
                              leafTriangles.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dAreas_, state.areas.data(),
                              triangleCount_ * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dReflectivity_, state.rho.data(),
                              triangleCount_ * sizeof(val_t), cudaMemcpyHostToDevice));
        if (timestepCount_ != 0) {
            CUDA_CHECK(cudaMemcpy(dEmission_, state.radE.data(),
                                  timestepCount_ * triangleCount_ * sizeof(val_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(dReflected_, 0,
                                  timestepCount_ * triangleCount_ * sizeof(val_t)));
        }
    }

    ~CudaSimulation() {
        cudaFree(dDistances_);
        cudaFree(dReflected_);
        cudaFree(dEmission_);
        cudaFree(dTimeDelays_);
        cudaFree(dFormFactors_);
        cudaFree(dReflectivity_);
        cudaFree(dAreas_);
        cudaFree(dLeafTriangles_);
        cudaFree(dNodes_);
        cudaFree(dCenters_);
        cudaFree(dTriangles_);
    }

    void computeTimeDelays(SimulationState& state) {
        const uint64_t pairCount = static_cast<uint64_t>(triangleCount_) * triangleCount_;
        const int blocks = launchBlocks(pairCount);
        timeDelayKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            dCenters_, static_cast<uint32_t>(triangleCount_), dTimeDelays_);
        synchronizeKernel();
        CUDA_CHECK(cudaMemcpy(state.tau.data(), dTimeDelays_,
                              pairCount * sizeof(int), cudaMemcpyDeviceToHost));
    }

    void computeFormFactors(SimulationState& state) {
        const uint64_t pairCount = static_cast<uint64_t>(triangleCount_) * triangleCount_;
        const int blocks = launchBlocks(pairCount);
        formFactorKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
            dTriangles_, static_cast<uint32_t>(triangleCount_), dNodes_,
            dLeafTriangles_, dFormFactors_);
        synchronizeKernel();
        CUDA_CHECK(cudaMemcpy(state.kij.data(), dFormFactors_,
                              pairCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

    void runSimulation(SimulationState& state) {
        for (uint32_t timestep = 0; timestep < timestepCount_; ++timestep) {
            radiosityKernel<<<static_cast<unsigned int>(triangleCount_), CUDA_BLOCK_SIZE>>>(
                dFormFactors_, dTimeDelays_, dAreas_, dReflectivity_, dEmission_,
                dReflected_, static_cast<uint32_t>(triangleCount_), timestep);
            CUDA_CHECK(cudaGetLastError());

            if ((timestep + 1) % 10 == 0 || timestep + 1 == timestepCount_) {
                printf("  Timestep %u/%zu\n", timestep + 1, timestepCount_);
            }
        }
        synchronizeKernel();
        if (timestepCount_ != 0) {
            CUDA_CHECK(cudaMemcpy(state.radB.data(), dReflected_,
                                  timestepCount_ * triangleCount_ * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        }
    }

    void computeDistances(SimulationState& state) {
        if (timestepCount_ == 0) {
            CUDA_CHECK(cudaMemset(dDistances_, 0, triangleCount_ * sizeof(val_t)));
            CUDA_CHECK(cudaMemcpy(state.distances.data(), dDistances_,
                                  triangleCount_ * sizeof(val_t), cudaMemcpyDeviceToHost));
            return;
        }
        distanceKernel<<<static_cast<unsigned int>(triangleCount_), CUDA_BLOCK_SIZE>>>(
            dReflected_, static_cast<uint32_t>(triangleCount_),
            static_cast<uint32_t>(timestepCount_),
            static_cast<uint32_t>(state.sourceIndex), dDistances_);
        synchronizeKernel();
        CUDA_CHECK(cudaMemcpy(state.distances.data(), dDistances_,
                              triangleCount_ * sizeof(val_t), cudaMemcpyDeviceToHost));
    }

private:
    static int launchBlocks(uint64_t elements) {
        const uint64_t needed = (elements + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        const uint64_t maxGrid = 2147483647ULL;
        return static_cast<int>(std::min(needed, maxGrid));
    }

    static void synchronizeKernel() {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t triangleCount_;
    size_t timestepCount_;
    Triangle* dTriangles_ = nullptr;
    Vec3* dCenters_ = nullptr;
    DeviceOctreeNode* dNodes_ = nullptr;
    uint32_t* dLeafTriangles_ = nullptr;
    val_t* dAreas_ = nullptr;
    val_t* dReflectivity_ = nullptr;
    val_t* dFormFactors_ = nullptr;
    int* dTimeDelays_ = nullptr;
    val_t* dEmission_ = nullptr;
    val_t* dReflected_ = nullptr;
    val_t* dDistances_ = nullptr;
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
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

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

void computeFormFactors(SimulationState& state, CudaSimulation& cuda) {
    printf("Computing form factors (Kij)...\n");
    cuda.computeFormFactors(state);
    printf("  Completed %zu/%zu pair rows on CUDA\n",
           state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state, CudaSimulation& cuda) {
    printf("Computing time delays (Tau)...\n");
    cuda.computeTimeDelays(state);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, CudaSimulation& cuda) {
    printf("Running wave propagation simulation...\n");
    cuda.runSimulation(state);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, CudaSimulation& cuda) {
    printf("Computing distances via cross-correlation...\n");
    cuda.computeDistances(state);
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
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();
    CudaSimulation cuda(state);

    computeTimeDelays(state, cuda);
    computeFormFactors(state, cuda);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, cuda);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, cuda);

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
