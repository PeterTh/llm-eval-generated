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
#include <cfloat>
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

#if defined(__CUDACC__)
#define ROOMSIM_HD __host__ __device__
#else
#define ROOMSIM_HD
#endif

MPI_Comm g_mpiComm = MPI_COMM_WORLD;
int g_mpiRank = 0;
int g_mpiSize = 1;

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

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    ROOMSIM_HD Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    ROOMSIM_HD Vec3 center() const { return (a + b + c) / 3.0f; }
    ROOMSIM_HD Vec3 normal() const { return _normal; }

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

// A compact, pointer-free representation used by the CUDA visibility kernel.
// The host octree remains the authoritative structure; flattening preserves
// its child and triangle traversal order.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    int triangleBegin;
    int triangleCount;
};

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
    void flatten(std::vector<FlatOctreeNode>& nodes,
                 std::vector<int>& leafTriangles) const {
        const int nodeId = static_cast<int>(nodes.size());
        FlatOctreeNode flat{};
        flat.center = center;
        flat.halfExtent = halfExtent;
        flat.triangleBegin = -1;
        flat.triangleCount = 0;
        for (int i = 0; i < 8; ++i) flat.children[i] = -1;
        nodes.push_back(flat);

        if (!triangleIndices.empty()) {
            nodes[nodeId].triangleBegin = static_cast<int>(leafTriangles.size());
            nodes[nodeId].triangleCount = static_cast<int>(triangleIndices.size());
            for (size_t idx : triangleIndices) {
                leafTriangles.push_back(static_cast<int>(idx));
            }
            return;
        }

        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                nodes[nodeId].children[i] = static_cast<int>(nodes.size());
                children[i]->flatten(nodes, leafTriangles);
            }
        }
    }

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
// CUDA kernels and accelerator context
// ============================================================================

inline void checkCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line,
                     expression, cudaGetErrorString(status));
        MPI_Abort(g_mpiComm, static_cast<int>(status));
        std::abort();
    }
}

#define ROOMSIM_CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

#if defined(__CUDACC__)

__device__ inline uint32_t deviceMix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ inline val_t deviceRandom(uint32_t i, uint32_t j,
                                     uint32_t ray, uint32_t component) {
    uint32_t seed = 42u;
    seed ^= deviceMix(i + 0x9e3779b9u);
    seed ^= deviceMix(j + 0x85ebca6bu);
    seed ^= deviceMix(ray + 0xc2b2ae35u);
    seed ^= deviceMix(component + 0x27d4eb2fu);
    return (static_cast<val_t>(deviceMix(seed)) + 1.0f) * 2.3283064365386963e-10f;
}

__device__ inline bool deviceRayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                              const FlatOctreeNode& node) {
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

__device__ inline val_t deviceRayTriangleIntersect(const Vec3& orig,
                                                   const Vec3& dir,
                                                   const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ inline bool deviceRayBlocked(const Vec3& from, const Vec3& to,
                                        const Triangle* triangles,
                                        const FlatOctreeNode* nodes,
                                        const int* leafTriangles,
                                        int nodeCount, int srcTriIdx,
                                        int dstTriIdx) {
    Vec3 direction = to - from;
    val_t rayLength = sqrtf(direction.squaredNorm());
    if (rayLength < EPSILON) return true;
    Vec3 directionNormalized = direction / rayLength;

    // The generated octree is shallow (and has at most eight children per
    // level), so a private fixed stack avoids device-side allocation.
    int stack[96];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize != 0) {
        const int nodeId = stack[--stackSize];
        const FlatOctreeNode& node = nodes[nodeId];
        if (!deviceRayIntersectsBox(from, to, node)) continue;

        if (node.triangleCount != 0) {
            for (int k = 0; k < node.triangleCount; ++k) {
                const int idx = leafTriangles[node.triangleBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const val_t distance = deviceRayTriangleIntersect(
                    from, directionNormalized, triangles[idx]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }

        for (int child = 7; child >= 0; --child) {
            const int childId = node.children[child];
            if (childId >= 0 && stackSize < 96) stack[stackSize++] = childId;
        }
    }
    return false;
}

__device__ inline Vec3 deviceRandomPoint(const Triangle& tri, uint32_t i,
                                          uint32_t j, uint32_t ray,
                                          uint32_t component) {
    val_t u = deviceRandom(i, j, ray, component);
    val_t v = deviceRandom(i, j, ray, component + 1u);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__device__ inline val_t deviceCosPhi(const Vec3& v, const Vec3& normal) {
    val_t norm = sqrtf(v.squaredNorm());
    if (norm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / norm);
}

__device__ val_t deviceComputeKij(int i, int j, const Triangle* triangles,
                                  const FlatOctreeNode* nodes,
                                  const int* leafTriangles, int nodeCount) {
    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;
    for (uint32_t ray = 0; ray < NUM_RAYS; ++ray) {
        const Vec3 pI = deviceRandomPoint(triI, static_cast<uint32_t>(i),
                                          static_cast<uint32_t>(j), ray, 0u);
        const Vec3 pJ = deviceRandomPoint(triJ, static_cast<uint32_t>(i),
                                          static_cast<uint32_t>(j), ray, 2u);
        if (deviceRayBlocked(pI, pJ, triangles, nodes, leafTriangles,
                             nodeCount, i, j)) continue;

        const Vec3 v = pJ - pI;
        const val_t distanceSquared = v.squaredNorm();
        if (distanceSquared < EPSILON) continue;

        const val_t cosI = deviceCosPhi(v, triI.normal());
        const val_t cosJ = deviceCosPhi(-v, triJ.normal());
        if (cosI <= ZERO || cosJ <= ZERO) continue;
        kij += (cosI * cosJ) / (PI * distanceSquared);
    }
    return kij * INV_NUM_RAYS;
}

__global__ void formFactorKernel(const Triangle* triangles,
                                 const FlatOctreeNode* nodes,
                                 const int* leafTriangles, int nodeCount,
                                 int numTriangles, int rowBegin, int rowCount,
                                 val_t* output) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rowCount) * numTriangles;
    if (linear >= total) return;
    const int i = rowBegin + static_cast<int>(linear / numTriangles);
    const int j = static_cast<int>(linear % numTriangles);
    output[linear] = (i == j) ? ZERO : deviceComputeKij(
        i, j, triangles, nodes, leafTriangles, nodeCount);
}

__global__ void timeDelayKernel(const Triangle* triangles, int numTriangles,
                                int rowBegin, int rowCount, int* output) {
    const size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rowCount) * numTriangles;
    if (linear >= total) return;
    const int i = rowBegin + static_cast<int>(linear / numTriangles);
    const int j = static_cast<int>(linear % numTriangles);
    if (i == j) {
        output[linear] = 0;
        return;
    }
    const Vec3 delta = triangles[i].center() - triangles[j].center();
    output[linear] = static_cast<int>(ceilf(sqrtf(delta.squaredNorm()) * INV_WAVE_SPEED));
}

__global__ void propagationKernel(const val_t* kij, const int* tau,
                                  const val_t* areas, const val_t* rho,
                                  val_t* radB, int timestep, int numTriangles,
                                  int rowBegin, int rowCount, int numTimesteps,
                                  int sourceIndex) {
    const int localI = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (localI >= rowCount) return;
    const int i = rowBegin + localI;
    val_t sumB = ZERO;
    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;
        const int delay = tau[static_cast<size_t>(localI) * numTriangles + j];
        if (timestep < delay) continue;
        const val_t formFactor = kij[static_cast<size_t>(localI) * numTriangles + j];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radB[
            (static_cast<size_t>(timestep - delay) * numTriangles) + j];
        if (sourceRadiosity <= ZERO) continue;
        sumB += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }
    const val_t emission = (i == sourceIndex && timestep < numTimesteps / 2) ? ONE : ZERO;
    radB[static_cast<size_t>(timestep) * numTriangles + i] = rho[i] * sumB + emission;
}

__global__ void distanceKernel(const val_t* radB, int numTriangles,
                               int numTimesteps, int rowBegin, int rowCount,
                               int sourceIndex, val_t* output) {
    const int localI = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (localI >= rowCount) return;
    const int i = rowBegin + localI;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            const val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            const val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    output[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
}

#endif // defined(__CUDACC__)

class CudaContext {
public:
    int rowBegin;
    int rowCount;
    int numTriangles;
    int numTimesteps;
    int nodeCount = 0;

    Triangle* d_triangles = nullptr;
    FlatOctreeNode* d_nodes = nullptr;
    int* d_leafTriangles = nullptr;
    val_t* d_localKij = nullptr;
    int* d_localTau = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

    CudaContext(const SimulationState& state, int begin, int count)
        : rowBegin(begin), rowCount(count),
          numTriangles(static_cast<int>(state.numTriangles)),
          numTimesteps(static_cast<int>(state.numTimesteps)) {
        std::vector<FlatOctreeNode> hostNodes;
        std::vector<int> hostLeafTriangles;
        state.octree.flatten(hostNodes, hostLeafTriangles);
        nodeCount = static_cast<int>(hostNodes.size());

        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles),
                                      state.triangles.size() * sizeof(Triangle)));
        ROOMSIM_CUDA_CHECK(cudaMemcpy(d_triangles, state.triangles.data(),
                                      state.triangles.size() * sizeof(Triangle),
                                      cudaMemcpyHostToDevice));
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nodes),
                                      hostNodes.size() * sizeof(FlatOctreeNode)));
        ROOMSIM_CUDA_CHECK(cudaMemcpy(d_nodes, hostNodes.data(),
                                      hostNodes.size() * sizeof(FlatOctreeNode),
                                      cudaMemcpyHostToDevice));
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_leafTriangles),
                                      hostLeafTriangles.size() * sizeof(int)));
        ROOMSIM_CUDA_CHECK(cudaMemcpy(d_leafTriangles, hostLeafTriangles.data(),
                                      hostLeafTriangles.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));

        const size_t localElements = static_cast<size_t>(rowCount) * numTriangles;
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localKij),
                                      std::max<size_t>(1, localElements) * sizeof(val_t)));
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_localTau),
                                      std::max<size_t>(1, localElements) * sizeof(int)));
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_areas),
                                      state.areas.size() * sizeof(val_t)));
        ROOMSIM_CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(),
                                      state.areas.size() * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rho),
                                      state.rho.size() * sizeof(val_t)));
        ROOMSIM_CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(),
                                      state.rho.size() * sizeof(val_t),
                                      cudaMemcpyHostToDevice));
    }

    ~CudaContext() {
        cudaFree(d_distances);
        cudaFree(d_radB);
        cudaFree(d_rho);
        cudaFree(d_areas);
        cudaFree(d_localTau);
        cudaFree(d_localKij);
        cudaFree(d_leafTriangles);
        cudaFree(d_nodes);
        cudaFree(d_triangles);
    }

    void computeFormFactors(std::vector<val_t>& output) {
        if (rowCount == 0) return;
        const size_t elements = static_cast<size_t>(rowCount) * numTriangles;
        const int blockSize = 256;
        const unsigned gridSize = static_cast<unsigned>(
            (elements + blockSize - 1) / blockSize);
        formFactorKernel<<<gridSize, blockSize>>>(
            d_triangles, d_nodes, d_leafTriangles, nodeCount, numTriangles,
            rowBegin, rowCount, d_localKij);
        ROOMSIM_CUDA_CHECK(cudaGetLastError());
        ROOMSIM_CUDA_CHECK(cudaMemcpy(output.data(), d_localKij,
                                      elements * sizeof(val_t),
                                      cudaMemcpyDeviceToHost));
    }

    void computeTimeDelays(std::vector<int>& output) {
        if (rowCount == 0) return;
        const size_t elements = static_cast<size_t>(rowCount) * numTriangles;
        const int blockSize = 256;
        const unsigned gridSize = static_cast<unsigned>(
            (elements + blockSize - 1) / blockSize);
        timeDelayKernel<<<gridSize, blockSize>>>(
            d_triangles, numTriangles, rowBegin, rowCount, d_localTau);
        ROOMSIM_CUDA_CHECK(cudaGetLastError());
        ROOMSIM_CUDA_CHECK(cudaMemcpy(output.data(), d_localTau,
                                      elements * sizeof(int),
                                      cudaMemcpyDeviceToHost));
    }

    void prepareSimulation() {
        // d_localKij and d_localTau were filled by the distributed CUDA
        // precomputation and remain resident on this rank's accelerator.
        const size_t responseElements = static_cast<size_t>(numTimesteps) * numTriangles;
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radB),
                                      std::max<size_t>(1, responseElements) * sizeof(val_t)));
        ROOMSIM_CUDA_CHECK(cudaMemset(d_radB, 0,
                                      std::max<size_t>(1, responseElements) * sizeof(val_t)));
    }

    void propagate(int timestep, int sourceIndex) {
        if (rowCount == 0) return;
        const int blockSize = 256;
        const unsigned gridSize = static_cast<unsigned>((rowCount + blockSize - 1) / blockSize);
        propagationKernel<<<gridSize, blockSize>>>(
            d_localKij, d_localTau, d_areas, d_rho, d_radB, timestep,
            numTriangles, rowBegin, rowCount, numTimesteps, sourceIndex);
        ROOMSIM_CUDA_CHECK(cudaGetLastError());
    }

    void computeDistances(std::vector<val_t>& output, int sourceIndex) {
        if (rowCount == 0) return;
        const int blockSize = 256;
        const unsigned gridSize = static_cast<unsigned>((rowCount + blockSize - 1) / blockSize);
        ROOMSIM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                                      std::max(1, rowCount) * sizeof(val_t)));
        distanceKernel<<<gridSize, blockSize>>>(
            d_radB, numTriangles, numTimesteps, rowBegin, rowCount,
            sourceIndex, d_distances);
        ROOMSIM_CUDA_CHECK(cudaGetLastError());
        ROOMSIM_CUDA_CHECK(cudaMemcpy(output.data(), d_distances,
                                      static_cast<size_t>(rowCount) * sizeof(val_t),
                                      cudaMemcpyDeviceToHost));
        cudaFree(d_distances);
        d_distances = nullptr;
    }
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

    if (g_mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (g_mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.rho[i] = reflectivity;
    }

    // Initialize matrices
    // Only rank zero needs complete host-side result arrays.  Other ranks
    // retain their local CUDA rows and use compact communication buffers.
    if (g_mpiRank == 0) {
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.tau.resize(state.numTriangles * state.numTriangles, 0);
        state.radE.resize(timesteps * state.numTriangles, ZERO);
        state.radB.resize(timesteps * state.numTriangles, ZERO);
        state.distances.resize(state.numTriangles, ZERO);
    }

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    if (g_mpiRank == 0) {
        #pragma omp parallel for schedule(static)
        for (long long t = static_cast<long long>(timeOn);
             t < static_cast<long long>(timeOff); ++t) {
            state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
        }
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

struct RowPartition {
    int begin = 0;
    int rows = 0;
    std::vector<int> counts;
    std::vector<int> displacements;
};

RowPartition makeRowPartition(size_t numRows, size_t rowWidth) {
    RowPartition partition;
    partition.counts.resize(g_mpiSize);
    partition.displacements.resize(g_mpiSize);
    for (int rank = 0; rank < g_mpiSize; ++rank) {
        const size_t first = (numRows * static_cast<size_t>(rank)) /
                             static_cast<size_t>(g_mpiSize);
        const size_t last = (numRows * static_cast<size_t>(rank + 1)) /
                            static_cast<size_t>(g_mpiSize);
        const size_t count = (last - first) * rowWidth;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            first * rowWidth > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (g_mpiRank == 0) {
                std::fprintf(stderr, "MPI row partition exceeds 32-bit MPI count limits\n");
            }
            MPI_Abort(g_mpiComm, 1);
        }
        partition.counts[rank] = static_cast<int>(count);
        partition.displacements[rank] = static_cast<int>(first * rowWidth);
        if (rank == g_mpiRank) {
            partition.begin = static_cast<int>(first);
            partition.rows = static_cast<int>(last - first);
        }
    }
    return partition;
}

void computeFormFactors(SimulationState& state, CudaContext& cuda,
                        const RowPartition& partition) {
    if (g_mpiRank == 0) printf("Computing form factors (Kij) on CUDA...\n");
    std::vector<val_t> local(static_cast<size_t>(partition.rows) * state.numTriangles);
    cuda.computeFormFactors(local);

    MPI_Gatherv(local.data(), partition.counts[g_mpiRank], MPI_FLOAT,
                state.kij.data(), partition.counts.data(),
                partition.displacements.data(), MPI_FLOAT, 0, g_mpiComm);
    if (g_mpiRank == 0) {
        printf("  Distributed %zu form-factor rows across %d MPI ranks\n",
               state.numTriangles, g_mpiSize);
    }
}

void computeTimeDelays(SimulationState& state, CudaContext& cuda,
                       const RowPartition& partition) {
    if (g_mpiRank == 0) printf("Computing time delays (Tau) on CUDA...\n");
    std::vector<int> local(static_cast<size_t>(partition.rows) * state.numTriangles);
    cuda.computeTimeDelays(local);

    MPI_Gatherv(local.data(), partition.counts[g_mpiRank], MPI_INT,
                state.tau.data(), partition.counts.data(),
                partition.displacements.data(), MPI_INT, 0, g_mpiComm);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, CudaContext& cuda) {
    if (g_mpiRank == 0) {
        printf("Running wave propagation simulation (CUDA + MPI row exchange)...\n");
    }

    cuda.prepareSimulation();
    RowPartition currentPartition = makeRowPartition(state.numTriangles, 1);
    std::vector<val_t> localCurrent(static_cast<size_t>(currentPartition.rows));
    std::vector<val_t> synchronizedRow(state.numTriangles);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        cuda.propagate(static_cast<int>(t), static_cast<int>(state.sourceIndex));

        if (currentPartition.rows != 0) {
            ROOMSIM_CUDA_CHECK(cudaMemcpy(
                localCurrent.data(),
                cuda.d_radB + static_cast<size_t>(t) * state.numTriangles + currentPartition.begin,
                static_cast<size_t>(currentPartition.rows) * sizeof(val_t),
                cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(localCurrent.data(), currentPartition.counts[g_mpiRank], MPI_FLOAT,
                       synchronizedRow.data(), currentPartition.counts.data(),
                       currentPartition.displacements.data(), MPI_FLOAT, g_mpiComm);

        if (g_mpiRank == 0) {
            std::copy(synchronizedRow.begin(), synchronizedRow.end(),
                      state.radB.begin() + t * state.numTriangles);
        }

        // Every rank needs the complete current time slice for later delays.
        ROOMSIM_CUDA_CHECK(cudaMemcpy(
            cuda.d_radB + t * state.numTriangles,
            synchronizedRow.data(),
            state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (g_mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, CudaContext& cuda) {
    if (g_mpiRank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    RowPartition distancePartition = makeRowPartition(state.numTriangles, 1);
    std::vector<val_t> local(static_cast<size_t>(distancePartition.rows));
    cuda.computeDistances(local, static_cast<int>(state.sourceIndex));

    MPI_Gatherv(local.data(), distancePartition.counts[g_mpiRank], MPI_FLOAT,
                state.distances.data(), distancePartition.counts.data(),
                distancePartition.displacements.data(), MPI_FLOAT, 0, g_mpiComm);
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
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);
    MPI_Comm_rank(g_mpiComm, &g_mpiRank);
    MPI_Comm_size(g_mpiComm, &g_mpiSize);
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(g_mpiComm, 1);
    }

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
            if (g_mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (timesteps < 0 || targetTriangles < 1) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "Triangle and timestep counts must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Bind each MPI process to a GPU local to its node.  MPI ranks share the
    // mesh and exchange only row blocks of the matrices and response slices.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(g_mpiComm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    ROOMSIM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (g_mpiRank == 0) std::fprintf(stderr, "No CUDA accelerator is available\n");
        MPI_Abort(g_mpiComm, 1);
    }
    const int deviceId = localRank % deviceCount;
    ROOMSIM_CUDA_CHECK(cudaSetDevice(deviceId));
    MPI_Comm_free(&localComm);

    omp_set_dynamic(0);
    if (g_mpiRank == 0) {
        printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA device %d/rank\n",
               g_mpiSize, omp_get_max_threads(), deviceId);
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (g_mpiRank == 0) printf("\n");

    const RowPartition matrixPartition = makeRowPartition(
        state.numTriangles, state.numTriangles);
    int exitCode = 0;
    {
        CudaContext cuda(state, matrixPartition.begin, matrixPartition.rows);

        // Precomputation: each rank owns rows, with CUDA evaluating pairs and
        // MPI assembling the complete matrices needed by every timestep.
        MPI_Barrier(g_mpiComm);
        const double preStart = MPI_Wtime();
        computeTimeDelays(state, cuda, matrixPartition);
        computeFormFactors(state, cuda, matrixPartition);
        MPI_Barrier(g_mpiComm);
        double localElapsed = MPI_Wtime() - preStart;
        double preElapsed = 0.0;
        MPI_Reduce(&localElapsed, &preElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, g_mpiComm);
        const long preDuration = g_mpiRank == 0 ? static_cast<long>(preElapsed * 1000.0) : 0;
        if (g_mpiRank == 0) {
            printf("Precomputation time: %ld ms\n\n", preDuration);
        }

        // Simulation: timestep dependencies are preserved; independent
        // receiver rows run on CUDA and are exchanged after each timestep.
        MPI_Barrier(g_mpiComm);
        const double simStart = MPI_Wtime();
        runSimulation(state, cuda);
        MPI_Barrier(g_mpiComm);
        localElapsed = MPI_Wtime() - simStart;
        double simElapsed = 0.0;
        MPI_Reduce(&localElapsed, &simElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, g_mpiComm);
        const long simDuration = g_mpiRank == 0 ? static_cast<long>(simElapsed * 1000.0) : 0;
        if (g_mpiRank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

        MPI_Barrier(g_mpiComm);
        const double distStart = MPI_Wtime();
        computeDistances(state, cuda);
        MPI_Barrier(g_mpiComm);
        localElapsed = MPI_Wtime() - distStart;
        double distElapsed = 0.0;
        MPI_Reduce(&localElapsed, &distElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, g_mpiComm);
        const long distDuration = g_mpiRank == 0 ? static_cast<long>(distElapsed * 1000.0) : 0;
        if (g_mpiRank == 0) printf("Distance computation time: %ld ms\n\n", distDuration);

        if (g_mpiRank == 0) {
            const long totalTime = preDuration + simDuration + distDuration;
            const size_t n = state.numTriangles;
            const size_t t = state.numTimesteps;
            const double kijOps = static_cast<double>(n) * n;
            const double simOps = static_cast<double>(n) * n * t;
            const double distOps = static_cast<double>(n) * t * t;

            printf("Total computation time: %ld ms\n", totalTime);
            printf("\nPerformance:\n");
            printf("  Triangles: %zu\n", n);
            printf("  Timesteps: %zu\n", t);
            printf("  Form factor computations: %.2e\n", kijOps);
            printf("  Simulation operations: %.2e\n", simOps);
            printf("  Distance computations: %.2e\n", distOps);
            printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

            const size_t memKij = n * n * sizeof(val_t);
            const size_t memTau = n * n * sizeof(int);
            const size_t memRad = 2 * t * n * sizeof(val_t);
            const size_t totalMem = memKij + memTau + memRad;
            printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

            const uint64_t resultHash = computeHash(state);
            printf("  Result hash: %016llX\n", static_cast<unsigned long long>(resultHash));
            printf("\n");

            if (printResults) {
                std::vector<double> distData(state.distances.begin(), state.distances.end());
                print_results(distData, "Distances");
            }
        }

        if (validate) {
            int valid = 1;
            if (g_mpiRank == 0) valid = validateResults(state) ? 1 : 0;
            MPI_Bcast(&valid, 1, MPI_INT, 0, g_mpiComm);
            if (!valid) exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
