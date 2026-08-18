/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI + OpenMP + CUDA room impulse-response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
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

#define HD __host__ __device__

[[noreturn]] void cudaAbort(cudaError_t error, const char* expression,
                            const char* file, int line) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d CUDA error at %s:%d while evaluating %s: %s\n",
                 rank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                        \
        if (cuda_check_error != cudaSuccess)                                      \
            cudaAbort(cuda_check_error, #expression, __FILE__, __LINE__);         \
    } while (false)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
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

    HD Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
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

// Pointer-free representation used by CUDA kernels.  Leaf triangle references
// retain the exact ordering of the host octree traversal.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    uint32_t triangleOffset;
    uint32_t triangleCount;
    uint32_t subtreeEnd;
};

int flattenOctreeNode(const Octree& source, std::vector<FlatOctreeNode>& nodes,
                      std::vector<uint32_t>& triangleRefs) {
    const int nodeIndex = static_cast<int>(nodes.size());
    FlatOctreeNode flat{};
    flat.center = source.center;
    flat.halfExtent = source.halfExtent;
    flat.triangleOffset = static_cast<uint32_t>(triangleRefs.size());
    flat.triangleCount = static_cast<uint32_t>(source.triangleIndices.size());
    for (size_t triangleIndex : source.triangleIndices) {
        triangleRefs.push_back(static_cast<uint32_t>(triangleIndex));
    }
    nodes.push_back(flat);

    for (int child = 0; child < 8; ++child) {
        if (source.children[child]) {
            flattenOctreeNode(*source.children[child], nodes, triangleRefs);
        }
    }
    nodes[static_cast<size_t>(nodeIndex)].subtreeEnd =
        static_cast<uint32_t>(nodes.size());
    return nodeIndex;
}

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
// CUDA kernels
// ============================================================================

__device__ uint64_t mixBits(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ val_t randomUnit(uint64_t pairIndex, int ray, int component) {
    const uint64_t bits = mixBits(42ULL ^ (pairIndex * 0xd2b74407b1ce6e93ULL) ^
                                  (static_cast<uint64_t>(ray) << 16) ^
                                  static_cast<uint64_t>(component));
    return static_cast<val_t>((bits >> 40) + 0.5) * (1.0f / 16777216.0f);
}

__device__ Vec3 randomPointInTriangleDevice(const Triangle& triangle,
                                             uint64_t pairIndex, int ray,
                                             int componentOffset) {
    val_t u = randomUnit(pairIndex, ray, componentOffset);
    val_t v = randomUnit(pairIndex, ray, componentOffset + 1);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u +
           (triangle.c - triangle.a) * v;
}

__device__ bool rayIntersectsBoxDevice(const Vec3& p1, const Vec3& p2,
                                        const FlatOctreeNode& node) {
    const Vec3 d = (p2 - p1) * 0.5f;
    const Vec3 c = p1 + d - node.center;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));

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

__device__ val_t rayTriangleIntersectDevice(const Vec3& origin, const Vec3& direction,
                                             const Triangle& triangle) {
    const Vec3 e1 = triangle.b - triangle.a;
    const Vec3 e2 = triangle.c - triangle.a;
    const Vec3 pvec = direction.cross(e2);
    const val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    const val_t invDet = ONE / det;
    const Vec3 tvec = origin - triangle.a;
    const val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;
    const Vec3 qvec = tvec.cross(e1);
    const val_t v = direction.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                                    const Triangle* triangles,
                                    const FlatOctreeNode* nodes,
                                    const uint32_t* triangleRefs,
                                    uint32_t nodeCount, uint32_t srcTriangle,
                                    uint32_t dstTriangle) {
    const Vec3 ray = to - from;
    const val_t rayLength = ray.norm();
    if (rayLength < EPSILON) return true;
    const Vec3 direction = ray / rayLength;

    // Nodes are stored in depth-first order.  subtreeEnd makes traversal
    // stackless: a rejected box skips its complete subtree.
    uint32_t nodeIndex = 0;
    while (nodeIndex < nodeCount) {
        const FlatOctreeNode& node = nodes[nodeIndex];
        if (!rayIntersectsBoxDevice(from, to, node)) {
            nodeIndex = node.subtreeEnd;
            continue;
        }
        for (uint32_t ref = 0; ref < node.triangleCount; ++ref) {
            const uint32_t triangleIndex = triangleRefs[node.triangleOffset + ref];
            if (triangleIndex == srcTriangle || triangleIndex == dstTriangle) continue;
            const val_t distance = rayTriangleIntersectDevice(
                from, direction, triangles[triangleIndex]);
            if (distance > EPSILON && distance < rayLength - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

__global__ void computeTimeDelaysKernel(const Triangle* triangles, int* tau,
                                        size_t numTriangles, size_t localBegin,
                                        size_t pairCount) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= pairCount) return;
    const size_t localI = pair / numTriangles;
    const size_t j = pair - localI * numTriangles;
    const size_t i = localBegin + localI;
    if (i == j) {
        tau[pair] = 0;
        return;
    }
    const val_t distance = (triangles[i].center() - triangles[j].center()).norm();
    tau[pair] = static_cast<int>(ceilf(distance * INV_WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(const Triangle* triangles,
                                         const FlatOctreeNode* nodes,
                                         const uint32_t* triangleRefs,
                                         uint32_t nodeCount, val_t* kij,
                                         size_t numTriangles, size_t localBegin,
                                         size_t pairCount) {
    constexpr unsigned RAY_GROUP_SIZE = NUM_RAYS;
    const size_t linearThread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pair = linearThread / RAY_GROUP_SIZE;
    const unsigned ray = static_cast<unsigned>(linearThread % RAY_GROUP_SIZE);
    if (pair >= pairCount) return;

    const size_t localI = pair / numTriangles;
    const size_t j = pair - localI * numTriangles;
    const size_t i = localBegin + localI;
    const Triangle& triangleI = triangles[i];
    const Triangle& triangleJ = triangles[j];
    val_t sample = ZERO;

    if (i != j && triangleI.normal().dot(triangleJ.normal()) <= 0.99f) {
        const uint64_t globalPair = static_cast<uint64_t>(i) * numTriangles + j;
        const Vec3 pointI = randomPointInTriangleDevice(triangleI, globalPair,
                                                        static_cast<int>(ray), 0);
        const Vec3 pointJ = randomPointInTriangleDevice(triangleJ, globalPair,
                                                        static_cast<int>(ray), 2);
        if (!isRayBlockedDevice(pointI, pointJ, triangles, nodes, triangleRefs,
                                nodeCount, static_cast<uint32_t>(i),
                                static_cast<uint32_t>(j))) {
            const Vec3 direction = pointJ - pointI;
            const val_t distanceSquared = direction.squaredNorm();
            if (distanceSquared >= EPSILON) {
                const val_t inverseDistance = rsqrtf(distanceSquared);
                const val_t cosineI = fmaxf(ZERO,
                    direction.dot(triangleI.normal()) * inverseDistance);
                const val_t cosineJ = fmaxf(ZERO,
                    (-direction).dot(triangleJ.normal()) * inverseDistance);
                if (cosineI > ZERO && cosineJ > ZERO) {
                    sample = (cosineI * cosineJ) / (PI * distanceSquared);
                }
            }
        }
    }

    const unsigned laneInWarp = threadIdx.x & 31U;
    const unsigned groupBase = laneInWarp & ~15U;
    const unsigned mask = 0x0000ffffU << groupBase;
    for (unsigned offset = RAY_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        sample += __shfl_down_sync(mask, sample, offset, RAY_GROUP_SIZE);
    }
    if (ray == 0) kij[pair] = sample * INV_NUM_RAYS;
}

__global__ void propagateKernel(const val_t* kij, const int* tau,
                                const val_t* areas, val_t* radiosity,
                                size_t numTriangles, size_t numTimesteps,
                                size_t localBegin, size_t localCount,
                                size_t timestep, size_t sourceIndex,
                                val_t reflectivity) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localCount) return;
    const size_t i = localBegin + localI;
    val_t sum = ZERO;
    const size_t matrixRow = localI * numTriangles;

    for (size_t j = 0; j < numTriangles; ++j) {
        if (i == j) continue;
        const int delay = tau[matrixRow + j];
        if (timestep < static_cast<size_t>(delay)) continue;
        const val_t formFactor = kij[matrixRow + j];
        if (formFactor <= ZERO) continue;
        const val_t previous = radiosity[(timestep - static_cast<size_t>(delay)) *
                                         numTriangles + j];
        if (previous <= ZERO) continue;
        sum += fminf(formFactor * areas[j], ONE) * previous;
    }

    const val_t emission = (i == sourceIndex && timestep < numTimesteps / 2) ? ONE : ZERO;
    radiosity[timestep * numTriangles + i] = reflectivity * sum + emission;
}

__global__ void computeDistancesKernel(const val_t* radiosity, val_t* distances,
                                       size_t numTriangles, size_t numTimesteps,
                                       size_t localBegin, size_t localCount,
                                       size_t sourceIndex) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localCount) return;
    const size_t i = localBegin + localI;
    val_t maximumCorrelation = ZERO;
    size_t bestTimestep = 0;
    for (size_t lag = 0; lag < numTimesteps; ++lag) {
        val_t sum = ZERO;
        for (size_t t = lag; t < numTimesteps; ++t) {
            sum += radiosity[(t - lag) * numTriangles + sourceIndex] *
                   radiosity[t * numTriangles + i];
        }
        if (sum > maximumCorrelation) {
            maximumCorrelation = sum;
            bestTimestep = lag;
        }
    }
    distances[localI] = WAVE_SPEED * static_cast<val_t>(bestTimestep);
}

__global__ void countFormFactorsKernel(const val_t* kij, size_t count,
                                       unsigned long long* nonzero) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && kij[index] > EPSILON) atomicAdd(nonzero, 1ULL);
}

__global__ void countEnergyKernel(const val_t* radiosity, size_t numTriangles,
                                  size_t numTimesteps, size_t localBegin,
                                  size_t localCount, unsigned long long* received) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localCount) return;
    const size_t i = localBegin + localI;
    for (size_t t = 0; t < numTimesteps; ++t) {
        if (radiosity[t * numTriangles + i] > EPSILON) {
            atomicAdd(received, 1ULL);
            return;
        }
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> distances;   // Computed distances from source

    std::vector<FlatOctreeNode> flatNodes;
    std::vector<uint32_t> flatTriangleRefs;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    int rank = 0;
    int worldSize = 1;
    int device = 0;
    size_t localBegin = 0;
    size_t localCount = 0;
    val_t reflectivity = 0.8f;

    Triangle* deviceTriangles = nullptr;
    FlatOctreeNode* deviceNodes = nullptr;
    uint32_t* deviceTriangleRefs = nullptr;
    val_t* deviceAreas = nullptr;
    val_t* deviceKij = nullptr;
    int* deviceTau = nullptr;
    val_t* deviceRadB = nullptr;
    val_t* deviceDistances = nullptr;
    val_t* hostLocalRow = nullptr;
    val_t* hostGlobalRow = nullptr;
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
    state.reflectivity = reflectivity;
    state.localBegin = state.numTriangles * static_cast<size_t>(state.rank) /
                       static_cast<size_t>(state.worldSize);
    const size_t localEnd = state.numTriangles * static_cast<size_t>(state.rank + 1) /
                            static_cast<size_t>(state.worldSize);
    state.localCount = localEnd - state.localBegin;

    if (state.rank == 0)
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctreeNode(state.octree, state.flatNodes, state.flatTriangleRefs);

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i)
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();

    const size_t pairCount = state.localCount * state.numTriangles;
    const size_t historyCount = state.numTimesteps * state.numTriangles;
    const size_t localAllocation = std::max<size_t>(state.localCount, 1);
    const size_t pairAllocation = std::max<size_t>(pairCount, 1);
    CUDA_CHECK(cudaMalloc(&state.deviceTriangles,
                          state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.deviceNodes,
                          state.flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&state.deviceTriangleRefs,
                          state.flatTriangleRefs.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&state.deviceAreas, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.deviceKij, pairAllocation * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.deviceTau, pairAllocation * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.deviceRadB,
                          std::max<size_t>(historyCount, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.deviceDistances, localAllocation * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.deviceTriangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.deviceNodes, state.flatNodes.data(),
                          state.flatNodes.size() * sizeof(FlatOctreeNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.deviceTriangleRefs, state.flatTriangleRefs.data(),
                          state.flatTriangleRefs.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.deviceAreas, state.areas.data(),
                          state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.deviceKij, 0, pairAllocation * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.deviceTau, 0, pairAllocation * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.deviceRadB, 0,
                          std::max<size_t>(historyCount, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&state.hostLocalRow, localAllocation * sizeof(val_t)));
    CUDA_CHECK(cudaMallocHost(&state.hostGlobalRow,
                              state.numTriangles * sizeof(val_t)));
}

void releaseSimulation(SimulationState& state) {
    CUDA_CHECK(cudaFreeHost(state.hostGlobalRow));
    CUDA_CHECK(cudaFreeHost(state.hostLocalRow));
    CUDA_CHECK(cudaFree(state.deviceDistances));
    CUDA_CHECK(cudaFree(state.deviceRadB));
    CUDA_CHECK(cudaFree(state.deviceTau));
    CUDA_CHECK(cudaFree(state.deviceKij));
    CUDA_CHECK(cudaFree(state.deviceAreas));
    CUDA_CHECK(cudaFree(state.deviceTriangleRefs));
    CUDA_CHECK(cudaFree(state.deviceNodes));
    CUDA_CHECK(cudaFree(state.deviceTriangles));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij) on CUDA devices...\n");
    const size_t pairCount = state.localCount * state.numTriangles;
    if (pairCount == 0) return;
    constexpr unsigned threads = 256;
    const size_t rayThreads = pairCount * static_cast<size_t>(NUM_RAYS);
    const size_t blocks = (rayThreads + threads - 1) / threads;
    computeFormFactorsKernel<<<static_cast<unsigned>(blocks), threads>>>(
        state.deviceTriangles, state.deviceNodes, state.deviceTriangleRefs,
        static_cast<uint32_t>(state.flatNodes.size()), state.deviceKij,
        state.numTriangles, state.localBegin, pairCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau) on CUDA devices...\n");
    const size_t pairCount = state.localCount * state.numTriangles;
    if (pairCount == 0) return;
    constexpr unsigned threads = 256;
    const size_t blocks = (pairCount + threads - 1) / threads;
    computeTimeDelaysKernel<<<static_cast<unsigned>(blocks), threads>>>(
        state.deviceTriangles, state.deviceTau, state.numTriangles,
        state.localBegin, pairCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running distributed wave propagation simulation...\n");
    std::vector<int> receiveCounts(static_cast<size_t>(state.worldSize));
    std::vector<int> receiveDisplacements(static_cast<size_t>(state.worldSize));
    for (int rank = 0; rank < state.worldSize; ++rank) {
        const size_t begin = state.numTriangles * static_cast<size_t>(rank) /
                             static_cast<size_t>(state.worldSize);
        const size_t end = state.numTriangles * static_cast<size_t>(rank + 1) /
                           static_cast<size_t>(state.worldSize);
        receiveCounts[static_cast<size_t>(rank)] = static_cast<int>(end - begin);
        receiveDisplacements[static_cast<size_t>(rank)] = static_cast<int>(begin);
    }

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((state.localCount + threads - 1) / threads);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localCount != 0) {
            propagateKernel<<<blocks, threads>>>(
                state.deviceKij, state.deviceTau, state.deviceAreas,
                state.deviceRadB, state.numTriangles, state.numTimesteps,
                state.localBegin, state.localCount, t, state.sourceIndex,
                state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(
                state.hostLocalRow,
                state.deviceRadB + t * state.numTriangles + state.localBegin,
                state.localCount * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(state.hostLocalRow, static_cast<int>(state.localCount), MPI_FLOAT,
                       state.hostGlobalRow, receiveCounts.data(),
                       receiveDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(state.deviceRadB + t * state.numTriangles,
                              state.hostGlobalRow,
                              state.numTriangles * sizeof(val_t),
                              cudaMemcpyHostToDevice));

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    constexpr unsigned threads = 256;
    if (state.localCount != 0) {
        const unsigned blocks = static_cast<unsigned>((state.localCount + threads - 1) /
                                                       threads);
        computeDistancesKernel<<<blocks, threads>>>(
            state.deviceRadB, state.deviceDistances, state.numTriangles,
            state.numTimesteps, state.localBegin, state.localCount,
            state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.hostLocalRow, state.deviceDistances,
                              state.localCount * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<int> receiveCounts;
    std::vector<int> receiveDisplacements;
    if (state.rank == 0) {
        state.distances.resize(state.numTriangles);
        receiveCounts.resize(static_cast<size_t>(state.worldSize));
        receiveDisplacements.resize(static_cast<size_t>(state.worldSize));
        for (int rank = 0; rank < state.worldSize; ++rank) {
            const size_t begin = state.numTriangles * static_cast<size_t>(rank) /
                                 static_cast<size_t>(state.worldSize);
            const size_t end = state.numTriangles * static_cast<size_t>(rank + 1) /
                               static_cast<size_t>(state.worldSize);
            receiveCounts[static_cast<size_t>(rank)] = static_cast<int>(end - begin);
            receiveDisplacements[static_cast<size_t>(rank)] = static_cast<int>(begin);
        }
    }
    MPI_Gatherv(state.hostLocalRow, static_cast<int>(state.localCount), MPI_FLOAT,
                state.rank == 0 ? state.distances.data() : nullptr,
                state.rank == 0 ? receiveCounts.data() : nullptr,
                state.rank == 0 ? receiveDisplacements.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    bool valid = true;
    if (state.rank == 0) printf("\nValidation:\n");

    // Check that distances are non-negative
    int allNonNegative = 1;
    int allFinite = 1;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    long long nonZeroCount = 0;

    if (state.rank == 0) {
#pragma omp parallel for reduction(min:minDist) reduction(max:maxDist) \
    reduction(+:sumDist,nonZeroCount) reduction(&:allNonNegative,allFinite) schedule(static)
        for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
            const val_t distance = state.distances[static_cast<size_t>(i)];
            allNonNegative &= distance >= ZERO;
            allFinite &= std::isfinite(distance);
            minDist = std::min(minDist, distance);
            maxDist = std::max(maxDist, distance);
            sumDist += distance;
            nonZeroCount += distance > EPSILON;
        }
        if (!allNonNegative) printf("  ERROR: At least one distance is negative\n");
        if (!allFinite) printf("  ERROR: At least one distance is non-finite\n");
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %lld/%zu\n", nonZeroCount, state.numTriangles);

        const val_t srcDist = state.distances[state.sourceIndex];
        if (srcDist > WAVE_SPEED * 2)
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
        valid = allNonNegative && allFinite;
    }

    unsigned long long* deviceCount = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceCount, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(deviceCount, 0, sizeof(unsigned long long)));
    constexpr unsigned threads = 256;
    const size_t pairCount = state.localCount * state.numTriangles;
    if (pairCount != 0) {
        const unsigned blocks = static_cast<unsigned>((pairCount + threads - 1) / threads);
        countFormFactorsKernel<<<blocks, threads>>>(state.deviceKij, pairCount, deviceCount);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long localNonzeroKij = 0;
    CUDA_CHECK(cudaMemcpy(&localNonzeroKij, deviceCount, sizeof(localNonzeroKij),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(deviceCount, 0, sizeof(unsigned long long)));
    if (state.localCount != 0) {
        const unsigned blocks = static_cast<unsigned>((state.localCount + threads - 1) /
                                                       threads);
        countEnergyKernel<<<blocks, threads>>>(state.deviceRadB, state.numTriangles,
                                               state.numTimesteps, state.localBegin,
                                               state.localCount, deviceCount);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long localReceivedEnergy = 0;
    CUDA_CHECK(cudaMemcpy(&localReceivedEnergy, deviceCount, sizeof(localReceivedEnergy),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceCount));

    unsigned long long nonzeroKij = 0;
    unsigned long long receivedEnergy = 0;
    MPI_Reduce(&localNonzeroKij, &nonzeroKij, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (state.rank == 0) {
        const size_t totalPairs = state.numTriangles * state.numTriangles;
        printf("  Triangles receiving energy: %llu/%zu\n", receivedEnergy,
               state.numTriangles);
        printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonzeroKij,
               totalPairs, 100.0 * static_cast<double>(nonzeroKij) /
               static_cast<double>(totalPairs));
        if (receivedEnergy == 0) {
            printf("  ERROR: No triangles received energy - simulation failed\n");
            valid = false;
        }
        if (nonzeroKij == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
            valid = false;
        }
        if (valid) printf("  Validation: PASSED\n");
    }
    int validInt = valid ? 1 : 0;
    MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return validInt != 0;
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
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    omp_set_dynamic(0);
    int openmpThreads = 1;
#pragma omp parallel
    {
#pragma omp master
        openmpThreads = omp_get_num_threads();
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

    if (targetTriangles <= 0 || timesteps <= 0 || sourceIdx < 0 ||
        reflectivity < ZERO || reflectivity > ONE) {
        if (rank == 0) {
            std::fprintf(stderr,
                "Invalid arguments: triangles and timesteps must be positive, "
                "source must be non-negative, and reflectivity must be in [0,1].\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &nodeCommunicator);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);
    MPI_Comm_size(nodeCommunicator, &nodeSize);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int device = nodeRank % deviceCount;
    const int localGpuSharing = nodeSize > deviceCount ? 1 : 0;
    int anyGpuSharing = 0;
    MPI_Allreduce(&localGpuSharing, &anyGpuSharing, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid execution: %d MPI rank(s), %d OpenMP thread(s)/rank, CUDA GPU %s\n",
               worldSize, openmpThreads, deviceProperties.name);
        if (anyGpuSharing)
            printf("Note: node-local MPI ranks may share GPUs; one rank per GPU is recommended.\n");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.worldSize = worldSize;
    state.device = device;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localPreDuration = MPI_Wtime() - startPre;
    double preSeconds = 0.0;
    MPI_Reduce(&localPreDuration, &preSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) printf("Precomputation time: %.3f ms\n\n", preSeconds * 1000.0);

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startSim = MPI_Wtime();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localSimDuration = MPI_Wtime() - startSim;
    double simSeconds = 0.0;
    MPI_Reduce(&localSimDuration, &simSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %.3f ms\n\n", simSeconds * 1000.0);

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startDist = MPI_Wtime();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localDistDuration = MPI_Wtime() - startDist;
    double distSeconds = 0.0;
    MPI_Reduce(&localDistDuration, &distSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
        printf("Distance computation time: %.3f ms\n\n", distSeconds * 1000.0);

    // Total time
    const double totalSeconds = preSeconds + simSeconds + distSeconds;
    if (rank == 0) printf("Total computation time: %.3f ms\n", totalSeconds * 1000.0);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    const double kijOps = static_cast<double>(n) * static_cast<double>(n);
    const double simOps = kijOps * static_cast<double>(t);
    const double distOps = static_cast<double>(n) * static_cast<double>(t) *
                           static_cast<double>(t);

    if (rank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", totalSeconds * 1000.0 /
               static_cast<double>(n));
    }

    // Memory usage
    const size_t maximumLocalRows = (n + static_cast<size_t>(worldSize) - 1) /
                                    static_cast<size_t>(worldSize);
    const size_t memKij = maximumLocalRows * n * sizeof(val_t);
    const size_t memTau = maximumLocalRows * n * sizeof(int);
    const size_t memRad = t * n * sizeof(val_t);
    const size_t totalMem = memKij + memTau + memRad;
    if (rank == 0)
        printf("  Maximum GPU matrix/history memory per rank: %.2f MB\n",
               totalMem / (1024.0 * 1024.0));

    // Hash
    if (rank == 0) {
        const uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n\n", hash);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    bool validationPassed = true;
    if (validate) validationPassed = validateResults(state);

    releaseSimulation(state);
    MPI_Comm_free(&nodeCommunicator);
    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
