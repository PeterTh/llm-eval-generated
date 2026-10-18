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
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

int worldRank = 0, worldSize = 1;
#define HD __host__ __device__
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
// Only the reporting rank writes benchmark output.
template<class... Args> int rankPrintf(const char* format, Args... args) {
    if (worldRank != 0) return 0;
    if constexpr (sizeof...(args) == 0) return std::fputs(format, stdout);
    else return std::printf(format, args...);
}
#define printf rankPrintf

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
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// MT19937, including its in-place twist, matches std::mt19937 exactly. Row
// checkpoints preserve the original stream without storing O(N*N*NUM_RAYS) samples.
struct MTState {
    uint32_t words[624];
    int pos = 624;
    MTState() {
        words[0] = 42;
        for (int i = 1; i < 624; ++i)
            words[i] = 1812433253U * (words[i-1] ^ (words[i-1] >> 30)) + i;
    }
    HD void twist() {
        // Split at the recurrence wrap so host compilers can vectorize the
        // stream advance; integer division/modulo in this hot loop is avoided.
        for (int i = 0; i < 227; ++i) {
            uint32_t y = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
            words[i] = words[i+397] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
        }
        for (int i = 227; i < 623; ++i) {
            uint32_t y = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
            words[i] = words[i-227] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
        }
        uint32_t y = (words[623] & 0x80000000U) | (words[0] & 0x7fffffffU);
        words[623] = words[396] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
        pos = 0;
    }
    void discard(size_t count) {
        while (count) {
            if (pos == 624) twist();
            size_t step = std::min(count, size_t(624-pos));
            pos += static_cast<int>(step);
            count -= step;
        }
    }
};

struct FlatNode {
    Vec3 center, half;
    int end, first, count;
    __device__ bool intersects(Vec3 p1, Vec3 p2) const {
        Vec3 d = (p2-p1)*0.5f, c = p1+d-center;
        Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
        if (fabsf(c.x)>half.x+ad.x || fabsf(c.y)>half.y+ad.y || fabsf(c.z)>half.z+ad.z) return false;
        if (fabsf(d.y*c.z-d.z*c.y)>half.y*ad.z+half.z*ad.y+EPSILON) return false;
        if (fabsf(d.z*c.x-d.x*c.z)>half.z*ad.x+half.x*ad.z+EPSILON) return false;
        if (fabsf(d.x*c.y-d.y*c.x)>half.x*ad.y+half.y*ad.x+EPSILON) return false;
        return true;
    }
};

void flattenTree(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<size_t>& ids) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, int(ids.size()), int(tree.triangleIndices.size())});
    ids.insert(ids.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenTree(*child, nodes, ids);
    nodes[index].end = int(nodes.size());
}

__device__ bool blocked(Vec3 from, Vec3 to, const Triangle* tris,
                       const FlatNode* nodes, const size_t* ids, size_t i, size_t j) {
    Vec3 dir = to-from;
    float len = dir.norm();
    if (len < EPSILON) return true;
    dir = dir/len;
    for (int node = 0; node < nodes[0].end;) {
        const FlatNode& n = nodes[node];
        // The original traversal tests child boxes, but not the root box.
        if (node && !n.intersects(from, to)) { node = n.end; continue; }
        for (int k = 0; k < n.count; ++k) {
            size_t id = ids[n.first+k];
            if (id == i || id == j) continue;
            const Triangle& tri = tris[id];
            float d = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < len-EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 samplePoint(const Triangle& tri, float u, float v) {
    if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a+(tri.b-tri.a)*u+(tri.c-tri.a)*v;
}

__device__ void warpSamples(uint32_t* words, int* position, float* samples) {
    int lane = threadIdx.x & 31;
    int produced = 0;
    while (produced < 64) {
        if (*position == 624) {
            // Each phase uses the old adjacent word and the already-updated
            // recurrence segment. Read all operands before writing shared state.
            for (int phase = 0; phase < 3; ++phase) {
                int begin = phase*227, end = phase == 2 ? 624 : begin+227;
                uint32_t values[8];
                for (int k = 0, i = begin+lane; i < end; i += 32, ++k) {
                    uint32_t y = (words[i]&0x80000000U) | (words[i == 623 ? 0 : i+1]&0x7fffffffU);
                    values[k] = words[i < 227 ? i+397 : i-227] ^ (y>>1) ^ ((y&1) ? 0x9908b0dfU : 0U);
                }
                __syncwarp();
                for (int k = 0, i = begin+lane; i < end; i += 32, ++k) words[i] = values[k];
                __syncwarp();
            }
            if (lane == 0) *position = 0;
            __syncwarp();
        }
        int start = *position;
        int count = min(64-produced, 624-start);
        for (int k = lane; k < count; k += 32) {
            uint32_t y = words[start+k];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680U;
            y ^= (y << 15) & 0xefc60000U;
            y ^= y >> 18;
            samples[produced+k] = fminf(float(y)*0x1p-32f, 0x1.fffffep-1f);
        }
        __syncwarp();
        if (lane == 0) *position = start+count;
        __syncwarp();
        produced += count;
    }
}

// One warp per receiver. Rays are independent; the final 16-term sum remains
// ordered, avoiding changes to the floating-point reduction and correlation ties.
__global__ void formKernel(const Triangle* tris, const FlatNode* nodes, const size_t* ids,
                           const MTState* rngs, float* kij, int* tau, size_t n, size_t begin, size_t rows) {
    size_t row = size_t(blockIdx.x)*4+threadIdx.x/32, i = begin+row;
    // The final block may contain inactive whole warps.
    if (row >= rows) return;
    int lane = threadIdx.x & 31, warp = threadIdx.x/32;
    __shared__ float allSamples[4][64], allContributions[4][16];
    __shared__ uint32_t allWords[4][624];
    __shared__ int positions[4];
    float* samples = allSamples[warp];
    float* contributions = allContributions[warp];
    uint32_t* words = allWords[warp];
    for (int k = lane; k < 624; k += 32) words[k] = rngs[row].words[k];
    if (lane == 0) positions[warp] = rngs[row].pos;
    __syncwarp();
    Triangle a = tris[i];
    for (size_t j = 0; j < n; ++j) {
        size_t offset = row*n+j;
        Triangle b = tris[j];
        if (lane == 0) {
            Vec3 ca = (a.a+a.b+a.c)/3.0f, cb = (b.a+b.b+b.c)/3.0f;
            tau[offset] = i == j ? 0 : int(ceilf((ca-cb).norm()*INV_WAVE_SPEED));
            kij[offset] = ZERO;
        }
        if (i == j || a._normal.dot(b._normal) > 0.99f) continue;
        warpSamples(words, positions+warp, samples);
        if (lane < NUM_RAYS) {
            Vec3 p = samplePoint(a, samples[4*lane], samples[4*lane+1]);
            Vec3 q = samplePoint(b, samples[4*lane+2], samples[4*lane+3]);
            float value = ZERO;
            if (!blocked(p, q, tris, nodes, ids, i, j)) {
                Vec3 v = q-p;
                float ds = v.squaredNorm();
                if (ds >= EPSILON) {
                    float c1 = cosPhi(v, a._normal), c2 = cosPhi(-v, b._normal);
                    if (c1 > ZERO && c2 > ZERO) value = (c1*c2)/(PI*ds);
                }
            }
            contributions[lane] = value;
        }
        __syncwarp();
        if (lane == 0) {
            float sum = ZERO;
            for (int r = 0; r < NUM_RAYS; ++r) sum += contributions[r];
            kij[offset] = sum*INV_NUM_RAYS;
        }
        __syncwarp();
    }
}

__global__ void propagationKernel(const float* kij, const int* tau, const float* areas,
                                  const float* rho, const float* history, float* result,
                                  size_t n, size_t rows, size_t begin, size_t t,
                                  size_t steps, size_t source) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    size_t i = begin+row;
    float sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[row*n+j];
        if (t < size_t(delay)) continue;
        float k = kij[row*n+j];
        if (k <= ZERO) continue;
        float b = history[(t-delay)*n+j];
        if (b <= ZERO) continue;
        sum += fminf(k*areas[j], ONE)*b;
    }
    result[row] = rho[i]*sum + ((i == source && t < steps/2) ? ONE : ZERO);
}

__global__ void countFactorsKernel(const float* kij, size_t count, unsigned long long* partial) {
    unsigned long long local = 0;
    for (size_t k = size_t(blockIdx.x)*blockDim.x+threadIdx.x; k < count;
         k += size_t(gridDim.x)*blockDim.x) local += kij[k] > EPSILON;
    __shared__ unsigned long long values[256];
    values[threadIdx.x] = local;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = values[0];
}

__global__ void distanceKernel(const float* history, float* distances, size_t n,
                               size_t rows, size_t begin, size_t steps, size_t source) {
    size_t row = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    float maxCorr = ZERO;
    size_t best = 0, i = begin+row;
    for (size_t t = 0; t < steps; ++t) {
        float sum = ZERO;
        for (size_t tt = t; tt < steps; ++tt)
            sum += history[(tt-t)*n+source]*history[tt*n+i];
        if (sum > maxCorr) { maxCorr = sum; best = t; }
    }
    distances[row] = WAVE_SPEED*float(best);
}

template<class T> struct DeviceArray {
    T* data = nullptr;
    void allocate(size_t count) { if (count) cudaCheck(cudaMalloc(&data, count*sizeof(T))); }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) cudaCheck(cudaMemcpy(data, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { if (data) cudaFree(data); }
};

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
    size_t rowBegin, localRows;
    std::vector<int> counts, displacements;
    unsigned long long nonZeroFactors = 0;
    DeviceArray<Triangle> dTriangles;
    DeviceArray<FlatNode> dNodes;
    DeviceArray<size_t> dIds;
    DeviceArray<float> dKij, dAreas, dRho, dHistory, dResult;
    DeviceArray<int> dTau;

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
    state.counts.resize(worldSize);
    state.displacements.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t begin = state.numTriangles*rank/worldSize;
        size_t end = state.numTriangles*(rank+1)/worldSize;
        state.counts[rank] = int(end-begin);
        state.displacements[rank] = int(begin);
    }
    state.rowBegin = state.displacements[worldRank];
    state.localRows = state.counts[worldRank];

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

    // Initialize matrices
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Emission is implicit in propagationKernel (first half of timesteps).
    // Dense pair matrices stay distributed on the GPUs throughout the run.
    state.dTriangles.upload(state.triangles);
    std::vector<FlatNode> nodes;
    std::vector<size_t> ids;
    flattenTree(state.octree, nodes, ids);
    state.dNodes.upload(nodes);
    state.dIds.upload(ids);
    state.dKij.allocate(state.localRows*state.numTriangles);
    state.dTau.allocate(state.localRows*state.numTriangles);
    state.dAreas.upload(state.areas);
    state.dRho.upload(state.rho);
    state.dHistory.allocate(timesteps*state.numTriangles);
    state.dResult.allocate(state.localRows);
    cudaCheck(cudaMemset(state.dHistory.data, 0, timesteps*state.numTriangles*sizeof(float)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    std::vector<size_t> draws(state.localRows);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < state.localRows; ++row) {
        size_t i = state.rowBegin+row, count = 0;
        for (size_t j = 0; j < state.numTriangles; ++j)
            if (i != j && !(state.triangles[i]._normal.dot(state.triangles[j]._normal) > 0.99f))
                count += 4*NUM_RAYS;
        draws[row] = count;
    }
    MTState rng;
    // Pipeline the stream checkpoints between ranks. No random samples or dense
    // form-factor matrices are communicated, and earlier ranks can start the GPU
    // while later ranks prepare their own checkpoints.
    if (worldRank) MPI_Recv(&rng, sizeof(rng), MPI_BYTE, worldRank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<MTState> checkpoints(state.localRows);
    for (size_t row = 0; row < state.localRows; ++row) {
        checkpoints[row] = rng;
        rng.discard(draws[row]);
    }
    if (worldRank+1 < worldSize) MPI_Send(&rng, sizeof(rng), MPI_BYTE, worldRank+1, 0, MPI_COMM_WORLD);
    DeviceArray<MTState> dRng;
    dRng.upload(checkpoints);
    unsigned long long count = 0;
    if (state.localRows) {
        formKernel<<<(state.localRows+3)/4, 128>>>(state.dTriangles.data, state.dNodes.data, state.dIds.data,
            dRng.data, state.dKij.data, state.dTau.data, state.numTriangles, state.rowBegin, state.localRows);
        cudaCheck(cudaGetLastError());
        size_t entries = state.localRows*state.numTriangles;
        size_t blocks = std::min(size_t(4096), (entries+255)/256);
        DeviceArray<unsigned long long> dCounts;
        dCounts.allocate(blocks);
        countFactorsKernel<<<blocks, 256>>>(state.dKij.data, entries, dCounts.data);
        cudaCheck(cudaGetLastError());
        std::vector<unsigned long long> counts(blocks);
        cudaCheck(cudaMemcpy(counts.data(), dCounts.data, blocks*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (auto value : counts) count += value;
    }
    MPI_Allreduce(&count, &state.nonZeroFactors, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    (void)state; // Fused into formKernel to reuse triangle loads.
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    float* staging = nullptr;
    cudaCheck(cudaMallocHost(&staging, state.numTriangles*sizeof(float)));
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows) {
            propagationKernel<<<(state.localRows+127)/128, 128>>>(state.dKij.data, state.dTau.data,
                state.dAreas.data, state.dRho.data, state.dHistory.data, state.dResult.data,
                state.numTriangles, state.localRows, state.rowBegin, t, state.numTimesteps, state.sourceIndex);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(staging+state.rowBegin, state.dResult.data,
                state.localRows*sizeof(float), cudaMemcpyDeviceToHost));
        }
        // Positive off-diagonal delays make a timestep independent across rows.
        // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, staging, state.counts.data(),
                       state.displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(state.dHistory.data+t*state.numTriangles, staging,
            state.numTriangles*sizeof(float), cudaMemcpyHostToDevice));
        std::memcpy(state.radB.data()+t*state.numTriangles, staging, state.numTriangles*sizeof(float));
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaFreeHost(staging));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    if (state.localRows) {
        distanceKernel<<<(state.localRows+127)/128, 128>>>(state.dHistory.data, state.dResult.data,
            state.numTriangles, state.localRows, state.rowBegin, state.numTimesteps, state.sourceIndex);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(state.distances.data()+state.rowBegin, state.dResult.data,
            state.localRows*sizeof(float), cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(worldRank == 0 ? MPI_IN_PLACE : state.distances.data()+state.rowBegin,
        int(state.localRows), MPI_FLOAT, state.distances.data(), state.counts.data(),
        state.displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    auto nonZeroKij = state.nonZeroFactors;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalizeMPI;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 1 || !std::isfinite(reflectivity)) {
        printf("Timesteps must be positive and reflectivity must be finite.\n");
        return 1;
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank, localSize, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, omp_get_num_procs()/localSize));
    MPI_Comm_free(&localComm);

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

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    if (worldRank != 0) {
        int status;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return status;
    }

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
    size_t maxLocalRows = (n+worldSize-1)/worldSize;
    size_t memKij = maxLocalRows * n * sizeof(val_t);
    size_t memTau = maxLocalRows * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Matrix memory usage (largest rank): %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
    int status = validate && !validateResults(state) ? 1 : 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}
