/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI/OpenMP/CUDA room impulse response simulation using radiosity-based
 * wave propagation. It models how sound/light
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
#include <climits>
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
#include <string>
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

#define ROOMSIM_HD __host__ __device__

static int worldRank = 0;

[[noreturn]] void abortWithMessage(const char* kind, const char* expression,
                                   const char* file, int line, const char* detail) {
    std::fprintf(stderr, "Rank %d: %s failure at %s:%d while evaluating %s: %s\n",
                 worldRank, kind, file, line, expression, detail);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

inline void checkCuda(cudaError_t error, const char* expression,
                      const char* file, int line) {
    if (error != cudaSuccess) {
        abortWithMessage("CUDA", expression, file, line, cudaGetErrorString(error));
    }
}

inline void checkMpi(int error, const char* expression, const char* file, int line) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        message[length] = '\0';
        abortWithMessage("MPI", expression, file, line, message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr val_t ROOM_RADIUS = 10.0f;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
static_assert(2.0f * ROOM_RADIUS * INV_WAVE_SPEED < 255.0f,
              "uint8_t Tau storage must cover the room diameter");

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
    ROOMSIM_HD val_t norm() const { return ::sqrtf(squaredNorm()); }
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
    ROOMSIM_HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    ROOMSIM_HD Vec3 center() const { return (a + b + c) / 3.0f; }
    ROOMSIM_HD Vec3 normal() const { return _normal; }

    ROOMSIM_HD val_t area() const {
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

// Preorder layout with an escape index permits stackless GPU traversal.  A
// rejected node jumps over its complete subtree; an accepted internal node
// simply advances into its first child.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    idx_t triangleOffset;
    idx_t triangleCount;
    idx_t escapeIndex;
};

idx_t flattenOctreeNode(const Octree& source,
                        std::vector<FlatOctreeNode>& nodes,
                        std::vector<idx_t>& triangleIndices) {
    if (nodes.size() >= std::numeric_limits<idx_t>::max()) {
        abortWithMessage("octree", "flattenOctreeNode", __FILE__, __LINE__,
                         "too many nodes for 32-bit GPU indices");
    }

    const idx_t nodeIndex = static_cast<idx_t>(nodes.size());
    FlatOctreeNode node{};
    node.center = source.center;
    node.halfExtent = source.halfExtent;
    node.triangleOffset = static_cast<idx_t>(triangleIndices.size());
    node.triangleCount = static_cast<idx_t>(source.triangleIndices.size());
    nodes.push_back(node);

    for (size_t triangleIndex : source.triangleIndices) {
        triangleIndices.push_back(static_cast<idx_t>(triangleIndex));
    }
    for (const auto& child : source.children) {
        if (child) {
            flattenOctreeNode(*child, nodes, triangleIndices);
        }
    }

    nodes[nodeIndex].escapeIndex = static_cast<idx_t>(nodes.size());
    return nodeIndex;
}

void flattenOctree(const Octree& source,
                   std::vector<FlatOctreeNode>& nodes,
                   std::vector<idx_t>& triangleIndices) {
    nodes.clear();
    triangleIndices.clear();
    flattenOctreeNode(source, nodes, triangleIndices);
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
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t rowBegin;
    size_t localRows;
    int worldSize;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity, retained for validation
    std::vector<val_t> distances;   // Computed distances from source
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    val_t reflectivity;
    bool retainRadiosity;
    bool cudaAwareMpi;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

struct DeviceState {
    Triangle* triangles = nullptr;
    val_t* areas = nullptr;
    FlatOctreeNode* octreeNodes = nullptr;
    idx_t* octreeTriangleIndices = nullptr;
    val_t* kij = nullptr;
    uint8_t* tau = nullptr;
    val_t* radB = nullptr;
    val_t* localRadiosity = nullptr;
    val_t* localDistances = nullptr;
    size_t localPairs = 0;
    idx_t octreeNodeCount = 0;
};

void partitionRows(size_t count, int rank, int ranks,
                   size_t& begin, size_t& localCount) {
    const size_t quotient = count / static_cast<size_t>(ranks);
    const size_t remainder = count % static_cast<size_t>(ranks);
    localCount = quotient + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    begin = static_cast<size_t>(rank) * quotient +
            std::min(static_cast<size_t>(rank), remainder);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int ranks,
                          bool retainRadiosity, bool cudaAwareMpi) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    state.worldSize = ranks;
    state.reflectivity = reflectivity;
    state.retainRadiosity = retainRadiosity;
    state.cudaAwareMpi = cudaAwareMpi;
    partitionRows(state.numTriangles, rank, ranks, state.rowBegin, state.localRows);

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // OpenMP prepares independent per-triangle geometry on every rank.
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    if (retainRadiosity) {
        state.radB.resize(timesteps * state.numTriangles, ZERO);
    }
    state.distances.resize(state.numTriangles, ZERO);

    state.rowCounts.resize(ranks);
    state.rowDisplacements.resize(ranks);
    for (int r = 0; r < ranks; ++r) {
        size_t first = 0;
        size_t rows = 0;
        partitionRows(state.numTriangles, r, ranks, first, rows);
        if (rows > static_cast<size_t>(INT_MAX) || first > static_cast<size_t>(INT_MAX)) {
            abortWithMessage("MPI", "row partition", __FILE__, __LINE__,
                             "triangle count exceeds MPI_Allgatherv integer limits");
        }
        state.rowCounts[r] = static_cast<int>(rows);
        state.rowDisplacements[r] = static_cast<int>(first);
    }
}

// ============================================================================
// CUDA kernels
// ============================================================================

__device__ __forceinline__ uint32_t mixBits(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

__device__ __forceinline__ val_t nextRandom(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<val_t>(state >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ Vec3 randomPointDevice(const Triangle& triangle,
                                                   uint32_t& randomState) {
    val_t u = nextRandom(randomState);
    val_t v = nextRandom(randomState);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u +
           (triangle.c - triangle.a) * v;
}

__device__ __forceinline__ bool rayIntersectsBoxDevice(
        const Vec3& p1, const Vec3& p2, const FlatOctreeNode& node) {
    const Vec3 d = (p2 - p1) * 0.5f;
    const Vec3 c = p1 + d - node.center;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

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

__device__ __forceinline__ val_t rayTriangleIntersectDevice(
        const Vec3& origin, const Vec3& direction, const Triangle& triangle) {
    const Vec3 edge1 = triangle.b - triangle.a;
    const Vec3 edge2 = triangle.c - triangle.a;
    const Vec3 pvec = direction.cross(edge2);
    const val_t determinant = edge1.dot(pvec);
    if (fabsf(determinant) < EPSILON) return FLT_MAX;

    const val_t inverseDeterminant = ONE / determinant;
    const Vec3 tvec = origin - triangle.a;
    const val_t u = tvec.dot(pvec) * inverseDeterminant;
    if (u < ZERO || u > ONE) return FLT_MAX;
    const Vec3 qvec = tvec.cross(edge1);
    const val_t v = direction.dot(qvec) * inverseDeterminant;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return edge2.dot(qvec) * inverseDeterminant;
}

__device__ __forceinline__ bool isRayBlockedDevice(
        const Vec3& from, const Vec3& to, const Triangle* triangles,
        const FlatOctreeNode* nodes, idx_t nodeCount,
        const idx_t* triangleIndices, idx_t sourceIndex, idx_t destinationIndex) {
    const Vec3 ray = to - from;
    const val_t rayLength = ray.norm();
    if (rayLength < EPSILON) return true;
    const Vec3 direction = ray / rayLength;

    idx_t nodeIndex = 0;
    while (nodeIndex < nodeCount) {
        const FlatOctreeNode& node = nodes[nodeIndex];
        if (nodeIndex != 0 && !rayIntersectsBoxDevice(from, to, node)) {
            nodeIndex = node.escapeIndex;
            continue;
        }
        for (idx_t offset = 0; offset < node.triangleCount; ++offset) {
            const idx_t triangleIndex = triangleIndices[node.triangleOffset + offset];
            if (triangleIndex == sourceIndex || triangleIndex == destinationIndex) continue;
            const val_t distance = rayTriangleIntersectDevice(
                from, direction, triangles[triangleIndex]);
            if (distance > EPSILON && distance < rayLength - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

__device__ __forceinline__ val_t cosineDevice(const Vec3& vector,
                                               const Vec3& normal) {
    const val_t length = vector.norm();
    return length > EPSILON ? fmaxf(ZERO, vector.dot(normal) / length) : ZERO;
}

__device__ __forceinline__ val_t formFactorDevice(
        idx_t receiverIndex, idx_t emitterIndex, const Triangle* triangles,
        const FlatOctreeNode* nodes, idx_t nodeCount,
        const idx_t* triangleIndices) {
    const Triangle& receiver = triangles[receiverIndex];
    const Triangle& emitter = triangles[emitterIndex];
    if (receiver.normal().dot(emitter.normal()) > 0.99f) return ZERO;

    uint32_t randomState = mixBits(receiverIndex * 0x9e3779b9u ^
                                   emitterIndex * 0x85ebca6bu ^ 42u);
    if (randomState == 0) randomState = 0x6d2b79f5u;
    val_t result = ZERO;
    #pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const Vec3 receiverPoint = randomPointDevice(receiver, randomState);
        const Vec3 emitterPoint = randomPointDevice(emitter, randomState);
        if (isRayBlockedDevice(receiverPoint, emitterPoint, triangles, nodes,
                               nodeCount, triangleIndices, receiverIndex,
                               emitterIndex)) continue;

        const Vec3 vector = emitterPoint - receiverPoint;
        const val_t distanceSquared = vector.squaredNorm();
        if (distanceSquared < EPSILON) continue;
        const val_t receiverCosine = cosineDevice(vector, receiver.normal());
        const val_t emitterCosine = cosineDevice(-vector, emitter.normal());
        if (receiverCosine > ZERO && emitterCosine > ZERO) {
            result += receiverCosine * emitterCosine / (PI * distanceSquared);
        }
    }
    return result * INV_NUM_RAYS;
}

__global__ void precomputeKernel(const Triangle* triangles,
                                 const FlatOctreeNode* nodes, idx_t nodeCount,
                                 const idx_t* triangleIndices, val_t* kij,
                                 uint8_t* tau, size_t triangleCount,
                                 size_t rowBegin, size_t localRows) {
    const uint64_t workItems = static_cast<uint64_t>(localRows) * triangleCount;
    for (uint64_t item = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < workItems;
         item += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        const size_t localRow = item / triangleCount;
        const idx_t receiver = static_cast<idx_t>(rowBegin + localRow);
        const idx_t emitter = static_cast<idx_t>(item - localRow * triangleCount);
        if (receiver == emitter) {
            kij[item] = ZERO;
            tau[item] = 0;
            continue;
        }
        const val_t distance = (triangles[receiver].center() -
                                triangles[emitter].center()).norm();
        tau[item] = static_cast<uint8_t>(ceilf(distance * INV_WAVE_SPEED));
        kij[item] = formFactorDevice(receiver, emitter, triangles, nodes,
                                     nodeCount, triangleIndices);
    }
}

__global__ void simulateTimestepKernel(
        const val_t* __restrict__ kij, const uint8_t* __restrict__ tau,
        const val_t* __restrict__ areas, const val_t* __restrict__ radB,
        val_t* __restrict__ output, size_t triangleCount, size_t rowBegin,
        size_t localRows, size_t timestep, size_t sourceIndex,
        size_t sourceOffTime, val_t reflectivity) {
    __shared__ val_t partialSums[256];
    for (size_t localRow = blockIdx.x;
         localRow < localRows;
         localRow += gridDim.x) {
        const size_t receiver = rowBegin + localRow;
        const size_t matrixOffset = localRow * triangleCount;
        val_t sum = ZERO;
        for (size_t emitter = threadIdx.x; emitter < triangleCount;
             emitter += blockDim.x) {
            if (receiver == emitter) continue;
            const uint8_t delay = tau[matrixOffset + emitter];
            if (timestep < static_cast<size_t>(delay)) continue;
            const val_t factor = kij[matrixOffset + emitter];
            if (factor <= ZERO) continue;
            const val_t emitterRadiosity =
                radB[(timestep - static_cast<size_t>(delay)) * triangleCount + emitter];
            if (emitterRadiosity > ZERO) {
                sum += fminf(factor * areas[emitter], ONE) * emitterRadiosity;
            }
        }
        partialSums[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const val_t emission = receiver == sourceIndex && timestep < sourceOffTime
                                 ? ONE : ZERO;
            output[localRow] = reflectivity * partialSums[0] + emission;
        }
        __syncthreads();
    }
}

__global__ void distanceKernel(const val_t* __restrict__ radB,
                               val_t* __restrict__ distances,
                               size_t triangleCount, size_t timesteps,
                               size_t rowBegin, size_t localRows,
                               size_t sourceIndex) {
    for (size_t localRow = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         localRow < localRows;
         localRow += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t triangle = rowBegin + localRow;
        val_t maximumCorrelation = ZERO;
        size_t bestTimestep = 0;
        for (size_t delay = 0; delay < timesteps; ++delay) {
            val_t correlation = ZERO;
            for (size_t time = delay; time < timesteps; ++time) {
                correlation += radB[(time - delay) * triangleCount + sourceIndex] *
                               radB[time * triangleCount + triangle];
            }
            if (correlation > maximumCorrelation) {
                maximumCorrelation = correlation;
                bestTimestep = delay;
            }
        }
        distances[localRow] = WAVE_SPEED * static_cast<val_t>(bestTimestep);
    }
}

__global__ void countPositiveKernel(const val_t* values, uint64_t count,
                                    unsigned long long* result) {
    __shared__ unsigned int blockCounts[256];
    unsigned int localCount = 0;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
        localCount += values[index] > EPSILON ? 1u : 0u;
    }
    blockCounts[threadIdx.x] = localCount;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) blockCounts[threadIdx.x] += blockCounts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, static_cast<unsigned long long>(blockCounts[0]));
}

int launchBlockCount(uint64_t workItems, int blockSize, int maximumBlocks = 65535) {
    if (workItems == 0) return 0;
    return static_cast<int>(std::min<uint64_t>(
        (workItems + static_cast<uint64_t>(blockSize) - 1) / blockSize,
        static_cast<uint64_t>(maximumBlocks)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void initializeDeviceState(const SimulationState& state, DeviceState& device) {
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<idx_t> flatTriangleIndices;
    flattenOctree(state.octree, flatNodes, flatTriangleIndices);
    device.octreeNodeCount = static_cast<idx_t>(flatNodes.size());
    device.localPairs = state.localRows * state.numTriangles;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.triangles),
                          state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.areas),
                          state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.octreeNodes),
                          flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.octreeTriangleIndices),
                          flatTriangleIndices.size() * sizeof(idx_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.kij),
                          std::max<size_t>(device.localPairs, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.tau),
                          std::max<size_t>(device.localPairs, 1) * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.radB),
                          state.numTimesteps * state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.localRadiosity),
                          std::max<size_t>(state.localRows, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device.localDistances),
                          std::max<size_t>(state.localRows, 1) * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(device.triangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.areas, state.areas.data(),
                          state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.octreeNodes, flatNodes.data(),
                          flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.octreeTriangleIndices, flatTriangleIndices.data(),
                          flatTriangleIndices.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device.radB, 0,
                          state.numTimesteps * state.numTriangles * sizeof(val_t)));
}

void computePrecomputation(const SimulationState& state, DeviceState& device) {
    if (worldRank == 0) {
        printf("Computing time delays (Tau) and form factors (Kij) on GPUs...\n");
    }
    constexpr int threads = 128;
    const int blocks = launchBlockCount(device.localPairs, threads);
    if (blocks > 0) {
        precomputeKernel<<<blocks, threads>>>(
            device.triangles, device.octreeNodes, device.octreeNodeCount,
            device.octreeTriangleIndices, device.kij, device.tau,
            state.numTriangles, state.rowBegin, state.localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Geometry is no longer needed after Kij and Tau have been materialized.
    CUDA_CHECK(cudaFree(device.triangles));
    CUDA_CHECK(cudaFree(device.octreeNodes));
    CUDA_CHECK(cudaFree(device.octreeTriangleIndices));
    device.triangles = nullptr;
    device.octreeNodes = nullptr;
    device.octreeTriangleIndices = nullptr;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, DeviceState& device) {
    if (worldRank == 0) printf("Running wave propagation simulation...\n");

    val_t* hostLocal = nullptr;
    val_t* hostGlobal = nullptr;
    if (!state.cudaAwareMpi) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostLocal),
                                  std::max<size_t>(state.localRows, 1) * sizeof(val_t)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostGlobal),
                                  state.numTriangles * sizeof(val_t)));
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>(state.localRows, static_cast<size_t>(65535)));
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        if (blocks > 0) {
            simulateTimestepKernel<<<blocks, threads>>>(
                device.kij, device.tau, device.areas, device.radB,
                device.localRadiosity, state.numTriangles, state.rowBegin,
                state.localRows, timestep, state.sourceIndex,
                state.numTimesteps / 2, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
        }

        val_t* destination = device.radB + timestep * state.numTriangles;
        if (state.cudaAwareMpi) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_CHECK(MPI_Allgatherv(device.localRadiosity,
                                     static_cast<int>(state.localRows), MPI_FLOAT,
                                     destination, state.rowCounts.data(),
                                     state.rowDisplacements.data(), MPI_FLOAT,
                                     MPI_COMM_WORLD));
        } else {
            CUDA_CHECK(cudaMemcpy(hostLocal, device.localRadiosity,
                                  state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_CHECK(MPI_Allgatherv(hostLocal, static_cast<int>(state.localRows), MPI_FLOAT,
                                     hostGlobal, state.rowCounts.data(),
                                     state.rowDisplacements.data(), MPI_FLOAT,
                                     MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpy(destination, hostGlobal,
                                  state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        if (worldRank == 0 && ((timestep + 1) % 10 == 0 ||
                               timestep + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (hostLocal) CUDA_CHECK(cudaFreeHost(hostLocal));
    if (hostGlobal) CUDA_CHECK(cudaFreeHost(hostGlobal));

    if (state.retainRadiosity) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), device.radB,
                              state.radB.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, DeviceState& device) {
    if (worldRank == 0) printf("Computing distances via cross-correlation...\n");

    constexpr int threads = 128;
    const int blocks = launchBlockCount(state.localRows, threads);
    if (blocks > 0) {
        distanceKernel<<<blocks, threads>>>(
            device.radB, device.localDistances, state.numTriangles,
            state.numTimesteps, state.rowBegin, state.localRows, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
    }

    val_t* hostLocal = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostLocal),
                              std::max<size_t>(state.localRows, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(hostLocal, device.localDistances,
                          state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Allgatherv(hostLocal, static_cast<int>(state.localRows), MPI_FLOAT,
                             state.distances.data(), state.rowCounts.data(),
                             state.rowDisplacements.data(), MPI_FLOAT, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFreeHost(hostLocal));
}

unsigned long long countNonZeroFormFactors(const DeviceState& device) {
    unsigned long long* deviceCount = nullptr;
    unsigned long long localCount = 0;
    unsigned long long globalCount = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCount), sizeof(*deviceCount)));
    CUDA_CHECK(cudaMemset(deviceCount, 0, sizeof(*deviceCount)));
    constexpr int threads = 256;
    const int blocks = launchBlockCount(device.localPairs, threads, 4096);
    if (blocks > 0) {
        countPositiveKernel<<<blocks, threads>>>(device.kij, device.localPairs, deviceCount);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(&localCount, deviceCount, sizeof(localCount), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceCount));
    MPI_CHECK(MPI_Reduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, 0, MPI_COMM_WORLD));
    return globalCount;
}

void releaseDeviceState(DeviceState& device) {
    if (device.triangles) CUDA_CHECK(cudaFree(device.triangles));
    if (device.areas) CUDA_CHECK(cudaFree(device.areas));
    if (device.octreeNodes) CUDA_CHECK(cudaFree(device.octreeNodes));
    if (device.octreeTriangleIndices) CUDA_CHECK(cudaFree(device.octreeTriangleIndices));
    if (device.kij) CUDA_CHECK(cudaFree(device.kij));
    if (device.tau) CUDA_CHECK(cudaFree(device.tau));
    if (device.radB) CUDA_CHECK(cudaFree(device.radB));
    if (device.localRadiosity) CUDA_CHECK(cudaFree(device.localRadiosity));
    if (device.localDistances) CUDA_CHECK(cudaFree(device.localDistances));
    device = {};
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state,
                     unsigned long long nonZeroKij) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    int negativeCount = 0;
    int nonFiniteCount = 0;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    #pragma omp parallel for schedule(static) reduction(+:negativeCount,nonFiniteCount,sumDist,nonZeroCount) reduction(min:minDist) reduction(max:maxDist)
    for (int64_t i = 0; i < static_cast<int64_t>(state.numTriangles); ++i) {
        const val_t d = state.distances[i];
        negativeCount += d < ZERO ? 1 : 0;
        nonFiniteCount += std::isfinite(d) ? 0 : 1;
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        nonZeroCount += d > EPSILON ? 1 : 0;
    }

    if (negativeCount != 0) printf("  ERROR: %d negative distances\n", negativeCount);
    if (nonFiniteCount != 0) printf("  ERROR: %d non-finite distances\n", nonFiniteCount);

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
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
    for (int64_t i = 0; i < static_cast<int64_t>(state.numTriangles); ++i) {
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

    const unsigned long long totalFactors =
        static_cast<unsigned long long>(state.numTriangles) * state.numTriangles;
    printf("  Non-zero form factors: %llu/%llu (%.2f%%)\n",
           nonZeroKij, totalFactors,
           100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalFactors));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (negativeCount != 0 || nonFiniteCount != 0) {
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
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120,
    //               5->20480, 6->81920
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
    printf("               Actual count will be rounded up to an icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480, 81920\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
    printf("Environment:\n");
    printf("  OMP_NUM_THREADS          CPU threads per MPI rank\n");
    printf("  ROOMSIM_CUDA_AWARE_MPI=1 Use direct GPU MPI buffers when supported\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel));
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage("MPI", "MPI_Init_thread", __FILE__, __LINE__,
                         "MPI_THREAD_FUNNELED support is required");
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (targetTriangles <= 0 || timesteps <= 0 || sourceIdx < 0 ||
        !std::isfinite(reflectivity) || reflectivity < ZERO || reflectivity > ONE) {
        if (worldRank == 0) {
            printf("Invalid arguments: counts and source must be non-negative, and "
                   "reflectivity must be in [0, 1].\n");
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    // Assign ranks to GPUs by their node-local rank. Multiple ranks per device
    // remain supported for schedulers that deliberately oversubscribe GPUs.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  worldRank, MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage("CUDA", "cudaGetDeviceCount", __FILE__, __LINE__,
                         "no CUDA accelerator is available");
    }
    const int deviceIndex = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceIndex));
    CUDA_CHECK(cudaFree(nullptr));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    int cudaAwareFlag = 0;
    if (const char* value = std::getenv("ROOMSIM_CUDA_AWARE_MPI")) {
        cudaAwareFlag = std::strcmp(value, "0") != 0 ? 1 : 0;
    }
    int minimumCudaAwareFlag = 0;
    int maximumCudaAwareFlag = 0;
    MPI_CHECK(MPI_Allreduce(&cudaAwareFlag, &minimumCudaAwareFlag, 1, MPI_INT,
                            MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&cudaAwareFlag, &maximumCudaAwareFlag, 1, MPI_INT,
                            MPI_MAX, MPI_COMM_WORLD));
    if (minimumCudaAwareFlag != maximumCudaAwareFlag) {
        abortWithMessage("MPI", "ROOMSIM_CUDA_AWARE_MPI", __FILE__, __LINE__,
                         "the setting must be identical on every rank");
    }
    const bool cudaAwareMpi = minimumCudaAwareFlag != 0;

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (worldRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
               worldSize, omp_get_max_threads());
        printf("GPU assignment: node-local rank modulo %d visible device(s)\n", deviceCount);
        printf("MPI GPU buffers: %s\n", cudaAwareMpi ? "CUDA-aware direct" : "pinned-host staged");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity,
                         worldRank, worldSize, validate, cudaAwareMpi);

    if (worldRank == 0) printf("\n");

    DeviceState device;
    double localDuration = 0.0;
    double maximumDuration = 0.0;

    // Precomputation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double phaseStart = MPI_Wtime();
    initializeDeviceState(state, device);
    computePrecomputation(state, device);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    localDuration = MPI_Wtime() - phaseStart;
    MPI_CHECK(MPI_Reduce(&localDuration, &maximumDuration, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    const long preDuration = static_cast<long>(maximumDuration * 1000.0);
    if (worldRank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    // Simulation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    phaseStart = MPI_Wtime();
    runSimulation(state, device);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    localDuration = MPI_Wtime() - phaseStart;
    MPI_CHECK(MPI_Reduce(&localDuration, &maximumDuration, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    const long simDuration = static_cast<long>(maximumDuration * 1000.0);
    if (worldRank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    // Distance computation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    phaseStart = MPI_Wtime();
    computeDistances(state, device);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    localDuration = MPI_Wtime() - phaseStart;
    MPI_CHECK(MPI_Reduce(&localDuration, &maximumDuration, 1, MPI_DOUBLE,
                         MPI_MAX, 0, MPI_COMM_WORLD));
    const long distDuration = static_cast<long>(maximumDuration * 1000.0);
    if (worldRank == 0) printf("Distance computation time: %ld ms\n\n", distDuration);

    const unsigned long long nonZeroKij = validate
        ? countNonZeroFormFactors(device) : 0;

    // Total time
    const long totalTime = preDuration + simDuration + distDuration;
    if (worldRank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;
    if (worldRank == 0) {
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = kijOps * static_cast<double>(t);
        const double distOps = static_cast<double>(n) * t * t;

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        const size_t maximumRows = (n + static_cast<size_t>(worldSize) - 1) /
                                   static_cast<size_t>(worldSize);
        const double acceleratorBytes =
            static_cast<double>(maximumRows) * n * (sizeof(val_t) + sizeof(uint8_t)) +
            static_cast<double>(t) * n * sizeof(val_t) +
            static_cast<double>(n) * sizeof(val_t);
        printf("  Memory usage: %.2f MB per rank (largest accelerator allocation)\n",
               acceleratorBytes / (1024.0 * 1024.0));

        const uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    
    // Print results for external validation
    if (worldRank == 0 && printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int validationPassed = 1;
    if (worldRank == 0 && validate) {
        validationPassed = validateResults(state, nonZeroKij) ? 1 : 0;
    }
    MPI_CHECK(MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD));

    releaseDeviceState(device);
    MPI_CHECK(MPI_Finalize());
    return validationPassed ? 0 : 1;
}
