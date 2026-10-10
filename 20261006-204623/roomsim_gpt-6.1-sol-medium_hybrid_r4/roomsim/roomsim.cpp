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
#include <climits>
#include <cfloat>

#define HD __host__ __device__

static int worldRank = 0, worldSize = 1;

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Only rank zero writes the benchmark report.
#define printf(...) do { if (worldRank == 0) std::printf(__VA_ARGS__); } while (0)


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
constexpr val_t ROOM_RADIUS = 10.0f;
static_assert(2.0f * ROOM_RADIUS * INV_WAVE_SPEED <= UINT8_MAX,
              "Propagation delays must fit in one byte");
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
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
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

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
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
    unsigned long long nonZeroKij = 0;
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

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);  // Radius 10 units
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
    #pragma omp parallel for schedule(static) if(state.numTriangles >= 2048)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices

    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);


}

// ============================================================================
// Precomputation Phase
// ============================================================================


// A portable checkpoint of the original MT19937 stream. Discarding does not
// temper outputs: only the 624-word state transition is needed to advance it.
struct RandomCheckpoint {
    uint32_t words[624];
    uint32_t position = 624;
    RandomCheckpoint() {
        words[0] = 42;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253U * (words[i-1] ^ (words[i-1] >> 30)) + i;
    }
    void twist() {
        // Split the wraparound ranges so the compiler can vectorize the
        // recurrence without integer remainder operations in the inner loop.
        for (uint32_t i = 0; i < 227; ++i) {
            uint32_t x = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
            words[i] = words[i+397] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
        }
        for (uint32_t i = 227; i < 623; ++i) {
            uint32_t x = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
            words[i] = words[i-227] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
        }
        uint32_t x = (words[623] & 0x80000000U) | (words[0] & 0x7fffffffU);
        words[623] = words[396] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
        position = 0;
    }
    void discard(size_t count) {
        while (count) {
            if (position == 624) twist();
            size_t advance = std::min(count, size_t(624 - position));
            position += static_cast<uint32_t>(advance);
            count -= advance;
        }
    }
    val_t next() {
        if (position == 624) twist();
        uint32_t y = words[position++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        // libstdc++ uniform_real_distribution<float> uses one engine word,
        // rounded to float, scaled by 2^-32 and clamped below one.
        return std::min(static_cast<float>(y) * 0x1p-32f, 0x1.fffffep-1f);
    }
};

struct FlatNode {
    Vec3 center, halfExtent;
    uint32_t first, count, escape;
};

static void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                          std::vector<uint32_t>& indices) {
    const size_t id = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent,
                     static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size()), 0});
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<uint32_t>(i));
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[id].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool intersectsNode(const Vec3& p1, const Vec3& p2, const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y ||
        fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blockedOnDevice(const Vec3& from, const Vec3& to,
                                const Triangle* triangles, const FlatNode* nodes,
                                const uint32_t* indices, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    Vec3 direction = dir / length;
    // Preorder traversal with escape links avoids per-ray recursion or stacks.
    uint32_t node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& current = nodes[node];
        if (node != 0 && !intersectsNode(from, to, current)) {
            node = current.escape;
            continue;
        }
        for (uint32_t k = 0; k < current.count; ++k) {
            uint32_t index = indices[current.first + k];
            if (index == src || index == dst) continue;
            const Triangle& tri = triangles[index];
            val_t distance = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__device__ val_t rayContribution(const Triangle* triangles, const FlatNode* nodes,
                                 const uint32_t* indices, const val_t* samples,
                                 size_t i, size_t j) {
    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    Vec3 pI = sampledPoint(triI, samples[0], samples[1]);
    Vec3 pJ = sampledPoint(triJ, samples[2], samples[3]);
    if (blockedOnDevice(pI, pJ, triangles, nodes, indices, i, j)) return ZERO;
    Vec3 v = pJ - pI;
    val_t distanceSquared = v.squaredNorm();
    if (distanceSquared < EPSILON) return ZERO;
    val_t ci = cosPhi(v, triI.normal()), cj = cosPhi(-v, triJ.normal());
    if (ci <= ZERO || cj <= ZERO) return ZERO;
    return (ci * cj) / (PI * distanceSquared);
}

__global__ void formFactorKernel(const Triangle* triangles, const FlatNode* nodes,
                                 const uint32_t* indices, const val_t* random,
                                 val_t* kij, uint8_t* tau, size_t n, size_t localN,
                                 size_t firstRow, size_t tileStart, size_t tileRows) {
    static_assert(NUM_RAYS == 16, "Ray groups occupy half a CUDA warp");
    size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t pair = thread / NUM_RAYS;
    unsigned ray = threadIdx.x % NUM_RAYS;
    if (pair >= tileRows * n) return;
    size_t row = tileStart + pair / n;
    size_t i = firstRow + row, j = pair % n;
    // Column-major local receiver rows coalesce propagation loads on the GPU.
    size_t output = j * localN + row;
    if (i == j) {
        if (ray == 0) { kij[output] = ZERO; tau[output] = 0; }
        return;
    }
    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    if (ray == 0)
        tau[output] = static_cast<uint8_t>(ceilf((triI.center() - triJ.center()).norm() * INV_WAVE_SPEED));
    if (triI.normal().dot(triJ.normal()) > 0.99f) {
        if (ray == 0) kij[output] = ZERO;
        return;
    }
    val_t contribution = rayContribution(triangles, nodes, indices,
                                         random + pair * NUM_RAYS * 4 + ray * 4, i, j);
    // Rays execute concurrently, but lane zero sums them in original ray order.
    unsigned mask = 0xffffU << (threadIdx.x & 16U);
    val_t sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t value = __shfl_sync(mask, contribution, r, NUM_RAYS);
        if (ray == 0) sum += value;
    }
    if (ray == 0) kij[output] = sum * INV_NUM_RAYS;
}

__global__ void countFactorsKernel(val_t* kij, const uint8_t* tau, const val_t* areas,
                                   size_t pairs, size_t localN,
                                   unsigned long long* count, int* zeroDelay) {
    __shared__ unsigned counts[256];
    unsigned subtotal = 0;
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < pairs; p += size_t(gridDim.x) * blockDim.x) {
        if (kij[p] > EPSILON) ++subtotal;
        if (tau[p] == 0 && kij[p] > ZERO) atomicExch(zeroDelay, 1);
        // The fixed radius bounds ceil(distance / WAVE_SPEED) by 40, so byte
        // delays are lossless. Cache the timestep-invariant transfer weight.
        if (kij[p] > ZERO) kij[p] = fminf(kij[p] * areas[p / localN], ONE);
    }
    counts[threadIdx.x] = subtotal;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(count, static_cast<unsigned long long>(counts[0]));
}

__global__ void propagationKernel(const val_t* kij, const uint8_t* tau,
                                  const val_t* rho, val_t* radB,
                                  size_t n, size_t localN, size_t firstRow,
                                  size_t t, size_t source, size_t timeOff,
                                  size_t orderedRow) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= localN) return;
    size_t i = firstRow + row;
    if (orderedRow < n && i != orderedRow) return;
    val_t sum = ZERO;
    // Preserve increasing-emitter accumulation order; parallelize receivers.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j * localN + row];
        if (t < static_cast<size_t>(delay)) continue;
        val_t factor = kij[j * localN + row];
        if (factor <= ZERO) continue;
        val_t sourceRad = radB[(t - delay) * n + j];
        if (sourceRad <= ZERO) continue;
        sum += factor * sourceRad;
    }
    val_t emission = (i == source && t < timeOff) ? ONE : ZERO;
    radB[t * n + i] = rho[i] * sum + emission;
}

__global__ void correlationKernel(const val_t* radB, val_t* correlations,
                                  size_t n, size_t localN, size_t firstRow,
                                  size_t timesteps, size_t source) {
    size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair >= localN * timesteps) return;
    size_t lag = pair / localN, row = pair % localN;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < timesteps; ++tt)
        sum += radB[(tt - lag) * n + source] * radB[tt * n + firstRow + row];
    correlations[pair] = sum;
}

__global__ void distanceKernel(const val_t* correlations, val_t* distances,
                               size_t localN, size_t timesteps) {
    size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= localN) return;
    val_t best = ZERO;
    size_t lag = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = correlations[t * localN + row];
        if (sum > best) { best = sum; lag = t; }
    }
    distances[row] = WAVE_SPEED * static_cast<val_t>(lag);
}

template<class T> class DeviceBuffer {
    T* data_ = nullptr;
public:
    explicit DeviceBuffer(size_t count = 0) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    void upload(const T* source, size_t count) {
        if (count) cudaCheck(cudaMemcpy(data_, source, count*sizeof(T), cudaMemcpyHostToDevice));
    }
};

class HybridSimulation {
    SimulationState& state;
    size_t n, localN, firstRow;
    std::vector<int> counts, offsets;
    std::vector<FlatNode> nodes;
    std::vector<uint32_t> indices;
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<val_t> areas, rho, kij, radB;
    DeviceBuffer<uint8_t> tau;
    val_t* exchange = nullptr;
    bool ordered = false;

    static size_t beginRow(int rank, size_t size) {
        return size * static_cast<size_t>(rank) / static_cast<size_t>(worldSize);
    }
public:
    explicit HybridSimulation(SimulationState& s)
        : state(s), n(s.numTriangles),
          localN(beginRow(worldRank + 1, n) - beginRow(worldRank, n)),
          firstRow(beginRow(worldRank, n)), counts(worldSize), offsets(worldSize),
          triangles(n), areas(n), rho(n), kij(n * localN),
          radB(n * s.numTimesteps), tau(n * localN) {
        for (int rank = 0; rank < worldSize; ++rank) {
            offsets[rank] = static_cast<int>(beginRow(rank, n));
            counts[rank] = static_cast<int>(beginRow(rank+1, n) - beginRow(rank, n));
        }
        flattenOctree(state.octree, nodes, indices);
        triangles.upload(state.triangles.data(), n);
        areas.upload(state.areas.data(), n);
        rho.upload(state.rho.data(), n);
        cudaCheck(cudaMemset(radB.get(), 0, n * state.numTimesteps * sizeof(val_t)));
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&exchange), n * sizeof(val_t)));
    }
    ~HybridSimulation() { cudaFreeHost(exchange); }

    void precompute() {
        printf("Computing time delays and form factors (MPI/OpenMP/CUDA)...\n");
        std::vector<RandomCheckpoint> snapshots;
        std::vector<size_t> draws;
        if (worldRank == 0) {
            snapshots.resize(n);
            draws.resize(n);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                size_t active = 0;
                for (size_t j = 0; j < n; ++j)
                    if (i != j && !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f)) ++active;
                draws[i] = active * NUM_RAYS * 4;
            }
            RandomCheckpoint stream;
            for (size_t i = 0; i < n; ++i) {
                snapshots[i] = stream;
                stream.discard(draws[i]);
            }
        }
        // Each receiver gets the exact stream position it had in the original.
        std::vector<RandomCheckpoint> localSnapshots(localN);
        std::vector<int> byteCounts(worldSize), byteOffsets(worldSize);
        for (int rank = 0; rank < worldSize; ++rank) {
            byteCounts[rank] = counts[rank] * static_cast<int>(sizeof(RandomCheckpoint));
            byteOffsets[rank] = offsets[rank] * static_cast<int>(sizeof(RandomCheckpoint));
        }
        MPI_Scatterv(snapshots.data(), byteCounts.data(), byteOffsets.data(), MPI_BYTE,
                     localSnapshots.data(), byteCounts[worldRank], MPI_BYTE, 0, MPI_COMM_WORLD);
        snapshots.clear(); snapshots.shrink_to_fit();

        DeviceBuffer<FlatNode> deviceNodes(nodes.size());
        DeviceBuffer<uint32_t> deviceIndices(indices.size());
        deviceNodes.upload(nodes.data(), nodes.size());
        deviceIndices.upload(indices.data(), indices.size());
        // Double buffered, bounded sample storage overlaps CPU preparation with
        // transfer and ray tracing, without an O(N^2 * NUM_RAYS) allocation.
        size_t tileRows = std::min(localN, std::max(size_t(1),
                               (8UL * 1024 * 1024) / (n * NUM_RAYS * 4 * sizeof(val_t))));
        size_t sampleCount = tileRows * n * NUM_RAYS * 4;
        DeviceBuffer<val_t> samples0(sampleCount), samples1(sampleCount);
        val_t* hostSamples[2] = {nullptr, nullptr};
        cudaStream_t streams[2];
        for (int b = 0; b < 2; ++b) {
            cudaCheck(cudaStreamCreate(&streams[b]));
            if (sampleCount) cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostSamples[b]), sampleCount*sizeof(val_t)));
        }
        size_t tile = 0;
        for (size_t start = 0; start < localN; start += tileRows, ++tile) {
            int b = tile % 2;
            cudaCheck(cudaStreamSynchronize(streams[b]));
            size_t rows = std::min(tileRows, localN - start);
            val_t* host = hostSamples[b];
            #pragma omp parallel for schedule(static) num_threads(std::min(omp_get_max_threads(), static_cast<int>(rows)))
            for (size_t r = 0; r < rows; ++r) {
                RandomCheckpoint rng = localSnapshots[start + r];
                size_t i = firstRow + start + r;
                for (size_t j = 0; j < n; ++j) {
                    if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                    val_t* pair = host + (r*n+j) * NUM_RAYS * 4;
                    for (int k = 0; k < NUM_RAYS * 4; ++k) pair[k] = rng.next();
                }
            }
            val_t* device = b == 0 ? samples0.get() : samples1.get();
            cudaCheck(cudaMemcpyAsync(device, host, rows*n*NUM_RAYS*4*sizeof(val_t),
                                      cudaMemcpyHostToDevice, streams[b]));
            formFactorKernel<<<static_cast<unsigned>((rows*n*NUM_RAYS+127)/128),128,0,streams[b]>>>(
                triangles.get(), deviceNodes.get(), deviceIndices.get(), device,
                kij.get(), tau.get(), n, localN, firstRow, start, rows);
            cudaCheck(cudaGetLastError());
        }
        for (int b = 0; b < 2; ++b) {
            cudaCheck(cudaStreamSynchronize(streams[b]));
            cudaCheck(cudaStreamDestroy(streams[b]));
            if (hostSamples[b]) cudaCheck(cudaFreeHost(hostSamples[b]));
        }
        DeviceBuffer<unsigned long long> deviceCount(1);
        DeviceBuffer<int> deviceZeroDelay(1);
        cudaCheck(cudaMemset(deviceCount.get(), 0, sizeof(unsigned long long)));
        cudaCheck(cudaMemset(deviceZeroDelay.get(), 0, sizeof(int)));
        if (localN) {
            countFactorsKernel<<<256,256>>>(kij.get(), tau.get(), areas.get(), n*localN, localN,
                                           deviceCount.get(), deviceZeroDelay.get());
            cudaCheck(cudaGetLastError());
        }
        unsigned long long localCount;
        int localZero, globalZero;
        cudaCheck(cudaMemcpy(&localCount, deviceCount.get(), sizeof(localCount), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(&localZero, deviceZeroDelay.get(), sizeof(localZero), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localZero, &globalZero, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        ordered = globalZero != 0;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void propagate() {
        printf("Running wave propagation simulation (MPI/CUDA)...\n");
        unsigned blocks = static_cast<unsigned>((localN + 127) / 128);
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (!ordered) {
                if (localN) {
                    propagationKernel<<<blocks,128>>>(kij.get(), tau.get(), rho.get(),
                        radB.get(), n, localN, firstRow, t, state.sourceIndex, state.numTimesteps/2, n);
                    cudaCheck(cudaGetLastError());
                    cudaCheck(cudaMemcpy(exchange+firstRow, radB.get()+t*n+firstRow,
                                         localN*sizeof(val_t), cudaMemcpyDeviceToHost));
                }
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange, counts.data(), offsets.data(),
                               MPI_FLOAT, MPI_COMM_WORLD);
                cudaCheck(cudaMemcpy(radB.get()+t*n, exchange, n*sizeof(val_t), cudaMemcpyHostToDevice));
            } else {
                // Zero-delay interactions require the original increasing-i
                // Gauss-Seidel updates, including communication between owners.
                int owner = 0;
                for (size_t i = 0; i < n; ++i) {
                    while (owner+1 < worldSize && i >= size_t(offsets[owner]+counts[owner])) ++owner;
                    if (worldRank == owner) {
                        propagationKernel<<<blocks,128>>>(kij.get(), tau.get(), rho.get(),
                            radB.get(), n, localN, firstRow, t, state.sourceIndex, state.numTimesteps/2, i);
                        cudaCheck(cudaGetLastError());
                        cudaCheck(cudaMemcpy(exchange+i, radB.get()+t*n+i, sizeof(val_t), cudaMemcpyDeviceToHost));
                    }
                    MPI_Bcast(exchange+i, 1, MPI_FLOAT, owner, MPI_COMM_WORLD);
                    cudaCheck(cudaMemcpy(radB.get()+t*n+i, exchange+i, sizeof(val_t), cudaMemcpyHostToDevice));
                }
            }
            std::copy(exchange, exchange+n, state.radB.begin()+t*n);
            if ((t+1)%10 == 0 || t+1 == state.numTimesteps)
                printf("  Timestep %zu/%zu\n", t+1, state.numTimesteps);
        }
        cudaCheck(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void correlate() {
        printf("Computing distances via cross-correlation (MPI/CUDA)...\n");
        DeviceBuffer<val_t> correlations(localN*state.numTimesteps), distances(localN);
        if (localN) {
            correlationKernel<<<static_cast<unsigned>((localN*state.numTimesteps+127)/128),128>>>(
                radB.get(), correlations.get(), n, localN, firstRow, state.numTimesteps, state.sourceIndex);
            cudaCheck(cudaGetLastError());
            distanceKernel<<<static_cast<unsigned>((localN+127)/128),128>>>(
                correlations.get(), distances.get(), localN, state.numTimesteps);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(exchange+firstRow, distances.get(), localN*sizeof(val_t), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, exchange, counts.data(), offsets.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        std::copy(exchange, exchange+n, state.distances.begin());
        MPI_Barrier(MPI_COMM_WORLD);
    }
};


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
    const unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0 * nonZeroKij / static_cast<double>(state.numTriangles * state.numTriangles));
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

struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm shared;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                            MPI_INFO_NULL, &shared);
        int localRank, localSize;
        MPI_Comm_rank(shared, &localRank);
        MPI_Comm_size(shared, &localSize);
        MPI_Comm_free(&shared);
        int devices = 0;
        cudaCheck(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaCheck(cudaSetDevice(localRank % devices));
        // Honor the OpenMP environment; avoid every MPI rank claiming the node.
        if (!std::getenv("OMP_NUM_THREADS")) {
            omp_set_num_threads(std::max(1, std::min(32, omp_get_max_threads() / localSize)));
        }
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

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

    if (timesteps <= 0) {
        printf("Number of timesteps must be positive\n");
        return 1;
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

    HybridSimulation hybrid(state);
    hybrid.precompute();

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    hybrid.propagate();

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    hybrid.correlate();

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

    // Core device storage on the largest receiver partition.
    size_t maxLocalRows = (n + worldSize - 1) / worldSize;
    size_t memKij = maxLocalRows * n * sizeof(val_t);
    size_t memTau = maxLocalRows * n * sizeof(uint8_t);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Matrix/history device storage per rank: <= %.2f MB\n", totalMem / (1024.0 * 1024.0));
    printf("  MPI ranks: %d; OpenMP threads on rank zero: %d\n", worldSize, omp_get_max_threads());

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
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
