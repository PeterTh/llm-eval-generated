/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed-memory MPI implementation of room impulse response
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
#include <mpi.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;
using tau_t = uint8_t;

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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    // The form-factor calculation is decomposed by rows.  Saving the engine
    // state at each row-block boundary lets every MPI rank reproduce exactly
    // the random-number subsequence used by the serial implementation.
    std::string serialize() const {
        std::ostringstream stream;
        stream << rng << '\n' << dist;
        return stream.str();
    }

    bool deserialize(const std::string& data) {
        std::istringstream stream(data);
        stream >> rng >> dist;
        return !stream.fail();
    }
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
    size_t localBegin;
    size_t localCount;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    // Kij and Tau own only this rank's receiver rows.  Every rank keeps the
    // geometry because every local row references all emitter triangles.
    std::vector<val_t> kij;
    std::vector<tau_t> tau;
    std::vector<val_t> radB;         // Local radiosity, T x localCount
    std::vector<val_t> radHistory;  // Global radiosity ring buffer
    std::vector<val_t> sourceRadB;  // Global source history, T entries
    std::vector<val_t> distances;   // Local computed distances
    std::vector<val_t> globalDistances; // Root-only, ordered global result
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    size_t historyLength;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t localI, size_t j) const { return localI * numTriangles + j; }
    size_t idxTN(size_t t, size_t localI) const { return t * localCount + localI; }
};

struct RowRange {
    size_t begin;
    size_t end;
};

RowRange rowRangeForRank(size_t numRows, int rank, int worldSize) {
    return {
        numRows * static_cast<size_t>(rank) / static_cast<size_t>(worldSize),
        numRows * static_cast<size_t>(rank + 1) / static_cast<size_t>(worldSize)
    };
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          int rank, int worldSize) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    const RowRange localRange = rowRangeForRank(state.numTriangles, rank, worldSize);
    state.localBegin = localRange.begin;
    state.localCount = localRange.end - localRange.begin;

    if (rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (rank == 0) {
        printf("Building octree...\n");
    }
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.localCount * state.numTriangles, ZERO);
    state.tau.resize(state.localCount * state.numTriangles, 0);
    state.radB.resize(timesteps * state.localCount, ZERO);
    state.distances.resize(state.localCount, ZERO);
    state.sourceRadB.resize(timesteps, ZERO);
    state.rowCounts.resize(worldSize);
    state.rowDisplacements.resize(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const RowRange processRange = rowRangeForRank(state.numTriangles, process, worldSize);
        state.rowCounts[process] = static_cast<int>(processRange.end - processRange.begin);
        state.rowDisplacements[process] = static_cast<int>(processRange.begin);
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void advanceRandomStreamForRows(RandomGenerator& rng, const SimulationState& state,
                                const RowRange& rows) {
    for (size_t i = rows.begin; i < rows.end; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            if (state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
            for (int ray = 0; ray < NUM_RAYS; ++ray) {
                rng.rand();
                rng.rand();
            }
        }
    }
}

RandomGenerator initializeLocalRandomStream(const SimulationState& state, int rank,
                                            int worldSize, MPI_Comm communicator) {
    RandomGenerator localRng;

    if (rank == 0) {
        RandomGenerator serialRng(42);
        for (int process = 0; process < worldSize; ++process) {
            const std::string snapshot = serialRng.serialize();
            if (process == 0) {
                if (!localRng.deserialize(snapshot)) {
                    MPI_Abort(communicator, 1);
                }
            } else {
                const uint64_t length = static_cast<uint64_t>(snapshot.size());
                MPI_Send(&length, 1, MPI_UINT64_T, process, 0, communicator);
                MPI_Send(snapshot.data(), static_cast<int>(length), MPI_CHAR,
                         process, 1, communicator);
            }
            advanceRandomStreamForRows(serialRng, state,
                                       rowRangeForRank(state.numTriangles, process, worldSize));
        }
    } else {
        uint64_t length = 0;
        MPI_Recv(&length, 1, MPI_UINT64_T, 0, 0, communicator, MPI_STATUS_IGNORE);
        std::string snapshot(static_cast<size_t>(length), '\0');
        MPI_Recv(snapshot.data(), static_cast<int>(length), MPI_CHAR,
                 0, 1, communicator, MPI_STATUS_IGNORE);
        if (!localRng.deserialize(snapshot)) {
            MPI_Abort(communicator, 1);
        }
    }

    return localRng;
}

void computeFormFactors(SimulationState& state, RandomGenerator& rng) {
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.idx2d(localI, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localBegin + localI;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(localI, j)] = static_cast<tau_t>(computeTau(
                state.triangles[i], state.triangles[j]));
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void initializeRadiosityHistory(SimulationState& state, MPI_Comm communicator) {
    int localMaxTau = 0;
    for (tau_t value : state.tau) {
        localMaxTau = std::max(localMaxTau, static_cast<int>(value));
    }
    int maxTau = 0;
    MPI_Allreduce(&localMaxTau, &maxTau, 1, MPI_INT, MPI_MAX, communicator);

    state.historyLength = std::min(state.numTimesteps,
                                   static_cast<size_t>(maxTau) + 1);
    state.radHistory.assign(state.historyLength * state.numTriangles, ZERO);
}

void runSimulation(SimulationState& state, int rank, MPI_Comm communicator) {
    if (rank == 0) {
        printf("Running wave propagation simulation...\n");
    }

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        for (size_t localI = 0; localI < state.localCount; ++localI) {
            const size_t i = state.localBegin + localI;
            val_t sumB = ZERO;

            for (size_t j = 0; j < state.numTriangles; ++j) {
                if (i == j) continue;

                const int tauij = static_cast<int>(state.tau[state.idx2d(localI, j)]);

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                const val_t kij = state.kij[state.idx2d(localI, j)];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                const size_t srcTime = t - static_cast<size_t>(tauij);
                const val_t radJ = state.radHistory[
                    (srcTime % state.historyLength) * state.numTriangles + j];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + source emission.  radE has only
            // one non-zero column, so materializing its distributed matrix is
            // unnecessary.
            const val_t emission = (i == state.sourceIndex && t < state.numTimesteps / 2)
                ? ONE : ZERO;
            state.radB[state.idxTN(t, localI)] = state.rho[i] * sumB + emission;
        }

        const size_t historySlot = t % state.historyLength;
        const val_t* localRadiosity = state.localCount == 0 ? nullptr :
            state.radB.data() + state.idxTN(t, 0);
        MPI_Allgatherv(localRadiosity,
                       static_cast<int>(state.localCount), MPI_FLOAT,
                       state.radHistory.data() + historySlot * state.numTriangles,
                       state.rowCounts.data(), state.rowDisplacements.data(), MPI_FLOAT,
                       communicator);
        state.sourceRadB[t] = state.radHistory[
            historySlot * state.numTriangles + state.sourceIndex];

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, int rank) {
    if (rank == 0) {
        printf("Computing distances via cross-correlation...\n");
    }

    for (size_t localI = 0; localI < state.localCount; ++localI) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                const val_t pB = state.radB[state.idxTN(tt, localI)];
                const val_t pS = state.sourceRadB[tt - t];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

void gatherDistances(SimulationState& state, int rank, MPI_Comm communicator) {
    if (rank == 0) {
        state.globalDistances.resize(state.numTriangles);
    }
    MPI_Gatherv(state.distances.data(), static_cast<int>(state.localCount), MPI_FLOAT,
                rank == 0 ? state.globalDistances.data() : nullptr,
                state.rowCounts.data(), state.rowDisplacements.data(), MPI_FLOAT,
                0, communicator);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, int rank, MPI_Comm communicator) {
    int localNonNegative = 1;
    int localFinite = 1;
    val_t localMinDist = std::numeric_limits<val_t>::max();
    val_t localMaxDist = std::numeric_limits<val_t>::lowest();
    val_t localSumDist = ZERO;
    uint64_t localNonZeroDistances = 0;

    for (val_t distance : state.distances) {
        if (distance < ZERO) localNonNegative = 0;
        if (!std::isfinite(distance)) localFinite = 0;
        localMinDist = std::min(localMinDist, distance);
        localMaxDist = std::max(localMaxDist, distance);
        localSumDist += distance;
        if (distance > EPSILON) ++localNonZeroDistances;
    }

    val_t localSourceDistance = ZERO;
    if (state.sourceIndex >= state.localBegin &&
        state.sourceIndex < state.localBegin + state.localCount) {
        localSourceDistance = state.distances[state.sourceIndex - state.localBegin];
    }

    uint64_t localReceivedEnergy = 0;
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, localI)] > EPSILON) {
                ++localReceivedEnergy;
                break;
            }
        }
    }

    uint64_t localNonZeroKij = 0;
    for (val_t value : state.kij) {
        if (value > EPSILON) ++localNonZeroKij;
    }

    int allNonNegative = 0;
    int allFinite = 0;
    val_t minDist = ZERO;
    val_t maxDist = ZERO;
    val_t sumDist = ZERO;
    val_t sourceDistance = ZERO;
    uint64_t nonZeroDistances = 0;
    uint64_t receivedEnergy = 0;
    uint64_t nonZeroKij = 0;
    MPI_Allreduce(&localNonNegative, &allNonNegative, 1, MPI_INT, MPI_MIN, communicator);
    MPI_Allreduce(&localFinite, &allFinite, 1, MPI_INT, MPI_MIN, communicator);
    MPI_Allreduce(&localMinDist, &minDist, 1, MPI_FLOAT, MPI_MIN, communicator);
    MPI_Allreduce(&localMaxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, communicator);
    MPI_Allreduce(&localSumDist, &sumDist, 1, MPI_FLOAT, MPI_SUM, communicator);
    MPI_Allreduce(&localSourceDistance, &sourceDistance, 1, MPI_FLOAT, MPI_SUM, communicator);
    MPI_Allreduce(&localNonZeroDistances, &nonZeroDistances, 1, MPI_UINT64_T, MPI_SUM, communicator);
    MPI_Allreduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_UINT64_T, MPI_SUM, communicator);
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, communicator);

    const bool valid = allNonNegative != 0 && allFinite != 0 &&
                       receivedEnergy != 0 && nonZeroKij != 0;
    if (rank != 0) return valid;

    printf("\nValidation:\n");
    if (!allNonNegative) {
        printf("  ERROR: A negative distance was produced\n");
    }
    if (!allFinite) {
        printf("  ERROR: A non-finite distance was produced\n");
    }
    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %llu/%zu\n",
           static_cast<unsigned long long>(nonZeroDistances), state.numTriangles);

    if (sourceDistance > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", sourceDistance);
    }

    printf("  Triangles receiving energy: %llu/%zu\n",
           static_cast<unsigned long long>(receivedEnergy), state.numTriangles);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
    }

    const size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), totalKij,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalKij));
    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
    }

    if (valid) {
        printf("  Validation: PASSED\n");
    }
    return valid;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&distances[i]);
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

long synchronizedElapsedMilliseconds(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end,
    MPI_Comm communicator) {
    const long localMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    long maximumMilliseconds = 0;
    MPI_Reduce(&localMilliseconds, &maximumMilliseconds, 1, MPI_LONG, MPI_MAX,
               0, communicator);
    return maximumMilliseconds;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    if (timesteps <= 0 || sourceIdx < 0) {
        if (rank == 0) {
            printf("The timestep count must be positive and the source index must be non-negative.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n", worldSize);
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
                         static_cast<size_t>(sourceIdx), reflectivity, rank, worldSize);

    if (rank == 0) printf("\n");

    // Precomputation
    if (rank == 0) {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij) across %d MPI ranks...\n", worldSize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    RandomGenerator localRng = initializeLocalRandomStream(state, rank, worldSize, MPI_COMM_WORLD);
    computeFormFactors(state, localRng);
    initializeRadiosityHistory(state, MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    const long preDuration = synchronizedElapsedMilliseconds(startPre, endPre, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, rank, MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    const long simDuration = synchronizedElapsedMilliseconds(startSim, endSim, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, rank);
    gatherDistances(state, rank, MPI_COMM_WORLD);

    auto endDist = std::chrono::high_resolution_clock::now();
    const long distDuration = synchronizedElapsedMilliseconds(startDist, endDist, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    const long totalTime = preDuration + simDuration + distDuration;

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    const unsigned long long localStorage = static_cast<unsigned long long>(
        state.kij.size() * sizeof(val_t) + state.tau.size() * sizeof(tau_t) +
        state.radB.size() * sizeof(val_t) + state.radHistory.size() * sizeof(val_t) +
        state.sourceRadB.size() * sizeof(val_t));
    unsigned long long maxRankStorage = 0;
    MPI_Reduce(&localStorage, &maxRankStorage, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
        printf("  Maximum rank working memory: %.2f MB\n",
               maxRankStorage / (1024.0 * 1024.0));

        const uint64_t hash = computeHash(state.globalDistances);
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.globalDistances.begin(), state.globalDistances.end());
            print_results(distData, "Distances");
        }
    }

    const bool valid = !validate || validateResults(state, rank, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
