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
#include <vector>

#include <mpi.h>

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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    // Skip ahead in the underlying engine stream (used by MPI ranks to start
    // at the exact position the sequential sweep would have reached).
    void discard(unsigned long long z) { rng.discard(z); }
    bool operator==(const RandomGenerator& o) const { return rng == o.rng; }
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

// Cull triangles facing the same direction (no random numbers are consumed
// for culled pairs; the MPI skip-ahead bookkeeping relies on this test).
inline bool formFactorCulled(const Vec3& normalI, const Vec3& normalJ) {
    return normalI.dot(normalJ) > 0.99f;
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (formFactorCulled(triI.normal(), triJ.normal())) return ZERO;

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
//
// MPI decomposition: the N triangles (rows of Kij / Tau) are distributed
// cyclically over the ranks (global row i belongs to rank i mod P, local row
// il of rank r is global row r + il * P).  Consecutive triangle indices are
// spatially clustered, so the cyclic interleaving gives every rank a
// representative sample of the mesh and keeps the per-row visibility-test
// cost balanced.  Each rank owns the Kij/Tau rows and the emission
// radiosity of its own triangles.  The reflected radiosity of all triangles
// for every timestep is replicated on all ranks (everybody needs it as
// input) and assembled with one Allgatherv per timestep.  Geometry (mesh,
// octree, areas, reflectivity) is cheap and deterministic and is therefore
// simply rebuilt on every rank.

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    // MPI topology / row decomposition
    int rank = 0;
    int nprocs = 1;
    size_t numLocal = 0;            // Number of rows owned by this rank
    std::vector<int> rowCounts;     // Rows owned by each rank
    std::vector<int> rowDispls;     // Offset of each rank's rows in gathered (rank-ordered) buffers
    std::vector<idx_t> gatherToGlobal;  // Position in a gathered buffer -> global row index
    std::vector<val_t> gatherBuf;   // Scratch buffer for rank-ordered gathers (N)

    std::vector<Triangle> triangles;  // Replicated
    std::vector<val_t> areas;       // Area of each triangle (replicated)
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0) (replicated)
    std::vector<val_t> kij;         // Form factors (numLocal x N matrix, row-major)
    std::vector<int> tau;           // Time delays (numLocal x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x numLocal matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, replicated)
    std::vector<val_t> distances;   // Computed distances from source (N, replicated)

    Octree octree;                  // Spatial acceleration structure (replicated)

    size_t sourceIndex;

    // Row-local index helpers (il = local row index, j = global column index)
    size_t globalRow(size_t il) const { return static_cast<size_t>(rank) + il * static_cast<size_t>(nprocs); }
    bool ownsRow(size_t i) const { return i % static_cast<size_t>(nprocs) == static_cast<size_t>(rank); }
    size_t localRow(size_t i) const { return i / static_cast<size_t>(nprocs); }
    size_t idxL2d(size_t il, size_t j) const { return il * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
    size_t idxTL(size_t t, size_t il) const { return t * numLocal + il; }
};

// ============================================================================
// Initialization
// ============================================================================

static void setupDecomposition(SimulationState& state) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.nprocs);

    const size_t n = state.numTriangles;
    const size_t p = static_cast<size_t>(state.nprocs);
    state.rowCounts.resize(p);
    state.rowDispls.resize(p);
    state.gatherToGlobal.resize(n);
    size_t pos = 0;
    for (size_t r = 0; r < p; ++r) {
        size_t cnt = (n > r) ? (n - r + p - 1) / p : 0;
        state.rowDispls[r] = static_cast<int>(pos);
        state.rowCounts[r] = static_cast<int>(cnt);
        for (size_t il = 0; il < cnt; ++il) {
            state.gatherToGlobal[pos + il] = static_cast<idx_t>(r + il * p);
        }
        pos += cnt;
    }
    state.numLocal = static_cast<size_t>(state.rowCounts[state.rank]);
    state.gatherBuf.resize(n);
}

// Gather one value per local row from all ranks and scatter the result into
// a global-order array (dst[i] for every global row i).
static void allgatherRows(const SimulationState& state, const val_t* local, val_t* dst,
                          std::vector<val_t>& scratch) {
    MPI_Allgatherv(local, static_cast<int>(state.numLocal), MPI_FLOAT,
                   scratch.data(), state.rowCounts.data(), state.rowDispls.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
    for (size_t pos = 0; pos < state.numTriangles; ++pos) {
        dst[state.gatherToGlobal[pos]] = scratch[pos];
    }
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (identical on all ranks)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    setupDecomposition(state);

    if (state.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("MPI ranks: %d (%zu-%zu rows per rank)\n", state.nprocs,
               state.numTriangles / static_cast<size_t>(state.nprocs),
               (state.numTriangles + static_cast<size_t>(state.nprocs) - 1) / static_cast<size_t>(state.nprocs));
    }

    // Build octree for spatial acceleration
    if (state.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (row-distributed)
    state.kij.resize(state.numLocal * state.numTriangles, ZERO);
    state.tau.resize(state.numLocal * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numLocal, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    if (state.ownsRow(state.sourceIndex)) {
        size_t srcLocal = state.localRow(state.sourceIndex);
        size_t timeOn = 0;
        size_t timeOff = timesteps / 2;
        for (size_t t = timeOn; t < timeOff; ++t) {
            state.radE[state.idxTL(t, srcLocal)] = 1.0f;
        }
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Determine how many raw engine outputs a single RandomGenerator::rand() call
// consumes, so that ranks can skip ahead in the shared random stream exactly.
static unsigned long long engineStepsPerRand() {
    for (unsigned long long k = 1; k <= 8; ++k) {
        std::mt19937 a(42), b(42);
        std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
        for (int i = 0; i < 16; ++i) (void)dist(a);
        b.discard(16 * k);
        if (a == b) return k;
    }
    return 0;
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;

    // The sequential reference draws random numbers from a single mt19937
    // stream while sweeping (i, j) in row-major order.  Every non-culled pair
    // (i != j) consumes exactly the same number of engine outputs, so the
    // stream position at the start of every row is known in advance.  Each
    // rank skips ahead to the start of each of its rows, which keeps the
    // results bit-identical to the sequential code.
    const unsigned long long stepsPerRand = engineStepsPerRand();
    if (stepsPerRand == 0) {
        if (state.rank == 0)
            fprintf(stderr, "ERROR: unable to determine RNG consumption per draw\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const unsigned long long stepsPerPair = 2ULL * 2ULL * NUM_RAYS * stepsPerRand;

    // Count non-culled pairs in the local rows (stored as float so the
    // generic row gather can be reused; a count is < N <= 81920 < 2^24, so
    // it is represented exactly)
    std::vector<val_t> localPairs(state.numLocal, ZERO);
    for (size_t il = 0; il < state.numLocal; ++il) {
        const size_t i = state.globalRow(il);
        const Vec3 ni = state.triangles[i].normal();
        unsigned cnt = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!formFactorCulled(ni, state.triangles[j].normal())) ++cnt;
        }
        localPairs[il] = static_cast<val_t>(cnt);
    }

    // Stream position (in engine outputs) at the start of every global row
    std::vector<val_t> pairsPerRow(n, ZERO);
    allgatherRows(state, localPairs.data(), pairsPerRow.data(), state.gatherBuf);
    std::vector<unsigned long long> rowOffset(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        rowOffset[i + 1] = rowOffset[i] +
            static_cast<unsigned long long>(pairsPerRow[i]) * stepsPerPair;
    }

    RandomGenerator rng(42);
    unsigned long long streamPos = 0;
    bool verified = false;

    for (size_t il = 0; il < state.numLocal; ++il) {
        const size_t i = state.globalRow(il);
        val_t* kijRow = &state.kij[state.idxL2d(il, 0)];

        // Skip ahead to the start of this row
        rng.discard(rowOffset[i] - streamPos);
        RandomGenerator rngCheck = rng;

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            kijRow[j] = computeKij(i, j, state.triangles, state.octree, rng);
        }
        streamPos = rowOffset[i + 1];

        // Verify once that the stream advanced exactly as predicted (guards the
        // skip-ahead bookkeeping against any change in RNG consumption).
        if (!verified && rowOffset[i + 1] > rowOffset[i]) {
            rngCheck.discard(rowOffset[i + 1] - rowOffset[i]);
            if (!(rngCheck == rng)) {
                fprintf(stderr, "ERROR: rank %d: RNG stream bookkeeping mismatch\n", state.rank);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
            verified = true;
        }

        if (state.rank == 0 && ((il + 1) % 100 == 0 || il + 1 == state.numLocal)) {
            printf("  Progress: %zu/%zu triangles (rank 0)\n", il + 1, state.numLocal);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    for (size_t il = 0; il < state.numLocal; ++il) {
        const size_t i = state.globalRow(il);
        int* tauRow = &state.tau[state.idxL2d(il, 0)];
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            tauRow[j] = computeTau(state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    const size_t n = state.numTriangles;
    const val_t* areas = state.areas.data();
    std::vector<val_t> localB(state.numLocal, ZERO);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        val_t* radBt = &state.radB[state.idxTN(t, 0)];
        const int ti = static_cast<int>(t);

        for (size_t il = 0; il < state.numLocal; ++il) {
            const size_t i = state.globalRow(il);
            const int* tauRow = &state.tau[state.idxL2d(il, 0)];
            const val_t* kijRow = &state.kij[state.idxL2d(il, 0)];
            val_t sumB = ZERO;

            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;

                int tauij = tauRow[j];

                // Skip if wave hasn't yet propagated from j to i
                if (ti < tauij) continue;

                val_t kij = kijRow[j];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            localB[il] = state.rho[i] * sumB + state.radE[state.idxTL(t, il)];
        }

        // Assemble the full radiosity row for this timestep on all ranks
        allgatherRows(state, localB.data(), radBt, state.gatherBuf);

        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t T = state.numTimesteps;
    std::vector<val_t> localDist(state.numLocal, ZERO);

    for (size_t il = 0; il < state.numLocal; ++il) {
        const size_t i = state.globalRow(il);
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDist[il] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    // Replicate the distance vector (in global order) on all ranks
    allgatherRows(state, localDist.data(), state.distances.data(), state.gatherBuf);
}

// ============================================================================
// Validation
// ============================================================================

// Collective: must be called by all ranks; the result is identical everywhere.
bool validateResults(const SimulationState& state) {
    // Count non-zero form factors over the distributed Kij matrix
    long long localNonZeroKij = 0;
    for (size_t i = 0; i < state.numLocal * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) localNonZeroKij++;
    }
    long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if (state.rank != 0) {
        int ok = 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return ok != 0;
    }

    int ok = 1;
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles && ok; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            ok = 0;
            break;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    if (ok) {
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
            ok = 0;
        }
    }

    if (ok) {
        // Check Kij matrix (should have some non-zero entries)
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * static_cast<val_t>(nonZeroKij) /
                   static_cast<val_t>(state.numTriangles * state.numTriangles));

        if (nonZeroKij == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
            ok = 0;
        }
    }

    if (ok && !allNonNegative) {
        ok = 0;
    }

    if (ok) printf("  Validation: PASSED\n");

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
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

// Accumulate local work seconds and report maximum phase milliseconds on rank 0.
static long syncedMillis(double start, double& localTotal) {
    double elapsed = MPI_Wtime() - start;
    localTotal += elapsed;
    MPI_Barrier(MPI_COMM_WORLD);
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return static_cast<long>(maxElapsed * 1000.0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
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

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double localTotal = 0.0;
    double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state);

    long preDuration = syncedMillis(startPre, localTotal);

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    double startSim = MPI_Wtime();

    runSimulation(state);

    long simDuration = syncedMillis(startSim, localTotal);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    double startDist = MPI_Wtime();

    computeDistances(state);

    long distDuration = syncedMillis(startDist, localTotal);

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    double maxTotal = 0.0;
    MPI_Reduce(&localTotal, &maxTotal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long totalTime = static_cast<long>(maxTotal * 1000.0);

    if (rank == 0) {
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
        printf("  MPI ranks: %d\n", state.nprocs);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage (aggregate over all ranks; Kij/Tau are row-distributed)
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
    }

    // Validation (collective)
    int exitCode = 0;
    if (validate) {
        if (!validateResults(state)) {
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
