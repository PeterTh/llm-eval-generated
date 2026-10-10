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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
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
    using EngineState = std::array<uint64_t, std::mt19937::state_size + 1>;

    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    void discard(uint64_t count) { rng.discard(count); }

    // Use the standard engine serialization rather than transmitting its
    // implementation-dependent object layout between cluster nodes.
    EngineState save() const {
        std::ostringstream output;
        output << rng;
        std::istringstream input(output.str());
        EngineState state{};
        for (auto& value : state) input >> value;
        if (!input) throw std::runtime_error("Cannot serialize random generator");
        return state;
    }

    void restore(const EngineState& state) {
        std::ostringstream output;
        for (auto value : state) output << value << ' ';
        std::istringstream input(output.str());
        input >> rng;
        if (!input) throw std::runtime_error("Cannot restore random generator");
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
    size_t firstTriangle;
    size_t localTriangles;
    int rank;
    int ranks;
    MPI_Comm communicator;
    std::vector<int> rowCounts;
    std::vector<int> rowOffsets;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Owned rows only; later weighted by emitter area
    std::vector<int> tau;           // Owned rows of the time-delay matrix
    std::vector<val_t> radB;        // Owned triangle histories, triangle-major
    std::vector<val_t> distances;   // Full result on rank zero only
    std::vector<val_t> history;     // Bounded global history for delayed propagation
    size_t historyLength = 0;
    int minDelay = 1;
    int maxDelay = 0;
    uint64_t nonZeroKij = 0;
    uint64_t receivedEnergy = 0;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t localRow) const { return localRow * numTimesteps + t; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int ranks,
                          MPI_Comm communicator) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.rank = rank;
    state.ranks = ranks;
    state.communicator = communicator;

    // Contiguous, balanced receiver rows keep the quadratic matrices distributed
    // and retain the serial summation order over emitter triangles.
    state.rowCounts.resize(ranks);
    state.rowOffsets.resize(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        size_t count = state.numTriangles / ranks +
                       (static_cast<size_t>(r) < state.numTriangles % ranks);
        state.rowCounts[r] = static_cast<int>(count);
        state.rowOffsets[r] = static_cast<int>(offset);
        offset += count;
    }
    state.firstTriangle = state.rowOffsets[rank];
    state.localTriangles = state.rowCounts[rank];

    if (rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.localTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.localTriangles * state.numTriangles, 0);
    state.radB.resize(timesteps * state.localTriangles, ZERO);
    if (rank == 0) state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");

    // Each unculled pair consumes exactly four float draws per ray. Visibility
    // tests never change that count. Count pairs in parallel, then distribute
    // engine states at row-block boundaries to reproduce the original stream.
    static_assert(std::numeric_limits<val_t>::digits <= std::mt19937::word_size);
    uint64_t pairs = 0;
    for (size_t row = 0; row < state.localTriangles; ++row) {
        size_t i = state.firstTriangle + row;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i != j && !(state.triangles[i].normal().dot(
                              state.triangles[j].normal()) > 0.99f)) ++pairs;
        }
    }
    std::vector<uint64_t> pairCounts(state.rank == 0 ? state.ranks : 0);
    MPI_Gather(&pairs, 1, MPI_UINT64_T, pairCounts.data(), 1, MPI_UINT64_T,
               0, state.communicator);

    RandomGenerator rng(42);
    std::vector<RandomGenerator::EngineState> initialStates;
    if (state.rank == 0) {
        initialStates.resize(state.ranks);
        for (int r = 0; r < state.ranks; ++r) {
            initialStates[r] = rng.save();
            if (r + 1 < state.ranks) rng.discard(pairCounts[r] * (4 * NUM_RAYS));
        }
    }
    RandomGenerator::EngineState initialState;
    constexpr int stateWords = static_cast<int>(RandomGenerator::EngineState{}.size());
    MPI_Scatter(initialStates.data(), stateWords, MPI_UINT64_T,
                initialState.data(), stateWords, MPI_UINT64_T, 0, state.communicator);
    rng.restore(initialState);

    for (size_t row = 0; row < state.localTriangles; ++row) {
        size_t i = state.firstTriangle + row;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.idx2d(row, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
        if (state.rank == 0 && ((row + 1) % 100 == 0 || row + 1 == state.localTriangles)) {
            printf("  Rank 0 progress: %zu/%zu assigned triangles\n", row + 1, state.localTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");

    for (size_t row = 0; row < state.localTriangles; ++row) {
        size_t i = state.firstTriangle + row;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(row, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

void preparePropagation(SimulationState& state) {
    int minDelay = std::numeric_limits<int>::max();
    int maxDelay = 0;
    for (size_t row = 0; row < state.localTriangles; ++row) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            size_t index = state.idx2d(row, j);
            val_t kij = state.kij[index];
            if (kij > EPSILON) ++state.nonZeroKij;
            if (kij <= ZERO) continue;
            // Cache the same rounded coefficient used by the original inner loop.
            state.kij[index] = std::min(kij * state.areas[j], ONE);
            int delay = state.tau[index];
            if (static_cast<size_t>(delay) < state.numTimesteps) {
                minDelay = std::min(minDelay, delay);
                maxDelay = std::max(maxDelay, delay);
            }
        }
    }
    int limits[2] = {minDelay, -maxDelay};
    MPI_Allreduce(MPI_IN_PLACE, limits, 2, MPI_INT, MPI_MIN, state.communicator);
    // Distinct triangles in the generated mesh always have a positive delay.
    state.minDelay = limits[0];
    state.maxDelay = -limits[1];
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");

    // No delayed contribution can arrive within this simulation window.
    if (state.maxDelay == 0) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            for (size_t row = 0; row < state.localTriangles; ++row) {
                size_t i = state.firstTriangle + row;
                val_t emission = i == state.sourceIndex && t < state.numTimesteps / 2 ? ONE : ZERO;
                state.radB[state.idxTN(t, row)] = state.rho[i] * ZERO + emission;
            }
            if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
            }
        }
        return;
    }

    // All dependencies precede the current batch, so a single collective can
    // exchange minDelay timesteps without changing the original recurrence.
    size_t batchSize = std::min(static_cast<size_t>(state.minDelay),
                               static_cast<size_t>(std::numeric_limits<int>::max()) / state.numTriangles);
    state.historyLength = std::min(state.numTimesteps,
                                  static_cast<size_t>(state.maxDelay) + batchSize);
    state.history.assign(state.historyLength * state.numTriangles, ZERO);
    std::vector<val_t> incoming(batchSize * state.numTriangles);
    std::vector<const val_t*> delayedRows(state.maxDelay + 1, nullptr);
    std::vector<int> counts(state.ranks), offsets(state.ranks);
    MPI_Datatype sendType = MPI_DATATYPE_NULL;
    size_t previousBatch = 0;

    for (size_t begin = 0; begin < state.numTimesteps; begin += batchSize) {
        size_t batch = std::min(batchSize, state.numTimesteps - begin);
        if (batch != previousBatch) {
            if (sendType != MPI_DATATYPE_NULL) MPI_Type_free(&sendType);
            if (state.localTriangles != 0) {
                // Send each owned triangle's batch directly from its full history.
                MPI_Type_vector(static_cast<int>(state.localTriangles), static_cast<int>(batch),
                                static_cast<int>(state.numTimesteps), MPI_FLOAT, &sendType);
                MPI_Type_commit(&sendType);
            }
            for (int r = 0; r < state.ranks; ++r) {
                counts[r] = state.rowCounts[r] * static_cast<int>(batch);
                offsets[r] = state.rowOffsets[r] * static_cast<int>(batch);
            }
            previousBatch = batch;
        }

        for (size_t t = begin; t < begin + batch; ++t) {
            // Hoist the circular-buffer modulo out of the quadratic inner loop.
            for (int delay = state.minDelay; delay <= state.maxDelay; ++delay) {
                if (static_cast<size_t>(delay) <= t) {
                    delayedRows[delay] = state.history.data() +
                        ((t - static_cast<size_t>(delay)) % state.historyLength) * state.numTriangles;
                }
            }
            for (size_t row = 0; row < state.localTriangles; ++row) {
                size_t i = state.firstTriangle + row;
                val_t sumB = ZERO;
                const int* delays = state.tau.data() + state.idx2d(row, 0);
                const val_t* weights = state.kij.data() + state.idx2d(row, 0);
                for (size_t j = 0; j < state.numTriangles; ++j) {
                    int delay = delays[j];
                    if (static_cast<size_t>(delay) > t || weights[j] <= ZERO) continue;
                    val_t radJ = delayedRows[delay][j];
                    if (radJ <= ZERO) continue;
                    // Retain the original clamp expression and accumulation
                    // order, including its intermediate floating-point rounding.
                    sumB += std::min(weights[j], ONE) * radJ;
                }
                val_t emission = i == state.sourceIndex && t < state.numTimesteps / 2 ? ONE : ZERO;
                state.radB[state.idxTN(t, row)] = state.rho[i] * sumB + emission;
            }
            if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
            }
        }

        // The final batch is already retained locally for cross-correlation.
        if (begin + batch == state.numTimesteps) break;
        MPI_Allgatherv(state.localTriangles ? state.radB.data() + begin : nullptr,
                       state.localTriangles ? 1 : 0,
                       state.localTriangles ? sendType : MPI_FLOAT,
                       incoming.data(), counts.data(), offsets.data(), MPI_FLOAT, state.communicator);
        for (size_t step = 0; step < batch; ++step) {
            val_t* target = state.history.data() + ((begin + step) % state.historyLength) * state.numTriangles;
            for (size_t j = 0; j < state.numTriangles; ++j) {
                target[j] = incoming[j * batch + step];
            }
        }
    }
    if (sendType != MPI_DATATYPE_NULL) MPI_Type_free(&sendType);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");

    int sourceRank = 0;
    while (state.sourceIndex >= static_cast<size_t>(state.rowOffsets[sourceRank] +
                                                  state.rowCounts[sourceRank])) ++sourceRank;
    std::vector<val_t> sourceHistory(state.numTimesteps);
    if (state.rank == sourceRank && state.numTimesteps != 0) {
        const val_t* source = state.radB.data() +
            state.idxTN(0, state.sourceIndex - state.firstTriangle);
        std::copy_n(source, state.numTimesteps, sourceHistory.data());
    }
    MPI_Bcast(sourceHistory.data(), static_cast<int>(state.numTimesteps),
              MPI_FLOAT, sourceRank, state.communicator);

    std::vector<val_t> localDistances(state.localTriangles);
    for (size_t row = 0; row < state.localTriangles; ++row) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, row)];
                val_t pS = sourceHistory[tt - t];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        localDistances[row] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
    MPI_Gatherv(localDistances.data(), state.rowCounts[state.rank], MPI_FLOAT,
                state.distances.data(), state.rowCounts.data(), state.rowOffsets.data(),
                MPI_FLOAT, 0, state.communicator);
}

// ============================================================================
// Validation
// ============================================================================

void collectValidationCounts(SimulationState& state) {
    uint64_t counts[2] = {0, state.nonZeroKij};
    for (size_t row = 0; row < state.localTriangles; ++row) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, row)] > EPSILON) {
                ++counts[0];
                break;
            }
        }
    }
    uint64_t totals[2] = {};
    MPI_Reduce(counts, totals, 2, MPI_UINT64_T, MPI_SUM, 0, state.communicator);
    if (state.rank == 0) {
        state.receivedEnergy = totals[0];
        state.nonZeroKij = totals[1];
    }
}

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
    uint64_t receivedEnergy = state.receivedEnergy;

    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    uint64_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
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

template<typename Function>
long timePhase(Function&& function, int rank, MPI_Comm communicator) {
    MPI_Barrier(communicator);
    double start = MPI_Wtime();
    function();
    double elapsed = MPI_Wtime() - start;
    double maximum = 0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    return rank == 0 ? static_cast<long>(maximum * 1000.0) : 0;
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    int numTriangles = 20;
    for (int level = 0; level < subdivisions; ++level) numTriangles *= 4;
    int activeRanks = std::min(ranks, numTriangles);
    MPI_Comm communicator = MPI_COMM_WORLD;
    if (activeRanks < ranks) {
        // Ranks without receiver rows need no geometry, matrices, or wave data.
        MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED,
                       rank, &communicator);
        if (communicator == MPI_COMM_NULL) {
            int result;
            MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return result;
        }
    }

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        if (activeRanks != ranks) printf("Active MPI ranks: %d\n", activeRanks);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, activeRanks,
                         communicator);

    if (rank == 0) printf("\n");

    // Precomputation
    long preDuration = timePhase([&] {
        computeTimeDelays(state);
        computeFormFactors(state);
        preparePropagation(state);
    }, rank, communicator);
    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    long simDuration = timePhase([&] { runSimulation(state); }, rank, communicator);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    long distDuration = timePhase([&] { computeDistances(state); }, rank, communicator);
    if (validate) collectValidationCounts(state);
    int result = 0;
    if (rank == 0) {
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
        size_t maxRows = state.rowCounts[0];
        size_t memKij = maxRows * n * sizeof(val_t);
        size_t memTau = maxRows * n * sizeof(int);
        size_t memRad = (t * maxRows + state.historyLength * n) * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB per rank (maximum)\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016" PRIX64 "\n", hash);
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
                result = 1;
            }
        }
    }
    if (communicator != MPI_COMM_WORLD) MPI_Comm_free(&communicator);
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result;
    try {
        result = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
