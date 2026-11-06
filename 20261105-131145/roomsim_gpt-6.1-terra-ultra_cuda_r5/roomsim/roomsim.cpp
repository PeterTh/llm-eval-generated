/**
 * Room Response Simulation Benchmark
 * 
 * This implementation uses CUDA for room impulse response
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
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
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

void cudaPrecompute(SimulationState& state);
void cudaRunSimulation(SimulationState& state);
void cudaComputeDistances(SimulationState& state);

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

    cudaPrecompute(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    cudaRunSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    cudaComputeDistances(state);

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

// CUDA implementations of propagation and distance estimation.  The matrix
// transpose makes adjacent receiver threads read adjacent values for each j,
// while each thread still sums contributions in the original order.
static void roomCudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class RoomDeviceArray {
public:
    T* data = nullptr;
    explicit RoomDeviceArray(size_t count) {
        roomCudaCheck(cudaMalloc(reinterpret_cast<void**>(&data),
                                 std::max(size_t(1), count) * sizeof(T)), "cudaMalloc");
    }
    ~RoomDeviceArray() { if (data) cudaFree(data); }
    RoomDeviceArray(const RoomDeviceArray&) = delete;
    RoomDeviceArray& operator=(const RoomDeviceArray&) = delete;
};

__global__ void roomTransposeMatrices(float* kij, int* tau, size_t n) {
    // Each block owns a pair of off-diagonal tiles, so both matrices can be
    // transposed without allocating another N-by-N device buffer.
    if (blockIdx.x < blockIdx.y) return;
    __shared__ float kTileA[32][33], kTileB[32][33];
    __shared__ int tTileA[32][33], tTileB[32][33];
    const size_t x = blockIdx.x * 32 + threadIdx.x;
    const size_t y = blockIdx.y * 32 + threadIdx.y;
    const size_t x2 = blockIdx.y * 32 + threadIdx.x;
    const size_t y2 = blockIdx.x * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8) {
        if (x < n && y + k < n) {
            kTileA[threadIdx.y + k][threadIdx.x] = kij[(y + k) * n + x];
            tTileA[threadIdx.y + k][threadIdx.x] = tau[(y + k) * n + x];
        }
        if (blockIdx.x != blockIdx.y && x2 < n && y2 + k < n) {
            kTileB[threadIdx.y + k][threadIdx.x] = kij[(y2 + k) * n + x2];
            tTileB[threadIdx.y + k][threadIdx.x] = tau[(y2 + k) * n + x2];
        }
    }
    __syncthreads();
    for (int k = 0; k < 32; k += 8) {
        if (x2 < n && y2 + k < n) {
            kij[(y2 + k) * n + x2] = kTileA[threadIdx.x][threadIdx.y + k];
            tau[(y2 + k) * n + x2] = tTileA[threadIdx.x][threadIdx.y + k];
        }
        if (blockIdx.x != blockIdx.y && x < n && y + k < n) {
            kij[(y + k) * n + x] = kTileB[threadIdx.x][threadIdx.y + k];
            tau[(y + k) * n + x] = tTileB[threadIdx.x][threadIdx.y + k];
        }
    }
}

__global__ void roomPropagationStep(const float* kij, const int* tau,
                                    const float* areas, const float* rho,
                                    const float* emission, float* radiosity,
                                    size_t n, size_t t) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        const size_t matrixIndex = j * n + i;
        const int delay = tau[matrixIndex];
        if (static_cast<int>(t) < delay) continue;
        const float factor = kij[matrixIndex];
        if (factor <= 0.0f) continue;
        const float radJ = radiosity[(t - static_cast<size_t>(delay)) * n + j];
        if (radJ <= 0.0f) continue;
        const float product = factor * areas[j];
        // std::min(product, 1.0f) returns product for unordered comparisons.
        const float weight = 1.0f < product ? 1.0f : product;
        sum += weight * radJ;
    }
    radiosity[t * n + i] = rho[i] * sum + emission[t * n + i];
}

void cudaRunSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    const size_t count = n * n;
    const size_t radCount = n * state.numTimesteps;
    RoomDeviceArray<float> dKijT(count), dAreas(n), dRho(n), dEmission(radCount), dRad(radCount);
    RoomDeviceArray<int> dTauT(count);
    roomCudaCheck(cudaMemcpy(dKijT.data, state.kij.data(), count * sizeof(float), cudaMemcpyHostToDevice), "copy form factors");
    roomCudaCheck(cudaMemcpy(dTauT.data, state.tau.data(), count * sizeof(int), cudaMemcpyHostToDevice), "copy delays");
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>((n + 31) / 32));
    roomTransposeMatrices<<<grid, block>>>(dKijT.data, dTauT.data, n);
    roomCudaCheck(cudaGetLastError(), "transpose form factors and delays");
    roomCudaCheck(cudaMemcpy(dAreas.data, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy areas");
    roomCudaCheck(cudaMemcpy(dRho.data, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copy reflectivity");
    if (radCount) {
        roomCudaCheck(cudaMemcpy(dEmission.data, state.radE.data(), radCount * sizeof(float), cudaMemcpyHostToDevice), "copy emission");
        roomCudaCheck(cudaMemcpy(dRad.data, state.radB.data(), radCount * sizeof(float), cudaMemcpyHostToDevice), "copy radiosity");
    }
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        roomPropagationStep<<<blocks, 256>>>(dKijT.data, dTauT.data, dAreas.data, dRho.data,
                                             dEmission.data, dRad.data, n, t);
        roomCudaCheck(cudaGetLastError(), "wave propagation kernel");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
    }
    if (radCount) roomCudaCheck(cudaMemcpy(state.radB.data(), dRad.data, radCount * sizeof(float), cudaMemcpyDeviceToHost), "copy radiosity results");
}

__global__ void roomCorrelations(const float* radiosity, float* correlations,
                                 size_t n, size_t steps, size_t source, size_t lagOffset) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t lag = lagOffset + blockIdx.y;
    if (i >= n || lag >= steps) return;
    float sum = 0.0f;
    for (size_t tt = lag; tt < steps; ++tt) {
        float pB = radiosity[tt * n + i];
        float pS = radiosity[(tt - lag) * n + source];
        sum += pS * pB;
    }
    correlations[lag * n + i] = sum;
}

__global__ void roomBestCorrelations(const float* correlations, float* distances,
                                     size_t n, size_t steps) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float maxCorr = 0.0f;
    int best = 0;
    for (size_t lag = 0; lag < steps; ++lag) {
        const float corr = correlations[lag * n + i];
        if (corr > maxCorr) {
            maxCorr = corr;
            best = static_cast<int>(lag);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(best);
}

void cudaComputeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles;
    const size_t steps = state.numTimesteps;
    const size_t count = n * steps;
    RoomDeviceArray<float> dRad(count), dCorrelations(count), dDistances(n);
    if (count) roomCudaCheck(cudaMemcpy(dRad.data, state.radB.data(), count * sizeof(float), cudaMemcpyHostToDevice), "copy radiosity for correlation");
    const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
    // Split lag dimension into chunks to accommodate CUDA's 65535 grid-y limit.
    // Typical benchmark time series fit in a single launch.
    for (size_t lagOffset = 0; lagOffset < steps; lagOffset += 65535) {
        dim3 grid(blocks, static_cast<unsigned>(std::min(size_t(65535), steps - lagOffset)));
        roomCorrelations<<<grid, 256>>>(dRad.data, dCorrelations.data, n, steps, state.sourceIndex, lagOffset);
        roomCudaCheck(cudaGetLastError(), "correlation kernel");
    }
    roomBestCorrelations<<<blocks, 256>>>(dCorrelations.data, dDistances.data, n, steps);
    roomCudaCheck(cudaGetLastError(), "select correlation peaks");
    roomCudaCheck(cudaMemcpy(state.distances.data(), dDistances.data, n * sizeof(float), cudaMemcpyDeviceToHost), "copy distances");
}

// CUDA implementation of form-factor and propagation-delay precomputation.
namespace roomsim_cuda_precompute {

inline void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

struct Node {
    Vec3 center, halfExtent;
    int children[8];
    unsigned first, count;
};

int flatten(const Octree& tree, std::vector<Node>& nodes, std::vector<unsigned>& indices) {
    const int index = static_cast<int>(nodes.size());
    Node node{};
    node.center = tree.center;
    node.halfExtent = tree.halfExtent;
    node.first = static_cast<unsigned>(indices.size());
    node.count = static_cast<unsigned>(tree.triangleIndices.size());
    for (size_t tri : tree.triangleIndices) indices.push_back(static_cast<unsigned>(tri));
    for (int& child : node.children) child = -1;
    nodes.push_back(node);
    for (int c = 0; c < 8; ++c) {
        if (tree.children[c]) {
            const int child = flatten(*tree.children[c], nodes, indices);
            nodes[index].children[c] = child;
        }
    }
    return index;
}

struct V { float x, y, z; };
__device__ inline V makev(float x, float y, float z) { return {x,y,z}; }
__device__ inline V from(const Vec3& x) { return {x.x, x.y, x.z}; }
__device__ inline V add(V a, V b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ inline V sub(V a, V b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ inline V mul(V a, float s) { return {a.x*s,a.y*s,a.z*s}; }
__device__ inline V divide(V a, float s) { return {a.x/s,a.y/s,a.z/s}; }
__device__ inline float dot(V a, V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ inline V cross(V a, V b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
__device__ inline float norm(V a) { return sqrtf(dot(a,a)); }

__device__ inline V random_point(const Triangle& tr, const float* random) {
    float u = random[0], v = random[1];
    if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    V a=from(tr.a);
    return add(add(a, mul(sub(from(tr.b), a), u)), mul(sub(from(tr.c), a), v));
}

__device__ inline bool intersects_box(V p1, V p2, const Node& n) {
    V d = mul(sub(p2,p1), 0.5f);
    V c = sub(add(p1,d),from(n.center));
    V ad = makev(fabsf(d.x),fabsf(d.y),fabsf(d.z));
    V h = from(n.halfExtent);
    if (fabsf(c.x)>h.x+ad.x || fabsf(c.y)>h.y+ad.y || fabsf(c.z)>h.z+ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y)>h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z)>h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x)>h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ inline bool hit_triangle(V orig, V dir, const Triangle& tr, float raylen) {
    V v0=from(tr.a), e1=sub(from(tr.b),v0), e2=sub(from(tr.c),v0);
    V pvec=cross(dir,e2);
    float det=dot(e1,pvec);
    if (fabsf(det)<EPSILON) return false;
    float inv=1.0f/det;
    V tv=sub(orig,v0);
    float u=dot(tv,pvec)*inv;
    if (u<0.0f || u>1.0f) return false;
    V q=cross(tv,e1);
    float v=dot(dir,q)*inv;
    if (v<0.0f || u+v>1.0f) return false;
    float distance=dot(e2,q)*inv;
    return distance>EPSILON && distance<raylen-EPSILON;
}

__device__ inline bool blocked(V p1, V p2, unsigned src, unsigned dst,
                               const Node* nodes, const unsigned* indices,
                               const Triangle* triangles) {
    V delta=sub(p2,p1);
    float len=norm(delta);
    if (len<EPSILON) return true;
    V dir=divide(delta,len);
    // The octree stops subdividing before depth eight. At most seven siblings
    // per level can be pending during this depth-first traversal.
    int stack[64];
    int top=0;
    stack[top++]=0;
    while (top) {
        const Node& node=nodes[stack[--top]];
        if (node.count) {
            for (unsigned k=0; k<node.count; ++k) {
                unsigned idx=indices[node.first+k];
                if (idx!=src && idx!=dst && hit_triangle(p1,dir,triangles[idx],len)) return true;
            }
        } else {
            for (int c=7; c>=0; --c) {
                int child=node.children[c];
                if (child>=0 && intersects_box(p1,p2,nodes[child])) stack[top++]=child;
            }
        }
    }
    return false;
}

__global__ void precompute(const Triangle* triangles, const Node* nodes,
                           const unsigned* indices, const float* random,
                           const unsigned char* active, size_t start, size_t count,
                           size_t n, float* kij, int* tau) {
    size_t local=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (local>=count) return;
    size_t pair=start+local;
    size_t i=pair/n, j=pair%n;
    if (i==j) { kij[pair]=0.0f; tau[pair]=0; return; }
    Triangle a=triangles[i], b=triangles[j];
    V ca=divide(add(add(from(a.a),from(a.b)),from(a.c)),3.0f);
    V cb=divide(add(add(from(b.a),from(b.b)),from(b.c)),3.0f);
    tau[pair]=static_cast<int>(ceilf(norm(sub(ca,cb))*INV_WAVE_SPEED));
    if (!active[local]) { kij[pair]=0.0f; return; }
    float sum=0.0f;
    V na=from(a._normal), nb=from(b._normal);
    const float* rnd=random+local*(NUM_RAYS*4);
    for (int r=0; r<NUM_RAYS; ++r) {
        V pi=random_point(a,rnd+4*r), pj=random_point(b,rnd+4*r+2);
        if (blocked(pi,pj,static_cast<unsigned>(i),static_cast<unsigned>(j),nodes,indices,triangles)) continue;
        V v=sub(pj,pi);
        float dist2=dot(v,v);
        if (dist2<EPSILON) continue;
        float length=norm(v);
        float ci=length<=EPSILON ? 0.0f : fmaxf(0.0f,dot(v,na)/length);
        V neg=mul(v,-1.0f);
        float cj=length<=EPSILON ? 0.0f : fmaxf(0.0f,dot(neg,nb)/length);
        if (ci<=0.0f || cj<=0.0f) continue;
        sum+=(ci*cj)/(PI*dist2);
    }
    kij[pair]=sum*INV_NUM_RAYS;
}

} // namespace roomsim_cuda_precompute

void cudaPrecompute(SimulationState& state) {
    using namespace roomsim_cuda_precompute;
    printf("Computing time delays (Tau) and form factors (Kij) on CUDA...\n");
    const size_t n=state.numTriangles, pairs=n*n;
    std::vector<Node> nodes;
    std::vector<unsigned> indices;
    flatten(state.octree,nodes,indices);
    Triangle* dtri=nullptr;
    Node* dnodes=nullptr;
    unsigned* dindices=nullptr;
    float *dkij=nullptr;
    int* dtau=nullptr;
    float* drandom[2]={nullptr,nullptr};
    float* random[2]={nullptr,nullptr};
    unsigned char* dactive[2]={nullptr,nullptr};
    unsigned char* active[2]={nullptr,nullptr};
    cudaStream_t streams[2];
    const size_t batch=std::min<size_t>(65536,pairs);
    check(cudaMalloc(reinterpret_cast<void**>(&dtri),n*sizeof(Triangle)),"allocate triangles");
    check(cudaMalloc(reinterpret_cast<void**>(&dnodes),nodes.size()*sizeof(Node)),"allocate octree");
    check(cudaMalloc(reinterpret_cast<void**>(&dindices),indices.size()*sizeof(unsigned)),"allocate octree indices");
    check(cudaMalloc(reinterpret_cast<void**>(&dkij),pairs*sizeof(float)),"allocate form factors");
    check(cudaMalloc(reinterpret_cast<void**>(&dtau),pairs*sizeof(int)),"allocate time delays");
    for (int b=0; b<2; ++b) {
        check(cudaStreamCreateWithFlags(&streams[b],cudaStreamNonBlocking),"create precomputation stream");
        check(cudaMalloc(reinterpret_cast<void**>(&drandom[b]),batch*NUM_RAYS*4*sizeof(float)),"allocate random values");
        check(cudaMalloc(reinterpret_cast<void**>(&dactive[b]),batch*sizeof(unsigned char)),"allocate pair flags");
        check(cudaMallocHost(reinterpret_cast<void**>(&random[b]),batch*NUM_RAYS*4*sizeof(float)),"allocate pinned random values");
        check(cudaMallocHost(reinterpret_cast<void**>(&active[b]),batch*sizeof(unsigned char)),"allocate pinned pair flags");
    }
    check(cudaMemcpy(dtri,state.triangles.data(),n*sizeof(Triangle),cudaMemcpyHostToDevice),"copy triangles");
    check(cudaMemcpy(dnodes,nodes.data(),nodes.size()*sizeof(Node),cudaMemcpyHostToDevice),"copy octree");
    check(cudaMemcpy(dindices,indices.data(),indices.size()*sizeof(unsigned),cudaMemcpyHostToDevice),"copy octree indices");
    RandomGenerator rng(42);
    for (size_t start=0; start<pairs; start+=batch) {
        const size_t count=std::min(batch,pairs-start);
        const int b=static_cast<int>((start/batch)%2);
        // Wait until both the transfer and kernel have finished using this slot.
        check(cudaStreamSynchronize(streams[b]),"wait for precomputation batch");
        for (size_t x=0; x<count; ++x) {
            size_t pair=start+x, i=pair/n, j=pair%n;
            bool valid=i!=j && !(state.triangles[i].normal().dot(state.triangles[j].normal())>0.99f);
            active[b][x]=static_cast<unsigned char>(valid);
            if (valid) for (int k=0; k<NUM_RAYS*4; ++k) random[b][x*NUM_RAYS*4+k]=rng.rand();
        }
        check(cudaMemcpyAsync(drandom[b],random[b],count*NUM_RAYS*4*sizeof(float),cudaMemcpyHostToDevice,streams[b]),"copy random values");
        check(cudaMemcpyAsync(dactive[b],active[b],count*sizeof(unsigned char),cudaMemcpyHostToDevice,streams[b]),"copy pair flags");
        precompute<<<static_cast<unsigned>((count+127)/128),128,0,streams[b]>>>(dtri,dnodes,dindices,drandom[b],dactive[b],start,count,n,dkij,dtau);
        check(cudaGetLastError(),"launch precomputation");
    }
    for (int b=0; b<2; ++b)
        check(cudaStreamSynchronize(streams[b]),"finish precomputation");
    check(cudaMemcpy(state.kij.data(),dkij,pairs*sizeof(float),cudaMemcpyDeviceToHost),"copy form factors");
    check(cudaMemcpy(state.tau.data(),dtau,pairs*sizeof(int),cudaMemcpyDeviceToHost),"copy time delays");
    check(cudaFree(dtri),"free triangles");
    check(cudaFree(dnodes),"free octree");
    check(cudaFree(dindices),"free octree indices");
    check(cudaFree(dkij),"free form factors");
    check(cudaFree(dtau),"free time delays");
    for (int b=0; b<2; ++b) {
        check(cudaFree(drandom[b]),"free random values");
        check(cudaFree(dactive[b]),"free pair flags");
        check(cudaFreeHost(random[b]),"free pinned random values");
        check(cudaFreeHost(active[b]),"free pinned pair flags");
        check(cudaStreamDestroy(streams[b]),"destroy precomputation stream");
    }
}
