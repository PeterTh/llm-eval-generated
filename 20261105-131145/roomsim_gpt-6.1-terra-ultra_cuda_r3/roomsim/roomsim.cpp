/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light
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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Persistent device matrices shared by the three simulation phases.
static val_t* deviceKij = nullptr;
static int* deviceTau = nullptr;
static val_t* deviceRadB = nullptr;

static void cudaCheck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void launchFormFactorKernel(const SimulationState&, const float*, const size_t*,
                            float*, int*, size_t, size_t);
void cudaRunSimulation(SimulationState&, const float*, const int*, float*);
void cudaComputeDistances(SimulationState&, const float*);

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

void computeFormFactors(SimulationState& state, bool validate) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const size_t total = n * n;
    cudaCheck(cudaMalloc(&deviceKij, total * sizeof(val_t)), "allocate form factors");
    cudaCheck(cudaMalloc(&deviceTau, total * sizeof(int)), "allocate delays");
    cudaCheck(cudaMalloc(&deviceRadB, std::max(size_t(1), state.radB.size()) * sizeof(val_t)), "allocate radiosity");

    // The original generator advances only for pairs that are not culled.  Stage
    // the exact same random sequence in bounded chunks in row-major pair order.
    constexpr size_t chunkPairs = 1u << 20;
    const size_t capacity = std::min(total, chunkPairs);
    std::vector<size_t> offsets(capacity);
    std::vector<val_t> random;
    random.reserve(capacity * NUM_RAYS * 4);
    size_t* deviceOffsets = nullptr;
    val_t* deviceRandom = nullptr;
    cudaCheck(cudaMalloc(&deviceOffsets, capacity * sizeof(size_t)), "allocate offsets");
    cudaCheck(cudaMalloc(&deviceRandom, capacity * NUM_RAYS * 4 * sizeof(val_t)), "allocate random samples");
    RandomGenerator rng(42);
    for (size_t start = 0; start < total; start += capacity) {
        size_t count = std::min(capacity, total - start);
        random.clear();
        for (size_t k = 0; k < count; ++k) {
            size_t pair = start + k;
            size_t i = pair / n, j = pair % n;
            if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) {
                offsets[k] = SIZE_MAX;
            } else {
                offsets[k] = random.size();
                for (int r = 0; r < NUM_RAYS * 4; ++r) random.push_back(rng.rand());
            }
        }
        cudaCheck(cudaMemcpy(deviceOffsets, offsets.data(), count * sizeof(size_t), cudaMemcpyHostToDevice), "copy offsets");
        if (!random.empty())
            cudaCheck(cudaMemcpy(deviceRandom, random.data(), random.size() * sizeof(val_t), cudaMemcpyHostToDevice), "copy random samples");
        launchFormFactorKernel(state, deviceRandom, deviceOffsets, deviceKij, deviceTau, start, count);
    }
    cudaCheck(cudaDeviceSynchronize(), "compute form factors");
    if (validate) {
        state.kij.resize(total);
        cudaCheck(cudaMemcpy(state.kij.data(), deviceKij, total * sizeof(val_t), cudaMemcpyDeviceToHost), "copy form factors");
    }
    cudaCheck(cudaFree(deviceOffsets), "free offsets");
    cudaCheck(cudaFree(deviceRandom), "free random samples");
}

void computeTimeDelays(SimulationState&) {
    printf("Computing time delays (Tau)...\n");
    // Computed together with the form factors on the GPU.
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    cudaRunSimulation(state, deviceKij, deviceTau, deviceRadB);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    cudaComputeDistances(state, deviceRadB);
    cudaCheck(cudaFree(deviceKij), "free form factors");
    cudaCheck(cudaFree(deviceTau), "free delays");
    cudaCheck(cudaFree(deviceRadB), "free radiosity");
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

    computeTimeDelays(state);
    computeFormFactors(state, validate);

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


// CUDA wave propagation and cross-correlation.  All time steps are launched
// in the same stream so a step completes before the next one begins.

static void checkRoomsimCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// A warp cooperates on each row, loading consecutive emitters.  The lane-zero
// accumulation is deliberately in emitter order, matching the serial sum.
__global__ void roomsimWaveStep(const float* kij, const int* tau,
                                const float* areas, const float* rho,
                                const float* radE, float* radB,
                                size_t n, size_t timestep) {
    const unsigned lane = threadIdx.x & 31u;
    const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (i >= n) return;
    float sum = 0.0f;
    for (size_t base = 0; base < n; base += 32) {
        const size_t j = base + lane;
        float contribution = 0.0f;
        if (j < n && i != j) {
            const size_t index = i * n + j;
            const int delay = tau[index];
            if (static_cast<int>(timestep) >= delay) {
                const float factor = kij[index];
                if (factor > 0.0f) {
                    const float rad = radB[(timestep - static_cast<size_t>(delay)) * n + j];
                    if (rad > 0.0f) {
                        contribution = fminf(factor * areas[j], 1.0f) * rad;
                    }
                }
            }
        }
        const unsigned count = static_cast<unsigned>((n - base < 32) ? n - base : 32);
        for (unsigned k = 0; k < count; ++k) {
            float value = __shfl_sync(0xffffffffu, contribution, k);
            if (lane == 0) sum += value;
        }
    }
    if (lane == 0) {
        radB[timestep * n + i] = rho[i] * sum + radE[timestep * n + i];
    }
}

// Each block processes one receiver and each lane evaluates independent lags.
// A thread sums each lag in the original order. The reduction resolves ties in
// favor of the earliest lag, as in the sequential implementation.
__global__ void roomsimDistanceKernel(const float* radB, float* distances,
                                       size_t n, size_t timesteps, size_t source) {
    const size_t i = blockIdx.x;
    float best = 0.0f;
    size_t bestLag = 0;
    for (size_t lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        float sum = 0.0f;
        for (size_t tt = lag; tt < timesteps; ++tt) {
            const float pB = radB[tt * n + i];
            const float pS = radB[(tt - lag) * n + source];
            sum += pS * pB;
        }
        if (sum > best || (sum == best && lag < bestLag)) {
            best = sum;
            bestLag = lag;
        }
    }
    __shared__ float bestValues[128];
    __shared__ size_t bestLags[128];
    bestValues[threadIdx.x] = best;
    bestLags[threadIdx.x] = bestLag;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = bestValues[threadIdx.x + stride];
            const size_t otherLag = bestLags[threadIdx.x + stride];
            if (other > bestValues[threadIdx.x] ||
                (other == bestValues[threadIdx.x] && otherLag < bestLags[threadIdx.x])) {
                bestValues[threadIdx.x] = other;
                bestLags[threadIdx.x] = otherLag;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[i] = WAVE_SPEED * static_cast<float>(bestLags[0]);
}

void cudaRunSimulation(SimulationState& state, const float* d_kij,
                       const int* d_tau, float* d_radB) {
    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;
    float *d_areas = nullptr, *d_rho = nullptr, *d_radE = nullptr;
    checkRoomsimCuda(cudaMalloc(&d_areas, n * sizeof(float)), "allocating areas");
    checkRoomsimCuda(cudaMalloc(&d_rho, n * sizeof(float)), "allocating reflectivity");
    if (t != 0) checkRoomsimCuda(cudaMalloc(&d_radE, t * n * sizeof(float)), "allocating emission");
    checkRoomsimCuda(cudaMemcpy(d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copying areas");
    checkRoomsimCuda(cudaMemcpy(d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice), "copying reflectivity");
    if (t != 0) checkRoomsimCuda(cudaMemcpy(d_radE, state.radE.data(), t * n * sizeof(float), cudaMemcpyHostToDevice), "copying emission");
    constexpr unsigned threads = 128;
    const unsigned blocks = static_cast<unsigned>((n + threads / 32 - 1) / (threads / 32));
    for (size_t step = 0; step < t; ++step) {
        roomsimWaveStep<<<blocks, threads>>>(d_kij, d_tau, d_areas, d_rho, d_radE, d_radB, n, step);
        checkRoomsimCuda(cudaGetLastError(), "launching wave propagation");
    }
    if (t != 0) checkRoomsimCuda(cudaMemcpy(state.radB.data(), d_radB, t * n * sizeof(float), cudaMemcpyDeviceToHost), "copying radiosity");
    checkRoomsimCuda(cudaFree(d_areas), "freeing areas");
    checkRoomsimCuda(cudaFree(d_rho), "freeing reflectivity");
    if (d_radE) checkRoomsimCuda(cudaFree(d_radE), "freeing emission");
}

void cudaComputeDistances(SimulationState& state, const float* d_radB) {
    float* d_distances = nullptr;
    const size_t n = state.numTriangles;
    checkRoomsimCuda(cudaMalloc(&d_distances, n * sizeof(float)), "allocating distances");
    roomsimDistanceKernel<<<static_cast<unsigned>(n), 128>>>(d_radB, d_distances, n,
                                                             state.numTimesteps, state.sourceIndex);
    checkRoomsimCuda(cudaGetLastError(), "launching cross-correlation");
    checkRoomsimCuda(cudaMemcpy(state.distances.data(), d_distances, n * sizeof(float), cudaMemcpyDeviceToHost), "copying distances");
    checkRoomsimCuda(cudaFree(d_distances), "freeing distances");
}


// Flat, stackless octree representation for GPU visibility queries.  Nodes are
// laid out in preorder; escape identifies the first node after each subtree.
#include <cfloat>

struct GpuOctreeNode {
    Vec3 center, halfExtent;
    unsigned int escape, first, count;
};

static void flattenOctree(const Octree& node, std::vector<GpuOctreeNode>& nodes,
                          std::vector<unsigned int>& indices) {
    const size_t pos = nodes.size();
    nodes.push_back({node.center, node.halfExtent, 0, 0, 0});
    if (!node.triangleIndices.empty()) {
        nodes[pos].first = static_cast<unsigned int>(indices.size());
        nodes[pos].count = static_cast<unsigned int>(node.triangleIndices.size());
        for (size_t i : node.triangleIndices) indices.push_back(static_cast<unsigned int>(i));
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) flattenOctree(*node.children[i], nodes, indices);
        }
    }
    nodes[pos].escape = static_cast<unsigned int>(nodes.size());
}

__device__ inline Vec3 gpuAdd(Vec3 a, Vec3 b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline Vec3 gpuSub(Vec3 a, Vec3 b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline Vec3 gpuMul(Vec3 a, float b) { return Vec3(a.x*b, a.y*b, a.z*b); }
__device__ inline Vec3 gpuDiv(Vec3 a, float b) { return Vec3(a.x/b, a.y/b, a.z/b); }
__device__ inline float gpuDot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline Vec3 gpuCross(Vec3 a, Vec3 b) {
    return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ inline float gpuNorm(Vec3 a) { return sqrtf(gpuDot(a,a)); }

__device__ inline bool gpuBox(const GpuOctreeNode& n, Vec3 p1, Vec3 p2) {
    Vec3 d = gpuMul(gpuSub(p2, p1), 0.5f);
    Vec3 c = gpuSub(gpuAdd(p1, d), n.center);
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > n.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > n.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > n.halfExtent.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > n.halfExtent.y*ad.z+n.halfExtent.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > n.halfExtent.z*ad.x+n.halfExtent.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > n.halfExtent.x*ad.y+n.halfExtent.y*ad.x+EPSILON) return false;
    return true;
}

__device__ inline float gpuRayTriangle(Vec3 orig, Vec3 dir, const Triangle& tri) {
    Vec3 e1 = gpuSub(tri.b, tri.a), e2 = gpuSub(tri.c, tri.a);
    Vec3 pvec = gpuCross(dir, e2);
    float det = gpuDot(e1, pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    float invDet = 1.0f / det;
    Vec3 tvec = gpuSub(orig, tri.a);
    float u = gpuDot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    Vec3 qvec = gpuCross(tvec, e1);
    float v = gpuDot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return gpuDot(e2, qvec) * invDet;
}

__device__ bool gpuBlocked(Vec3 from, Vec3 to, unsigned int src, unsigned int dst,
                           const Triangle* triangles, const GpuOctreeNode* nodes,
                           const unsigned int* indices, unsigned int nodeCount) {
    Vec3 dir = gpuSub(to, from);
    float len = gpuNorm(dir);
    if (len < EPSILON) return true;
    Vec3 dirNorm = gpuDiv(dir, len);
    unsigned int n = 0;
    while (n < nodeCount) {
        const GpuOctreeNode& node = nodes[n];
        // The original traversal tests child boxes but never the root box.
        if (n != 0 && !gpuBox(node, from, to)) { n = node.escape; continue; }
        if (node.count) {
            for (unsigned int k = 0; k < node.count; ++k) {
                unsigned int i = indices[node.first+k];
                if (i == src || i == dst) continue;
                float dist = gpuRayTriangle(from, dirNorm, triangles[i]);
                if (dist > EPSILON && dist < len - EPSILON) return true;
            }
            n = node.escape;
        } else {
            ++n;
        }
    }
    return false;
}

__device__ inline Vec3 gpuPoint(const Triangle& t, const float* random) {
    float u = random[0], v = random[1];
    if (u+v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return gpuAdd(gpuAdd(t.a, gpuMul(gpuSub(t.b,t.a),u)), gpuMul(gpuSub(t.c,t.a),v));
}

__global__ void gpuFormFactors(const Triangle* triangles, const GpuOctreeNode* nodes,
                               const unsigned int* indices, unsigned int nodeCount,
                               size_t n, const float* random, const size_t* offsets,
                               float* kij, int* tau, size_t start, size_t count) {
    size_t local = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (local >= count) return;
    size_t pair = start + local;
    unsigned int i = static_cast<unsigned int>(pair/n);
    unsigned int j = static_cast<unsigned int>(pair%n);
    if (i == j) { kij[pair] = 0.0f; tau[pair] = 0; return; }
    Triangle ti = triangles[i], tj = triangles[j];
    Vec3 ci = gpuDiv(gpuAdd(gpuAdd(ti.a,ti.b),ti.c), 3.0f);
    Vec3 cj = gpuDiv(gpuAdd(gpuAdd(tj.a,tj.b),tj.c), 3.0f);
    tau[pair] = static_cast<int>(ceilf(gpuNorm(gpuSub(ci,cj))*INV_WAVE_SPEED));
    size_t offset = offsets[local];
    if (offset == static_cast<size_t>(-1)) { kij[pair] = 0.0f; return; }
    float sum = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pi = gpuPoint(ti, random + offset + r*4);
        Vec3 pj = gpuPoint(tj, random + offset + r*4 + 2);
        if (gpuBlocked(pi, pj, i, j, triangles, nodes, indices, nodeCount)) continue;
        Vec3 v = gpuSub(pj, pi);
        float distSqr = gpuDot(v,v);
        if (distSqr < EPSILON) continue;
        float vNorm = gpuNorm(v);
        float cosI = vNorm <= EPSILON ? 0.0f : fmaxf(0.0f, gpuDot(v,ti._normal)/vNorm);
        Vec3 negv(-v.x,-v.y,-v.z);
        float cosJ = vNorm <= EPSILON ? 0.0f : fmaxf(0.0f, gpuDot(negv,tj._normal)/vNorm);
        if (cosI <= 0.0f || cosJ <= 0.0f) continue;
        sum += (cosI*cosJ)/(PI*distSqr);
    }
    kij[pair] = sum*INV_NUM_RAYS;
}

static void formCudaCheck(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", action, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void launchFormFactorKernel(const SimulationState& state, const float* d_random,
                            const size_t* d_offsets, float* d_kij, int* d_tau,
                            size_t start, size_t count) {
    static Triangle* d_triangles = nullptr;
    static GpuOctreeNode* d_nodes = nullptr;
    static unsigned int* d_indices = nullptr;
    static unsigned int nodeCount = 0;
    if (!d_triangles) {
        std::vector<GpuOctreeNode> nodes;
        std::vector<unsigned int> indices;
        flattenOctree(state.octree, nodes, indices);
        nodeCount = static_cast<unsigned int>(nodes.size());
        formCudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_triangles), state.triangles.size()*sizeof(Triangle)), "triangle allocation");
        formCudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_nodes), nodes.size()*sizeof(GpuOctreeNode)), "octree allocation");
        formCudaCheck(cudaMalloc(reinterpret_cast<void**>(&d_indices), indices.size()*sizeof(unsigned int)), "octree index allocation");
        formCudaCheck(cudaMemcpy(d_triangles, state.triangles.data(), state.triangles.size()*sizeof(Triangle), cudaMemcpyHostToDevice), "triangle upload");
        formCudaCheck(cudaMemcpy(d_nodes, nodes.data(), nodes.size()*sizeof(GpuOctreeNode), cudaMemcpyHostToDevice), "octree upload");
        formCudaCheck(cudaMemcpy(d_indices, indices.data(), indices.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "octree index upload");
    }
    constexpr unsigned int threads = 128;
    gpuFormFactors<<<static_cast<unsigned int>((count+threads-1)/threads), threads>>>(
        d_triangles, d_nodes, d_indices, nodeCount,
        state.numTriangles, d_random, d_offsets, d_kij, d_tau, start, count);
    formCudaCheck(cudaGetLastError(), "form factor kernel launch");
    formCudaCheck(cudaDeviceSynchronize(), "form factor kernel");
    if (start + count == state.numTriangles * state.numTriangles) {
        formCudaCheck(cudaFree(d_indices), "octree index release");
        formCudaCheck(cudaFree(d_nodes), "octree release");
        formCudaCheck(cudaFree(d_triangles), "triangle release");
        d_indices = nullptr; d_nodes = nullptr; d_triangles = nullptr;
    }
}
