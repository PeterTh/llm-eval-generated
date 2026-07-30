/**
 * Room Response Simulation Benchmark - Hybrid MPI+OpenMP+CUDA
 * 
 * Parallelized using:
 * - MPI: Distributes triangle rows across cluster nodes
 * - OpenMP: Thread-level parallelism within each node
 * - CUDA: GPU acceleration for simulation, distances, and time delays
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

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Kernels (compiled separately via nvcc, included as host-callable)
// ============================================================================

// We declare CUDA kernels inline using __host__ __device__ where needed
// For the actual CUDA code, we use a separate compilation approach

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
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
// Triangle-Box Overlap Test
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

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
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

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

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];
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

    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

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

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
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

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

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

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation (deterministic per-pair seeding)
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Deterministic per-pair RNG: seeds based on (i, j, ray_index, global_seed)
inline uint32_t pairSeed(size_t i, size_t j, int ray, uint32_t globalSeed) {
    // Use a hash combining all inputs
    uint32_t h = globalSeed;
    h ^= static_cast<uint32_t>(i) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<uint32_t>(j) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= static_cast<uint32_t>(ray) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

inline Vec3 randomPointInTriangleSeeded(const Triangle& t, size_t i, size_t j, int ray, uint32_t globalSeed) {
    RandomGenerator rng1(pairSeed(i, j, ray * 2, globalSeed));
    RandomGenerator rng2(pairSeed(i, j, ray * 2 + 1, globalSeed));
    val_t u = rng1.rand();
    val_t v = rng2.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore)
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
// Visibility Testing
// ============================================================================

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
            return true;
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation
// ============================================================================

val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor using octree-accelerated visibility (CPU path)
val_t computeKijOctree(size_t idxI, size_t idxJ,
                        const std::vector<Triangle>& triangles,
                        const Octree& octree,
                        uint32_t globalSeed) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleSeeded(triI, idxI, idxJ, r * 2, globalSeed);
        Vec3 pJ = randomPointInTriangleSeeded(triJ, idxI, idxJ, r * 2 + 1, globalSeed);

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
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex;
    uint32_t rngSeed;

    // MPI distribution info
    int mpiRank;
    int mpiSize;
    size_t localRowStart;  // First row this rank owns
    size_t localRowCount;  // Number of rows this rank owns

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                           size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.rngSeed = 42;

    // MPI distribution: split rows across ranks
    size_t rowsPerRank = state.numTriangles / state.mpiSize;
    size_t remainder = state.numTriangles % state.mpiSize;
    state.localRowStart = state.mpiRank * rowsPerRank + std::min((size_t)state.mpiRank, remainder);
    state.localRowCount = rowsPerRank + ((size_t)state.mpiRank < remainder ? 1 : 0);

    if (state.mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("MPI: %d ranks, each handling ~%zu rows\n", state.mpiSize, rowsPerRank);
    }

    // Build octree on all ranks (needed for form factor computation)
    if (state.mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices - only allocate local rows for kij and tau
    state.kij.resize(state.localRowCount * state.numTriangles, ZERO);
    state.tau.resize(state.localRowCount * state.numTriangles, 0);
    
    // radE and radB need full T×N on all ranks (for simulation)
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation: Form Factors (MPI + OpenMP)
// ============================================================================

void computeFormFactors(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing form factors (Kij) with MPI+OpenMP...\n");

    const size_t N = state.numTriangles;
    const size_t rowStart = state.localRowStart;
    const size_t rowCount = state.localRowCount;
    const uint32_t seed = state.rngSeed;

    #pragma omp parallel for schedule(dynamic, 4) collapse(1)
    for (size_t localI = 0; localI < rowCount; ++localI) {
        size_t i = rowStart + localI;
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            state.kij[localI * N + j] = computeKijOctree(
                i, j, state.triangles, state.octree, seed);
        }
        if ((localI + 1) % 100 == 0 || localI + 1 == rowCount) {
            #pragma omp critical
            printf("  Rank %d Progress: %zu/%zu local rows\n", state.mpiRank, localI + 1, rowCount);
        }
    }
}

// ============================================================================
// Precomputation: Time Delays (CUDA)
// ============================================================================

// CUDA kernel for computing time delays
__global__ void computeTauKernel(
    const float* triCentersX, const float* triCentersY, const float* triCentersZ,
    int* tauLocal, size_t N, size_t rowStart, size_t rowCount,
    float invWaveSpeed)
{
    size_t localI = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (localI >= rowCount || j >= N) return;
    
    size_t i = rowStart + localI;
    if (i == j) return;
    
    float dx = triCentersX[i] - triCentersX[j];
    float dy = triCentersY[i] - triCentersY[j];
    float dz = triCentersZ[i] - triCentersZ[j];
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    
    tauLocal[localI * N + j] = static_cast<int>(ceilf(dist * invWaveSpeed));
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau) with CUDA...\n");

    const size_t N = state.numTriangles;
    const size_t rowCount = state.localRowCount;
    const size_t rowStart = state.localRowStart;

    // Precompute triangle centers
    std::vector<float> centerX(N), centerY(N), centerZ(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        Vec3 c = state.triangles[i].center();
        centerX[i] = c.x;
        centerY[i] = c.y;
        centerZ[i] = c.z;
    }

    // Allocate GPU memory
    float *d_cx, *d_cy, *d_cz;
    int *d_tau;
    
    CUDA_CHECK(cudaMalloc(&d_cx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, rowCount * N * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_cx, centerX.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy, centerY.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cz, centerZ.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((rowCount + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    computeTauKernel<<<grid, block>>>(
        d_cx, d_cy, d_cz, d_tau, N, rowStart, rowCount, INV_WAVE_SPEED);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau, rowCount * N * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_cx));
    CUDA_CHECK(cudaFree(d_cy));
    CUDA_CHECK(cudaFree(d_cz));
    CUDA_CHECK(cudaFree(d_tau));
}

// ============================================================================
// Simulation: Wave Propagation (CUDA)
// ============================================================================

// CUDA kernel for one timestep of the simulation
__global__ void simulationTimestepKernel(
    const float* kij, const int* tau, const float* areas, const float* rho,
    const float* radE, const float* radB, float* radBNew,
    size_t N, size_t t, size_t rowStart, size_t rowCount)
{
    size_t localI = blockIdx.x * blockDim.x + threadIdx.x;
    if (localI >= rowCount) return;
    
    size_t i = rowStart + localI;
    float sumB = 0.0f;
    
    for (size_t j = 0; j < N; ++j) {
        if (i == j) continue;
        
        int tauij = tau[localI * N + j];
        if (static_cast<int>(t) < tauij) continue;
        
        float k = kij[localI * N + j];
        if (k <= 0.0f) continue;
        
        size_t srcTime = t - static_cast<size_t>(tauij);
        float radJ = radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;
        
        float weight = fminf(k * areas[j], 1.0f);
        sumB += weight * radJ;
    }
    
    radBNew[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

void runSimulation(SimulationState& state) {
    if (state.mpiRank == 0) printf("Running wave propagation simulation with CUDA+MPI...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t rowCount = state.localRowCount;
    const size_t rowStart = state.localRowStart;

    // Allocate GPU memory for kij, tau (local rows), and full radB, radE
    float *d_kij, *d_areas, *d_rho, *d_radE, *d_radB;
    int *d_tau;
    
    CUDA_CHECK(cudaMalloc(&d_kij, rowCount * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, rowCount * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), rowCount * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), rowCount * N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_radB, 0, T * N * sizeof(float)));

    // Each rank simulates its own rows
    // But radB needs to be shared across all ranks (each row reads from all columns)
    // Strategy: each timestep, all-to-all exchange of radB column data
    
    // For simplicity in the benchmark, we use a different approach:
    // Each rank computes its local rows, then we do MPI_Allgather to share radB
    
    // Temporary buffer for gathering full radB
    std::vector<float> fullRadB(T * N, 0.0f);
    
    dim3 block(256);
    dim3 grid((rowCount + block.x - 1) / block.x);

    for (size_t t = 0; t < T; ++t) {
        // Copy current radB state to GPU (only need up to timestep t)
        // Since radB is built incrementally, we maintain it on GPU
        // Actually, the kernel reads from d_radB which we update each step
        
        simulationTimestepKernel<<<grid, block>>>(
            d_kij, d_tau, d_areas, d_rho, d_radE, d_radB, d_radB,
            N, t, rowStart, rowCount);
        CUDA_CHECK(cudaGetLastError());
        
        // After each timestep, we need to synchronize radB across all ranks
        // Copy local rows from GPU
        std::vector<float> localRadBRow(rowCount);
        CUDA_CHECK(cudaMemcpy(localRadBRow.data(), d_radB + t * N + rowStart, 
                               rowCount * sizeof(float), cudaMemcpyDeviceToHost));
        
        // MPI_Allgather to share all rows
        // Each rank sends its local rows, all ranks receive full row
        std::vector<int> recvCounts(state.mpiSize);
        std::vector<int> displs(state.mpiSize);
        
        // We need to gather row data for timestep t
        // Use MPI_Allgatherv since rows may not be evenly divided
        std::vector<int> sendCounts(state.mpiSize);
        std::vector<int> sdispls(state.mpiSize);
        
        int localCount = static_cast<int>(rowCount);
        MPI_Allgather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        displs[0] = 0;
        for (int r = 1; r < state.mpiSize; ++r) {
            displs[r] = displs[r-1] + recvCounts[r-1];
        }
        
        std::vector<float> gatheredRadB(N);
        MPI_Allgatherv(localRadBRow.data(), localCount, MPI_FLOAT,
                       gatheredRadB.data(), recvCounts.data(), displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        
        // Copy gathered radB back to GPU for all ranks
        CUDA_CHECK(cudaMemcpy(d_radB + t * N, gatheredRadB.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        
        if ((t + 1) % 10 == 0 || t + 1 == T) {
            if (state.mpiRank == 0) printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Copy final radB back to host
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, T * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
}

// ============================================================================
// Distance Computation (CUDA)
// ============================================================================

// CUDA kernel for cross-correlation distance computation
__global__ void computeDistancesKernel(
    const float* radB, float* distances,
    size_t N, size_t T, size_t sourceIndex, float waveSpeed)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float maxCorr = 0.0f;
    int bestT = 0;
    
    for (size_t t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (size_t tt = t; tt < T; ++tt) {
            float pB = radB[tt * N + i];
            float pS = radB[(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    
    distances[i] = waveSpeed * static_cast<float>(bestT);
}

void computeDistances(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing distances via cross-correlation with CUDA...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Ensure all ranks have the full radB (should already be the case after simulation)
    // radB is already synchronized across all ranks

    float *d_radB, *d_distances;
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    computeDistancesKernel<<<grid, block>>>(d_radB, d_distances, N, T, state.sourceIndex, WAVE_SPEED);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_distances));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

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

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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

    int nonZeroKij = 0;
    for (size_t i = 0; i < state.kij.size(); ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors (local): %d/%zu\n", nonZeroKij, state.kij.size());

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
// Helper
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
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
    printf("               Actual count: 20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

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
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("============================================================\n");
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;
    
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time (max across all ranks)
    long localTotal = preDuration + simDuration + distDuration;
    long totalTime;
    MPI_Reduce(&localTotal, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Total computation time: %ld ms\n", totalTime);

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

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t hashVal = computeHash(state);
        printf("  Result hash: %016lX\n", hashVal);
        printf("\n");
    }

    if (printResults) {
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        if (mpiRank == 0) {
            print_results(distData, "Distances");
        }
    }

    if (validate) {
        if (!validateResults(state)) {
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
