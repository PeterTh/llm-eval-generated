/**
 * Room Response Simulation Benchmark
 * Hybrid MPI + OpenMP + CUDA implementation
 *
 * Parallelization strategy:
 * - MPI: Distributes triangles across processes for all phases
 * - OpenMP: Thread-level parallelism for form factors and tau on CPU
 * - CUDA: GPU acceleration for wave propagation simulation and distance computation
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

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
// Ray-Triangle Intersection
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

// Compute form factor using pre-generated random values for deterministic parallelism
val_t computeKijPrecomputed(size_t idxI, size_t idxJ,
                             const std::vector<Triangle>& triangles,
                             const Octree& octree,
                             const val_t* randomValues,
                             size_t rngOffset) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;
    size_t off = rngOffset;

    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t u1 = randomValues[off++];
        val_t v1 = randomValues[off++];
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        Vec3 abI = triI.b - triI.a;
        Vec3 acI = triI.c - triI.a;
        Vec3 pI = triI.a + abI * u1 + acI * v1;

        val_t u2 = randomValues[off++];
        val_t v2 = randomValues[off++];
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        Vec3 abJ = triJ.b - triJ.a;
        Vec3 acJ = triJ.c - triJ.a;
        Vec3 pJ = triJ.a + abJ * u2 + acJ * v2;

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 dv = pJ - pI;
        val_t distSqr = dv.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cphiI = cosPhi(dv, triI.normal());
        val_t cphiJ = cosPhi(-dv, triJ.normal());

        if (cphiI <= ZERO || cphiJ <= ZERO) continue;

        kij += (cphiI * cphiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void simulationStepKernel(
    const float* __restrict__ local_kij,
    const int* __restrict__ local_tau,
    const float* __restrict__ areas,
    const float* __restrict__ rho,
    const float* __restrict__ radE,
    float* __restrict__ radB,
    int N, int t, int start_i, int count_i)
{
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= count_i) return;
    int i = start_i + local_i;

    float sumB = 0.0f;
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        int local_idx = local_i * N + j;
        int tauij = local_tau[local_idx];
        if (t < tauij) continue;
        float kij_val = local_kij[local_idx];
        if (kij_val <= 0.0f) continue;
        int srcTime = t - tauij;
        float radJ = radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;
        sumB += fminf(kij_val * areas[j], 1.0f) * radJ;
    }
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

__global__ void distanceKernel(
    const float* __restrict__ radB,
    float* __restrict__ distances,
    int N, int T, int sourceIdx, int start_i, int count_i)
{
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= count_i) return;
    int i = start_i + local_i;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < T; t++) {
        float sum = 0.0f;
        for (int tt = t; tt < T; tt++) {
            float pB = radB[tt * N + i];
            float pS = radB[(tt - t) * N + sourceIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[i] = WAVE_SPEED * (float)bestT;
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

    int mpiRank, mpiSize;
    size_t startRow, endRow;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization (MPI-aware)
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    MPI_Comm_rank(MPI_COMM_WORLD, &state.mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.mpiSize);
    int rank = state.mpiRank;
    int size = state.mpiSize;

    // Rank 0 generates the mesh
    size_t numTriangles = 0;
    std::vector<val_t> triData;

    if (rank == 0) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        numTriangles = mesh.triangles.size();
        // Serialize triangles: each triangle has 9 floats for vertices + 3 for normal = 12 floats
        triData.resize(numTriangles * 12);
        for (size_t i = 0; i < numTriangles; i++) {
            triData[i*12 + 0] = mesh.triangles[i].a.x;
            triData[i*12 + 1] = mesh.triangles[i].a.y;
            triData[i*12 + 2] = mesh.triangles[i].a.z;
            triData[i*12 + 3] = mesh.triangles[i].b.x;
            triData[i*12 + 4] = mesh.triangles[i].b.y;
            triData[i*12 + 5] = mesh.triangles[i].b.z;
            triData[i*12 + 6] = mesh.triangles[i].c.x;
            triData[i*12 + 7] = mesh.triangles[i].c.y;
            triData[i*12 + 8] = mesh.triangles[i].c.z;
            triData[i*12 + 9] = mesh.triangles[i]._normal.x;
            triData[i*12 + 10] = mesh.triangles[i]._normal.y;
            triData[i*12 + 11] = mesh.triangles[i]._normal.z;
        }
    }

    // Broadcast triangle count
    MPI_Bcast(&numTriangles, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Broadcast triangle data
    if (rank != 0) triData.resize(numTriangles * 12);
    MPI_Bcast(triData.data(), (int)(numTriangles * 12 * sizeof(val_t)), MPI_BYTE, 0, MPI_COMM_WORLD);

    // All ranks reconstruct triangles
    state.numTriangles = numTriangles;
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % numTriangles;
    state.triangles.resize(numTriangles);
    for (size_t i = 0; i < numTriangles; i++) {
        state.triangles[i].a = Vec3(triData[i*12+0], triData[i*12+1], triData[i*12+2]);
        state.triangles[i].b = Vec3(triData[i*12+3], triData[i*12+4], triData[i*12+5]);
        state.triangles[i].c = Vec3(triData[i*12+6], triData[i*12+7], triData[i*12+8]);
        state.triangles[i]._normal = Vec3(triData[i*12+9], triData[i*12+10], triData[i*12+11]);
    }

    if (rank == 0) printf("Generated icosphere mesh with %zu triangles\n", numTriangles);

    // Each rank builds its own octree (same data, same result)
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Compute row distribution
    size_t rowsPerRank = numTriangles / size;
    size_t remainder = numTriangles % size;
    state.startRow = (size_t)rank * rowsPerRank + std::min((size_t)rank, remainder);
    state.endRow = state.startRow + rowsPerRank + ((size_t)rank < remainder ? 1 : 0);

    // Initialize arrays
    state.areas.resize(numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(numTriangles, reflectivity);
    state.kij.resize(numTriangles * numTriangles, ZERO);
    state.tau.resize(numTriangles * numTriangles, 0);
    state.radE.resize(timesteps * numTriangles, ZERO);
    state.radB.resize(timesteps * numTriangles, ZERO);
    state.distances.resize(numTriangles, ZERO);

    // Set source emission
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precompute Random Numbers (deterministic, same on all ranks)
// ============================================================================

void pregenerateRandomNumbers(const SimulationState& state,
                               std::vector<val_t>& allRandomValues,
                               std::vector<size_t>& rngOffset,
                               std::vector<bool>& isCulled) {
    size_t N = state.numTriangles;
    rngOffset.resize(N * N, 0);
    isCulled.resize(N * N, true);
    size_t totalRandomValues = 0;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j) continue;
            if (state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
            isCulled[i * N + j] = false;
            rngOffset[i * N + j] = totalRandomValues;
            totalRandomValues += NUM_RAYS * 4;
        }
    }

    allRandomValues.resize(totalRandomValues);
    RandomGenerator rng(42);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (isCulled[i * N + j]) continue;
            size_t off = rngOffset[i * N + j];
            for (int r = 0; r < NUM_RAYS * 4; r++) {
                allRandomValues[off + r] = rng.rand();
            }
        }
    }
}

// ============================================================================
// Form Factor Computation (MPI + OpenMP)
// ============================================================================

void computeFormFactors(SimulationState& state) {
    int rank = state.mpiRank;
    size_t N = state.numTriangles;

    if (rank == 0) printf("Computing form factors (Kij)...\n");

    // All ranks pre-generate the same random numbers (deterministic)
    std::vector<val_t> allRandomValues;
    std::vector<size_t> rngOffset;
    std::vector<bool> isCulled;
    pregenerateRandomNumbers(state, allRandomValues, rngOffset, isCulled);

    // Each rank computes form factors for its assigned rows using OpenMP
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = state.startRow; i < state.endRow; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j || isCulled[i * N + j]) continue;
            state.kij[state.idx2d(i, j)] = computeKijPrecomputed(
                i, j, state.triangles, state.octree,
                allRandomValues.data(), rngOffset[i * N + j]);
        }
    }

    if (rank == 0) printf("  Form factors computed.\n");
}

// ============================================================================
// Time Delay Computation (OpenMP)
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    int rank = state.mpiRank;
    size_t N = state.numTriangles;

    if (rank == 0) printf("Computing time delays (Tau)...\n");

    #pragma omp parallel for schedule(static)
    for (size_t i = state.startRow; i < state.endRow; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (MPI + OpenMP + CUDA)
// ============================================================================

void runSimulation(SimulationState& state) {
    int rank = state.mpiRank;
    int size = state.mpiSize;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    if (rank == 0) printf("Running wave propagation simulation...\n");

    // Select GPU based on rank
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    size_t localCount = state.endRow - state.startRow;

    // Allocate device memory for local kij and tau
    float *d_kij = nullptr, *d_tau_int = nullptr;
    float *d_areas = nullptr, *d_rho = nullptr, *d_radE = nullptr, *d_radB = nullptr;
    int *d_tau = nullptr;

    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_kij, localCount * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tau, localCount * N * sizeof(int)));
    }
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));

    // Copy data to device
    if (localCount > 0) {
        // Copy local rows of kij (stored as float)
        std::vector<float> localKij(localCount * N);
        for (size_t i = 0; i < localCount; i++) {
            size_t globalI = state.startRow + i;
            for (size_t j = 0; j < N; j++) {
                localKij[i * N + j] = state.kij[globalI * N + j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_kij, localKij.data(), localCount * N * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Copy local rows of tau
        std::vector<int> localTau(localCount * N);
        for (size_t i = 0; i < localCount; i++) {
            size_t globalI = state.startRow + i;
            for (size_t j = 0; j < N; j++) {
                localTau[i * N + j] = state.tau[globalI * N + j];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_tau, localTau.data(), localCount * N * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_radB, 0, T * N * sizeof(float)));

    // Compute MPI distribution info for Allgatherv
    std::vector<int> recvCounts(size);
    std::vector<int> recvDispls(size);
    size_t rowsPerRank = N / size;
    size_t remainder = N % size;
    for (int r = 0; r < size; r++) {
        size_t r_start = (size_t)r * rowsPerRank + std::min((size_t)r, remainder);
        size_t r_count = rowsPerRank + ((size_t)r < remainder ? 1 : 0);
        recvCounts[r] = (int)r_count;
        recvDispls[r] = (int)r_start;
    }

    // Host buffers for MPI communication
    std::vector<val_t> localRadB(localCount);
    std::vector<val_t> globalRadB(N);

    int blockSize = 256;
    int numBlocks = ((int)localCount + blockSize - 1) / blockSize;
    if (numBlocks == 0) numBlocks = 1;

    for (size_t t = 0; t < T; t++) {
        // Launch CUDA kernel for local rows
        if (localCount > 0) {
            simulationStepKernel<<<numBlocks, blockSize>>>(
                d_kij, d_tau, d_areas, d_rho, d_radE, d_radB,
                (int)N, (int)t, (int)state.startRow, (int)localCount);
            CUDA_CHECK(cudaGetLastError());

            // Copy local result from device
            CUDA_CHECK(cudaMemcpy(localRadB.data(),
                                  d_radB + t * N + state.startRow,
                                  localCount * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        // MPI_Allgatherv to collect complete row
        MPI_Allgatherv(localRadB.data(), (int)localCount, MPI_FLOAT,
                        globalRadB.data(), recvCounts.data(), recvDispls.data(),
                        MPI_FLOAT, MPI_COMM_WORLD);

        // Copy complete row back to device
        CUDA_CHECK(cudaMemcpy(d_radB + t * N, globalRadB.data(),
                              N * sizeof(float), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            if (rank == 0) printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Copy full radB back to host
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, T * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    if (d_kij) CUDA_CHECK(cudaFree(d_kij));
    if (d_tau) CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
}

// ============================================================================
// Distance Computation (MPI + CUDA)
// ============================================================================

void computeDistances(SimulationState& state) {
    int rank = state.mpiRank;
    int size = state.mpiSize;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    if (rank == 0) printf("Computing distances via cross-correlation...\n");

    // Select GPU
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    size_t localCount = state.endRow - state.startRow;

    // Allocate device memory
    float *d_radB = nullptr, *d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_radB, T * N * sizeof(float)));
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_distances, localCount * sizeof(float)));
    }

    // Copy radB to device
    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), T * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Launch distance kernel
    int blockSize = 256;
    int numBlocks = ((int)localCount + blockSize - 1) / blockSize;
    if (numBlocks == 0) numBlocks = 1;

    if (localCount > 0) {
        distanceKernel<<<numBlocks, blockSize>>>(
            d_radB, d_distances,
            (int)N, (int)T, (int)state.sourceIndex,
            (int)state.startRow, (int)localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local distances back
        std::vector<val_t> localDist(localCount);
        CUDA_CHECK(cudaMemcpy(localDist.data(), d_distances, localCount * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < localCount; i++) {
            state.distances[state.startRow + i] = localDist[i];
        }
    }

    // Gather all distances to rank 0
    MPI_Gather(state.distances.data() + state.startRow, (int)localCount, MPI_FLOAT,
               state.distances.data(), (int)localCount, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // For non-root ranks, we don't need the full distances, but let's broadcast
    // so validation can work on all ranks if needed
    // Actually, only rank 0 needs the full distances for output/validation
    // But let's broadcast for consistency
    MPI_Bcast(state.distances.data(), (int)N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_radB));
    if (d_distances) CUDA_CHECK(cudaFree(d_distances));
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
// Hash
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printRes = false;

    // All ranks parse the same arguments (launched with same command line)
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
            printRes = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI processes: %d\n", size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA devices: %d\n", deviceCount);
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
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    if (rank == 0) {
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

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    if (printRes) {
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
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
