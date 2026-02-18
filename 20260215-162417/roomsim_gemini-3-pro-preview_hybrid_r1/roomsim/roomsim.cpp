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

#include <mpi.h>
#include <omp.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
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

// CUDA Error checking
#ifdef USE_CUDA
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA Kernels
__global__ void computeKijKernel(int numTriangles, int numRays, const Triangle* triangles, 
                                 const float* randomNums, float* kij_chunk, int startIdx, int endIdx) {
    // This is complex due to octree and random sampling. 
    // For now, let's keep computeKij on CPU with OpenMP for simplicity 
    // unless we can port the octree traversal to GPU easily.
    // Given constraints, CPU OpenMP for octree traversal is safer/easier to keep correct.
    // However, the prompt asks for CUDA as appropriate.
    // Ray tracing is great for GPU. But porting the pointer-based octree to GPU is hard in one go.
    // Let's focus CUDA on the dense matrix operations in runSimulation.
}

// CUDA Kernel for Simulation Step
__global__ void simulationStepKernel(int numTriangles, int t, int startIdx, int endIdx,
                                     const float* kij_chunk, const int* tau_chunk,
                                     const float* radB_all, const float* radE_chunk, const float* rho_chunk, const float* areas,
                                     float* radB_new_chunk) {
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = startIdx + local_i;

    if (i < endIdx) {
        float sumB = 0.0f;
        
        // Row in kij/tau chunk corresponds to local_i
        int rowOffset = local_i * numTriangles;

        for (int j = 0; j < numTriangles; ++j) {
            if (i == j) continue;

            int tauij = tau_chunk[rowOffset + j];

            // Skip if wave hasn't yet propagated from j to i
            if (t < tauij) continue;

            float kij = kij_chunk[rowOffset + j];
            if (kij <= 1e-6f) continue;

            // Get radiosity from source triangle at time when emission occurred
            int srcTime = t - tauij;
            // radB is stored as [Timesteps * NumTriangles] or just [NumTriangles] for current/prev?
            // The CPU code stores all history. GPU memory might be limited.
            // We need random access to history. 
            // Let's assume we copy the necessary history or the full history if it fits.
            // For N=20k, T=50, Matrix is 1M floats = 4MB. Fits easily.
            
            float radJ = radB_all[srcTime * numTriangles + j];
            if (radJ <= 1e-6f) continue;

            // Accumulate contribution
            // sumB += std::min(kij * state.areas[j], ONE) * radJ;
            float term = kij * areas[j];
            if (term > 1.0f) term = 1.0f;
            sumB += term * radJ;
        }

        // Update radiosity: reflection + emission
        // state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        // radE_chunk handles the [t, i] indexing inside it or we pass the specific row
        
        // Actually radE is [Timesteps x Triangles].
        float radE = radE_chunk[t * (endIdx - startIdx) + local_i]; 
        
        radB_new_chunk[local_i] = rho_chunk[local_i] * sumB + radE;
    }
}
#endif

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
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);

    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.idx2d(i, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
        if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        for (size_t i = 0; i < state.numTriangles; ++i) {
            val_t sumB = ZERO;

            for (size_t j = 0; j < state.numTriangles; ++j) {
                if (i == j) continue;

                int tauij = state.tau[state.idx2d(i, j)];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[state.idx2d(i, j)];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
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
        printf("MPI Size: %d\n", size);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    // Calculate local range
    size_t local_n = state.numTriangles / size;
    size_t remainder = state.numTriangles % size;
    size_t startIdx = rank * local_n + std::min(static_cast<size_t>(rank), remainder);
    if (rank < remainder) local_n++;
    size_t endIdx = startIdx + local_n;
    
    // Resize local storage for distributed matrices
    // Kij and Tau are N x N globally, but we only store rows [startIdx, endIdx)
    // Actually, InitializeSimulation allocates full matrices. We should probably clear them to save memory
    // or just use the relevant parts.
    // For simplicity with existing code structure, let's keep the vectors but only fill local parts.
    // Wait, Kij is N*N. If N=20000, N*N*4B = 1.6GB. Fits in RAM.
    // But for scalability, we should ideally distribute.
    // However, the `computeKij` function writes to `state.kij`.
    // Let's modify `computeKij` to only compute local rows.
    
    if (rank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    // computeTimeDelays(state); -> needs parallelization
    // computeFormFactors(state); -> needs parallelization

    // Parallelize computeTimeDelays
    if (rank == 0) printf("Computing time delays (Tau)...\n");
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = startIdx; i < endIdx; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
    
    // Gather Tau? Or just compute all on all nodes? 
    // Tau is cheap to compute (distance), but N^2 is large.
    // Distributed computation + Gather is better.
    // But wait, do we need full Tau on every node?
    // In runSimulation, for local `i`, we need `tau[i][j]` for all `j`.
    // So each node ONLY needs rows `i` that it owns.
    // It does NOT need the full Tau matrix.
    // So we don't need to gather Tau!
    
    // Parallelize computeFormFactors
    if (rank == 0) printf("Computing form factors (Kij)...\n");
    
    // Use OpenMP for local rows
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = startIdx; i < endIdx; ++i) {
        // Create thread-local RNG
        RandomGenerator rng(42 + i * 1234); // Deterministic seed per row
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[state.idx2d(i, j)] = computeKij(
                i, j, state.triangles, state.octree, rng);
        }
        if (rank == 0 && ((i + 1) % 100 == 0 || i + 1 == endIdx)) {
             // Progress reporting only from rank 0 (approximate)
             // printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
        }
    }

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    if (rank == 0) printf("Running wave propagation simulation...\n");

    #ifdef USE_CUDA
    bool useCUDA = false;
    // Check for CUDA devices
    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    useCUDA = (cudaStatus == cudaSuccess && deviceCount > 0);
    
    // Allocate device memory if CUDA is available
    float *d_kij = nullptr, *d_radB = nullptr, *d_radE = nullptr;
    float *d_rho = nullptr, *d_areas = nullptr, *d_radB_new = nullptr;
    int *d_tau = nullptr;
    
    if (useCUDA) {
        if (rank == 0) printf("Using CUDA acceleration with %d devices\n", deviceCount);
        int deviceId = rank % deviceCount;
        checkCuda(cudaSetDevice(deviceId));
        
        size_t numLocal = endIdx - startIdx;
        size_t totalTris = state.numTriangles;
        
        // Allocate memory
        // Kij: only local rows. Size: numLocal * totalTris
        checkCuda(cudaMalloc(&d_kij, numLocal * totalTris * sizeof(float)));
        // Tau: only local rows. Size: numLocal * totalTris
        checkCuda(cudaMalloc(&d_tau, numLocal * totalTris * sizeof(int)));
        
        // RadB: Need full history for all triangles?
        // Kernel accesses radB[srcTime, j]. srcTime < t. j can be any triangle.
        // So yes, we need full history for all triangles on GPU.
        // Size: numTimesteps * totalTris
        checkCuda(cudaMalloc(&d_radB, state.numTimesteps * totalTris * sizeof(float)));
        checkCuda(cudaMemset(d_radB, 0, state.numTimesteps * totalTris * sizeof(float)));
        
        // RadE: Need local rows for current time t? Or full?
        // Kernel: radE_chunk[t * numLocal + local_i]
        // Size: numTimesteps * numLocal
        checkCuda(cudaMalloc(&d_radE, state.numTimesteps * numLocal * sizeof(float)));
        
        // Rho: local rows
        checkCuda(cudaMalloc(&d_rho, numLocal * sizeof(float)));
        
        // Areas: all triangles (for j loop)
        checkCuda(cudaMalloc(&d_areas, totalTris * sizeof(float)));
        
        // Output buffer for one step: local rows
        checkCuda(cudaMalloc(&d_radB_new, numLocal * sizeof(float)));
        
        // Copy static data
        // Copy Kij local rows
        // Note: state.kij is flat N*N. Local rows start at startIdx * N.
        checkCuda(cudaMemcpy(d_kij, &state.kij[startIdx * totalTris], 
                             numLocal * totalTris * sizeof(float), cudaMemcpyHostToDevice));
                             
        // Copy Tau local rows
        checkCuda(cudaMemcpy(d_tau, &state.tau[startIdx * totalTris], 
                             numLocal * totalTris * sizeof(int), cudaMemcpyHostToDevice));
                             
        // Copy Rho local
        checkCuda(cudaMemcpy(d_rho, &state.rho[startIdx], 
                             numLocal * sizeof(float), cudaMemcpyHostToDevice));
                             
        // Copy Areas (all)
        checkCuda(cudaMemcpy(d_areas, state.areas.data(), 
                             totalTris * sizeof(float), cudaMemcpyHostToDevice));
                             
        // Copy RadE local parts
        // state.radE is T * N. We need sub-matrix [T, startIdx..endIdx].
        // This is strided in the original array (row-major T x N).
        // Actually, storage is idxTN(t, n) = t * numTriangles + n.
        // So for each t, we have a chunk. We can copy row by row or just use the full array pointer if we copy it all.
        // Copying all radE is easier if memory permits (T*N is small enough).
        // Let's copy the needed parts to a contiguous buffer to save space on GPU if needed, 
        // but since we copy radB full, might as well copy radE full?
        // No, radE is only read for local_i. 
        // Let's just create a temporary host buffer for the compact radE and upload it.
        std::vector<float> radE_compact(state.numTimesteps * numLocal);
        for(size_t t=0; t<state.numTimesteps; ++t) {
            for(size_t i=0; i<numLocal; ++i) {
                radE_compact[t * numLocal + i] = state.radE[state.idxTN(t, startIdx + i)];
            }
        }
        checkCuda(cudaMemcpy(d_radE, radE_compact.data(), 
                             state.numTimesteps * numLocal * sizeof(float), cudaMemcpyHostToDevice));
    }
    #endif

    std::vector<val_t> stepBuffer(state.numTriangles);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        
        bool executedOnGPU = false;
        #ifdef USE_CUDA
        if (useCUDA) {
             size_t numLocal = endIdx - startIdx;
             int threadsPerBlock = 256;
             int blocks = (numLocal + threadsPerBlock - 1) / threadsPerBlock;
             
             simulationStepKernel<<<blocks, threadsPerBlock>>>(
                 state.numTriangles, t, startIdx, endIdx,
                 d_kij, d_tau, d_radB, d_radE, d_rho, d_areas, d_radB_new
             );
             checkCuda(cudaGetLastError());
             
             // Copy result back to host
             // We put it directly into state.radB at the right place
             checkCuda(cudaMemcpy(&state.radB[state.idxTN(t, startIdx)], d_radB_new, 
                                  numLocal * sizeof(float), cudaMemcpyDeviceToHost));
             executedOnGPU = true;
        }
        #endif
        
        if (!executedOnGPU) {
            #pragma omp parallel for schedule(static)
            for (size_t i = startIdx; i < endIdx; ++i) {
                val_t sumB = ZERO;

                for (size_t j = 0; j < state.numTriangles; ++j) {
                    if (i == j) continue;

                    int tauij = state.tau[state.idx2d(i, j)];

                    // Skip if wave hasn't yet propagated from j to i
                    if (static_cast<int>(t) < tauij) continue;

                    val_t kij = state.kij[state.idx2d(i, j)];
                    if (kij <= ZERO) continue;

                    // Get radiosity from source triangle at time when emission occurred
                    size_t srcTime = t - static_cast<size_t>(tauij);
                    val_t radJ = state.radB[state.idxTN(srcTime, j)];
                    
                    if (radJ <= ZERO) continue;

                    // Accumulate contribution
                    sumB += std::min(kij * state.areas[j], ONE) * radJ;
                }

                state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
            }
        }
        
        // Gather the results for timestep t from all ranks
        // We need to gather state.radB[t * N + startIdx ... t * N + endIdx] 
        // into the global state.radB[t * N ... ]
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        for(int r=0; r<size; ++r) {
             size_t r_local_n = state.numTriangles / size;
             size_t r_remainder = state.numTriangles % size;
             if (r < r_remainder) r_local_n++;
             recvcounts[r] = static_cast<int>(r_local_n);
             
             size_t r_startIdx = r * (state.numTriangles / size) + std::min(static_cast<size_t>(r), r_remainder);
             displs[r] = static_cast<int>(r_startIdx);
        }
        
        val_t* currentStepPtr = &state.radB[state.idxTN(t, 0)];
        val_t* sendBuf = &state.radB[state.idxTN(t, startIdx)];
        
        MPI_Allgatherv(sendBuf, static_cast<int>(endIdx - startIdx), MPI_FLOAT, 
                       currentStepPtr, recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        // If using CUDA, update the device's full radB for this timestep so future steps can read it
        #ifdef USE_CUDA
        if (useCUDA) {
            // Copy the now-complete row t to device
            checkCuda(cudaMemcpy(&d_radB[t * state.numTriangles], currentStepPtr,
                                 state.numTriangles * sizeof(float), cudaMemcpyHostToDevice));
        }
        #endif

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    
    // Cleanup CUDA
    #ifdef USE_CUDA
    if (useCUDA) {
        checkCuda(cudaFree(d_kij));
        checkCuda(cudaFree(d_tau));
        checkCuda(cudaFree(d_radB));
        checkCuda(cudaFree(d_radE));
        checkCuda(cudaFree(d_rho));
        checkCuda(cudaFree(d_areas));
        checkCuda(cudaFree(d_radB_new));
    }
    #endif

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    // computeDistances(state);
    if (rank == 0) printf("Computing distances via cross-correlation...\n");

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = startIdx; i < endIdx; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
    
    // Gather distances (only needed for validation/output on rank 0)
    // But validation runs on all ranks? No, validation usually on rank 0.
    // Let's gather distances to all ranks to simplify validation code or just rank 0.
    
    std::vector<val_t> allDistances(state.numTriangles);
    
    // Recalculate counts/displs
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for(int r=0; r<size; ++r) {
            size_t r_local_n = state.numTriangles / size;
            size_t r_remainder = state.numTriangles % size;
            if (r < r_remainder) r_local_n++;
            recvcounts[r] = static_cast<int>(r_local_n);
            
            size_t r_startIdx = r * (state.numTriangles / size) + std::min(static_cast<size_t>(r), r_remainder);
            displs[r] = static_cast<int>(r_startIdx);
    }

    MPI_Allgatherv(&state.distances[startIdx], static_cast<int>(local_n), MPI_FLOAT,
                   &state.distances[0], recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
