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
#include <cstdarg>
#include <cfloat>
#include <climits>
#include <type_traits>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// MPI calls are confined to the main thread. Each node maps its local ranks to
// the visible GPUs, allowing CUDA_VISIBLE_DEVICES to control scheduler placement.
int worldRank = 0, worldSize = 1;
void mpiCheck(int result) {
    if (result != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING]; int length = 0;
        MPI_Error_string(result, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, result);
    }
}
void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA error: %s\n", worldRank, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                    worldRank, MPI_INFO_NULL, &local));
        int localRank, localSize, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        mpiCheck(MPI_Comm_size(local, &localSize));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs()/localSize)));
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "A CUDA GPU is required.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        cudaCheck(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&local));
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};
int rankPrintf(const char* format, ...) {
    if (worldRank) return 0;
    va_list args; va_start(args, format);
    int result = vprintf(format, args); va_end(args); return result;
}
#define printf rankPrintf
#define HD __host__ __device__

// Zero-length partitions are valid when ranks outnumber triangles.
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

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

        #pragma omp parallel
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
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
                Octree* child = children[i].get();
                #pragma omp task firstprivate(child, childMin, childMax, i) shared(childIndices) if(indices.size() > 128)
                child->buildNode(childIndices[i], childMin, childMax);
            }
        }
        // Child index vectors remain alive until every recursive task finishes.
        #pragma omp taskwait
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

// MT19937 uses 32-bit words. Keep that representation even on LP64 hosts,
// and vectorize its three dependency-free twist ranges. This is the standard
// engine's exact sequence, including seeding and discard, rather than a new RNG.
class Mt19937 {
    uint32_t words[624];
    size_t position = 624;
    void twist() {
        #pragma omp simd
        for (int k = 0; k < 227; ++k) {
            uint32_t y = (words[k]&0x80000000U) | (words[k+1]&0x7fffffffU);
            words[k] = words[k+397] ^ (y>>1) ^ ((0U-(y&1U))&0x9908b0dfU);
        }
        #pragma omp simd
        for (int k = 227; k < 454; ++k) {
            uint32_t y = (words[k]&0x80000000U) | (words[k+1]&0x7fffffffU);
            words[k] = words[k-227] ^ (y>>1) ^ ((0U-(y&1U))&0x9908b0dfU);
        }
        #pragma omp simd
        for (int k = 454; k < 623; ++k) {
            uint32_t y = (words[k]&0x80000000U) | (words[k+1]&0x7fffffffU);
            words[k] = words[k-227] ^ (y>>1) ^ ((0U-(y&1U))&0x9908b0dfU);
        }
        uint32_t y = (words[623]&0x80000000U) | (words[0]&0x7fffffffU);
        words[623] = words[396] ^ (y>>1) ^ ((0U-(y&1U))&0x9908b0dfU);
        position = 0;
    }
public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }
    explicit Mt19937(uint32_t seed = 42) {
        words[0] = seed;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253U*(words[i-1]^(words[i-1]>>30))+i;
    }
    result_type operator()() {
        if (position == 624) twist();
        uint32_t y = words[position++];
        y ^= y>>11; y ^= (y<<7)&0x9d2c5680U;
        y ^= (y<<15)&0xefc60000U; y ^= y>>18;
        return y;
    }
    void discard(uint64_t count) {
        while (count > 624-position) {
            count -= 624-position;
            twist();
        }
        position += static_cast<size_t>(count);
    }
};
class RandomGenerator {
    Mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(uint64_t count) { rng.discard(count); }
};

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
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t firstRow = 0, localRows = 0;
    std::vector<int> counts, offsets;
    DeviceBuffer<Triangle> dTriangles;
    DeviceBuffer<val_t> dKij, dAreas, dRad, dOutput, dDistances;
    DeviceBuffer<int> dTau;
    int minDelay = 1;
    uint64_t nonZeroKij = 0;
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Receiver-row ownership keeps the quadratic matrices distributed.
    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t first = state.numTriangles * rank / worldSize;
        size_t last = state.numTriangles * (rank + 1) / worldSize;
        state.counts[rank] = static_cast<int>(last - first);
        state.offsets[rank] = static_cast<int>(first);
    }
    state.firstRow = state.offsets[worldRank];
    state.localRows = state.counts[worldRank];
    const size_t pairs = state.localRows * state.numTriangles;
    state.dTriangles.allocate(state.numTriangles);
    state.dAreas.allocate(state.numTriangles);
    state.dKij.allocate(pairs);
    state.dTau.allocate(pairs);
    state.dRad.allocate(timesteps * state.numTriangles);
    state.dDistances.allocate(state.localRows);
    cudaCheck(cudaMemcpy(state.dTriangles.data, state.triangles.data(),
                         state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(state.dAreas.data, state.areas.data(),
                         state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(state.dRad.data, 0, timesteps * state.numTriangles * sizeof(val_t)));
    state.distances.resize(state.numTriangles, ZERO);

}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Flattened preorder octree: escape skips an entire subtree, so GPU rays
// traverse without recursion, a fixed-depth stack, or per-ray allocations.
struct FlatNode {
    Vec3 center, halfExtent;
    int escape;
    size_t begin, count;
};
void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                   std::vector<size_t>& indices) {
    size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, indices.size(), tree.triangleIndices.size()});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) if (child) flattenOctree(*child, nodes, indices);
    nodes[here].escape = static_cast<int>(nodes.size());
}
__device__ bool intersectsBox(const Vec3& p1, const Vec3& p2, const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}
__device__ bool blocked(const Vec3& from, const Vec3& to, size_t src, size_t dst,
                        const Triangle* tris, const FlatNode* nodes, const size_t* indices) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    int node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& n = nodes[node];
        // Original traversal always visits the root, including root leaves.
        if (node && !intersectsBox(from, to, n)) { node = n.escape; continue; }
        for (size_t k = 0; k < n.count; ++k) {
            size_t idx = indices[n.begin + k];
            if (idx == src || idx == dst) continue;
            const Triangle& tri = tris[idx];
            val_t distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}
__device__ Vec3 sampledPoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}
__global__ void formKernel(const Triangle* tris, const FlatNode* nodes, const size_t* indices,
                           const val_t* samples, val_t* factors, size_t n, size_t first,
                           size_t rows, size_t batchStart, size_t batchRows) {
    size_t thread = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t pair = thread/NUM_RAYS;
    int ray = thread%NUM_RAYS;
    unsigned mask = __ballot_sync(0xffffffffU, pair < batchRows*n);
    if (pair >= batchRows*n) return;
    size_t row = batchStart + pair/n, j = pair%n, i = first+row;
    const Triangle& a = tris[i]; const Triangle& b = tris[j];
    val_t contribution = ZERO;
    if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
        const val_t* sample = samples + pair*(4*NUM_RAYS) + 4*ray;
        Vec3 from = sampledPoint(a, sample[0], sample[1]);
        Vec3 to = sampledPoint(b, sample[2], sample[3]);
        if (!blocked(from, to, i, j, tris, nodes, indices)) {
            Vec3 v = to-from;
            val_t squared = v.squaredNorm();
            if (squared >= EPSILON) {
                val_t ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
                if (ci > ZERO && cj > ZERO) contribution = (ci*cj)/(PI*squared);
            }
        }
    }
    // Each half warp traces one pair. Shuffles preserve the sequential r-order
    // rather than using a tree reduction that changes floating-point rounding.
    val_t sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r)
        sum += __shfl_sync(mask, contribution, r, NUM_RAYS);
    if (!ray) factors[row*n + j] = sum*INV_NUM_RAYS;
}
__global__ void delayKernel(const Triangle* tris, int* delays, int* minDelay,
                            size_t n, size_t first, size_t rows) {
    size_t p = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    int candidate = INT_MAX;
    if (p < rows*n) {
        size_t row = p/n, j = p%n, i = first+row;
        int delay = i == j ? 0 : computeTau(tris[i], tris[j]);
        delays[p] = delay;
        if (i != j) candidate = delay;
    }
    __shared__ int minimum[256];
    minimum[threadIdx.x] = candidate; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            minimum[threadIdx.x] = min(minimum[threadIdx.x], minimum[threadIdx.x+stride]);
        __syncthreads();
    }
    if (!threadIdx.x) atomicMin(minDelay, minimum[0]);
}
__global__ void countFactors(const val_t* factors, size_t count, unsigned long long* total) {
    size_t p = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    unsigned long long found = 0;
    for (; p < count; p += size_t(gridDim.x)*blockDim.x) found += factors[p] > EPSILON;
    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = found; __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(total, sums[0]);
}
void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on CUDA...\n");
    DeviceBuffer<int> minimum; minimum.allocate(1);
    int value = std::numeric_limits<int>::max();
    cudaCheck(cudaMemcpy(minimum.data, &value, sizeof(value), cudaMemcpyHostToDevice));
    size_t pairs = state.localRows * state.numTriangles;
    if (pairs) delayKernel<<<(pairs+255)/256,256>>>(state.dTriangles.data, state.dTau.data,
        minimum.data, state.numTriangles, state.firstRow, state.localRows);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&value, minimum.data, sizeof(value), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Allreduce(&value, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    // Distinct icosphere triangle centers have strictly positive delays.
    if (state.minDelay < 1) {
        fprintf(stderr, "Zero inter-triangle delay would require ordered updates.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (MPI/OpenMP/CUDA)...\n");
    const size_t n = state.numTriangles, rows = state.localRows;
    // Culling happens before sampling in the original loop. Record the exact
    // consumption per row before splitting the single seed-42 MT19937 stream.
    std::vector<uint64_t> draws(n);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        size_t i = state.firstRow+row;
        uint64_t count = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f)
                count += 4*NUM_RAYS;
        draws[i] = count;
    }
    mpiCheck(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_UINT64_T, draws.data(),
        state.counts.data(), state.offsets.data(), MPI_UINT64_T, MPI_COMM_WORLD));
    // Only rank zero walks the prefix stream; transmit engine checkpoints to
    // other ranks. All ranks execute the same binary (and hence engine ABI).
    static_assert(std::is_trivially_copyable<RandomGenerator>::value, "RNG checkpoints must be byte-copyable");
    RandomGenerator start(42);
    if (!worldRank) {
        size_t cursor = 0;
        for (int rank = 1; rank < worldSize; ++rank) {
            size_t boundary = state.offsets[rank];
            uint64_t skip = 0;
            while (cursor < boundary) skip += draws[cursor++];
            start.discard(skip);
            mpiCheck(MPI_Send(&start, sizeof(start), MPI_BYTE, rank, 42, MPI_COMM_WORLD));
        }
        start = RandomGenerator(42);
    } else mpiCheck(MPI_Recv(&start, sizeof(start), MPI_BYTE, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    std::vector<RandomGenerator> generators;
    generators.reserve(rows);
    for (size_t row = 0; row < rows; ++row) {
        generators.push_back(start);
        start.discard(draws[state.firstRow+row]);
    }
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    DeviceBuffer<FlatNode> dNodes; dNodes.allocate(nodes.size());
    DeviceBuffer<size_t> dIndices; dIndices.allocate(indices.size());
    cudaCheck(cudaMemcpy(dNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    // Bound staging memory independently of the quadratic problem size.
    size_t batchRows = std::min(rows, std::max(size_t(1), size_t(65536)/n));
    if (rows) {
        size_t capacity = batchRows*n*4*NUM_RAYS;
        val_t* samples;
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&samples), 2*capacity*sizeof(val_t)));
        DeviceBuffer<val_t> dSamples; dSamples.allocate(2*capacity);
        cudaStream_t streams[2];
        for (auto& stream : streams) cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        size_t batch = 0;
        for (size_t base = 0; base < rows; base += batchRows) {
            size_t count = std::min(batchRows, rows-base);
            int slot = batch++%2;
            cudaCheck(cudaStreamSynchronize(streams[slot]));
            val_t* hostBatch = samples+slot*capacity;
            val_t* deviceBatch = dSamples.data+slot*capacity;
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < count; ++r) {
                size_t i = state.firstRow+base+r;
                RandomGenerator& rng = generators[base+r];
                for (size_t j = 0; j < n; ++j) {
                    if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                    val_t* sample = hostBatch + (r*n+j)*4*NUM_RAYS;
                    for (int k = 0; k < 4*NUM_RAYS; ++k) sample[k] = rng.rand();
                }
            }
            cudaCheck(cudaMemcpyAsync(deviceBatch, hostBatch, count*n*4*NUM_RAYS*sizeof(val_t),
                                      cudaMemcpyHostToDevice, streams[slot]));
            formKernel<<<(count*n*NUM_RAYS+127)/128,128,0,streams[slot]>>>(state.dTriangles.data,
                dNodes.data, dIndices.data, deviceBatch, state.dKij.data, n, state.firstRow, rows, base, count);
            cudaCheck(cudaGetLastError());
        }
        for (auto stream : streams) {
            cudaCheck(cudaStreamSynchronize(stream));
            cudaCheck(cudaStreamDestroy(stream));
        }
        cudaCheck(cudaFreeHost(samples));
    }
    DeviceBuffer<unsigned long long> dCount; dCount.allocate(1);
    cudaCheck(cudaMemset(dCount.data, 0, sizeof(unsigned long long)));
    if (rows) countFactors<<<std::min(size_t(1024), (rows*n+255)/256),256>>>(state.dKij.data, rows*n, dCount.data);
    cudaCheck(cudaGetLastError());
    uint64_t localCount;
    cudaCheck(cudaMemcpy(&localCount, dCount.data, sizeof(localCount), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
}

__global__ void propagationKernel(const val_t* factors, const int* delays, const val_t* areas,
                                  val_t* rad, val_t* output, size_t n, size_t first, size_t rows,
                                  size_t t, size_t batchBase, size_t source, size_t timeOff, val_t rho) {
    size_t thread = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    size_t row = thread/32;
    int lane = thread%32;
    if (row >= rows) return;
    size_t i = first+row;
    val_t sum = ZERO;
    // A warp loads 32 emitters concurrently, then adds them in original j-order.
    // Only lane zero accumulates; every lane participates in the shuffles.
    for (size_t base = 0; base < n; base += 32) {
        size_t j = base+lane;
        val_t contribution = ZERO;
        if (j < n && i != j) {
            int delay = delays[row*n+j];
            if (t >= size_t(delay)) {
                val_t factor = factors[row*n+j];
                if (!(factor <= ZERO)) {
                    val_t incoming = rad[(t-size_t(delay))*n+j];
                    if (!(incoming <= ZERO)) contribution = fminf(factor*areas[j], ONE)*incoming;
                }
            }
        }
        for (int k = 0; k < 32 && base+size_t(k) < n; ++k) {
            val_t value = __shfl_sync(0xffffffffU, contribution, k);
            if (!lane) sum += value;
        }
    }
    if (!lane) {
        val_t value = rho*sum + ((i == source && t < timeOff) ? ONE : ZERO);
        rad[t*n+i] = value;
        output[(t-batchBase)*rows+row] = value;
    }
}
void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    size_t n = state.numTriangles, rows = state.localRows;
    // No contribution can cross a rank boundary within minDelay timesteps.
    // Exchange blocks of that many steps to amortize collective latency.
    size_t block = std::min(state.numTimesteps, size_t(state.minDelay));
    state.dOutput.allocate(block*rows);
    val_t *local, *gathered, *ordered;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&local), std::max(size_t(1),block*rows)*sizeof(val_t)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gathered), std::max(size_t(1),block*n)*sizeof(val_t)));
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&ordered), std::max(size_t(1),block*n)*sizeof(val_t)));
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t base = 0; base < state.numTimesteps; base += block) {
        size_t steps = std::min(block, state.numTimesteps-base);
        for (size_t t = base; t < base+steps; ++t) {
            if (rows) propagationKernel<<<(rows*32+127)/128,128>>>(state.dKij.data, state.dTau.data,
                state.dAreas.data, state.dRad.data, state.dOutput.data, n, state.firstRow, rows,
                t, base, state.sourceIndex, state.numTimesteps/2, state.rho[0]);
            cudaCheck(cudaGetLastError());
        }
        if (worldSize > 1) {
            cudaCheck(cudaMemcpy(local, state.dOutput.data, steps*rows*sizeof(val_t), cudaMemcpyDeviceToHost));
            for (int rank = 0; rank < worldSize; ++rank) {
                counts[rank] = static_cast<int>(steps*state.counts[rank]);
                offsets[rank] = static_cast<int>(steps*state.offsets[rank]);
            }
            mpiCheck(MPI_Allgatherv(local, static_cast<int>(steps*rows), MPI_FLOAT,
                gathered, counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD));
            #pragma omp parallel for schedule(static)
            for (int rank = 0; rank < worldSize; ++rank)
                for (size_t t = 0; t < steps; ++t)
                    std::copy_n(gathered+offsets[rank]+t*state.counts[rank], state.counts[rank],
                                ordered+t*n+state.offsets[rank]);
            cudaCheck(cudaMemcpy(state.dRad.data+base*n, ordered, steps*n*sizeof(val_t), cudaMemcpyHostToDevice));
        }
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFreeHost(local)); cudaCheck(cudaFreeHost(gathered)); cudaCheck(cudaFreeHost(ordered));
}
__global__ void correlationKernel(const val_t* rad, val_t* distances, size_t n, size_t times,
                                  size_t first, size_t rows, size_t source) {
    size_t row = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    val_t maxCorr = ZERO; size_t best = 0;
    for (size_t t = 0; t < times; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < times; ++tt)
            sum += rad[(tt-t)*n+source]*rad[tt*n+first+row];
        if (sum > maxCorr) { maxCorr = sum; best = t; }
    }
    distances[row] = WAVE_SPEED*static_cast<val_t>(best);
}
void computeDistances(SimulationState& state) {
    printf("Computing distances via CUDA cross-correlation...\n");
    if (state.localRows) correlationKernel<<<(state.localRows+127)/128,128>>>(state.dRad.data,
        state.dDistances.data, state.numTriangles, state.numTimesteps, state.firstRow,
        state.localRows, state.sourceIndex);
    cudaCheck(cudaGetLastError());
    std::vector<val_t> local(state.localRows);
    if (!local.empty())
        cudaCheck(cudaMemcpy(local.data(), state.dDistances.data, local.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
    mpiCheck(MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_FLOAT, state.distances.data(),
        state.counts.data(), state.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
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
    uint64_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
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

    if (timesteps < 0) { printf("Timesteps must be non-negative.\n"); return 1; }
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
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &preDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &simDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    mpiCheck(MPI_Allreduce(MPI_IN_PLACE, &distDuration, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    if (validate && !worldRank) {
        state.radB.resize(state.numTriangles * state.numTimesteps);
        if (!state.radB.empty())
            cudaCheck(cudaMemcpy(state.radB.data(), state.dRad.data,
                state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    // All collectives are complete; only the root writes benchmark results.
    int validationStatus = 0;
    if (!worldRank) {
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
        size_t memKij = state.localRows * n * sizeof(val_t);
        size_t memTau = state.localRows * n * sizeof(int);
        size_t memRad = t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Core device memory per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
                validationStatus = 1;
            }
        }
    }
    mpiCheck(MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return validationStatus;
}
