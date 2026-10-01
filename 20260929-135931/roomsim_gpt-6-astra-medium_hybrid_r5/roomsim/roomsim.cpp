/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed MPI/OpenMP/CUDA implementation of room impulse response
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
#include <vector>
#include <cstdarg>
#include <climits>
#include <exception>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// MPI calls are confined to the main thread; each local rank owns one GPU.
int worldRank = 0, worldSize = 1;
void rankPrintf(const char* format, ...) {
    if (worldRank != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}
void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)
struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: roomsim requires a CUDA GPU\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        // Respect OMP_NUM_THREADS; otherwise share available CPUs among local ranks.
        if (!getenv("OMP_NUM_THREADS")) {
            int localSize;
            MPI_Comm_size(local, &localSize);
            omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
        }
        MPI_Comm_free(&local);
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void resize(size_t count) {
        if (data) CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};
template<class T> struct PinnedBuffer {
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
        #pragma omp single
        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax, int depth = 0) {
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
                #pragma omp task firstprivate(i, childMin, childMax, depth) shared(childIndices) if(depth < 2)
                children[i]->buildNode(childIndices[i], childMin, childMax, depth + 1);
            }
        }
        #pragma omp taskwait
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < faces.size(); ++f) {
            const auto& face = faces[f];
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
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
    size_t rowBegin, localRows;
    std::vector<int> counts, displacements;
    DeviceBuffer<val_t> kijDevice, historyDevice, areasDevice;
    DeviceBuffer<int> tauDevice;
    unsigned long long nonZeroKij = 0;
    int minDelay = 1;
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

    rankPrintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    rankPrintf("Building octree...\n");
    state.octree.build(state.triangles);

    state.counts.resize(worldSize);
    state.displacements.resize(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        size_t begin = state.numTriangles * r / worldSize;
        size_t end = state.numTriangles * (r + 1) / worldSize;
        state.counts[r] = static_cast<int>(end - begin);
        state.displacements[r] = static_cast<int>(begin);
    }
    state.rowBegin = state.displacements[worldRank];
    state.localRows = state.counts[worldRank];

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    const size_t entries = state.localRows * state.numTriangles;
    state.kijDevice.resize(entries);
    state.tauDevice.resize(entries);
    state.historyDevice.resize(timesteps * state.numTriangles);
    state.areasDevice.resize(state.numTriangles);
    CUDA_CHECK(cudaMemcpy(state.areasDevice.data, state.areas.data(), state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.historyDevice.data, 0, timesteps * state.numTriangles * sizeof(val_t)));
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// A stackless preorder octree preserves the original overlap/intersection tests.
// escape is the first node after this subtree, eliminating per-ray stacks.
struct FlatNode {
    Vec3 center, halfExtent;
    size_t first, count, escape;
};
void flattenTree(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<size_t>& indices) {
    size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(), tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (int c = 0; c < 8; ++c)
        if (tree.children[c]) flattenTree(*tree.children[c], nodes, indices);
    nodes[here].escape = nodes.size();
}
__device__ bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}
__device__ bool blocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                        const FlatNode* nodes, const size_t* indices, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    size_t node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& current = nodes[node];
        if (node && !intersectsBox(current, from, to)) { node = current.escape; continue; }
        for (size_t k = 0; k < current.count; ++k) {
            size_t index = indices[current.first + k];
            if (index == src || index == dst) continue;
            const Triangle& tri = triangles[index];
            val_t distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}
__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

// Explicit standard MT19937 state makes rank-to-rank checkpoints portable.
// Skipping consumes exactly the draws skipped by the original normal culling.
struct RandomStream {
    uint32_t words[625];
    RandomStream() {
        words[0] = 42;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253U * (words[i-1] ^ (words[i-1] >> 30)) + i;
        words[624] = 624;
    }
    void twist() {
        // Split at the recurrence dependencies to allow host SIMD execution.
        #pragma omp simd
        for (int i = 0; i < 227; ++i) {
            uint32_t x = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
            words[i] = words[i+397] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
        }
        for (int base = 227; base < 623; base += 227) {
            int end = std::min(base + 227, 623);
            #pragma omp simd
            for (int i = base; i < end; ++i) {
                uint32_t x = (words[i] & 0x80000000U) | (words[i+1] & 0x7fffffffU);
                words[i] = words[i-227] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
            }
        }
        uint32_t x = (words[623] & 0x80000000U) | (words[0] & 0x7fffffffU);
        words[623] = words[396] ^ (x >> 1) ^ ((x & 1) ? 0x9908b0dfU : 0U);
        words[624] = 0;
    }
    void discard(size_t count) {
        while (count) {
            if (words[624] == 624) twist();
            size_t step = std::min(count, size_t(624 - words[624]));
            words[624] += static_cast<uint32_t>(step);
            count -= step;
        }
    }
    val_t next() {
        if (words[624] == 624) twist();
        uint32_t y = words[words[624]++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return std::min(static_cast<val_t>(y) * 0x1p-32f, 0x1.fffffep-1f);
    }
};
static_assert(sizeof(RandomStream) == 625 * sizeof(uint32_t));

// Sixteen neighboring lanes trace a pair's rays; lane zero sums in the original
// ray order so floating-point reduction order never depends on the launch size.
__global__ void formFactorKernel(const Triangle* triangles, const FlatNode* nodes,
                                 const size_t* indices, const val_t* samples,
                                 val_t* kij, int* tau, size_t n, size_t rows,
                                 size_t rowBegin, size_t batchBegin, size_t pairs) {
    size_t pair = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / NUM_RAYS;
    int ray = threadIdx.x % NUM_RAYS;
    __shared__ val_t contributions[128];
    val_t value = ZERO;
    size_t i = rowBegin + batchBegin + pair / n, j = pair % n;
    if (pair < pairs && i != j && triangles[i].normal().dot(triangles[j].normal()) <= 0.99f) {
        const val_t* sample = samples + pair * (4 * NUM_RAYS) + ray * 4;
        Vec3 pI = samplePoint(triangles[i], sample[0], sample[1]);
        Vec3 pJ = samplePoint(triangles[j], sample[2], sample[3]);
        if (!blocked(pI, pJ, triangles, nodes, indices, i, j)) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cI = cosPhi(v, triangles[i].normal());
                val_t cJ = cosPhi(-v, triangles[j].normal());
                if (cI > ZERO && cJ > ZERO) value = (cI*cJ)/(PI*distSqr);
            }
        }
    }
    contributions[threadIdx.x] = value;
    __syncthreads();
    if (pair < pairs && ray == 0) {
        val_t sum = ZERO;
        for (int r = 0; r < NUM_RAYS; ++r) sum += contributions[threadIdx.x+r];
        size_t index = j * rows + batchBegin + pair / n;
        kij[index] = sum * INV_NUM_RAYS;
        tau[index] = i == j ? 0 : computeTau(triangles[i], triangles[j]);
    }
}
__global__ void countFactors(const val_t* kij, const int* tau, size_t count, unsigned long long* result, int* minDelay) {
    unsigned long long local = 0;
    int delay = INT_MAX;
    for (size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += size_t(gridDim.x)*blockDim.x) {
        local += kij[i] > EPSILON;
        if (tau[i] > 0) delay = min(delay, tau[i]);
    }
    __shared__ unsigned long long values[256];
    __shared__ int delays[256];
    values[threadIdx.x] = local;
    delays[threadIdx.x] = delay;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) {
            values[threadIdx.x] += values[threadIdx.x+step];
            delays[threadIdx.x] = min(delays[threadIdx.x], delays[threadIdx.x+step]);
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        atomicAdd(result, values[0]);
        atomicMin(minDelay, delays[0]);
    }
}
void computeFormFactors(SimulationState& state) {
    rankPrintf("Computing distributed form factors and time delays...\n");
    size_t n = state.numTriangles;
    std::vector<RandomStream> checkpoints;
    if (worldRank == 0) {
        std::vector<size_t> active(n, 0);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                active[i] += i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f;
        checkpoints.resize(n);
        RandomStream stream;
        for (size_t i = 0; i < n; ++i) {
            checkpoints[i] = stream;
            stream.discard(active[i] * NUM_RAYS * 4);
        }
    }
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = state.counts[r] * 625;
        offsets[r] = state.displacements[r] * 625;
    }
    std::vector<RandomStream> streams(state.localRows);
    MPI_Scatterv(checkpoints.data(), counts.data(), offsets.data(), MPI_UINT32_T,
                 streams.data(), counts[worldRank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    checkpoints.clear();
    checkpoints.shrink_to_fit();

    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenTree(state.octree, nodes, indices);
    DeviceBuffer<Triangle> trianglesDevice;
    DeviceBuffer<FlatNode> nodesDevice;
    DeviceBuffer<size_t> indicesDevice;
    trianglesDevice.resize(n); nodesDevice.resize(nodes.size()); indicesDevice.resize(indices.size());
    CUDA_CHECK(cudaMemcpy(trianglesDevice.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nodesDevice.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indicesDevice.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    // Bound staging memory independently of the quadratic distributed matrices.
    size_t batchRows = std::min(state.localRows, std::max(size_t(1), (size_t(1)<<18) / n));
    batchRows = std::max(size_t(1), batchRows);
    size_t sampleCount = batchRows*n*NUM_RAYS*4;
    PinnedBuffer<val_t> samples(sampleCount);
    DeviceBuffer<val_t> samplesDevice;
    samplesDevice.resize(sampleCount);
    for (size_t begin = 0; begin < state.localRows; begin += batchRows) {
        size_t rows = std::min(batchRows, state.localRows - begin);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row) {
            RandomStream stream = streams[begin+row];
            size_t i = state.rowBegin + begin + row;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                val_t* sample = samples.data + (row*n+j)*NUM_RAYS*4;
                for (int d = 0; d < NUM_RAYS*4; ++d) sample[d] = stream.next();
            }
        }
        CUDA_CHECK(cudaMemcpy(samplesDevice.data, samples.data, rows*n*NUM_RAYS*4*sizeof(val_t), cudaMemcpyHostToDevice));
        formFactorKernel<<<static_cast<unsigned>((rows*n+7)/8), 128>>>(trianglesDevice.data, nodesDevice.data,
            indicesDevice.data, samplesDevice.data, state.kijDevice.data, state.tauDevice.data,
            n, state.localRows, state.rowBegin, begin, rows*n);
        CUDA_CHECK(cudaGetLastError());
    }
    DeviceBuffer<unsigned long long> countDevice;
    countDevice.resize(1);
    CUDA_CHECK(cudaMemset(countDevice.data, 0, sizeof(unsigned long long)));
    DeviceBuffer<int> delayDevice;
    delayDevice.resize(1);
    int delay = INT_MAX;
    CUDA_CHECK(cudaMemcpy(delayDevice.data, &delay, sizeof(delay), cudaMemcpyHostToDevice));
    countFactors<<<256,256>>>(state.kijDevice.data, state.tauDevice.data, state.localRows*n, countDevice.data, delayDevice.data);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long count;
    CUDA_CHECK(cudaMemcpy(&count, countDevice.data, sizeof(count), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&count, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(&delay, delayDevice.data, sizeof(delay), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&delay, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
}

// Each lane owns one receiver. The transposed local matrix coalesces loads while
// retaining the original emitter accumulation order (no unordered atomics).
__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
                                val_t* history, val_t* output, size_t n, size_t rows,
                                size_t begin, size_t t, size_t source, size_t timeOff, val_t rho) {
    size_t row = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t i = begin + row;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t index = j*rows+row;
        int delay = tau[index];
        if (t < static_cast<size_t>(delay)) continue;
        val_t factor = kij[index];
        if (factor <= ZERO) continue;
        val_t rad = history[(t-delay)*n+j];
        if (rad <= ZERO) continue;
        sum += fminf(factor*areas[j], ONE)*rad;
    }
    val_t result = rho*sum + ((i == source && t < timeOff) ? ONE : ZERO);
    output[row] = result;
    history[t*n+i] = result;
}
__global__ void unpackHistory(const val_t* packed, val_t* history, size_t n,
                               size_t steps, size_t t, int ranks) {
    size_t k = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (k >= n*steps) return;
    size_t j = k % n, step = k / n;
    size_t rank = ((j+1)*ranks-1)/n;
    size_t begin = n*rank/ranks, rows = n*(rank+1)/ranks-begin;
    history[t*n+k] = packed[begin*steps+step*rows+j-begin];
}
void runSimulation(SimulationState& state) {
    rankPrintf("Running distributed wave propagation simulation...\n");
    size_t n = state.numTriangles;
    // No receiver can depend on another row within this time window. Exchange
    // several timesteps per collective without approximating the delayed model.
    size_t window = std::max(size_t(1), std::min(size_t(state.minDelay), state.numTimesteps));
    DeviceBuffer<val_t> output, packed;
    output.resize(state.localRows*window);
    packed.resize(worldSize > 1 ? n*window : 1);
    PinnedBuffer<val_t> gathered(worldSize > 1 ? n*window : 1);
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t t = 0; t < state.numTimesteps; t += window) {
        size_t steps = std::min(window, state.numTimesteps-t);
        if (state.localRows) {
            for (size_t step = 0; step < steps; ++step) {
                propagateKernel<<<static_cast<unsigned>((state.localRows+127)/128),128>>>(
                    state.kijDevice.data, state.tauDevice.data, state.areasDevice.data,
                    state.historyDevice.data, output.data+step*state.localRows, n, state.localRows, state.rowBegin,
                    t+step, state.sourceIndex, state.numTimesteps/2, state.rho[0]);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (worldSize > 1) {
            CUDA_CHECK(cudaMemcpy(gathered.data+state.rowBegin*steps, output.data,
                                  state.localRows*steps*sizeof(val_t), cudaMemcpyDeviceToHost));
            for (int r = 0; r < worldSize; ++r) {
                counts[r] = static_cast<int>(state.counts[r]*steps);
                offsets[r] = static_cast<int>(state.displacements[r]*steps);
            }
            // Pinned staging works with MPI implementations without CUDA awareness.
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, gathered.data,
                           counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(packed.data, gathered.data, n*steps*sizeof(val_t), cudaMemcpyHostToDevice));
            unpackHistory<<<static_cast<unsigned>((n*steps+255)/256),256>>>(
                packed.data, state.historyDevice.data, n, steps, t, worldSize);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
}

// Lags are independent. Each lane sums its correlation in the original order;
// a deterministic argmax keeps the earliest lag, including all-zero histories.
__global__ void distanceKernel(const val_t* history, val_t* distances, size_t n,
                               size_t timesteps, size_t begin, size_t source) {
    size_t i = begin + blockIdx.x;
    val_t best = ZERO;
    size_t lag = 0;
    for (size_t t = threadIdx.x; t < timesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt)
            sum += history[(tt-t)*n+source]*history[tt*n+i];
        if (sum > best) { best = sum; lag = t; }
    }
    __shared__ val_t values[128];
    __shared__ size_t lags[128];
    values[threadIdx.x] = best; lags[threadIdx.x] = lag;
    __syncthreads();
    for (int step = 64; step; step /= 2) {
        if (threadIdx.x < step) {
            int other = threadIdx.x+step;
            if (values[other] > values[threadIdx.x] ||
                (values[other] == values[threadIdx.x] && lags[other] < lags[threadIdx.x])) {
                values[threadIdx.x] = values[other]; lags[threadIdx.x] = lags[other];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) distances[blockIdx.x] = WAVE_SPEED*static_cast<val_t>(lags[0]);
}
void computeDistances(SimulationState& state) {
    rankPrintf("Computing distributed distances via cross-correlation...\n");
    DeviceBuffer<val_t> distances;
    distances.resize(state.localRows);
    if (state.localRows) {
        distanceKernel<<<static_cast<unsigned>(state.localRows),128>>>(state.historyDevice.data,
            distances.data, state.numTriangles, state.numTimesteps, state.rowBegin, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<val_t> local(state.localRows);
    if (state.localRows)
        CUDA_CHECK(cudaMemcpy(local.data(), distances.data, state.localRows*sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_Gatherv(local.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                state.distances.data(), state.counts.data(), state.displacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    rankPrintf("\nValidation:\n");

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
            rankPrintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rankPrintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rankPrintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rankPrintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rankPrintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rankPrintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
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

    rankPrintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rankPrintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
    rankPrintf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        rankPrintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rankPrintf("  Validation: PASSED\n");
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
    rankPrintf("Usage: %s [options]\n", progName);
    rankPrintf("Options:\n");
    rankPrintf("  -n <num>     Target number of triangles (default: 320)\n");
    rankPrintf("               Actual count will be rounded to nearest icosphere level:\n");
    rankPrintf("               20, 80, 320, 1280, 5120, 20480\n");
    rankPrintf("  -t <num>     Number of timesteps (default: 50)\n");
    rankPrintf("  -s <num>     Source triangle index (default: 0)\n");
    rankPrintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rankPrintf("  -v           Enable validation\n");
    rankPrintf("  -o           Print results for external validation\n");
    rankPrintf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
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
            rankPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0) {
        rankPrintf("Number of timesteps must be non-negative\n");
        return 1;
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rankPrintf("Room Response Simulation Benchmark\n");
    rankPrintf("===================================\n");
    rankPrintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rankPrintf("Timesteps: %d\n", timesteps);
    rankPrintf("Source triangle: %d\n", sourceIdx);
    rankPrintf("Reflectivity: %.2f\n", reflectivity);
    rankPrintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rankPrintf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rankPrintf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rankPrintf("Precomputation time: %ld ms\n", preDuration);
    rankPrintf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rankPrintf("Simulation time: %ld ms\n", simDuration);
    rankPrintf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rankPrintf("Distance computation time: %ld ms\n", distDuration);
    rankPrintf("\n");

    // Total time
    long localTime = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startPre).count();
    long totalTime = 0;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    rankPrintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    rankPrintf("\nPerformance:\n");
    rankPrintf("  Triangles: %zu\n", n);
    rankPrintf("  Timesteps: %zu\n", t);
    rankPrintf("  Form factor computations: %.2e\n", kijOps);
    rankPrintf("  Simulation operations: %.2e\n", simOps);
    rankPrintf("  Distance computations: %.2e\n", distOps);
    rankPrintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(int);
    size_t memRad = t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    rankPrintf("  Matrix/history memory on rank 0: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    rankPrintf("  Result hash: %016lX\n", hash);
    rankPrintf("\n");
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    int status = 0;
    if (validate && worldRank == 0) {
        state.radB.resize(t*n);
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.historyDevice.data, t*n*sizeof(val_t), cudaMemcpyDeviceToHost));
        if (!validateResults(state)) status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
    try {
        return runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
