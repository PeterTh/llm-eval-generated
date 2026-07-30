/**
 * Room Response Simulation Benchmark - CUDA Parallel Implementation
 *
 * GPU-parallelized version using CUDA for maximum performance.
 * All major computational phases are executed on GPU:
 * - Form factor (Kij) computation with grid-based spatial hashing
 * - Time delay (Tau) computation
 * - Wave propagation simulation
 * - Distance estimation via cross-correlation
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
#include <cfloat>

#include "../common/results_output.hpp"

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
    Vec3 cross(const Vec3& o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }
    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const { val_t n = norm(); return n > EPSILON ? *this / n : Vec3(); }
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
    val_t area() const { Vec3 ab = b - a; Vec3 ac = c - a; return 0.5f * ab.cross(ac).norm(); }
    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter, v1 = tri.b - boxCenter, v2 = tri.c - boxCenter;
    Vec3 e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;
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
    val_t r = boxHalfSize.x * std::abs(triNormal.x) + boxHalfSize.y * std::abs(triNormal.y) + boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0), p1 = axis.dot(v1), p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) + boxHalfSize.y * std::abs(axis.y) + boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };
    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON && !testAxis(crossAxis)) return false;
        }
    }
    return true;
}

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound, halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;
    Octree() : allTriangles(nullptr) {}
    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;
        minBound = triangles[0].a; maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x); minBound.y = std::min(minBound.y, v->y); minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x); maxBound.y = std::max(maxBound.y, v->y); maxBound.z = std::max(maxBound.z, v->z);
            }
        }
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;
        buildNode(allIndices, minBound, maxBound);
    }
private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin; maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f; center = (minBound + maxBound) * 0.5f;
        if (indices.size() <= MAX_OCTREE_TRIS || (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices; return;
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
                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) childIndices[i].push_back(idx);
            }
        }
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) { canSplit = true; break; }
        }
        if (!canSplit) { triangleIndices = indices; return; }
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center, childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x; childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y; childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z; childMax.z = (i & 4) ? maxBound.z : center.z;
                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }
};

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;
    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;
        std::vector<Vec3> vertices = {
            Vec3(-1, t, 0).normalized() * radius, Vec3(1, t, 0).normalized() * radius,
            Vec3(-1, -t, 0).normalized() * radius, Vec3(1, -t, 0).normalized() * radius,
            Vec3(0, -1, t).normalized() * radius, Vec3(0, 1, t).normalized() * radius,
            Vec3(0, -1, -t).normalized() * radius, Vec3(0, 1, -t).normalized() * radius,
            Vec3(t, 0, -1).normalized() * radius, Vec3(t, 0, 1).normalized() * radius,
            Vec3(-t, 0, -1).normalized() * radius, Vec3(-t, 0, 1).normalized() * radius
        };
        std::vector<std::array<idx_t, 3>> faces = {
            {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
            {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
            {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
            {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
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
                midpointCache[key] = idx; return idx;
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
        for (const auto& face : faces)
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
    }
};

// ============================================================================
// CUDA Grid-based Spatial Hash for fast visibility testing
// ============================================================================

struct DeviceVec3 { float x, y, z; };
struct DeviceTriangle { DeviceVec3 a, b, c, normal; };

// Grid cell: stores triangle indices
// Using a linked-list approach: each cell has a start index and count
struct GridCell {
    int triStartIdx;  // index into flatTriIndices array
    int triCount;
};

// Build grid on CPU, transfer to GPU
void buildGrid(const std::vector<Triangle>& triangles,
               float gridMin[3], float gridMax[3], float cellSize,
               int gridSize[3],
               std::vector<GridCell>& gridCells,
               std::vector<int>& flatTriIndices) {
    int totalCells = gridSize[0] * gridSize[1] * gridSize[2];
    gridCells.resize(totalCells, {0, 0});
    float invCellSize = 1.0f / cellSize;

    // First pass: count triangles per cell
    std::vector<int> cellCounts(totalCells, 0);
    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& tri = triangles[i];
        float minx = std::min({tri.a.x, tri.b.x, tri.c.x});
        float maxx = std::max({tri.a.x, tri.b.x, tri.c.x});
        float miny = std::min({tri.a.y, tri.b.y, tri.c.y});
        float maxy = std::max({tri.a.y, tri.b.y, tri.c.y});
        float minz = std::min({tri.a.z, tri.b.z, tri.c.z});
        float maxz = std::max({tri.a.z, tri.b.z, tri.c.z});

        int ix0 = std::max(0, static_cast<int>(floorf((minx - gridMin[0]) * invCellSize)));
        int ix1 = std::min(gridSize[0] - 1, static_cast<int>(floorf((maxx - gridMin[0]) * invCellSize)));
        int iy0 = std::max(0, static_cast<int>(floorf((miny - gridMin[1]) * invCellSize)));
        int iy1 = std::min(gridSize[1] - 1, static_cast<int>(floorf((maxy - gridMin[1]) * invCellSize)));
        int iz0 = std::max(0, static_cast<int>(floorf((minz - gridMin[2]) * invCellSize)));
        int iz1 = std::min(gridSize[2] - 1, static_cast<int>(floorf((maxz - gridMin[2]) * invCellSize)));

        for (int iz = iz0; iz <= iz1; ++iz)
            for (int iy = iy0; iy <= iy1; ++iy)
                for (int ix = ix0; ix <= ix1; ++ix) {
                    int cellIdx = iz * gridSize[0] * gridSize[1] + iy * gridSize[0] + ix;
                    cellCounts[cellIdx]++;
                }
    }

    // Compute prefix sums for cell start indices
    int running = 0;
    for (int i = 0; i < totalCells; ++i) {
        gridCells[i].triStartIdx = running;
        gridCells[i].triCount = cellCounts[i];
        running += cellCounts[i];
    }

    flatTriIndices.resize(running, -1);

    // Second pass: fill triangle indices
    std::vector<int> cellCursor(totalCells);
    for (int i = 0; i < totalCells; ++i) cellCursor[i] = gridCells[i].triStartIdx;

    for (size_t i = 0; i < triangles.size(); ++i) {
        const Triangle& tri = triangles[i];
        float minx = std::min({tri.a.x, tri.b.x, tri.c.x});
        float maxx = std::max({tri.a.x, tri.b.x, tri.c.x});
        float miny = std::min({tri.a.y, tri.b.y, tri.c.y});
        float maxy = std::max({tri.a.y, tri.b.y, tri.c.y});
        float minz = std::min({tri.a.z, tri.b.z, tri.c.z});
        float maxz = std::max({tri.a.z, tri.b.z, tri.c.z});

        int ix0 = std::max(0, static_cast<int>(floorf((minx - gridMin[0]) * invCellSize)));
        int ix1 = std::min(gridSize[0] - 1, static_cast<int>(floorf((maxx - gridMin[0]) * invCellSize)));
        int iy0 = std::max(0, static_cast<int>(floorf((miny - gridMin[1]) * invCellSize)));
        int iy1 = std::min(gridSize[1] - 1, static_cast<int>(floorf((maxy - gridMin[1]) * invCellSize)));
        int iz0 = std::max(0, static_cast<int>(floorf((minz - gridMin[2]) * invCellSize)));
        int iz1 = std::min(gridSize[2] - 1, static_cast<int>(floorf((maxz - gridMin[2]) * invCellSize)));

        for (int iz = iz0; iz <= iz1; ++iz)
            for (int iy = iy0; iy <= iy1; ++iy)
                for (int ix = ix0; ix <= ix1; ++ix) {
                    int cellIdx = iz * gridSize[0] * gridSize[1] + iy * gridSize[0] + ix;
                    flatTriIndices[cellCursor[cellIdx]++] = static_cast<int>(i);
                }
    }
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__device__ float devRayTriangleIntersect(const DeviceVec3& orig, const DeviceVec3& dir,
                                          const DeviceVec3& v0, const DeviceVec3& v1, const DeviceVec3& v2) {
    DeviceVec3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    DeviceVec3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
    DeviceVec3 pvec = {dir.y * e2.z - dir.z * e2.y, dir.z * e2.x - dir.x * e2.z, dir.x * e2.y - dir.y * e2.x};
    float det = e1.x * pvec.x + e1.y * pvec.y + e1.z * pvec.z;
    if (fabsf(det) < EPSILON) return FLT_MAX;
    float invDet = 1.0f / det;
    DeviceVec3 tvec = {orig.x - v0.x, orig.y - v0.y, orig.z - v0.z};
    float u = (tvec.x * pvec.x + tvec.y * pvec.y + tvec.z * pvec.z) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    DeviceVec3 qvec = {tvec.y * e1.z - tvec.z * e1.y, tvec.z * e1.x - tvec.x * e1.z, tvec.x * e1.y - tvec.y * e1.x};
    float v = (dir.x * qvec.x + dir.y * qvec.y + dir.z * qvec.z) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return (e2.x * qvec.x + e2.y * qvec.y + e2.z * qvec.z) * invDet;
}

__device__ DeviceVec3 devRandomPointInTriangle(const DeviceTriangle& t, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    DeviceVec3 ab = {t.b.x - t.a.x, t.b.y - t.a.y, t.b.z - t.a.z};
    DeviceVec3 ac = {t.c.x - t.a.x, t.c.y - t.a.y, t.c.z - t.a.z};
    return {t.a.x + ab.x * u + ac.x * v, t.a.y + ab.y * u + ac.y * v, t.a.z + ab.z * u + ac.z * v};
}

__device__ float devCosPhi(const DeviceVec3& v, const DeviceVec3& normal) {
    float vNorm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, (v.x * normal.x + v.y * normal.y + v.z * normal.z) / vNorm);
}

__device__ uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state; x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return state = x;
}
__device__ float xorshiftFloat(uint32_t& state) {
    return static_cast<float>(xorshift32(state)) / 4294967295.0f;
}

// Grid-based visibility test using 3D DDA ray traversal
__device__ bool devIsRayBlockedGrid(const DeviceVec3& from, const DeviceVec3& to,
                                     const GridCell* gridCells, const int* flatTriIndices,
                                     const DeviceTriangle* triangles,
                                     const float* gridMin,
                                     float cellSize, int gridSizeX, int gridSizeY, int gridSizeZ,
                                     int srcTriIdx, int dstTriIdx) {
    float rayLen = sqrtf((to.x - from.x) * (to.x - from.x) + (to.y - from.y) * (to.y - from.y) + (to.z - from.z) * (to.z - from.z));
    if (rayLen < EPSILON) return true;
    DeviceVec3 dir = {(to.x - from.x) / rayLen, (to.y - from.y) / rayLen, (to.z - from.z) / rayLen};

    // 3D DDA ray traversal
    float invCellSize = 1.0f / cellSize;
    int ix = static_cast<int>(floorf((from.x - gridMin[0]) * invCellSize));
    int iy = static_cast<int>(floorf((from.y - gridMin[1]) * invCellSize));
    int iz = static_cast<int>(floorf((from.z - gridMin[2]) * invCellSize));

    // Clamp to grid bounds
    ix = max(0, min(gridSizeX - 1, ix));
    iy = max(0, min(gridSizeY - 1, iy));
    iz = max(0, min(gridSizeZ - 1, iz));

    // Compute step directions and TMax
    int stepX = (dir.x >= 0) ? 1 : -1;
    int stepY = (dir.y >= 0) ? 1 : -1;
    int stepZ = (dir.z >= 0) ? 1 : -1;

    float tMaxX, tMaxY, tMaxZ;
    float tDeltaX, tDeltaY, tDeltaZ;

    if (fabsf(dir.x) < 1e-10f) { tMaxX = FLT_MAX; tDeltaX = FLT_MAX; }
    else {
        tDeltaX = cellSize / fabsf(dir.x);
        tMaxX = ((stepX > 0) ? (ix + 1) : ix) * cellSize + gridMin[0] - from.x;
        tMaxX /= dir.x;
    }
    if (fabsf(dir.y) < 1e-10f) { tMaxY = FLT_MAX; tDeltaY = FLT_MAX; }
    else {
        tDeltaY = cellSize / fabsf(dir.y);
        tMaxY = ((stepY > 0) ? (iy + 1) : iy) * cellSize + gridMin[1] - from.y;
        tMaxY /= dir.y;
    }
    if (fabsf(dir.z) < 1e-10f) { tMaxZ = FLT_MAX; tDeltaZ = FLT_MAX; }
    else {
        tDeltaZ = cellSize / fabsf(dir.z);
        tMaxZ = ((stepZ > 0) ? (iz + 1) : iz) * cellSize + gridMin[2] - from.z;
        tMaxZ /= dir.z;
    }

    // Track visited cells to avoid duplicate triangle checks
    // Use a small local set for visited triangle indices
    int visited[64];
    int visitedCount = 0;

    auto markVisited = [&](int idx) {
        for (int k = 0; k < visitedCount; ++k) {
            if (visited[k] == idx) return true;
        }
        if (visitedCount < 64) { visited[visitedCount++] = idx; return false; }
        return true;
    };

    float t = 0.0f;
    while (t < rayLen) {
        int cellIdx = iz * gridSizeX * gridSizeY + iy * gridSizeX + ix;
        const GridCell& cell = gridCells[cellIdx];

        for (int k = 0; k < cell.triCount; ++k) {
            int triIdx = flatTriIndices[cell.triStartIdx + k];
            if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;
            if (markVisited(triIdx)) continue;

            const DeviceTriangle& tri = triangles[triIdx];
            float dist = devRayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        }

        // Advance to next cell
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                t = tMaxX; ix += stepX; tMaxX += tDeltaX;
            } else {
                t = tMaxZ; iz += stepZ; tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                t = tMaxY; iy += stepY; tMaxY += tDeltaY;
            } else {
                t = tMaxZ; iz += stepZ; tMaxZ += tDeltaZ;
            }
        }

        // Check bounds
        if (ix < 0 || ix >= gridSizeX || iy < 0 || iy >= gridSizeY || iz < 0 || iz >= gridSizeZ) break;
    }

    return false;
}

__global__ void computeTauKernel(const DeviceTriangle* d_triangles, int* d_tau, int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numTriangles || j >= numTriangles || i == j) return;
    const DeviceTriangle& triI = d_triangles[i];
    const DeviceTriangle& triJ = d_triangles[j];
    float ci_x = (triI.a.x + triI.b.x + triI.c.x) / 3.0f;
    float ci_y = (triI.a.y + triI.b.y + triI.c.y) / 3.0f;
    float ci_z = (triI.a.z + triI.b.z + triI.c.z) / 3.0f;
    float cj_x = (triJ.a.x + triJ.b.x + triJ.c.x) / 3.0f;
    float cj_y = (triJ.a.y + triJ.b.y + triJ.c.y) / 3.0f;
    float cj_z = (triJ.a.z + triJ.b.z + triJ.c.z) / 3.0f;
    float dist = sqrtf((ci_x - cj_x) * (ci_x - cj_x) + (ci_y - cj_y) * (ci_y - cj_y) + (ci_z - cj_z) * (ci_z - cj_z));
    d_tau[i * numTriangles + j] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void computeKijKernel(const DeviceTriangle* d_triangles,
                                  const GridCell* d_gridCells,
                                  const int* d_flatTriIndices,
                                  const float* d_gridMin, float cellSize,
                                  int gridSizeX, int gridSizeY, int gridSizeZ,
                                  float* d_kij, int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numTriangles || j >= numTriangles || i == j) return;

    const DeviceTriangle& triI = d_triangles[i];
    const DeviceTriangle& triJ = d_triangles[j];

    float normalDot = triI.normal.x * triJ.normal.x + triI.normal.y * triJ.normal.y + triI.normal.z * triJ.normal.z;
    if (normalDot > 0.99f) { d_kij[i * numTriangles + j] = ZERO; return; }

    uint32_t rngState = static_cast<uint32_t>(i * 73856093u + j * 19349663u + 42u);
    float kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        float u = xorshiftFloat(rngState), v = xorshiftFloat(rngState);
        DeviceVec3 pI = devRandomPointInTriangle(triI, u, v);
        u = xorshiftFloat(rngState); v = xorshiftFloat(rngState);
        DeviceVec3 pJ = devRandomPointInTriangle(triJ, u, v);

        if (devIsRayBlockedGrid(pI, pJ, d_gridCells, d_flatTriIndices, d_triangles,
                                 d_gridMin, cellSize, gridSizeX, gridSizeY, gridSizeZ, i, j)) continue;

        DeviceVec3 vVec = {pJ.x - pI.x, pJ.y - pI.y, pJ.z - pI.z};
        float distSqr = vVec.x * vVec.x + vVec.y * vVec.y + vVec.z * vVec.z;
        if (distSqr < EPSILON) continue;

        float cosPhiI = devCosPhi(vVec, triI.normal);
        DeviceVec3 negV = {-vVec.x, -vVec.y, -vVec.z};
        float cosPhiJ = devCosPhi(negV, triJ.normal);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    d_kij[i * numTriangles + j] = kij * INV_NUM_RAYS;
}

__global__ void simulateStepKernel(const float* d_kij, const int* d_tau,
                                    const float* d_areas, const float* d_rho,
                                    const float* d_radE, float* d_radB,
                                    int numTriangles, int numTimesteps, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;
    float sumB = ZERO;
    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;
        int tauij = d_tau[i * numTriangles + j];
        if (t < tauij) continue;
        float kijVal = d_kij[i * numTriangles + j];
        if (kijVal <= ZERO) continue;
        int srcTime = t - tauij;
        float radJ = d_radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;
        sumB += fminf(kijVal * d_areas[j], ONE) * radJ;
    }
    d_radB[t * numTriangles + i] = d_rho[i] * sumB + d_radE[t * numTriangles + i];
}

__global__ void computeDistanceKernel(const float* d_radB, float* d_distances,
                                       int numTriangles, int numTimesteps, int sourceIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;
    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < numTimesteps; ++t) {
        float sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            float pB = d_radB[tt * numTriangles + i];
            float pS = d_radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
    }
    d_distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

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
    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) state.areas[i] = state.triangles[i].area();
    state.rho.resize(state.numTriangles, reflectivity);
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
    for (size_t t = 0; t < timesteps / 2; ++t)
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void computeTimeDelaysGPU(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");
    int n = static_cast<int>(state.numTriangles);
    size_t n2 = static_cast<size_t>(n) * n;
    DeviceTriangle* d_triangles = nullptr;
    int* d_tau = nullptr;
    std::vector<DeviceTriangle> h_triangles(n);
    for (int i = 0; i < n; ++i) {
        h_triangles[i].a = {state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z};
        h_triangles[i].b = {state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z};
        h_triangles[i].c = {state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z};
        h_triangles[i].normal = {state.triangles[i]._normal.x, state.triangles[i]._normal.y, state.triangles[i]._normal.z};
    }
    CUDA_CHECK(cudaMalloc(&d_triangles, n * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(&d_tau, n2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), n * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_tau, 0, n2 * sizeof(int)));
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    computeTauKernel<<<grid, block>>>(d_triangles, d_tau, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    state.tau.resize(n2);
    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau, n2 * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_tau));
}

void computeFormFactorsGPU(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");
    int n = static_cast<int>(state.numTriangles);
    size_t n2 = static_cast<size_t>(n) * n;

    // Build grid for spatial hashing
    float gridMin[3], gridMax[3];
    gridMin[0] = state.octree.minBound.x; gridMin[1] = state.octree.minBound.y; gridMin[2] = state.octree.minBound.z;
    gridMax[0] = state.octree.maxBound.x; gridMax[1] = state.octree.maxBound.y; gridMax[2] = state.octree.maxBound.z;
    float cellSize = MAX_OCTREE_LEAF_SIZE;
    int gridSize[3];
    gridSize[0] = static_cast<int>(ceilf((gridMax[0] - gridMin[0]) / cellSize)) + 1;
    gridSize[1] = static_cast<int>(ceilf((gridMax[1] - gridMin[1]) / cellSize)) + 1;
    gridSize[2] = static_cast<int>(ceilf((gridMax[2] - gridMin[2]) / cellSize)) + 1;

    std::vector<GridCell> gridCells;
    std::vector<int> flatTriIndices;
    buildGrid(state.triangles, gridMin, gridMax, cellSize, gridSize, gridCells, flatTriIndices);

    DeviceTriangle* d_triangles = nullptr;
    GridCell* d_gridCells = nullptr;
    int* d_flatTriIndices = nullptr;
    float* d_gridMin = nullptr;
    float* d_kij = nullptr;

    std::vector<DeviceTriangle> h_triangles(n);
    for (int i = 0; i < n; ++i) {
        h_triangles[i].a = {state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z};
        h_triangles[i].b = {state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z};
        h_triangles[i].c = {state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z};
        h_triangles[i].normal = {state.triangles[i]._normal.x, state.triangles[i]._normal.y, state.triangles[i]._normal.z};
    }

    CUDA_CHECK(cudaMalloc(&d_triangles, n * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(&d_gridCells, gridCells.size() * sizeof(GridCell)));
    CUDA_CHECK(cudaMalloc(&d_flatTriIndices, flatTriIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridMin, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kij, n2 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), n * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gridCells, gridCells.data(), gridCells.size() * sizeof(GridCell), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flatTriIndices, flatTriIndices.data(), flatTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gridMin, gridMin, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_kij, 0, n2 * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    computeKijKernel<<<grid, block>>>(d_triangles, d_gridCells, d_flatTriIndices,
                                       d_gridMin, cellSize, gridSize[0], gridSize[1], gridSize[2],
                                       d_kij, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    state.kij.resize(n2);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), d_kij, n2 * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_gridCells));
    CUDA_CHECK(cudaFree(d_flatTriIndices));
    CUDA_CHECK(cudaFree(d_gridMin));
    CUDA_CHECK(cudaFree(d_kij));
}

void runSimulationGPU(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");
    int n = static_cast<int>(state.numTriangles);
    int timesteps = static_cast<int>(state.numTimesteps);
    size_t n2 = static_cast<size_t>(n) * n;
    size_t tn = static_cast<size_t>(timesteps) * n;

    float* d_kij = nullptr; int* d_tau = nullptr;
    float* d_areas = nullptr; float* d_rho = nullptr;
    float* d_radE = nullptr; float* d_radB = nullptr;

    CUDA_CHECK(cudaMalloc(&d_kij, n2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, n2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, tn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radB, tn * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), n2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), n2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), tn * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_radB, 0, tn * sizeof(float)));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int t = 0; t < timesteps; ++t) {
        simulateStepKernel<<<numBlocks, blockSize>>>(d_kij, d_tau, d_areas, d_rho, d_radE, d_radB, n, timesteps, t);
        CUDA_CHECK(cudaDeviceSynchronize());
        if ((t + 1) % 10 == 0 || t + 1 == timesteps)
            printf("  Timestep %d/%d\n", t + 1, timesteps);
    }

    state.radB.resize(tn);
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, tn * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_kij)); CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas)); CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE)); CUDA_CHECK(cudaFree(d_radB));
}

void computeDistancesGPU(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");
    int n = static_cast<int>(state.numTriangles);
    int timesteps = static_cast<int>(state.numTimesteps);
    size_t tn = static_cast<size_t>(timesteps) * n;

    float* d_radB = nullptr; float* d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_radB, tn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), tn * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    computeDistanceKernel<<<numBlocks, blockSize>>>(d_radB, d_distances, n, timesteps, static_cast<int>(state.sourceIndex));
    CUDA_CHECK(cudaDeviceSynchronize());

    state.distances.resize(n);
    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_radB)); CUDA_CHECK(cudaFree(d_distances));
}

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) { allNonNegative = false; printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d); }
        if (!std::isfinite(d)) { printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d); return false; }
        minDist = std::min(minDist, d); maxDist = std::max(maxDist, d); sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }
    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) { receivedEnergy++; break; }
        }
    }
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
    if (receivedEnergy == 0) { printf("  ERROR: No triangles received energy - simulation failed\n"); return false; }
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n", nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));
    if (nonZeroKij == 0) { printf("  ERROR: All form factors are zero - visibility computation failed\n"); return false; }
    if (!allNonNegative) return false;
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

int getSubdivisionsForTriangleCount(int targetTriangles) {
    int subdivisions = 0, triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) { subdivisions++; triangles *= 4; }
    return subdivisions;
}

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
    int targetTriangles = 320, timesteps = 50, sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) targetTriangles = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) timesteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sourceIdx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) reflectivity = static_cast<val_t>(atof(argv[++i]));
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-o") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    printf("\n");
    auto startPre = std::chrono::high_resolution_clock::now();
    computeTimeDelaysGPU(state);
    computeFormFactorsGPU(state);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    printf("Precomputation time: %ld ms\n", preDuration); printf("\n");
    auto startSim = std::chrono::high_resolution_clock::now();
    runSimulationGPU(state);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    printf("Simulation time: %ld ms\n", simDuration); printf("\n");
    auto startDist = std::chrono::high_resolution_clock::now();
    computeDistancesGPU(state);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    printf("Distance computation time: %ld ms\n", distDuration); printf("\n");
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);
    size_t n = state.numTriangles, t = state.numTimesteps;
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
    printf("  Result hash: %016lX\n", hash); printf("\n");
    if (printResults) {
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }
    if (validate) {
        if (!validateResults(state)) return 1;
    }
    return 0;
}
