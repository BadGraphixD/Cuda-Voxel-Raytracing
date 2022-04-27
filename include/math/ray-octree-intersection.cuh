#pragma once

#include <cuda_runtime.h>

#include "math/rt-math.cuh"
#include "util/buffer.cuh"
#include "voxel/chunk.h"
#include "voxel/material.h"

#define NODE_STACK_SIZE 10

struct Octree {

    struct Stack {

        struct SubtreeData {
            float tx0, ty0, tz0;
            float txm, tym, tzm;
            float tx1, ty1, tz1;
            int nodeIdx;
            int nextQuadrant;
        };

        int idx = 0;
        SubtreeData data[NODE_STACK_SIZE];

        __host__ __device__ inline void push(SubtreeData&& data) {
            this->data[idx] = data;
            idx++;
        }
        
        __host__ __device__ inline void pop() {
            idx--;
        }
        
        __host__ __device__ inline SubtreeData* top() {
            return data + (idx - 1);
        }

        __host__ __device__ inline bool isEmpty() {
            return idx <= 0;
        }

    };

    float3 min, max, center;

    Octree() = default;
    Octree(const Chunk& chunk) {
        min = {
            (float)chunk.posx,
            (float)chunk.posy,
            (float)chunk.posz
        };

        center = {
            (float)(chunk.posx + 0.5f),
            (float)(chunk.posy + 0.5f),
            (float)(chunk.posz + 0.5f)
        };

        max = {
            (float)(chunk.posx + 1.0f),
            (float)(chunk.posy + 1.0f),
            (float)(chunk.posz + 1.0f)
        };
    }

    __host__ __device__ int traverse(Ray ray, const Node* nodes, const Material* materials) {
        unsigned char a = 0;
        
        if (ray.dir.x < 0.0f) {
            ray.org.x = center.x * 2.0f - ray.org.x;
            ray.invDir.x = -ray.invDir.x;
            a |= 4;
        }
        if (ray.dir.y < 0.0f) {
            ray.org.y = center.y * 2.0f - ray.org.y;
            ray.invDir.y = -ray.invDir.y;
            a |= 2;
        }
        if (ray.dir.z < 0.0f) {
            ray.org.z = center.z * 2.0f - ray.org.z;
            ray.invDir.z = -ray.invDir.z;
            a |= 1;
        }

        const float tx0 = (min.x - ray.org.x) * ray.invDir.x;
        const float tx1 = (max.x - ray.org.x) * ray.invDir.x;
        const float ty0 = (min.y - ray.org.y) * ray.invDir.y;
        const float ty1 = (max.y - ray.org.y) * ray.invDir.y;
        const float tz0 = (min.z - ray.org.z) * ray.invDir.z;
        const float tz1 = (max.z - ray.org.z) * ray.invDir.z;

        if (fmaxf(fmaxf(tx0, ty0), tz0) < fminf(fminf(tx1, ty1), tz1)) {
            
            int foundNode = -1;
            Stack stack;

            foundNode = traverseNewNode(tx0, ty0, tz0, tx1, ty1, tz1, 0, stack, nodes, materials);

            while (!stack.isEmpty() && foundNode == -1) {
                Stack::SubtreeData* data = stack.top();
                foundNode = traverseChildNodes(data, a, stack, nodes, materials);
            }
            
            return foundNode;
        }

        return -1;
    }

    __host__ __device__ int traverseChildNodes(Stack::SubtreeData* data, const unsigned char& a, Stack& stack, const Node* nodes, const Material* materials) {

        switch (data->nextQuadrant) {
        case 0:
            data->nextQuadrant = getNextQuadrant(data->txm, 4, data->tym, 2, data->tzm, 1);
            return traverseNewNode(data->tx0, data->ty0, data->tz0, data->txm, data->tym, data->tzm, nodes[data->nodeIdx].firstChildIdx + a, stack, nodes, materials);
        case 1:
            data->nextQuadrant = getNextQuadrant(data->txm, 5, data->tym, 3, data->tz1, 8);
            return traverseNewNode(data->tx0, data->ty0, data->tzm, data->txm, data->tym, data->tz1, nodes[data->nodeIdx].firstChildIdx + (1 ^ a), stack, nodes, materials);
        case 2:
            data->nextQuadrant = getNextQuadrant(data->txm, 6, data->ty1, 8, data->tzm, 3);
            return traverseNewNode(data->tx0, data->tym, data->tz0, data->txm, data->ty1, data->tzm, nodes[data->nodeIdx].firstChildIdx + (2 ^ a), stack, nodes, materials);
        case 3:
            data->nextQuadrant = getNextQuadrant(data->txm, 7, data->ty1, 8, data->tz1, 8);
            return traverseNewNode(data->tx0, data->tym, data->tzm, data->txm, data->ty1, data->tz1, nodes[data->nodeIdx].firstChildIdx + (3 ^ a), stack, nodes, materials);
        case 4:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->tym, 6, data->tzm, 5);
            return traverseNewNode(data->txm, data->ty0, data->tz0, data->tx1, data->tym, data->tzm, nodes[data->nodeIdx].firstChildIdx + (4 ^ a), stack, nodes, materials);
        case 5:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->tym, 7, data->tz1, 8);
            return traverseNewNode(data->txm, data->ty0, data->tzm, data->tx1, data->tym, data->tz1, nodes[data->nodeIdx].firstChildIdx + (5 ^ a), stack, nodes, materials);
        case 6:
            data->nextQuadrant = getNextQuadrant(data->tx1, 8, data->ty1, 8, data->tzm, 7);
            return traverseNewNode(data->txm, data->tym, data->tz0, data->tx1, data->ty1, data->tzm, nodes[data->nodeIdx].firstChildIdx + (6 ^ a), stack, nodes, materials);
        case 7:
            data->nextQuadrant = 8;
            return traverseNewNode(data->txm, data->tym, data->tzm, data->tx1, data->ty1, data->tz1, nodes[data->nodeIdx].firstChildIdx + (7 ^ a), stack, nodes, materials);
        case 8:
            stack.pop();
            return -1;
        }

        return -1;
    }

    __host__ __device__ int traverseNewNode(const float& tx0, const float& ty0, const float& tz0, const float& tx1, const float& ty1, const float& tz1, const int& nodeIdx, Stack& stack, const Node* nodes, const Material* materials) {
        
        if (tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) return -1;

        if (nodes[nodeIdx].isVoxel) {
            return materials[nodes[nodeIdx].materialIdx].transparent ? -1 : nodeIdx;
        }

        const float txm = 0.5f * (tx0 + tx1);
        const float tym = 0.5f * (ty0 + ty1);
        const float tzm = 0.5f * (tz0 + tz1);
    
        stack.push({
            tx0, ty0, tz0,
            txm, tym, tzm,
            tx1, ty1, tz1,
            nodeIdx,
            getFirstQuadrant(tx0, ty0, tz0, txm, tym, tzm)
        });
        
        return -1;
    }

    __host__ __device__ int getFirstQuadrant(const float& tx0, const float& ty0, const float& tz0, const float& txm, const float& tym, const float& tzm) {
        unsigned char a = 0;

        if (tx0 > ty0) {
            if (tx0 > tz0){
                if (tym < tx0) a |= 2;
                if (tzm < tx0) a |= 1;
                return (int) a;
            }
        }

        else {
            if (ty0 > tz0) {
                if (txm < ty0) a |= 4;
                if (tzm < ty0) a |= 1;
                return (int) a;
            }
        }

        if (txm < tz0) a |= 4;
        if (tym < tz0) a |= 2;
        return (int) a;
    }

    __host__ __device__ int getNextQuadrant(const float& txm, const int& x, const float& tym, const int& y, const float& tzm, const int& z) {

        if (txm < tym) {
            if (txm < tzm) return x;
        }

        else {
            if (tym < tzm) return y;
        }
        
        return z;
    }

};
