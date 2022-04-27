#include "voxel/chunk.h"

Node::Node(int parentIdx, int firstChildIdx, int materialIdx, bool isVoxel, float3 color)
    : parentIdx(parentIdx), firstChildIdx(firstChildIdx), materialIdx(materialIdx), isVoxel(isVoxel), color(color) {}

Chunk::Chunk(int posx, int posy, int posz, int materialIdx)
    : posx(posx), posy(posy), posz(posz) {
    nodes.push_back(Node(-1, -1, materialIdx, true));
}

int Chunk::splitNode(int nodeIdx) {
    if (!nodes[nodeIdx].isVoxel) return -1;

    int firstChildIdx = addChildNodes(nodeIdx, nodes[nodeIdx].materialIdx, nodes[nodeIdx].color);

    nodes[nodeIdx].isVoxel = false;
    nodes[nodeIdx].firstChildIdx = firstChildIdx;

    return firstChildIdx;
}

int Chunk::combineNode(int nodeIdx, bool ignoreChildMaterials, int materialIdx) {
    if (nodes[nodeIdx].isVoxel) return -1;

    int firstChildIdx = nodes[nodeIdx].firstChildIdx;
    
    for (int i = 0; i < 8; i++) {
        if (!nodes[firstChildIdx + i].isVoxel) return -1;
    }

    if (!ignoreChildMaterials) {
        int firstChildMaterial = nodes[firstChildIdx].materialIdx;

        for (int i = 1; i < 8; i++) {
            if (nodes[firstChildIdx + i].materialIdx != firstChildMaterial) return -1;
        }
    }

    if (materialIdx != -1) {
        nodes[nodeIdx].materialIdx = materialIdx;
    }
    else {
        nodes[nodeIdx].materialIdx = nodes[firstChildIdx].materialIdx;
    }

    unusedStorageIndices.push(firstChildIdx);

    nodes[nodeIdx].firstChildIdx = -1;
    nodes[nodeIdx].isVoxel = true;

    return nodeIdx;
}

int Chunk::addChildNodes(int parentNodeIdx, int materialIdx, float3 color) {
    int idx;

    if (unusedStorageIndices.empty()) {
        idx = (int)nodes.size();
        for (int i = 0; i < 8; i++) {
            nodes.push_back(Node(parentNodeIdx, -1, materialIdx, true, color));
        }
    }
    else {
        idx = unusedStorageIndices.top();
        unusedStorageIndices.pop();
        for (int i = 0; i < 8; i++) {
            nodes[idx + i] = Node(parentNodeIdx, -1, materialIdx, true, color);
        }
    }

    return idx;
}