#pragma once

#include "util/buffer.cuh"

struct Material {
    float3 color;
    float transparent;

    Material() = default;
    Material(float3 color, float transparent);
};

class MaterialList {
public:
    MaterialList() = default;
    int registerMaterial(const Material& material);

    inline Material getMaterial(const int& idx) { return materials[idx]; }
    inline Material* getData() { return materials.data(); }
    inline int getAmount() { return (int)materials.size(); }

private:
    std::vector<Material> materials;
};