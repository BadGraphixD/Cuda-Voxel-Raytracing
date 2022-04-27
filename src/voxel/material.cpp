#include "voxel/material.h"

Material::Material(float3 color, float transparent)
    : color(color), transparent(transparent) {}

int MaterialList::registerMaterial(const Material& material) {
    materials.push_back(material);
    return (int)materials.size() - 1;
}