module;

#include <vector>

export module SamplesCommon:LoadObj;
import :MeshPrimitives;
 
// import std;

using namespace mesh;

export struct LoadObjResult {
	std::vector<Vertex> vertices;
	std::vector<u32>    indices;
};

export auto LoadObj(std::string_view obj_path) -> LoadObjResult;
