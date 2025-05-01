export module SamplesCommon:MeshPrimitives;
import std;

export using u32 = std::uint32_t;

export struct CubeVertex {
	float pos[4];
	// int pad;
};

export template <typename VertexType, typename IndexType>
struct Mesh {
	std::vector<VertexType> vertices;
	std::vector<IndexType>  indices;
};

export struct UVSphereVertex {
	float pos[3];
	float uv[2];
};

export auto GetCubeVertices() -> std::span<CubeVertex const>;

export auto GenerateUVSphereVerticesAndIndices(
	float radius, u32 segments, u32 rings,
	std::vector<UVSphereVertex>& vertices,
	std::vector<u32>&            indices) -> void;
