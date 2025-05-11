export module SamplesCommon:MeshPrimitives;
import std;
export namespace mesh {

using u32 = std::uint32_t;

struct CubeVertex {
	float pos[4];
	// int pad;
};

template <typename VertexType, typename IndexType>
struct Mesh {
	std::vector<VertexType> vertices;
	std::vector<IndexType>  indices;
};

class UVSphere {
public:
	static constexpr float pi = 3.14159265358979323846f;

	struct Vertex {
		float pos[3];
		float u;
		float normal[3];
		float v;
	};

	using IndexType = u32;

	UVSphere(float radius, u32 segments, u32 rings) : radius(radius), segments(segments), rings(rings) {}

	auto GetVertexCount() const -> u32 { return (rings + 1) * (segments + 1); }
	auto GetIndexCount() const -> u32 { return rings * segments * 6; }

	auto WriteVertices(Vertex* vertices) const -> void;

	auto WriteIndices(u32* indices) const -> void;

private:
	float radius;
	u32   segments;
	u32   rings;
};


auto GetCubeVertices() -> std::span<CubeVertex const>;
 

} // namespace mesh
