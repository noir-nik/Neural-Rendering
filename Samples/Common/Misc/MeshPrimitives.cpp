module SamplesCommon;
import :MeshPrimitives;
import std;

namespace mesh {
// clang-format off
static constexpr CubeVertex cube_vertices[] = {
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f,  0.5f, -0.5f, 0.0f},
	{ 0.5f,  0.5f, -0.5f, 0.0f},
	{-0.5f,  0.5f, -0.5f, 0.0f},
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{-0.5f, -0.5f,  0.5f, 0.0f},
	{ 0.5f, -0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f, -0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f, -0.5f, 0.0f},
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{-0.5f, -0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f, -0.5f, 0.0f},
	{ 0.5f, -0.5f,  0.5f, 0.0f},
	{ 0.5f, -0.5f,  0.5f, 0.0f},
	{-0.5f, -0.5f,  0.5f, 0.0f},
	{-0.5f, -0.5f, -0.5f, 0.0f},
	{-0.5f,  0.5f, -0.5f, 0.0f},
	{ 0.5f,  0.5f, -0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{ 0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f,  0.5f, 0.0f},
	{-0.5f,  0.5f, -0.5f, 0.0f},
};
// clang-format on

auto GetCubeVertices() -> std::span<CubeVertex const> { return {cube_vertices}; }

auto GenerateUVSphereVerticesAndIndices(
	float radius, u32 segments, u32 rings,
	std::vector<UVSphere::Vertex>& vertices,
	std::vector<u32>&              indices) -> void {

	constexpr float pi = 3.14159265358979323846f;

	float const segment_angle = 2.0f * pi / segments;
	float const ring_angle    = pi / rings;

	vertices.reserve((rings + 1) * (segments + 1));
	indices.reserve(rings * segments * 6);

	for (int i = 0; i <= rings; ++i) {
		float const t = float(i) / rings;
		float const y = std::cos(t * pi) * radius;
		float const r = std::sin(t * pi) * radius;

		for (int j = 0; j <= segments; ++j) {
			float const s = float(j) / segments;
			float const x = std::cos(segment_angle * j) * r;
			float const z = std::sin(segment_angle * j) * r;

			vertices.push_back({{x, y, z}, {s, t}});
		}
	}

	for (int i = 0; i < rings; ++i) {
		for (int j = 0; j < segments; ++j) {
			indices.push_back(static_cast<u32>(i * (segments + 1) + j));
			indices.push_back(static_cast<u32>(i * (segments + 1) + j + 1));
			indices.push_back(static_cast<u32>((i + 1) * (segments + 1) + j + 1));

			indices.push_back(static_cast<u32>(i * (segments + 1) + j));
			indices.push_back(static_cast<u32>((i + 1) * (segments + 1) + j + 1));
			indices.push_back(static_cast<u32>((i + 1) * (segments + 1) + j));
		}
	}
}

auto UVSphere::WriteVertices(Vertex* vertices) const -> void {
	float const segment_angle = 2.0f * pi / segments;
	float const ring_angle    = pi / rings;

	for (int i = 0; i <= rings; ++i) {
		float const t = float(i) / rings;
		float const y = std::cos(t * pi) * radius;
		float const r = std::sin(t * pi) * radius;

		for (int j = 0; j <= segments; ++j) {
			float const s = float(j) / segments;
			float const x = std::cos(segment_angle * j) * r;
			float const z = std::sin(segment_angle * j) * r;

			auto index = i * (segments + 1) + j;

			vertices[index] = {{x, y, z}, {s, t}};
		}
	}
}

auto UVSphere::WriteIndices(u32* indices) const -> void {
	for (u32 i = 0; i < rings; ++i) {
		for (u32 j = 0; j < segments; ++j) {

			auto base_index = i * segments * 6 + j * 6;

			indices[base_index + 0] = i * (segments + 1) + j + 0;
			indices[base_index + 1] = i * (segments + 1) + j + 1;
			indices[base_index + 2] = (i + 1) * (segments + 1) + j + 1;

			indices[base_index + 3] = i * (segments + 1) + j + 0;
			indices[base_index + 4] = (i + 1) * (segments + 1) + j + 1;
			indices[base_index + 5] = (i + 1) * (segments + 1) + j + 0;
		}
	}
};

} // namespace mesh
