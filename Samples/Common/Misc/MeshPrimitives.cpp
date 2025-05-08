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
}

auto UVSphere::WriteVertices(Vertex* vertices) const -> void {
	float const segment_angle = 2.0f * pi / segments;
	float const ring_angle    = pi / rings;

	for (int v = 0; v <= rings; ++v) {
		float const phi = float(v) / rings * pi;
		float const y = std::cos(phi);
		float const r = std::sin(phi);

		for (int u = 0; u <= segments; ++u) {
			float const uvx = float(u) / segments;
			float const x = std::cos(segment_angle * u) * r;
			float const z = std::sin(segment_angle * u) * r;

			auto array_index = v * (segments + 1) + u;

			vertices[array_index] = {{x * radius, y * radius, z * radius}, uvx, {x, y, z}, phi};
		}
	}
}

auto UVSphere::WriteIndices(u32* indices) const -> void {
	auto segment_cap = segments + 1;

	for (u32 v = 0; v < rings; ++v) {
		for (u32 u = 0; u < segments; ++u) {

			u32 index0 = v * segment_cap + u;
			u32 index1 = v * segment_cap + (u + 1) % segment_cap;
			u32 index2 = (v + 1) * segment_cap + u;
			u32 index3 = (v + 1) * segment_cap + (u + 1) % segment_cap;

			auto base_index = v * segments * 6 + u * 6;

			indices[base_index + 0] = index0;
			indices[base_index + 1] = index1;
			indices[base_index + 2] = index2;

			indices[base_index + 3] = index2;
			indices[base_index + 4] = index1;
			indices[base_index + 5] = index3;
		}
	}
};

} // namespace mesh
