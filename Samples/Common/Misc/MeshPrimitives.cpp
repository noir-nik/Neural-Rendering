module SamplesCommon;
import :MeshPrimitives;
import std;

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
	std::vector<UVSphereVertex>& vertices,
	std::vector<u32>&            indices) -> void {

	float const pi            = 3.14159265358979323846f;
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

