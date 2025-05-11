#ifndef BRDFCONSTANTS_H
#define BRDFCONSTANTS_H

#if defined(__cplusplus)
using std::uint32_t;
#endif

struct Material {
	float4 base_color;
	float  metallic;
	float  roughness;
	int    pad[2];
};

struct Light {
	float3 position;
	float  range;
	float3 color;
	float  intensity;
	float3 ambient_color;
	float  ambient_intensity;
};

struct BRDFConstants {
	float4x4 view_proj;
	Material material;
	Light    light;
	float3   camera_pos;
	int      pad;

	uint32_t weights_offsets[5];
	uint32_t bias_offsets[5];
	int      pad2[2];
};

#if defined(__cplusplus) || defined(__SLANG__)
enum class BrdfFunctionType : int {
	eDefault = 0,
	eCoopVec = 1,
	eCount
};
#endif

#endif // BRDFCONSTANTS_H
