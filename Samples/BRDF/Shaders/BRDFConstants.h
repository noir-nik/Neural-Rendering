#ifndef BRDFCONSTANTS_H
#define BRDFCONSTANTS_H

#define MAX_KAN_LAYERS 4

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

struct FastKanLayerBufferOffsets {
	uint32_t rbf_grid_offset;
	uint32_t spline_weight_offset;
	uint32_t base_weight_offset;
	uint32_t base_bias_offset;
};

struct FastKanConstants {
	int                       num_layers;
	int                       pad[3];
	FastKanLayerBufferOffsets offsets[MAX_KAN_LAYERS];
};

struct BRDFConstants {
	float4x4 view_proj;
	Material material;
	Light    light;
	float3   camera_pos;
	int      pad;

	// ENABLE_MLP
	// uint32_t weights_offsets[5];
	// uint32_t bias_offsets[5];
	// int      pad2[2];

	FastKanConstants fast_kan;
};

// constexpr auto s = sizeof(BRDFConstants);

#if defined(__cplusplus) || defined(__SLANG__)
enum class BrdfFunctionType : int {
	eClassic,
	eCoopVec,
	eWeightsInBuffer,
	eWeightsInBufferF16,
	eWeightsInHeader,
	eKan,
	eCount
};
#endif

#endif // BRDFCONSTANTS_H
