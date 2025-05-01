#ifndef BRDFCONSTANTS_H
#define BRDFCONSTANTS_H

#ifdef __cplusplus
struct BRDFConstants {
	float4x4 view_proj/* [16] */;

	float3 camera_pos/* [3] */;
	int   pad[1];

	unsigned weights_offsets[5];
	unsigned bias_offsets[5];
	int      pad2[2];
};
#endif

#ifdef __SLANG__
struct BRDFConstants {
	float4x4 view_proj;
	
	float3   camera_pos;
	int      pad[1];

	uint32_t weights_offsets[5];
	uint32_t bias_offsets[5];
	int      pad2[2];
};
#endif

#if defined(__cplusplus) || defined(__SLANG__)
enum class BrdfFunctionType : int {
	eDefault = 0,
	eCoopVec = 1,
	eCount
};
#endif

#endif // BRDFCONSTANTS_H
