#ifndef SDFCONSTANTS_H
#define SDFCONSTANTS_H

#if defined(__cplusplus)
using std::uint32_t;
#endif

struct SDFConstants {
	float3 camera_pos;
	float  resolution_x;
	float3 camera_forward;
	float  resolution_y;
	float3 camera_up;
	float  fov;
	float3 camera_right;
	int    pad;

	uint32_t weights_offsets[6];
	int      pad3[3];
	uint32_t bias_offsets[6];
	int      pad4[3];
};

enum class SdfFunctionType : int {
	eCoopVec,
	eWeightsInBuffer,
	eWeightsInBufferF16,
	eWeightsInHeader,
	eVec4,

	eCount
};

#endif // SDFCONSTANTS_H
