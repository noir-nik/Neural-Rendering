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

	uint32_t weights_offsets[4];
	uint32_t bias_offsets[4];
};

enum class SdfFunctionType : int {
	eCoopVec         = 0,
	eWeightsInHeader = 1,
	eWeightsInBuffer = 2,
	eVec4            = 3,

	eCount
};

#endif // SDFCONSTANTS_H
