#ifndef SDFCONSTANTS_H
#define SDFCONSTANTS_H

#ifdef __cplusplus
struct SDFConstants {
	float resolution[2];
	float mouse[2];
	unsigned weights_offsets[4];
	unsigned bias_offsets[4];
};
#endif

#ifdef GL_core_profile
struct SDFConstants {
	vec2 resolution;
	vec2 mouse;
	uint32_t weights_offsets[4];
	uint32_t bias_offsets[4];
};
#endif

#ifdef __SLANG__
struct SDFConstants {
	float2 resolution;
	float2 mouse;
	uint32_t weights_offsets[4];
	uint32_t bias_offsets[4];
};
#endif

#endif // SDFCONSTANTS_H
