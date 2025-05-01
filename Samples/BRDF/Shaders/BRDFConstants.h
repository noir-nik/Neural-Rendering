#ifndef BRDFCONSTANTS_H
#define BRDFCONSTANTS_H

#ifdef __cplusplus
struct BRDFConstants {
	float resolution[2];
	float mouse[2];
	unsigned weights_offsets[4];
	unsigned bias_offsets[4];
};
#endif

#ifdef __SLANG__
struct BRDFConstants {
	float2 resolution;
	float2 mouse;
	uint32_t weights_offsets[4];
	uint32_t bias_offsets[4];
};
#endif

#if defined (__cplusplus) || defined (__SLANG__)
enum class BrdfFunctionType : int {
	eDefault = 0,
	eCoopVec = 1,
	eCount
};
#endif

#endif // BRDFCONSTANTS_H
