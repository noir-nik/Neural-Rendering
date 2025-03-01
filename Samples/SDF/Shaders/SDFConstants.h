#ifndef SDFCONSTANTS_H
#define SDFCONSTANTS_H

#ifdef __cplusplus 
struct SDFConstants {
	float resolution[2];
	float mouse[2];
};
#endif

#ifdef GL_core_profile 
struct SDFConstants {
	vec2 resolution;
	vec2 mouse;
};
#endif

#ifdef __SLANG__ 
struct SDFConstants {
	float2 resolution;
	float2 mouse;
};
#endif

#endif