#ifndef SHADER_TYPES_H
#define SHADER_TYPES_H

struct GPUCamera {
	float4x4 projection_view_inv;
	float3   position;
	int      pad[1];
};

#endif // SHADER_TYPES_H
