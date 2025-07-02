#include "SDFConfig.h"
#include "SDFConstants.h"

import SDFVec4;
import SDFCoopVec;
import SDFWeightsInBuffer;
import SDFWeightsInHeader;

// clang-format off1

[[vk::binding(0, 0)]]
StructuredBuffer<COOPVEC_TYPE> gSDFWeights;

[[vk::push_constant]]
SDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

func RotateAxisAngle(float3 vector, float3 axis, float angle)->float3 {
	return lerp(dot(vector, axis) * axis, vector, cos(angle)) + sin(angle) * cross(axis, vector);
}

func Scene(float3 p)->float {
	vector<COOPVEC_TYPE, 3> pos = { p.x, p.y, p.z };
	if (shader_type == SdfFunctionType::eCoopVec) {
		return SDFCoopVec<COOPVEC_TYPE>(pos, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == SdfFunctionType::eWeightsInHeader) {
		return SDFWeightsInHeader<COOPVEC_TYPE>(pos);
	} else if (shader_type == SdfFunctionType::eWeightsInBuffer) {
		return SDFWeightsInBuffer<COOPVEC_TYPE>(pos, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == SdfFunctionType::eVec4) {
		return SDFVec4<COOPVEC_TYPE>(pos);
	} else {
		return SDFCoopVec<COOPVEC_TYPE>(pos, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	}
	return 0;
}

func GetNormal(float3 p)->float3 {
	const COOPVEC_TYPE kDiffEpsilon = COOPVEC_TYPE(0.1);
	return normalize(Scene(p) - float3(Scene(p - float3(kDiffEpsilon, 0, 0)), Scene(p - float3(0, kDiffEpsilon, 0)), Scene(p - float3(0, 0, kDiffEpsilon))));
}

[shader("fragment")]
void main(float2 fragCoord, out float4 fragColor: SV_Target0) {
	const float PI         = 3.141592653589793238462643383279;
	const float DEG_TO_RAD = PI / 180.0f;
	const float kEpsilon   = 1e-4;
	const float kMaxDist   = 1000.0f;

	float2 uv = (fragCoord - 0.5); // center of the pixel

	float aspect_ratio = ctx.resolution_x / ctx.resolution_y;

	float2 fov = tan(ctx.fov * DEG_TO_RAD);

	if (aspect_ratio > 1) {
		uv.y /= aspect_ratio;
	} else {
		uv.x *= aspect_ratio;
	}

	let factor = uv * fov;

	float3 ray_dir =
		ctx.camera_forward
		+ factor.x * ctx.camera_right
		+ factor.y * ctx.camera_up;

	ray_dir = normalize(ray_dir);

	float3 init = ctx.camera_pos;

	float3 p   = init;
	bool   hit = false;
	for (int i = 0; i < 75 && !hit; ++i) {
		float dist = Scene(p);

		// hit = dist * dist < kEpsilon;
		hit = dist < kEpsilon;
		p += dist * ray_dir;
		if (distance(p, init) > kMaxDist)
			break;
	}

	float3 n = GetNormal(p);
	// float3 n = D_SDFScalar<COOPVEC_TYPE>({p.x, p.y, p.z});
	float3 r    = reflect(ray_dir, n);
	float3 nz   = p - RotateAxisAngle(p, float3(1, 1, 1), 2.) + RotateAxisAngle(p, float3(1, 1, 1), 4.);
	float  spec = length(sin(r * 3.5 + sin(nz * 120.) * .15) * .4 + .6) / sqrt(3.);
	// float spec = length(sin(r * 3.5) * .4 + .6) / sqrt(3.);

	spec *= smoothstep(-.3, .2, Scene(p + r * .2));
	float3 col    = float3(.1, .1, .12) * spec + pow(spec, 8.);
	float  bgdot  = length(sin(ray_dir * 8.) * .4 + .6) / 2.;
	float3 bg     = float3(.1, .1, .11) * bgdot + pow(bgdot, 10.);
	fragColor.xyz = hit ? col : bg;
	fragColor     = smoothstep(-.02, 1.05, sqrt(fragColor)) * (1. - dot(uv, uv) * .5);
}


