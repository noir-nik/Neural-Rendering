#include "SDFConfig.h"
#include "SDFConstants.h"

import SDFVec4;
import SDFCoopVec;
import SDFScalarInline;
import SDFScalarBuffer;

// clang-format off

[[vk::binding(0, 0)]] StructuredBuffer<COOPVEC_TYPE> gSDFWeights;

[[vk::push_constant]] SDFConstants ctx;


func RotateAxisAngle(float3 vector, float3 axis, float angle) -> float3 {
	return lerp(dot(vector, axis) * axis, vector, cos(angle)) + sin(angle) * cross(axis, vector);
}

func Scene(float3 p) -> float {
	return SDFCoopVec<COOPVEC_TYPE>({p.x, p.y, p.z}, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	// return SDFScalarBuffer<COOPVEC_TYPE>({p.x, p.y, p.z}, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	// return SDFScalarInline<COOPVEC_TYPE>({p.x, p.y, p.z});
	// return SDFVector<COOPVEC_TYPE>({p.x, p.y, p.z});
}

func GetNormal(float3 p) -> float3 {
	return normalize(Scene(p) - float3(Scene(p - float3(0.1, 0, 0)), Scene(p - float3(0, 0.1, 0)), Scene(p - float3(0, 0, 0.1))));
}

[shader("fragment")]
void main(float2 fragCoord, out float4 fragColor: SV_Target0) {
	float2 uv = (fragCoord - 0.5 );
	if (ctx.resolution.x < ctx.resolution.y) {
		uv.y *= ctx.resolution.y / ctx.resolution.x;
	} else {
		uv.x *= ctx.resolution.x / ctx.resolution.y;
	}
	float2 mouse = (ctx.mouse.xy - 0.5 * ctx.resolution.xy) / ctx.resolution.y;
	
	float3 cam = normalize(float3(1.5, uv));
	float3 init = float3(-3., 0, 0);

	float yrot = 0.5;
	float zrot = 0.5;
	zrot = 4. * mouse.x;
	if (true) {
		yrot += -4.*mouse.y;
		zrot = 4.*mouse.x;
	}
	cam = RotateAxisAngle(cam, float3(0, 1, 0), yrot);
	init = RotateAxisAngle(init, float3(0, 1, 0), yrot);
	cam = RotateAxisAngle(cam, float3(0, 0, 1), zrot);
	init = RotateAxisAngle(init, float3(0, 0, 1), zrot);

	float3 p = init;
	bool hit = false;
	for(int i = 0; i < 75 && !hit; ++i) {
		float dist = Scene(p);
		hit = dist * dist < 1e-4;
		p += dist * cam;
		if(distance(p, init) > 5.)
			break;
	}
	float3 n = GetNormal(p);
	// float3 n = D_SDFScalar<COOPVEC_TYPE>({p.x, p.y, p.z});
	float3 r = reflect(cam, n);
	float3 nz = p - RotateAxisAngle(p, float3(1, 1, 1), 2.) + RotateAxisAngle(p, float3(1, 1, 1), 4.);
	float spec = length(sin(r*3.5+sin(nz*120.)*.15)*.4+.6)/sqrt(3.);
	// float spec = length(sin(r * 3.5) * .4 + .6) / sqrt(3.);

	spec *= smoothstep(-.3, .2, Scene(p + r * .2));
	float3 col = float3(.1, .1, .12) * spec + pow(spec, 8.);
	// float3 col = n;
	float bgdot = length(sin(cam * 8.) * .4 + .6) / 2.;
	float3 bg = float3(.1, .1, .11) * bgdot + pow(bgdot, 10.);
	fragColor.xyz = hit ? col : bg;
	fragColor = smoothstep(-.02, 1.05, sqrt(fragColor)) * (1. - dot(uv, uv) * .5);
}