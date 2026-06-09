#include "BRDFBindings.h"
#include "BRDFConstants.h"

import Utils;

// clang-format off
struct VSOutput {
    [[vk::location(0)]] float4 position : SV_POSITION;
    [[vk::location(1)]] float3 view_ray : TEXCOORD0;
};

[[vk::push_constant]] BRDFConstants ctx;
[[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;
// clang-format on

static const float3 tri[3] = {
	float3(-1.0, -1.0, 1.0),
	float3(3.0, -1.0, 1.0),
	float3(-1.0, 3.0, 1.0)
};

[[shader("vertex")]]
VSOutput vs_main(uint vertexId: SV_VertexID) {
	VSOutput output;

	float2 uv = tri[vertexId].xy;

	float2 pos = uv * 2.0 - 1.0;

	// Position in NDC (covers entire screen, at far plane)
	output.position = float4(pos, 1.0, 1.0); // z=1.0 for far plane

	// Reconstruct view ray for environment map sampling
	// Transform NDC position to world space direction
	float4 world_pos = mul((ctx.view_proj), float4(pos, 1.0, 1.0));
	output.view_ray  = world_pos.xyz / world_pos.w;

	return output;
}

float2 SampleEquirectangular(float3 dir) {
	const float PI = 3.1415926535897932384626433832795;
	// Horizontal angle (longitude) and vertical angle (latitude)
	float phi   = atan2(dir.z, dir.x);
	float theta = acos(dir.y);

	// Map angles to [0.0, 1.0] range
	// phi outputs [-PI, PI] -> map to [0, 1]
	// theta outputs [0, PI] -> map to [0, 1]
	float2 uv = float2(phi / (2.0 * PI) + 0.5, theta / PI);
	return uv;
}

[[shader("fragment")]]
float4 ps_main(VSOutput input) : SV_TARGET {
	// Normalize the view ray to get direction
	float3 direction = normalize(input.view_ray);

	// Sample the cube map
	float4 color = gEnvMap.Sample(direction);

	// color.rgb *= 0.05;

	// float3 out = ToneMapR1Gamma(color.rgb * 0.0005);

	float exposure = 0.05;

	float3 out = color.rgb;
	// float3 out = float3(1.0) - exp(-color.rgb * exposure);

	// float3 out = color.rgb / (color.rgb + float3(1.0));

	// out = pow(out, 1);
	// out = GammaF3(out, 1 / 2.2);
	// out = float3(1.0) - exp(-color.rgb * exposure);
	// out = ToneMapR(out);
	// out = GammaF3(out, 2.2);

	return float4(out.rgb, 1.0);
}


