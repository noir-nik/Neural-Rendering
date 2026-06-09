#include "BRDFBindings.h"
#include "BRDFConstants.h"

import Utils;

// clang-format off
struct VSOutput {
    [[vk::location(0)]] float4 position : SV_POSITION;
    [[vk::location(1)]] float3 view_ray : TEXCOORD0;
};

[[vk::push_constant]] BRDFConstants ctx;
// [[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;
[[vk::binding(BINDING_TEXTURE, 0)]] Sampler2D gEnvMap;
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


[[shader("fragment")]]
float4 ps_main(VSOutput input) : SV_TARGET {
	// Normalize the view ray to get direction
	float3 direction = normalize(input.view_ray);

	float2 uv = SampleEquirectangular(direction);
	// Sample the cube map
	float4 color = gEnvMap.Sample(uv);

	// color.rgb *= 0.05;

	// float3 out = ToneMapR1Gamma(color.rgb * 0.0005);

	// float exposure = 0.05;

	float3 out = color.rgb;
	// float3 out = float3(1.0) - exp(-color.rgb * exposure);

	// float3 out = color.rgb / (color.rgb + float3(1.0));

	// out = pow(out, 1);
	// out = GammaF3(out, 1 / 2.2);
	// out = float3(1.0) - exp(-color.rgb * exposure);
	
	out = ToneMapR1Gamma(out);
	// out = GammaF3(out, 2.2);

	return float4(out.rgb, 1.0);
}


