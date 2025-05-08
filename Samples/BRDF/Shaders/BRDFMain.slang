#include "BRDFConfig.h"
#include "BRDFConstants.h"

[[vk::push_constant]]
BRDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

// clang-format off
struct VSInput {
	[[vk::location(0)]] float4 pos : POSITION0;
	[[vk::location(1)]] float4 normal : COLOR0;
};

struct VSOutput {
	[[vk::location(0)]] float4 clip_pos : SV_POSITION;
	[[vk::location(1)]] float3 normal : NORMAL0;
	[[vk::location(2)]] float3 frag_pos : POSITION;
}
// clang-format on

[[shader("vertex")]]
func vs_main(VSInput in_vertex)->VSOutput {
	VSOutput output;
	output.clip_pos = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	output.frag_pos = in_vertex.pos.xyz;
	output.normal   = in_vertex.normal.xyz;

	return output;
}

[[shader("fragment")]]
func ps_main(VSOutput input) -> float4{

	float4 color = float4(input.normal.xyz, 1.0f);
	return color;
}


