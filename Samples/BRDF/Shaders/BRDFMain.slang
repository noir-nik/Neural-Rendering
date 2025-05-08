#include "BRDFConfig.h"
#include "BRDFConstants.h"


[[vk::push_constant]]
BRDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

// clang-format off
struct VSInput {
	[[vk::location(0)]] float4 pos : POSITION0;
};
// clang-format on

[[shader("vertex")]]
void vs_main(VSInput in_vertex, out float4 out_clip_pos: SV_Position, out float3 frag_pos) {
	// out_clip_pos = mul(float4(in_vertex.pos.xyz * 0.5, 1.0f), mat);
	out_clip_pos = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	// out_clip_pos = mul(in_vertex.pos, ctx.view_proj);
	frag_pos = in_vertex.pos.xyz;
}

[[shader("fragment")]]
void ps_main(in float3 frag_pos, out float4 frag_color: SV_Target0) {
	// frag_color = float4(0.2f, 0.8f, 0.0f, 1.0f);
	frag_color = float4(frag_pos, 1.0f);
}

