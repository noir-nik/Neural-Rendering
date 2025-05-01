#include "BRDFConfig.h"
#include "BRDFConstants.h"


[[vk::push_constant]]
BRDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

// clang-format off
struct VSInput {
	[[vk::location(0)]] float3 pos : POSITION0;
};
// clang-format on

[[shader("vertex")]]
void vs_main(VSInput in_vertex, out float4 out_clip_pos: SV_Position, out float3 frag_pos) {
	out_clip_pos = mul(ctx.view_proj, float4(in_vertex.pos, 1.0f));
	frag_pos = in_vertex.pos;
}

[[shader("fragment")]]
void ps_main(float2 frag_coord, out float4 frag_color: SV_Target0) {
	frag_color = float4(0.2f, 0.8f, 0.0f, 1.0f);
}

