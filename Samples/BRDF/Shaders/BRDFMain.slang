#include "BRDFConfig.h"

import Pbr;
import BRDFScalarBuffer;
import BRDFCoopVec;
import BRDFClassic;
import BRDFWeightsInHeader;

// clang-format off
[[vk::binding(0, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(0, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;

[[vk::push_constant]] BRDFConstants ctx;

[SpecializationConstant] const int shader_type = 0;

struct VSInput {
	[[vk::location(0)]]	float4 pos : POSITION0;
	[[vk::location(1)]]	float4 normal : COLOR0;
};

struct VSOutput {
	[[vk::location(0)]]	float4 clip_pos : SV_POSITION;
	[[vk::location(1)]]	float3 normal : NORMAL0; 
	[[vk::location(3)]]	float3 world_pos : POSITION1;
};
// clang-format on

[[shader("vertex")]]
VSOutput vs_main(VSInput in_vertex) {
	VSOutput output;
	output.clip_pos  = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	output.world_pos = in_vertex.pos.xyz;
	output.normal    = normalize(in_vertex.normal.xyz);
	return output;
}

[[shader("fragment")]]
func ps_main(VSOutput input)->float4 {
	float3 normal = normalize(input.normal);
	float3 view   = normalize(ctx.camera_pos - input.world_pos);

	float3 light_dir = normalize(ctx.light.position - input.world_pos);

	float3 helper    = abs(normal.x) > 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 tangent   = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));

	float3x3 tbn = float3x3(tangent, bitangent, normal);

	float3x3 inv_tbn = (tbn);

	float3 view_tbn  = mul(inv_tbn, view);
	float3 light_tbn = mul(inv_tbn, light_dir);

	view_tbn  = normalize(view_tbn);
	light_tbn = normalize(light_tbn);

	float3 color = float3(0, 0, 0);

	float3 albedo    = ctx.material.base_color.rgb;
	float  metallic  = ctx.material.metallic;
	float  roughness = ctx.material.roughness;

	vector<COOPVEC_TYPE, 3> view_typed  = { view_tbn.x, view_tbn.y, view_tbn.z };
	vector<COOPVEC_TYPE, 3> light_typed = { light_tbn.x, light_tbn.y, light_tbn.z };
	if (shader_type == BrdfFunctionType::eCoopVec) {
		color = BRDFCoopVec<COOPVEC_TYPE>(view_typed, light_typed, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eWeightsInBuffer) {
		color = BRDFWeightsInBuffer<float>(view_tbn, light_tbn, gWeights, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eWeightsInBufferF16) {
		color = BRDFWeightsInBuffer<half>(view_typed, light_typed, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eWeightsInHeader) {
		color = BRDFWeightsInHeader<float>(view_tbn, light_tbn);
		// color = BRDFWeightsInHeader<half>(view_tbn, light_tbn);
	} else {
		color = BRDFClassic(input.world_pos, view, ctx.light, normal, albedo, metallic, roughness);
	}

	// Reinhard tone mapping
	color = color / (color + float3(1.0, 1.0, 1.0));

	// Gamma correction
	color = pow(color, 1.0 / 2.2);

	return float4(color, 1.0);
}


