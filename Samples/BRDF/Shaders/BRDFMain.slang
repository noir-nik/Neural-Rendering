#include "BRDFConfig.h"

import Pbr;
import BRDFCoopVec;
import BRDFClassic;
import BRDFWeightsInBuffer;
import BRDFWeightsInHeader;

// #define _TY half
// #define _TY int

import BRDFWeightsInHeader_6_8_8_8_6_3_275;
import BRDFWeightsInHeader_6_16_16_16_6_3_779;
import BRDFWeightsInHeader_6_32_32_32_6_3_2555;
import BRDFWeightsInHeader_6_64_48_6_3_3883;
import BRDFWeightsInHeader_6_48_48_48_6_3_5355;
import BRDFWeightsInHeader_6_64_64_64_6_3_9179;

// clang-format off
[[vk::binding(0, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(0, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;

[[vk::push_constant]] BRDFConstants ctx;

[SpecializationConstant] const int shader_type = 0;
[SpecializationConstant] const int func_id = 0;

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
func ps_main(VSOutput vs_input)->float4 {
	float3 normal = normalize(vs_input.normal);
	float3 view   = normalize(ctx.camera_pos - vs_input.world_pos);

	float3 light_dir = normalize(ctx.light.position - vs_input.world_pos);

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

	// typealias Ty = _TY;
	// typealias Ty = float;
	typealias Ty       = half;
	Ty        input[6] = { view_tbn.x, view_tbn.y, view_tbn.z, light_tbn.x, light_tbn.y, light_tbn.z };
	Ty        output[3];

	if (shader_type == BrdfFunctionType::eCoopVec) {
		color = BRDFCoopVec<COOPVEC_TYPE>(view_typed, light_typed, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eWeightsInBuffer) {
		typealias Ty = float;

		Ty input[6] = { view_tbn.x, view_tbn.y, view_tbn.z, light_tbn.x, light_tbn.y, light_tbn.z };
		Ty output[3];
		BRDFWeightsInBuffer<float>(input, output, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		color = float3(output[0], output[1], output[2]);
	} else if (shader_type == BrdfFunctionType::eWeightsInBufferF16) {
		BRDFWeightsInBuffer<half>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		color = float3(output[0], output[1], output[2]);
	} else if (shader_type == BrdfFunctionType::eWeightsInHeader) {

		// BRDFWeightsInHeader_6_8_8_8_6_3_275<Ty>(input, output);
		// BRDFWeightsInHeader_6_16_16_16_6_3_779<Ty>(input, output);
		// BRDFWeightsInHeader_6_32_32_32_6_3_2555<Ty>(input, output);
		// BRDFWeightsInHeader_6_64_48_6_3_3883<Ty>(input, output);
		// BRDFWeightsInHeader_6_48_48_48_6_3_5355<Ty>(input, output);
		// BRDFWeightsInHeader_6_64_64_64_6_3_9179<Ty>(input, output);
		// BRDFWeightsInHeader<float>(input, output);

		if (func_id == 0) BRDFWeightsInHeader_6_8_8_8_6_3_275<Ty>(input, output);
		else if (func_id == 1) BRDFWeightsInHeader_6_16_16_16_6_3_779<Ty>(input, output);
		else if (func_id == 2) BRDFWeightsInHeader_6_32_32_32_6_3_2555<Ty>(input, output);
		else if (func_id == 3) BRDFWeightsInHeader_6_64_48_6_3_3883<Ty>(input, output);
		else if (func_id == 4) BRDFWeightsInHeader_6_48_48_48_6_3_5355<Ty>(input, output);
		else if (func_id == 5) BRDFWeightsInHeader_6_64_64_64_6_3_9179<Ty>(input, output);
		// else if (func_id == 5) BRDFWeightsInHeader<Ty>(input, output);
		// else if (func_id == 6) BRDFWeightsInHeader<float>(input, output);
		else return float4(1, 0, 1, 1.0);

		color = float3(output[0], output[1], output[2]);
		// color = BRDFWeightsInHeader<half>(view_tbn, light_tbn);
	} else {
		color = BRDFClassic(vs_input.world_pos, view, ctx.light, normal, albedo, metallic, roughness);
	}

	// Reinhard tone mapping
	color = color / (color + float3(1.0, 1.0, 1.0));

	// Gamma correction
	color = pow(color, 1.0 / 2.2);

	return float4(color, 1.0);
}


