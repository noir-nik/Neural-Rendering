#include "BRDFConfig.h"

import Pbr;
import BRDFCoopVec;
import BRDFClassic;
import BRDFWeightsInBuffer;
// import BRDFWeightsInHeader;

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
	// float3 color = float3(1, 1, 1);

	float3 albedo    = ctx.material.base_color.rgb;
	float  metallic  = ctx.material.metallic;
	float  roughness = ctx.material.roughness;

	vector<COOPVEC_TYPE, 3> view_typed  = { view_tbn.x, view_tbn.y, view_tbn.z };
	vector<COOPVEC_TYPE, 3> light_typed = { light_tbn.x, light_tbn.y, light_tbn.z };

	// view_typed = abs(view_typed);
	// light_typed = abs(light_typed);

	// typealias Ty = _TY;
	// typealias Ty = float;
	typealias Ty  = half;
	typealias FTy = float;

	Ty input[6] = { view_tbn.x, view_tbn.y, view_tbn.z, light_tbn.x, light_tbn.y, light_tbn.z };
	Ty output[3];

	FTy inputf[6] = { view_tbn.x, view_tbn.y, view_tbn.z, light_tbn.x, light_tbn.y, light_tbn.z };
	FTy outputf[3];

	uint wo4[4] = { ctx.weights_offsets[0], ctx.weights_offsets[1], ctx.weights_offsets[2], ctx.weights_offsets[3] };
	uint bo4[4] = { ctx.bias_offsets[0], ctx.bias_offsets[1], ctx.bias_offsets[2], ctx.bias_offsets[3] };

	if (shader_type == BrdfFunctionType::eCoopVec) {
		// BRDFCoopVec<COOPVEC_TYPE>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		if (func_id == 0) BRDFCoopVec<COOPVEC_TYPE>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		// if (func_id == 0) BRDFCoopVec_6_8_8_8_6_3_275<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 1) BRDFCoopVec_6_16_16_16_6_3_779<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 2) BRDFCoopVec_6_32_32_32_6_3_2555<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 3) BRDFCoopVec_6_64_48_6_3_3883<Ty>(input, output, gWeightsCoopVec, wo4, bo4);
		else if (func_id == 4) BRDFCoopVec_6_48_48_48_6_3_5355<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 5) BRDFCoopVec_6_64_64_64_6_3_9179<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		// else if (func_id == 5) BRDFCoopVec<COOPVEC_TYPE>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else return float4(1, 0, 1, 1.0);
	} else if (shader_type == BrdfFunctionType::eWeightsInBuffer) {
		// BRDFWeightsInBuffer<float>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);

		if (func_id == 0) BRDFWeightsInBuffer_6_8_8_8_6_3_275<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 1) BRDFWeightsInBuffer_6_16_16_16_6_3_779<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 2) BRDFWeightsInBuffer_6_32_32_32_6_3_2555<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 3) BRDFWeightsInBuffer_6_64_48_6_3_3883<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (func_id == 4) BRDFWeightsInBuffer_6_48_48_48_6_3_5355<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		// else if (func_id == 5) BRDFWeightsInBuffer_6_64_64_64_6_3_9179<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 5) BRDFWeightsInBuffer<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		// else if (func_id == 6) BRDFWeightsInBuffer<float>(input, output, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		else return float4(1, 0, 1, 1.0);
	} else if (shader_type == BrdfFunctionType::eWeightsInBufferF16) {
		// BRDFWeightsInBuffer<half>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		if (func_id == 0) BRDFWeightsInBuffer_6_8_8_8_6_3_275<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 1) BRDFWeightsInBuffer_6_16_16_16_6_3_779<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 2) BRDFWeightsInBuffer_6_32_32_32_6_3_2555<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 3) BRDFWeightsInBuffer_6_64_48_6_3_3883<Ty>(input, output, gWeightsCoopVec, wo4, bo4);
		else if (func_id == 4) BRDFWeightsInBuffer_6_48_48_48_6_3_5355<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else if (func_id == 5) BRDFWeightsInBuffer_6_64_64_64_6_3_9179<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		// else if (func_id == 5) BRDFWeightsInBuffer<Ty>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		// else if (func_id == 6) BRDFWeightsInBuffer<float>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		else return float4(1, 0, 1, 1.0);
	} else if (shader_type == BrdfFunctionType::eWeightsInHeader) {
		if (func_id == 0) BRDFWeightsInHeader_6_8_8_8_6_3_275<Ty>(input, output);
		else if (func_id == 1) BRDFWeightsInHeader_6_16_16_16_6_3_779<Ty>(input, output);
		else if (func_id == 2) BRDFWeightsInHeader_6_32_32_32_6_3_2555<Ty>(input, output);
		else if (func_id == 3) BRDFWeightsInHeader_6_64_48_6_3_3883<Ty>(input, output);
		else if (func_id == 4) BRDFWeightsInHeader_6_48_48_48_6_3_5355<Ty>(input, output);
		else if (func_id == 5) BRDFWeightsInHeader_6_64_64_64_6_3_9179<Ty>(input, output);
		// else if (func_id == 5) BRDFWeightsInHeader<Ty>(input, output);
		// else if (func_id == 6) BRDFWeightsInHeader<float>(input, output);
		else return float4(1, 0, 1, 1.0);

	} else {
		FTy inputc[6] = { view.x, view.y, view.z, light_dir.x, light_dir.y, light_dir.z };
		BRDFClassic(inputc, outputf, ctx.light, vs_input.world_pos, normal, albedo, metallic, roughness);
	}

	if (shader_type == BrdfFunctionType::eWeightsInBuffer) {
		color = float3(outputf[0], outputf[1], outputf[2]);
	} else {
		color = float3(output[0], output[1], output[2]);
	}

	// Reinhard tone mapping
	color = color / (color + float3(1.0, 1.0, 1.0));

	// Gamma correction
	color = pow(color, 1.0 / 2.2);

	// color = float3(color.x, color.y, color.z);
	// color = float3(color.x, color.z, color.y);
	// color = float3(color.y, color.x, color.z);
	// color = float3(color.y, color.z, color.x);
	// color = float3(color.z, color.y, color.x);
	// color = float3(color.z, color.x, color.y);

	// color = float3(color.z, color.y, color.x);

	return float4(color, 1.0);
}


