#include "BRDFConfig.h"

import Pbr;
import BRDFCoopVec;
import BRDFClassic;
import BRDFWeightsInBuffer;
import BRDFWeightsInHeader;

// import NBRDF_6_8_8_3_155_155;
// import NBRDF_6_16_16_3_435_435;
// import NBRDF_6_21_21_3_675_675;
// import NBRDF_6_32_32_3_1379_1379;
// import NBRDF_6_64_64_3_4803_4803;

// import BRDFFASTKAN;
// import BRDFSINEKAN;
import BRDFCHEBYKAN;
// import BRDFRELUKAN;

import Utils; // rusinkiewicz_transform

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

#ifndef LAYERS
#define LAYERS 3
#endif

[[shader("fragment")]]
func ps_main(VSOutput vs_input)->float4 {
	float3 normal   = normalize(vs_input.normal);
	float3 view_dir = normalize(ctx.camera_pos - vs_input.world_pos);

	float3 ldir      = ctx.light.position - vs_input.world_pos;
	float  dist      = length(ldir);
	float3 light_dir = normalize(ldir);

	float3 helper    = abs(normal.x) > 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 tangent   = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));

	float3x3 tbn = float3x3(tangent, bitangent, normal);

	float3x3 inv_tbn = (tbn);

	float3 view_tbn  = mul(inv_tbn, view_dir);
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
		BRDFCoopVec<COOPVEC_TYPE>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);

	} else if (shader_type == BrdfFunctionType::eWeightsInBuffer) {
		BRDFWeightsInBuffer<float>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);

	} else if (shader_type == BrdfFunctionType::eWeightsInBufferF16) {
		BRDFWeightsInBuffer<half>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eWeightsInHeader) {
		// BRDFWeightsInHeader<float>(inputf, outputf);
		// BRDFCoopVec<COOPVEC_TYPE>(input, output, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
		output = { 0.5h, 0.5h, 0.5h };
	} else if (shader_type == BrdfFunctionType::eKan) {
		float3x3 tbn_inv = transpose(tbn);
		float3   wi      = mul(tbn_inv, light_dir);
		float3   wo      = mul(tbn_inv, view_dir);
		// float3 wi = mul(-light_dir, tbn_inv);
		// float3 wo = mul(view_dir, tbn_inv);
		// wi = normalize(wi);
		// wo = normalize(wo);

		float3 half = float3(0.f, 0.f, 0.f);
		float3 diff = float3(0.f, 0.f, 0.f);

		// rusinkiewicz_transform(wi, wo, inputf);

		RusinkiewiczTransform(wi, wo, &half, &diff);
		inputf = { half.x, half.y, half.z, diff.x, diff.y, diff.z };

#if LAYERS == 2
		// BRDFFASTKAN2(inputf, outputf);
		// BRDFSINEKAN2(inputf, outputf);
		BRDFCHEBYKAN2(inputf, outputf);
		// BRDFRELUKAN2(inputf, outputf);
#elif LAYERS == 3
		// BRDFFASTKAN3(inputf, outputf);
		// BRDFSINEKAN3(inputf, outputf);
		BRDFCHEBYKAN3(inputf, outputf);
		// BRDFRELUKAN3(inputf, outputf);
#endif

		// BRDFMLP(inputf, outputf);

		// NBRDF_6_8_8_3_155_155(inputf, outputf);
		// NBRDF_6_16_16_3_435_435(inputf, outputf);
		// NBRDF_6_21_21_3_675_675(inputf, outputf);
		// NBRDF_6_32_32_3_1379_1379(inputf, outputf);
		// NBRDF_6_64_64_3_4803_4803(inputf, outputf);

		// NBRDF_6_64_64_3_4803_4803(input, output);
		// NBRDF_6_21_21_3_675_675(input, output);

	} else {
		FTy inputc[6] = { view_dir.x, view_dir.y, view_dir.z, light_dir.x, light_dir.y, light_dir.z };
		BRDFClassic(inputc, outputf, ctx.light, vs_input.world_pos, normal, albedo, metallic, roughness);
	}

	if (shader_type == BrdfFunctionType::eWeightsInBuffer
		|| shader_type == BrdfFunctionType::eClassic
		|| shader_type == BrdfFunctionType::eKan) {
		color = float3(outputf[0], outputf[1], outputf[2]);
	} else {
		color = float3(output[0], output[1], output[2]);
	}
	// color = float3(output[0], output[1], output[2]);

	float attenuation = 1.0 / (dist * dist);
	// float attenuation = 1.0;

	// Reinhard tone mapping
	// color = color / (color + float3(1.0, 1.0, 1.0));

	// Gamma correction
	// color = pow(color, 1.0 / 2.2);

	// Final color with Lambert cosine term
	// float3 brdf_color = float3(outputf[0], outputf[1], outputf[2]);
	// float  n_dot_l    = max(0.0, dot(normal, light_dir));
	// color             = brdf_color * n_dot_l;

	// color = float3(color.z, color.y, color.x);
	// color = color * attenuation;
	return float4(color, 1.0);
}


