#include "BRDFBindings.h"
#include "BRDFConfig.h"
#include "BRDFConstants.h"

import Pbr;
// import NeuroBRDF;
import Utils; // rusinkiewicz_transform
import BRDFSINEKAN;

// clang-format off
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;
// Cubemap texture
[[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;
// Accumulation buffer (previous frame)
// [[vk::binding(BINDING_STORAGE_IMAGE, 0)]] Texture2D<float4> gAccumBuffer;
[[vk::binding(BINDING_STORAGE_IMAGE, 0)]] RWTexture2D<float4> gAccumBuffer;	

[[vk::push_constant]] BRDFConstants ctx;

[SpecializationConstant] const int shader_type = 0;
[SpecializationConstant] const int func_id = 0;

struct VSInput {
    [[vk::location(0)]] float4 pos : POSITION0;
    [[vk::location(1)]] float4 normal : COLOR0;
};

struct VSOutput {
    [[vk::location(0)]] float4 clip_pos : SV_POSITION;
    [[vk::location(1)]] float3 normal : NORMAL0; 
    [[vk::location(3)]] float3 world_pos : POSITION1;
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

#define NUM_SAMPLES 8

[[shader("fragment")]]
float4 ps_main(VSOutput ps_input) : SV_TARGET {
	float3 normal   = normalize(ps_input.normal);
	float3 view_dir = normalize(ctx.camera_pos - ps_input.world_pos);

	// Build tangent space
	float3   helper    = abs(normal.x) > 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3   tangent   = normalize(cross(helper, normal));
	float3   bitangent = normalize(cross(normal, tangent));
	float3x3 tbn       = float3x3(tangent, bitangent, normal);
	float3x3 inv_tbn   = transpose(tbn);

	float3 view_tbn = normalize(mul(inv_tbn, view_dir));

	uint2 pixel_coord = uint2(ps_input.clip_pos.xy);

	float3 prev_color;
	float  prev_sample_count;

	bool camera_moved = ctx.reset_accumulation;

	if (camera_moved) {
		prev_color        = float3(0, 0, 0);
		prev_sample_count = 0.0;
	} else {
		float4 prev_accum = gAccumBuffer[pixel_coord];
		prev_color        = prev_accum.rgb;
		prev_sample_count = prev_accum.a;
	}

	uint frame_number = ctx.frame_count;

	float shininess       = 1.0;
	float specular_weight = 0.99;

	float3 light_tbn;
	float  pdf = 0.;

	float3 current_sample = float3(0, 0, 0);

	for (uint i = 0; i < NUM_SAMPLES; i++) {
		let rinput2 = pixel_coord;
		// uint2 rinput2 = uint2(ps_input.world_pos.x, ps_input.world_pos.y);
		float2 random_val = random2D(rinput2, frame_number * NUM_SAMPLES + i);

		if (random_val.x < specular_weight) {
			float2 rand = float2(random_val.x / specular_weight, random_val.y);
			light_tbn   = SampleBlinnPhong(rand, shininess, view_tbn, pdf);

			pdf *= specular_weight;
		} else {

			float2 rand = float2((random_val.x - specular_weight) / (1.0 - specular_weight), random_val.y);
			light_tbn   = SampleCosineHemisphere(rand, pdf);

			pdf *= (1.0 - specular_weight);
		}


		if (light_tbn.z > 0.0 && pdf > 0.01) {
			// if (1) {
			float3 light_dir = TangentToWorld(light_tbn, normal, tangent, bitangent);

			float3 half_vec = float3(0.f, 0.f, 0.f);
			float3 diff_vec = float3(0.f, 0.f, 0.f);
			RusinkiewiczTransform(view_tbn, light_tbn, &half_vec, &diff_vec);

			float inputf[6] = { half_vec.x, half_vec.y, half_vec.z,
								diff_vec.x, diff_vec.y, diff_vec.z };
			float outputf[3];

			BRDFSINEKAN3(inputf, outputf);
			float3 brdf = float3(outputf[0], outputf[1], outputf[2]);

			float4 env_sample   = gEnvMap.Sample(light_dir) * 3;
			float3 env_radiance = env_sample.rgb;

			float NdotL = max(dot(normal, light_dir), 0.0);

			current_sample += (brdf * env_radiance * NdotL) / pdf;
		}
	}

	float  new_sample_count  = prev_sample_count + NUM_SAMPLES;
	float3 accumulated_color = (prev_color * prev_sample_count + current_sample) / new_sample_count;

	gAccumBuffer[pixel_coord] = float4(accumulated_color, new_sample_count);

	return float4(accumulated_color, 1.0);
}


