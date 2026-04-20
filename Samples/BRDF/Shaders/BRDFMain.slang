#include "BRDFBindings.h"
#include "BRDFConfig.h"

import BRDFConstants;
import Pbr;

import Utils;

#if !defined(CHEBYKAN) && !defined(FASTKAN) && !defined(RELUKAN) && !defined(SINEKAN) && !defined(NBRDF)
#define FASTKAN
#endif

#ifdef CHEBYKAN
#define INC "CHEBYKAN_2_6_6_630_651.slang"
#include "BRDFCHEBYKAN.slang"
#define FF BRDFCHEBYKAN3
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifdef FASTKAN
import BRDFFASTKAN;
#if LAYERS == 3
#define FF BRDFFASTKAN3
#endif
#if LAYERS == 2
#define FF BRDFFASTKAN2
#endif
#endif

#ifdef RELUKAN
// #define INC "RELUKAN_2_5_3_625_643.slang"
// #define INC "blue-acrylic-relu.slang"
#define INC "green-metallic-paint2.slang"
#include "BRDFRELUKAN.slang"
#define FF BRDFRELUKAN3
#endif

#ifdef SINEKAN
#define INC "SINEKAN_2_6_6_573_681.slang"
#include "BRDFSINEKAN.slang"
#define FF BRDFSINEKAN3
#endif

#ifdef NBRDF

#ifndef INC
#define INC irid_flake_paint2_rgb_NBRDF_2_16_
#endif

import INC;
#define FF INC

// #define FF irid_flake_paint2_rgb_NBRDF_2_48_
// #define FF irid_flake_paint2_rgb_NBRDF_2_64_
// #define FF irid_flake_paint2_rgb_NBRDF_2_96_
// import FF;
#endif

#ifndef NUM_SAMPLES
// #define NUM_SAMPLES 32
// #define NUM_SAMPLES 16
#define NUM_SAMPLES 64
// #define NUM_SAMPLES 1
#endif

#ifndef IS_POINT
#define IS_POINT false
#endif

#ifndef _T
#define _T float
// #define _T half
#endif
typealias Tp = _T;

// import BRDFCoopVec;
// import BRDFClassic;

// clang-format off
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;
[[vk::binding(BINDING_TEXTURE, 0)]]        SamplerCube gEnvMap;
[[vk::binding(BINDING_STORAGE_IMAGE, 0)]]  RWTexture2D<float4> gAccumBuffer;	

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

struct PSOutput {
	float4 color : SV_TARGET0;
}
// clang-format on

[[shader("vertex")]]
VSOutput vs_main(VSInput in_vertex) {
	VSOutput output;
	output.clip_pos = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));

	output.world_pos = in_vertex.pos.xyz;
	output.normal    = normalize(in_vertex.normal.xyz);

	return output;
}

[[shader("fragment")]]
float4 ps_main(VSOutput ps_input) : SV_TARGET {
	float3 normal = normalize(ps_input.normal);

	float3 view_dir = normalize(ctx.camera_pos - ps_input.world_pos);

	float3 helper = float3(0, 1, 0);

	float3 tangent   = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));

	float3x3 tbn     = float3x3(tangent, bitangent, normal);
	float3x3 inv_tbn = transpose(tbn);

	float3 view_tbn = normalize(WorldToTangent(view_dir, tangent, bitangent, normal));

	float3 normal_tbn = normalize(WorldToTangent(normal, tangent, bitangent, normal));

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

	uint rng_state = rng_init(pixel_coord, ctx.frame_count);

	float shininess       = 32.0;
	float specular_weight = 0.9;

	float3 current_sample = float3(0, 0, 0);

	for (uint i = 0; i < NUM_SAMPLES; i++) {
		let rinput2 = pixel_coord;

		float2 random_val = float2(rng_float(rng_state), rng_float(rng_state));

		float3 light_tbn;
		float  pdf = 1.;
		float  pdf_other;
		float  mis_weight;
		bool   is_specular_sample = random_val.x < specular_weight;

		if (IS_POINT) {
			float3 ldir      = ctx.light.position - ps_input.world_pos;
			float  dist      = length(ldir);
			float3 light_dir = normalize(ldir);
			light_tbn        = WorldToTangent(light_dir, tangent, bitangent, normal);
		} else {
			if (is_specular_sample) {
				float2 rand        = float2(random_val.x / specular_weight, random_val.y);
				float3 light_blinn = SampleBlinnPhong(rand, shininess, view_tbn, pdf);
				light_tbn          = (light_blinn);
				pdf_other          = PDFCosineHemisphere(light_tbn) * (1.0 - specular_weight);

				float pdf_cosine = CosinePDF(light_tbn.z);
				pdf *= specular_weight;

			} else {

				float2 rand = float2((random_val.x - specular_weight) / (1.0 - specular_weight), random_val.y);
				light_tbn   = SampleCosineHemisphere(rand, pdf);

				pdf_other = PDFBlinnPhong(view_tbn, light_tbn, shininess) * specular_weight;
				pdf *= (1.0 - specular_weight);
			}
		}

		if (light_tbn.z > 0.0 && pdf > 0.00001) {
			float3 light_dir = TangentToWorld(light_tbn, tangent, bitangent, normal);

			float3 half_vec = float3(0.f, 0.f, 0.f);
			float3 diff_vec = float3(0.f, 0.f, 0.f);
			RusinkiewiczTransform(view_tbn, light_tbn, &half_vec, &diff_vec);

			// typealias Tp = half;
			typealias Tp = float;

			Tp inputf[6] = {
				half_vec.x, half_vec.y, half_vec.z,
				diff_vec.x, diff_vec.y, diff_vec.z
			};
			Tp outputf[3];

			// BRDFSINEKAN3(inputf, outputf);
			if (BrdfFunctionType(shader_type) == BrdfFunctionType::eClassic) {
			}

			uint wo3[3] = { ctx.weights_offsets[0], ctx.weights_offsets[1], ctx.weights_offsets[2] };
			uint bo3[3] = { ctx.bias_offsets[0], ctx.bias_offsets[1], ctx.bias_offsets[2] };

			if (shader_type == BrdfFunctionType::eCoopVec) {
				// FF(inputf, outputf, gWeightsCoopVec, wo3, bo3);
			} else {
			}
			FF(inputf, outputf);

			float3 brdf = float3(outputf[0], outputf[1], outputf[2]);

			float NdotL = max(dot(normal, light_dir), 0.0);

			float3 eval;
			if (IS_POINT) {
				// eval = (brdf);
				eval = (brdf * NdotL);
			} else {
				float4 env_sample   = gEnvMap.Sample(light_dir) * 5.;
				float3 env_radiance = env_sample.rgb;
				mis_weight          = MISWeight(pdf, pdf_other, 2.0);
				eval                = (brdf * env_radiance * NdotL * mis_weight) / pdf;
			}
			eval = min(eval, float3(10.0, 10.0, 10.0));
			current_sample += eval;
		}
	}

	float  new_sample_count  = prev_sample_count + NUM_SAMPLES;
	float3 accumulated_color = (prev_color * prev_sample_count + current_sample) / new_sample_count;

	gAccumBuffer[pixel_coord] = float4(accumulated_color, new_sample_count);

	return (float4(accumulated_color, 1.0));
}


