#include "BRDFBindings.h"
#include "BRDFConfig.h"
#include "BRDFConstants.h"

import Pbr;

import Utils;
import BRDFSINEKAN;

// clang-format off
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(BINDING_STORAGE_BUFFER, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;
[[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;
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




#ifndef LAYERS
#define LAYERS 3
#endif
#ifndef NUM_SAMPLES
#define NUM_SAMPLES 32

#endif

	uint pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

uint rng_init(uint2 pixel, uint frame) {
	return pcg_hash(pixel.x + pcg_hash(pixel.y + pcg_hash(frame)));
}

uint rng_next(inout uint rng_state) {
	rng_state = pcg_hash(rng_state);
	return rng_state;
}

float rng_float(inout uint rng_state) {
	return float(rng_next(rng_state)) / 4294967296.0;
}

float rng_float_range(inout uint rng_state, float min_val, float max_val) {
	return min_val + rng_float(rng_state) * (max_val - min_val);
}

float3 EvaluateBlinnPhong(float3 view_dir, float3 light_dir, float3 normal,
						  float3 diffuse_color, float3 specular_color, float shininess) {
	float3 half_vec = normalize(view_dir + light_dir);
	float  NdotL    = max(dot(normal, light_dir), 0.0);
	float  NdotH    = max(dot(normal, half_vec), 0.0);

	float3 diffuse = diffuse_color / 3.14159265359;

	float  spec_factor   = pow(NdotH, shininess);
	float  normalization = (shininess + 2.0) / (2.0 * 3.14159265359);
	float3 specular      = specular_color * normalization * spec_factor;

	return (diffuse + specular) * NdotL;
}

float3 TangentToWorld(float3 sample_dir, float3 tangent, float3 bitangent, float3 normal) {
	return sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * normal;
}

float3 WorldToTangent(float3 world_dir, float3 tangent, float3 bitangent, float3 normal) {

	float3 tangent_dir;
	tangent_dir.x = dot(world_dir, tangent);
	tangent_dir.y = dot(world_dir, bitangent);
	tangent_dir.z = dot(world_dir, normal);

	return tangent_dir;
}

float3 Reflect(float3 incident, float3 normal) {
	return incident - 2.0 * dot(incident, normal) * normal;
}

float3 SampleCosineHemisphere(float2 random_val, out float pdf) {

	float phi = 2.0 * 3.14159265359 * random_val.x;
	float r   = sqrt(random_val.y);

	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(max(0.0, 1.0 - random_val.y));

	pdf = z / 3.14159265359;

	return float3(x, y, z);
}

float3 SampleBlinnPhong(float2 random_val, float shininess, float3 view_tbn, out float pdf) {

	float phi      = 2.0 * 3.14159265359 * random_val.x;
	float cosTheta = pow(random_val.y, 1.0 / (shininess + 1.0));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

	float3 half_vec;
	half_vec.x = sinTheta * cos(phi);
	half_vec.y = sinTheta * sin(phi);
	half_vec.z = cosTheta;

	float3 light_dir = reflect(-view_tbn, half_vec);

	if (light_dir.z <= 0.0) {
		pdf = 0.0;
		return float3(0.0, 0.0, 1.0);
	}

	float pdf_half = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cosTheta, shininess);

	float VdotH = max(dot(view_tbn, half_vec), 0.0001);
	pdf         = pdf_half / (4.0 * VdotH);

	return light_dir;
}

float3 SampleBlinnPhongReflection(float2 random_val, float shininess, float3 view_tbn, out float pdf) {

	float3 reflect_dir = float3(-view_tbn.x, -view_tbn.y, view_tbn.z);

	float phi      = 2.0 * 3.14159265359 * random_val.x;
	float cosTheta = pow(random_val.y, 1.0 / (shininess + 1.0));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

	float3 up        = abs(reflect_dir.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
	float3 tangent   = normalize(cross(up, reflect_dir));
	float3 bitangent = cross(reflect_dir, tangent);

	float3 sample_dir;
	sample_dir.x = sinTheta * cos(phi);
	sample_dir.y = sinTheta * sin(phi);
	sample_dir.z = cosTheta;

	float3 light_dir = sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * reflect_dir;

	light_dir = normalize(light_dir);

	if (light_dir.z <= 0.0) {
		pdf = 0.0;
		return float3(0.0, 0.0, 1.0);
	}

	pdf = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cosTheta, shininess);

	return light_dir;
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

	float3 light_tbn;
	float  pdf = 0.1;

	float3 current_sample = float3(0, 0, 0);

	for (uint i = 0; i < NUM_SAMPLES; i++) {
		let rinput2 = pixel_coord;

		float2 random_val = float2(rng_float(rng_state), rng_float(rng_state));

		if (random_val.x < specular_weight) {
			float2 rand        = float2(random_val.x / specular_weight, random_val.y);
			float3 light_blinn = SampleBlinnPhong(rand, shininess, view_tbn, pdf);
			light_tbn          = (light_blinn);

			pdf *= specular_weight;

		} else {

			float2 rand = float2((random_val.x - specular_weight) / (1.0 - specular_weight), random_val.y);
			light_tbn   = SampleCosineHemisphere(rand, pdf);

			pdf *= (1.0 - specular_weight);
		}

		if (light_tbn.z > 0.0 && pdf > 0.00001) {

			float3 light_dir = TangentToWorld(light_tbn, tangent, bitangent, normal);

			float3 half_vec = float3(0.f, 0.f, 0.f);
			float3 diff_vec = float3(0.f, 0.f, 0.f);
			RusinkiewiczTransform(view_tbn, light_tbn, &half_vec, &diff_vec);

			float inputf[6] = { half_vec.x, half_vec.y, half_vec.z,
								diff_vec.x, diff_vec.y, diff_vec.z };
			float outputf[3];

			BRDFSINEKAN3(inputf, outputf);
			float3 brdf = float3(outputf[0], outputf[1], outputf[2]);

			float4 env_sample   = gEnvMap.Sample(light_dir) * 15.;
			float3 env_radiance = env_sample.rgb;

			float NdotL = max(dot(normal, light_dir), 0.0);

			float3 eval = (brdf * env_radiance * NdotL) / pdf;
			eval        = min(eval, float3(10.0, 10.0, 10.0));
			current_sample += eval;
		}
	}

	float  new_sample_count  = prev_sample_count + NUM_SAMPLES;
	float3 accumulated_color = (prev_color * prev_sample_count + current_sample) / new_sample_count;

	gAccumBuffer[pixel_coord] = float4(accumulated_color, new_sample_count);

	return (float4(accumulated_color, 1.0));
}


