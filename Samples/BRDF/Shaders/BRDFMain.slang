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

struct PSOutput {
	float4 color : SV_TARGET0;
		// float depth : SV_Depth; // Explicit depth output
}
// clang-format on

[[shader("vertex")]]
VSOutput vs_main(VSInput in_vertex) {
	VSOutput output;
	output.clip_pos = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	// output.clip_pos = mul(float4(in_vertex.pos.xyz, 1.0f),ctx.view_proj);
	output.world_pos = in_vertex.pos.xyz;
	output.normal    = normalize(in_vertex.normal.xyz);

	return output;
}
#ifndef LAYERS
#define LAYERS 3
#endif
#ifndef NUM_SAMPLES
#define NUM_SAMPLES 32 // Samples per frame for accumulation
#endif

// ============================================================================
// High-quality random number generation using PCG hash
// ============================================================================
uint pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

// Initialize RNG state from pixel coordinates and frame count
uint rng_init(uint2 pixel, uint frame) {
	return pcg_hash(pixel.x + pcg_hash(pixel.y + pcg_hash(frame)));
}

// Generate next random number in sequence
uint rng_next(inout uint rng_state) {
	rng_state = pcg_hash(rng_state);
	return rng_state;
}

// Generate random float in [0, 1)
float rng_float(inout uint rng_state) {
	return float(rng_next(rng_state)) / 4294967296.0;
}

// Generate random float in [min, max)
float rng_float_range(inout uint rng_state, float min_val, float max_val) {
	return min_val + rng_float(rng_state) * (max_val - min_val);
}

// ============================================================================
// Blinn-Phong Importance Sampling
// ============================================================================

// Sample a direction according to Blinn-Phong distribution
// Returns the sampled direction in tangent space and its PDF
// void SampleBlinnPhong(float2 random_sample, float shininess,
// 					  out float3 half_vector, out float pdf) {
// 	// Sample according to cos^n(theta) distribution
// 	float phi = 2.0 * 3.14159265359 * random_sample.x;
// 	float cos_theta = pow(random_sample.y, 1.0 / (shininess + 1.0));
// 	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

// 	// Half vector in tangent space (aligned with normal)
// 	half_vector = float3(
// 		cos(phi) * sin_theta,
// 		sin(phi) * sin_theta,
// 		cos_theta
//     );

// 	// PDF for sampling half vector: (n+1)/(2*PI) * cos^n(theta)
// 	pdf = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cos_theta, shininess);
// }

// Evaluate Blinn-Phong BRDF
float3 EvaluateBlinnPhong(float3 view_dir, float3 light_dir, float3 normal,
						  float3 diffuse_color, float3 specular_color, float shininess) {
	float3 half_vec = normalize(view_dir + light_dir);
	float  NdotL    = max(dot(normal, light_dir), 0.0);
	float  NdotH    = max(dot(normal, half_vec), 0.0);

	// Diffuse component
	float3 diffuse = diffuse_color / 3.14159265359;

	// Specular component (Blinn-Phong)
	float  spec_factor   = pow(NdotH, shininess);
	float  normalization = (shininess + 2.0) / (2.0 * 3.14159265359);
	float3 specular      = specular_color * normalization * spec_factor;

	return (diffuse + specular) * NdotL;
}

// Transform hemisphere sample to world space
float3 TangentToWorld(float3 sample_dir, float3 tangent, float3 bitangent, float3 normal) {
	return sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * normal;
}

float3 WorldToTangent(float3 world_dir, float3 tangent, float3 bitangent, float3 normal) {
	// Transform world-space direction into tangent space by projecting
	// onto the TBN basis axes. This is equivalent to multiplying by the
	// transpose of the TBN matrix (valid when TBN is orthonormal).
	float3 tangent_dir;
	tangent_dir.x = dot(world_dir, tangent);
	tangent_dir.y = dot(world_dir, bitangent);
	tangent_dir.z = dot(world_dir, normal);

	return tangent_dir;
}

// Reflect direction around normal
float3 Reflect(float3 incident, float3 normal) {
	return incident - 2.0 * dot(incident, normal) * normal;
}

[[shader("fragment")]]
float4 ps_main(VSOutput ps_input) : SV_TARGET {
	float3 normal = normalize(ps_input.normal);

	float3 view_dir = normalize(ctx.camera_pos - ps_input.world_pos);

	// Build tangent space
	// float3 helper = abs(normal.x) > 0.999f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 helper = float3(0, 1, 0);
	// float3 h2 = (ps_input.world_pos - float3(0.5, 0.5, 0.5));
	// return float4(h2, 1.0);
	float3 tangent   = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));

	float3x3 tbn     = float3x3(tangent, bitangent, normal);
	float3x3 inv_tbn = transpose(tbn);
	// float3x3 inv_tbn = (tbn);
	// float3 view_tbn = normalize(mul(inv_tbn, view_dir));
	// float3 view_tbn = normalize(mul(view_dir, tbn));
	// float3 view_tbn = normalize(mul(tbn, view_dir));
	float3 view_tbn = normalize(WorldToTangent(view_dir, tangent, bitangent, normal));

	float3 normal_tbn = normalize(WorldToTangent(normal, tangent, bitangent, normal));
	// normal_tbn = normalize(TangentToWorld(normal_tbn, tangent, bitangent, normal));
	// normal_tbn = normalize(WorldToTangent(normal_tbn, tangent, bitangent, normal));
	// float3 normal_tbn = normalize(mul(normal, tbn));
	// float3 normal_tbn = normalize(mul(tbn, normal));
	// normal_tbn = mul(tbn, mul(inv_tbn, normal_tbn));
	// return float4(normal_tbn, 1.0);

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
	float specular_weight = 0.1;

	float3 light_tbn;
	float  pdf = 0.1;

	float3 current_sample = float3(0, 0, 0);

	for (uint i = 0; i < NUM_SAMPLES; i++) {
		let rinput2 = pixel_coord;
		// uint2 rinput2 = uint2(ps_input.world_pos.x, ps_input.world_pos.y);
		float2 random_val = float2(rng_float(rng_state), rng_float(rng_state));
		// float2 random_val = random2D(rinput2, frame_number * NUM_SAMPLES + i);

		// if (1) {
		if (random_val.x < specular_weight) {
			float2 rand        = float2(random_val.x / specular_weight, random_val.y);
			float3 light_blinn = SampleBlinnPhong(rand, shininess, view_tbn, pdf);
			// float cospdf;
			// float3 coss = SampleCosineHemisphere(rand, cospdf);
			light_tbn = (light_blinn);
			// light_tbn = normalize(float3(0.f, 0.f, 50.f) + light_blinn);
			// light_tbn = normalize(normal_tbn * 50. + light_blinn);
			// light_tbn = normalize(normal_tbn);
			// light_tbn = normalize(float3(0.f, 0.f, 1.f) );
			// light_tbn = (/* float3(0.f, 0.f, 1.f) + */ light_blinn);

			pdf *= specular_weight;
			// pdf = 1.;
		} else {

			float2 rand = float2((random_val.x - specular_weight) / (1.0 - specular_weight), random_val.y);
			light_tbn   = SampleCosineHemisphere(rand, pdf);
			// light_tbn   = normalize(float3(0.f, 0.f, 50.f) + light_tbn);
			// light_tbn = float3(0.f, 0.f, 1.f);

			pdf *= (1.0 - specular_weight);
		}
		// return float4(light_tbn, 1.0);
		// return float4(normal_tbn - light_tbn, 1.0);
		// float3 light_dir = TangentToWorld(light_tbn, normal, tangent, bitangent);
		// return float4(light_dir, 1.0);

		if (light_tbn.z > 0.0 && pdf > 0.00001) {
		// if (1) {
			float3 light_dir = TangentToWorld(light_tbn, tangent, bitangent, normal);

			// light_dir = TangentToWorld(float3(0.f, 0.f, 1.f), normal, tangent, bitangent);
			// pdf = 1.0;

			float3 half_vec = float3(0.f, 0.f, 0.f);
			float3 diff_vec = float3(0.f, 0.f, 0.f);
			RusinkiewiczTransform(view_tbn, light_tbn, &half_vec, &diff_vec);

			float inputf[6] = { half_vec.x, half_vec.y, half_vec.z,
								diff_vec.x, diff_vec.y, diff_vec.z };
			float outputf[3];

			BRDFSINEKAN3(inputf, outputf);
			float3 brdf = float3(outputf[0], outputf[1], outputf[2]);

			float4 env_sample   = gEnvMap.Sample(light_dir) * 1.;
			float3 env_radiance = env_sample.rgb;

			// float NdotL = min(max(dot(normal, light_dir), 0.0), 1.0);
			float NdotL = max(dot(normal, light_dir), 0.0);
			// float NdotL = dot(normal_tbn, light_tbn);
			// return float4(-(light_dir - normal), 1.0);
			// return float4(NdotL, NdotL, NdotL, 1.0);
			// return  float4(normal, 1.0);
			// return float4(light_dir, 1.0);
			// return float4(light_tbn, 1.0);
			// NdotL = 1.0;

			float3 eval = (brdf * env_radiance * NdotL) / pdf;
			eval        = min(eval, float3(10.0, 10.0, 10.0));
			current_sample += eval;
		}
	}

	// current_sample /= NUM_SAMPLES;
	float  new_sample_count  = prev_sample_count + NUM_SAMPLES;
	float3 accumulated_color = (prev_color * prev_sample_count + current_sample) / new_sample_count;

	gAccumBuffer[pixel_coord] = float4(accumulated_color, new_sample_count);

	// return PSOutput(float4(accumulated_color, 1.0), ps_input.clip_pos.z);
	return (float4(accumulated_color, 1.0));
}


