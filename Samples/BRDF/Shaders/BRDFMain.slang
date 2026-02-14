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

#define NUM_SAMPLES 1

// PCG Hash - High quality pseudorandom number generator
uint pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

// Initialize RNG state from pixel coordinates and frame count
uint rng_init(uint2 pixel, uint frame) {
	// Combine pixel coordinates and frame for unique seed
	uint seed = pixel.x + pixel.y * 8192u + frame * 67108864u;
	return pcg_hash(seed);
}

// Generate next random number and update state
uint rng_next(inout uint state) {
	state = pcg_hash(state);
	return state;
}

// Generate random float in [0, 1)
float rng_float(inout uint state) {
	return float(rng_next(state)) * (1.0 / 4294967296.0);
}

// Generate random float in [min, max)
float rng_range(inout uint state, float min_val, float max_val) {
	return min_val + rng_float(state) * (max_val - min_val);
}

[[shader("fragment")]]
float4 ps_main(VSOutput vs_input) : SV_Target {
	// Get pixel coordinates
	uint2 pixel_coord = uint2(vs_input.clip_pos.xy);

	// Initialize or reset accumulation
	float3 accumulated;
	float prev_sample_count;

	if (ctx.reset_accumulation || ctx.frame_count == 0) {
		accumulated = float3(0, 0, 0);
		prev_sample_count = 0;
	} else {
		float4 prev_accum = gAccumBuffer[pixel_coord];
		accumulated = prev_accum.rgb;
		// prev_sample_count = ctx.frame_count * NUM_SAMPLES;
		prev_sample_count =  (prev_accum.a);
	}

	// Setup geometry
	float3 normal = normalize(vs_input.normal);
	float3 view_dir = normalize(ctx.camera_pos - vs_input.world_pos);

	// Build tangent space (orthonormal basis)
	float3 helper = abs(normal.x) > 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 tangent = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));
	float3x3 tbn = float3x3(tangent, bitangent, normal);
	float3x3 inv_tbn = transpose(tbn);

	float3 view_tbn = normalize(mul(inv_tbn, view_dir));

	// Material roughness parameter (can be passed in or derived from material)
	// Lower values = more specular, higher values = more diffuse
	float roughness = 0.5;				 // Default value, adjust as needed
	float alpha = roughness * roughness; // GGX uses alpha = roughness^2

	// Initialize RNG for this pixel and frame
	uint rng_state = rng_init(pixel_coord, ctx.frame_count);

	// Accumulate lighting for this frame
	float3 frame_contribution = float3(0, 0, 0);

	// Monte Carlo integration with Multiple Importance Sampling (MIS)
	for (int i = 0; i < NUM_SAMPLES; i++) {
		// Generate random numbers for this sample
		float2 u = float2(rng_float(rng_state), rng_float(rng_state));
		float sample_choice = rng_float(rng_state);

		float3 light_dir;
		float3 light_tbn;
		float sampling_pdf;

		// Use MIS: alternate between cosine sampling and GGX sampling
		// Each strategy contributes with appropriate weight
		bool use_cosine_sampling = (sample_choice < 0.5);
		use_cosine_sampling = 0;

		float mis_weight;
		if (use_cosine_sampling) {
			// ============================================================
			// Strategy 1: Cosine-weighted hemisphere sampling
			// ============================================================
			float pdf_cosine;
			float3 sample_local = SampleCosineHemisphere(u, pdf_cosine);

			// Transform to world space
			light_dir = TangentToWorld(sample_local, normal, tangent, bitangent);
			light_tbn = normalize(mul(inv_tbn, light_dir));

			// Compute half vector for GGX PDF evaluation
			float3 half_tbn = normalize(view_tbn + light_tbn);

			// Evaluate both PDFs for MIS weight
			float pdf_ggx = GGXVNDF_PDF(view_tbn, half_tbn, light_tbn, alpha);

			// MIS weight using power heuristic (beta = 2)
			mis_weight = MISWeight(pdf_cosine, pdf_ggx, 2.0);

			// Combined PDF accounting for 50% probability of choosing this strategy
			sampling_pdf = 0.5 * pdf_cosine;

			// Apply MIS weight to the contribution
			// We'll multiply by mis_weight after BRDF evaluation

		} else {
			// ============================================================
			// Strategy 2: GGX VNDF sampling
			// ============================================================
			float pdf_ggx;

			// Sample half vector in tangent space
			float3 half_tbn = SampleGGXVNDF(view_tbn, alpha, u, pdf_ggx);

			// let N  = normalize(normal);
			// let V = normalize(view_tbn);
			// float2 Xi = hammersleySequence(uint(prev_sample_count) + i, uint(prev_sample_count) + NUM_SAMPLES);
			// float3 H = importanceSampleGGX(Xi, N, roughness);
			// float3 L = reflect(-V, H);

			// float dotNH = max(dot(N, H), 0.0);
			// float D = dGgx(dotNH, roughness);
			// float pdf = 0.25 * D + 0.0001;
			// pdf_ggx = pdf;//* 0.1;

			// let half_tbn = normalize(L);

			// Convert half vector to light direction
			light_tbn = Reflect(-view_tbn, half_tbn);

			// Check if light direction is above surface
			if (light_tbn.z <= 0.0 || pdf_ggx <= 0.0) {
				continue; // Skip invalid samples
			}

			// Transform to world space
			light_dir = TangentToWorld(light_tbn, normal, tangent, bitangent);

			// Evaluate both PDFs for MIS weight
			float pdf_cosine = CosinePDF(light_tbn.z);

			// Convert VNDF half-vector PDF to light direction PDF
			float VdotH = dot(view_tbn, half_tbn);
			pdf_ggx = pdf_ggx / (4.0 * VdotH);

			// MIS weight using power heuristic (beta = 2)
			mis_weight = MISWeight(pdf_ggx, pdf_cosine, 2.0);

			// Combined PDF accounting for 50% probability of choosing this strategy
			sampling_pdf = 0.5 * pdf_ggx;
		}

		// ============================================================
		// Common evaluation for both strategies
		// ============================================================

		// Prepare inputs for neural BRDF using Rusinkiewicz parameterization
		float3 half_vec = float3(0.f, 0.f, 0.f);
		float3 diff_vec = float3(0.f, 0.f, 0.f);
		RusinkiewiczTransform(view_tbn, light_tbn, &half_vec, &diff_vec);

		float inputf[6] = { half_vec.x, half_vec.y, half_vec.z,
							diff_vec.x, diff_vec.y, diff_vec.z };
		float outputf[3];

		// Evaluate neural BRDF
		BRDFSINEKAN3(inputf, outputf);
		float3 brdf = float3(outputf[0], outputf[1], outputf[2]);

		// Sample environment map
		float4 env_sample = gEnvMap.Sample(light_dir);
		float3 env_radiance = env_sample.rgb * 3;

		// Cosine term
		float NdotL = max(dot(normal, light_dir), 0.0);

		// Monte Carlo estimator with MIS: (BRDF * L * cos(theta)) / pdf
		if (sampling_pdf > 1e-6) {
			// if (1) {
			float3 contribution = (brdf * env_radiance * NdotL /* * mis_weight */) / sampling_pdf;

			// Clamp to prevent fireflies (very bright samples from low PDF)
			contribution = min(contribution, float3(10.0, 10.0, 10.0));

			frame_contribution += contribution;
		}
	}

	// Average samples for this frame
	frame_contribution  /= float(NUM_SAMPLES);

	float new_sample_count = prev_sample_count + NUM_SAMPLES;

	// Accumulate with previous frames
	float3 final_color;
	if (ctx.reset_accumulation || ctx.frame_count == 0) {
		final_color = frame_contribution;
		// final_color = float3(0.0, 0.0, 0.0);
	} else {
		// Progressive averaging: new_avg = (old_avg * old_count + new_value) / (old_count + 1)
		float weight = (prev_sample_count) / new_sample_count;
		final_color = accumulated.rgb * weight + frame_contribution * (1.0 - weight);
	}

	// float3 accumulated_color = (prev_color * prev_sample_count + current_sample) / new_sample_count;

	// Store in accumulation buffer
	gAccumBuffer[pixel_coord] = float4(final_color, new_sample_count);

	return float4(final_color, 1.0);
}
