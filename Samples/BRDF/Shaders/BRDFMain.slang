#include "BRDFConfig.h"
#include "BRDFConstants.h"
import Pbr;
// import NeuroBRDF;
import Utils; // rusinkiewicz_transform
import BRDFSINEKAN;



// clang-format off
[[vk::binding(0, 0)]] StructuredBuffer<float> gWeights;
[[vk::binding(0, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeightsCoopVec;
// Cubemap texture
[[vk::binding(1, 0)]] SamplerCube gEnvMap;
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

#ifndef NUM_ENV_SAMPLES
#define NUM_ENV_SAMPLES 128  // Adjust based on quality needs
#endif

// Generate sampling direction using Fibonacci sphere
float3 FibonacciSphere(int i, int n) {
    const float phi = 1.618033988749895; // golden ratio
    float y = 1.0 - (2.0 * float(i)) / float(n - 1);
    float radius = sqrt(1.0 - y * y);
    float theta = 2.0 * 3.14159265359 * float(i) / phi;
    return float3(cos(theta) * radius, y, sin(theta) * radius);
}

// Transform hemisphere sample to world space
float3 TangentToWorld(float3 sample_dir, float3 normal, float3 tangent, float3 bitangent) {
    return sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * normal;
}

[[shader("fragment")]]
float4 ps_main(VSOutput vs_input) {
	float3 normal = normalize(vs_input.normal);
	float3 view_dir = normalize(ctx.camera_pos - vs_input.world_pos);

	// Build tangent space
	float3 helper = abs(normal.x) > 0.99f ? float3(0, 1, 0) : float3(1, 0, 0);
	float3 tangent = normalize(cross(helper, normal));
	float3 bitangent = normalize(cross(normal, tangent));
	float3x3 tbn = float3x3(tangent, bitangent, normal);
	float3x3 inv_tbn = transpose(tbn);

	float3 view_tbn = normalize(mul(inv_tbn, view_dir));

	// Accumulate lighting from environment map
	float3 accumulated_color = float3(0, 0, 0);
	float total_weight = 0.0; // Material properties
	float3 albedo = ctx.material.base_color.rgb;
	float metallic = ctx.material.metallic;
	float roughness = ctx.material.roughness;

	// Monte Carlo integration over hemisphere
	for (int i = 0; i < NUM_ENV_SAMPLES; i++) {
		// Generate sample direction in hemisphere
		float3 sample_local = FibonacciSphere(i, NUM_ENV_SAMPLES);

		// Only consider upper hemisphere
		if (sample_local.z < 0.0) continue;

		// Transform to world space
		float3 light_dir = TangentToWorld(sample_local, normal, tangent, bitangent);

		// Transform to tangent space for neural BRDF
		float3 light_tbn = normalize(mul(inv_tbn, light_dir));

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
		float3 env_radiance = env_sample.rgb * 3.;

		// Cosine term
		float NdotL = max(dot(normal, light_dir), 0.0);

		// Accumulate contribution
		accumulated_color += brdf * env_radiance * NdotL;
		total_weight += NdotL;
	}

	// Average and apply Monte Carlo normalization
	// The 2*PI factor accounts for hemisphere integration
	float3 color = accumulated_color * (2.0 * 3.14159265359 / float(NUM_ENV_SAMPLES));

	// Apply albedo/color modulation
	float3 final_color = color * albedo;

	// Add specular reflection for extra realism (optional)
	float3 R = reflect(-view_dir, normal);
	float4 spec_env = gEnvMap.Sample(R);
	float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
	float3 fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);
	final_color += spec_env.rgb * fresnel * (1.0 - roughness);

	return float4(final_color, 1.0);
}
