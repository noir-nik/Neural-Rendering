#include "BRDFConfig.h"
#include "BRDFConstants.h"

import Pbr;
import BRDFScalarBuffer;
import BRDFCoopVec;

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

static const float PI = 3.14159265359;

[[shader("vertex")]]
VSOutput vs_main(VSInput in_vertex) {
	VSOutput output;
	output.clip_pos  = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	output.world_pos = in_vertex.pos.xyz;
	output.normal    = normalize(in_vertex.normal.xyz);
	return output;
}

func ClassicLight(float3 world_pos, float3 view, float3 normal, float3 albedo, float metallic, float roughness)->float3 {
	float3 F0 = float3(0.04, 0.04, 0.04);

	F0 = lerp(F0, albedo, metallic);

	// Initialize lighting components to zero
	float3 Lo = float3(0.0, 0.0, 0.0);

	{
		float3 L           = normalize(ctx.light.position - world_pos);
		float3 H           = normalize(view + L);
		float  distance    = length(ctx.light.position - world_pos);
		float  attenuation = 1.0 / (distance * distance);
		float3 radiance    = ctx.light.color * ctx.light.intensity * attenuation;

		float NdotL = max(dot(normal, L), 0.0);
		float NdotV = max(dot(normal, view), 0.0);
		float NdotH = max(dot(normal, H), 0.0);
		float HdotV = max(dot(H, view), 0.0);

		// BRDF
		float  NDF = DistributionGGX(NdotH, roughness);
		float  G   = GeometrySmith(NdotL, NdotV, roughness);
		float3 F   = fresnelSchlick(HdotV, F0);

		// specular
		float3 numerator   = NDF * G * F;
		float  denominator = 4.0 * NdotV * NdotL;
		float3 specular    = numerator / max(denominator, 0.001);

		// Energy conservation
		float3 kS = F;
		float3 kD = float3(1.0, 1.0, 1.0) - kS;
		kD *= 1.0 - metallic;

		Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	}

	// Ambient
	float3 ambient = ctx.light.ambient_color * ctx.light.ambient_intensity * albedo;

	// Final color
	float3 color = ambient + Lo;

	return color;
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

	// float3x3 inv_tbn = transpose(tbn);
	float3x3 inv_tbn = (tbn);

	float3 view_tbn  = mul(inv_tbn, view);
	float3 light_tbn = mul(inv_tbn, light_dir);

	view_tbn  = normalize(view_tbn);
	light_tbn = normalize(light_tbn);

	float3 color = float3(0, 0, 0);

	float3 albedo    = ctx.material.base_color.rgb;
	float  metallic  = ctx.material.metallic;
	float  roughness = ctx.material.roughness;

	if (shader_type == BrdfFunctionType::eCoopVec) {
		color = BRDFCoopVec<COOPVEC_TYPE>( { view_tbn.x, view_tbn.y, view_tbn.z }, { light_tbn.x, light_tbn.y, light_tbn.z }, gWeightsCoopVec, ctx.weights_offsets, ctx.bias_offsets);
	} else if (shader_type == BrdfFunctionType::eScalarBuffer) {
		color = BRDFScalarBuffer(view_tbn, light_tbn, gWeights, ctx.weights_offsets, ctx.bias_offsets);
	} else {
		color = ClassicLight(input.world_pos, view, normal, albedo, metallic, roughness);
	}

	// Simple Reinhard tone mapping
	color = color / (color + float3(1.0, 1.0, 1.0));

	// Gamma correction
	color = pow(color, 1.0 / 2.2);

	// color   = f.xyz;
	return float4(color, 1.0);
}


