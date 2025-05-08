#include "BRDFConfig.h"
#include "BRDFConstants.h"

[[vk::push_constant]]
BRDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

// clang-format off
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

float3 fresnelSchlick(float cosTheta, float3 F0) {
	return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness) {
	float a      = roughness * roughness;
	float a2     = a * a;
	float NdotH  = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float num   = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom       = PI * denom * denom;

	return num / max(denom, 0.001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float num   = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return num / max(denom, 0.001);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2  = GeometrySchlickGGX(NdotV, roughness);
	float ggx1  = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

[[shader("vertex")]]
VSOutput vs_main(VSInput in_vertex) {
	VSOutput output;
	output.clip_pos  = mul(ctx.view_proj, float4(in_vertex.pos.xyz, 1.0f));
	output.world_pos = in_vertex.pos.xyz;
	output.normal    = normalize(in_vertex.normal.xyz);
	return output;
}

[[shader("fragment")]]
func ps_main(VSOutput input)->float4 {
	float3 N = normalize(input.normal);
	float3 V = normalize(ctx.camera_pos - input.world_pos);

	float3 albedo    = ctx.material.base_color.rgb;
	float  metallic  = ctx.material.metallic;
	float  roughness = ctx.material.roughness;

	float3 F0 = float3(0.04, 0.04, 0.04);

	F0 = lerp(F0, albedo, metallic);

	// Initialize lighting components to zero
	float3 Lo = float3(0.0, 0.0, 0.0);

	{
		float3 L           = normalize(ctx.light.position - input.world_pos);
		float3 H           = normalize(V + L);
		float  distance    = length(ctx.light.position - input.world_pos);
		float  attenuation = 1.0 / (distance * distance);
		float3 radiance    = ctx.light.color * ctx.light.intensity * attenuation;

		// BRDF
		float  NDF = DistributionGGX(N, H, roughness);
		float  G   = GeometrySmith(N, V, L, roughness);
		float3 F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

		// specular
		float3 numerator   = NDF * G * F;
		float  denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
		float3 specular    = numerator / max(denominator, 0.001);

		// Energy conservation
		float3 kS = F;
		float3 kD = float3(1.0, 1.0, 1.0) - kS;
		kD *= 1.0 - metallic;

		float NdotL = max(dot(N, L), 0.0);
		Lo += (kD * albedo / PI + specular) * radiance * NdotL;
	}

	// Ambient
	float3 ambient = ctx.light.ambient_color * ctx.light.ambient_intensity * albedo;

	// Final color
	float3 color = ambient + Lo;

	// Simple Reinhard tone mapping
	color = color / (color + float3(1.0, 1.0, 1.0));
	// Gamma correction
	color = pow(color, float3(1.0 / 2.2)); 

	return float4(color, 1.0);
}


