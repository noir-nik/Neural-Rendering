import Pbr;
#include "BRDFConstants.h"

func BRDFClassic(float3 world_pos, float3 view, Light light, float3 normal, float3 albedo, float metallic, float roughness)->float3 {
	static const float PI = 3.14159265359;
	float3 F0 = float3(0.04, 0.04, 0.04);

	F0 = lerp(F0, albedo, metallic);

	// Initialize lighting components to zero
	float3 Lo = float3(0.0, 0.0, 0.0);

	{
		float3 L           = normalize(light.position - world_pos);
		float3 H           = normalize(view + L);
		float  distance    = length(light.position - world_pos);
		float  attenuation = 1.0 / (distance * distance);
		float3 radiance    = light.color * light.intensity * attenuation;

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
	float3 ambient = light.ambient_color * light.ambient_intensity * albedo;

	// Final color
	float3 color = ambient + Lo;

	return color;
}


