float3 fresnelSchlick(float cos_theta, float3 F0) {
	return F0 + (1.0 - F0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

float DistributionGGX(float NdotH, float roughness) {
	const float PI = 3.14159265359;

	float alpha  = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom  = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
	return alpha2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float num   = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return num / max(denom, 0.001);
}

float GeometrySmith(float NdotL, float NdotV, float roughness) {
	float ggxL = GeometrySchlickGGX(NdotL, roughness);
	float ggxV = GeometrySchlickGGX(NdotV, roughness);
	return ggxL * ggxV;
}